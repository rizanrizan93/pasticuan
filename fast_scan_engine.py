from __future__ import annotations

"""Lean single-pass orchestration for the production Streamlit UI.

The analytical engine remains unchanged.  What is intentionally removed from
this default path is durable scan-job orchestration (scan_jobs, item leases,
heartbeats, per-chunk checkpoint reads/writes and artifact repository polling).
Supabase is treated as a cache/persistence accelerator, never as a prerequisite
for producing a ranking in the current Streamlit session.
"""

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence
import hashlib
import os
import time

import pandas as pd

from scanner_database import (
    DatabaseSettings,
    DatabaseTransportError,
    ScannerDatabaseBridge,
)
from resumable_app_engine import process_daily_scan_chunk, finalize_daily_scan_job, _expected_completed_session

FAST_SCAN_VERSION = "9.8.7-production-hardening"


class FastDatabaseBridge(ScannerDatabaseBridge):
    """Short-timeout bridge with independent read/write transport circuits.

    A slow optional cache read must never disable database writes.  The previous
    single circuit classified one scanner_feature_cache timeout as a complete
    Supabase outage, which made a healthy project look unavailable and prevented
    persistence for the rest of the scan.  Read and write health are now tracked
    independently and a read circuit only opens after two consecutive transport
    failures.  Any successful read resets that failure streak.
    """

    def __init__(self) -> None:
        base = DatabaseSettings.from_env()
        timeout = max(4.0, min(12.0, float(os.environ.get("SCANNER_FAST_DATABASE_TIMEOUT", "8"))))
        connect = max(1.0, min(4.0, float(os.environ.get("SCANNER_FAST_DATABASE_CONNECT_TIMEOUT", "3"))))
        settings = replace(
            base,
            timeout_seconds=timeout,
            connect_timeout_seconds=connect,
            read_attempts=1,
            write_attempts=1,
            retry_backoff_seconds=0.1,
            # Smaller REST batches are more predictable for JSONB payloads.
            read_batch_size=max(40, min(80, int(base.read_batch_size))),
        )
        super().__init__(settings=settings)
        self.read_circuit_open = False
        self.write_circuit_open = False
        self.read_transport_failures = 0
        self.write_transport_failures = 0
        self.read_transport_error = ""
        self.write_transport_error = ""

    @property
    def transport_circuit_open(self) -> bool:
        # Compatibility alias: only a complete read+write outage is a global
        # circuit-open state.  A degraded read path may still persist results.
        return bool(self.read_circuit_open and self.write_circuit_open)

    @property
    def transport_error(self) -> str:
        return " | ".join(v for v in (self.read_transport_error, self.write_transport_error) if v)

    def transport_state(self) -> str:
        if self.read_circuit_open and self.write_circuit_open:
            return "READ_WRITE_DEGRADED_FAIL_SOFT"
        if self.read_circuit_open:
            return "READ_DEGRADED_WRITE_AVAILABLE"
        if self.write_circuit_open:
            return "WRITE_DEGRADED_READ_AVAILABLE"
        return self.settings.mode

    def _request(self, method: str, endpoint: str, *, operation: str, retry_safe: bool, **kwargs: Any) -> Any:
        verb = str(method or "").strip().lower()
        is_read = verb in {"get", "head"}
        if self.settings.mode == "SUPABASE_REST":
            if is_read and self.read_circuit_open:
                raise DatabaseTransportError(operation, endpoint, self.read_transport_error or "FAST_DATABASE_READ_CIRCUIT_OPEN", 1)
            if not is_read and self.write_circuit_open:
                raise DatabaseTransportError(operation, endpoint, self.write_transport_error or "FAST_DATABASE_WRITE_CIRCUIT_OPEN", 1)
        try:
            response = super()._request(method, endpoint, operation=operation, retry_safe=retry_safe, **kwargs)
            status = int(getattr(response, "status_code", 0) or 0)
            if status in {408, 425, 429, 500, 502, 503, 504}:
                if is_read:
                    self.read_transport_failures += 1
                    self.read_transport_error = f"HTTP {status} during {operation}"
                    if self.read_transport_failures >= 2:
                        self.read_circuit_open = True
                else:
                    self.write_transport_failures += 1
                    self.write_transport_error = f"HTTP {status} during {operation}"
                    if self.write_transport_failures >= 2:
                        self.write_circuit_open = True
            else:
                if is_read:
                    self.read_transport_failures = 0
                    self.read_transport_error = ""
                else:
                    self.write_transport_failures = 0
                    self.write_transport_error = ""
            return response
        except DatabaseTransportError as exc:
            if is_read:
                self.read_transport_failures += 1
                self.read_transport_error = str(exc)[:500]
                if self.read_transport_failures >= 2:
                    self.read_circuit_open = True
            else:
                self.write_transport_failures += 1
                self.write_transport_error = str(exc)[:500]
                if self.write_transport_failures >= 2:
                    self.write_circuit_open = True
            raise


def _ticker(value: Any) -> str:
    text = str(value or "").upper().strip()
    return text if text else ""


def _job_id(tickers: Sequence[str]) -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    digest = hashlib.sha256(("|".join(tickers) + "|" + now).encode("utf-8")).hexdigest()[:16]
    return f"FAST-{digest}"


def _items(tickers: Sequence[str]) -> pd.DataFrame:
    rows = []
    for idx, ticker in enumerate(tickers):
        rows.append({
            "job_id": "IN_MEMORY",
            "item_key": f"T{idx:04d}-{ticker}",
            "ticker": ticker,
            "phase": "TECHNICAL",
            "status": "RUNNING",
            # One provider pass only. A missing symbol is checkpointed as a
            # data-availability state instead of waiting through retry leases.
            "attempt_count": 1,
            "max_attempts": 1,
        })
    return pd.DataFrame(rows)


def _outcomes_to_items(items: pd.DataFrame, outcomes: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw in items.to_dict("records"):
        key = str(raw.get("item_key", ""))
        outcome = outcomes.get(key)
        row = dict(raw)
        if outcome is None:
            row.update({"status": "FAILED", "last_error": "PROCESSOR_RETURNED_NO_OUTCOME", "result_payload": {"ticker": raw.get("ticker")}})
        elif bool(getattr(outcome, "success", False)):
            row.update({"status": "COMPLETE", "last_error": None, "result_payload": dict(getattr(outcome, "payload", {}) or {})})
        else:
            # Keep one-pass semantics. Provider unavailability is represented in
            # the payload/audit and does not trigger a durable retry loop.
            payload = dict(getattr(outcome, "payload", {}) or {})
            payload.setdefault("ticker", raw.get("ticker"))
            row.update({"status": "COMPLETE", "last_error": str(getattr(outcome, "error", "") or "DATA_UNAVAILABLE"), "result_payload": payload})
        rows.append(row)
    return pd.DataFrame(rows)



def _compact_scalar_mapping(value: Any, *, max_text: int = 500) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key, item in value.items():
        if item is None or isinstance(item, (bool, int, float)):
            out[str(key)] = item
        elif isinstance(item, str):
            out[str(key)] = item[:max_text]
        elif isinstance(item, pd.Timestamp):
            out[str(key)] = item.isoformat()
    return out


def _compact_feature_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only fields required to reconstruct all-eligible discovery.

    Narrative event/outcome lists belong to durable research tables, not the
    lightweight feature cache.  Removing nested/heavy objects keeps the cache a
    real accelerator instead of another large JSON transfer.
    """
    local = dict(payload or {})
    compact = {
        "ticker": local.get("ticker"),
        "completion_state": local.get("completion_state", "TECHNICAL_READY"),
        "technical_ready": bool(local.get("technical_ready", False)),
        "signal": _compact_scalar_mapping(local.get("signal")),
        "silent_profile": _compact_scalar_mapping(local.get("silent_profile")),
        "narrative_profile": _compact_scalar_mapping(local.get("narrative_profile")),
        "narrative_events": [],
        "narrative_outcomes": [],
        "breadth": _compact_scalar_mapping(local.get("breadth")),
        "ohlcv_state": local.get("ohlcv_state"),
        "ohlcv_bars": local.get("ohlcv_bars"),
        "ohlcv_last_bar_date": local.get("ohlcv_last_bar_date"),
        "ohlcv_session_lag": local.get("ohlcv_session_lag"),
        "ohlcv_source_tier": local.get("ohlcv_source_tier"),
        "completed_at": local.get("completed_at"),
    }
    return compact

def run_fast_single_scan(
    tickers: Sequence[str],
    *,
    config: Mapping[str, Any] | None = None,
    runtime: Mapping[str, str] | None = None,
    progress: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    """Run the complete 400-name workflow without scan-job repository I/O."""
    started = time.perf_counter()
    universe = list(dict.fromkeys(_ticker(v) for v in tickers if _ticker(v)))[:400]
    if not universe:
        raise ValueError("Universe ticker kosong")
    cfg = dict(config or {})
    # Lean production policy. These are research budgets, not user-facing modes.
    cfg.setdefault("period", "5y")
    cfg.setdefault("provider_batch_size", 80)
    cfg.setdefault("evidence_refresh_cap", 20)
    cfg.setdefault("decision_evidence_cap", 12)
    cfg.setdefault("evidence_fundamental_cap", 20)
    cfg.setdefault("evidence_official_cap", 12)
    cfg.setdefault("evidence_snapshot_cap", 16)
    cfg.setdefault("evidence_market_cap", 6)
    cfg.setdefault("evidence_news_cap", 10)
    cfg.setdefault("execution_verification_cap", 8)
    cfg.setdefault("macro_external_enabled", True)
    cfg.setdefault("macro_timeout_seconds", 3)
    cfg.setdefault("lean_persistence", True)
    cfg.setdefault("lean_skip_narrative_history", True)
    cfg.setdefault("daily_market_refresh_limit", 6)
    cfg.setdefault("portfolio_records", [])

    jid = _job_id(universe)
    job = {
        "job_id": jid,
        "job_type": "FAST_DAILY_SCAN",
        "universe_payload": universe,
        "config_payload": cfg,
        "status": "RUNNING",
        "phase": "TECHNICAL",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "total_items": len(universe),
        "completed_items": 0,
        "failed_items": 0,
        "chunk_size": len(universe),
    }
    bridge = FastDatabaseBridge()
    raw_items = _items(universe)

    # ALL_ELIGIBLE_LITE: every ticker remains eligible, but a current technical
    # payload is reused from one lightweight row instead of transferring and
    # recomputing 800-900 OHLCV bars again. Missing/stale names are the only
    # ones sent through the full technical engine.
    expected_session = _expected_completed_session()
    feature_hits: dict[str, dict[str, Any]] = {}
    feature_audit = pd.DataFrame()
    feature_read_started = time.perf_counter()
    if hasattr(bridge, "read_feature_cache"):
        try:
            feature_hits, feature_audit = bridge.read_feature_cache(
                universe, expected_session=expected_session, scanner_version=FAST_SCAN_VERSION,
            )
        except Exception as exc:
            feature_hits, feature_audit = {}, pd.DataFrame([{
                "provider": "SUPABASE_FEATURE_CACHE", "status": "READ_FAIL_SOFT",
                "error": f"{type(exc).__name__}: {str(exc)[:240]}",
            }])
    feature_read_elapsed = time.perf_counter() - feature_read_started
    cached_names = set(feature_hits)
    compute_items = raw_items.loc[~raw_items["ticker"].isin(cached_names)].copy()
    cached_rows: list[dict[str, Any]] = []
    for row in raw_items.loc[raw_items["ticker"].isin(cached_names)].to_dict("records"):
        payload = dict(feature_hits.get(str(row.get("ticker")), {}) or {})
        payload["worker_id"] = "feature-cache"
        payload["feature_cache_hit"] = True
        cached_rows.append({**row, "status": "COMPLETE", "last_error": None, "result_payload": payload})
    cached_completed = pd.DataFrame(cached_rows)

    def emit(label: str, pct: float) -> None:
        if progress is not None:
            progress(label, float(pct))

    emit(
        f"ALL_ELIGIBLE_LITE: {len(cached_names)} feature-cache hit, {len(compute_items)} ticker perlu technical refresh",
        0.08,
    )
    technical_started = time.perf_counter()
    if not compute_items.empty:
        outcomes = process_daily_scan_chunk(
            job, compute_items, "single-pass", runtime=dict(runtime or {}), bridge_override=bridge,
        )
        computed_completed = _outcomes_to_items(compute_items, outcomes)
    else:
        outcomes = {}
        computed_completed = pd.DataFrame(columns=list(raw_items.columns) + ["last_error", "result_payload"])
    technical_elapsed = time.perf_counter() - technical_started

    # Persist only newly computed technical payloads. This write is optional and
    # fail-soft; ranking never waits for a durable job repository.
    feature_write_report = pd.DataFrame()
    feature_write_started = time.perf_counter()
    if not computed_completed.empty and hasattr(bridge, "write_feature_cache") and not bridge.write_circuit_open:
        fresh_payloads: dict[str, Mapping[str, Any]] = {}
        for row in computed_completed.to_dict("records"):
            payload = row.get("result_payload") or {}
            ticker = _ticker(row.get("ticker") or (payload.get("ticker") if isinstance(payload, Mapping) else ""))
            if ticker and isinstance(payload, Mapping) and bool(payload.get("technical_ready", False)):
                fresh_payloads[ticker] = _compact_feature_payload(payload)
        if fresh_payloads:
            feature_write_report = bridge.write_feature_cache(fresh_payloads, scanner_version=FAST_SCAN_VERSION)
    feature_write_elapsed = time.perf_counter() - feature_write_started

    completed_items = pd.concat(
        [frame for frame in (cached_completed, computed_completed) if isinstance(frame, pd.DataFrame) and not frame.empty],
        ignore_index=True, sort=False,
    ) if (not cached_completed.empty or not computed_completed.empty) else pd.DataFrame()
    if not completed_items.empty and "ticker" in completed_items.columns:
        order_map = {ticker: idx for idx, ticker in enumerate(universe)}
        completed_items["_universe_order"] = completed_items["ticker"].map(order_map).fillna(999999)
        completed_items = completed_items.sort_values("_universe_order").drop(columns=["_universe_order"]).reset_index(drop=True)
    ready_count = int(completed_items["result_payload"].map(lambda x: bool(x.get("technical_ready")) if isinstance(x, Mapping) else False).sum()) if not completed_items.empty else 0
    job["completed_items"] = len(completed_items)
    job["phase"] = "FINAL_RANKING"

    emit(f"Technical discovery selesai ({ready_count}/{len(universe)} ready); enrichment shortlist", 0.70)
    finalized = finalize_daily_scan_job(
        job,
        bridge,
        "single-pass",
        runtime=dict(runtime or {}),
        items_override=completed_items,
        durable_updates=False,
        persist_artifacts=False,
        return_result=True,
    )
    result = dict(finalized.get("result") or {})
    elapsed = time.perf_counter() - started
    result["scan_elapsed_seconds"] = round(elapsed, 3)
    result["scanner_version"] = FAST_SCAN_VERSION
    result["mode"] = "all_eligible_lite_database_accelerated"
    result["all_eligible_state"] = "ALL_ELIGIBLE_LITE"
    result["feature_cache_hits"] = int(len(cached_names))
    result["feature_cache_refreshes"] = int(len(compute_items))
    result["feature_cache_audit"] = feature_audit
    result["feature_cache_write_report"] = feature_write_report
    result["scan_finished_at"] = datetime.now(timezone.utc).isoformat()
    result["database_transport_state"] = bridge.transport_state() if hasattr(bridge, "transport_state") else ("CIRCUIT_OPEN_FAIL_SOFT" if bool(getattr(bridge, "transport_circuit_open", False)) else str(getattr(getattr(bridge, "settings", None), "mode", "UNKNOWN")))
    result["database_transport_error"] = str(getattr(bridge, "transport_error", "") or "")
    result["scan_id"] = jid
    stage_frame = result.get("stage_timings")
    if isinstance(stage_frame, pd.DataFrame) and not stage_frame.empty:
        stage_frame = stage_frame.copy()
        stage_frame["FEATURE_CACHE_READ"] = round(feature_read_elapsed, 3)
        stage_frame["TECHNICAL_DISCOVERY"] = round(technical_elapsed, 3)
        stage_frame["FEATURE_CACHE_WRITE"] = round(feature_write_elapsed, 3)
        stage_frame["TOTAL_SCAN"] = round(elapsed, 3)
        result["stage_timings"] = stage_frame
    emit("Ranking final selesai", 1.0)
    return result


__all__ = ["FAST_SCAN_VERSION", "FastDatabaseBridge", "run_fast_single_scan"]
