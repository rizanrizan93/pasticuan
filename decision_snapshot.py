from __future__ import annotations

"""Immutable, compact final-decision snapshots for exact scan readback."""

from hashlib import sha256
from typing import Any, Iterable, Mapping, Sequence
import json

import numpy as np
import pandas as pd


FINAL_DECISION_SNAPSHOT_VERSION = "FINAL_DECISION_SNAPSHOT_V1"


def _ticker(value: Any) -> str:
    return str(value or "").upper().strip()


def _safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _canonical(value: Any) -> str:
    return json.dumps(_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _first(row: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        return value
    return default


def _summary(row: Mapping[str, Any], names: Iterable[str]) -> dict[str, Any]:
    return {
        name: _safe(row.get(name))
        for name in names
        if name in row and _safe(row.get(name)) is not None
    }


def _provider_degradation(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in (
        "database_sync_report", "macro_source_report", "independent_provider_report",
        "fundamental_history_report", "feature_cache_audit",
    ):
        frame = result.get(key)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        for raw in frame.to_dict("records"):
            state = str(_first(raw, ("status", "state", "database_read_state"), "UNKNOWN") or "UNKNOWN").upper()
            if state in {"OK", "CURRENT", "READY", "VERIFIED", "HIT_CURRENT", "DATABASE_READY"}:
                continue
            rows.append({
                "source": key,
                "provider": _first(raw, ("provider", "source_family", "table"), "UNKNOWN"),
                "state": state,
                "ticker": _ticker(raw.get("ticker")) or None,
                "error": str(raw.get("error") or "")[:240] or None,
            })
    return rows[:100]


def build_final_decision_snapshot(
    result: Mapping[str, Any],
    *,
    universe: Sequence[object],
    completed_session: Any,
) -> dict[str, Any]:
    focus = result.get("focus_screens", {}) if isinstance(result.get("focus_screens"), Mapping) else {}
    lanes = (
        ("NEXT_LEADER", focus.get("next_leaders_all", focus.get("next_leaders", pd.DataFrame()))),
        ("SWING_READY", focus.get("swing_ready_all", focus.get("swing_ready", pd.DataFrame()))),
    )
    decisions: list[dict[str, Any]] = []
    for lane, frame in lanes:
        if not isinstance(frame, pd.DataFrame) or frame.empty or "ticker" not in frame.columns:
            continue
        for raw in frame.to_dict("records"):
            ticker = _ticker(raw.get("ticker"))
            if not ticker:
                continue
            decisions.append({
                "ticker": ticker,
                "lane": lane,
                "rank": _safe(_first(raw, ("rank", "production_rank", "multibagger_selection_rank"))),
                "ranking_score": _safe(_first(raw, ("ranking_score", "final_score"))),
                "production_score": _safe(_first(raw, ("v9_next_leader_score", "v9_swing_score", "final_score"))),
                "coverage_pct": _safe(_first(raw, ("score_coverage_pct", "v8_production_score_coverage_pct"))),
                "confidence_pct": _safe(_first(raw, ("thesis_confidence_pct", "overall_research_confidence", "silent_accumulation_confidence"))),
                "guardrail_state": _safe(_first(raw, ("ranking_guardrail_state", "methodology_gate_state", "methodology_gate_reason"), "UNKNOWN")),
                "authorization_state": _safe(_first(raw, ("real_money_authorization_state", "execution_gate_state", "production_gate_reason"), "UNKNOWN")),
                "eligible": bool(_first(raw, ("production_rank_eligible", "rank_eligible"), False)),
                "entry": _safe(_first(raw, ("execution_entry", "entry"))),
                "stop_loss": _safe(raw.get("stop_loss")),
                "tp1": _safe(raw.get("tp1")),
                "tp2": _safe(raw.get("tp2")),
                "rr": _safe(_first(raw, ("rr2", "rr1"))),
                "evidence_freshness_summary": _summary(raw, (
                    "fundamental_freshness_state", "fundamental_reporting_cadence_state",
                    "market_context_provenance_state", "independent_price_state",
                    "execution_plan_freshness_state", "statement_age_days",
                )),
                "provenance_summary": _summary(raw, (
                    "fundamental_official_state", "fundamental_source_families",
                    "market_context_provenance_state", "independent_price_verified",
                    "ohlcv_source_tier", "production_scoring_version",
                )),
            })
    normalized_universe = list(dict.fromkeys(_ticker(value) for value in universe if _ticker(value)))
    universe_identity = sha256(_canonical(normalized_universe).encode("utf-8")).hexdigest()
    body = {
        "snapshot_schema_version": FINAL_DECISION_SNAPSHOT_VERSION,
        "scan_id": str(result.get("scan_id") or ""),
        "scanner_version": str(result.get("scanner_version") or ""),
        "completed_market_session": str(pd.Timestamp(completed_session).date()),
        "universe_identity": universe_identity,
        "universe_ticker_count": len(normalized_universe),
        "decisions": decisions,
        "provider_degradation_summary": _provider_degradation(result),
    }
    return {"snapshot_id": sha256(_canonical(body).encode("utf-8")).hexdigest(), **body}


def reload_final_decision_snapshot(artifact_rows: pd.DataFrame | Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = artifact_rows.to_dict("records") if isinstance(artifact_rows, pd.DataFrame) else list(artifact_rows)
    matching = [row for row in records if str(row.get("artifact_type") or "") == FINAL_DECISION_SNAPSHOT_VERSION]
    if not matching:
        raise ValueError("FINAL_DECISION_SNAPSHOT_NOT_FOUND")
    payload = matching[-1].get("payload")
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, Mapping):
        raise ValueError("FINAL_DECISION_SNAPSHOT_INVALID_PAYLOAD")
    snapshot = _safe(dict(payload))
    claimed = str(snapshot.pop("snapshot_id", ""))
    actual = sha256(_canonical(snapshot).encode("utf-8")).hexdigest()
    if not claimed or claimed != actual:
        raise ValueError("FINAL_DECISION_SNAPSHOT_HASH_MISMATCH")
    return {"snapshot_id": claimed, **snapshot}


__all__ = [
    "FINAL_DECISION_SNAPSHOT_VERSION",
    "build_final_decision_snapshot",
    "reload_final_decision_snapshot",
]
