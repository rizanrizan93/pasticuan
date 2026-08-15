from __future__ import annotations

"""Hot-reload-safe production integrity hooks for Super Scanner."""

from functools import wraps
from typing import Any, Iterable
import copy
import hashlib
import time

import pandas as pd

from evidence_governance import apply_three_rank_contract
from official_evidence_bridge import combine_project_management, corporate_events_to_narrative
from live_forward_evidence import collect_live_forward_evidence, collection_coverage

PATCH_VERSION = "1.2.1-live-forward"
_NEGATIVE_RESULTS: dict[tuple[str, str], tuple[float, Any, BaseException | None]] = {}
_GOVERNED_EVIDENCE_CACHE: dict[tuple[int, tuple[str, ...]], tuple[float, dict[str, pd.DataFrame], pd.DataFrame]] = {}
_PROVIDER_TTLS = {
    "fetch_resilient_fundamentals": 1800,
    "fetch_idx_fundamental_history": 3600,
    "fetch_yahoo_fundamental_history": 1800,
    "fetch_resilient_market_status": 900,
    "fetch_resilient_news_review": 900,
}


def _ticker(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text if text.endswith(".JK") else f"{text}.JK" if text else ""


def _clone(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    if isinstance(value, list):
        return list(value)
    if isinstance(value, dict):
        return dict(value)
    try:
        return copy.copy(value)
    except Exception:
        return value


def _negative(value: Any) -> bool:
    if isinstance(value, pd.DataFrame):
        return value.empty
    if isinstance(value, tuple):
        frames = [item for item in value if isinstance(item, pd.DataFrame)]
        return bool(frames) and all(frame.empty for frame in frames)
    if isinstance(value, dict):
        return not value
    return value is None


def _request_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    def norm(value: Any) -> str:
        if isinstance(value, pd.DataFrame):
            return f"DF:{len(value)}:{','.join(map(str, value.columns[:8]))}"
        if isinstance(value, (list, tuple, set)):
            return "[" + ",".join(sorted(map(str, value))) + "]"
        if isinstance(value, dict):
            return "{" + ",".join(
                f"{key}={value[key]}" for key in sorted(value)
                if "key" not in str(key).lower() and "secret" not in str(key).lower()
            ) + "}"
        return str(value)
    raw = "|".join([*(norm(item) for item in args), norm(kwargs)])
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()[:24]


def _wrap_negative_cache(owner: Any, name: str) -> None:
    original = getattr(owner, name, None)
    if not callable(original) or getattr(original, "__provider_negative_cache_v1__", False):
        return
    ttl = int(_PROVIDER_TTLS.get(name, 900))

    @wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        key = (name, _request_key(args, kwargs))
        cached = _NEGATIVE_RESULTS.get(key)
        now = time.monotonic()
        if cached and cached[0] > now:
            if cached[2] is not None:
                raise type(cached[2])(str(cached[2]))
            return _clone(cached[1])
        if cached:
            _NEGATIVE_RESULTS.pop(key, None)
        try:
            result = original(*args, **kwargs)
        except Exception as exc:
            _NEGATIVE_RESULTS[key] = (now + ttl, None, exc)
            raise
        if _negative(result):
            _NEGATIVE_RESULTS[key] = (now + ttl, _clone(result), None)
        else:
            _NEGATIVE_RESULTS.pop(key, None)
        return result

    wrapped.__provider_negative_cache_v1__ = True
    setattr(owner, name, wrapped)


def _wrap_focus_builder(owner: Any, name: str, *, swing: bool = False) -> None:
    original = getattr(owner, name, None)
    if not callable(original) or getattr(original, "__three_rank_contract_v1__", False):
        return

    @wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> pd.DataFrame:
        out = original(*args, **kwargs)
        if not isinstance(out, pd.DataFrame) or out.empty:
            return out
        raw_col = "raw_ranking_score" if "raw_ranking_score" in out.columns else ("v9_swing_score" if swing else "v9_next_leader_score")
        guarded_col = "ranking_score" if "ranking_score" in out.columns else raw_col
        production_col = next((column for column in (
            "real_money_candidate_score", "execution_priority_score",
            "v9_swing_score" if swing else "v9_next_leader_score", guarded_col,
        ) if column in out.columns), guarded_col)
        ranked = apply_three_rank_contract(
            out,
            raw_score_col=raw_col,
            guarded_score_col=guarded_col,
            production_score_col=production_col,
            research_eligible_col="rank_eligible" if "rank_eligible" in out.columns else None,
            guarded_eligible_col="rank_eligible" if "rank_eligible" in out.columns else None,
            production_eligible_cols=("real_money_authorization_pass",),
        )
        ranked["ranking_contract_version"] = PATCH_VERSION
        ranked["production_rank_is_strict_real_money_authorization"] = True
        return ranked

    wrapped.__three_rank_contract_v1__ = True
    setattr(owner, name, wrapped)


def _read_governed_evidence(bridge: Any, tickers: Iterable[Any]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    names = tuple(sorted(dict.fromkeys(_ticker(value) for value in tickers if _ticker(value))))
    key = (id(bridge), names)
    now = time.monotonic()
    cached = _GOVERNED_EVIDENCE_CACHE.get(key)
    if cached and cached[0] > now:
        return ({name: frame.copy() for name, frame in cached[1].items()}, cached[2].copy())
    empty = {name: pd.DataFrame() for name in ("project_events", "management_roles", "ownership_events", "corporate_events")}
    if not names or getattr(getattr(bridge, "settings", None), "mode", "") != "SUPABASE_REST":
        return empty, pd.DataFrame()
    universe = set(names)
    specs = {
        "project_events": {"project_source_quorum_verified": "eq.true", "entity_match_verified": "eq.true", "order": "evidence_date.desc"},
        "management_roles": {"verified": "eq.true", "source_quorum_verified": "eq.true", "entity_match_verified": "eq.true", "order": "updated_at.desc"},
        "ownership_events": {"verified": "eq.true", "source_quorum_verified": "eq.true", "entity_match_verified": "eq.true", "order": "report_date.desc"},
        "corporate_events": {"verified": "eq.true", "source_quorum_verified": "eq.true", "entity_match_verified": "eq.true", "order": "event_date.desc"},
    }
    frames: dict[str, pd.DataFrame] = {}
    audits: list[dict[str, Any]] = []
    for table, filters in specs.items():
        params = {"select": "*", "limit": str(max(500, min(5000, len(names) * 8))), **filters}
        try:
            frame = pd.DataFrame(bridge._get_rows(table, params))
            if not frame.empty and "ticker" in frame.columns:
                frame["ticker"] = frame["ticker"].map(_ticker)
                frame = frame.loc[frame["ticker"].isin(universe)].copy()
            frames[table] = frame
            audits.append({"provider": "SUPABASE_GOVERNED_OFFICIAL_EVIDENCE", "scope": table, "status": "DATABASE_CURRENT" if len(frame) else "NO_ITEMS", "rows": len(frame)})
        except Exception as exc:
            frames[table] = pd.DataFrame()
            audits.append({"provider": "SUPABASE_GOVERNED_OFFICIAL_EVIDENCE", "scope": table, "status": "READ_FAIL_SOFT", "rows": 0, "error": f"{type(exc).__name__}: {str(exc)[:240]}"})
    audit = pd.DataFrame(audits)
    _GOVERNED_EVIDENCE_CACHE[key] = (now + 300.0, {name: frame.copy() for name, frame in frames.items()}, audit.copy())
    while len(_GOVERNED_EVIDENCE_CACHE) > 8:
        _GOVERNED_EVIDENCE_CACHE.pop(next(iter(_GOVERNED_EVIDENCE_CACHE)))
    return frames, audit


def _recent_collection_tickers(frame: pd.DataFrame, max_age_hours: float = 24.0) -> set[str]:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "ticker" not in frame.columns:
        return set()
    local = frame.copy()
    local["ticker"] = local["ticker"].map(_ticker)
    state = local.get("forward_collection_state", pd.Series("", index=local.index)).fillna("").astype(str)
    stamp = pd.Series(pd.NaT, index=local.index, dtype="datetime64[ns, UTC]")
    for column in ("source_checked_at", "database_source_checked_at", "last_verified_at"):
        if column in local.columns:
            parsed = pd.to_datetime(local[column], errors="coerce", utc=True)
            stamp = stamp.where(stamp.notna(), parsed)
    age = (pd.Timestamp.now(tz="UTC") - stamp).dt.total_seconds() / 3600.0
    return set(local.loc[state.str.len().gt(0) & age.between(0.0, float(max_age_hours)), "ticker"])


def _persist_live_forward(bridge: Any, frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    try:
        report = bridge.persist_scan_result(
            {"mode": "live_forward_evidence_refresh", "scanner_version": getattr(bridge, "scanner_version", ""), "as_of": pd.Timestamp.now(tz="UTC"), "project_management_review": frame},
            tables=("forward_quality_cache",),
        )
        return report if isinstance(report, pd.DataFrame) else pd.DataFrame()
    except Exception as exc:
        return pd.DataFrame([{"provider": "LIVE_FORWARD_EVIDENCE_PERSIST", "state": "FAIL_SOFT", "error": f"{type(exc).__name__}: {str(exc)[:240]}"}])


def _wrap_forward_quality_reader(database_cls: Any) -> None:
    original = getattr(database_cls, "read_forward_quality_cache", None)
    if not callable(original) or getattr(original, "__live_forward_evidence_v2__", False):
        return

    @wraps(original)
    def wrapped(self: Any, tickers: Iterable[Any]):
        names = list(dict.fromkeys(_ticker(value) for value in tickers if _ticker(value)))
        cached_forward, cache_audit = original(self, names)
        governed, governed_audit = _read_governed_evidence(self, names)
        combined = combine_project_management(cached_forward, governed.get("project_events"), governed.get("management_roles"), governed.get("ownership_events"), governed.get("corporate_events"))
        bridge_rows = sum(len(frame) for frame in governed.values() if isinstance(frame, pd.DataFrame))

        recent = _recent_collection_tickers(cached_forward)
        missing = [ticker for ticker in names if ticker not in recent]
        live = pd.DataFrame()
        persist_report = pd.DataFrame()
        if missing and getattr(getattr(self, "settings", None), "mode", "") == "SUPABASE_REST":
            live = collect_live_forward_evidence(missing, lookback_days=120, max_workers=12, timeout=4.0)
            persist_report = _persist_live_forward(self, live)
            if not live.empty:
                combined = pd.concat([combined, live], ignore_index=True, sort=False) if not combined.empty else live.copy()

        audits = [frame for frame in (cache_audit, governed_audit, persist_report) if isinstance(frame, pd.DataFrame) and not frame.empty]
        audit = pd.concat(audits, ignore_index=True, sort=False) if audits else pd.DataFrame()
        if bridge_rows:
            audit = pd.concat([audit, pd.DataFrame([{
                "provider": "OFFICIAL_EVIDENCE_BRIDGE", "scope": "PROJECT_MANAGEMENT_OWNERSHIP_CAPITAL", "status": "BRIDGED", "rows": bridge_rows,
                "detail": "Strict governed raw facts converted to scanner-consumable evidence without relaxing production quorum.",
            }])], ignore_index=True, sort=False)
        cov = collection_coverage(combined, names)
        audit = pd.concat([audit, pd.DataFrame([{
            "provider": "LIVE_FORWARD_EVIDENCE", "scope": "FULL_SCAN_UNIVERSE",
            "status": "COLLECTED_AND_PERSISTED" if not missing or not live.empty else "COLLECTION_UNAVAILABLE",
            "rows": len(combined), "requested_tickers": len(names), "refresh_tickers": len(missing), "collection_coverage_pct": cov["coverage_pct"],
            "detail": "Research collection coverage is separate from strict official scoring coverage.",
        }])], ignore_index=True, sort=False)
        return combined.reset_index(drop=True), audit

    wrapped.__live_forward_evidence_v2__ = True
    setattr(database_cls, "read_forward_quality_cache", wrapped)


def _wrap_narrative_event_reader(database_cls: Any) -> None:
    original = getattr(database_cls, "read_narrative_events", None)
    if not callable(original) or getattr(original, "__governed_event_bridge_v1__", False):
        return

    @wraps(original)
    def wrapped(self: Any, tickers: Iterable[Any], *, limit: int = 10000):
        existing = original(self, tickers, limit=limit)
        governed, _ = _read_governed_evidence(self, tickers)
        direct = corporate_events_to_narrative(governed.get("corporate_events"))
        frames = [frame for frame in (existing, direct) if isinstance(frame, pd.DataFrame) and not frame.empty]
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True, sort=False)
        if "narrative_event_id" in out.columns:
            out = out.drop_duplicates("narrative_event_id", keep="last")
        return out.reset_index(drop=True)

    wrapped.__governed_event_bridge_v1__ = True
    setattr(database_cls, "read_narrative_events", wrapped)


def _ensure_forward_cache_final_persistence(engine_module: Any) -> None:
    for name in ("LEAN_FINAL_PERSISTENCE_TABLES", "FULL_FINAL_PERSISTENCE_TABLES"):
        current = tuple(getattr(engine_module, name, ()) or ())
        if "forward_quality_cache" not in current:
            setattr(engine_module, name, (*current, "forward_quality_cache"))


def install(expected_release: str = "") -> dict[str, Any]:
    import simple_focus
    import resumable_app_engine
    import scanner_database

    _wrap_focus_builder(simple_focus, "build_next_leaders", swing=False)
    _wrap_focus_builder(simple_focus, "build_swing_ready", swing=True)
    for name in _PROVIDER_TTLS:
        _wrap_negative_cache(resumable_app_engine, name)
    _wrap_forward_quality_reader(scanner_database.ScannerDatabaseBridge)
    _wrap_narrative_event_reader(scanner_database.ScannerDatabaseBridge)
    _ensure_forward_cache_final_persistence(resumable_app_engine)
    return {
        "patch_version": PATCH_VERSION,
        "release": expected_release,
        "ranking_contract": "RAW_RESEARCH|GUARDED_DECISION_PRIORITY|PRODUCTION_REAL_MONEY",
        "official_evidence_bridge": "PROJECT|MANAGEMENT|OWNERSHIP|CAPITAL_ACTION",
        "live_forward_collection": "FULL_UNIVERSE_MISSING_OR_STALE_24H",
        "live_forward_persistence": "FORWARD_QUALITY_CACHE",
    }


__all__ = ["PATCH_VERSION", "install"]
