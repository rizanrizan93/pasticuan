from __future__ import annotations

"""Hot-reload-safe integrity patch installed by runtime_release.

Production hooks keep ranking labels consistent, cache provider failures, and
bridge governed official evidence into the same inputs consumed by the scanner.
Raw evidence tables remain factual stores; scoring is performed by
``official_evidence_bridge`` and is explicitly labelled scanner-derived.
"""

from functools import wraps
from typing import Any, Iterable
import copy
import hashlib
import time

import pandas as pd

from evidence_governance import apply_three_rank_contract
from official_evidence_bridge import combine_project_management, corporate_events_to_narrative

PATCH_VERSION = "1.1.0"
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
            return "{" + ",".join(f"{k}={value[k]}" for k in sorted(value) if 'key' not in str(k).lower() and 'secret' not in str(k).lower()) + "}"
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
    wrapped.__provider_negative_cache_original__ = original
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
        production_col = next((column for column in ("real_money_candidate_score", "execution_priority_score", "v9_swing_score" if swing else "v9_next_leader_score", guarded_col) if column in out.columns), guarded_col)
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
    wrapped.__three_rank_contract_original__ = original
    setattr(owner, name, wrapped)


def _read_governed_evidence(bridge: Any, tickers: Iterable[Any]) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    names = tuple(sorted(dict.fromkeys(_ticker(value) for value in tickers if _ticker(value))))
    key = (id(bridge), names)
    now = time.monotonic()
    cached = _GOVERNED_EVIDENCE_CACHE.get(key)
    if cached and cached[0] > now:
        return ({name: frame.copy() for name, frame in cached[1].items()}, cached[2].copy())
    if not names or getattr(getattr(bridge, "settings", None), "mode", "") != "SUPABASE_REST":
        return ({name: pd.DataFrame() for name in ("project_events", "management_roles", "ownership_events", "corporate_events")}, pd.DataFrame())
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
            rows = bridge._get_rows(table, params)
            frame = pd.DataFrame(rows)
            if not frame.empty and "ticker" in frame.columns:
                frame = frame.loc[frame["ticker"].map(_ticker).isin(universe)].copy()
                frame["ticker"] = frame["ticker"].map(_ticker)
            frames[table] = frame
            audits.append({
                "provider": "SUPABASE_GOVERNED_OFFICIAL_EVIDENCE",
                "scope": table,
                "status": "DATABASE_CURRENT" if len(frame) else "NO_ITEMS",
                "rows": len(frame),
                "bridge_version": PATCH_VERSION,
            })
        except Exception as exc:
            frames[table] = pd.DataFrame()
            audits.append({
                "provider": "SUPABASE_GOVERNED_OFFICIAL_EVIDENCE",
                "scope": table,
                "status": "READ_FAIL_SOFT",
                "rows": 0,
                "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                "bridge_version": PATCH_VERSION,
            })
    audit = pd.DataFrame(audits)
    _GOVERNED_EVIDENCE_CACHE[key] = (
        now + 300.0,
        {name: frame.copy() for name, frame in frames.items()},
        audit.copy(),
    )
    while len(_GOVERNED_EVIDENCE_CACHE) > 8:
        oldest = next(iter(_GOVERNED_EVIDENCE_CACHE))
        _GOVERNED_EVIDENCE_CACHE.pop(oldest, None)
    return frames, audit


def _wrap_forward_quality_reader(database_cls: Any) -> None:
    original = getattr(database_cls, "read_forward_quality_cache", None)
    if not callable(original) or getattr(original, "__governed_evidence_bridge_v1__", False):
        return

    @wraps(original)
    def wrapped(self: Any, tickers: Iterable[Any]):
        cached_forward, cache_audit = original(self, tickers)
        governed, governed_audit = _read_governed_evidence(self, tickers)
        combined = combine_project_management(
            cached_forward,
            governed.get("project_events"),
            governed.get("management_roles"),
            governed.get("ownership_events"),
            governed.get("corporate_events"),
        )
        audits = [frame for frame in (cache_audit, governed_audit) if isinstance(frame, pd.DataFrame) and not frame.empty]
        audit = pd.concat(audits, ignore_index=True, sort=False) if audits else pd.DataFrame()
        if not combined.empty:
            audit = pd.concat([audit, pd.DataFrame([{
                "provider": "OFFICIAL_EVIDENCE_BRIDGE",
                "scope": "FORWARD_AND_MANAGEMENT",
                "status": "BRIDGED",
                "rows": len(combined),
                "detail": "Cached forward evidence plus strict project/management/ownership/capital facts.",
            }])], ignore_index=True, sort=False)
        return combined, audit

    wrapped.__governed_evidence_bridge_v1__ = True
    wrapped.__governed_evidence_bridge_original__ = original
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
    wrapped.__governed_event_bridge_original__ = original
    setattr(database_cls, "read_narrative_events", wrapped)


def install(expected_release: str = "") -> dict[str, Any]:
    import simple_focus
    import resumable_app_engine
    import scanner_database

    _wrap_focus_builder(simple_focus, "build_next_leaders", swing=False)
    _wrap_focus_builder(simple_focus, "build_swing_ready", swing=True)
    for name in _PROVIDER_TTLS:
        # resumable_app_engine imports provider callables into module globals;
        # patching those references avoids touching technical/OHLCV code paths.
        _wrap_negative_cache(resumable_app_engine, name)
    _wrap_forward_quality_reader(scanner_database.ScannerDatabaseBridge)
    _wrap_narrative_event_reader(scanner_database.ScannerDatabaseBridge)
    return {
        "patch_version": PATCH_VERSION,
        "release": expected_release,
        "negative_cache_entries": len(_NEGATIVE_RESULTS),
        "ranking_contract": "RAW_RESEARCH|GUARDED_DECISION_PRIORITY|PRODUCTION_REAL_MONEY",
        "official_evidence_bridge": "PROJECT|MANAGEMENT|OWNERSHIP|CAPITAL_ACTION",
    }


__all__ = ["PATCH_VERSION", "install"]
