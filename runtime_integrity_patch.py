from __future__ import annotations

"""Hot-reload-safe integrity patch installed by runtime_release.

Only public ranking labels and failure-cache behaviour are changed here.  Score
formulae remain owned by their native modules.  Provider failures are cached in
process for a short TTL; successful/partial responses are never cached here.
"""

from functools import wraps
from typing import Any
import copy
import hashlib
import time

import pandas as pd

from evidence_governance import apply_three_rank_contract

PATCH_VERSION = "1.0.0"
_NEGATIVE_RESULTS: dict[tuple[str, str], tuple[float, Any, BaseException | None]] = {}
_PROVIDER_TTLS = {
    "fetch_resilient_fundamentals": 1800,
    "fetch_idx_fundamental_history": 3600,
    "fetch_yahoo_fundamental_history": 1800,
    "fetch_resilient_market_status": 900,
    "fetch_resilient_news_review": 900,
}


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


def install(expected_release: str = "") -> dict[str, Any]:
    import simple_focus
    import resumable_app_engine

    _wrap_focus_builder(simple_focus, "build_next_leaders", swing=False)
    _wrap_focus_builder(simple_focus, "build_swing_ready", swing=True)
    for name in _PROVIDER_TTLS:
        # resumable_app_engine imports provider callables into module globals;
        # patching those references avoids touching technical/OHLCV code paths.
        _wrap_negative_cache(resumable_app_engine, name)
    return {
        "patch_version": PATCH_VERSION,
        "release": expected_release,
        "negative_cache_entries": len(_NEGATIVE_RESULTS),
        "ranking_contract": "RAW_RESEARCH|GUARDED_DECISION_PRIORITY|PRODUCTION_REAL_MONEY",
    }


__all__ = ["PATCH_VERSION", "install"]
