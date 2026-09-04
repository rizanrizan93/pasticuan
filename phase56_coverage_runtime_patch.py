from __future__ import annotations

"""Phase 5.6 coverage semantics for PASTICUAN.

The adapter closes three research/coverage gaps without weakening production
execution gates:
- expose public ownership-concentration context with explicit non-KSEI lineage;
- allow non-quorum forward evidence to contribute only to a capped research
  Future Fundamental tier below the 40% direct-evidence authorization floor;
- calculate technical research readiness from observed EMA/ADX/CMF/RSI/ROC
  measurements when the legacy synthetic momentum field is absent.

No missing RR, direct forward quorum, regulatory free float, or execution
authorization is inferred by this module.
"""

import time
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import requests

from scanner_database import DatabaseSettings

PATCH_VERSION = "1.0.0-phase5.6-coverage-semantics"
CACHE_TTL_SECONDS = 6 * 60 * 60

_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_AT = 0.0


def _ticker(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    return text if text.endswith(".JK") else f"{text}.JK"


def _bare(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text[:-3] if text.endswith(".JK") else text


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _clip(value: Any, lower: float = 0.0, upper: float = 100.0) -> float:
    number = _finite(value)
    return float(np.clip(number, lower, upper)) if np.isfinite(number) else np.nan


def _load_public_ownership(tickers: Iterable[str] | None = None) -> dict[str, dict[str, Any]]:
    """Read the scanner-neutral projection using the existing backend DB credential."""
    global _CACHE, _CACHE_AT
    now = time.monotonic()
    if _CACHE and now - _CACHE_AT < CACHE_TTL_SECONDS:
        return {key: dict(value) for key, value in _CACHE.items()}

    settings = DatabaseSettings.from_env()
    if settings.mode != "SUPABASE_REST" or not settings.supabase_url or not settings.supabase_key:
        return {key: dict(value) for key, value in _CACHE.items()}

    symbols = list(dict.fromkeys(_bare(value) for value in (tickers or []) if _bare(value)))
    chunks: list[list[str] | None] = [None] if not symbols else [symbols[start:start + 80] for start in range(0, len(symbols), 80)]
    rows: list[dict[str, Any]] = []
    headers = {
        "apikey": settings.supabase_key,
        "Accept": "application/json",
        "Accept-Profile": settings.schema,
    }
    if settings.supabase_key_type == "SERVICE_ROLE":
        headers["Authorization"] = f"Bearer {settings.supabase_key}"
    try:
        with requests.Session() as http:
            for chunk in chunks:
                params: dict[str, Any] = {
                    "select": "ticker,source_period,observed_on,insiders_held_pct,institutions_held_pct,institutions_float_held_pct,institutions_count,coverage_pct,source_authority,official_verified,provenance_state,source_state,refreshed_at",
                    "limit": 1000,
                }
                if chunk:
                    quoted = ",".join(f'"{symbol}"' for symbol in chunk)
                    params["ticker"] = f"in.({quoted})"
                response = http.get(
                    f"{settings.supabase_url}/rest/v1/phase56_public_ownership_snapshots",
                    params=params,
                    headers=headers,
                    timeout=min(10.0, float(settings.timeout_seconds)),
                )
                if response.status_code != 200:
                    return {key: dict(value) for key, value in _CACHE.items()}
                payload = response.json()
                if not isinstance(payload, list):
                    return {key: dict(value) for key, value in _CACHE.items()}
                rows.extend(dict(item) for item in payload if isinstance(item, Mapping))
    except Exception:
        return {key: dict(value) for key, value in _CACHE.items()}

    fresh: dict[str, dict[str, Any]] = {}
    for row in rows:
        ticker = _ticker(row.get("ticker"))
        if not ticker:
            continue
        fresh[ticker] = {
            "ownership_public_insiders_held_pct": row.get("insiders_held_pct"),
            "ownership_public_institutions_held_pct": row.get("institutions_held_pct"),
            "ownership_public_institutions_float_held_pct": row.get("institutions_float_held_pct"),
            "ownership_public_institutions_count": row.get("institutions_count"),
            "ownership_public_context_coverage_pct": row.get("coverage_pct") or 0.0,
            "ownership_public_source_period": row.get("source_period"),
            "ownership_public_observed_on": row.get("observed_on"),
            "ownership_public_source_authority": str(row.get("source_authority") or "PUBLIC_PROVIDER"),
            "ownership_public_official_verified": bool(row.get("official_verified", False)),
            "ownership_public_context_provenance_state": str(row.get("provenance_state") or "PUBLIC_PROVIDER_NOT_IDX_KSEI"),
            "ownership_public_context_state": "CONTEXT_ONLY_NOT_REGULATORY_FREE_FLOAT",
        }
    if fresh:
        _CACHE = fresh
        _CACHE_AT = now
    return {key: dict(value) for key, value in _CACHE.items()}


def _technical_component(row: Mapping[str, Any]) -> tuple[float, float, str]:
    """Research readiness from observed indicators; never invent execution RR."""
    state = str(row.get("sig_setup_status") or row.get("sig_status") or row.get("sig_decision_state") or "").upper()
    state_cap = {
        "EXECUTION_READY": 100.0,
        "ENTRY_PLAN_READY": 90.0,
        "READY_FOR_PRICE_VERIFY": 82.0,
        "READY_FOR_STOCKBIT_VERIFY": 80.0,
        "WATCHLIST_ENTRY": 72.0,
        "WATCHLIST": 68.0,
        "PENDING": 50.0,
        "REJECT": 20.0,
    }.get(state, 100.0)

    quality = next(
        (_finite(row.get(name)) for name in ("sig_quality_score", "sig_analyst_fusion_score", "sig_setup_score") if np.isfinite(_finite(row.get(name)))),
        np.nan,
    )
    last = _finite(row.get("sig_last_price"))
    ema20 = _finite(row.get("sig_ema20"))
    ema50 = _finite(row.get("sig_ema50"))
    ema200 = _finite(row.get("sig_ema200"))
    adx = _finite(row.get("sig_adx14"))
    cmf = _finite(row.get("sig_cmf20"))
    if not np.isfinite(quality):
        quality_parts: list[float] = []
        if all(np.isfinite(value) and value > 0 for value in (last, ema20, ema50, ema200)):
            trend_votes = int(last > ema20) + int(ema20 > ema50) + int(ema50 > ema200)
            quality_parts.append(35.0 + 20.0 * trend_votes)
        if np.isfinite(adx):
            quality_parts.append(_clip(25.0 + 2.0 * adx))
        if np.isfinite(cmf):
            quality_parts.append(_clip(50.0 + 250.0 * cmf))
        quality = float(np.mean(quality_parts)) if quality_parts else np.nan

    momentum = _finite(row.get("sig_momentum_score"))
    if np.isfinite(momentum) and momentum <= 12:
        momentum = momentum * 100.0 / 12.0
    if not np.isfinite(momentum):
        rsi = _finite(row.get("sig_rsi14"))
        roc60 = _finite(row.get("sig_roc60"))
        rs60 = _finite(row.get("sig_relative_strength60"))
        momentum_parts: list[float] = []
        if np.isfinite(rsi):
            momentum_parts.append(_clip(100.0 - abs(rsi - 60.0) * 2.5))
        if np.isfinite(roc60):
            momentum_parts.append(_clip(50.0 + 2.0 * roc60))
        if np.isfinite(rs60):
            momentum_parts.append(_clip(rs60 if abs(rs60) > 2.0 else 50.0 + 200.0 * rs60))
        momentum = float(np.mean(momentum_parts)) if momentum_parts else np.nan

    rr1 = _finite(row.get("sig_rr1"))
    rr_score = _clip(40.0 + 25.0 * rr1) if np.isfinite(rr1) else np.nan
    values = [(quality, 0.45, "QUALITY"), (momentum, 0.30, "MOMENTUM"), (rr_score, 0.25, "RR")]
    observed = [(value, weight, label) for value, weight, label in values if np.isfinite(value)]
    if not observed:
        return np.nan, 0.0, ""
    observed_weight = sum(weight for _, weight, _ in observed)
    score = sum(_clip(value) * weight for value, weight, _ in observed) / observed_weight
    score = min(float(score), state_cap)
    basis = " | ".join(label for _, _, label in observed)
    if state:
        basis = f"{basis} | STATE_CAP:{state}"
    if not np.isfinite(rr1):
        basis = f"{basis} | RR_MISSING_NO_EXECUTION_INFERENCE"
    return score, 100.0 * observed_weight, basis


def _future_component(original: Any, row: Mapping[str, Any]) -> tuple[float, float, str]:
    direct_score, direct_coverage, direct_basis = original(row)
    if np.isfinite(_finite(direct_score)) and float(direct_coverage or 0.0) > 0.0:
        return direct_score, direct_coverage, direct_basis

    # Research-only non-quorum forward evidence. Coverage is hard-capped below
    # NEXT_LEADER_MIN_FUTURE_COVERAGE_PCT (40), so it cannot authorize the direct lane.
    pipeline = _finite(row.get("nar_forward_project_pipeline_score"))
    impact = _finite(row.get("nar_forward_future_fundamental_impact_score"))
    evidence_coverage = _finite(row.get("nar_forward_project_data_coverage_pct"), 0.0)
    parts = [(pipeline, 0.45, "PROJECT_PIPELINE"), (impact, 0.55, "FUTURE_IMPACT")]
    observed = [(value, weight, label) for value, weight, label in parts if np.isfinite(value)]
    if observed:
        observed_weight = sum(weight for _, weight, _ in observed)
        score = sum(_clip(value) * weight for value, weight, _ in observed) / observed_weight
        coverage = min(35.0, max(5.0, evidence_coverage) * observed_weight)
        return min(float(score), 68.0), coverage, "RESEARCH_NON_QUORUM_FORWARD | " + " | ".join(label for _, _, label in observed)

    issuer_score = _finite(row.get("nar_issuer_alignment_effective_score"))
    issuer_coverage = _finite(row.get("nar_issuer_alignment_coverage_pct"), 0.0)
    if np.isfinite(issuer_score) and issuer_coverage > 0.0:
        return (
            min(_clip(issuer_score), 60.0),
            min(25.0, issuer_coverage * 0.25),
            "RESEARCH_ISSUER_EVENT_PROXY_NOT_DIRECT_FORWARD",
        )
    return direct_score, direct_coverage, direct_basis


def _merge_context(frame: pd.DataFrame, context: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "ticker" not in frame.columns:
        return frame
    rows = [{"ticker": ticker, **dict(values)} for ticker, values in context.items()]
    if not rows:
        return frame
    left = frame.copy()
    left["ticker"] = left["ticker"].map(_ticker)
    right = pd.DataFrame(rows)
    return left.merge(right, on="ticker", how="left", suffixes=("", "_phase56"))


def install() -> dict[str, str]:
    import simple_focus as focus

    if getattr(focus, "_phase56_coverage_runtime_patch", "") == PATCH_VERSION:
        return {"patch_version": PATCH_VERSION, "state": "ALREADY_INSTALLED"}

    original_future = focus._future_component
    original_build = focus.build_simple_focus

    def future_with_research_tier(row: Mapping[str, Any]):
        return _future_component(original_future, row)

    def build_with_public_ownership(*args: Any, **kwargs: Any):
        result = original_build(*args, **kwargs)
        context = _load_public_ownership(kwargs.get("universe_tickers") or [])
        if isinstance(result, Mapping) and context:
            result = dict(result)
            for key, value in list(result.items()):
                if isinstance(value, pd.DataFrame) and not value.empty and "ticker" in value.columns:
                    result[key] = _merge_context(value, context)
        return result

    focus._future_component = future_with_research_tier
    focus._technical_component = _technical_component
    focus.build_simple_focus = build_with_public_ownership
    focus._phase56_coverage_runtime_patch = PATCH_VERSION

    # resumable_app_engine imports build_simple_focus by value at module import.
    try:
        import resumable_app_engine as app_engine
        app_engine.build_simple_focus = build_with_public_ownership
    except Exception:
        pass

    return {
        "patch_version": PATCH_VERSION,
        "ownership": "PUBLIC_CONTEXT_EXPOSED_NOT_KSEI",
        "future": "RESEARCH_FALLBACK_CAPPED_BELOW_DIRECT_GATE",
        "technical": "OBSERVED_INDICATORS_FILL_RESEARCH_COVERAGE_NO_RR_INFERENCE",
    }


__all__ = ["PATCH_VERSION", "install", "_technical_component", "_load_public_ownership"]
