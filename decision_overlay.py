from __future__ import annotations

"""Pure decision guardrails layered on top of the v9 core scanner.

This module deliberately has no dependency on scanner.py, database code, Streamlit,
or network providers.  It can therefore be tested independently and changed without
risking data acquisition or core fundamental scoring.
"""

from typing import Any, Mapping

import numpy as np
import pandas as pd


DECISION_OVERLAY_VERSION = "1.1.0-execution-plan-integrity"
HORIZONS = (20, 60, 120, 252, 504, 756)
HORIZON_WEIGHTS = {20: 0.22, 60: 0.22, 120: 0.18, 252: 0.15, 504: 0.13, 756: 0.10}


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


def _clip(value: Any, lower: float = 0.0, upper: float = 100.0) -> float:
    parsed = _finite(value, np.nan)
    if not np.isfinite(parsed):
        return np.nan
    return float(np.clip(parsed, lower, upper))


def _series(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _prepare_inventory_features(frame: pd.DataFrame) -> pd.DataFrame:
    close = _series(frame, "Close")
    high = _series(frame, "High")
    low = _series(frame, "Low")
    volume = _series(frame, "Volume")
    valid = close.notna() & high.notna() & low.notna() & volume.notna() & volume.ge(0)
    span = (high - low).replace(0.0, np.nan)
    location = ((close - low) / span).clip(0.0, 1.0).fillna(0.5)
    returns = close.pct_change().fillna(0.0)
    value = (close * volume).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    avg_volume = volume.rolling(20, min_periods=3).mean().replace(0.0, np.nan)
    vol_ratio = (volume / avg_volume).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    obv_step = np.sign(close.diff()).fillna(0.0) * volume.fillna(0.0)
    return pd.DataFrame({
        "close": close,
        "volume": volume,
        "valid": valid,
        "location": location,
        "returns": returns,
        "value": value,
        "vol_ratio": vol_ratio,
        "obv_step": obv_step,
    }, index=frame.index)


def _window_inventory_score(features: pd.DataFrame, horizon: int) -> tuple[float, float, float]:
    """Return inventory score, distribution score and coverage for one horizon."""
    if features is None or features.empty or len(features) < min(horizon, 20):
        return np.nan, np.nan, 0.0
    local = features.tail(horizon)
    valid = local["valid"].fillna(False).astype(bool)
    local_count = int(valid.sum())
    expected = min(horizon, len(local))
    if local_count < max(15, min(horizon, 40) // 2):
        return np.nan, np.nan, 100.0 * local_count / max(1, expected)
    local = local.loc[valid]
    close = local["close"].astype(float)
    volume = local["volume"].astype(float)
    location = local["location"].astype(float)
    returns = local["returns"].astype(float)
    value = local["value"].astype(float)
    vol_ratio = local["vol_ratio"].astype(float)

    positive = ((returns >= -0.004) & (location >= 0.58) & vol_ratio.between(0.60, 1.90))
    absorption = ((returns < 0.0) & (location >= 0.55) & (vol_ratio >= 1.05))
    distribution = ((returns < -0.006) & (location <= 0.38) & (vol_ratio >= 1.25))

    positive_share = float((positive | absorption).mean())
    distribution_share = float(distribution.mean())
    positive_value = float(value[(positive | absorption)].sum())
    negative_value = float(value[distribution].sum())
    value_balance = positive_value / (positive_value + negative_value) if positive_value + negative_value > 0 else 0.5

    obv_delta = _finite(local["obv_step"].sum(), 0.0)
    total_volume = max(_finite(volume.sum(), 0.0), 1.0)
    obv_norm = float(np.clip(obv_delta / total_volume, -1.0, 1.0))
    weighted_location = float(np.average(location, weights=np.maximum(volume, 1.0))) if len(location) else 0.5
    ret = _finite(close.iloc[-1] / close.iloc[0] - 1.0, 0.0) if len(close) > 1 and close.iloc[0] else 0.0
    price_acceptance = np.clip(50.0 + 80.0 * (weighted_location - 0.5), 0.0, 100.0)
    obv_score = np.clip(50.0 + 120.0 * obv_norm, 0.0, 100.0)
    persistence_score = np.clip(100.0 * positive_share / max(0.12, positive_share + distribution_share), 0.0, 100.0)
    balance_score = 100.0 * value_balance
    trend_penalty = max(0.0, min(18.0, 35.0 * max(0.0, -ret - 0.12)))

    score = 0.30 * persistence_score + 0.28 * balance_score + 0.22 * price_acceptance + 0.20 * obv_score - trend_penalty
    distribution_score = np.clip(
        55.0 * min(1.0, distribution_share / 0.18)
        + 30.0 * (1.0 - value_balance)
        + 15.0 * max(0.0, 0.45 - weighted_location) / 0.45,
        0.0,
        100.0,
    )
    coverage = 100.0 * local_count / max(1, expected)
    return float(np.clip(score, 0.0, 100.0)), float(distribution_score), float(np.clip(coverage, 0.0, 100.0))


def inventory_lifecycle_profile(frame: pd.DataFrame) -> dict[str, Any]:
    """Build a multi-horizon inventory/lifecycle profile from OHLCV only.

    This is an issuer-agnostic proxy.  It never claims to identify individual
    brokers or shareholders; direct broker evidence remains a separate evidence class.
    """
    base: dict[str, Any] = {
        "inventory_overlay_version": DECISION_OVERLAY_VERSION,
        "inventory_multi_horizon_score": np.nan,
        "inventory_multi_horizon_coverage_pct": 0.0,
        "distribution_risk_score": np.nan,
        "inventory_lifecycle": "INSUFFICIENT_HISTORY",
        "anti_chase_gate": False,
        "markup_extension_pct": np.nan,
        "reaccumulation_quality_score": np.nan,
        "inventory_reason": "OHLCV history insufficient",
    }
    if not isinstance(frame, pd.DataFrame) or frame.empty or "Close" not in frame.columns:
        return base

    close = _series(frame, "Close").dropna()
    if len(close) < 20:
        return base

    features = _prepare_inventory_features(frame)
    scores: list[tuple[float, float]] = []
    dists: list[tuple[float, float]] = []
    coverages: list[tuple[float, float]] = []
    details: list[str] = []
    for horizon in HORIZONS:
        if len(frame) < min(horizon, 20):
            continue
        score, dist, coverage = _window_inventory_score(features, horizon)
        weight = HORIZON_WEIGHTS[horizon]
        if np.isfinite(score):
            scores.append((score, weight))
            dists.append((dist, weight))
            coverages.append((coverage, weight))
            base[f"inventory_score_{horizon}d"] = round(score, 1)
            base[f"distribution_score_{horizon}d"] = round(dist, 1)
            details.append(f"{horizon}d {score:.0f}")

    if not scores:
        return base
    weight_sum = sum(weight for _, weight in scores)
    inventory = sum(value * weight for value, weight in scores) / weight_sum
    distribution = sum(value * weight for value, weight in dists) / weight_sum
    coverage = sum(value * weight for value, weight in coverages) / weight_sum

    last = float(close.iloc[-1])
    ema20 = _finite(close.ewm(span=20, adjust=False).mean().iloc[-1], np.nan)
    ema50 = _finite(close.ewm(span=50, adjust=False).mean().iloc[-1], np.nan)
    ret20 = _finite(close.iloc[-1] / close.iloc[-21] - 1.0, np.nan) if len(close) >= 21 and close.iloc[-21] else np.nan
    ret60 = _finite(close.iloc[-1] / close.iloc[-61] - 1.0, np.nan) if len(close) >= 61 and close.iloc[-61] else np.nan
    ret120 = _finite(close.iloc[-1] / close.iloc[-121] - 1.0, np.nan) if len(close) >= 121 and close.iloc[-121] else np.nan
    low120 = _finite(close.tail(min(120, len(close))).min(), np.nan)
    extension = 100.0 * (last / low120 - 1.0) if np.isfinite(low120) and low120 > 0 else np.nan

    # Recent range contraction is a reaccumulation proxy after an established advance.
    high = _series(frame, "High")
    low = _series(frame, "Low")
    recent_range = ((high - low) / close.reindex(frame.index)).replace([np.inf, -np.inf], np.nan)
    range20 = _finite(recent_range.tail(20).median(), np.nan)
    range120 = _finite(recent_range.tail(min(120, len(recent_range))).median(), np.nan)
    contraction = range20 / range120 if np.isfinite(range20) and np.isfinite(range120) and range120 > 0 else np.nan
    reaccum_quality = np.clip(
        0.50 * inventory
        + 25.0 * (1.0 - min(1.0, max(0.0, _finite(contraction, 1.0) - 0.65) / 0.75))
        + 25.0 * (1.0 - min(1.0, abs(_finite(ret20, 0.0)) / 0.18)),
        0.0,
        100.0,
    )

    advanced_markup = bool(
        np.isfinite(ret60) and ret60 >= 0.25
        and np.isfinite(extension) and extension >= 35.0
        and np.isfinite(ema50) and last > ema50
    )
    reaccumulating = bool(
        np.isfinite(ret120) and ret120 >= 0.15
        and np.isfinite(ret20) and -0.10 <= ret20 <= 0.12
        and inventory >= 58.0
        and _finite(contraction, 1.0) <= 0.95
        and distribution < 60.0
    )

    if distribution >= 68.0:
        lifecycle = "DISTRIBUTION"
    elif reaccumulating:
        lifecycle = "REACCUMULATION"
    elif advanced_markup:
        lifecycle = "MARKUP"
    elif inventory >= 68.0 and (not np.isfinite(ret60) or ret60 <= 0.20):
        lifecycle = "INVENTORY_COLLECTION"
    elif inventory >= 58.0 and (not np.isfinite(ret20) or ret20 >= -0.05):
        lifecycle = "EARLY_CONVERGENCE"
    else:
        lifecycle = "NEUTRAL"

    anti_chase = bool(lifecycle == "MARKUP" and not reaccumulating and (extension >= 35.0 or _finite(ret60, 0.0) >= 0.30))
    reason = f"Inventory {inventory:.0f}; distribution {distribution:.0f}; lifecycle {lifecycle}; " + ", ".join(details)

    base.update({
        "inventory_multi_horizon_score": round(float(inventory), 1),
        "inventory_multi_horizon_coverage_pct": round(float(coverage), 1),
        "distribution_risk_score": round(float(distribution), 1),
        "inventory_lifecycle": lifecycle,
        "anti_chase_gate": anti_chase,
        "markup_extension_pct": round(float(extension), 1) if np.isfinite(extension) else np.nan,
        "reaccumulation_quality_score": round(float(reaccum_quality), 1),
        "inventory_reason": reason,
        "inventory_proxy_evidence_type": "MULTI_HORIZON_OHLCV_PROXY",
    })
    return base


def enrich_silent_profile(frame: pd.DataFrame, profile: Mapping[str, Any] | None = None) -> dict[str, Any]:
    merged = dict(profile or {})
    merged.update(inventory_lifecycle_profile(frame))
    silent = _finite(merged.get("silent_accumulation_score"), np.nan)
    inventory = _finite(merged.get("inventory_multi_horizon_score"), np.nan)
    distribution = _finite(merged.get("distribution_risk_score"), np.nan)
    values: list[tuple[float, float]] = []
    if np.isfinite(silent):
        values.append((silent, 0.55))
    if np.isfinite(inventory):
        values.append((inventory, 0.45))
    if values:
        denominator = sum(weight for _, weight in values)
        dominance = sum(value * weight for value, weight in values) / denominator
        if np.isfinite(distribution):
            dominance = 0.85 * dominance + 0.15 * (100.0 - distribution)
        merged["accumulation_dominance_pct"] = round(float(np.clip(dominance, 0.0, 100.0)), 1)
    else:
        merged["accumulation_dominance_pct"] = np.nan
    return merged


def lifecycle_priority(value: Any) -> int:
    return {
        "REACCUMULATION": 0,
        "INVENTORY_COLLECTION": 1,
        "EARLY_CONVERGENCE": 2,
        "NEUTRAL": 3,
        "MARKUP": 5,
        "DISTRIBUTION": 9,
        "INSUFFICIENT_HISTORY": 8,
    }.get(str(value or "").strip().upper(), 7)


def apply_execution_plan_integrity(frame: pd.DataFrame, *, model: str) -> pd.DataFrame:
    """Expire consumed or geometrically invalid execution plans.

    Research ranking is preserved, but stale order levels are removed so an old
    setup can never look like a current re-entry instruction after price has
    already reached its first target.  An executable entry zone must also sit
    completely above its stop; otherwise part of the displayed zone would imply
    negative/undefined risk.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    out = frame.copy()

    def num(name: str) -> pd.Series:
        if name not in out.columns:
            return pd.Series(np.nan, index=out.index, dtype=float)
        return pd.to_numeric(out[name], errors="coerce")

    entry = num("entry")
    low = num("entry_low").where(num("entry_low").notna(), entry)
    high = num("entry_high").where(num("entry_high").notna(), entry)
    trigger = num("trigger").where(num("trigger").notna(), num("trigger_price")).where(lambda s: s.notna(), entry)
    stop = num("stop_loss")
    tp1 = num("tp1")
    tp2 = num("tp2")
    last = num("independent_last_price").where(num("independent_last_price").notna(), num("quote_last_price"))
    last = last.where(last.notna(), num("last_price"))
    zone_exec = out.get("entry_zone_is_executable", pd.Series(True, index=out.index)).fillna(False).astype(bool)

    has_plan = low.notna() & high.notna() & stop.notna() & tp1.notna() & tp2.notna()
    invalid_zone = has_plan & ((low > high) | (zone_exec & (stop >= low)))
    invalid_targets = has_plan & ((tp1 <= trigger) | (tp2 <= tp1))
    consumed = has_plan & ~invalid_zone & ~invalid_targets & last.notna() & (last >= tp1)
    invalid = invalid_zone | invalid_targets
    expired = invalid | consumed

    out["execution_plan_integrity_state"] = np.select(
        [invalid_zone, invalid_targets, consumed, has_plan],
        ["INVALID_ENTRY_STOP_GEOMETRY", "INVALID_TARGET_GEOMETRY", "STALE_TARGET_ALREADY_REACHED", "CURRENT_PLAN"],
        default="NO_EXECUTION_PLAN",
    )
    out["execution_plan_is_current"] = has_plan & ~expired
    out["execution_plan_block_reason"] = np.select(
        [invalid_zone, invalid_targets, consumed],
        ["STOP_NOT_BELOW_FULL_EXECUTABLE_ENTRY_ZONE", "TARGETS_NOT_ABOVE_TRIGGER_IN_ORDER", "LATEST_PRICE_ALREADY_REACHED_TP1"],
        default="",
    )

    # Stale/invalid levels are intentionally removed from the current order
    # contract.  The research score and historical thesis remain untouched.
    order_fields = ["entry", "execution_entry", "entry_low", "entry_high", "trigger", "trigger_price", "stop_loss", "tp1", "tp2", "rr1", "rr2"]
    for name in order_fields:
        if name in out.columns:
            out.loc[expired, name] = np.nan
    if "entry_zone_is_executable" in out.columns:
        out.loc[expired, "entry_zone_is_executable"] = False
    if "order_builder_eligible" in out.columns:
        out.loc[expired, "order_builder_eligible"] = False
    if "order_ready" in out.columns:
        out.loc[expired, "order_ready"] = False
    if "recommended_allocation_idr" in out.columns:
        out.loc[expired, "recommended_allocation_idr"] = 0.0
    if "recommended_lots" in out.columns:
        out.loc[expired, "recommended_lots"] = 0
    if "stockbit_order_lots" in out.columns:
        out.loc[expired, "stockbit_order_lots"] = 0

    model_name = str(model or "").strip().upper()
    if "status" in out.columns:
        if model_name == "NEXT_LEADER":
            out.loc[consumed, "status"] = "WAIT"
            out.loc[invalid, "status"] = "RESEARCH_ONLY"
        elif model_name == "SWING_READY":
            out.loc[consumed, "status"] = "WATCHLIST"
            out.loc[invalid, "status"] = "RESEARCH_ONLY"
    if "next_action" in out.columns:
        out.loc[consumed, "next_action"] = "WAIT_NEW_SETUP_AFTER_TARGET_REACHED"
        out.loc[invalid, "next_action"] = "REBUILD_INVALID_EXECUTION_PLAN"
    return out


def apply_methodology_guardrails(frame: pd.DataFrame, *, model: str) -> pd.DataFrame:
    """Apply execution-integrity, anti-chase and distribution guards without altering research score math."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    out = apply_execution_plan_integrity(frame, model=model)
    model_name = str(model or "").strip().upper()
    lifecycle = out.get("inventory_lifecycle", pd.Series("NEUTRAL", index=out.index)).astype(str).str.upper()
    distribution_source = out["distribution_risk_score"] if "distribution_risk_score" in out.columns else pd.Series(np.nan, index=out.index)
    distribution = pd.to_numeric(distribution_source, errors="coerce")
    anti_chase = out.get("anti_chase_gate", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    distribution_block = lifecycle.eq("DISTRIBUTION") | distribution.ge(68.0).fillna(False)

    out["methodology_gate_pass"] = ~distribution_block
    out["methodology_priority"] = lifecycle.map(lifecycle_priority).fillna(7).astype(int)
    out["decision_overlay_state"] = np.select(
        [distribution_block, anti_chase, lifecycle.eq("REACCUMULATION"), lifecycle.eq("INVENTORY_COLLECTION"), lifecycle.eq("EARLY_CONVERGENCE")],
        ["V9_DISTRIBUTION_BLOCK", "V9_WAIT_REACCUMULATION", "V9_REACCUMULATION", "V9_INVENTORY_COLLECTION", "V9_EARLY_CONVERGENCE"],
        default="V9_NEUTRAL_LIFECYCLE",
    )

    if model_name == "NEXT_LEADER":
        if "status" in out.columns:
            out.loc[distribution_block, "status"] = "RESEARCH_ONLY"
            buy_chase = anti_chase & out["status"].astype(str).eq("BUY_ZONE")
            out.loc[buy_chase, "status"] = "WAIT"
        if "recommended_allocation_idr" in out.columns:
            out.loc[distribution_block | anti_chase, "recommended_allocation_idr"] = 0.0
        if "recommended_lots" in out.columns:
            out.loc[distribution_block | anti_chase, "recommended_lots"] = 0
        if "research_recommendation_status" in out.columns and "status" in out.columns:
            out["research_recommendation_status"] = out["status"]
        if "multibagger_status" in out.columns and "status" in out.columns:
            out["multibagger_status"] = np.where(
                out["decision_overlay_state"].eq("V9_WAIT_REACCUMULATION"),
                "WAIT_REACCUMULATION",
                np.where(out["decision_overlay_state"].eq("V9_DISTRIBUTION_BLOCK"), "RESEARCH_ONLY", out["multibagger_status"]),
            )
        if "multibagger_rank_eligible" in out.columns and "status" in out.columns:
            out["multibagger_rank_eligible"] = out["status"].isin(["BUY_ZONE", "WATCH", "WAIT"]) & out["methodology_gate_pass"]
        if "research_eligible" in out.columns and "status" in out.columns:
            out["research_eligible"] = ~out["status"].isin(["REJECT", "DATA_PENDING", "RESEARCH_ONLY"])
    elif model_name == "SWING_READY":
        if "status" in out.columns:
            out.loc[distribution_block, "status"] = "RESEARCH_ONLY"
            chase_exec = anti_chase & out["status"].astype(str).isin(["EXECUTION_READY", "ENTRY_PLAN_READY"])
            out.loc[chase_exec, "status"] = "WATCHLIST"
        if "next_action" in out.columns:
            out.loc[distribution_block, "next_action"] = "AVOID_DISTRIBUTION"
            out.loc[anti_chase & ~distribution_block, "next_action"] = "WAIT_REACCUMULATION"
        if "order_builder_eligible" in out.columns:
            out.loc[distribution_block | anti_chase, "order_builder_eligible"] = False
        if "order_ready" in out.columns:
            out.loc[distribution_block | anti_chase, "order_ready"] = False
        if "decision_state" in out.columns and "status" in out.columns:
            out["decision_state"] = out["status"]
    return out


__all__ = [
    "DECISION_OVERLAY_VERSION",
    "HORIZONS",
    "inventory_lifecycle_profile",
    "enrich_silent_profile",
    "lifecycle_priority",
    "apply_execution_plan_integrity",
    "apply_methodology_guardrails",
]
