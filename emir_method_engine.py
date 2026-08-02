from __future__ import annotations

"""Public-method reconstruction inspired by Emir Parengkuan's disclosed process.

This module does not reproduce proprietary material and does not claim to
identify a real beneficial owner from public OHLCV.  It translates the public
framework into auditable states:

1. know a bounded stock universe deeply;
2. classify narrative lifecycle instead of chasing headlines;
3. require agreement between price action, volume/flow and (when supplied)
   broker-summary evidence;
4. react to confirmation, reject crowded distribution;
5. size risk from evidence quality rather than conviction language.
"""

from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


EMIR_METHOD_ENGINE_VERSION = "1.3.0-broker-provenance-contract"


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return _text(value).upper() in {"1", "TRUE", "YES", "Y", "ON", "READY"}


def _clip(value: Any, low: float = 0.0, high: float = 100.0) -> float:
    number = _finite(value, low)
    return max(low, min(high, number))


def _price_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    local = frame.copy()
    if not isinstance(local.index, pd.DatetimeIndex):
        local.index = pd.to_datetime(local.index, errors="coerce")
    local = local[~local.index.isna()].sort_index()
    for column in ("Open", "High", "Low", "Close", "Volume"):
        if column in local:
            local[column] = pd.to_numeric(local[column], errors="coerce")
    return local.dropna(subset=["Close"])


def _latest_number(row: Mapping[str, Any], names: Iterable[str], default: float = np.nan) -> float:
    for name in names:
        value = _finite(row.get(name), np.nan)
        if np.isfinite(value):
            return value
    return default


def _liquidity_score(bucket: str, adtv20_idr: float) -> float:
    mapping = {
        "VERY_LIQUID": 95.0,
        "LIQUID": 85.0,
        "MEDIUM": 70.0,
        "ILLIQUID": 45.0,
        "VERY_ILLIQUID": 25.0,
    }
    key = _text(bucket).upper()
    if key in mapping:
        return mapping[key]
    if adtv20_idr >= 100_000_000_000:
        return 95.0
    if adtv20_idr >= 25_000_000_000:
        return 85.0
    if adtv20_idr >= 7_500_000_000:
        return 70.0
    if adtv20_idr >= 1_500_000_000:
        return 45.0
    if adtv20_idr > 0:
        return 25.0
    return 0.0


def _broker_summary_component(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Score explicit broker-code transaction evidence when the user supplies it.

    The scanner never interprets a broker code as the beneficial owner.  The
    legacy CSV contract supplies `broksum_net_ratio` (net/gross transaction
    value) and up to ten observed dates; richer normalized fields can be
    supplied by future providers without changing this interface.
    """
    days = max(0.0, _finite(profile.get("broksum_days"), 0.0))
    explicit_coverage = _finite(profile.get("broker_summary_coverage_pct"), np.nan)
    coverage = _clip(
        explicit_coverage if np.isfinite(explicit_coverage) else 100.0 * min(1.0, days / 10.0),
        0.0,
        100.0,
    )
    quality = _finite(profile.get("broksum_data_quality_pct"), np.nan)
    if np.isfinite(quality):
        coverage *= _clip(quality) / 100.0
    provenance_state = _text(profile.get("broksum_provenance_state")).upper()
    direct_verified = bool(
        _truthy(profile.get("broksum_direct_evidence_eligible"))
        and _truthy(profile.get("broksum_source_verified"))
        and provenance_state == "DIRECT_SOURCE_VERIFIED"
    )
    net_buy = _finite(profile.get("broker_net_buy_20d_pct_adtv"), np.nan)
    legacy_net_ratio = _finite(profile.get("broksum_net_ratio"), np.nan)
    consistency = _finite(profile.get("broker_accumulation_consistency_pct"), np.nan)
    concentration = _finite(profile.get("broker_top3_buy_concentration_pct"), np.nan)
    distribution = _finite(profile.get("broker_distribution_consistency_pct"), np.nan)
    signal = _text(profile.get("broksum_signal")).upper()
    observed = [
        value for value in (net_buy, legacy_net_ratio, consistency, concentration, distribution)
        if np.isfinite(value)
    ]
    if not observed or coverage <= 0.0 or signal == "UNAVAILABLE":
        return {
            "score": np.nan,
            "coverage_pct": 0.0,
            "state": "BROKER_SUMMARY_NOT_AVAILABLE",
            "basis": "",
            "direct_verified": False,
            "provenance_state": provenance_state or "NOT_AVAILABLE",
        }
    components: list[tuple[float, float]] = []
    basis: list[str] = []
    if np.isfinite(net_buy):
        components.append((_clip(50.0 + 2.5 * net_buy), 0.35))
        basis.append("NET_BUY_PCT_ADTV")
    elif np.isfinite(legacy_net_ratio):
        # net/gross ratio is naturally bounded around [-1, +1].
        components.append((_clip(50.0 + 250.0 * legacy_net_ratio), 0.45))
        basis.append("NET_GROSS_RATIO")
    if np.isfinite(consistency):
        components.append((_clip(consistency), 0.25))
        basis.append("CONSISTENCY")
    if np.isfinite(concentration):
        concentration_score = (
            85.0 if 20.0 <= concentration <= 55.0
            else 65.0 if 10.0 <= concentration < 20.0 or 55.0 < concentration <= 70.0
            else 35.0
        )
        components.append((concentration_score, 0.15))
        basis.append("CONCENTRATION")
    if np.isfinite(distribution):
        components.append((_clip(100.0 - distribution), 0.15))
        basis.append("DISTRIBUTION_CONSISTENCY")
    elif signal in {"ACCUMULATION_PROXY", "DISTRIBUTION_PROXY", "NEUTRAL"}:
        signal_score = {
            "ACCUMULATION_PROXY": 80.0,
            "NEUTRAL": 50.0,
            "DISTRIBUTION_PROXY": 20.0,
        }[signal]
        components.append((signal_score, 0.20))
        basis.append("BROKER_SIGNAL")
    weight = sum(item[1] for item in components)
    score = sum(value * component_weight for value, component_weight in components) / max(weight, 1e-9)
    effective = 50.0 + (score - 50.0) * coverage / 100.0
    state = (
        "BROKER_SUMMARY_DISTRIBUTION"
        if signal == "DISTRIBUTION_PROXY"
        else "BROKER_SUMMARY_DIRECT_VERIFIED"
        if direct_verified
        else "BROKER_SUMMARY_OBSERVED_UNVERIFIED"
        if coverage >= 70.0
        else "BROKER_SUMMARY_PARTIAL"
    )
    return {
        "score": round(_clip(effective), 1),
        "coverage_pct": round(coverage, 1),
        "state": state,
        "basis": " | ".join(basis),
        "direct_verified": direct_verified,
        "provenance_state": provenance_state or "LEGACY_OR_UNVERIFIED",
    }


def _pre_event_flow(frame: pd.DataFrame, latest_detection: Any) -> dict[str, float]:
    if frame.empty or not latest_detection:
        return {"return_20d_pct": np.nan, "volume_ratio": np.nan, "coverage_pct": 0.0}
    detected = pd.to_datetime(latest_detection, errors="coerce", utc=True)
    if pd.isna(detected):
        return {"return_20d_pct": np.nan, "volume_ratio": np.nan, "coverage_pct": 0.0}
    detected = detected.tz_convert(None)
    local = frame.copy()
    index = local.index
    if index.tz is not None:
        local.index = index.tz_convert(None)
    prior = local.loc[local.index < detected].tail(60)
    if len(prior) < 21:
        return {"return_20d_pct": np.nan, "volume_ratio": np.nan, "coverage_pct": 0.0}
    close = pd.to_numeric(prior["Close"], errors="coerce")
    volume = pd.to_numeric(prior.get("Volume"), errors="coerce")
    return_20d = 100.0 * (close.iloc[-1] / close.iloc[-21] - 1.0)
    recent_volume = _finite(volume.tail(10).mean(), np.nan)
    baseline_volume = _finite(volume.iloc[-50:-10].mean(), np.nan)
    ratio = recent_volume / baseline_volume if np.isfinite(recent_volume) and np.isfinite(baseline_volume) and baseline_volume > 0 else np.nan
    coverage = 100.0 if np.isfinite(return_20d) and np.isfinite(ratio) else 50.0
    return {
        "return_20d_pct": round(return_20d, 2),
        "volume_ratio": round(ratio, 3) if np.isfinite(ratio) else np.nan,
        "coverage_pct": coverage,
    }


def build_emir_method_profile(
    *,
    ticker: str,
    frame: pd.DataFrame | None,
    active_events: pd.DataFrame | None,
    outcomes: Mapping[str, Any] | None,
    fundamental: Mapping[str, Any] | None,
    silent_profile: Mapping[str, Any] | None,
    narrative_effective_score: float,
    narrative_evidence_coverage_pct: float,
    narrative_evidence_mode: str,
    alignment_effective_score: float,
    alignment_coverage_pct: float,
    adoption_stage: str,
    crowding_risk_score: float,
    hard_block: bool,
    growth_max_adjustment_points: float = 14.0,
    turnaround_max_adjustment_points: float = 16.0,
    swing_max_adjustment_points: float = 18.0,
) -> dict[str, Any]:
    """Return an auditable public-framework selection profile."""
    local = _price_frame(frame)
    events = active_events.copy() if isinstance(active_events, pd.DataFrame) else pd.DataFrame()
    outcome = dict(outcomes or {})
    fund = dict(fundamental or {})
    silent = dict(silent_profile or {})

    history_bars = len(local)
    history_years = _latest_number(
        fund,
        ("fundamental_history_years", "history_years"),
        default=0.0,
    )
    history_quarters = _latest_number(
        fund,
        ("fundamental_history_quarters", "history_quarters"),
        default=0.0,
    )
    source_count = _latest_number(
        fund,
        ("fundamental_source_count", "fundamental_history_source_count"),
        default=0.0,
    )
    resolved_20d = int(max(0.0, _finite(outcome.get("narrative_conversion_resolved_20d"), 0.0)))
    event_history = int(len(events))
    flow_confidence = _clip(
        silent.get("silent_accumulation_confidence", silent.get("silent_accumulation_data_coverage")),
    )
    familiarity_components = {
        "OHLCV_HISTORY": min(35.0, 35.0 * history_bars / 500.0),
        "FINANCIAL_HISTORY": min(20.0, 3.0 * history_quarters + 4.0 * history_years),
        "SOURCE_LINEAGE": min(15.0, 5.0 * source_count),
        "EVENT_MEMORY": min(15.0, 3.0 * event_history),
        "OUTCOME_MEMORY": min(15.0, 1.5 * resolved_20d),
    }
    familiarity_raw = sum(familiarity_components.values())
    familiarity_coverage = min(
        100.0,
        40.0 * min(1.0, history_bars / 220.0)
        + 25.0 * min(1.0, (history_quarters + 4.0 * history_years) / 12.0)
        + 20.0 * min(1.0, source_count / 2.0)
        + 15.0 * min(1.0, (event_history + resolved_20d) / 10.0),
    )
    familiarity_score = (
        50.0 + (familiarity_raw - 50.0) * familiarity_coverage / 100.0
        if familiarity_coverage > 0.0 else np.nan
    )
    familiarity_state = (
        "UNIVERSE_MEMORY_NOT_SCORED" if not np.isfinite(familiarity_score)
        else "DEEP_UNIVERSE_MEMORY" if familiarity_score >= 75.0 and familiarity_coverage >= 75.0
        else "WORKING_UNIVERSE_MEMORY" if familiarity_score >= 60.0 and familiarity_coverage >= 55.0
        else "LIMITED_UNIVERSE_MEMORY"
    )

    effective_silent = _clip(
        silent.get("effective_silent_accumulation_score", silent.get("silent_accumulation_score", 50.0)),
    )
    persistent_bid = _clip(silent.get("persistent_bid_score"), 0.0, 100.0)
    accumulation_persistence = _clip(silent.get("accumulation_persistence_score"), 0.0, 100.0)
    weighted_clv = _finite(silent.get("weighted_close_location20"), np.nan)
    close_acceptance = _clip(100.0 * weighted_clv) if np.isfinite(weighted_clv) else 50.0
    absorption_days = max(0.0, _finite(silent.get("absorption_confirmed_days20"), 0.0))
    churning_days = max(0.0, _finite(silent.get("churning_support_days20"), 0.0))
    failed_absorption = max(0.0, _finite(silent.get("failed_absorption_days20"), 0.0))
    distribution_days = max(0.0, _finite(silent.get("distribution_days20"), 0.0))
    supply_absorption = _clip(
        35.0 + 12.0 * absorption_days + 5.0 * churning_days
        - 18.0 * failed_absorption - 10.0 * distribution_days,
    )
    adtv20_idr = _latest_number(silent, ("adtv20_idr",), default=np.nan)
    if not np.isfinite(adtv20_idr) and not local.empty and "Volume" in local:
        value = pd.to_numeric(local["Close"], errors="coerce") * pd.to_numeric(local["Volume"], errors="coerce")
        adtv20_idr = _finite(value.tail(20).mean(), 0.0)
    liquidity = _liquidity_score(_text(silent.get("liquidity_bucket")), adtv20_idr)
    broker = _broker_summary_component(silent)
    broker_score = _finite(broker.get("score"), np.nan)
    price_volume_score = (
        0.38 * effective_silent
        + 0.18 * persistent_bid
        + 0.15 * accumulation_persistence
        + 0.12 * close_acceptance
        + 0.10 * supply_absorption
        + 0.07 * liquidity
    )
    broker_direct_verified = _truthy(broker.get("direct_verified"))
    if np.isfinite(broker_score):
        smart_money_raw = 0.72 * price_volume_score + 0.28 * broker_score
        flow_evidence_mode = (
            "DIRECT_BROKER_SUMMARY_VERIFIED"
            if broker_direct_verified
            else "BROKER_SUMMARY_OBSERVED_UNVERIFIED"
        )
        smart_money_coverage = min(100.0, 0.70 * flow_confidence + 0.30 * _finite(broker.get("coverage_pct"), 0.0))
    else:
        smart_money_raw = price_volume_score
        flow_evidence_mode = "OHLCV_PRICE_VOLUME_PROXY_ONLY"
        smart_money_coverage = min(85.0, flow_confidence)
    broker_distribution = _text(broker.get("state")).upper() == "BROKER_SUMMARY_DISTRIBUTION"
    state_distribution = _text(silent.get("silent_accumulation_state")).upper() in {
        "DISTRIBUTION_RISK", "WEAK_OR_DISTRIBUTION",
    }
    proxy_distribution_raw = min(
        1.0,
        0.38 * min(1.0, distribution_days / 6.0)
        + 0.34 * min(1.0, failed_absorption / 3.0)
        + 0.18 * float(state_distribution)
        + 0.10 * min(1.0, max(0.0, 55.0 - effective_silent) / 30.0),
    )
    proxy_evidence_confidence = min(
        1.0,
        max(0.25, smart_money_coverage / 100.0)
        * max(0.35, liquidity / 100.0),
    )
    distribution_severity = 1.0 if broker_distribution else proxy_distribution_raw * proxy_evidence_confidence
    severe_distribution = bool(broker_distribution or distribution_severity >= 0.68)
    distribution_risk = bool(broker_distribution or distribution_severity >= 0.35)
    if distribution_risk:
        smart_money_raw = min(smart_money_raw, 50.0 - 28.0 * distribution_severity)
    smart_money_score = (
        50.0 + (smart_money_raw - 50.0) * smart_money_coverage / 100.0
        if smart_money_coverage > 0.0 else np.nan
    )
    smart_money_state = (
        "FLOW_NOT_SCORED" if not np.isfinite(smart_money_score)
        else "DISTRIBUTION_OR_FAILED_ABSORPTION" if distribution_risk
        else "SMART_MONEY_BEHAVIOR_CONFIRMED" if smart_money_score >= 70.0 and smart_money_coverage >= 65.0
        else "EARLY_FLOW_CONFIRMATION" if smart_money_score >= 58.0 and smart_money_coverage >= 50.0
        else "FLOW_NOT_CONFIRMED"
    )

    latest_detection = ""
    latest_materiality = np.nan
    latest_bridge = np.nan
    if not events.empty:
        local_events = events.copy()
        local_events["detected_at"] = pd.to_datetime(local_events.get("detected_at"), errors="coerce", utc=True)
        local_events = local_events.sort_values("detected_at", ascending=False)
        latest_event = local_events.iloc[0]
        latest_detection = _text(latest_event.get("detected_at"))
        latest_materiality = _finite(latest_event.get("materiality_score"), np.nan)
        latest_bridge = _finite(latest_event.get("financial_bridge_score"), np.nan)
    pre_event = _pre_event_flow(local, latest_detection)
    flow_led_before_story = bool(
        np.isfinite(pre_event["return_20d_pct"])
        and np.isfinite(pre_event["volume_ratio"])
        and pre_event["return_20d_pct"] >= 4.0
        and pre_event["volume_ratio"] >= 1.15
    )
    narrative_input = _finite(narrative_effective_score, np.nan)
    narrative_score = _clip(narrative_input) if np.isfinite(narrative_input) else np.nan
    narrative_coverage = _clip(narrative_evidence_coverage_pct)
    materiality_score = latest_materiality if np.isfinite(latest_materiality) else narrative_score
    bridge_score = latest_bridge if np.isfinite(latest_bridge) else narrative_score
    stage = _text(adoption_stage).upper()
    stage_score_map = {
        "SEED_STORY_AHEAD_OF_FLOW": 58.0,
        "STRUCTURED_OPERATING_STORY": 58.0,
        "EARLY_DISCOVERY_STRUCTURED": 70.0,
        "EARLY_DISCOVERY": 78.0,
        "EXPANSION": 88.0,
        "CONSENSUS_CROWDED": 35.0,
        "EXHAUSTION_OR_DISTRIBUTION": 15.0,
        "DORMANT_OR_UNCONVERTED": 40.0,
        "FLOW_WITHOUT_SOURCED_STORY": 48.0,
        "NO_ACTIVE_STORY": 35.0,
    }
    stage_score = stage_score_map.get(stage, 45.0)
    if narrative_coverage > 0.0 and np.isfinite(narrative_score):
        lifecycle_raw = (
            0.40 * narrative_score
            + 0.20 * _clip(materiality_score)
            + 0.18 * _clip(bridge_score)
            + 0.22 * stage_score
        )
        if flow_led_before_story and stage not in {"CONSENSUS_CROWDED", "EXHAUSTION_OR_DISTRIBUTION"}:
            lifecycle_raw += 5.0
        lifecycle_raw -= 0.20 * _clip(crowding_risk_score)
        lifecycle_score = 50.0 + (_clip(lifecycle_raw) - 50.0) * narrative_coverage / 100.0
    else:
        lifecycle_raw = np.nan
        lifecycle_score = np.nan
    lifecycle_state = (
        "NARRATIVE_LIFECYCLE_NOT_SCORED" if not np.isfinite(lifecycle_score)
        else "LATE_CROWDED_OR_DISTRIBUTION" if stage in {"CONSENSUS_CROWDED", "EXHAUSTION_OR_DISTRIBUTION"} or distribution_risk
        else "FLOW_LED_STORY_CONFIRMED" if flow_led_before_story and narrative_score >= 55.0
        else "EARLY_NARRATIVE_FLOW_CONVERGENCE" if stage in {"EARLY_DISCOVERY", "EARLY_DISCOVERY_STRUCTURED"} and np.isfinite(smart_money_score) and smart_money_score >= 58.0
        else "EXPANSION_CONFIRMED" if stage == "EXPANSION" and np.isfinite(smart_money_score) and smart_money_score >= 60.0
        else "STORY_AHEAD_OF_FLOW" if narrative_score >= 58.0 and (not np.isfinite(smart_money_score) or smart_money_score < 52.0)
        else "FLOW_AHEAD_OF_STORY" if np.isfinite(smart_money_score) and smart_money_score >= 62.0 and narrative_coverage < 25.0
        else "MIXED_OR_UNCONFIRMED"
    )

    alignment_input = _finite(alignment_effective_score, np.nan)
    alignment_reliability = _clip(alignment_coverage_pct) / 100.0
    effective_alignment = (
        50.0 + (_clip(alignment_input) - 50.0) * alignment_reliability
        if alignment_reliability > 0.0 and np.isfinite(alignment_input)
        else np.nan
    )
    method_components = (
        (familiarity_score, 0.22),
        (lifecycle_score, 0.31),
        (smart_money_score, 0.37),
        (effective_alignment, 0.10),
    )
    available_method_weight = sum(weight for value, weight in method_components if np.isfinite(value))
    method_raw = (
        sum(value * weight for value, weight in method_components if np.isfinite(value))
        / available_method_weight
        if available_method_weight > 0.0 else np.nan
    )
    method_coverage = (
        0.22 * familiarity_coverage
        + 0.31 * narrative_coverage
        + 0.37 * smart_money_coverage
        + 0.10 * (100.0 * alignment_reliability)
    )
    risk_flags: list[str] = []
    if hard_block:
        risk_flags.append("OFFICIAL_CRITICAL_CONTRADICTION")
    if distribution_risk:
        risk_flags.append("DISTRIBUTION_RISK")
    if _clip(crowding_risk_score) >= 60.0:
        risk_flags.append("CROWDED_LATE_ENTRY")
    if smart_money_coverage < 55.0:
        risk_flags.append("FLOW_EVIDENCE_WEAK")
    if narrative_coverage < 35.0:
        risk_flags.append("NARRATIVE_EVIDENCE_WEAK")
    if familiarity_coverage < 55.0:
        risk_flags.append("UNIVERSE_MEMORY_LIMITED")
    if liquidity < 45.0:
        risk_flags.append("LIQUIDITY_RISK")
    penalty = (
        28.0 * int(hard_block)
        + 24.0 * int(broker_distribution)
        + 18.0 * distribution_severity * int(not broker_distribution)
        + 0.18 * max(0.0, _clip(crowding_risk_score) - 55.0)
    )
    method_score = _clip(method_raw - penalty) if np.isfinite(method_raw) else np.nan
    production_eligible = bool(
        not hard_block
        and not severe_distribution
        and _clip(crowding_risk_score) < 65.0
        and narrative_coverage >= 35.0
        and smart_money_coverage >= 55.0
        and familiarity_coverage >= 45.0
        and liquidity >= 45.0
        and np.isfinite(method_score)
        and method_coverage >= 55.0
        and method_score >= 62.0
    )
    if production_eligible and method_score >= 75.0 and lifecycle_state in {
        "FLOW_LED_STORY_CONFIRMED", "EARLY_NARRATIVE_FLOW_CONVERGENCE", "EXPANSION_CONFIRMED",
    }:
        method_state = "EMIR_FRAMEWORK_READY"
    elif hard_block or severe_distribution:
        method_state = "EMIR_FRAMEWORK_REJECT"
    elif (not np.isfinite(method_score)) or method_coverage < 55.0 or narrative_coverage < 35.0 or smart_money_coverage < 55.0:
        method_state = "EMIR_FRAMEWORK_EVIDENCE_PENDING"
    else:
        method_state = "EMIR_FRAMEWORK_WATCH"

    reliability = min(
        familiarity_coverage,
        max(narrative_coverage, 0.0),
        smart_money_coverage,
        method_coverage,
    ) / 100.0
    adjustment_unit = (
        (method_score - 50.0) / 50.0 * reliability
        if np.isfinite(method_score) else 0.0
    )
    if not production_eligible:
        adjustment_unit = min(adjustment_unit, 0.0)
    growth_cap = max(0.0, min(20.0, _finite(growth_max_adjustment_points, 14.0)))
    turnaround_cap = max(0.0, min(22.0, _finite(turnaround_max_adjustment_points, 16.0)))
    swing_cap = max(0.0, min(25.0, _finite(swing_max_adjustment_points, 18.0)))
    growth_adjustment = growth_cap * adjustment_unit
    turnaround_adjustment = turnaround_cap * adjustment_unit
    swing_adjustment = swing_cap * adjustment_unit
    if hard_block or broker_distribution:
        growth_adjustment = min(growth_adjustment, -0.55 * growth_cap)
        turnaround_adjustment = min(turnaround_adjustment, -0.60 * turnaround_cap)
        swing_adjustment = min(swing_adjustment, -0.65 * swing_cap)
    elif distribution_risk:
        # OHLCV-only distribution is a confidence-weighted continuous penalty,
        # not a uniform hard floor applied to hundreds of issuers.
        growth_adjustment = min(growth_adjustment, -0.45 * growth_cap * distribution_severity)
        turnaround_adjustment = min(turnaround_adjustment, -0.50 * turnaround_cap * distribution_severity)
        swing_adjustment = min(swing_adjustment, -0.58 * swing_cap * distribution_severity)

    if not production_eligible:
        position_cap = 0.0
    elif method_score >= 82.0 and liquidity >= 70.0:
        position_cap = 15.0
    elif method_score >= 75.0:
        position_cap = 12.0
    elif method_score >= 68.0:
        position_cap = 8.0
    else:
        position_cap = 5.0
    if flow_evidence_mode == "OHLCV_PRICE_VOLUME_PROXY_ONLY":
        position_cap = min(position_cap, 10.0)

    reasons = [familiarity_state, lifecycle_state, smart_money_state]
    if flow_led_before_story:
        reasons.append("FLOW_PRECEDED_PUBLIC_STORY")
    if np.isfinite(broker_score):
        reasons.append(_text(broker.get("state")))
    return {
        "emir_method_engine_version": EMIR_METHOD_ENGINE_VERSION,
        "emir_public_framework_disclaimer": "PUBLIC_METHOD_RECONSTRUCTION_NOT_PROPRIETARY_OR_VERIFIED_TRACK_RECORD",
        "stock_universe_familiarity_score": round(_clip(familiarity_score), 1) if np.isfinite(familiarity_score) else np.nan,
        "stock_universe_familiarity_coverage_pct": round(_clip(familiarity_coverage), 1),
        "stock_universe_familiarity_state": familiarity_state,
        "stock_universe_familiarity_basis": " | ".join(
            f"{name}={value:.1f}" for name, value in familiarity_components.items()
        ),
        "smart_money_behavior_score": round(_clip(smart_money_score), 1) if np.isfinite(smart_money_score) else np.nan,
        "smart_money_behavior_raw_score": round(_clip(smart_money_raw), 1),
        "smart_money_behavior_coverage_pct": round(_clip(smart_money_coverage), 1),
        "smart_money_behavior_state": smart_money_state,
        "smart_money_flow_evidence_mode": flow_evidence_mode,
        "distribution_severity_score": round(100.0 * distribution_severity, 1),
        "distribution_penalty_points": round(float(penalty), 2),
        "distribution_evidence_state": (
            "DIRECT_BROKER_DISTRIBUTION" if broker_distribution
            else "SEVERE_OHLCV_DISTRIBUTION" if severe_distribution
            else "MODERATE_OHLCV_DISTRIBUTION" if distribution_risk
            else "NO_MATERIAL_DISTRIBUTION"
        ),
        "broker_summary_score": round(broker_score, 1) if np.isfinite(broker_score) else np.nan,
        "broker_summary_coverage_pct": round(_finite(broker.get("coverage_pct"), 0.0), 1),
        "broker_summary_direct_verified": bool(broker_direct_verified),
        "broker_summary_provenance_state": _text(broker.get("provenance_state")) or "NOT_AVAILABLE",
        "narrative_lifecycle_score": round(_clip(lifecycle_score), 1) if np.isfinite(lifecycle_score) else np.nan,
        "narrative_lifecycle_state": lifecycle_state,
        "pre_narrative_flow_return_20d_pct": pre_event["return_20d_pct"],
        "pre_narrative_volume_ratio": pre_event["volume_ratio"],
        "flow_preceded_narrative": bool(flow_led_before_story),
        "emir_method_score": round(method_score, 1) if np.isfinite(method_score) else np.nan,
        "emir_method_coverage_pct": round(_clip(method_coverage), 1),
        "emir_method_score_state": (
            "SCORED_PRODUCTION_ELIGIBLE" if production_eligible
            else "SCORED_REJECTED" if np.isfinite(method_score) and (hard_block or severe_distribution)
            else "SCORED_RESEARCH_ONLY" if np.isfinite(method_score)
            else "NOT_SCORED_INSUFFICIENT_EVIDENCE"
        ),
        "emir_method_state": method_state,
        "emir_method_production_eligible": production_eligible,
        "emir_method_reliability_pct": round(100.0 * reliability, 1),
        "emir_selection_reason": " • ".join(reasons),
        "emir_risk_flags": " | ".join(risk_flags) or "NO_MAJOR_FRAMEWORK_RISK",
        "emir_position_cap_pct": round(position_cap, 1),
        "emir_growth_rank_adjustment": round(growth_adjustment, 2),
        "emir_turnaround_rank_adjustment": round(turnaround_adjustment, 2),
        "emir_swing_rank_adjustment": round(swing_adjustment, 2),
    }


__all__ = ["EMIR_METHOD_ENGINE_VERSION", "build_emir_method_profile"]
