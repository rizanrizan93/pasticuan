from __future__ import annotations

"""Simplified decision layer for the macro-first scanner.

The module produces only two decision universes:
- The Next Leader: Jos-style business/future-fundamental selection, confirmed
  by macro transmission and Emir-style narrative/flow.
- Swing Ready: technically executable opportunities that remain aligned with
  macro, business quality, narrative, and risk.

It deliberately avoids EOFF, time-cycle, AI overlays, and additive legacy
scores. Missing evidence remains missing; no displayed component is filled with
an artificial neutral 50.
"""

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from macro_engine import MacroRegimeResult
from narrative_engine import build_narrative_intelligence
from scanner import ScanConfig, safe_number, silent_accumulation_profile


SIMPLE_FOCUS_VERSION = "9.5.0-database-first-ranking"


@dataclass(frozen=True)
class NextLeaderWeights:
    business_quality: float = 0.25
    future_fundamental: float = 0.20
    valuation_mos: float = 0.15
    management_capital: float = 0.10
    macro_sector: float = 0.10
    narrative_flow: float = 0.15
    technical_readiness: float = 0.05

    def validate(self) -> None:
        if not np.isclose(sum(self.__dict__.values()), 1.0):
            raise ValueError("Next Leader weights must total 1.0")


@dataclass(frozen=True)
class SwingWeights:
    technical_execution: float = 0.40
    macro_sector: float = 0.15
    narrative_flow: float = 0.25
    business_quality: float = 0.10
    risk_data: float = 0.10

    def validate(self) -> None:
        if not np.isclose(sum(self.__dict__.values()), 1.0):
            raise ValueError("Swing weights must total 1.0")


NEXT_LEADER_WEIGHTS = NextLeaderWeights()
SWING_WEIGHTS = SwingWeights()
NEXT_LEADER_WEIGHTS.validate()
SWING_WEIGHTS.validate()


def _ticker(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    return text if text.endswith(".JK") else f"{text}.JK"


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _clip(value: Any, lower: float = 0.0, upper: float = 100.0) -> float:
    number = _finite(value, np.nan)
    if not np.isfinite(number):
        return np.nan
    return float(np.clip(number, lower, upper))


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value or "").strip().upper() in {"1", "TRUE", "YES", "Y", "PASS", "VALID", "VERIFIED"}


def _fraction(value: Any) -> float:
    number = _finite(value, np.nan)
    if not np.isfinite(number):
        return np.nan
    return number / 100.0 if abs(number) > 2.0 else number


def _latest(frame: pd.DataFrame | None) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "ticker" not in frame.columns:
        return pd.DataFrame(columns=["ticker"])
    out = frame.copy()
    out["ticker"] = out["ticker"].map(_ticker)
    out = out[out["ticker"].ne("")]
    return out.drop_duplicates("ticker", keep="last")


def _prefixed(frame: pd.DataFrame | None, prefix: str) -> pd.DataFrame:
    out = _latest(frame)
    if out.empty:
        return out
    return out.rename(columns={column: f"{prefix}{column}" for column in out.columns if column != "ticker"})


def _first(row: Mapping[str, Any], names: Sequence[str], default: Any = np.nan) -> Any:
    for name in names:
        value = row.get(name, np.nan)
        if isinstance(value, str):
            if value.strip():
                return value
        elif value is not None and not (isinstance(value, float) and np.isnan(value)):
            return value
    return default


def _first_num(row: Mapping[str, Any], names: Sequence[str]) -> float:
    for name in names:
        value = _finite(row.get(name), np.nan)
        if np.isfinite(value):
            return value
    return np.nan


def _metric_score(
    row: Mapping[str, Any],
    specs: Sequence[tuple[Sequence[str], float, Callable[[float], float]]],
) -> tuple[float, float, list[str]]:
    numerator = 0.0
    observed_weight = 0.0
    total_weight = sum(weight for _, weight, _ in specs)
    evidence: list[str] = []
    for names, weight, transform in specs:
        value = _first_num(row, names)
        if not np.isfinite(value):
            continue
        score = _clip(transform(value))
        if not np.isfinite(score):
            continue
        numerator += weight * score
        observed_weight += weight
        evidence.append(names[0])
    if observed_weight <= 0 or total_weight <= 0:
        return np.nan, 0.0, []
    return numerator / observed_weight, 100.0 * observed_weight / total_weight, evidence


def _growth_score(value: float) -> float:
    fraction = _fraction(value)
    return 50.0 + 200.0 * fraction if np.isfinite(fraction) else np.nan


def _ratio_score(value: float, weak: float, strong: float) -> float:
    if not np.isfinite(value):
        return np.nan
    if strong == weak:
        return 50.0
    return 100.0 * (value - weak) / (strong - weak)


def _inverse_ratio_score(value: float, good: float, bad: float) -> float:
    return 100.0 - _ratio_score(value, good, bad)


def _positive_cash_score(value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    if value > 0:
        return 75.0 + min(25.0, np.log10(max(abs(value), 1.0)) * 2.0)
    return 15.0


def _build_silent_profiles(prepared: Mapping[str, pd.DataFrame]) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    profiles: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for ticker, frame in prepared.items():
        key = _ticker(ticker)
        try:
            profile = dict(silent_accumulation_profile(frame))
        except Exception as exc:
            profile = {
                "silent_accumulation_score": np.nan,
                "silent_accumulation_confidence": 0.0,
                "silent_accumulation_state": "FAIL_SOFT",
                "silent_accumulation_reason": f"{type(exc).__name__}: {str(exc)[:120]}",
            }
        profiles[key] = profile
        rows.append({"ticker": key, **profile})
    return profiles, pd.DataFrame(rows)


def build_silent_profiles(prepared: Mapping[str, pd.DataFrame]) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Public chunk-safe silent-accumulation profile builder."""
    return _build_silent_profiles(prepared)


def _business_component(row: Mapping[str, Any]) -> tuple[float, float, str]:
    score, coverage, evidence = _metric_score(row, (
        (("fund_revenue_growth", "fund_history_revenue_growth_yoy"), 0.13, _growth_score),
        (("fund_earnings_growth", "fund_history_net_income_growth_yoy"), 0.13, _growth_score),
        (("fund_roe", "fund_history_roe"), 0.12, lambda v: _ratio_score(_fraction(v), 0.05, 0.22)),
        (("fund_roa", "fund_history_roa"), 0.07, lambda v: _ratio_score(_fraction(v), 0.02, 0.12)),
        (("fund_net_margin", "fund_history_net_margin"), 0.10, lambda v: _ratio_score(_fraction(v), 0.03, 0.20)),
        (("fund_operating_cash_flow",), 0.10, _positive_cash_score),
        (("fund_free_cash_flow",), 0.10, _positive_cash_score),
        (("fund_history_cash_conversion", "fund_cash_conversion_ttm"), 0.08, lambda v: _ratio_score(v, 0.45, 1.25)),
        (("fund_debt_equity",), 0.07, lambda v: _inverse_ratio_score(v, 0.20, 2.00)),
        (("fund_current_ratio",), 0.05, lambda v: _ratio_score(v, 0.70, 2.00)),
        (("fund_cash_to_debt",), 0.05, lambda v: _ratio_score(v, 0.10, 1.20)),
    ))
    return score, coverage, " | ".join(evidence)


def _future_component(row: Mapping[str, Any]) -> tuple[float, float, str]:
    score, coverage, evidence = _metric_score(row, (
        (("fund_future_fundamental_impact_score", "fund_future_impact_score"), 0.30, lambda v: v),
        (("fund_project_pipeline_score",), 0.20, lambda v: v),
        (("fund_reinvestment_runway_pillar",), 0.18, lambda v: v),
        (("fund_fundamental_inflection_score",), 0.17, lambda v: v),
        (("fund_revenue_growth",), 0.08, _growth_score),
        (("fund_earnings_growth",), 0.07, _growth_score),
    ))
    return score, coverage, " | ".join(evidence)


def _valuation_component(row: Mapping[str, Any]) -> tuple[float, float, str]:
    # valuation_score in the legacy engine is 0..8; other fields are raw ratios.
    score, coverage, evidence = _metric_score(row, (
        (("fund_valuation_score",), 0.35, lambda v: 12.5 * v if v <= 10 else v),
        (("fund_peg_ratio",), 0.20, lambda v: _inverse_ratio_score(v, 0.50, 2.50)),
        (("fund_fcf_yield",), 0.20, lambda v: _ratio_score(_fraction(v), 0.00, 0.10)),
        (("fund_trailing_pe",), 0.15, lambda v: _inverse_ratio_score(v, 6.0, 35.0)),
        (("fund_price_to_book",), 0.10, lambda v: _inverse_ratio_score(v, 0.70, 5.00)),
    ))
    return score, coverage, " | ".join(evidence)


def _management_component(row: Mapping[str, Any]) -> tuple[float, float, str]:
    score, coverage, evidence = _metric_score(row, (
        (("fund_management_quality_score",), 0.25, lambda v: v),
        (("fund_capital_allocation_score",), 0.25, lambda v: v),
        (("fund_history_roic_proxy", "fund_roic_proxy"), 0.20, lambda v: _ratio_score(_fraction(v), 0.04, 0.20)),
        (("fund_history_share_dilution_yoy", "fund_share_dilution_yoy"), 0.15,
         lambda v: _inverse_ratio_score(_fraction(v), 0.00, 0.15)),
        (("nar_issuer_action_alignment_effective_score",), 0.15, lambda v: v),
    ))
    return score, coverage, " | ".join(evidence)


def _macro_component(row: Mapping[str, Any]) -> tuple[float, float, str]:
    score = _first_num(row, ("mac_issuer_macro_alignment_score",))
    coverage = _first_num(row, ("mac_issuer_macro_alignment_coverage_pct",))
    if not np.isfinite(score):
        return np.nan, 0.0, ""
    return _clip(score), _clip(coverage) if np.isfinite(coverage) else 60.0, str(row.get("mac_issuer_macro_alignment_basis", ""))


def _flow_score_from_row(row: Mapping[str, Any]) -> tuple[float, float]:
    score = _first_num(row, ("flow_silent_accumulation_score", "sig_silent_accumulation_score"))
    coverage = _first_num(row, ("flow_silent_accumulation_confidence", "flow_silent_accumulation_data_coverage"))
    return (_clip(score), _clip(coverage) if np.isfinite(coverage) else (70.0 if np.isfinite(score) else 0.0))


def _narrative_flow_component(row: Mapping[str, Any]) -> tuple[float, float, str]:
    narrative = _first_num(row, ("nar_narrative_event_effective_score", "nar_narrative_effective_score"))
    narrative_cov = _first_num(row, ("nar_narrative_event_coverage_pct", "nar_narrative_evidence_coverage_pct"))
    flow, flow_cov = _flow_score_from_row(row)
    parts: list[tuple[float, float, float, str]] = []
    if np.isfinite(narrative):
        parts.append((narrative, 0.60, narrative_cov if np.isfinite(narrative_cov) else 60.0, "SOURCED_NARRATIVE"))
    if np.isfinite(flow):
        parts.append((flow, 0.40, flow_cov, "SILENT_ACCUMULATION"))
    if not parts:
        return np.nan, 0.0, ""
    denominator = sum(weight for _, weight, _, _ in parts)
    score = sum(value * weight for value, weight, _, _ in parts) / denominator
    coverage = sum(weight * cov for _, weight, cov, _ in parts) / denominator
    return _clip(score), _clip(coverage), " | ".join(label for _, _, _, label in parts)


def _technical_component(row: Mapping[str, Any]) -> tuple[float, float, str]:
    state = str(_first(row, ("sig_setup_status", "sig_status", "sig_decision_state"), "")).upper()
    state_map = {
        "EXECUTION_READY": 95.0,
        "ENTRY_PLAN_READY": 82.0,
        "READY_FOR_PRICE_VERIFY": 76.0,
        "READY_FOR_STOCKBIT_VERIFY": 74.0,
        "WATCHLIST": 55.0,
        "PENDING": 42.0,
        "REJECT": 10.0,
    }
    state_score = state_map.get(state, np.nan)
    quality = _first_num(row, ("sig_quality_score", "sig_analyst_fusion_score", "sig_setup_score"))
    momentum = _first_num(row, ("sig_momentum_score", "fund_momentum_score"))
    if np.isfinite(momentum) and momentum <= 12:
        momentum = momentum * 100.0 / 12.0
    rr1 = _first_num(row, ("sig_rr1",))
    rr_score = _clip(40.0 + 25.0 * rr1) if np.isfinite(rr1) else np.nan
    values = [(state_score, 0.35, "SETUP_STATE"), (quality, 0.30, "QUALITY"),
              (momentum, 0.20, "MOMENTUM"), (rr_score, 0.15, "RR")]
    observed = [(v, w, label) for v, w, label in values if np.isfinite(v)]
    if not observed:
        return np.nan, 0.0, ""
    weight = sum(w for _, w, _ in observed)
    score = sum(_clip(v) * w for v, w, _ in observed) / weight
    return score, 100.0 * weight, " | ".join(label for _, _, label in observed)


def _risk_data_component(row: Mapping[str, Any]) -> tuple[float, float, str]:
    data_quality = _first_num(row, ("sig_data_quality_score", "fund_fundamental_coverage", "fund_fundamental_overall_coverage"))
    validation = _first_num(row, ("sig_validation_score",))
    adtv = _first_num(row, ("sig_adtv20_idr", "flow_adtv20_idr", "fund_adtv20_idr"))
    liquidity = np.nan
    if np.isfinite(adtv) and adtv > 0:
        liquidity = _clip(20.0 * np.log10(max(adtv, 1.0)) - 90.0)
    values = [(data_quality, 0.45, "DATA_QUALITY"), (validation, 0.20, "OOS_VALIDATION"),
              (liquidity, 0.35, "LIQUIDITY")]
    observed = [(v, w, label) for v, w, label in values if np.isfinite(v)]
    if not observed:
        return np.nan, 0.0, ""
    weight = sum(w for _, w, _ in observed)
    score = sum(_clip(v) * w for v, w, _ in observed) / weight
    return score, 100.0 * weight, " | ".join(label for _, _, label in observed)


def _weighted_final(components: Mapping[str, tuple[float, float]], weights: Mapping[str, float], *, min_coverage: float) -> tuple[float, float]:
    raw_sum = 0.0
    coverage_sum = 0.0
    for name, weight in weights.items():
        score, coverage = components[name]
        observed = np.isfinite(score) and coverage > 0
        effective = score if observed else 50.0
        raw_sum += weight * effective
        coverage_sum += weight * (coverage if observed else 0.0)
    if coverage_sum < min_coverage:
        return np.nan, coverage_sum
    # Coverage shrinkage is internal only. Component display remains NaN when missing.
    final = 50.0 + coverage_sum / 100.0 * (raw_sum - 50.0)
    return _clip(final), _clip(coverage_sum)


def _hard_block(row: Mapping[str, Any]) -> tuple[bool, str]:
    reasons: list[str] = []
    if _truthy(row.get("nar_narrative_hard_block")):
        reasons.append("OFFICIAL_NARRATIVE_CONTRADICTION")
    for name in ("sig_market_status", "sig_listing_status", "sig_universe_status", "sig_fca_status"):
        text = str(row.get(name, "")).upper()
        if any(token in text for token in ("SUSPEND", "DELIST", "FCA_BLOCK", "REJECT")):
            reasons.append(text or name.upper())
    if _truthy(row.get("fund_severe_fundamental_flags")):
        reasons.append("SEVERE_FUNDAMENTAL_FLAGS")
    sharia = row.get("fund_is_sharia", row.get("sig_is_sharia", np.nan))
    if sharia is not np.nan and str(sharia).strip() and not _truthy(sharia):
        if str(sharia).strip().upper() in {"0", "FALSE", "NO", "N"}:
            reasons.append("NOT_SHARIA")
    return bool(reasons), " | ".join(dict.fromkeys(reasons))


def _candidate_type(row: Mapping[str, Any], future_score: float) -> str:
    sector = str(_first(row, ("fund_sector", "mac_sector"), "")).upper()
    revenue_growth = _fraction(_first_num(row, ("fund_revenue_growth",)))
    earnings_growth = _fraction(_first_num(row, ("fund_earnings_growth",)))
    roe = _fraction(_first_num(row, ("fund_roe",)))
    project = _first_num(row, ("fund_project_pipeline_score", "fund_future_fundamental_impact_score"))
    if np.isfinite(project) and project >= 65:
        return "CAPACITY_EXPANSION"
    if any(token in sector for token in ("ENERGY", "MATERIAL", "MINING", "PLANTATION")):
        return "CYCLICAL_RECOVERY"
    if np.isfinite(earnings_growth) and earnings_growth > 0.20 and (not np.isfinite(revenue_growth) or revenue_growth < 0.10):
        return "TURNAROUND"
    if np.isfinite(revenue_growth) and revenue_growth >= 0.12 and np.isfinite(roe) and roe >= 0.12:
        return "TRUE_COMPOUNDER"
    if np.isfinite(future_score) and future_score >= 65:
        return "EVENT_DRIVEN_RERATING"
    return "GROWTH_VALUE_CANDIDATE"


def _merge_universe(
    prepared: Mapping[str, pd.DataFrame],
    fundamentals: pd.DataFrame | None,
    signals: pd.DataFrame | None,
    narrative_profiles: pd.DataFrame | None,
    macro_result: MacroRegimeResult,
    silent_frame: pd.DataFrame,
) -> pd.DataFrame:
    base = pd.DataFrame({"ticker": [_ticker(ticker) for ticker in prepared]})
    for frame, prefix in (
        (fundamentals, "fund_"),
        (signals, "sig_"),
        (narrative_profiles, "nar_"),
        (macro_result.issuer_map, "mac_"),
        (silent_frame, "flow_"),
    ):
        part = _prefixed(frame, prefix)
        if not part.empty:
            base = base.merge(part, on="ticker", how="left")
    return base.drop_duplicates("ticker", keep="last").reset_index(drop=True)


def build_next_leaders(universe: pd.DataFrame, config: ScanConfig | None = None) -> pd.DataFrame:
    cfg = config or ScanConfig()
    rows: list[dict[str, Any]] = []
    weights = NEXT_LEADER_WEIGHTS.__dict__
    for _, source in universe.iterrows():
        row = source.to_dict()
        business = _business_component(row)
        future = _future_component(row)
        valuation = _valuation_component(row)
        management = _management_component(row)
        macro = _macro_component(row)
        narrative_flow = _narrative_flow_component(row)
        technical = _technical_component(row)
        components = {
            "business_quality": business[:2],
            "future_fundamental": future[:2],
            "valuation_mos": valuation[:2],
            "management_capital": management[:2],
            "macro_sector": macro[:2],
            "narrative_flow": narrative_flow[:2],
            "technical_readiness": technical[:2],
        }
        final_score, coverage = _weighted_final(components, weights, min_coverage=55.0)
        blocked, blocked_reason = _hard_block(row)
        if blocked and np.isfinite(final_score):
            final_score = min(final_score, 30.0)

        technical_score = technical[0]
        entry = _first_num(row, ("sig_entry", "sig_entry_low"))
        rr1 = _first_num(row, ("sig_rr1",))
        adtv = _first_num(row, ("sig_adtv20_idr", "flow_adtv20_idr"))
        liquidity_ok = not np.isfinite(adtv) or adtv >= 250_000_000.0
        if blocked:
            status = "REJECT"
        elif not np.isfinite(final_score):
            status = "DATA_PENDING"
        elif final_score >= 75 and np.isfinite(technical_score) and technical_score >= 68 and liquidity_ok and (not np.isfinite(rr1) or rr1 >= 1.5):
            status = "BUY_ZONE"
        elif final_score >= 66:
            status = "WATCH"
        elif final_score >= 56:
            status = "WAIT"
        else:
            status = "RESEARCH_ONLY"
        candidate_type = _candidate_type(row, future[0])
        lane = "TURNAROUND_CYCLICAL" if candidate_type in {"TURNAROUND", "CYCLICAL_RECOVERY"} else "GROWTH_COMPOUNDER"

        account = _finite(getattr(cfg, "account_size_idr", 0.0), 0.0)
        max_weight = 0.25 if final_score >= 82 else 0.18 if final_score >= 74 else 0.12 if final_score >= 66 else 0.0
        allocation = account * max_weight if status == "BUY_ZONE" else 0.0
        lots = int(allocation // (entry * 100.0)) if allocation > 0 and np.isfinite(entry) and entry > 0 else 0
        reasons = [
            f"Business {business[0]:.1f}" if np.isfinite(business[0]) else "Business pending",
            f"Future {future[0]:.1f}" if np.isfinite(future[0]) else "Future evidence pending",
            f"Macro {macro[0]:.1f}" if np.isfinite(macro[0]) else "Macro exposure pending",
            f"Narrative-flow {narrative_flow[0]:.1f}" if np.isfinite(narrative_flow[0]) else "Narrative-flow pending",
        ]
        primary_risk = blocked_reason or str(_first(row, ("nar_narrative_primary_risk", "sig_warnings", "fund_fundamental_conflicts"), "Execution/fundamental evidence can change"))
        rows.append({
            "ticker": row["ticker"],
            "sector": _first(row, ("fund_sector", "mac_sector"), "UNKNOWN"),
            "candidate_type": candidate_type,
            "multibagger_lane": lane,
            "status": status,
            "v9_next_leader_score": round(final_score, 1) if np.isfinite(final_score) else np.nan,
            "score_coverage_pct": round(coverage, 1),
            "business_quality_score": round(business[0], 1) if np.isfinite(business[0]) else np.nan,
            "business_quality_coverage_pct": round(business[1], 1),
            "future_fundamental_score": round(future[0], 1) if np.isfinite(future[0]) else np.nan,
            "future_fundamental_coverage_pct": round(future[1], 1),
            "valuation_mos_score": round(valuation[0], 1) if np.isfinite(valuation[0]) else np.nan,
            "valuation_mos_coverage_pct": round(valuation[1], 1),
            "management_capital_score": round(management[0], 1) if np.isfinite(management[0]) else np.nan,
            "management_capital_coverage_pct": round(management[1], 1),
            "issuer_macro_alignment_score": round(macro[0], 1) if np.isfinite(macro[0]) else np.nan,
            "issuer_macro_alignment_coverage_pct": round(macro[1], 1),
            "narrative_flow_score": round(narrative_flow[0], 1) if np.isfinite(narrative_flow[0]) else np.nan,
            "narrative_flow_coverage_pct": round(narrative_flow[1], 1),
            "technical_readiness_score": round(technical[0], 1) if np.isfinite(technical[0]) else np.nan,
            "technical_readiness_coverage_pct": round(technical[1], 1),
            "silent_accumulation_score": _first_num(row, ("flow_silent_accumulation_score", "sig_silent_accumulation_score")),
            "narrative_event_score": _first_num(row, ("nar_narrative_event_effective_score",)),
            "retail_adoption_stage": _first(row, ("nar_retail_adoption_stage",), "UNKNOWN"),
            "macro_alignment_basis": macro[2],
            "last_price": _first_num(row, ("sig_last_price", "sig_close", "fund_last_price")),
            "entry": entry,
            "entry_low": _first_num(row, ("sig_entry_low",)),
            "entry_high": _first_num(row, ("sig_entry_high",)),
            "trigger": _first_num(row, ("sig_trigger", "sig_trigger_price")),
            "stop_loss": _first_num(row, ("sig_stop_loss",)),
            "tp1": _first_num(row, ("sig_tp1",)),
            "tp2": _first_num(row, ("sig_tp2",)),
            "rr1": rr1,
            "rr2": _first_num(row, ("sig_rr2",)),
            "adtv20_idr": adtv,
            "recommended_allocation_idr": round(allocation, 0),
            "recommended_lots": lots,
            "selected_reason": " | ".join(reasons),
            "primary_risk": primary_risk,
            "hard_block": blocked,
            "hard_block_reason": blocked_reason,
            "production_scoring_version": SIMPLE_FOCUS_VERSION,
            # Compatibility aliases for existing database/allocation readers.
            "v8_strategic_score": round(final_score, 1) if np.isfinite(final_score) else np.nan,
            "v8_production_score_coverage_pct": round(coverage, 1),
            "final_score": round(final_score, 1) if np.isfinite(final_score) else np.nan,
            "multibagger_quality_score": round(business[0], 1) if np.isfinite(business[0]) else np.nan,
            "multibagger_status": "MULTIBAGGER_A_CANDIDATE" if status == "BUY_ZONE" else "MULTIBAGGER_B_WATCH" if status == "WATCH" else status,
            "research_recommendation_status": status,
            "research_eligible": status not in {"REJECT", "DATA_PENDING"},
            "multibagger_rank_eligible": status not in {"REJECT", "DATA_PENDING", "RESEARCH_ONLY"},
            "multibagger_scoring_state": "SCORED" if np.isfinite(final_score) else "DATA_PENDING",
            "multibagger_evidence_class": "MACRO_JOS_EMIR",
            "multibagger_metric_coverage_pct": round(coverage, 1),
            "multibagger_metric_data_gate": "PASS" if coverage >= 55 else "PENDING",
            "execution_readiness_score": round(technical[0], 1) if np.isfinite(technical[0]) else np.nan,
            "multibagger_candidate_type": candidate_type,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rank_eligible"] = (
        pd.to_numeric(out["v9_next_leader_score"], errors="coerce").notna()
        & ~out["status"].isin(["DATA_PENDING", "REJECT"])
    )
    out = out.sort_values(
        ["rank_eligible", "v9_next_leader_score", "score_coverage_pct", "ticker"],
        ascending=[False, False, False, True], na_position="last", kind="stable",
    ).reset_index(drop=True)
    out["rank"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    eligible_index = out.index[out["rank_eligible"]]
    out.loc[eligible_index, "rank"] = np.arange(1, len(eligible_index) + 1)
    out["multibagger_selection_rank"] = out["rank"]
    out["multibagger_production_rank"] = out["rank"]
    return out


def build_swing_ready(universe: pd.DataFrame, next_leaders: pd.DataFrame) -> pd.DataFrame:
    leader_lookup = _latest(next_leaders).set_index("ticker").to_dict("index") if not next_leaders.empty else {}
    rows: list[dict[str, Any]] = []
    weights = SWING_WEIGHTS.__dict__
    for _, source in universe.iterrows():
        row = source.to_dict()
        technical = _technical_component(row)
        macro = _macro_component(row)
        narrative_flow = _narrative_flow_component(row)
        business = _business_component(row)
        risk_data = _risk_data_component(row)
        components = {
            "technical_execution": technical[:2],
            "macro_sector": macro[:2],
            "narrative_flow": narrative_flow[:2],
            "business_quality": business[:2],
            "risk_data": risk_data[:2],
        }
        final_score, coverage = _weighted_final(components, weights, min_coverage=60.0)
        blocked, blocked_reason = _hard_block(row)
        if blocked and np.isfinite(final_score):
            final_score = min(final_score, 25.0)
        setup_state = str(_first(row, ("sig_setup_status", "sig_status", "sig_decision_state"), "WATCHLIST")).upper()
        entry = _first_num(row, ("sig_entry", "sig_entry_low"))
        stop = _first_num(row, ("sig_stop_loss",))
        tp1 = _first_num(row, ("sig_tp1",))
        rr1 = _first_num(row, ("sig_rr1",))
        atomic_plan = all(np.isfinite(v) and v > 0 for v in (entry, stop, tp1)) and stop < entry < tp1
        if blocked:
            status = "REJECT"
        elif not np.isfinite(final_score):
            status = "DATA_PENDING"
        elif final_score >= 74 and atomic_plan and (not np.isfinite(rr1) or rr1 >= 1.5):
            status = "EXECUTION_READY"
        elif final_score >= 65 and atomic_plan:
            status = "ENTRY_PLAN_READY"
        elif final_score >= 56:
            status = "WATCHLIST"
        else:
            status = "WAIT"
        leader = leader_lookup.get(row["ticker"], {})
        rows.append({
            "ticker": row["ticker"],
            "sector": _first(row, ("fund_sector", "mac_sector"), "UNKNOWN"),
            "status": status,
            "v9_swing_score": round(final_score, 1) if np.isfinite(final_score) else np.nan,
            "score_coverage_pct": round(coverage, 1),
            "technical_execution_score": round(technical[0], 1) if np.isfinite(technical[0]) else np.nan,
            "issuer_macro_alignment_score": round(macro[0], 1) if np.isfinite(macro[0]) else np.nan,
            "narrative_flow_score": round(narrative_flow[0], 1) if np.isfinite(narrative_flow[0]) else np.nan,
            "business_quality_score": round(business[0], 1) if np.isfinite(business[0]) else np.nan,
            "risk_data_score": round(risk_data[0], 1) if np.isfinite(risk_data[0]) else np.nan,
            "next_leader_score": leader.get("v9_next_leader_score", np.nan),
            "setup_status": setup_state,
            "strategy": _first(row, ("sig_strategy", "sig_setup"), "CORE_SWING"),
            "last_price": _first_num(row, ("sig_last_price", "sig_close", "fund_last_price")),
            "entry": entry,
            "entry_low": _first_num(row, ("sig_entry_low",)),
            "entry_high": _first_num(row, ("sig_entry_high",)),
            "trigger_price": _first_num(row, ("sig_trigger_price", "sig_trigger")),
            "stop_loss": stop,
            "tp1": tp1,
            "tp2": _first_num(row, ("sig_tp2",)),
            "rr1": rr1,
            "rr2": _first_num(row, ("sig_rr2",)),
            "stockbit_order_lots": _first_num(row, ("sig_stockbit_order_lots", "sig_lots")),
            "next_action": status,
            "selected_reason": f"Technical {technical[0]:.1f} | Macro {macro[0]:.1f} | Narrative-flow {narrative_flow[0]:.1f}" if all(np.isfinite(v) for v in (technical[0], macro[0], narrative_flow[0])) else "Evidence partially pending",
            "primary_risk": blocked_reason or str(_first(row, ("nar_narrative_primary_risk", "sig_warnings"), "Setup can invalidate below stop loss")),
            "hard_block": blocked,
            "hard_block_reason": blocked_reason,
            "production_scoring_version": SIMPLE_FOCUS_VERSION,
            # Compatibility aliases.
            "core_priority_score": round(final_score, 1) if np.isfinite(final_score) else np.nan,
            "final_score": round(final_score, 1) if np.isfinite(final_score) else np.nan,
            "decision_state": status,
            "order_builder_eligible": status in {"EXECUTION_READY", "ENTRY_PLAN_READY"},
            "order_ready": status == "EXECUTION_READY",
            "v8_production_score_coverage_pct": round(coverage, 1),
            "v8_technical_score": round(technical[0], 1) if np.isfinite(technical[0]) else np.nan,
            "v8_market_sector_score": round(macro[0], 1) if np.isfinite(macro[0]) else np.nan,
            "v8_narrative_alignment_score": round(narrative_flow[0], 1) if np.isfinite(narrative_flow[0]) else np.nan,
            "v8_flow_score": _flow_score_from_row(row)[0],
            "v8_data_validation_score": round(risk_data[0], 1) if np.isfinite(risk_data[0]) else np.nan,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rank_eligible"] = (
        pd.to_numeric(out["v9_swing_score"], errors="coerce").notna()
        & ~out["status"].isin(["DATA_PENDING", "REJECT"])
    )
    out = out.sort_values(
        ["rank_eligible", "v9_swing_score", "score_coverage_pct", "ticker"],
        ascending=[False, False, False, True], na_position="last", kind="stable",
    ).reset_index(drop=True)
    out["rank"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    eligible_index = out.index[out["rank_eligible"]]
    out.loc[eligible_index, "rank"] = np.arange(1, len(eligible_index) + 1)
    out["profit_rank"] = out["rank"]
    out["strategy_rank"] = out["rank"]
    return out


def build_evidence_audit(next_leaders: pd.DataFrame, swing_ready: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    leader_specs = (
        ("BUSINESS_QUALITY", "business_quality_score", "business_quality_coverage_pct", 25.0),
        ("FUTURE_FUNDAMENTAL", "future_fundamental_score", "future_fundamental_coverage_pct", 20.0),
        ("VALUATION_MOS", "valuation_mos_score", "valuation_mos_coverage_pct", 15.0),
        ("MANAGEMENT_CAPITAL", "management_capital_score", "management_capital_coverage_pct", 10.0),
        ("MACRO_SECTOR", "issuer_macro_alignment_score", "issuer_macro_alignment_coverage_pct", 10.0),
        ("NARRATIVE_FLOW", "narrative_flow_score", "narrative_flow_coverage_pct", 15.0),
        ("TECHNICAL_READINESS", "technical_readiness_score", "technical_readiness_coverage_pct", 5.0),
    )
    for _, row in next_leaders.iterrows():
        for component, score_col, cov_col, weight in leader_specs:
            score = _finite(row.get(score_col), np.nan)
            coverage = _finite(row.get(cov_col), 0.0)
            rows.append({
                "model": "THE_NEXT_LEADER", "ticker": row.get("ticker", ""), "component": component,
                "weight_pct": weight, "score": score, "coverage_pct": coverage,
                "evidence_state": "SCORED" if np.isfinite(score) and coverage >= 50 else "PARTIAL" if coverage > 0 else "MISSING",
                "model_version": SIMPLE_FOCUS_VERSION,
            })
    swing_specs = (
        ("TECHNICAL_EXECUTION", "technical_execution_score", 40.0),
        ("MACRO_SECTOR", "issuer_macro_alignment_score", 15.0),
        ("NARRATIVE_FLOW", "narrative_flow_score", 25.0),
        ("BUSINESS_QUALITY", "business_quality_score", 10.0),
        ("RISK_DATA", "risk_data_score", 10.0),
    )
    for _, row in swing_ready.iterrows():
        for component, score_col, weight in swing_specs:
            score = _finite(row.get(score_col), np.nan)
            rows.append({
                "model": "SWING_READY", "ticker": row.get("ticker", ""), "component": component,
                "weight_pct": weight, "score": score, "coverage_pct": row.get("score_coverage_pct", np.nan),
                "evidence_state": "SCORED" if np.isfinite(score) else "MISSING",
                "model_version": SIMPLE_FOCUS_VERSION,
            })
    return pd.DataFrame(rows)


def scoring_contract_frame() -> pd.DataFrame:
    rows = []
    for name, weight in NEXT_LEADER_WEIGHTS.__dict__.items():
        rows.append({"model": "THE_NEXT_LEADER", "component": name.upper(), "weight_pct": 100.0 * weight,
                     "purpose": "Jos-style quality/future/value with macro and Emir confirmation",
                     "model_version": SIMPLE_FOCUS_VERSION})
    for name, weight in SWING_WEIGHTS.__dict__.items():
        rows.append({"model": "SWING_READY", "component": name.upper(), "weight_pct": 100.0 * weight,
                     "purpose": "Executable swing with macro, narrative-flow and risk gates",
                     "model_version": SIMPLE_FOCUS_VERSION})
    return pd.DataFrame(rows)


def build_simple_focus(
    prepared: Mapping[str, pd.DataFrame],
    *,
    fundamentals: pd.DataFrame | None,
    signals: pd.DataFrame | None,
    news_review: pd.DataFrame | None,
    market_status: pd.DataFrame | None,
    benchmark: pd.DataFrame | None,
    macro_result: MacroRegimeResult,
    config: ScanConfig | None = None,
    existing_narrative_events: pd.DataFrame | None = None,
    existing_narrative_outcomes: pd.DataFrame | None = None,
    universe_tickers: Sequence[str] | None = None,
    precomputed_silent_frame: pd.DataFrame | None = None,
    precomputed_narrative: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    cfg = config or ScanConfig()
    base_prepared: Mapping[str, pd.DataFrame]
    if universe_tickers is not None:
        ordered = list(dict.fromkeys(_ticker(value) for value in universe_tickers if _ticker(value)))
        base_prepared = {ticker: prepared.get(ticker, pd.DataFrame()) for ticker in ordered}
    else:
        base_prepared = prepared

    if isinstance(precomputed_silent_frame, pd.DataFrame):
        silent_frame = precomputed_silent_frame.copy()
        silent_profiles = {
            _ticker(row.get("ticker")): {key: value for key, value in row.items() if key != "ticker"}
            for row in silent_frame.to_dict("records") if _ticker(row.get("ticker"))
        }
    else:
        silent_profiles, silent_frame = _build_silent_profiles(base_prepared)

    if isinstance(precomputed_narrative, Mapping):
        narrative = {
            "events": precomputed_narrative.get("events", pd.DataFrame()),
            "outcomes": precomputed_narrative.get("outcomes", pd.DataFrame()),
            "profiles": precomputed_narrative.get("profiles", pd.DataFrame()),
            "audit": precomputed_narrative.get("audit", pd.DataFrame()),
        }
    else:
        narrative = build_narrative_intelligence(
            prepared=base_prepared,
            fundamentals=fundamentals,
            news_review=news_review,
            market_status=market_status,
            existing_events=existing_narrative_events,
            existing_outcomes=existing_narrative_outcomes,
            benchmark=benchmark,
            silent_profiles=silent_profiles,
            scan_config=cfg,
        )
    universe = _merge_universe(
        base_prepared,
        fundamentals,
        signals,
        narrative.get("profiles", pd.DataFrame()),
        macro_result,
        silent_frame,
    )
    next_leaders_all = build_next_leaders(universe, cfg)
    swing_ready_all = build_swing_ready(universe, next_leaders_all)
    next_leaders = next_leaders_all.loc[next_leaders_all.get("rank_eligible", False)].copy() if not next_leaders_all.empty else next_leaders_all
    swing_ready = swing_ready_all.loc[swing_ready_all.get("rank_eligible", False)].copy() if not swing_ready_all.empty else swing_ready_all
    evidence = build_evidence_audit(next_leaders_all, swing_ready_all)
    return {
        "next_leaders": next_leaders,
        "swing_ready": swing_ready,
        "next_leaders_all": next_leaders_all,
        "swing_ready_all": swing_ready_all,
        "multibagger": next_leaders,
        "multibagger_research": next_leaders_all,
        "core_swing": swing_ready,
        "profit_order_builder": swing_ready,
        "narrative_events": narrative.get("events", pd.DataFrame()),
        "narrative_event_outcomes": narrative.get("outcomes", pd.DataFrame()),
        "narrative_profiles": narrative.get("profiles", pd.DataFrame()),
        "narrative_engine_audit": narrative.get("audit", pd.DataFrame()),
        "production_evidence_detail": evidence,
        "production_scoring_audit": scoring_contract_frame(),
        "universe_decision_frame": universe,
    }


__all__ = [
    "SIMPLE_FOCUS_VERSION",
    "NextLeaderWeights",
    "SwingWeights",
    "NEXT_LEADER_WEIGHTS",
    "SWING_WEIGHTS",
    "build_silent_profiles",
    "build_next_leaders",
    "build_swing_ready",
    "build_evidence_audit",
    "scoring_contract_frame",
    "build_simple_focus",
]
