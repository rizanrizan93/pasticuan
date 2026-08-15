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
from issuer_classification import canonical_sector
from scanner import ScanConfig, safe_number, silent_accumulation_profile, round_idx_price
from decision_overlay import apply_methodology_guardrails, enrich_silent_profile
from real_money_guard import apply_real_money_authorization, fundamental_conviction_profile
from fundamental_calibration import reporting_refresh_profile, latest_growth_profile, classify_thesis_archetype
from release_contract import SCANNER_RELEASE_VERSION


SIMPLE_FOCUS_VERSION = SCANNER_RELEASE_VERSION


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

NEXT_LEADER_MIN_COVERAGE_PCT = 70.0
NEXT_LEADER_MIN_BUSINESS_COVERAGE_PCT = 55.0
NEXT_LEADER_MIN_FUTURE_COVERAGE_PCT = 40.0
NEXT_LEADER_MAX_STATEMENT_AGE_DAYS = 300.0
SWING_MIN_COVERAGE_PCT = 65.0
SWING_MIN_TECHNICAL_SCORE = 40.0

DIRECT_FORWARD_SCORE_FIELDS = (
    "nar_forward_project_pipeline_score",
    "nar_forward_future_fundamental_impact_score",
)

# Research quality and decision priority are deliberately separate.  These
# bounded penalties only affect which names deserve attention first; they never
# rewrite the underlying business/future-fundamental thesis score.
DECISION_PRIORITY_ANTI_CHASE_PENALTY = 3.0
DECISION_PRIORITY_TECHNICAL_FLOOR = 50.0
DECISION_PRIORITY_TECHNICAL_SLOPE = 0.08
DECISION_PRIORITY_TECHNICAL_MAX_PENALTY = 4.0
DECISION_PRIORITY_FLOW_FLOOR = 40.0
DECISION_PRIORITY_FLOW_SLOPE = 0.06
DECISION_PRIORITY_FLOW_MAX_PENALTY = 2.4
DECISION_PRIORITY_DISTRIBUTION_FLOOR = 45.0
DECISION_PRIORITY_DISTRIBUTION_SLOPE = 0.08
DECISION_PRIORITY_DISTRIBUTION_MAX_PENALTY = 2.0
DECISION_PRIORITY_DISTRIBUTION_BLOCK_PENALTY = 8.0


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
    """Score cash-flow direction without rewarding issuer size.

    Absolute rupiah cash flow is not comparable across issuers.  Magnitude is
    therefore deliberately ignored here; cash-conversion/FCF-yield fields carry
    the relative quality information elsewhere in the model.
    """
    if not np.isfinite(value):
        return np.nan
    if value > 0:
        return 82.0
    if value == 0:
        return 45.0
    return 15.0


def _ownership_alignment_score(value: float) -> float:
    fraction = _fraction(value)
    if not np.isfinite(fraction):
        return np.nan
    pct = 100.0 * fraction
    if pct <= 0.0:
        return 40.0
    if pct < 5.0:
        return 50.0 + 4.0 * pct
    if pct <= 60.0:
        return 70.0 + 20.0 * (pct - 5.0) / 55.0
    if pct <= 75.0:
        return 90.0 - 2.0 * (pct - 60.0)
    return max(25.0, 60.0 - 1.4 * (pct - 75.0))


def _build_silent_profiles(prepared: Mapping[str, pd.DataFrame]) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    profiles: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for ticker, frame in prepared.items():
        key = _ticker(ticker)
        try:
            profile = enrich_silent_profile(frame, dict(silent_accumulation_profile(frame)))
        except Exception as exc:
            profile = {
                "silent_accumulation_score": np.nan,
                "silent_accumulation_confidence": 0.0,
                "silent_accumulation_state": "FAIL_SOFT",
                "silent_accumulation_reason": f"{type(exc).__name__}: {str(exc)[:120]}",
            }
            profile = enrich_silent_profile(frame, profile)
        profiles[key] = profile
        rows.append({"ticker": key, **profile})
    return profiles, pd.DataFrame(rows)


def build_silent_profiles(prepared: Mapping[str, pd.DataFrame]) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Public chunk-safe silent-accumulation profile builder."""
    return _build_silent_profiles(prepared)


def _business_component(row: Mapping[str, Any]) -> tuple[float, float, str]:
    # Latest-period inflection and acceleration are explicitly represented so
    # a genuine turnaround is not buried by weaker trailing ratios.
    score, coverage, evidence = _metric_score(row, (
        (("fund_revenue_growth", "fund_history_revenue_growth_yoy"), 0.12, _growth_score),
        (("fund_earnings_growth", "fund_history_net_income_growth_yoy"), 0.12, _growth_score),
        (("fund_history_revenue_growth_acceleration",), 0.07, lambda v: _ratio_score(v, -0.10, 0.10)),
        (("fund_history_earnings_growth_acceleration",), 0.07, lambda v: _ratio_score(v, -0.20, 0.20)),
        (("fund_history_gross_profit_growth", "fund_gross_profit_growth"), 0.07, _growth_score),
        (("fund_fundamental_inflection_score",), 0.08, lambda v: v),
        (("fund_roe", "fund_history_roe"), 0.10, lambda v: _ratio_score(_fraction(v), 0.05, 0.22)),
        (("fund_roa", "fund_history_roa"), 0.05, lambda v: _ratio_score(_fraction(v), 0.02, 0.12)),
        (("fund_operating_margin", "fund_history_operating_margin"), 0.07, lambda v: _ratio_score(_fraction(v), 0.02, 0.18)),
        (("fund_net_margin", "fund_history_net_margin"), 0.07, lambda v: _ratio_score(_fraction(v), 0.03, 0.20)),
        (("fund_operating_cash_flow", "fund_history_ocf_ttm"), 0.05, _positive_cash_score),
        (("fund_free_cash_flow", "fund_history_fcf_ttm"), 0.05, _positive_cash_score),
        (("fund_history_cash_conversion", "fund_cash_conversion_ttm"), 0.04, lambda v: _ratio_score(v, 0.45, 1.25)),
        (("fund_debt_equity", "fund_history_debt_equity"), 0.02, lambda v: _inverse_ratio_score(v, 0.20, 2.00)),
        (("fund_current_ratio",), 0.01, lambda v: _ratio_score(v, 0.70, 2.00)),
        (("fund_cash_to_debt",), 0.01, lambda v: _ratio_score(v, 0.10, 1.20)),
    ))
    if np.isfinite(score):
        if coverage < 50.0:
            score = min(score, 75.0)
        elif coverage < 65.0:
            score = min(score, 88.0)
    return score, coverage, " | ".join(evidence)


def _future_component(row: Mapping[str, Any]) -> tuple[float, float, str]:
    # This pillar is deliberately direct-only.  Realised growth, ROIC, margins,
    # cash generation and their derived persistence/capacity proxies belong to
    # Business Quality.  Reusing them here produced an observed 0.895
    # cross-sectional correlation and double-counted the same financial facts.
    if not _truthy(row.get("nar_forward_source_quorum_verified")):
        return np.nan, 0.0, str(row.get("nar_forward_evidence_state", ""))
    evidence_coverage = _first_num(row, ("nar_forward_project_data_coverage_pct",))
    if not np.isfinite(evidence_coverage) or evidence_coverage <= 0.0:
        return np.nan, 0.0, "DIRECT_FORWARD_COVERAGE_MISSING"
    specs = (
        (("nar_forward_future_fundamental_impact_score",), 0.55, lambda value: value),
        (("nar_forward_project_pipeline_score",), 0.45, lambda value: value),
    )
    score, observed_pct, evidence = _metric_score(row, specs)
    if not np.isfinite(score):
        return np.nan, 0.0, "DIRECT_FORWARD_SCORE_MISSING"
    coverage = min(100.0, np.clip(evidence_coverage, 0.0, 100.0) * observed_pct / 100.0)
    if coverage < 40.0:
        score = min(score, 72.0)
    elif coverage < 70.0:
        score = min(score, 85.0)
    basis = str(row.get("nar_forward_evidence_basis", "")).strip()
    return score, coverage, " | ".join([*evidence, basis] if basis else evidence)


def _valuation_component(row: Mapping[str, Any]) -> tuple[float, float, str]:
    local = dict(row)
    price_to_book = _first_num(local, ("fund_price_to_book",))
    roe = _fraction(_first_num(local, ("fund_history_roe", "fund_roe")))
    if np.isfinite(price_to_book) and np.isfinite(roe) and roe > 0:
        local["fund_pb_to_roe_ratio"] = price_to_book / (100.0 * roe)
    sector = canonical_sector(_first(local, ("fund_sector", "mac_sector"), "UNKNOWN"))
    financial_model = str(local.get("fund_fundamental_model", "")).upper() == "FINANCIAL" or sector == "FINANCIALS"
    if financial_model:
        specs = (
            (("fund_trailing_pe",), 0.30, lambda v: _inverse_ratio_score(v, 6.0, 28.0)),
            (("fund_earnings_yield",), 0.25, lambda v: _ratio_score(_fraction(v), 0.02, 0.12)),
            (("fund_price_to_book",), 0.25, lambda v: _inverse_ratio_score(v, 0.70, 4.00)),
            (("fund_pb_to_roe_ratio",), 0.20, lambda v: _inverse_ratio_score(v, 0.05, 0.35)),
        )
    else:
        specs = (
            (("fund_trailing_pe",), 0.20, lambda v: _inverse_ratio_score(v, 6.0, 35.0)),
            (("fund_earnings_yield",), 0.15, lambda v: _ratio_score(_fraction(v), 0.00, 0.12)),
            (("fund_fcf_yield",), 0.25, lambda v: _ratio_score(_fraction(v), 0.00, 0.10)),
            (("fund_ev_ebitda",), 0.20, lambda v: _inverse_ratio_score(v, 3.0, 20.0)),
            (("fund_peg_ratio",), 0.10, lambda v: _inverse_ratio_score(v, 0.50, 2.50)),
            (("fund_price_to_book",), 0.10, lambda v: _inverse_ratio_score(v, 0.70, 5.00)),
        )
    score, coverage, evidence = _metric_score(local, specs)
    if np.isfinite(score):
        if coverage < 50.0:
            score = min(score, 80.0)
        elif coverage < 70.0:
            score = min(score, 90.0)
    return score, coverage, " | ".join(evidence)


def _management_component(row: Mapping[str, Any]) -> tuple[float, float, str]:
    specs = (
        (("nar_issuer_action_alignment_effective_score",), 0.55, lambda v: v),
        (("fund_insider_ownership_pct",), 0.15, _ownership_alignment_score),
        (("fund_governance_overall_risk", "fund_governance_audit_risk"), 0.15,
         lambda v: _inverse_ratio_score(v, 1.0, 10.0)),
        (("fund_history_share_dilution_yoy", "fund_share_dilution_yoy"), 0.15,
         lambda v: _inverse_ratio_score(_fraction(v), 0.00, 0.15)),
    )
    score, _, evidence = _metric_score(row, specs)
    coverage = 0.0
    for names, weight, _ in specs:
        value = _first_num(row, names)
        if not np.isfinite(value):
            continue
        selected = next((name for name in names if np.isfinite(_finite(row.get(name), np.nan))), names[0])
        coverage_name = {
            "nar_issuer_action_alignment_effective_score": "nar_issuer_action_alignment_coverage_pct",
        }.get(selected)
        evidence_coverage = _first_num(row, (coverage_name,)) if coverage_name else 100.0
        coverage += 100.0 * weight * (
            np.clip(evidence_coverage, 0.0, 100.0) / 100.0
            if np.isfinite(evidence_coverage) else 0.0
        )
    # "No dilution" alone is useful capital-structure evidence, but it cannot
    # establish management quality.  Require either sourced issuer-action
    # alignment or at least two independent management/capital observations.
    direct_alignment = "nar_issuer_action_alignment_effective_score" in evidence
    if not direct_alignment and len(evidence) < 2:
        return np.nan, 0.0, "MANAGEMENT_EVIDENCE_INSUFFICIENT"
    if coverage <= 0.0:
        return np.nan, 0.0, "MANAGEMENT_COVERAGE_MISSING"
    return score, coverage, " | ".join(evidence)


def _macro_component(row: Mapping[str, Any]) -> tuple[float, float, str]:
    score = _first_num(row, ("mac_issuer_macro_alignment_score",))
    coverage = _first_num(row, ("mac_issuer_macro_alignment_coverage_pct",))
    if not np.isfinite(score):
        return np.nan, 0.0, ""
    # A score without lineage/coverage is research information only.  Never
    # manufacture 60% evidence because that can leak into production gates.
    return _clip(score), _clip(coverage) if np.isfinite(coverage) else 0.0, str(row.get("mac_issuer_macro_alignment_basis", ""))


def _flow_score_from_row(row: Mapping[str, Any]) -> tuple[float, float]:
    score = _first_num(row, ("flow_silent_accumulation_score", "sig_silent_accumulation_score"))
    coverage = _first_num(row, ("flow_silent_accumulation_confidence", "flow_silent_accumulation_data_coverage"))
    # Missing confidence is UNKNOWN, not an inferred 70%.  The quality score may
    # remain visible for research, while weighted production coverage stays zero.
    return (_clip(score), _clip(coverage) if np.isfinite(coverage) else 0.0)


def _narrative_flow_component(row: Mapping[str, Any]) -> tuple[float, float, str]:
    # Pair score and coverage from the same evidence family.  A zero event-level
    # coverage must not shadow broad narrative evidence when the event-level score
    # itself is missing.
    event_score = _first_num(row, ("nar_narrative_event_effective_score",))
    if np.isfinite(event_score):
        narrative = event_score
        narrative_cov = _first_num(row, ("nar_narrative_event_coverage_pct",))
    else:
        narrative = _first_num(row, ("nar_narrative_effective_score",))
        narrative_cov = _first_num(row, ("nar_narrative_evidence_coverage_pct",))
    flow, flow_cov = _flow_score_from_row(row)
    parts: list[tuple[float, float, float, str]] = []
    if np.isfinite(narrative):
        # Missing narrative coverage is UNKNOWN (0), not a synthetic 60%.
        parts.append((narrative, 0.60, narrative_cov if np.isfinite(narrative_cov) else 0.0, "SOURCED_NARRATIVE"))
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
        "WATCHLIST_ENTRY": 60.0,
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


def _research_accumulation_plan(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build a research-only re-entry zone from observed technical supports.

    This is deliberately separate from the executable setup plan.  It may help a
    Next-Leader investor monitor a pullback even when SMC/ICT execution is not yet
    valid, but it never creates an order, RR, position size, or authorization.
    """
    existing_low = _first_num(row, ("sig_entry_low",))
    existing_high = _first_num(row, ("sig_entry_high",))
    existing_entry = _first_num(row, ("sig_entry",))
    if np.isfinite(existing_low) and np.isfinite(existing_high) and existing_low > 0 and existing_low <= existing_high:
        preferred = existing_entry if np.isfinite(existing_entry) and existing_entry > 0 else (existing_low + existing_high) / 2.0
        return {
            "research_accumulation_zone_low": existing_low,
            "research_accumulation_zone_high": existing_high,
            "research_preferred_reentry": preferred,
            "research_invalidation_reference": _first_num(row, ("sig_stop_loss",)),
            "research_zone_state": "TECHNICAL_PLAN_REFERENCE",
            "research_zone_basis": "EXISTING_SCANNER_SETUP_LEVELS",
        }

    last = _first_num(row, ("sig_last_price", "sig_close", "fund_last_price"))
    atr = _first_num(row, ("sig_atr14",))
    if not (np.isfinite(last) and last > 0 and np.isfinite(atr) and atr > 0):
        return {
            "research_accumulation_zone_low": np.nan, "research_accumulation_zone_high": np.nan,
            "research_preferred_reentry": np.nan, "research_invalidation_reference": np.nan,
            "research_zone_state": "UNAVAILABLE", "research_zone_basis": "STRUCTURAL_REFERENCE_MISSING",
        }
    named_levels = [
        ("EMA20", _first_num(row, ("sig_ema20",))),
        ("EMA50", _first_num(row, ("sig_ema50",))),
        ("VWAP20", _first_num(row, ("sig_vwap20",))),
        ("PIVOT_LOW", _first_num(row, ("sig_last_pivot_low",))),
    ]
    supports = [(name, value) for name, value in named_levels if np.isfinite(value) and value > 0 and value <= last * 1.02]
    if not supports:
        return {
            "research_accumulation_zone_low": np.nan, "research_accumulation_zone_high": np.nan,
            "research_preferred_reentry": np.nan, "research_invalidation_reference": np.nan,
            "research_zone_state": "UNAVAILABLE", "research_zone_basis": "NO_OBSERVED_SUPPORT_BELOW_PRICE",
        }
    # Nearest observed support anchors the monitoring zone; no synthetic price target is used.
    basis, anchor = max(supports, key=lambda item: item[1])
    low = round_idx_price(max(1.0, anchor - 0.35 * atr), "down")
    high = round_idx_price(min(last, anchor + 0.20 * atr), "up")
    preferred = round_idx_price(anchor, "nearest")
    lower_supports = [value for _, value in supports if value < anchor - 1e-9]
    invalidation_anchor = max(lower_supports) if lower_supports else anchor - 0.75 * atr
    invalidation = round_idx_price(max(1.0, invalidation_anchor - 0.25 * atr), "down")
    if not (low and high and preferred and low < high):
        return {
            "research_accumulation_zone_low": np.nan, "research_accumulation_zone_high": np.nan,
            "research_preferred_reentry": np.nan, "research_invalidation_reference": np.nan,
            "research_zone_state": "UNAVAILABLE", "research_zone_basis": "INVALID_STRUCTURAL_ZONE",
        }
    return {
        "research_accumulation_zone_low": float(low),
        "research_accumulation_zone_high": float(high),
        "research_preferred_reentry": float(preferred),
        "research_invalidation_reference": float(invalidation) if invalidation else np.nan,
        "research_zone_state": "RESEARCH_ONLY_AVAILABLE",
        "research_zone_basis": f"OBSERVED_{basis}_ATR_BAND",
    }


def _entry_zone_role(row: Mapping[str, Any], entry: float, stop: float, trigger: float) -> str:
    low = _first_num(row, ("sig_entry_low",))
    high = _first_num(row, ("sig_entry_high",))
    if not (np.isfinite(low) and np.isfinite(high) and low > 0 and low <= high):
        return "NO_ZONE"
    if np.isfinite(stop) and stop >= low and np.isfinite(trigger) and trigger > high:
        return "PULLBACK_OBSERVATION_ZONE"
    if np.isfinite(stop) and stop < low and (not np.isfinite(entry) or low <= entry <= max(high, entry)):
        return "EXECUTABLE_ENTRY_ZONE"
    return "REFERENCE_ZONE"


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
        reliability = np.clip(coverage, 0.0, 100.0) / 100.0 if observed else 0.0
        # Shrink each evidence family independently.  Applying one aggregate
        # shrink after mixing scores let a 15%-covered Management=100 influence
        # ranking as strongly as a fully covered component.
        effective = 50.0 + reliability * (score - 50.0) if observed else 50.0
        raw_sum += weight * effective
        coverage_sum += weight * (100.0 * reliability)
    if coverage_sum < min_coverage:
        return np.nan, coverage_sum
    return _clip(raw_sum), _clip(coverage_sum)


def _coverage_gap_profile(
    components: Mapping[str, tuple[float, float]],
    weights: Mapping[str, float],
) -> dict[str, Any]:
    """Explain fixed-denominator coverage without manufacturing evidence."""
    gaps: list[tuple[str, float]] = []
    for name, weight in weights.items():
        component_score, component_coverage = components[name]
        observed_coverage = (
            float(np.clip(component_coverage, 0.0, 100.0))
            if np.isfinite(component_score) and np.isfinite(component_coverage) and component_coverage > 0.0
            else 0.0
        )
        gaps.append((name, float(weight) * (100.0 - observed_coverage)))
    gaps.sort(key=lambda item: (-item[1], item[0]))
    total_gap = sum(points for _, points in gaps)
    material = [
        f"{name.upper()}:{points:.1f}"
        for name, points in gaps
        if points >= 0.5
    ]
    return {
        "coverage_gap_pct": round(total_gap, 1),
        "coverage_primary_gap": material[0] if material else "NONE",
        "coverage_recovery_priority": " | ".join(material[:3]),
    }


def _append_reason(existing: Any, reasons: Sequence[str]) -> str:
    existing_text = (
        "" if existing is None
        or (isinstance(existing, (float, np.floating)) and np.isnan(existing))
        else str(existing)
    )
    values = [part.strip() for part in existing_text.split("|") if part.strip()]
    values.extend(reason for reason in reasons if reason)
    return " | ".join(dict.fromkeys(values))


def _apply_decision_priority_guardrail(frame: pd.DataFrame) -> pd.DataFrame:
    """Downrank execution contradictions while preserving raw thesis scores.

    Missing technical/flow confidence is UNKNOWN and receives no penalty.  A
    penalty is only allowed when the corresponding evidence coverage is high
    enough to support a negative conclusion.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()

    out = frame.copy()
    out["raw_ranking_score"] = pd.to_numeric(out.get("ranking_score"), errors="coerce")
    for index, row in out.iterrows():
        raw_score = _finite(row.get("raw_ranking_score"), np.nan)
        if not np.isfinite(raw_score):
            continue

        penalty = 0.0
        distribution_penalty = 0.0
        reasons: list[str] = []

        methodology_pass = _truthy(row.get("methodology_gate_pass", True))
        distribution_risk = _finite(row.get("distribution_risk_score"), np.nan)
        if not methodology_pass or (np.isfinite(distribution_risk) and distribution_risk >= 68.0):
            distribution_penalty = DECISION_PRIORITY_DISTRIBUTION_BLOCK_PENALTY
            reasons.append("DISTRIBUTION_BLOCK")
        elif np.isfinite(distribution_risk) and distribution_risk >= DECISION_PRIORITY_DISTRIBUTION_FLOOR:
            distribution_penalty = min(
                DECISION_PRIORITY_DISTRIBUTION_MAX_PENALTY,
                (distribution_risk - DECISION_PRIORITY_DISTRIBUTION_FLOOR)
                * DECISION_PRIORITY_DISTRIBUTION_SLOPE,
            )
            if distribution_penalty > 0.0:
                reasons.append("ELEVATED_DISTRIBUTION_RISK")
        penalty += distribution_penalty

        if _truthy(row.get("anti_chase_gate")):
            penalty += DECISION_PRIORITY_ANTI_CHASE_PENALTY
            reasons.append("ANTI_CHASE_WAIT_REACCUMULATION")

        technical_score = _finite(row.get("technical_readiness_score"), np.nan)
        technical_coverage = _finite(row.get("technical_readiness_coverage_pct"), 0.0)
        if technical_coverage >= 50.0 and np.isfinite(technical_score) and technical_score < DECISION_PRIORITY_TECHNICAL_FLOOR:
            penalty += min(
                DECISION_PRIORITY_TECHNICAL_MAX_PENALTY,
                (DECISION_PRIORITY_TECHNICAL_FLOOR - technical_score)
                * DECISION_PRIORITY_TECHNICAL_SLOPE,
            )
            reasons.append("WEAK_TECHNICAL_CONFIRMATION")

        flow_score = _finite(row.get("silent_accumulation_score"), np.nan)
        flow_confidence = _finite(row.get("silent_accumulation_confidence"), 0.0)
        if flow_confidence >= 50.0 and np.isfinite(flow_score) and flow_score < DECISION_PRIORITY_FLOW_FLOOR:
            penalty += min(
                DECISION_PRIORITY_FLOW_MAX_PENALTY,
                (DECISION_PRIORITY_FLOW_FLOOR - flow_score)
                * DECISION_PRIORITY_FLOW_SLOPE,
            )
            reasons.append("WEAK_SMART_MONEY_CONFIRMATION")

        penalty = round(float(penalty), 2)
        guarded_score = max(0.0, min(100.0, raw_score - penalty))
        out.at[index, "ranking_score"] = round(guarded_score, 1)
        out.at[index, "ranking_guardrail_penalty_points"] = penalty
        out.at[index, "ranking_guardrail_state"] = "DOWNRANKED" if penalty > 0.0 else "CLEAN"
        out.at[index, "ranking_guardrail_reasons"] = " | ".join(reasons)
        out.at[index, "distribution_penalty_points"] = round(distribution_penalty, 2)
        out.at[index, "distribution_evidence_state"] = (
            "DISTRIBUTION_BLOCK" if "DISTRIBUTION_BLOCK" in reasons
            else "ELEVATED_DISTRIBUTION_RISK" if distribution_penalty > 0.0
            else "NO_DISTRIBUTION_PENALTY"
        )
        if penalty > 0.0:
            state = str(row.get("ranking_score_state") or "RANKING_SCORE")
            out.at[index, "ranking_score_state"] = (
                state if state.endswith("_GUARDED") else state + "_GUARDED"
            )
            out.at[index, "scoring_reason_codes"] = _append_reason(
                row.get("scoring_reason_codes"), reasons,
            )
            out.at[index, "top_negative_drivers"] = _append_reason(
                row.get("top_negative_drivers"), reasons,
            )
            out.at[index, "primary_risk"] = _append_reason(
                row.get("primary_risk"),
                [f"DECISION_PRIORITY_PENALTY:{penalty:.2f}", *reasons],
            )
    return out


def _hard_block(row: Mapping[str, Any], cfg: ScanConfig | None = None) -> tuple[bool, str]:
    reasons: list[str] = []
    if _truthy(row.get("nar_narrative_hard_block")):
        reasons.append("OFFICIAL_NARRATIVE_CONTRADICTION")
    for name in ("sig_market_status", "sig_listing_status", "sig_universe_status", "sig_fca_status"):
        text = str(row.get(name, "")).upper()
        if any(token in text for token in ("SUSPEND", "DELIST", "FCA_BLOCK", "REJECT")):
            reasons.append(text or name.upper())
    if _truthy(row.get("fund_severe_fundamental_flags")):
        reasons.append("SEVERE_FUNDAMENTAL_FLAGS")
    # Sharia eligibility is a universe/user policy, not a universal market-risk
    # hard block.  A generic all-IDX scan must not silently reject non-sharia
    # issuers unless the caller explicitly requests a sharia-only universe.
    if bool(getattr(cfg, "sharia_only", False)):
        sharia = row.get("fund_is_sharia", row.get("sig_is_sharia", np.nan))
        if sharia is not np.nan and str(sharia).strip() and not _truthy(sharia):
            if str(sharia).strip().upper() in {"0", "FALSE", "NO", "N"}:
                reasons.append("NOT_SHARIA")
    return bool(reasons), " | ".join(dict.fromkeys(reasons))


def _candidate_type(row: Mapping[str, Any], future_score: float, business_score: float = np.nan) -> str:
    sector = canonical_sector(_first(row, ("fund_sector", "mac_sector"), ""))
    revenue_growth = _fraction(_first_num(row, ("fund_revenue_growth",)))
    earnings_growth = _fraction(_first_num(row, ("fund_earnings_growth",)))
    roe = _fraction(_first_num(row, ("fund_roe",)))
    debt_equity = _first_num(row, ("fund_debt_equity", "fund_history_debt_equity"))
    cash_conversion = _first_num(row, ("fund_history_cash_conversion", "fund_cash_conversion_ttm"))
    project = _first_num(row, DIRECT_FORWARD_SCORE_FIELDS)
    if np.isfinite(project) and project >= 65:
        return "CAPACITY_EXPANSION"
    if sector in {"ENERGY", "BASIC MATERIALS"} and np.isfinite(future_score) and future_score >= 50:
        return "CYCLICAL_RECOVERY"
    if np.isfinite(earnings_growth) and earnings_growth > 0.20 and (not np.isfinite(revenue_growth) or revenue_growth < 0.10):
        return "TURNAROUND"
    quality_ok = np.isfinite(business_score) and business_score >= 65.0
    leverage_ok = not np.isfinite(debt_equity) or debt_equity <= 1.20
    cash_ok = not np.isfinite(cash_conversion) or cash_conversion >= 0.60
    if (
        np.isfinite(revenue_growth) and revenue_growth >= 0.12
        and np.isfinite(roe) and roe >= 0.12
        and quality_ok and leverage_ok and cash_ok
    ):
        return "TRUE_COMPOUNDER"
    if np.isfinite(revenue_growth) and revenue_growth >= 0.12 and np.isfinite(business_score) and business_score >= 50:
        return "EMERGING_GROWTH"
    if np.isfinite(project) and np.isfinite(future_score) and future_score >= 65:
        return "EVENT_DRIVEN_RERATING"
    if np.isfinite(future_score) and future_score >= 65:
        return "FINANCIAL_CAPACITY"
    return "GROWTH_VALUE_CANDIDATE"


def _fundamental_freshness_state(row: Mapping[str, Any]) -> str:
    explicit = str(_first(row, ("fund_fundamental_freshness_state",), "")).strip().upper()
    cadence = reporting_refresh_profile(row)
    cadence_state = str(cadence.get("fundamental_refresh_state", "")).upper()
    # Calendar-aware cadence outranks an old age-only CURRENT label.  A refresh
    # window does not remove the stock from research ranking; it only forces
    # bounded enrichment and prevents real-money authorization from treating
    # the period as fully current.
    if cadence_state in {"MISSING_DATE", "STALE", "REFRESH_WINDOW"}:
        return cadence_state
    if explicit:
        return explicit
    age = _first_num(row, ("fund_statement_age_days",))
    if not np.isfinite(age):
        return "MISSING_DATE"
    if age <= 210:
        return "CURRENT"
    if age <= NEXT_LEADER_MAX_STATEMENT_AGE_DAYS:
        return "ACCEPTABLE_STALE"
    return "STALE"


def _leader_production_gate(
    row: Mapping[str, Any],
    *,
    coverage: float,
    business: tuple[float, float, str],
    future: tuple[float, float, str],
    macro: tuple[float, float, str],
    candidate_type: str,
) -> tuple[bool, str]:
    reasons: list[str] = []
    sector = canonical_sector(_first(row, ("fund_sector", "mac_sector"), "UNKNOWN"))
    freshness = _fundamental_freshness_state(row)
    if coverage < NEXT_LEADER_MIN_COVERAGE_PCT:
        reasons.append(f"TOTAL_COVERAGE<{NEXT_LEADER_MIN_COVERAGE_PCT:.0f}")
    if business[1] < NEXT_LEADER_MIN_BUSINESS_COVERAGE_PCT:
        reasons.append(f"BUSINESS_COVERAGE<{NEXT_LEADER_MIN_BUSINESS_COVERAGE_PCT:.0f}")
    if future[1] < NEXT_LEADER_MIN_FUTURE_COVERAGE_PCT:
        reasons.append(f"FUTURE_COVERAGE<{NEXT_LEADER_MIN_FUTURE_COVERAGE_PCT:.0f}")
    if macro[1] < 50.0 or sector == "UNKNOWN":
        reasons.append("SECTOR_MACRO_UNRESOLVED")
    if freshness in {"STALE", "MISSING_DATE"}:
        reasons.append(f"FUNDAMENTAL_{freshness}")
    business_score = business[0]
    future_score = future[0]
    recovery_lane = candidate_type in {"TURNAROUND", "CYCLICAL_RECOVERY", "CAPACITY_EXPANSION"}
    project_score = _first_num(row, DIRECT_FORWARD_SCORE_FIELDS)
    if candidate_type in {"CAPACITY_EXPANSION", "EVENT_DRIVEN_RERATING"} and not np.isfinite(project_score):
        reasons.append("DIRECT_PROJECT_EVIDENCE_REQUIRED")
    if recovery_lane:
        if not (np.isfinite(business_score) and business_score >= 32.0 and np.isfinite(future_score) and future_score >= 55.0):
            reasons.append("RECOVERY_QUALITY_FLOOR")
    else:
        if not (np.isfinite(business_score) and business_score >= 45.0):
            reasons.append("BUSINESS_QUALITY_FLOOR")
        if not (np.isfinite(future_score) and future_score >= 40.0):
            reasons.append("FUTURE_FUNDAMENTAL_FLOOR")
    if candidate_type == "TRUE_COMPOUNDER" and (not np.isfinite(business_score) or business_score < 65.0):
        reasons.append("COMPOUNDER_QUALITY_FLOOR")
    return not reasons, " | ".join(reasons)


def _swing_production_gate(
    *, coverage: float, technical: tuple[float, float, str], macro: tuple[float, float, str]
) -> tuple[bool, str]:
    reasons: list[str] = []
    if coverage < SWING_MIN_COVERAGE_PCT:
        reasons.append(f"TOTAL_COVERAGE<{SWING_MIN_COVERAGE_PCT:.0f}")
    if not np.isfinite(technical[0]) or technical[0] < SWING_MIN_TECHNICAL_SCORE:
        reasons.append(f"TECHNICAL<{SWING_MIN_TECHNICAL_SCORE:.0f}")
    if technical[1] < 50.0:
        reasons.append("TECHNICAL_COVERAGE<50")
    if macro[1] < 50.0:
        reasons.append("MACRO_COVERAGE<50")
    return not reasons, " | ".join(reasons)


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
        coverage_gap = _coverage_gap_profile(components, weights)
        research_score, coverage = _weighted_final(components, weights, min_coverage=35.0)
        final_score, coverage = _weighted_final(components, weights, min_coverage=NEXT_LEADER_MIN_COVERAGE_PCT)
        fundamental_guard = fundamental_conviction_profile(row)
        pre_cap_score = final_score
        conviction_cap = _finite(fundamental_guard.get("fundamental_conviction_cap"), np.nan)
        if np.isfinite(final_score) and np.isfinite(conviction_cap):
            final_score = min(final_score, conviction_cap)
        fundamental_guard["fundamental_score_pre_cap"] = round(pre_cap_score, 1) if np.isfinite(pre_cap_score) else np.nan
        blocked, blocked_reason = _hard_block(row, cfg)
        if blocked and np.isfinite(final_score):
            final_score = min(final_score, 30.0)
        # Ranking thesis is intentionally separate from production eligibility.
        # When production coverage is not yet 70%, keep the lower-coverage
        # research score visible so ALL_ELIGIBLE can still rank the universe.
        # Real-money authorization remains fail-closed below.
        ranking_score = final_score if np.isfinite(final_score) else research_score
        if np.isfinite(ranking_score) and np.isfinite(conviction_cap):
            ranking_score = min(ranking_score, conviction_cap)
        ranking_score_state = "PRODUCTION_SCORE" if np.isfinite(final_score) else "RESEARCH_SCORE"

        technical_score = technical[0]
        entry = _first_num(row, ("sig_entry", "sig_entry_low"))
        rr1 = _first_num(row, ("sig_rr1",))
        adtv = _first_num(row, ("sig_adtv20_idr", "flow_adtv20_idr"))
        liquidity_ok = not np.isfinite(adtv) or adtv >= 250_000_000.0
        candidate_type = _candidate_type(row, future[0], business[0])
        freshness_profile = reporting_refresh_profile(row)
        growth_profile = latest_growth_profile(row)
        thesis_archetype = classify_thesis_archetype(row, business_score=business[0], future_score=future[0])
        production_gate_pass, production_gate_reason = _leader_production_gate(
            row, coverage=coverage, business=business, future=future, macro=macro, candidate_type=candidate_type
        )
        if blocked:
            status = "REJECT"
        elif not np.isfinite(ranking_score):
            status = "DATA_PENDING"
        elif not production_gate_pass:
            status = "RESEARCH_ONLY"
        elif final_score >= 75 and np.isfinite(technical_score) and technical_score >= 68 and liquidity_ok and (not np.isfinite(rr1) or rr1 >= 1.5):
            status = "BUY_ZONE"
        elif final_score >= 66:
            status = "WATCH"
        elif final_score >= 56:
            status = "WAIT"
        else:
            status = "RESEARCH_ONLY"
        if thesis_archetype == "FUNDAMENTAL_DETERIORATION":
            lane = "DETERIORATION_RESEARCH"
        elif thesis_archetype in {"TURNAROUND_RECOVERY", "BASE_EFFECT_REVIEW", "CYCLICAL_RECOVERY"}:
            lane = "TURNAROUND_CYCLICAL"
        else:
            lane = "GROWTH_COMPOUNDER"

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
        if thesis_archetype == "FUNDAMENTAL_DETERIORATION":
            primary_risk = "Latest-period revenue and earnings are both deteriorating | " + primary_risk
        elif str(freshness_profile.get("fundamental_refresh_state", "")).upper() == "REFRESH_WINDOW":
            primary_risk = "Latest quarterly reporting window should be refreshed | " + primary_risk
        research_plan = _research_accumulation_plan(row)
        rows.append({
            "ticker": row["ticker"],
            "sector": _first(row, ("fund_sector", "mac_sector"), "UNKNOWN"),
            "candidate_type": candidate_type,
            "thesis_archetype": thesis_archetype,
            "fundamental_trend_state": growth_profile.get("fundamental_trend_state"),
            "fundamental_growth_basis_state": growth_profile.get("fundamental_growth_basis_state"),
            "fundamental_growth_conflict_state": growth_profile.get("fundamental_growth_conflict_state"),
            "fundamental_latest_revenue_growth": growth_profile.get("fundamental_latest_revenue_growth"),
            "fundamental_latest_earnings_growth": growth_profile.get("fundamental_latest_earnings_growth"),
            "fundamental_extreme_earnings_base_review": growth_profile.get("fundamental_extreme_earnings_base_review"),
            "fundamental_refresh_state": freshness_profile.get("fundamental_refresh_state"),
            "fundamental_refresh_due": freshness_profile.get("fundamental_refresh_due"),
            "fundamental_latest_period": freshness_profile.get("fundamental_latest_period"),
            "fundamental_refresh_open_at": freshness_profile.get("fundamental_refresh_open_at"),
            "multibagger_lane": lane,
            "status": status,
            "v9_next_leader_score": round(final_score, 1) if np.isfinite(final_score) else np.nan,
            "research_score": round(research_score, 1) if np.isfinite(research_score) else np.nan,
            "ranking_score": round(ranking_score, 1) if np.isfinite(ranking_score) else np.nan,
            "ranking_score_state": ranking_score_state,
            "score_coverage_pct": round(coverage, 1),
            **coverage_gap,
            "production_gate_pass": bool(production_gate_pass),
            "production_gate_reason": production_gate_reason,
            "fundamental_freshness_state": _fundamental_freshness_state(row),
            "sector_source": _first(row, ("fund_sector_source",), "UNKNOWN"),
            "sector_confidence_pct": _first_num(row, ("fund_sector_confidence_pct",)),
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
            "silent_accumulation_confidence": _first_num(row, ("flow_silent_accumulation_confidence", "flow_silent_accumulation_data_coverage")),
            "accumulation_dominance_pct": _first_num(row, ("flow_accumulation_dominance_pct",)),
            "inventory_multi_horizon_score": _first_num(row, ("flow_inventory_multi_horizon_score",)),
            "inventory_multi_horizon_coverage_pct": _first_num(row, ("flow_inventory_multi_horizon_coverage_pct",)),
            "distribution_risk_score": _first_num(row, ("flow_distribution_risk_score",)),
            "inventory_lifecycle": _first(row, ("flow_inventory_lifecycle",), "UNKNOWN"),
            "anti_chase_gate": _truthy(_first(row, ("flow_anti_chase_gate",), False)),
            "markup_extension_pct": _first_num(row, ("flow_markup_extension_pct",)),
            "reaccumulation_quality_score": _first_num(row, ("flow_reaccumulation_quality_score",)),
            **fundamental_guard,
            "market_regime": _first(row, ("mac_market_regime", "mac_macro_regime"), "DATA_PENDING"),
            "market_context_score": _first_num(row, ("mac_market_context_score", "mac_macro_regime_score")),
            "market_context_coverage_pct": _first_num(row, ("mac_market_context_coverage_pct", "mac_macro_data_coverage_pct")),
            "market_context_provenance_state": _first(row, ("mac_market_context_provenance_state",), "UNKNOWN"),
            "independent_price_verified": _truthy(_first(row, ("sig_independent_price_verified",), False)),
            "independent_price_state": _first(row, ("sig_independent_price_state",), "MISSING_INDEPENDENT"),
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
            **research_plan,
            "execution_plan_state": "EXECUTABLE_PLAN_AVAILABLE" if np.isfinite(entry) and np.isfinite(_first_num(row, ("sig_stop_loss",))) else "WAIT_TECHNICAL_CONFIRMATION",
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
            "multibagger_scoring_state": "PRODUCTION_SCORED" if np.isfinite(final_score) else "RESEARCH_SCORED" if np.isfinite(ranking_score) else "DATA_PENDING",
            "multibagger_evidence_class": "MACRO_JOS_EMIR",
            "multibagger_metric_coverage_pct": round(coverage, 1),
            "multibagger_metric_data_gate": "PASS" if coverage >= NEXT_LEADER_MIN_COVERAGE_PCT else "PENDING",
            "execution_readiness_score": round(technical[0], 1) if np.isfinite(technical[0]) else np.nan,
            "multibagger_candidate_type": candidate_type,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = apply_methodology_guardrails(out, model="NEXT_LEADER")
    out = _apply_decision_priority_guardrail(out)
    out = apply_real_money_authorization(out, model="NEXT_LEADER", account_size_idr=_finite(getattr(cfg, "account_size_idr", np.nan), np.nan), requested_risk_budget_pct=100.0 * _finite(getattr(cfg, "risk_per_trade_pct", 0.005), 0.005))
    # Research, portfolio suitability and executable-order authorization are separate contracts.
    # Incomplete evidence may rank, but can never become production/order-ready
    # until the original production + methodology gates pass.
    out["production_rank_eligible"] = (
        pd.to_numeric(out["v9_next_leader_score"], errors="coerce").notna()
        & out["status"].isin(["BUY_ZONE", "WATCH", "WAIT"])
        & out["production_gate_pass"].fillna(False).astype(bool)
        & out["methodology_gate_pass"].fillna(False).astype(bool)
    )
    out["rank_eligible"] = (
        pd.to_numeric(out["ranking_score"], errors="coerce").notna()
        & ~out["status"].isin(["REJECT", "DATA_PENDING"])
        & ~out.get("hard_block", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    )
    adtv_series = pd.to_numeric(out.get("adtv20_idr", pd.Series(np.nan, index=out.index)), errors="coerce")
    score_series = pd.to_numeric(out.get("v9_next_leader_score", pd.Series(np.nan, index=out.index)), errors="coerce")
    out["portfolio_rank_eligible"] = (
        out["production_rank_eligible"]
        & score_series.ge(66.0)
        & (adtv_series.isna() | adtv_series.ge(250_000_000.0))
    )
    out["research_gate_state"] = np.where(out["rank_eligible"], "PASS", "BLOCKED")
    official_verified_series = out.get("fundamental_official_verified", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    data_q_series = pd.to_numeric(out.get("fundamental_data_quality_score", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    portfolio_full = out["portfolio_rank_eligible"] & official_verified_series & data_q_series.ge(70.0)
    out["portfolio_gate_state"] = np.where(portfolio_full, "PASS", np.where(out["portfolio_rank_eligible"] | out["rank_eligible"], "WATCH", "BLOCKED"))
    out["execution_gate_state"] = np.where(out.get("real_money_authorization_pass", pd.Series(False, index=out.index)).fillna(False).astype(bool), "PASS", "BLOCKED")
    official_cov_series = pd.to_numeric(out.get("fundamental_official_source_coverage_pct", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    cov_series = pd.to_numeric(out.get("score_coverage_pct", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    out["thesis_confidence_pct"] = (0.55 * cov_series + 0.35 * data_q_series + 0.10 * official_cov_series).clip(0, 100)
    not_official = ~official_verified_series
    out.loc[not_official, "thesis_confidence_pct"] = out.loc[not_official, "thesis_confidence_pct"].clip(upper=70.0)
    out["multibagger_rank_eligible"] = out["production_rank_eligible"]
    out = out.sort_values(
        ["rank_eligible", "ranking_score", "methodology_priority", "score_coverage_pct", "ticker"],
        ascending=[False, False, True, False, True], na_position="last", kind="stable",
    ).reset_index(drop=True)
    out["rank"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    eligible_index = out.index[out["rank_eligible"]]
    out.loc[eligible_index, "rank"] = np.arange(1, len(eligible_index) + 1)
    out["production_rank"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    production_index = out.index[out["production_rank_eligible"]]
    out.loc[production_index, "production_rank"] = np.arange(1, len(production_index) + 1)
    out["portfolio_rank"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    portfolio_order = out.loc[out["portfolio_rank_eligible"]].sort_values(
        ["v9_next_leader_score", "score_coverage_pct", "ticker"], ascending=[False, False, True], kind="stable"
    ).index
    out.loc[portfolio_order, "portfolio_rank"] = np.arange(1, len(portfolio_order) + 1)
    out["multibagger_selection_rank"] = out["rank"]
    out["multibagger_production_rank"] = out["production_rank"]
    return out


def build_swing_ready(universe: pd.DataFrame, next_leaders: pd.DataFrame, config: ScanConfig | None = None) -> pd.DataFrame:
    cfg = config or ScanConfig()
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
        fundamental_guard = fundamental_conviction_profile(row)
        components = {
            "technical_execution": technical[:2],
            "macro_sector": macro[:2],
            "narrative_flow": narrative_flow[:2],
            "business_quality": business[:2],
            "risk_data": risk_data[:2],
        }
        research_score, coverage = _weighted_final(components, weights, min_coverage=35.0)
        final_score, coverage = _weighted_final(components, weights, min_coverage=SWING_MIN_COVERAGE_PCT)
        ranking_score = final_score if np.isfinite(final_score) else research_score
        ranking_score_state = "PRODUCTION_SCORE" if np.isfinite(final_score) else "RESEARCH_SCORE"
        blocked, blocked_reason = _hard_block(row, cfg)
        if blocked and np.isfinite(final_score):
            final_score = min(final_score, 25.0)
        swing_gate_pass, swing_gate_reason = _swing_production_gate(coverage=coverage, technical=technical, macro=macro)
        setup_state = str(_first(row, ("sig_setup_status", "sig_status", "sig_decision_state"), "WATCHLIST")).upper()
        entry = _first_num(row, ("sig_entry", "sig_entry_low"))
        stop = _first_num(row, ("sig_stop_loss",))
        tp1 = _first_num(row, ("sig_tp1",))
        trigger = _first_num(row, ("sig_trigger_price", "sig_trigger"))
        rr1 = _first_num(row, ("sig_rr1",))
        execution_entry = trigger if np.isfinite(trigger) and trigger > 0 else entry
        zone_role = _entry_zone_role(row, execution_entry, stop, trigger)
        atomic_plan = all(np.isfinite(v) and v > 0 for v in (execution_entry, stop, tp1)) and stop < execution_entry < tp1
        if blocked:
            status = "REJECT"
        elif not np.isfinite(ranking_score):
            status = "DATA_PENDING"
        elif not swing_gate_pass:
            status = "RESEARCH_ONLY"
        elif final_score >= 74 and atomic_plan and (not np.isfinite(rr1) or rr1 >= 1.5):
            status = "EXECUTION_READY"
        elif final_score >= 65 and atomic_plan:
            status = "ENTRY_PLAN_READY"
        elif final_score >= 58:
            status = "WATCHLIST"
        else:
            status = "WAIT"
        leader = leader_lookup.get(row["ticker"], {})
        rows.append({
            "ticker": row["ticker"],
            "sector": _first(row, ("fund_sector", "mac_sector"), "UNKNOWN"),
            "status": status,
            "v9_swing_score": round(final_score, 1) if np.isfinite(final_score) else np.nan,
            "research_score": round(research_score, 1) if np.isfinite(research_score) else np.nan,
            "ranking_score": round(ranking_score, 1) if np.isfinite(ranking_score) else np.nan,
            "ranking_score_state": ranking_score_state,
            "score_coverage_pct": round(coverage, 1),
            "production_gate_pass": bool(swing_gate_pass),
            "production_gate_reason": swing_gate_reason,
            "technical_execution_score": round(technical[0], 1) if np.isfinite(technical[0]) else np.nan,
            "issuer_macro_alignment_score": round(macro[0], 1) if np.isfinite(macro[0]) else np.nan,
            "narrative_flow_score": round(narrative_flow[0], 1) if np.isfinite(narrative_flow[0]) else np.nan,
            "silent_accumulation_score": _first_num(row, ("flow_silent_accumulation_score", "sig_silent_accumulation_score")),
            "silent_accumulation_confidence": _first_num(row, ("flow_silent_accumulation_confidence", "flow_silent_accumulation_data_coverage")),
            "accumulation_dominance_pct": _first_num(row, ("flow_accumulation_dominance_pct",)),
            "inventory_multi_horizon_score": _first_num(row, ("flow_inventory_multi_horizon_score",)),
            "inventory_multi_horizon_coverage_pct": _first_num(row, ("flow_inventory_multi_horizon_coverage_pct",)),
            "distribution_risk_score": _first_num(row, ("flow_distribution_risk_score",)),
            "inventory_lifecycle": _first(row, ("flow_inventory_lifecycle",), "UNKNOWN"),
            "anti_chase_gate": _truthy(_first(row, ("flow_anti_chase_gate",), False)),
            "markup_extension_pct": _first_num(row, ("flow_markup_extension_pct",)),
            "reaccumulation_quality_score": _first_num(row, ("flow_reaccumulation_quality_score",)),
            **fundamental_guard,
            "market_regime": _first(row, ("mac_market_regime", "mac_macro_regime"), "DATA_PENDING"),
            "market_context_score": _first_num(row, ("mac_market_context_score", "mac_macro_regime_score")),
            "market_context_coverage_pct": _first_num(row, ("mac_market_context_coverage_pct", "mac_macro_data_coverage_pct")),
            "market_context_provenance_state": _first(row, ("mac_market_context_provenance_state",), "UNKNOWN"),
            "independent_price_verified": _truthy(_first(row, ("sig_independent_price_verified",), False)),
            "independent_price_state": _first(row, ("sig_independent_price_state",), "MISSING_INDEPENDENT"),
            "business_quality_score": round(business[0], 1) if np.isfinite(business[0]) else np.nan,
            "risk_data_score": round(risk_data[0], 1) if np.isfinite(risk_data[0]) else np.nan,
            "next_leader_score": leader.get("v9_next_leader_score", leader.get("ranking_score", np.nan)),
            "setup_status": setup_state,
            "strategy": _first(row, ("sig_strategy", "sig_setup"), "CORE_SWING"),
            "last_price": _first_num(row, ("sig_last_price", "sig_close", "fund_last_price")),
            "entry": entry,
            "execution_entry": execution_entry,
            "entry_low": _first_num(row, ("sig_entry_low",)),
            "entry_high": _first_num(row, ("sig_entry_high",)),
            "entry_zone_role": zone_role,
            "entry_zone_is_executable": zone_role == "EXECUTABLE_ENTRY_ZONE",
            "trigger_price": trigger,
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
    out = apply_methodology_guardrails(out, model="SWING_READY")
    out = apply_real_money_authorization(out, model="SWING_READY", account_size_idr=_finite(getattr(cfg, "account_size_idr", np.nan), np.nan), requested_risk_budget_pct=100.0 * _finite(getattr(cfg, "risk_per_trade_pct", 0.005), 0.005))
    out["production_rank_eligible"] = (
        pd.to_numeric(out["v9_swing_score"], errors="coerce").notna()
        & out["status"].isin(["EXECUTION_READY", "ENTRY_PLAN_READY", "WATCHLIST", "WAIT"])
        & out["production_gate_pass"].fillna(False).astype(bool)
        & out["methodology_gate_pass"].fillna(False).astype(bool)
        & out.get("execution_plan_is_current", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    )
    out["rank_eligible"] = (
        pd.to_numeric(out["ranking_score"], errors="coerce").notna()
        & ~out["status"].isin(["REJECT", "DATA_PENDING"])
        & ~out.get("hard_block", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    )
    out["actionable_rank_eligible"] = (
        out["production_rank_eligible"]
        & out["status"].isin(["EXECUTION_READY", "ENTRY_PLAN_READY"])
        & ~out.get("anti_chase_gate", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    )
    out["research_gate_state"] = np.where(out["rank_eligible"], "PASS", "BLOCKED")
    out["portfolio_gate_state"] = np.where(out["production_rank_eligible"], "PASS", np.where(out["rank_eligible"], "WATCH", "BLOCKED"))
    out["execution_gate_state"] = np.where(out.get("real_money_authorization_pass", pd.Series(False, index=out.index)).fillna(False).astype(bool), "PASS", "BLOCKED")
    out = out.sort_values(
        ["rank_eligible", "ranking_score", "methodology_priority", "score_coverage_pct", "ticker"],
        ascending=[False, False, True, False, True], na_position="last", kind="stable",
    ).reset_index(drop=True)
    out["rank"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    eligible_index = out.index[out["rank_eligible"]]
    out.loc[eligible_index, "rank"] = np.arange(1, len(eligible_index) + 1)
    out["production_rank"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    production_index = out.index[out["production_rank_eligible"]]
    out.loc[production_index, "production_rank"] = np.arange(1, len(production_index) + 1)
    out["actionable_rank"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    actionable_order = out.loc[out["actionable_rank_eligible"]].sort_values(
        ["v9_swing_score", "technical_execution_score", "score_coverage_pct", "ticker"],
        ascending=[False, False, False, True], kind="stable"
    ).index
    out.loc[actionable_order, "actionable_rank"] = np.arange(1, len(actionable_order) + 1)
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
    swing_ready_all = build_swing_ready(universe, next_leaders_all, cfg)
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
