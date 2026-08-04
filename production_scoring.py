from __future__ import annotations

"""Production scoring for IDX Super Scanner v8.

The module owns every score that is allowed to affect production ranking.
Each information family appears exactly once in the top-level formula:

Multibagger
    55% fundamental/future-fundamental
    25% sourced narrative + issuer actions (excludes fundamentals and flow)
    10% market/sector relative strength
    10% silent accumulation / observed flow

Core Swing
    45% technical structure, timing and target geometry
    15% market/sector relative strength and liquidity
    20% sourced narrative + issuer actions (excludes fundamentals and flow)
    15% observed flow
     5% data and OOS validation

AI, EOFF, lunar/planetary factors, Best Buy Date and legacy additive
narrative/Emir overlays have zero production weight.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


PRODUCTION_SCORING_VERSION = "8.0.0-slim-no-double-counting"


@dataclass(frozen=True)
class V8ProductionWeights:
    multibagger_fundamental: float = 0.55
    multibagger_narrative_alignment: float = 0.25
    multibagger_market_sector: float = 0.10
    multibagger_flow: float = 0.10

    core_technical: float = 0.45
    core_market_sector: float = 0.15
    core_narrative_alignment: float = 0.20
    core_flow: float = 0.15
    core_data_validation: float = 0.05

    def validate(self) -> None:
        mb = (
            self.multibagger_fundamental
            + self.multibagger_narrative_alignment
            + self.multibagger_market_sector
            + self.multibagger_flow
        )
        core = (
            self.core_technical
            + self.core_market_sector
            + self.core_narrative_alignment
            + self.core_flow
            + self.core_data_validation
        )
        if not np.isclose(mb, 1.0):
            raise ValueError(f"Multibagger weights must total 1.0, got {mb}")
        if not np.isclose(core, 1.0):
            raise ValueError(f"Core weights must total 1.0, got {core}")


DEFAULT_WEIGHTS = V8ProductionWeights()
DEFAULT_WEIGHTS.validate()


MULTIBAGGER_COMPONENT_SOURCES = {
    "fundamental_future": frozenset({
        "growth_persistence_pillar",
        "fundamental_inflection_score",
        "profitability_pillar",
        "cash_conversion_pillar",
        "balance_sheet_safety_pillar",
        "reinvestment_runway_pillar",
        "valuation_score",
    }),
    "narrative_alignment": frozenset({
        "narrative_event_effective_score",
        "issuer_action_alignment_effective_score",
    }),
    "market_sector": frozenset({
        "growth_sector_relative_score",
        "turnaround_sector_relative_score",
        "momentum_score",
    }),
    "flow": frozenset({
        "effective_silent_accumulation_score",
        "silent_accumulation_score",
    }),
}

CORE_COMPONENT_SOURCES = {
    "technical": frozenset({
        "v8_structure_pure_score",
        "timing_score",
        "target_quality_score",
    }),
    "market_sector": frozenset({
        "selector_trend_score",
        "selector_momentum_score",
        "selector_relative_strength_score",
        "sector_relative_strength_score",
        "liquidity_score",
    }),
    "narrative_alignment": frozenset({
        "narrative_event_effective_score",
        "issuer_action_alignment_effective_score",
    }),
    "flow": frozenset({
        "effective_silent_accumulation_score",
    }),
    "data_validation": frozenset({
        "data_quality_score",
        "validation_score",
    }),
}


def validate_disjoint_component_sources() -> None:
    """Fail at import if a raw score is assigned to two production pillars."""
    for model_name, components in (
        ("multibagger", MULTIBAGGER_COMPONENT_SOURCES),
        ("core", CORE_COMPONENT_SOURCES),
    ):
        seen: dict[str, str] = {}
        for component, fields in components.items():
            for field in fields:
                previous = seen.get(field)
                if previous is not None:
                    raise ValueError(
                        f"{model_name}: {field} belongs to both "
                        f"{previous} and {component}"
                    )
                seen[field] = component


validate_disjoint_component_sources()


MULTIBAGGER_DECISION_COLUMNS = (
    "ticker", "multibagger_selection_rank", "multibagger_production_rank",
    "growth_compounder_rank", "turnaround_rank", "multibagger_lane",
    "multibagger_status", "research_recommendation_status",
    "multibagger_scoring_state", "research_eligible",
    "multibagger_rank_eligible", "multibagger_evidence_class",
    "v8_strategic_score", "final_score",
    "v8_production_score_coverage_pct", "v8_fundamental_future_score",
    "v8_narrative_alignment_score", "v8_market_sector_score",
    "v8_flow_score", "growth_compounder_selection_score",
    "turnaround_selection_score", "fundamental_data_grade",
    "fundamental_overall_coverage", "fundamental_history_coverage",
    "fundamental_official_verified", "fundamental_as_of",
    "growth_persistence_pillar", "fundamental_inflection_score",
    "profitability_pillar", "cash_conversion_pillar",
    "balance_sheet_safety_pillar", "reinvestment_runway_pillar",
    "valuation_score", "narrative_event_effective_score",
    "narrative_event_coverage_pct",
    "issuer_action_alignment_effective_score",
    "issuer_action_alignment_coverage_pct",
    "effective_silent_accumulation_score",
    "silent_accumulation_confidence", "growth_sector_relative_score",
    "turnaround_sector_relative_score", "momentum_score", "last_price",
    "technical_entry_state", "entry_low", "entry_high", "entry",
    "trigger", "stop_loss", "tp1", "tp2", "rr1", "rr2",
    "compounding_state", "allocation_eligible", "capital_tier",
    "capital_conviction_score", "capital_priority_rank",
    "strategic_target_weight_pct", "deploy_now_weight_pct",
    "allocation_action", "recommended_allocation_idr",
    "recommended_lots", "selected_reason", "primary_risk",
    "not_entry_reason", "trigger_waiting", "invalidation_reason",
    "multibagger_final_score_formula", "production_scoring_version",
    "production_ai_weight_pct", "production_eoff_weight_pct",
    "production_legacy_overlay_weight_pct",
)

CORE_DECISION_COLUMNS = (
    "ticker", "profit_rank", "strategy_rank", "strategy", "horizon",
    "decision_state", "setup_status", "order_builder_eligible",
    "core_priority_score", "final_score", "profit_conviction_score",
    "v8_production_score_coverage_pct", "v8_technical_score",
    "v8_market_sector_score", "v8_narrative_alignment_score",
    "v8_flow_score", "v8_data_validation_score",
    "v8_structure_pure_score", "timing_score", "target_quality_score",
    "selector_trend_score", "selector_momentum_score",
    "selector_relative_strength_score", "sector_relative_strength_score",
    "liquidity_score", "narrative_event_effective_score",
    "narrative_event_coverage_pct",
    "issuer_action_alignment_effective_score",
    "issuer_action_alignment_coverage_pct",
    "effective_silent_accumulation_score",
    "silent_accumulation_confidence", "entry_low", "entry_high", "entry",
    "entry_type", "action", "trigger", "trigger_price", "stop_loss",
    "tp1", "tp2", "rr1", "rr2", "order_ready",
    "stockbit_order_lots", "next_action", "selected_reason",
    "primary_risk", "not_entry_reason", "trigger_waiting",
    "invalidation_reason", "warnings", "core_final_score_formula",
    "production_scoring_version", "production_ai_weight_pct",
    "production_eoff_weight_pct", "production_legacy_overlay_weight_pct",
)


def slim_production_frame(
    frame: pd.DataFrame | None,
    *,
    model: str,
) -> pd.DataFrame:
    """Return the bounded decision schema used by Streamlit and exports."""
    clean = drop_nonproduction_columns(frame)
    columns = (
        MULTIBAGGER_DECISION_COLUMNS
        if model.strip().lower() == "multibagger"
        else CORE_DECISION_COLUMNS
        if model.strip().lower() == "core"
        else ()
    )
    if not columns:
        raise ValueError(f"Unknown production model: {model}")
    selected = [column for column in columns if column in clean.columns]
    out = clean.loc[:, selected].copy()
    out.attrs = {}
    return out


def build_production_evidence_detail(
    multibagger: pd.DataFrame | None = None,
    core: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return one auditable row per ticker and top-level production pillar."""
    rows: list[dict[str, object]] = []
    specs = (
        (
            "MULTIBAGGER",
            multibagger,
            (
                ("FUNDAMENTAL_FUTURE", "v8_fundamental_future_score", 55.0),
                ("NARRATIVE_ISSUER_ACTION", "v8_narrative_alignment_score", 25.0),
                ("MARKET_SECTOR", "v8_market_sector_score", 10.0),
                ("SILENT_ACCUMULATION", "v8_flow_score", 10.0),
            ),
        ),
        (
            "CORE_SWING",
            core,
            (
                ("TECHNICAL_EXECUTION", "v8_technical_score", 45.0),
                ("MARKET_SECTOR", "v8_market_sector_score", 15.0),
                ("NARRATIVE_ISSUER_ACTION", "v8_narrative_alignment_score", 20.0),
                ("SILENT_ACCUMULATION", "v8_flow_score", 15.0),
                ("DATA_OOS", "v8_data_validation_score", 5.0),
            ),
        ),
    )
    for model, frame, components in specs:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        for _, row in frame.iterrows():
            for component, score_column, weight in components:
                value = pd.to_numeric(
                    pd.Series([row.get(score_column)]), errors="coerce",
                ).iloc[0]
                rows.append({
                    "model": model,
                    "ticker": row.get("ticker", ""),
                    "strategy": row.get("strategy", ""),
                    "component": component,
                    "production_weight_pct": weight,
                    "component_score": value,
                    "total_score": row.get(
                        "v8_strategic_score",
                        row.get("core_priority_score", np.nan),
                    ),
                    "score_coverage_pct": row.get(
                        "v8_production_score_coverage_pct", np.nan,
                    ),
                    "production_scoring_version": row.get(
                        "production_scoring_version",
                        PRODUCTION_SCORING_VERSION,
                    ),
                })
    return pd.DataFrame(rows)


def production_scoring_audit_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, weights, sources in (
        (
            "MULTIBAGGER",
            {
                "FUNDAMENTAL_FUTURE": 55.0,
                "NARRATIVE_ALIGNMENT": 25.0,
                "MARKET_SECTOR": 10.0,
                "FLOW": 10.0,
            },
            MULTIBAGGER_COMPONENT_SOURCES,
        ),
        (
            "CORE_SWING",
            {
                "TECHNICAL": 45.0,
                "MARKET_SECTOR": 15.0,
                "NARRATIVE_ALIGNMENT": 20.0,
                "FLOW": 15.0,
                "DATA_VALIDATION": 5.0,
            },
            CORE_COMPONENT_SOURCES,
        ),
    ):
        for component, weight in weights.items():
            source_key = component.lower()
            rows.append({
                "model": model,
                "component": component,
                "production_weight_pct": weight,
                "raw_sources": " | ".join(sorted(sources[source_key])),
                "source_overlap_count": 0,
                "ai_weight_pct": 0.0,
                "eoff_weight_pct": 0.0,
                "legacy_overlay_weight_pct": 0.0,
                "production_scoring_version": PRODUCTION_SCORING_VERSION,
            })
    return pd.DataFrame(rows)


def _series(frame: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


def _bool_series(frame: pd.DataFrame, name: str, default: bool = False) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=bool)
    values = frame[name]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(default).astype(bool)
    return values.fillna("").astype(str).str.strip().str.upper().isin(
        {"1", "TRUE", "YES", "Y", "ON", "VALID", "VERIFIED"}
    )


def _clip100(values: pd.Series | np.ndarray | float) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").clip(0.0, 100.0)
    return pd.Series(values, dtype=float).clip(0.0, 100.0)


def _weighted_component(
    frame: pd.DataFrame,
    items: Iterable[tuple[str, float, str | None, float]],
) -> tuple[pd.Series, pd.Series]:
    """Return raw score and weighted evidence coverage.

    Each item is ``(score_column, weight, coverage_column, score_scale)``.
    A missing observation contributes neutral 50 to the raw score and zero to
    coverage. Coverage is not used to renormalise the formula; instead the
    final production score is shrunk toward neutral so incomplete evidence
    cannot outrank complete evidence merely because only favourable fields are
    present.
    """

    score_sum = pd.Series(0.0, index=frame.index, dtype=float)
    coverage_sum = pd.Series(0.0, index=frame.index, dtype=float)
    total_weight = 0.0
    for score_col, weight, coverage_col, scale in items:
        if weight <= 0:
            continue
        total_weight += weight
        raw = _series(frame, score_col)
        observed = raw.notna()
        score = _clip100(raw * scale).where(observed, 50.0)
        if coverage_col:
            coverage = _clip100(_series(frame, coverage_col, 0.0)).where(observed, 0.0)
        else:
            coverage = pd.Series(np.where(observed, 100.0, 0.0), index=frame.index)
        score_sum += weight * score
        coverage_sum += weight * coverage
    if total_weight <= 0:
        return (
            pd.Series(50.0, index=frame.index, dtype=float),
            pd.Series(0.0, index=frame.index, dtype=float),
        )
    return score_sum / total_weight, coverage_sum / total_weight


def _coverage_shrink(raw: pd.Series, coverage: pd.Series) -> pd.Series:
    raw = _clip100(raw).fillna(50.0)
    coverage = _clip100(coverage).fillna(0.0)
    return (50.0 + coverage / 100.0 * (raw - 50.0)).clip(0.0, 100.0)


def _narrative_alignment_component(
    frame: pd.DataFrame,
    *,
    swing: bool,
) -> tuple[pd.Series, pd.Series]:
    """Score only sourced story and issuer actions.

    The ``swing`` argument is retained for API stability, but v8 intentionally
    does not use narrative conversion returns, financial bridges, structured
    operating proxies, crowding or flow here. Those inputs belong to other
    production pillars and including them would create double counting.
    """
    del swing
    raw, coverage = _weighted_component(
        frame,
        (
            (
                "narrative_event_effective_score",
                0.65,
                "narrative_event_coverage_pct",
                1.0,
            ),
            (
                "issuer_action_alignment_effective_score",
                0.35,
                "issuer_action_alignment_coverage_pct",
                1.0,
            ),
        ),
    )
    hard_block = _bool_series(frame, "narrative_hard_block")
    raw = raw.where(~hard_block, np.minimum(raw, 20.0))
    coverage = coverage.where(~hard_block, np.maximum(coverage, 60.0))
    return raw, coverage


def _fundamental_component(frame: pd.DataFrame, *, turnaround: bool) -> tuple[pd.Series, pd.Series]:
    valuation_scale = 12.5  # legacy valuation_score is 0..8 points
    if turnaround:
        return _weighted_component(
            frame,
            (
                ("fundamental_inflection_score", 0.28, "fundamental_inflection_coverage_pct", 1.0),
                ("profitability_pillar", 0.14, "profitability_pillar_coverage_pct", 1.0),
                ("cash_conversion_pillar", 0.18, "cashflow_pillar_coverage_pct", 1.0),
                ("balance_sheet_safety_pillar", 0.18, "safety_pillar_coverage_pct", 1.0),
                ("reinvestment_runway_pillar", 0.12, "runway_pillar_coverage_pct", 1.0),
                ("valuation_score", 0.10, "valuation_pillar_coverage_pct", valuation_scale),
            ),
        )
    return _weighted_component(
        frame,
        (
            ("growth_persistence_pillar", 0.22, "growth_pillar_coverage_pct", 1.0),
            ("profitability_pillar", 0.18, "profitability_pillar_coverage_pct", 1.0),
            ("cash_conversion_pillar", 0.18, "cashflow_pillar_coverage_pct", 1.0),
            ("balance_sheet_safety_pillar", 0.16, "safety_pillar_coverage_pct", 1.0),
            ("reinvestment_runway_pillar", 0.16, "runway_pillar_coverage_pct", 1.0),
            ("valuation_score", 0.10, "valuation_pillar_coverage_pct", valuation_scale),
        ),
    )


def _multibagger_market_component(frame: pd.DataFrame, *, turnaround: bool) -> tuple[pd.Series, pd.Series]:
    sector_col = "turnaround_sector_relative_score" if turnaround else "growth_sector_relative_score"
    # Legacy momentum_score is 0..12. Sector-relative score is 0..100.
    return _weighted_component(
        frame,
        (
            (sector_col, 0.55, None, 1.0),
            ("momentum_score", 0.45, None, 100.0 / 12.0),
        ),
    )


def apply_multibagger_production_scoring(
    frame: pd.DataFrame | None,
    *,
    weights: V8ProductionWeights = DEFAULT_WEIGHTS,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    weights.validate()
    out = frame.copy()

    lane = out.get(
        "multibagger_lane",
        pd.Series("GROWTH_COMPOUNDER", index=out.index),
    ).fillna("GROWTH_COMPOUNDER").astype(str).str.upper()
    turnaround = lane.eq("TURNAROUND_CYCLICAL")

    growth_fund_raw, growth_fund_cov = _fundamental_component(out, turnaround=False)
    turn_fund_raw, turn_fund_cov = _fundamental_component(out, turnaround=True)
    fundamental_raw = growth_fund_raw.where(~turnaround, turn_fund_raw)
    fundamental_cov = growth_fund_cov.where(~turnaround, turn_fund_cov)
    fundamental_effective = _coverage_shrink(fundamental_raw, fundamental_cov)

    narrative_raw, narrative_cov = _narrative_alignment_component(out, swing=False)
    narrative_effective = _coverage_shrink(narrative_raw, narrative_cov)

    growth_market_raw, growth_market_cov = _multibagger_market_component(out, turnaround=False)
    turn_market_raw, turn_market_cov = _multibagger_market_component(out, turnaround=True)
    market_raw = growth_market_raw.where(~turnaround, turn_market_raw)
    market_cov = growth_market_cov.where(~turnaround, turn_market_cov)
    market_effective = _coverage_shrink(market_raw, market_cov)

    flow_raw = _series(out, "effective_silent_accumulation_score")
    fallback_flow = _series(out, "silent_accumulation_score")
    flow_raw = flow_raw.where(flow_raw.notna(), fallback_flow)
    flow_cov = _series(out, "silent_accumulation_confidence", 0.0).fillna(0.0).clip(0.0, 100.0)
    flow_effective = _coverage_shrink(flow_raw, flow_cov)

    raw_score = (
        weights.multibagger_fundamental * fundamental_effective
        + weights.multibagger_narrative_alignment * narrative_effective
        + weights.multibagger_market_sector * market_effective
        + weights.multibagger_flow * flow_effective
    )
    total_coverage = (
        weights.multibagger_fundamental * fundamental_cov
        + weights.multibagger_narrative_alignment * narrative_cov
        + weights.multibagger_market_sector * market_cov
        + weights.multibagger_flow * flow_cov
    ).clip(0.0, 100.0)
    final_score = _coverage_shrink(raw_score, total_coverage)
    hard_block = _bool_series(out, "narrative_hard_block")
    final_score = final_score.where(~hard_block, np.minimum(final_score, 35.0))
    scoring_state = out.get(
        "multibagger_scoring_state",
        pd.Series("", index=out.index),
    ).fillna("").astype(str).str.upper()
    not_scored = scoring_state.str.startswith("DATA_NOT_SCORED")
    insufficient = total_coverage.lt(50.0)
    final_score = final_score.mask(not_scored | insufficient)

    out["v8_fundamental_future_score"] = fundamental_effective.round(1)
    out["v8_narrative_alignment_score"] = narrative_effective.round(1)
    out["v8_market_sector_score"] = market_effective.round(1)
    out["v8_flow_score"] = flow_effective.round(1)
    out["v8_production_score_coverage_pct"] = total_coverage.round(1)
    out["v8_strategic_score"] = final_score.round(1)
    out["multibagger_selection_score"] = final_score.round(1)
    out["growth_compounder_selection_score"] = final_score.where(~turnaround, np.nan).round(1)
    out["turnaround_selection_score"] = final_score.where(turnaround, np.nan).round(1)
    out["multibagger_quality_score"] = fundamental_effective.round(1)
    out["final_score"] = final_score.round(1)
    out["multibagger_final_score_formula"] = (
        "V8_55_FUNDAMENTAL_25_NARRATIVE_ALIGNMENT_10_MARKET_10_FLOW_NO_DOUBLE_COUNT"
    )
    out["production_scoring_version"] = PRODUCTION_SCORING_VERSION
    out["production_ai_weight_pct"] = 0.0
    out["production_eoff_weight_pct"] = 0.0
    out["production_legacy_overlay_weight_pct"] = 0.0

    for column in (
        "growth_narrative_contribution_points",
        "growth_emir_contribution_points",
        "turnaround_narrative_contribution_points",
        "turnaround_emir_contribution_points",
    ):
        out[column] = 0.0
    return out


def apply_core_production_scoring(
    frame: pd.DataFrame | None,
    *,
    weights: V8ProductionWeights = DEFAULT_WEIGHTS,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    weights.validate()
    out = frame.copy()

    technical_raw, technical_cov = _weighted_component(
        out,
        (
            (
                "v8_structure_pure_score",
                0.40,
                "v8_structure_pure_coverage_pct",
                1.0,
            ),
            ("timing_score", 0.35, "timing_coverage_pct", 1.0),
            (
                "target_quality_score",
                0.25,
                "target_quality_coverage_pct",
                1.0,
            ),
        ),
    )
    technical_effective = _coverage_shrink(technical_raw, technical_cov)

    market_raw, market_cov = _weighted_component(
        out,
        (
            ("selector_trend_score", 0.25, "technical_feature_coverage_pct", 1.0),
            ("selector_momentum_score", 0.25, "technical_feature_coverage_pct", 1.0),
            ("selector_relative_strength_score", 0.25, "technical_feature_coverage_pct", 1.0),
            ("sector_relative_strength_score", 0.15, None, 1.0),
            ("liquidity_score", 0.10, None, 1.0),
        ),
    )
    market_effective = _coverage_shrink(market_raw, market_cov)

    narrative_raw, narrative_cov = _narrative_alignment_component(out, swing=True)
    narrative_effective = _coverage_shrink(narrative_raw, narrative_cov)

    # One canonical flow score only. ``flow_score`` is a legacy blend of the
    # same demand/CMF/volume inputs and is intentionally excluded.
    flow_raw = _series(out, "effective_silent_accumulation_score")
    flow_cov = _series(
        out, "silent_accumulation_confidence", 0.0,
    ).fillna(0.0).clip(0.0, 100.0)
    flow_effective = _coverage_shrink(flow_raw, flow_cov)

    data_raw, data_cov = _weighted_component(
        out,
        (
            ("data_quality_score", 0.60, None, 1.0),
            ("validation_score", 0.40, None, 1.0),
        ),
    )
    data_effective = _coverage_shrink(data_raw, data_cov)

    raw_score = (
        weights.core_technical * technical_effective
        + weights.core_market_sector * market_effective
        + weights.core_narrative_alignment * narrative_effective
        + weights.core_flow * flow_effective
        + weights.core_data_validation * data_effective
    )
    total_coverage = (
        weights.core_technical * technical_cov
        + weights.core_market_sector * market_cov
        + weights.core_narrative_alignment * narrative_cov
        + weights.core_flow * flow_cov
        + weights.core_data_validation * data_cov
    ).clip(0.0, 100.0)
    final_score = _coverage_shrink(raw_score, total_coverage)
    hard_block = _bool_series(out, "narrative_hard_block")
    final_score = final_score.where(~hard_block, np.minimum(final_score, 35.0))
    final_score = final_score.mask(total_coverage.lt(55.0))

    out["v8_technical_score"] = technical_effective.round(1)
    out["v8_market_sector_score"] = market_effective.round(1)
    out["v8_narrative_alignment_score"] = narrative_effective.round(1)
    out["v8_flow_score"] = flow_effective.round(1)
    out["v8_data_validation_score"] = data_effective.round(1)
    out["v8_production_score_coverage_pct"] = total_coverage.round(1)
    out["core_priority_score"] = final_score.round(1)
    out["final_score"] = final_score.round(1)
    out["core_final_score_formula"] = (
        "V8_45_TECHNICAL_15_MARKET_20_NARRATIVE_ALIGNMENT_15_FLOW_5_DATA_NO_DOUBLE_COUNT"
    )
    out["production_scoring_version"] = PRODUCTION_SCORING_VERSION
    out["production_ai_weight_pct"] = 0.0
    out["production_eoff_weight_pct"] = 0.0
    out["production_legacy_overlay_weight_pct"] = 0.0
    out["core_narrative_contribution_points"] = 0.0
    out["core_emir_contribution_points"] = 0.0
    return out


def drop_nonproduction_columns(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Remove research-only columns from production-facing frames."""
    if frame is None:
        return pd.DataFrame()
    if not isinstance(frame, pd.DataFrame):
        return pd.DataFrame()
    prefixes = (
        "eoff_",
        "best_buy_",
        "time_cycle_",
        "lunar_",
        "cycle_",
        "ai_",
        "peer_ai_",
        "hybrid_",
    )
    exact = {
        "quick_buy_state",
        "quick_buy_action",
        "quick_buy_score",
        "quick_buy_reason",
        "next_reversal_window_start",
        "next_reversal_window_end",
        "bars_to_reversal_window",
        "dominant_cycle_bars",
    }
    removable = [
        column for column in frame.columns
        if column in exact or column.lower().startswith(prefixes)
    ]
    return frame.drop(columns=removable, errors="ignore")


__all__ = [
    "PRODUCTION_SCORING_VERSION",
    "V8ProductionWeights",
    "DEFAULT_WEIGHTS",
    "MULTIBAGGER_COMPONENT_SOURCES",
    "CORE_COMPONENT_SOURCES",
    "validate_disjoint_component_sources",
    "MULTIBAGGER_DECISION_COLUMNS",
    "CORE_DECISION_COLUMNS",
    "slim_production_frame",
    "build_production_evidence_detail",
    "production_scoring_audit_frame",
    "apply_multibagger_production_scoring",
    "apply_core_production_scoring",
    "drop_nonproduction_columns",
]
