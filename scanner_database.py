from __future__ import annotations

"""Database-first, fail-soft Supabase bridge for IDX Super Scanner v7.

The bridge has two independent responsibilities:
1. Read complete cached research payloads before external providers are called.
2. Persist bounded snapshots and durable cache payloads after a scan.

A database outage never stops the scanner. Unsafe publishable/anon keys are
rejected for backend reads and writes.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import hashlib
import json
import math
import os
import re

import numpy as np
import pandas as pd
try:
    import requests
except ModuleNotFoundError:  # Deployment requirements install it; local core tests remain importable.
    class _RequestsUnavailable:
        @staticmethod
        def get(*_args: Any, **_kwargs: Any) -> Any:
            raise ModuleNotFoundError("requests is required for Supabase REST")

        @staticmethod
        def post(*_args: Any, **_kwargs: Any) -> Any:
            raise ModuleNotFoundError("requests is required for Supabase REST")

    requests = _RequestsUnavailable()

from research_maintenance import MODEL_VERSIONS, semantic_refresh_reason, model_registry_frame
from ihsg_direction import ihsg_snapshot_frame
from selector_engine import selector_snapshot_frame

DATABASE_BRIDGE_VERSION = "11.0-v7.16.2"
DATABASE_SCHEMA_VERSION = "scanner_schema_v11"

TABLE_CONFLICT_TARGETS: dict[str, str] = {
    "fundamental_cache": "ticker",
    "fundamental_history_cache": "ticker",
    "forward_quality_cache": "ticker",
    "refresh_state": "entity_key",
    "scan_checkpoints": "checkpoint_id",
    "model_registry": "component",
    "source_events": "event_key",
    "research_outcomes": "outcome_id",
    "backfill_state": "entity_key",
    "idx_trading_calendar": "trade_date",
    "ai_execution_outcomes": "signal_id",
    "selector_snapshots": "snapshot_id",
    "selector_outcomes": "outcome_id",
    "selector_model_evaluations": "evaluation_id",
    "narrative_events": "narrative_event_id",
    "narrative_event_outcomes": "narrative_outcome_id",
}

TABLE_FIELD_TYPES: dict[str, dict[str, set[str]]] = {
    "fundamental_snapshots": {
        "text": {
            "snapshot_id", "scan_id", "ticker", "period_end", "statement_date",
            "fundamental_data_grade", "fundamental_source_families", "fundamental_reliability", "model_version",
            "schema_version", "as_of", "fundamental_fetched_at",
            "database_source_state", "content_hash",
        },
        "numeric": {
            "fundamental_score", "fundamental_score_10", "fundamental_coverage",
            "revenue_growth", "earnings_growth", "roe",
            "roa", "roic_proxy", "net_margin", "operating_margin",
            "operating_cash_flow", "free_cash_flow", "cash_conversion_ttm",
            "debt_equity", "net_debt_ebitda", "interest_coverage", "market_cap",
            "statement_age_days",
        },
        "boolean": {"fundamental_official_verified"},
        "integer": set(), "json": set(),
        "required": {"snapshot_id", "ticker", "as_of", "model_version", "schema_version"},
    },
    "multibagger_snapshots": {
        "text": {
            "snapshot_id", "scan_id", "ticker", "multibagger_status",
            "research_recommendation_status", "multibagger_candidate_type",
            "multibagger_lane", "research_eligibility_reason",
            "multibagger_scoring_state",
            "turnaround_research_state", "turnaround_gate_reasons",
            "silent_accumulation_state", "accumulation_regime",
            "overall_research_confidence_grade", "top_positive_drivers",
            "top_negative_drivers", "scoring_reason_codes", "project_stage",
            "best_buy_date", "best_buy_window_start", "best_buy_window_end",
            "eoff_state", "eoff_public_validation_state", "model_version",
            "schema_version", "as_of", "liquidity_bucket", "silent_accumulation_calibration_policy",
            "multibagger_evidence_class", "distribution_evidence_state",
        },
        "numeric": {
            "multibagger_quality_score", "execution_readiness_score",
            "economic_earnings_score", "economic_earnings_confidence",
            "minority_leakage_pct", "ocf_ebitda_conversion",
            "silent_accumulation_score", "silent_accumulation_raw_score", "silent_accumulation_confidence",
            "silent_accumulation_liquidity_adjustment", "silent_accumulation_liquidity_min_confirmation", "silent_accumulation_v4_adjustment", "accumulation_persistence_score",
            "accumulation_positive_windows_pct", "persistent_bid_score",
            "data_confidence_score", "fundamental_confidence_score",
            "future_fundamental_confidence_score", "technical_confidence_score",
            "eoff_confidence_score", "overall_research_confidence",
            "confidence_adjusted_multibagger_score", "project_pipeline_score",
            "growth_compounder_score", "growth_compounder_base_score",
            "growth_compounder_selection_score", "turnaround_recovery_score",
            "confidence_adjusted_turnaround_score", "turnaround_selection_score",
            "effective_silent_accumulation_score",
            "multibagger_metric_coverage_pct",
            "growth_pillar_coverage_pct",
            "profitability_pillar_coverage_pct",
            "cashflow_pillar_coverage_pct",
            "safety_pillar_coverage_pct",
            "runway_pillar_coverage_pct",
            "valuation_pillar_coverage_pct",
            "project_stage_probability_pct", "project_success_probability_pct",
            "future_fundamental_impact_score", "eoff_reconstruction_score",
            "eoff_public_lift", "last_price",
            "multibagger_score_comparability_pct", "multibagger_production_rank",
            "narrative_evidence_coverage_pct", "issuer_alignment_coverage_pct",
            "emir_method_coverage_pct", "distribution_severity_score",
            "distribution_penalty_points",
        },
        "integer": {
            "accumulation_longest_run", "absorption_confirmed_days20",
            "failed_absorption_days20", "effort_result_absorption20",
            "effort_result_distribution20", "turnaround_recovery_signals",
        },
        "boolean": {
            "research_eligible", "portfolio_allocation_eligible",
            "critical_research_flags", "operational_recovery_flags",
            "multibagger_metric_data_gate", "multibagger_rank_eligible",
        }, "json": set(),
        "required": {"snapshot_id", "ticker", "as_of", "model_version", "schema_version"},
    },
    "technical_snapshots": {
        "text": {
            "snapshot_id", "scan_id", "ticker", "active_setup",
            "technical_entry_state", "accumulation_regime", "model_version",
            "schema_version", "as_of",
        },
        "numeric": {
            "last_price", "entry", "stop_loss", "tp1", "tp2",
            "execution_readiness_score", "silent_accumulation_score",
            "silent_accumulation_confidence", "accumulation_persistence_score",
            "relative_strength60", "roc60", "roc120", "adtv20_idr",
        },
        "integer": set(), "boolean": set(), "json": set(),
        "required": {"snapshot_id", "ticker", "as_of", "model_version", "schema_version"},
    },
    "eoff_predictions": {
        "text": {
            "snapshot_id", "scan_id", "ticker", "best_buy_date", "best_buy_raw_date",
            "best_buy_calendar_state", "eoff_unique_anchor_signature",
            "best_buy_window_start", "best_buy_window_end", "eoff_state",
            "eoff_strength_label", "eoff_direction_bias",
            "eoff_public_validation_state", "model_version", "schema_version", "as_of",
        },
        "numeric": {
            "best_buy_score", "best_buy_confidence", "best_buy_entry_low",
            "best_buy_entry_high", "best_buy_trigger", "eoff_reconstruction_score",
            "eoff_public_lift", "time_cycle_confidence", "eoff_fib_unique_anchor_ratio", "eoff_fib_dominant_anchor_share",
        },
        "integer": {"eoff_public_directional_events", "best_buy_date_adjustment_days", "eoff_fib_unique_anchor_count"},
        "boolean": {"best_buy_calendar_verified", "eoff_unique_anchor_gate"}, "json": set(),
        "required": {"snapshot_id", "ticker", "as_of", "model_version", "schema_version"},
    },
    "ihsg_direction_snapshots": {
        "text": {
            "snapshot_id", "scan_id", "ticker", "as_of", "horizon",
            "raw_direction", "prediction_state", "validation_state", "data_state",
            "regime", "consensus_direction", "risk_action", "feature_hash",
            "model_version", "schema_version",
        },
        "numeric": {
            "benchmark_close", "prob_up_pct", "prob_sideways_pct", "prob_down_pct",
            "confidence_pct", "expected_return_pct", "return_p25_pct", "return_p75_pct",
            "neutral_band_pct", "effective_analogue_count", "median_distance",
            "directional_accuracy_pct", "directional_accuracy_ci_low_pct",
            "validation_coverage_pct", "brier_score", "baseline_brier_score",
            "brier_skill_pct", "regime_score", "consensus_confidence",
            "risk_budget_multiplier", "feature_coverage_pct", "breadth_ema50_pct",
        },
        "integer": {
            "horizon_bars", "analogue_count", "features_used",
            "validation_predictions", "directional_validation_predictions",
            "breadth_member_count",
        },
        "boolean": {"actionable", "eod_final"},
        "json": {"payload"},
        "required": {
            "snapshot_id", "ticker", "as_of", "horizon_bars",
            "model_version", "schema_version",
        },
    },
    "project_events": {
        "text": {
            "snapshot_id", "scan_id", "ticker", "project_name", "project_names",
            "project_stage", "project_source_families", "project_source_urls",
            "project_execution_flags", "last_verified_at", "review_origin",
            "event_date", "model_version", "schema_version", "as_of",
        },
        "numeric": {
            "project_completion_pct", "project_funding_secured_pct",
            "project_ownership_pct", "project_capex_idr",
            "project_expected_revenue_idr", "project_expected_ebitda_idr",
            "project_data_coverage",
        },
        "boolean": {"project_source_quorum_verified"},
        "integer": set(), "json": set(),
        "required": {"snapshot_id", "ticker", "as_of", "model_version", "schema_version"},
    },
    "provider_health": {
        "text": {
            "snapshot_id", "scan_id", "provider", "scope", "status", "asof",
            "error", "error_code", "source_family", "model_version", "schema_version",
            "as_of",
        },
        "numeric": {"rows"}, "integer": set(), "boolean": set(), "json": set(),
        "required": {"snapshot_id", "as_of", "model_version", "schema_version"},
    },
    "scan_runs": {
        "text": {
            "snapshot_id", "scan_id", "started_at", "finished_at", "database_mode",
            "model_version", "schema_version", "as_of",
        },
        "numeric": set(),
        "integer": {"ticker_count", "prepared_count", "multibagger_count", "core_swing_count"},
        "boolean": set(), "json": set(),
        "required": {"snapshot_id", "scan_id", "as_of", "model_version", "schema_version"},
    },
    "fundamental_cache": {
        "text": {
            "ticker", "source_families", "data_grade", "statement_date",
            "source_fetched_at", "source_checked_at", "content_hash", "refresh_state",
            "parser_version", "event_fingerprint", "next_check_at", "refresh_reason",
            "model_version", "schema_version",
        },
        "numeric": {"coverage"}, "integer": set(), "boolean": set(),
        "json": {"payload"},
        "required": {"ticker", "payload", "source_checked_at", "model_version", "schema_version"},
    },
    "fundamental_history_cache": {
        "text": {
            "ticker", "latest_period", "source_families", "source_checked_at",
            "content_hash", "refresh_state", "parser_version", "event_fingerprint",
            "next_check_at", "refresh_reason", "model_version", "schema_version",
        },
        "numeric": set(), "integer": {"period_count"}, "boolean": set(),
        "json": {"payload"},
        "required": {"ticker", "payload", "source_checked_at", "model_version", "schema_version"},
    },
    "forward_quality_cache": {
        "text": {
            "ticker", "source_families", "last_verified_at", "source_checked_at",
            "content_hash", "refresh_state", "parser_version", "event_fingerprint",
            "next_check_at", "refresh_reason", "model_version", "schema_version",
        },
        "numeric": set(), "integer": {"project_count"}, "boolean": set(),
        "json": {"payload"},
        "required": {"ticker", "payload", "source_checked_at", "model_version", "schema_version"},
    },
    "refresh_state": {
        "text": {
            "entity_key", "entity_type", "ticker", "source_family", "state",
            "last_checked_at", "last_changed_at", "valid_until", "content_hash",
            "detail", "parser_version", "event_fingerprint", "refresh_reason",
            "model_version", "schema_version",
        },
        "numeric": set(), "integer": set(), "boolean": set(),
        "json": {"payload_metadata"},
        "required": {"entity_key", "entity_type", "state", "model_version", "schema_version"},
    },
    "scan_checkpoints": {
        "text": {
            "checkpoint_id", "scan_id", "phase", "last_ticker", "status", "detail",
            "as_of", "model_version", "schema_version",
        },
        "numeric": set(), "integer": {"batch_number", "rows_completed", "rows_total"},
        "boolean": set(), "json": set(),
        "required": {"checkpoint_id", "scan_id", "phase", "status", "as_of", "model_version", "schema_version"},
    },
    "model_registry": {
        "text": {"component", "semantic_version", "released_at", "config_hash", "model_version", "schema_version"},
        "numeric": set(), "integer": set(), "boolean": {"is_active"}, "json": {"metadata"},
        "required": {"component", "semantic_version", "released_at", "model_version", "schema_version"},
    },
    "source_events": {
        "text": {"event_key", "ticker", "event_type", "event_date", "source_family", "content_hash", "event_fingerprint", "detected_at", "last_seen_at", "resolved_at", "model_version", "schema_version"},
        "numeric": set(), "integer": set(), "boolean": {"refresh_required"}, "json": {"payload"},
        "required": {"event_key", "event_type", "detected_at", "last_seen_at", "model_version", "schema_version"},
    },
    "research_outcomes": {
        "text": {"outcome_id", "ticker", "signal_family", "signal_timestamp", "signal_date", "anchor_id", "liquidity_bucket", "predicted_state", "predicted_direction", "prediction_window_start", "prediction_window_end", "outcome_status", "resolved_at", "actual_low_date", "actual_high_date", "model_version", "schema_version"},
        "numeric": {"signal_score", "signal_confidence", "entry_reference", "forward_return_1d", "forward_return_5d", "forward_return_10d", "forward_return_20d", "maximum_favourable_excursion", "maximum_adverse_excursion"},
        "integer": {"horizon_bars"}, "boolean": {"hit"}, "json": {"payload"},
        "required": {"outcome_id", "ticker", "signal_family", "signal_timestamp", "signal_date", "outcome_status", "model_version", "schema_version"},
    },
    "backfill_state": {
        "text": {"entity_key", "ticker", "entity_type", "status", "refresh_reason", "last_attempt_at", "last_success_at", "next_due_at", "model_version", "schema_version"},
        "numeric": set(), "integer": {"cohort", "active_cohort", "priority", "failure_count"}, "boolean": set(), "json": {"payload"},
        "required": {"entity_key", "ticker", "entity_type", "status", "model_version", "schema_version"},
    },
    "idx_trading_calendar": {
        "text": {"trade_date", "session_type", "source_family", "source_url", "content_hash", "verified_at", "notes", "model_version", "schema_version"},
        "numeric": set(), "integer": set(), "boolean": {"is_open"}, "json": set(),
        "required": {"trade_date", "is_open", "session_type", "model_version", "schema_version"},
    },
    "ai_execution_outcomes": {
        "text": {
            "signal_id", "ticker", "strategy", "signal_date", "memory_state",
            "result", "fill_date", "exit_date", "resolved_at", "outcome_quality",
            "no_fill_reason", "ai_version", "model_version", "schema_version",
        },
        "numeric": {
            "entry", "trigger_price", "stop_loss", "tp1", "tp2",
            "fill_price", "exit_price", "r_multiple", "expectancy_after_cost_r",
            "mfe_r", "mae_r", "mfe_pct", "mae_pct", "gross_return_pct",
            "net_return_pct", "roundtrip_cost_pct", "cost_r",
            "fill_slippage_pct",
        },
        "integer": {"fill_delay_bars"},
        "boolean": {
            "filled", "tp1_hit", "tp1_before_sl", "tp2_hit",
            "outcome_ambiguous",
        },
        "json": {"payload"},
        "required": {
            "signal_id", "ticker", "signal_date", "memory_state",
            "model_version", "schema_version",
        },
    },
    "selector_snapshots": {
        "text": {
            "snapshot_id", "ticker", "as_of", "horizon", "model_state",
            "selector_data_state", "selector_missing_features",
            "champion_model", "selected_reason", "selection_risks",
            "selector_universe_state",
            "model_version", "schema_version",
        },
        "numeric": {
            "selection_rank", "swing_selection_score",
            "production_selection_rank", "technical_feature_coverage_pct",
            "effective_silent_accumulation_score",
            "silent_accumulation_confidence",
            "multibagger_timing_selector_score", "technical_selection_score",
            "silent_accumulation_score", "relative_strength_score",
            "expected_excess_return_pct", "outperform_probability_pct",
            "selector_score", "ai_weight_pct", "relative_overlay_weight_pct",
        },
        "integer": {"horizon_bars", "selector_missing_feature_count"},
        "boolean": {"selector_rank_eligible", "score_inflation_guard_active"},
        "json": {"payload"},
        "required": {
            "snapshot_id", "ticker", "as_of", "horizon_bars",
            "model_version", "schema_version",
        },
    },
    "selector_outcomes": {
        "text": {
            "outcome_id", "snapshot_id", "ticker", "signal_date", "horizon",
            "model_state", "champion_model", "outcome_status", "resolved_at",
            "model_version", "schema_version",
        },
        "numeric": {
            "predicted_excess_return_pct", "outperform_probability_pct",
            "selector_score", "stock_return_pct", "benchmark_return_pct",
            "net_excess_return_pct",
        },
        "integer": {"horizon_bars"},
        "boolean": {"outperformed_after_cost"},
        "json": {"payload"},
        "required": {
            "outcome_id", "ticker", "signal_date", "horizon_bars",
            "outcome_status", "model_version", "schema_version",
        },
    },
    "selector_model_evaluations": {
        "text": {
            "evaluation_id", "as_of", "horizon", "model", "selector_version",
            "promoted_model", "walkforward_best_model", "ai_promotion_state", "model_version",
            "schema_version",
        },
        "numeric": {
            "brier_score", "baseline_brier_score", "brier_skill_pct",
            "net_excess_expectancy_pct", "net_absolute_expectancy_pct",
            "topk_hit_rate_pct", "spearman_ic", "max_drawdown_pct",
        },
        "integer": {
            "horizon_bars", "training_rows", "model_fit_rows", "calibration_rows",
            "evaluation_rows", "evaluation_dates", "evaluation_tickers",
            "challenger_rank",
        },
        "boolean": {"ai_can_influence"},
        "json": {"payload"},
        "required": {
            "evaluation_id", "as_of", "horizon_bars", "model",
            "model_version", "schema_version",
        },
    },
    "narrative_events": {
        "text": {
            "narrative_event_id", "ticker", "event_date", "detected_at",
            "event_type", "event_family", "headline", "summary",
            "source_url", "source_hostname", "registered_official_domain",
            "source_state", "source_family", "impact_direction",
            "content_hash", "event_cluster_key", "event_evidence_state",
            "detection_time_source", "entity_match_state", "event_status",
            "requested_event_status", "lifecycle_evidence_state",
            "resolved_at", "supersedes_event_id", "resolution_source_url",
            "narrative_engine_version",
            "model_version", "schema_version",
        },
        "numeric": {
            "source_quality_score", "materiality_score", "novelty_score",
            "financial_bridge_score", "narrative_decay_weight",
            "catalyst_proximity_score", "event_strength_score",
            "signed_event_strength", "event_age_days",
        },
        "integer": {"impact_sign"},
        "boolean": {
            "official_claimed", "official_verified", "source_present", "event_active",
            "future_detection_invalid",
        },
        "json": {"payload"},
        "required": {
            "narrative_event_id", "ticker", "detected_at",
            "event_type", "model_version", "schema_version",
        },
    },
    "narrative_event_outcomes": {
        "text": {
            "narrative_outcome_id", "narrative_event_id", "ticker",
            "event_type", "event_family", "impact_direction", "entry_policy",
            "signal_timestamp", "signal_date",
            "anchor_date", "outcome_status", "resolved_at",
            "narrative_engine_version", "model_version", "schema_version",
        },
        "numeric": {
            "entry_reference", "roundtrip_cost_pct",
            "stock_return_5d_pct", "benchmark_return_5d_pct",
            "net_excess_return_5d_pct", "mfe_5d_pct", "mae_5d_pct",
            "directional_excess_return_5d_pct",
            "stock_return_20d_pct", "benchmark_return_20d_pct",
            "net_excess_return_20d_pct", "mfe_20d_pct", "mae_20d_pct",
            "directional_excess_return_20d_pct",
            "stock_return_60d_pct", "benchmark_return_60d_pct",
            "net_excess_return_60d_pct", "mfe_60d_pct", "mae_60d_pct",
            "directional_excess_return_60d_pct",
        },
        "integer": {"impact_sign"},
        "boolean": {"converted_5d", "converted_20d", "converted_60d"},
        "json": {"payload"},
        "required": {
            "narrative_outcome_id", "narrative_event_id", "ticker",
            "signal_timestamp", "outcome_status",
            "model_version", "schema_version",
        },
    },
    "narrative_snapshots": {
        "text": {
            "snapshot_id", "scan_id", "ticker", "as_of",
            "narrative_as_of", "narrative_state",
            "latest_narrative_event", "latest_narrative_event_type",
            "latest_narrative_event_date", "issuer_alignment_state",
            "retail_adoption_stage", "retail_proxy_disclaimer",
            "narrative_flow_convergence_state",
            "narrative_silent_integration_state",
            "narrative_primary_reason", "narrative_primary_risk",
            "narrative_news_collection_state", "narrative_production_policy",
            "narrative_conversion_state_5d",
            "narrative_conversion_state_20d",
            "narrative_conversion_state_60d",
            "narrative_engine_version", "model_version", "schema_version",
        },
        "numeric": {
            "narrative_score", "narrative_effective_score",
            "narrative_evidence_coverage_pct",
            "narrative_source_quality_score", "narrative_novelty_score",
            "narrative_financial_bridge_score", "issuer_alignment_score",
            "issuer_alignment_effective_score",
            "issuer_alignment_coverage_pct",
            "retail_adoption_proxy_score",
            "retail_adoption_proxy_coverage_pct",
            "narrative_crowding_risk_score",
            "narrative_flow_convergence_score",
            "narrative_flow_effective_score",
            "narrative_flow_convergence_coverage_pct",
            "narrative_overlay_reliability_pct",
            "narrative_swing_overlay_reliability_pct",
            "narrative_growth_rank_adjustment",
            "narrative_turnaround_rank_adjustment",
            "narrative_swing_rank_adjustment",
            "narrative_conversion_rate_5d_pct",
            "narrative_conversion_effective_5d_score",
            "narrative_conversion_expectancy_5d_pct",
            "narrative_conversion_rate_20d_pct",
            "narrative_conversion_effective_20d_score",
            "narrative_conversion_expectancy_20d_pct",
            "narrative_conversion_rate_60d_pct",
            "narrative_conversion_effective_60d_score",
            "narrative_conversion_expectancy_60d_pct",
            "narrative_flow_proxy_score",
        },
        "integer": {
            "narrative_event_count", "narrative_active_event_count",
            "narrative_missing_source_event_count",
            "narrative_inactive_lifecycle_event_count",
            "narrative_entity_unverified_event_count",
            "narrative_event_cluster_count",
            "narrative_corroborated_cluster_count",
            "narrative_positive_event_count",
            "narrative_negative_event_count",
            "narrative_official_event_count",
            "issuer_alignment_positive_events",
            "issuer_alignment_negative_events",
            "narrative_contradiction_count",
            "narrative_items_reviewed",
            "narrative_conversion_resolved_5d",
            "narrative_conversion_resolved_20d",
            "narrative_conversion_resolved_60d",
        },
        "boolean": {"narrative_hard_block"},
        "json": {"payload"},
        "required": {
            "snapshot_id", "ticker", "as_of",
            "model_version", "schema_version",
        },
    },
}


class DatabaseWriteError(RuntimeError):
    def __init__(self, table: str, status_code: int | None, body: str, record_hint: str = "") -> None:
        status = f"HTTP {status_code}" if status_code is not None else "HTTP UNKNOWN"
        detail = " ".join(str(body or "").split())[:700]
        hint = f"; record={record_hint}" if record_hint else ""
        super().__init__(f"{table}: {status}; {detail}{hint}")
        self.table = table
        self.status_code = status_code
        self.body = detail
        self.record_hint = record_hint


class DatabaseReadError(RuntimeError):
    def __init__(self, table: str, status_code: int | None, body: str) -> None:
        status = f"HTTP {status_code}" if status_code is not None else "HTTP UNKNOWN"
        detail = " ".join(str(body or "").split())[:700]
        super().__init__(f"{table}: {status}; {detail}")
        self.table = table
        self.status_code = status_code
        self.body = detail


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        if isinstance(value, date) and not isinstance(value, datetime):
            return value.isoformat()
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        return stamp.isoformat()
    if isinstance(value, (dict, Mapping)):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value if isinstance(value, str) else str(value)


def _frame_records(frame: pd.DataFrame | None, columns: Iterable[str]) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    selected = [column for column in columns if column in frame.columns]
    if not selected:
        return []
    local = frame[selected].copy()
    return [{key: _json_safe(value) for key, value in row.items()} for row in local.to_dict("records")]


def _frame_records_with_payload(
    frame: pd.DataFrame | None,
    columns: Iterable[str],
) -> list[dict[str, Any]]:
    """Keep searchable columns plus the complete outcome for future retraining."""
    if frame is None or frame.empty:
        return []
    selected = [column for column in columns if column in frame.columns]
    records: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        record = {
            column: _json_safe(row.get(column))
            for column in selected
        }
        record["payload"] = {
            str(key): _json_safe(value)
            for key, value in row.items()
        }
        records.append(record)
    return records


def _coerce_numeric(value: Any) -> float | None:
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    if isinstance(value, (np.bool_, bool)):
        return float(bool(value))
    if isinstance(value, (np.integer, int, np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = _clean_text(value)
    if not text or text.upper() in {"N/A", "NA", "NONE", "NULL", "NAN", "INF", "-INF", "UNAVAILABLE"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").replace("Rp", "").replace("IDR", "").replace("USD", "")
    text = text.replace("%", "").replace(" ", "")
    if text.count(",") == 1 and text.count(".") == 0:
        text = text.replace(",", ".")
    else:
        text = text.replace(",", "")
    text = re.sub(r"[^0-9eE+\-.]", "", text)
    if not text or text in {"-", "+", "."}:
        return None
    try:
        number = float(text)
    except Exception:
        return None
    if negative:
        number = -abs(number)
    return number if math.isfinite(number) else None


def _coerce_integer(value: Any) -> int | None:
    number = _coerce_numeric(value)
    return None if number is None else int(round(number))


def _coerce_boolean(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int, np.floating, float)):
        if not math.isfinite(float(value)):
            return None
        return bool(int(value))
    text = _clean_text(value).lower()
    if text in {"true", "t", "1", "yes", "y", "on", "verified", "ok"}:
        return True
    if text in {"false", "f", "0", "no", "n", "off", "unverified", ""}:
        return False if text else None
    return None


def _coerce_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (dict, Mapping, list, tuple, set)):
        return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
    safe = _json_safe(value)
    if safe is None:
        return None
    text = str(safe).strip()
    return text or None


def _normalise_record(table: str, record: Mapping[str, Any]) -> dict[str, Any] | None:
    spec = TABLE_FIELD_TYPES.get(table)
    if not spec:
        return {str(k): _json_safe(v) for k, v in record.items()}
    allowed = spec["text"] | spec["numeric"] | spec["integer"] | spec["boolean"] | spec.get("json", set())
    output: dict[str, Any] = {}
    for key in allowed:
        if key not in record:
            continue
        value = record.get(key)
        if key in spec["numeric"]:
            output[key] = _coerce_numeric(value)
        elif key in spec["integer"]:
            output[key] = _coerce_integer(value)
        elif key in spec["boolean"]:
            output[key] = _coerce_boolean(value)
        elif key in spec.get("json", set()):
            output[key] = _json_safe(value)
        else:
            output[key] = _coerce_text(value)
            if output[key] == "" and (key.endswith("_date") or key.endswith("_at") or key in {"trade_date", "period_end", "statement_date"}):
                output[key] = None
    for field in spec["required"]:
        value = output.get(field)
        if value is None or value == "":
            return None
    return output


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


VOLATILE_SEMANTIC_FIELDS = {
    "as_of", "scan_id", "snapshot_id", "created_at", "updated_at",
    "database_source_state", "database_source_checked_at",
    "source_checked_at", "source_fetched_at", "fundamental_fetched_at",
    "generated_at", "last_seen_at", "age_days", "refresh_state",
    "refresh_reason", "next_check_at",
}


def _semantic_payload(payload: Any) -> Any:
    """Remove transport/audit timestamps before computing a fact-change hash."""
    if isinstance(payload, Mapping):
        return {
            str(key): _semantic_payload(value)
            for key, value in sorted(payload.items(), key=lambda item: str(item[0]))
            if str(key) not in VOLATILE_SEMANTIC_FIELDS
            and not str(key).endswith(("_checked_at", "_fetched_at", "_generated_at"))
        }
    if isinstance(payload, (list, tuple)):
        return [_semantic_payload(value) for value in payload]
    return _json_safe(payload)


def _semantic_hash(payload: Any) -> str:
    return _stable_hash(_semantic_payload(payload))


def _snapshot_id(table: str, record: Mapping[str, Any], as_of: str) -> str:
    as_of_day = _clean_text(as_of)[:10]
    identities = {
        "fundamental_snapshots": ("ticker", "period_end", "model_version"),
        "multibagger_snapshots": ("ticker", "model_version"),
        "technical_snapshots": ("ticker", "model_version"),
        "eoff_predictions": ("ticker", "best_buy_date", "model_version"),
        "ihsg_direction_snapshots": ("ticker", "horizon_bars", "model_version"),
        "project_events": ("ticker", "project_name", "project_stage", "event_date"),
        "provider_health": ("provider", "scope", "status", "model_version"),
        "scan_runs": ("scan_id",),
    }
    keys = identities.get(table, ("ticker", "model_version"))
    identity = "|".join(_clean_text(record.get(key)) for key in keys)
    date_part = "" if table in {"project_events", "scan_runs"} else as_of_day
    return hashlib.sha256(f"{table}|{identity}|{date_part}".encode("utf-8")).hexdigest()


def _download_report_frame(report: Any) -> pd.DataFrame:
    if isinstance(report, pd.DataFrame):
        return report.copy()
    if report is None:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    provider = _clean_text(getattr(report, "provider", "PRIMARY_OHLCV")) or "PRIMARY_OHLCV"
    downloaded_at = _json_safe(getattr(report, "downloaded_at", None))
    for ticker in list(getattr(report, "downloaded", []) or []):
        rows.append({"provider": provider, "scope": ticker, "status": "OK", "rows": None, "asof": downloaded_at})
    for ticker, error in dict(getattr(report, "failed", {}) or {}).items():
        rows.append({"provider": provider, "scope": ticker, "status": "FAILED", "rows": 0, "asof": downloaded_at, "error": error})
    for ticker, warning in dict(getattr(report, "warnings", {}) or {}).items():
        rows.append({"provider": provider, "scope": ticker, "status": "WARNING", "rows": None, "asof": downloaded_at, "error": warning})
    summary_fields = {
        "cache_hits": getattr(report, "cache_hits", None),
        "incremental_refreshes": getattr(report, "incremental_refreshes", None),
        "full_refreshes": getattr(report, "full_refreshes", None),
        "provider_calls": getattr(report, "provider_calls", None),
        "downloaded_bars": getattr(report, "downloaded_bars", None),
    }
    if any(value not in {None, 0} for value in summary_fields.values()):
        rows.append({
            "provider": provider, "scope": "__SUMMARY__", "status": "METRICS",
            "rows": summary_fields.get("downloaded_bars"), "asof": downloaded_at,
            "error": json.dumps(summary_fields, ensure_ascii=False, sort_keys=True),
        })
    return pd.DataFrame(rows)


def _provider_health_frame(result: Mapping[str, Any]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for key in (
        "provider_report", "fundamental_history_report", "independent_provider_report",
        "automatic_forward_report", "twelve_data_report", "download_report",
        "database_read_report",
    ):
        frame = _download_report_frame(result.get(key))
        if not frame.empty:
            frame = frame.copy()
            if "source_family" not in frame.columns:
                frame["source_family"] = key.upper()
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    keys = [column for column in ("provider", "scope", "status", "asof", "error") if column in combined.columns]
    return combined.drop_duplicates(subset=keys, keep="last") if keys else combined


def _as_utc_timestamp(value: Any) -> pd.Timestamp:
    stamp = pd.to_datetime(value, errors="coerce", utc=True)
    return pd.Timestamp(stamp) if pd.notna(stamp) else pd.NaT


def _freshness_state(value: Any, current_days: int, stale_days: int, now: Any | None = None) -> tuple[str, float]:
    stamp = _as_utc_timestamp(value)
    if pd.isna(stamp):
        return "MISSING_TIMESTAMP", math.inf
    current = _as_utc_timestamp(now or datetime.now(timezone.utc))
    age = max(0.0, (current - stamp).total_seconds() / 86400.0)
    if age <= max(0, int(current_days)):
        return "DATABASE_CURRENT", age
    if age <= max(int(current_days), int(stale_days)):
        return "DATABASE_STALE_USABLE", age
    return "DATABASE_EXPIRED", age


@dataclass(frozen=True)
class DatabaseSettings:
    enabled: bool = False
    mode: str = "DISABLED"
    supabase_url: str = ""
    supabase_key: str = ""
    supabase_key_type: str = "NONE"
    schema: str = "public"
    timeout_seconds: float = 8.0
    outbox_path: str = ".scanner_cache/database_outbox.jsonl"
    max_rows_per_table: int = 2000
    cache_max_rows_per_table: int = 5000
    read_enabled: bool = True
    read_batch_size: int = 100
    fundamental_max_age_days: int = 21
    history_max_age_days: int = 30
    forward_max_age_days: int = 7
    stale_max_age_days: int = 180

    @classmethod
    def from_env(cls) -> "DatabaseSettings":
        enabled = _truthy(os.getenv("SCANNER_DATABASE_ENABLED"))
        requested_mode = os.getenv("SCANNER_DATABASE_MODE", "").strip().upper()
        url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
        key_candidates = (
            ("SECRET", os.getenv("SUPABASE_SECRET_KEY", "").strip()),
            ("SERVICE_ROLE", os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()),
            ("PUBLISHABLE", os.getenv("SUPABASE_PUBLISHABLE_KEY", "").strip()),
            ("ANON", os.getenv("SUPABASE_ANON_KEY", "").strip()),
        )
        key_type, key = next(((kind, value) for kind, value in key_candidates if value), ("NONE", ""))
        if not enabled:
            mode = "DISABLED"
        elif requested_mode == "OUTBOX_ONLY":
            mode = "OUTBOX_ONLY"
        elif url and key and key_type in {"SECRET", "SERVICE_ROLE"}:
            mode = "SUPABASE_REST"
        elif url and key and key_type in {"PUBLISHABLE", "ANON"}:
            mode = "CONFIG_UNSAFE_KEY"
        else:
            mode = "CONFIG_INCOMPLETE"
        read_setting = os.getenv("SCANNER_DATABASE_READ_ENABLED", "true")
        return cls(
            enabled=enabled,
            mode=mode,
            supabase_url=url,
            supabase_key=key,
            supabase_key_type=key_type,
            schema=os.getenv("SCANNER_DATABASE_SCHEMA", "public").strip() or "public",
            timeout_seconds=max(2.0, float(os.getenv("SCANNER_DATABASE_TIMEOUT", "8"))),
            outbox_path=os.getenv("SCANNER_DATABASE_OUTBOX", ".scanner_cache/database_outbox.jsonl"),
            max_rows_per_table=max(20, int(os.getenv("SCANNER_DATABASE_MAX_ROWS", "2000"))),
            cache_max_rows_per_table=max(500, int(os.getenv("SCANNER_DATABASE_CACHE_MAX_ROWS", "5000"))),
            read_enabled=_truthy(read_setting),
            read_batch_size=max(20, min(250, int(os.getenv("SCANNER_DATABASE_READ_BATCH_SIZE", "100")))),
            fundamental_max_age_days=max(1, int(os.getenv("SCANNER_DATABASE_FUNDAMENTAL_MAX_AGE_DAYS", "21"))),
            history_max_age_days=max(1, int(os.getenv("SCANNER_DATABASE_HISTORY_MAX_AGE_DAYS", "30"))),
            forward_max_age_days=max(1, int(os.getenv("SCANNER_DATABASE_FORWARD_MAX_AGE_DAYS", "7"))),
            stale_max_age_days=max(30, int(os.getenv("SCANNER_DATABASE_STALE_MAX_AGE_DAYS", "180"))),
        )


class ScannerDatabaseBridge:
    """Database-first repository and bounded snapshot writer."""

    def __init__(self, settings: DatabaseSettings | None = None) -> None:
        self.settings = settings or DatabaseSettings.from_env()
        self._write_details: dict[str, str] = {}

    def status_row(self, state: str | None = None, detail: str = "") -> dict[str, Any]:
        return {
            "bridge_version": DATABASE_BRIDGE_VERSION,
            "schema_version": DATABASE_SCHEMA_VERSION,
            "database_mode": self.settings.mode,
            "database_key_type": self.settings.supabase_key_type,
            "read_enabled": bool(self.settings.read_enabled),
            "state": state or ("READY" if self.settings.mode in {"OUTBOX_ONLY", "SUPABASE_REST"} else self.settings.mode),
            "table": "", "rows_attempted": 0, "rows_written": 0, "detail": detail,
        }

    def _headers(self) -> dict[str, str]:
        headers = {
            "apikey": self.settings.supabase_key,
            "Content-Type": "application/json",
            "Accept-Profile": self.settings.schema,
            "Content-Profile": self.settings.schema,
            "Prefer": "resolution=merge-duplicates,return=minimal",
            "User-Agent": f"idx-scanner/{DATABASE_BRIDGE_VERSION}",
        }
        if self.settings.supabase_key_type in {"SERVICE_ROLE", "ANON"}:
            headers["Authorization"] = f"Bearer {self.settings.supabase_key}"
        return headers

    def _read_headers(self) -> dict[str, str]:
        headers = {
            "apikey": self.settings.supabase_key,
            "Accept-Profile": self.settings.schema,
            "User-Agent": f"idx-scanner/{DATABASE_BRIDGE_VERSION}",
        }
        if self.settings.supabase_key_type in {"SERVICE_ROLE", "ANON"}:
            headers["Authorization"] = f"Bearer {self.settings.supabase_key}"
        return headers

    def _get_rows(self, table: str, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        endpoint = f"{self.settings.supabase_url}/rest/v1/{table}"
        response = requests.get(endpoint, headers=self._read_headers(), params=dict(params), timeout=self.settings.timeout_seconds)
        status_code = int(getattr(response, "status_code", 0) or 0)
        if not bool(getattr(response, "ok", 200 <= status_code < 300)):
            raise DatabaseReadError(table, status_code or None, getattr(response, "text", ""))
        payload = response.json()
        return payload if isinstance(payload, list) else []

    def _read_cache_table(
        self,
        table: str,
        tickers: Sequence[str],
        max_age_days: int,
        payload_kind: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        names = list(dict.fromkeys(str(ticker) for ticker in tickers if str(ticker).strip()))
        audit_columns = [
            "ticker", "provider", "scope", "status", "database_read_state", "age_days",
            "refresh_required", "refresh_reason", "stored_model_version", "parser_version",
            "rows", "asof", "error", "source_family",
        ]
        if not names:
            return pd.DataFrame(), pd.DataFrame(columns=audit_columns)
        if self.settings.mode != "SUPABASE_REST" or not self.settings.read_enabled:
            state = "READ_DISABLED" if not self.settings.read_enabled else self.settings.mode
            audit = pd.DataFrame([{
                "ticker": ticker, "provider": "SUPABASE_DATABASE_FIRST", "scope": payload_kind,
                "status": state, "database_read_state": state, "age_days": np.nan,
                "refresh_required": True, "rows": 0, "asof": "", "error": "",
                "source_family": "DATABASE",
            } for ticker in names], columns=audit_columns)
            return pd.DataFrame(), audit

        by_ticker: dict[str, dict[str, Any]] = {}
        error_text = ""
        try:
            for start in range(0, len(names), self.settings.read_batch_size):
                chunk = names[start:start + self.settings.read_batch_size]
                rows = self._get_rows(table, {
                    "select": "*",
                    "ticker": f"in.({','.join(chunk)})",
                    "limit": str(max(len(chunk) * 2, 20)),
                })
                for row in rows:
                    ticker = _clean_text(row.get("ticker"))
                    if ticker:
                        by_ticker[ticker] = row
        except DatabaseReadError as exc:
            error_text = str(exc)
            migration_missing = exc.status_code in {404, 400} and any(token in exc.body.upper() for token in ("PGRST205", "NOT FIND", "RELATION", "SCHEMA CACHE"))
            state = "MIGRATION_REQUIRED_V4" if migration_missing else "DATABASE_READ_FAIL_SOFT"
            audit = pd.DataFrame([{
                "ticker": ticker, "provider": "SUPABASE_DATABASE_FIRST", "scope": payload_kind,
                "status": state, "database_read_state": state, "age_days": np.nan,
                "refresh_required": True, "rows": 0, "asof": "", "error": error_text,
                "source_family": "DATABASE",
            } for ticker in names], columns=audit_columns)
            return pd.DataFrame(), audit
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {str(exc)[:500]}"
            audit = pd.DataFrame([{
                "ticker": ticker, "provider": "SUPABASE_DATABASE_FIRST", "scope": payload_kind,
                "status": "DATABASE_READ_FAIL_SOFT", "database_read_state": "DATABASE_READ_FAIL_SOFT",
                "age_days": np.nan, "refresh_required": True, "rows": 0, "asof": "",
                "error": error_text, "source_family": "DATABASE",
            } for ticker in names], columns=audit_columns)
            return pd.DataFrame(), audit

        output_rows: list[dict[str, Any]] = []
        audits: list[dict[str, Any]] = []
        for ticker in names:
            row = by_ticker.get(ticker)
            if not row:
                audits.append({
                    "ticker": ticker, "provider": "SUPABASE_DATABASE_FIRST", "scope": payload_kind,
                    "status": "DATABASE_MISS", "database_read_state": "DATABASE_MISS",
                    "age_days": np.nan, "refresh_required": True, "rows": 0, "asof": "",
                    "error": "", "source_family": "DATABASE",
                })
                continue
            checked_at = row.get("source_checked_at") or row.get("updated_at") or row.get("created_at")
            state, age_days = _freshness_state(checked_at, max_age_days, self.settings.stale_max_age_days)
            expired_by_age = state == "DATABASE_EXPIRED"
            refresh_reason = "AGE_POLICY" if state != "DATABASE_CURRENT" else ""
            stored_refresh_state = _clean_text(row.get("refresh_state")).upper()
            if not expired_by_age and stored_refresh_state in {"EVENT_DUE", "FORCE_REFRESH", "REVISION_DETECTED"}:
                state = "DATABASE_EVENT_DUE"
                refresh_reason = stored_refresh_state
            parser_reason = semantic_refresh_reason(row.get("parser_version"), MODEL_VERSIONS["fundamental_parser"])
            model_reason = semantic_refresh_reason(row.get("model_version"), MODEL_VERSIONS["fundamental"])
            if not expired_by_age and parser_reason:
                state = "DATABASE_MODEL_STALE"
                refresh_reason = parser_reason
            elif not expired_by_age and model_reason and model_reason in {"MODEL_MAJOR_CHANGED", "MODEL_MINOR_CHANGED"}:
                state = "DATABASE_MODEL_STALE"
                refresh_reason = model_reason
            next_check = _as_utc_timestamp(row.get("next_check_at"))
            now_stamp = _as_utc_timestamp(datetime.now(timezone.utc))
            if state == "DATABASE_CURRENT" and pd.notna(next_check) and pd.notna(now_stamp) and now_stamp >= next_check:
                state = "DATABASE_CHECK_DUE"
                refresh_reason = "NEXT_CHECK_AT_REACHED"
            payload = row.get("payload")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    payload = None
            loaded = 0
            if expired_by_age:
                # Rows beyond stale_max_age_days are audit evidence only.  They
                # must not flow into scoring as a provider-failure fallback.
                payload = None
                refresh_reason = "HARD_EXPIRED_NOT_LOADED"
            elif isinstance(payload, dict):
                local = dict(payload)
                local["ticker"] = ticker
                local["database_source_state"] = state
                local["database_source_checked_at"] = checked_at
                local.setdefault("fundamental_fetched_at", row.get("source_fetched_at") or checked_at)
                output_rows.append(local)
                loaded = 1
            elif isinstance(payload, list):
                for item in payload:
                    if isinstance(item, Mapping):
                        local = dict(item)
                        local["ticker"] = ticker
                        local["database_source_state"] = state
                        local["database_source_checked_at"] = checked_at
                        output_rows.append(local)
                        loaded += 1
            else:
                state = "DATABASE_PAYLOAD_INVALID"
            audits.append({
                "ticker": ticker, "provider": "SUPABASE_DATABASE_FIRST", "scope": payload_kind,
                "status": state, "database_read_state": state, "age_days": round(age_days, 2) if math.isfinite(age_days) else np.nan,
                "refresh_required": state != "DATABASE_CURRENT", "refresh_reason": refresh_reason,
                "stored_model_version": _clean_text(row.get("model_version")),
                "parser_version": _clean_text(row.get("parser_version")), "rows": loaded,
                "asof": _clean_text(checked_at),
                "error": (
                    "" if loaded else
                    "Payload kedaluwarsa ditolak dari scoring" if expired_by_age else
                    "Payload kosong/tidak valid"
                ),
                "source_family": "DATABASE",
            })
        return pd.DataFrame(output_rows), pd.DataFrame(audits, columns=audit_columns)

    def read_fundamental_cache(self, tickers: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self._read_cache_table(
            "fundamental_cache", tickers, self.settings.fundamental_max_age_days, "FUNDAMENTAL_SNAPSHOT",
        )

    def read_fundamental_history_cache(self, tickers: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self._read_cache_table(
            "fundamental_history_cache", tickers, self.settings.history_max_age_days, "FUNDAMENTAL_HISTORY",
        )

    def read_forward_quality_cache(self, tickers: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self._read_cache_table(
            "forward_quality_cache", tickers, self.settings.forward_max_age_days, "FORWARD_QUALITY",
        )

    def read_pending_event_tickers(self, tickers: Sequence[str]) -> list[str]:
        """Return symbols with unresolved material events requiring refresh."""
        names = list(dict.fromkeys(str(ticker) for ticker in tickers if str(ticker).strip()))
        if not names or self.settings.mode != "SUPABASE_REST" or not self.settings.read_enabled:
            return []
        pending: list[str] = []
        try:
            for start in range(0, len(names), self.settings.read_batch_size):
                chunk = names[start:start + self.settings.read_batch_size]
                rows = self._get_rows("source_events", {
                    "select": "ticker,event_type,event_date,event_fingerprint",
                    "ticker": f"in.({','.join(chunk)})",
                    "refresh_required": "eq.true",
                    "resolved_at": "is.null",
                    "order": "detected_at.desc",
                    "limit": str(max(20, len(chunk) * 4)),
                })
                pending.extend(_clean_text(row.get("ticker")) for row in rows if _clean_text(row.get("ticker")))
        except Exception:
            return []
        return list(dict.fromkeys(pending))

    def read_research_outcomes(self, tickers: Sequence[str], *, limit: int = 5000) -> pd.DataFrame:
        names = list(dict.fromkeys(str(ticker) for ticker in tickers if str(ticker).strip()))
        if not names or self.settings.mode != "SUPABASE_REST" or not self.settings.read_enabled:
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        try:
            for start in range(0, len(names), self.settings.read_batch_size):
                chunk = names[start:start + self.settings.read_batch_size]
                rows.extend(self._get_rows("research_outcomes", {
                    "select": "*", "ticker": f"in.({','.join(chunk)})",
                    "order": "signal_timestamp.desc", "limit": str(min(max(20, int(limit)), 5000)),
                }))
        except Exception:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def _read_persistent_outcomes(
        self,
        table: str,
        tickers: Sequence[str],
        *,
        order_column: str,
        limit: int = 10000,
    ) -> pd.DataFrame:
        names = list(dict.fromkeys(
            str(ticker).upper().strip() for ticker in tickers if str(ticker).strip()
        ))
        if not names or self.settings.mode != "SUPABASE_REST" or not self.settings.read_enabled:
            return pd.DataFrame()
        rows: list[dict[str, Any]] = []
        try:
            for start in range(0, len(names), self.settings.read_batch_size):
                chunk = names[start:start + self.settings.read_batch_size]
                rows.extend(self._get_rows(table, {
                    "select": "*",
                    "ticker": f"in.({','.join(chunk)})",
                    "order": f"{order_column}.desc",
                    "limit": str(min(max(20, int(limit)), 10000)),
                }))
        except Exception:
            return pd.DataFrame()
        restored: list[dict[str, Any]] = []
        for raw in rows:
            payload = raw.get("payload")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    payload = {}
            record = dict(payload) if isinstance(payload, Mapping) else {}
            record.update({
                key: value for key, value in raw.items()
                if key not in {"payload", "created_at", "updated_at"} and value is not None
            })
            restored.append(record)
        return pd.DataFrame(restored)

    def read_ai_execution_outcomes(
        self,
        tickers: Sequence[str],
        *,
        limit: int = 10000,
    ) -> pd.DataFrame:
        return self._read_persistent_outcomes(
            "ai_execution_outcomes", tickers,
            order_column="signal_date", limit=limit,
        )

    def read_selector_outcomes(
        self,
        tickers: Sequence[str],
        *,
        limit: int = 10000,
    ) -> pd.DataFrame:
        return self._read_persistent_outcomes(
            "selector_outcomes", tickers,
            order_column="signal_date", limit=limit,
        )

    def read_narrative_events(
        self,
        tickers: Sequence[str],
        *,
        limit: int = 10000,
    ) -> pd.DataFrame:
        return self._read_persistent_outcomes(
            "narrative_events", tickers,
            order_column="detected_at", limit=limit,
        )

    def read_narrative_event_outcomes(
        self,
        tickers: Sequence[str],
        *,
        limit: int = 10000,
    ) -> pd.DataFrame:
        return self._read_persistent_outcomes(
            "narrative_event_outcomes", tickers,
            order_column="signal_timestamp", limit=limit,
        )

    def read_idx_trading_calendar(self, start_date: Any, end_date: Any) -> pd.DataFrame:
        if self.settings.mode != "SUPABASE_REST" or not self.settings.read_enabled:
            return pd.DataFrame()
        start = pd.to_datetime(start_date, errors="coerce")
        end = pd.to_datetime(end_date, errors="coerce")
        if pd.isna(start) or pd.isna(end):
            return pd.DataFrame()
        try:
            rows = self._get_rows("idx_trading_calendar", {
                "select": "*", "trade_date": f"gte.{pd.Timestamp(start).date().isoformat()}",
                "order": "trade_date.asc", "limit": "1000",
            })
            frame = pd.DataFrame(rows)
            if not frame.empty and "trade_date" in frame:
                frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
                frame = frame[frame["trade_date"] <= pd.Timestamp(end)]
            return frame
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def refresh_tickers(audit: pd.DataFrame, tickers: Sequence[str]) -> list[str]:
        if audit is None or audit.empty or "ticker" not in audit:
            return list(dict.fromkeys(tickers))
        state_map = {
            str(row.get("ticker")): str(row.get("database_read_state", ""))
            for _, row in audit.iterrows()
        }
        return [ticker for ticker in dict.fromkeys(tickers) if state_map.get(str(ticker)) != "DATABASE_CURRENT"]

    def build_payloads(self, result: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
        as_of = datetime.now(timezone.utc).isoformat()
        focus = result.get("focus_screens", {}) if isinstance(result.get("focus_screens", {}), Mapping) else {}
        multibagger = focus.get("multibagger", pd.DataFrame())
        fundamentals = result.get("fundamentals", pd.DataFrame())
        fundamental_history = result.get("fundamental_history", pd.DataFrame())
        projects = result.get("project_management_review", pd.DataFrame())
        core = focus.get("core_swing", pd.DataFrame())
        selector = focus.get("stock_selector", pd.DataFrame())
        selector_snapshots = selector_snapshot_frame(
            selector if isinstance(selector, pd.DataFrame) else pd.DataFrame()
        )
        selector_audit = focus.get("selector_model_audit", pd.DataFrame())
        selector_outcomes = result.get(
            "selector_outcomes", focus.get("selector_outcomes", pd.DataFrame()),
        )
        narrative_events = result.get(
            "narrative_events", focus.get("narrative_events", pd.DataFrame()),
        )
        narrative_outcomes = result.get(
            "narrative_event_outcomes",
            focus.get("narrative_event_outcomes", pd.DataFrame()),
        )
        narrative_profiles = result.get(
            "narrative_profiles",
            focus.get("narrative_profiles", pd.DataFrame()),
        )
        ai_execution_outcomes = result.get(
            "ai_outcome_memory", focus.get("ai_outcome_memory", pd.DataFrame()),
        )
        ihsg_snapshots = ihsg_snapshot_frame(result.get("ihsg_direction"))
        provider_report = _provider_health_frame(result)
        scan_id = _clean_text(result.get("scan_id")) or hashlib.sha256(
            f"{as_of}|{_clean_text(result.get('scanner_version'))}".encode("utf-8")
        ).hexdigest()[:24]
        model_version = _clean_text(result.get("scanner_version")) or "7.6.0"
        selector_evaluations: list[dict[str, Any]] = []
        if isinstance(selector_audit, pd.DataFrame) and not selector_audit.empty:
            evaluation_date = pd.Timestamp(as_of).date().isoformat()
            for row in selector_audit.to_dict("records"):
                evaluation_id = hashlib.sha256(
                    "|".join([
                        evaluation_date,
                        _clean_text(row.get("horizon")),
                        _clean_text(row.get("model")),
                        _clean_text(row.get("selector_version")),
                    ]).encode("utf-8")
                ).hexdigest()[:32]
                local = dict(row)
                local.update({
                    "evaluation_id": evaluation_id,
                    "as_of": as_of,
                    "payload": {str(key): _json_safe(value) for key, value in row.items()},
                })
                selector_evaluations.append(local)

        payloads: dict[str, list[dict[str, Any]]] = {
            "fundamental_snapshots": _frame_records(fundamentals, (
                "ticker", "period_end", "statement_date", "fundamental_score", "fundamental_score_10",
                "fundamental_coverage", "fundamental_data_grade", "fundamental_reliability",
                "revenue_growth", "earnings_growth", "roe", "roa", "roic_proxy", "net_margin",
                "operating_margin", "operating_cash_flow", "free_cash_flow", "cash_conversion_ttm",
                "debt_equity", "net_debt_ebitda", "interest_coverage", "market_cap",
                "fundamental_source_families", "fundamental_official_verified", "statement_age_days",
                "fundamental_fetched_at", "database_source_state", "content_hash",
            )),
            "multibagger_snapshots": _frame_records(multibagger, (
                "ticker", "multibagger_status", "multibagger_quality_score", "execution_readiness_score",
                "research_recommendation_status", "multibagger_candidate_type",
                "multibagger_lane", "research_eligible",
                "research_eligibility_reason", "multibagger_scoring_state",
                "portfolio_allocation_eligible", "multibagger_evidence_class",
                "multibagger_rank_eligible", "multibagger_score_comparability_pct",
                "multibagger_production_rank",
                "multibagger_metric_coverage_pct",
                "multibagger_metric_data_gate",
                "growth_pillar_coverage_pct",
                "profitability_pillar_coverage_pct",
                "cashflow_pillar_coverage_pct",
                "safety_pillar_coverage_pct",
                "runway_pillar_coverage_pct",
                "valuation_pillar_coverage_pct",
                "growth_compounder_score", "growth_compounder_base_score",
                "growth_compounder_selection_score",
                "turnaround_recovery_score", "confidence_adjusted_turnaround_score",
                "turnaround_selection_score", "turnaround_research_state",
                "turnaround_recovery_signals", "turnaround_gate_reasons",
                "critical_research_flags", "operational_recovery_flags",
                "effective_silent_accumulation_score", "economic_earnings_score",
                "economic_earnings_confidence", "minority_leakage_pct", "ocf_ebitda_conversion",
                "silent_accumulation_score", "silent_accumulation_raw_score", "silent_accumulation_state", "silent_accumulation_confidence",
                "silent_accumulation_liquidity_adjustment", "silent_accumulation_liquidity_min_confirmation",
                "silent_accumulation_calibration_policy", "liquidity_bucket",
                "silent_accumulation_v4_adjustment", "accumulation_persistence_score",
                "accumulation_positive_windows_pct", "accumulation_longest_run", "accumulation_regime",
                "absorption_confirmed_days20", "failed_absorption_days20", "effort_result_absorption20",
                "effort_result_distribution20", "persistent_bid_score", "data_confidence_score",
                "fundamental_confidence_score", "future_fundamental_confidence_score",
                "technical_confidence_score", "eoff_confidence_score", "overall_research_confidence",
                "overall_research_confidence_grade", "confidence_adjusted_multibagger_score",
                "growth_persistence_pillar", "profitability_pillar",
                "cash_conversion_pillar", "balance_sheet_safety_pillar",
                "reinvestment_runway_pillar", "quality_pillar_coverage_pct",
                "quality_pillars_strong", "quality_pillars_critical",
                "quality_pillar_gate", "quality_pillar_a_gate",
                "gross_profitability", "gross_margin", "gross_profit_growth",
                "accruals_to_assets", "leverage_change_yoy",
                "sector", "sector_peer_count", "sector_relative_quality_score",
                "sector_relative_state", "time_cycle_evaluation_mode",
                "top_positive_drivers", "top_negative_drivers", "scoring_reason_codes",
                "project_pipeline_score", "project_stage", "project_stage_probability_pct",
                "project_success_probability_pct", "future_fundamental_impact_score", "best_buy_date",
                "best_buy_window_start", "best_buy_window_end", "eoff_state", "eoff_reconstruction_score",
                "eoff_public_validation_state", "eoff_public_lift", "last_price",
                "narrative_evidence_coverage_pct", "issuer_alignment_coverage_pct",
                "emir_method_coverage_pct", "distribution_severity_score",
                "distribution_penalty_points", "distribution_evidence_state",
            )),
            "project_events": _frame_records(projects, (
                "ticker", "project_name", "project_names", "project_stage", "project_completion_pct",
                "project_funding_secured_pct", "project_ownership_pct", "project_capex_idr",
                "project_expected_revenue_idr", "project_expected_ebitda_idr", "project_data_coverage",
                "project_source_families", "project_source_urls", "project_source_quorum_verified",
                "project_execution_flags", "last_verified_at", "review_origin", "event_date",
            )),
            "technical_snapshots": _frame_records(
                multibagger if isinstance(multibagger, pd.DataFrame) and not multibagger.empty else core,
                ("ticker", "last_price", "active_setup", "technical_entry_state", "entry", "stop_loss",
                 "tp1", "tp2", "execution_readiness_score", "silent_accumulation_score",
                 "silent_accumulation_confidence", "accumulation_persistence_score", "accumulation_regime",
                 "relative_strength60", "roc60", "roc120", "adtv20_idr"),
            ),
            "eoff_predictions": _frame_records(multibagger, (
                "ticker", "best_buy_date", "best_buy_raw_date", "best_buy_calendar_state",
                "best_buy_calendar_verified", "best_buy_date_adjustment_days",
                "best_buy_window_start", "best_buy_window_end",
                "best_buy_score", "best_buy_confidence", "best_buy_entry_low", "best_buy_entry_high",
                "best_buy_trigger", "eoff_state", "eoff_reconstruction_score", "eoff_strength_label",
                "eoff_direction_bias", "eoff_public_validation_state", "eoff_public_directional_events",
                "eoff_public_lift", "eoff_fib_unique_anchor_count", "eoff_fib_unique_anchor_ratio",
                "eoff_fib_dominant_anchor_share", "eoff_unique_anchor_gate",
                "eoff_unique_anchor_signature", "time_cycle_confidence",
            )),
            "ihsg_direction_snapshots": _frame_records(ihsg_snapshots, (
                "ticker", "as_of", "horizon", "horizon_bars", "raw_direction",
                "prediction_state", "prob_up_pct", "prob_sideways_pct", "prob_down_pct",
                "confidence_pct", "expected_return_pct", "return_p25_pct", "return_p75_pct",
                "neutral_band_pct", "analogue_count", "effective_analogue_count",
                "features_used", "median_distance", "validation_state",
                "validation_predictions", "directional_validation_predictions",
                "directional_accuracy_pct", "directional_accuracy_ci_low_pct",
                "validation_coverage_pct", "brier_score", "baseline_brier_score",
                "brier_skill_pct", "actionable", "data_state", "eod_final",
                "benchmark_close", "regime", "regime_score", "consensus_direction",
                "consensus_confidence", "risk_budget_multiplier", "risk_action",
                "feature_coverage_pct", "breadth_member_count", "breadth_ema50_pct",
                "feature_hash", "payload", "model_version",
            )),
            "provider_health": _frame_records(provider_report, (
                "provider", "scope", "status", "rows", "asof", "error", "error_code", "source_family",
            )),
            "scan_runs": [{
                "scan_id": scan_id,
                "started_at": _json_safe(result.get("scan_started_at", as_of)),
                "finished_at": _json_safe(result.get("scan_finished_at", as_of)),
                "ticker_count": int(result.get("ticker_count", len(result.get("prepared", {})) if isinstance(result.get("prepared", {}), Mapping) else 0) or 0),
                "prepared_count": int(len(result.get("prepared", {})) if isinstance(result.get("prepared", {}), Mapping) else 0),
                "multibagger_count": int(len(multibagger)) if isinstance(multibagger, pd.DataFrame) else 0,
                "core_swing_count": int(len(core)) if isinstance(core, pd.DataFrame) else 0,
                "database_mode": self.settings.mode,
            }],
            "fundamental_cache": [],
            "fundamental_history_cache": [],
            "forward_quality_cache": [],
            "refresh_state": [],
            "scan_checkpoints": [{
                "checkpoint_id": hashlib.sha256(f"{scan_id}|SCAN_COMPLETE|0".encode("utf-8")).hexdigest(),
                "scan_id": scan_id,
                "phase": "SCAN_COMPLETE",
                "batch_number": 0,
                "last_ticker": (
                    list(result.get("prepared", {}).keys())[-1]
                    if isinstance(result.get("prepared", {}), Mapping) and result.get("prepared", {}) else ""
                ),
                "status": "COMPLETE",
                "rows_completed": int(len(result.get("prepared", {}))) if isinstance(result.get("prepared", {}), Mapping) else 0,
                "rows_total": int(result.get("ticker_count", len(result.get("prepared", {})) if isinstance(result.get("prepared", {}), Mapping) else 0) or 0),
                "detail": "Final durable checkpoint. Intermediate resume checkpoints are reserved for the next batch executor.",
                "as_of": as_of,
            }],
            "model_registry": model_registry_frame(as_of).to_dict("records"),
            "source_events": [],
            "research_outcomes": _frame_records(result.get("research_outcomes", pd.DataFrame()), (
                "outcome_id", "ticker", "signal_family", "signal_timestamp", "signal_date", "anchor_id",
                "liquidity_bucket", "predicted_state", "predicted_direction", "signal_score",
                "signal_confidence", "prediction_window_start", "prediction_window_end", "entry_reference",
                "horizon_bars", "outcome_status", "resolved_at", "actual_low_date", "actual_high_date",
                "forward_return_1d", "forward_return_5d", "forward_return_10d", "forward_return_20d",
                "maximum_favourable_excursion", "maximum_adverse_excursion", "hit", "payload",
                "model_version",
            )),
            "backfill_state": [],
            "idx_trading_calendar": _frame_records(result.get("idx_trading_calendar", pd.DataFrame()), (
                "trade_date", "is_open", "session_type", "source_family", "source_url",
                "content_hash", "verified_at", "notes",
            )),
            "ai_execution_outcomes": _frame_records_with_payload(
                ai_execution_outcomes if isinstance(ai_execution_outcomes, pd.DataFrame) else pd.DataFrame(),
                (
                    "signal_id", "ticker", "strategy", "signal_date", "memory_state",
                    "result", "fill_date", "exit_date", "resolved_at", "outcome_quality",
                    "no_fill_reason", "entry", "trigger_price", "stop_loss", "tp1", "tp2",
                    "fill_price", "exit_price", "r_multiple", "expectancy_after_cost_r",
                    "mfe_r", "mae_r", "mfe_pct", "mae_pct", "gross_return_pct",
                    "net_return_pct", "roundtrip_cost_pct", "cost_r",
                    "fill_delay_bars", "fill_slippage_pct", "filled", "tp1_hit",
                    "tp1_before_sl", "tp2_hit", "outcome_ambiguous", "ai_version",
                    "model_version",
                ),
            ),
            "selector_snapshots": _frame_records_with_payload(
                selector_snapshots,
                (
                    "snapshot_id", "ticker", "as_of", "horizon", "horizon_bars",
                    "selection_rank", "production_selection_rank",
                    "selector_rank_eligible", "selector_data_state",
                    "technical_feature_coverage_pct",
                    "selector_missing_feature_count",
                    "selector_missing_features", "swing_selection_score",
                    "multibagger_timing_selector_score", "technical_selection_score",
                    "silent_accumulation_score",
                    "effective_silent_accumulation_score",
                    "silent_accumulation_confidence",
                    "relative_strength_score",
                    "expected_excess_return_pct", "outperform_probability_pct",
                    "selector_score", "ai_weight_pct", "relative_overlay_weight_pct",
                    "selector_universe_state", "score_inflation_guard_active", "model_state",
                    "champion_model", "selected_reason", "selection_risks",
                    "model_version",
                ),
            ),
            "selector_outcomes": _frame_records_with_payload(
                selector_outcomes if isinstance(selector_outcomes, pd.DataFrame) else pd.DataFrame(),
                (
                    "outcome_id", "snapshot_id", "ticker", "signal_date", "horizon",
                    "horizon_bars", "predicted_excess_return_pct",
                    "outperform_probability_pct", "selector_score", "model_state",
                    "champion_model", "outcome_status", "resolved_at",
                    "stock_return_pct", "benchmark_return_pct", "net_excess_return_pct",
                    "outperformed_after_cost", "model_version",
                ),
            ),
            "selector_model_evaluations": selector_evaluations,
            "narrative_events": _frame_records_with_payload(
                narrative_events
                if isinstance(narrative_events, pd.DataFrame)
                else pd.DataFrame(),
                (
                    "narrative_event_id", "ticker", "event_date",
                    "detected_at", "event_type", "event_family",
                    "headline", "summary", "source_url", "source_hostname",
                    "registered_official_domain", "source_state",
                    "source_present", "source_family",
                    "source_quality_score", "official_claimed", "official_verified",
                    "materiality_score", "impact_direction", "impact_sign",
                    "financial_bridge_score", "content_hash",
                    "event_cluster_key", "detection_time_source",
                    "entity_match_state", "event_status",
                    "requested_event_status", "lifecycle_evidence_state",
                    "resolved_at", "supersedes_event_id",
                    "resolution_source_url",
                    "novelty_score", "event_age_days",
                    "narrative_decay_weight", "catalyst_proximity_score",
                    "event_active", "event_evidence_state",
                    "future_detection_invalid",
                    "event_strength_score", "signed_event_strength",
                    "narrative_engine_version",
                ),
            ),
            "narrative_event_outcomes": _frame_records_with_payload(
                narrative_outcomes
                if isinstance(narrative_outcomes, pd.DataFrame)
                else pd.DataFrame(),
                (
                    "narrative_outcome_id", "narrative_event_id",
                    "ticker", "event_type", "event_family", "impact_sign",
                    "impact_direction", "entry_policy",
                    "signal_timestamp", "signal_date", "anchor_date",
                    "entry_reference", "roundtrip_cost_pct",
                    "stock_return_5d_pct", "benchmark_return_5d_pct",
                    "net_excess_return_5d_pct", "mfe_5d_pct",
                    "mae_5d_pct", "directional_excess_return_5d_pct",
                    "converted_5d",
                    "stock_return_20d_pct", "benchmark_return_20d_pct",
                    "net_excess_return_20d_pct", "mfe_20d_pct",
                    "mae_20d_pct", "directional_excess_return_20d_pct",
                    "converted_20d",
                    "stock_return_60d_pct", "benchmark_return_60d_pct",
                    "net_excess_return_60d_pct", "mfe_60d_pct",
                    "mae_60d_pct", "directional_excess_return_60d_pct",
                    "converted_60d", "outcome_status",
                    "resolved_at", "narrative_engine_version",
                ),
            ),
            "narrative_snapshots": _frame_records_with_payload(
                narrative_profiles
                if isinstance(narrative_profiles, pd.DataFrame)
                else pd.DataFrame(),
                (
                    "ticker", "narrative_as_of", "narrative_state",
                    "narrative_score", "narrative_effective_score",
                    "narrative_evidence_coverage_pct",
                    "narrative_event_count",
                    "narrative_active_event_count",
                    "narrative_missing_source_event_count",
                    "narrative_inactive_lifecycle_event_count",
                    "narrative_entity_unverified_event_count",
                    "narrative_event_cluster_count",
                    "narrative_corroborated_cluster_count",
                    "narrative_positive_event_count",
                    "narrative_negative_event_count",
                    "narrative_official_event_count",
                    "latest_narrative_event",
                    "latest_narrative_event_type",
                    "latest_narrative_event_date",
                    "narrative_source_quality_score",
                    "narrative_novelty_score",
                    "narrative_financial_bridge_score",
                    "issuer_alignment_score",
                    "issuer_alignment_effective_score",
                    "issuer_alignment_coverage_pct",
                    "issuer_alignment_state",
                    "issuer_alignment_positive_events",
                    "issuer_alignment_negative_events",
                    "retail_adoption_stage",
                    "retail_adoption_proxy_score",
                    "retail_adoption_proxy_coverage_pct",
                    "retail_proxy_disclaimer",
                    "narrative_crowding_risk_score",
                    "narrative_flow_convergence_score",
                    "narrative_flow_effective_score",
                    "narrative_flow_convergence_coverage_pct",
                    "narrative_flow_convergence_state",
                    "narrative_silent_integration_state",
                    "narrative_contradiction_count",
                    "narrative_hard_block",
                    "narrative_primary_reason",
                    "narrative_primary_risk",
                    "narrative_overlay_reliability_pct",
                    "narrative_swing_overlay_reliability_pct",
                    "narrative_growth_rank_adjustment",
                    "narrative_turnaround_rank_adjustment",
                    "narrative_swing_rank_adjustment",
                    "narrative_news_collection_state",
                    "narrative_production_policy",
                    "narrative_items_reviewed",
                    "narrative_flow_proxy_score",
                    "narrative_conversion_rate_5d_pct",
                    "narrative_conversion_effective_5d_score",
                    "narrative_conversion_expectancy_5d_pct",
                    "narrative_conversion_resolved_5d",
                    "narrative_conversion_state_5d",
                    "narrative_conversion_rate_20d_pct",
                    "narrative_conversion_effective_20d_score",
                    "narrative_conversion_expectancy_20d_pct",
                    "narrative_conversion_resolved_20d",
                    "narrative_conversion_state_20d",
                    "narrative_conversion_rate_60d_pct",
                    "narrative_conversion_effective_60d_score",
                    "narrative_conversion_expectancy_60d_pct",
                    "narrative_conversion_resolved_60d",
                    "narrative_conversion_state_60d",
                    "narrative_engine_version",
                ),
            ),
        }

        if isinstance(fundamentals, pd.DataFrame) and not fundamentals.empty and "ticker" in fundamentals:
            for _, row in fundamentals.dropna(subset=["ticker"]).drop_duplicates("ticker", keep="last").iterrows():
                payload = {str(key): _json_safe(value) for key, value in row.to_dict().items()}
                ticker = _clean_text(payload.get("ticker"))
                if not ticker:
                    continue
                checked_at = payload.get("database_source_checked_at") or payload.get("fundamental_fetched_at") or as_of
                content_hash = _semantic_hash(payload)
                statement_date = payload.get("latest_statement_date") or payload.get("statement_date")
                event_fingerprint = _stable_hash({"ticker": ticker, "statement_date": statement_date, "content_hash": content_hash})
                next_check_at = (pd.to_datetime(checked_at, errors="coerce", utc=True) + timedelta(days=self.settings.fundamental_max_age_days))
                next_check_at = next_check_at.isoformat() if pd.notna(next_check_at) else as_of
                payloads["fundamental_cache"].append({
                    "ticker": ticker, "payload": payload,
                    "source_families": payload.get("fundamental_source_families"),
                    "data_grade": payload.get("fundamental_data_grade"),
                    "coverage": payload.get("fundamental_coverage"),
                    "statement_date": statement_date,
                    "source_fetched_at": payload.get("fundamental_fetched_at") or checked_at,
                    "source_checked_at": checked_at,
                    "content_hash": content_hash,
                    "refresh_state": "CURRENT", "parser_version": MODEL_VERSIONS["fundamental_parser"],
                    "event_fingerprint": event_fingerprint, "next_check_at": next_check_at,
                    "refresh_reason": _clean_text(payload.get("database_source_state")) or "PROVIDER_OR_DATABASE_CURRENT",
                    "model_version": MODEL_VERSIONS["fundamental"],
                })
                payloads["source_events"].append({
                    "event_key": hashlib.sha256(f"FUNDAMENTAL_PERIOD|{ticker}|{statement_date}|{content_hash}".encode("utf-8")).hexdigest(),
                    "ticker": ticker, "event_type": "FUNDAMENTAL_PERIOD", "event_date": statement_date,
                    "source_family": payload.get("fundamental_source_families"), "content_hash": content_hash,
                    "event_fingerprint": event_fingerprint, "refresh_required": False,
                    "detected_at": checked_at, "last_seen_at": as_of,
                    "payload": {"coverage": payload.get("fundamental_coverage"), "grade": payload.get("fundamental_data_grade")},
                })
                payloads["refresh_state"].append({
                    "entity_key": f"FUNDAMENTAL|{ticker}", "entity_type": "FUNDAMENTAL",
                    "ticker": ticker, "source_family": payload.get("fundamental_source_families"),
                    "state": "CURRENT", "last_checked_at": checked_at,
                    "last_changed_at": checked_at, "content_hash": content_hash,
                    "detail": "Complete normalised fundamental row persisted",
                    "parser_version": MODEL_VERSIONS["fundamental_parser"], "event_fingerprint": event_fingerprint,
                    "refresh_reason": _clean_text(payload.get("database_source_state")) or "CURRENT",
                    "payload_metadata": {"coverage": payload.get("fundamental_coverage"), "grade": payload.get("fundamental_data_grade")},
                })

        if isinstance(fundamental_history, pd.DataFrame) and not fundamental_history.empty and "ticker" in fundamental_history:
            for ticker, group in fundamental_history.groupby("ticker", sort=False):
                local = group.copy()
                period_column = "period_end" if "period_end" in local else None
                if period_column:
                    local[period_column] = pd.to_datetime(local[period_column], errors="coerce")
                    local = local.sort_values(period_column).tail(80)
                records = [{str(key): _json_safe(value) for key, value in row.items()} for row in local.to_dict("records")]
                source_families = "|".join(sorted(set(local.get("source_family", pd.Series(dtype=str)).dropna().astype(str))))
                checked_series = pd.to_datetime(local.get("database_source_checked_at", pd.Series(dtype=object)), errors="coerce", utc=True)
                checked_at = checked_series.dropna().max().isoformat() if not checked_series.dropna().empty else as_of
                latest_period = ""
                if period_column and local[period_column].notna().any():
                    latest_period = pd.Timestamp(local[period_column].dropna().max()).date().isoformat()
                content_hash = _semantic_hash(records)
                payloads["fundamental_history_cache"].append({
                    "ticker": str(ticker), "payload": records, "latest_period": latest_period,
                    "period_count": len(records), "source_families": source_families,
                    "source_checked_at": checked_at, "content_hash": content_hash,
                    "refresh_state": "CURRENT", "parser_version": MODEL_VERSIONS["fundamental_parser"],
                    "event_fingerprint": _stable_hash({"ticker": str(ticker), "latest_period": latest_period, "content_hash": content_hash}),
                    "next_check_at": (pd.to_datetime(checked_at, errors="coerce", utc=True) + timedelta(days=self.settings.history_max_age_days)).isoformat(),
                    "refresh_reason": "ROUND_ROBIN_OR_EVENT_REFRESH",
                    "model_version": MODEL_VERSIONS["fundamental"],
                })
                payloads["refresh_state"].append({
                    "entity_key": f"FUNDAMENTAL_HISTORY|{ticker}", "entity_type": "FUNDAMENTAL_HISTORY",
                    "ticker": str(ticker), "source_family": source_families,
                    "state": "CURRENT", "last_checked_at": checked_at,
                    "last_changed_at": checked_at, "content_hash": content_hash,
                    "detail": f"{len(records)} normalised history rows persisted",
                    "parser_version": MODEL_VERSIONS["fundamental_parser"],
                    "event_fingerprint": _stable_hash({"ticker": str(ticker), "latest_period": latest_period, "content_hash": content_hash}),
                    "refresh_reason": "ROUND_ROBIN_OR_EVENT_REFRESH",
                    "payload_metadata": {"period_count": len(records), "latest_period": latest_period},
                })

        if isinstance(projects, pd.DataFrame) and not projects.empty and "ticker" in projects:
            for ticker, group in projects.groupby("ticker", sort=False):
                records = [{str(key): _json_safe(value) for key, value in row.items()} for row in group.tail(40).to_dict("records")]
                source_families = "|".join(sorted(set(group.get("project_source_families", pd.Series(dtype=str)).dropna().astype(str))))
                last_verified = pd.to_datetime(group.get("last_verified_at", pd.Series(dtype=object)), errors="coerce", utc=True).dropna()
                checked_at = last_verified.max().isoformat() if not last_verified.empty else as_of
                content_hash = _semantic_hash(records)
                payloads["forward_quality_cache"].append({
                    "ticker": str(ticker), "payload": records, "project_count": len(records),
                    "source_families": source_families, "last_verified_at": checked_at,
                    "source_checked_at": checked_at, "content_hash": content_hash,
                    "refresh_state": "CURRENT", "parser_version": MODEL_VERSIONS["fundamental_parser"],
                    "event_fingerprint": _stable_hash({"ticker": str(ticker), "content_hash": content_hash}),
                    "next_check_at": (pd.to_datetime(checked_at, errors="coerce", utc=True) + timedelta(days=self.settings.forward_max_age_days)).isoformat(),
                    "refresh_reason": "EVENT_AWARE_FORWARD_REFRESH",
                    "model_version": MODEL_VERSIONS["fundamental"],
                })
                project_event_fingerprint = _stable_hash({"ticker": str(ticker), "content_hash": content_hash})
                payloads["refresh_state"].append({
                    "entity_key": f"FORWARD_QUALITY|{ticker}", "entity_type": "FORWARD_QUALITY",
                    "ticker": str(ticker), "source_family": source_families,
                    "state": "CURRENT", "last_checked_at": checked_at,
                    "last_changed_at": checked_at, "content_hash": content_hash,
                    "detail": f"{len(records)} forward-quality rows persisted",
                    "parser_version": MODEL_VERSIONS["fundamental_parser"],
                    "event_fingerprint": project_event_fingerprint, "refresh_reason": "EVENT_AWARE_FORWARD_REFRESH",
                    "payload_metadata": {"project_count": len(records)},
                })
                for project in records:
                    project_name = _clean_text(project.get("project_name") or project.get("project_names"))
                    project_stage = _clean_text(project.get("project_stage"))
                    event_date = _clean_text(project.get("event_date") or project.get("last_verified_at"))[:10]
                    event_hash = _stable_hash({"ticker": str(ticker), "name": project_name, "stage": project_stage, "event_date": event_date})
                    payloads["source_events"].append({
                        "event_key": hashlib.sha256(f"PROJECT_STAGE|{ticker}|{project_name}|{project_stage}|{event_date}".encode("utf-8")).hexdigest(),
                        "ticker": str(ticker), "event_type": "PROJECT_STAGE", "event_date": event_date,
                        "source_family": source_families, "content_hash": event_hash,
                        "event_fingerprint": event_hash, "refresh_required": False,
                        "detected_at": checked_at, "last_seen_at": as_of, "payload": project,
                    })

        for report_key, entity_type, scope_filter in (
            ("database_read_report", "FUNDAMENTAL_SNAPSHOT", "FUNDAMENTAL_SNAPSHOT"),
            ("fundamental_history_report", "FUNDAMENTAL_HISTORY", "FUNDAMENTAL_HISTORY"),
            ("automatic_forward_report", "FORWARD_QUALITY", "FORWARD_QUALITY"),
        ):
            report = result.get(report_key, pd.DataFrame())
            if isinstance(report, pd.DataFrame) and not report.empty and "provider" in report:
                scheduled = report[report["provider"].astype(str).eq("ROUND_ROBIN_BACKFILL")]
                if scope_filter and "scope" in scheduled:
                    scheduled = scheduled[scheduled["scope"].astype(str).eq(scope_filter)]
                for _, row in scheduled.iterrows():
                    ticker = _clean_text(row.get("ticker"))
                    if not ticker:
                        continue
                    selected = bool(row.get("selected_for_refresh"))
                    payloads["backfill_state"].append({
                        "entity_key": f"{entity_type}|{ticker}", "ticker": ticker, "entity_type": entity_type,
                        "cohort": int(_finite(row.get("ticker_cohort"), 0.0)),
                        "active_cohort": int(_finite(row.get("active_cohort"), 0.0)),
                        "priority": 1 if _clean_text(row.get("refresh_reason")) in {"PRIORITY_TICKER", "EVENT_TRIGGERED"} else 5,
                        "status": "SELECTED" if selected else "DEFERRED",
                        "refresh_reason": _clean_text(row.get("refresh_reason")),
                        "last_attempt_at": as_of if selected else None,
                        "next_due_at": (pd.Timestamp(as_of) + timedelta(days=1)).isoformat(),
                        "failure_count": 0, "payload": {"database_read_state": row.get("database_read_state")},
                    })

        for table, records in payloads.items():
            prepared: list[dict[str, Any]] = []
            row_limit = self.settings.cache_max_rows_per_table if table in {
                "fundamental_cache", "fundamental_history_cache", "forward_quality_cache", "refresh_state",
                "research_outcomes", "source_events", "backfill_state", "model_registry", "idx_trading_calendar",
                "ai_execution_outcomes", "selector_snapshots", "selector_outcomes",
                "selector_model_evaluations", "narrative_events",
                "narrative_event_outcomes", "narrative_snapshots",
            } else self.settings.max_rows_per_table
            for record in records[: row_limit]:
                local = dict(record)
                if not _clean_text(local.get("model_version")):
                    local["model_version"] = model_version
                local["schema_version"] = DATABASE_SCHEMA_VERSION
                if table not in TABLE_CONFLICT_TARGETS:
                    local["as_of"] = as_of
                    local["scan_id"] = local.get("scan_id") or scan_id
                    local["snapshot_id"] = _snapshot_id(table, local, as_of)
                normalised = _normalise_record(table, local)
                if normalised is not None:
                    prepared.append(normalised)
            conflict_key = TABLE_CONFLICT_TARGETS.get(table, "snapshot_id")
            deduped = {record[conflict_key]: record for record in prepared if conflict_key in record}
            payloads[table] = list(deduped.values())
        return payloads

    def persist_scan_result(self, result: Mapping[str, Any]) -> pd.DataFrame:
        if self.settings.mode == "DISABLED":
            return pd.DataFrame([self.status_row("DISABLED_NO_DATABASE", "Scanner tetap berjalan dengan cache lokal; database eksternal belum diaktifkan.")])
        if self.settings.mode == "CONFIG_INCOMPLETE":
            return pd.DataFrame([self.status_row("CONFIG_INCOMPLETE", "Aktifkan OUTBOX_ONLY atau isi SUPABASE_URL dan SUPABASE_SECRET_KEY.")])
        if self.settings.mode == "CONFIG_UNSAFE_KEY":
            return pd.DataFrame([self.status_row("CONFIG_UNSAFE_KEY", "Reader/writer database membutuhkan backend secret/service-role key; publishable/anon key ditolak.")])
        payloads = self.build_payloads(result)
        rows: list[dict[str, Any]] = []
        for table, records in payloads.items():
            if not records:
                row = self.status_row("NO_ROWS")
                row.update({"table": table})
                rows.append(row)
                continue
            try:
                written = self._write_outbox(table, records) if self.settings.mode == "OUTBOX_ONLY" else self._upsert_supabase(table, records)
                detail = self._write_details.get(table, "")
                state = "OK" if written == len(records) else ("PARTIAL_WRITE" if written > 0 else "DATABASE_FAIL_SOFT")
                row = self.status_row(state, detail)
                row.update({"table": table, "rows_attempted": len(records), "rows_written": written})
            except Exception as exc:
                row = self.status_row("DATABASE_FAIL_SOFT", f"{type(exc).__name__}: {str(exc)[:700]}")
                row.update({"table": table, "rows_attempted": len(records), "rows_written": 0})
            rows.append(row)
        return pd.DataFrame(rows)

    def health_check(self) -> dict[str, Any]:
        if self.settings.mode != "SUPABASE_REST":
            return self.status_row(self.settings.mode, "Health check requires SUPABASE_REST mode.")
        try:
            self._get_rows("scan_runs", {"select": "snapshot_id", "limit": "1"})
            try:
                self._get_rows("fundamental_cache", {"select": "ticker", "limit": "1"})
            except DatabaseReadError as exc:
                migration_missing = exc.status_code in {404, 400} and any(token in exc.body.upper() for token in ("PGRST205", "RELATION", "SCHEMA CACHE", "NOT FIND"))
                if migration_missing:
                    return self.status_row("MIGRATION_REQUIRED_V3", "Schema v2 tersambung, tetapi migration_v3_database_first.sql belum diterapkan.")
                raise
            try:
                self._get_rows("research_outcomes", {"select": "outcome_id", "limit": "1"})
                self._get_rows("model_registry", {"select": "component", "limit": "1"})
                self._get_rows("idx_trading_calendar", {"select": "trade_date", "limit": "1"})
            except DatabaseReadError as exc:
                migration_missing = exc.status_code in {404, 400} and any(token in exc.body.upper() for token in ("PGRST205", "RELATION", "SCHEMA CACHE", "NOT FIND"))
                if migration_missing:
                    return self.status_row("MIGRATION_REQUIRED_V4", "Migration v3 tersedia, tetapi migration_v4_research_memory.sql belum diterapkan.")
                raise
            try:
                self._get_rows("ihsg_direction_snapshots", {"select": "snapshot_id", "limit": "1"})
                self._get_rows("research_outcomes", {"select": "outcome_id,forward_return_1d", "limit": "1"})
            except DatabaseReadError as exc:
                migration_missing = exc.status_code in {404, 400} and any(
                    token in exc.body.upper()
                    for token in ("PGRST204", "PGRST205", "RELATION", "COLUMN", "SCHEMA CACHE", "NOT FIND")
                )
                if migration_missing:
                    return self.status_row(
                        "MIGRATION_REQUIRED_V5",
                        "Migration v4 tersedia, tetapi migration_v5_ihsg_direction.sql belum diterapkan.",
                    )
                raise
            try:
                self._get_rows("ai_execution_outcomes", {"select": "signal_id", "limit": "1"})
                self._get_rows("selector_snapshots", {"select": "snapshot_id", "limit": "1"})
                self._get_rows("selector_outcomes", {"select": "outcome_id", "limit": "1"})
                self._get_rows("selector_model_evaluations", {"select": "evaluation_id", "limit": "1"})
            except DatabaseReadError as exc:
                migration_missing = exc.status_code in {404, 400} and any(
                    token in exc.body.upper()
                    for token in ("PGRST204", "PGRST205", "RELATION", "COLUMN", "SCHEMA CACHE", "NOT FIND")
                )
                if migration_missing:
                    return self.status_row(
                        "MIGRATION_REQUIRED_V6",
                        "Migration v5 tersedia, tetapi migration_v6_selector_ai_outcomes.sql belum diterapkan.",
                    )
                raise
            try:
                self._get_rows(
                    "multibagger_snapshots",
                    {
                        "select": (
                            "snapshot_id,multibagger_lane,research_eligible,"
                            "portfolio_allocation_eligible,"
                            "growth_compounder_selection_score,"
                            "turnaround_selection_score"
                        ),
                        "limit": "1",
                    },
                )
            except DatabaseReadError as exc:
                migration_missing = exc.status_code in {404, 400} and any(
                    token in exc.body.upper()
                    for token in (
                        "PGRST204", "PGRST205", "RELATION", "COLUMN",
                        "SCHEMA CACHE", "NOT FIND",
                    )
                )
                if migration_missing:
                    return self.status_row(
                        "MIGRATION_REQUIRED_V7",
                        "Migration v6 tersedia, tetapi migration_v7_multibagger_lanes.sql belum diterapkan.",
                    )
                raise
            try:
                self._get_rows(
                    "multibagger_snapshots",
                    {
                        "select": (
                            "snapshot_id,multibagger_scoring_state,"
                            "multibagger_metric_coverage_pct,"
                            "multibagger_metric_data_gate"
                        ),
                        "limit": "1",
                    },
                )
                self._get_rows(
                    "selector_snapshots",
                    {
                        "select": (
                            "snapshot_id,production_selection_rank,"
                            "selector_rank_eligible,selector_data_state,"
                            "technical_feature_coverage_pct"
                        ),
                        "limit": "1",
                    },
                )
            except DatabaseReadError as exc:
                migration_missing = exc.status_code in {404, 400} and any(
                    token in exc.body.upper()
                    for token in (
                        "PGRST204", "PGRST205", "RELATION", "COLUMN",
                        "SCHEMA CACHE", "NOT FIND",
                    )
                )
                if migration_missing:
                    return self.status_row(
                        "MIGRATION_REQUIRED_V8",
                        "Migration v7 tersedia, tetapi "
                        "migration_v8_data_contract.sql belum diterapkan.",
                    )
                raise
            try:
                self._get_rows(
                    "narrative_events",
                    {
                        "select": (
                            "narrative_event_id,ticker,detected_at,"
                            "event_type,official_verified,source_state,"
                            "event_status,entity_match_state"
                        ),
                        "limit": "1",
                    },
                )
                self._get_rows(
                    "narrative_event_outcomes",
                    {
                        "select": (
                            "narrative_outcome_id,narrative_event_id,"
                            "signal_timestamp,outcome_status,entry_policy,"
                            "directional_excess_return_20d_pct"
                        ),
                        "limit": "1",
                    },
                )
                self._get_rows(
                    "narrative_snapshots",
                    {
                        "select": (
                            "snapshot_id,ticker,narrative_state,"
                            "issuer_alignment_state,"
                            "narrative_flow_convergence_state,"
                            "narrative_production_policy"
                        ),
                        "limit": "1",
                    },
                )
            except DatabaseReadError as exc:
                migration_missing = exc.status_code in {404, 400} and any(
                    token in exc.body.upper()
                    for token in (
                        "PGRST204", "PGRST205", "RELATION", "COLUMN",
                        "SCHEMA CACHE", "NOT FIND",
                    )
                )
                if migration_missing:
                    return self.status_row(
                        "MIGRATION_REQUIRED_V10",
                        "Migration v9 tersedia, tetapi "
                        "migration_v10_narrative_safety.sql belum diterapkan.",
                    )
                raise
            try:
                self._get_rows(
                    "multibagger_snapshots",
                    {
                        "select": (
                            "snapshot_id,multibagger_evidence_class,"
                            "multibagger_rank_eligible,"
                            "multibagger_score_comparability_pct,"
                            "multibagger_production_rank,"
                            "narrative_evidence_coverage_pct,"
                            "issuer_alignment_coverage_pct,"
                            "emir_method_coverage_pct,"
                            "distribution_severity_score,"
                            "distribution_penalty_points,"
                            "distribution_evidence_state"
                        ),
                        "limit": "1",
                    },
                )
                self._get_rows(
                    "selector_snapshots",
                    {
                        "select": (
                            "snapshot_id,relative_overlay_weight_pct,"
                            "selector_universe_state,"
                            "score_inflation_guard_active"
                        ),
                        "limit": "1",
                    },
                )
            except DatabaseReadError as exc:
                migration_missing = exc.status_code in {404, 400} and any(
                    token in exc.body.upper()
                    for token in (
                        "PGRST204", "PGRST205", "RELATION", "COLUMN",
                        "SCHEMA CACHE", "NOT FIND",
                    )
                )
                if migration_missing:
                    return self.status_row(
                        "MIGRATION_REQUIRED_V11",
                        "Migration v10 tersedia, tetapi "
                        "migration_v11_400_universe_evidence.sql belum diterapkan.",
                    )
                raise
            return self.status_row(
                "HEALTHY_V11_400_UNIVERSE_EVIDENCE",
                f"HTTP 200; schema {self.settings.schema}; v11 evidence comparability, production-rank contract, distribution evidence, and selector universe controls ready",
            )
        except Exception as exc:
            return self.status_row("DATABASE_FAIL_SOFT", f"{type(exc).__name__}: {str(exc)[:700]}")

    def _write_outbox(self, table: str, records: list[dict[str, Any]]) -> int:
        path = Path(self.settings.outbox_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps({"table": table, "record": record}, ensure_ascii=False) + "\n")
        self._write_details[table] = "OUTBOX_ONLY"
        return len(records)

    def _post_batch(self, table: str, batch: list[dict[str, Any]]) -> int:
        conflict = TABLE_CONFLICT_TARGETS.get(table, "snapshot_id")
        endpoint = f"{self.settings.supabase_url}/rest/v1/{table}?on_conflict={conflict}"
        response = requests.post(endpoint, headers=self._headers(), json=batch, timeout=self.settings.timeout_seconds)
        status_code = int(getattr(response, "status_code", 0) or 0)
        if bool(getattr(response, "ok", 200 <= status_code < 300)):
            return len(batch)
        hint_record = batch[0] if len(batch) == 1 else {}
        hint = _clean_text(hint_record.get("ticker") or hint_record.get("provider") or hint_record.get("scan_id") or hint_record.get(conflict))
        raise DatabaseWriteError(table, status_code or None, getattr(response, "text", ""), hint)

    def _write_with_isolation(self, table: str, batch: list[dict[str, Any]], failures: list[str]) -> int:
        if not batch:
            return 0
        try:
            return self._post_batch(table, batch)
        except DatabaseWriteError as exc:
            if len(batch) == 1:
                failures.append(str(exc))
                return 0
            midpoint = max(1, len(batch) // 2)
            return self._write_with_isolation(table, batch[:midpoint], failures) + self._write_with_isolation(table, batch[midpoint:], failures)

    def _upsert_supabase(self, table: str, records: list[dict[str, Any]]) -> int:
        written = 0
        failures: list[str] = []
        for start in range(0, len(records), 100):
            written += self._write_with_isolation(table, records[start:start + 100], failures)
        if failures:
            preview = " | ".join(failures[:3])
            suffix = f" | +{len(failures) - 3} row errors" if len(failures) > 3 else ""
            self._write_details[table] = f"isolated_failures={len(failures)}; {preview}{suffix}"[:1400]
        else:
            self._write_details[table] = "all_rows_written"
        return written


__all__ = [
    "DATABASE_BRIDGE_VERSION", "DATABASE_SCHEMA_VERSION", "DatabaseSettings",
    "DatabaseWriteError", "DatabaseReadError", "ScannerDatabaseBridge",
    "TABLE_FIELD_TYPES", "TABLE_CONFLICT_TARGETS", "_normalise_record",
    "_snapshot_id", "_provider_health_frame", "_freshness_state", "_semantic_hash",
]
