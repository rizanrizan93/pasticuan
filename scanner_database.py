from __future__ import annotations

"""Database-first, fail-soft Supabase bridge for IDX Super Scanner v8 Slim.

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
import base64
import hashlib
import io
import json
import math
import os
import re
import time
import zlib

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

        @staticmethod
        def patch(*_args: Any, **_kwargs: Any) -> Any:
            raise ModuleNotFoundError("requests is required for Supabase REST")

    requests = _RequestsUnavailable()

from research_maintenance import MODEL_VERSIONS, semantic_refresh_reason, model_registry_frame
from ihsg_direction import ihsg_snapshot_frame
from selector_engine import selector_snapshot_frame

DATABASE_BRIDGE_VERSION = "15.0-feature-cache-compact-ohlcv"
DATABASE_SCHEMA_VERSION = "scanner_schema_v15"
LEGACY_DATABASE_HEALTH_STATE_V14 = "HEALTHY_V14_GUARDED_REAL_MONEY"  # compatibility marker for v9.8.0 regression audit

DATABASE_VERIFICATION_TABLES: tuple[str, ...] = (
    "scan_runs",
    "fundamental_snapshots",
    "multibagger_snapshots",
    "technical_snapshots",
    "selector_snapshots",
    "narrative_snapshots",
    "project_events",
    "scan_jobs",
    "scan_job_items",
    "scan_job_artifacts",
    "ohlcv_daily_cache",
    "scanner_feature_cache",
)

TABLE_CONFLICT_TARGETS: dict[str, str] = {
    "fundamental_cache": "ticker",
    "fundamental_history_cache": "ticker",
    "forward_quality_cache": "ticker",
    "refresh_state": "entity_key",
    "scan_checkpoints": "checkpoint_id",
    "scan_jobs": "job_id",
    "scan_job_items": "item_key",
    "scan_job_artifacts": "artifact_key",
    "ohlcv_daily_cache": "ticker",
    "scanner_feature_cache": "ticker",
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
            "fundamental_data_grade", "fundamental_source_families", "fundamental_reliability", "fundamental_reconciliation_state", "model_version",
            "schema_version", "as_of", "fundamental_fetched_at",
            "database_source_state", "content_hash",
        },
        "numeric": {
            "fundamental_score", "fundamental_score_10", "fundamental_coverage",
            "revenue_growth", "earnings_growth", "roe",
            "roa", "roic_proxy", "net_margin", "operating_margin",
            "operating_cash_flow", "free_cash_flow", "cash_conversion_ttm",
            "debt_equity", "net_debt_ebitda", "interest_coverage", "market_cap",
            "statement_age_days", "fundamental_official_source_coverage_pct",
            "fundamental_cashflow_statement_coverage_pct", "fundamental_consensus_score",
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
            "real_money_authorization_state", "real_money_authorization_blockers", "real_money_manual_checks",
            "fundamental_score_cap_reason", "fundamental_cashflow_state", "fundamental_leverage_risk_state",
            "fundamental_official_state", "market_regime", "market_context_provenance_state",
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
            "distribution_penalty_points", "v9_next_leader_score", "final_score",
            "fundamental_conviction_cap", "fundamental_data_quality_score",
            "fundamental_official_source_coverage_pct", "fundamental_consensus_score",
            "fundamental_history_coverage_pct", "fundamental_cashflow_coverage_pct",
            "market_context_score", "market_context_coverage_pct", "real_money_risk_budget_cap_pct",
            "real_money_risk_budget_idr", "real_money_risk_per_share",
            "inventory_multi_horizon_score", "distribution_risk_score", "reaccumulation_quality_score",
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
            "real_money_authorization_pass", "fundamental_official_verified",
            "independent_price_verified", "anti_chase_gate",
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
    "scan_jobs": {
        "text": {
            "job_id", "job_type", "job_key", "universe_hash", "config_hash", "status", "phase",
            "active_worker", "lease_expires_at", "last_error", "created_at", "updated_at",
            "started_at", "finished_at", "model_version", "schema_version",
        },
        "numeric": set(),
        "integer": {"total_items", "completed_items", "failed_items", "retry_items", "chunk_size", "max_attempts"},
        "boolean": set(),
        "json": {"universe_payload", "config_payload", "result_summary"},
        "required": {"job_id", "job_type", "job_key", "universe_hash", "config_hash", "status", "phase", "model_version", "schema_version"},
    },
    "scan_job_items": {
        "text": {
            "item_key", "job_id", "ticker", "phase", "status", "next_attempt_at", "lease_owner",
            "lease_expires_at", "last_error", "created_at", "updated_at", "started_at", "completed_at",
            "model_version", "schema_version",
        },
        "numeric": set(),
        "integer": {"sequence_no", "attempt_count", "max_attempts"},
        "boolean": set(),
        "json": {"result_payload"},
        "required": {"item_key", "job_id", "ticker", "phase", "status", "model_version", "schema_version"},
    },
    "scan_job_artifacts": {
        "text": {
            "artifact_key", "job_id", "artifact_type", "created_at", "updated_at", "model_version", "schema_version",
        },
        "numeric": set(), "integer": {"chunk_number"}, "boolean": set(), "json": {"payload"},
        "required": {"artifact_key", "job_id", "artifact_type", "model_version", "schema_version"},
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


class DatabaseTransportError(RuntimeError):
    """Network/transport failure before a usable PostgREST response exists."""

    def __init__(self, operation: str, target: str, detail: str, attempts: int = 1) -> None:
        clean = " ".join(str(detail or "").split())[:700]
        super().__init__(f"{operation} {target}: transport failure after {max(1, int(attempts))} attempt(s); {clean}")
        self.operation = str(operation)
        self.target = str(target)
        self.detail = clean
        self.attempts = max(1, int(attempts))


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


def _finite(value: Any, default: float = np.nan) -> float:
    """Return a finite float or ``default`` for persistence metadata.

    Database reports may arrive from CSV/PostgREST as strings, numpy scalars,
    pandas missing values, or infinities.  Persistence must normalise those
    values before converting cohort fields to integers.
    """
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return number if math.isfinite(number) else float(default)


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


def _env_or_streamlit_secret(name: str, default: str = "") -> str:
    value = os.getenv(name, "")
    if str(value).strip():
        return str(value)
    try:
        import streamlit as st  # type: ignore
        return str(st.secrets.get(name, default) or default)
    except Exception:
        return str(default)


@dataclass(frozen=True)
class DatabaseSettings:
    enabled: bool = False
    mode: str = "DISABLED"
    supabase_url: str = ""
    supabase_key: str = ""
    supabase_key_type: str = "NONE"
    schema: str = "public"
    timeout_seconds: float = 18.0
    connect_timeout_seconds: float = 5.0
    read_attempts: int = 3
    write_attempts: int = 2
    retry_backoff_seconds: float = 0.8
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
        enabled = _truthy(_env_or_streamlit_secret("SCANNER_DATABASE_ENABLED"))
        requested_mode = _env_or_streamlit_secret("SCANNER_DATABASE_MODE", "").strip().upper()
        url = _env_or_streamlit_secret("SUPABASE_URL", "").strip().rstrip("/")
        key_candidates = (
            ("SECRET", _env_or_streamlit_secret("SUPABASE_SECRET_KEY", "").strip()),
            ("SERVICE_ROLE", _env_or_streamlit_secret("SUPABASE_SERVICE_ROLE_KEY", "").strip()),
            ("PUBLISHABLE", _env_or_streamlit_secret("SUPABASE_PUBLISHABLE_KEY", "").strip()),
            ("ANON", _env_or_streamlit_secret("SUPABASE_ANON_KEY", "").strip()),
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
        read_setting = _env_or_streamlit_secret("SCANNER_DATABASE_READ_ENABLED", "true")
        return cls(
            enabled=enabled,
            mode=mode,
            supabase_url=url,
            supabase_key=key,
            supabase_key_type=key_type,
            schema=_env_or_streamlit_secret("SCANNER_DATABASE_SCHEMA", "public").strip() or "public",
            timeout_seconds=max(15.0, float(_env_or_streamlit_secret("SCANNER_DATABASE_TIMEOUT", "18"))),
            connect_timeout_seconds=max(2.0, float(_env_or_streamlit_secret("SCANNER_DATABASE_CONNECT_TIMEOUT", "5"))),
            read_attempts=max(1, min(5, int(_env_or_streamlit_secret("SCANNER_DATABASE_READ_ATTEMPTS", "3")))),
            write_attempts=max(1, min(3, int(_env_or_streamlit_secret("SCANNER_DATABASE_WRITE_ATTEMPTS", "2")))),
            retry_backoff_seconds=max(0.1, min(5.0, float(_env_or_streamlit_secret("SCANNER_DATABASE_RETRY_BACKOFF", "0.8")))),
            outbox_path=_env_or_streamlit_secret("SCANNER_DATABASE_OUTBOX", ".scanner_cache/database_outbox.jsonl"),
            max_rows_per_table=max(20, int(_env_or_streamlit_secret("SCANNER_DATABASE_MAX_ROWS", "2000"))),
            cache_max_rows_per_table=max(500, int(_env_or_streamlit_secret("SCANNER_DATABASE_CACHE_MAX_ROWS", "5000"))),
            read_enabled=_truthy(read_setting),
            read_batch_size=max(20, min(250, int(_env_or_streamlit_secret("SCANNER_DATABASE_READ_BATCH_SIZE", "100")))),
            fundamental_max_age_days=max(1, int(_env_or_streamlit_secret("SCANNER_DATABASE_FUNDAMENTAL_MAX_AGE_DAYS", "21"))),
            history_max_age_days=max(1, int(_env_or_streamlit_secret("SCANNER_DATABASE_HISTORY_MAX_AGE_DAYS", "30"))),
            forward_max_age_days=max(1, int(_env_or_streamlit_secret("SCANNER_DATABASE_FORWARD_MAX_AGE_DAYS", "7"))),
            stale_max_age_days=max(30, int(_env_or_streamlit_secret("SCANNER_DATABASE_STALE_MAX_AGE_DAYS", "180"))),
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
            "table": "", "scan_id": "", "rows_attempted": 0, "rows_written": 0,
            "rows_verified": 0, "verification_pct": 0.0, "detail": detail,
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

    def _request_timeout(self) -> tuple[float, float]:
        return (float(self.settings.connect_timeout_seconds), float(self.settings.timeout_seconds))

    @staticmethod
    def _transient_status(status_code: int) -> bool:
        return int(status_code or 0) in {408, 425, 429, 500, 502, 503, 504}

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        operation: str,
        retry_safe: bool,
        **kwargs: Any,
    ) -> Any:
        if retry_safe:
            attempts = int(self.settings.read_attempts) if str(method).lower() == "get" else int(self.settings.write_attempts)
        else:
            attempts = 1
        attempts = max(1, attempts)
        kwargs = dict(kwargs)
        kwargs.setdefault("timeout", self._request_timeout())
        last_exc: Exception | None = None
        response: Any = None
        for attempt in range(1, attempts + 1):
            try:
                caller = getattr(requests, str(method).lower())
                response = caller(endpoint, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt >= attempts:
                    raise DatabaseTransportError(operation, endpoint, f"{type(exc).__name__}: {exc}", attempt) from exc
                time.sleep(float(self.settings.retry_backoff_seconds) * (2 ** (attempt - 1)))
                continue
            status_code = int(getattr(response, "status_code", 0) or 0)
            if retry_safe and self._transient_status(status_code) and attempt < attempts:
                time.sleep(float(self.settings.retry_backoff_seconds) * (2 ** (attempt - 1)))
                continue
            return response
        if last_exc is not None:
            raise DatabaseTransportError(operation, endpoint, f"{type(last_exc).__name__}: {last_exc}", attempts) from last_exc
        return response

    def _get_rows(self, table: str, params: Mapping[str, Any]) -> list[dict[str, Any]]:
        endpoint = f"{self.settings.supabase_url}/rest/v1/{table}"
        response = self._request(
            "get", endpoint, operation=f"GET {table}", retry_safe=True,
            headers=self._read_headers(), params=dict(params),
        )
        status_code = int(getattr(response, "status_code", 0) or 0)
        if not bool(getattr(response, "ok", 200 <= status_code < 300)):
            raise DatabaseReadError(table, status_code or None, getattr(response, "text", ""))
        try:
            payload = response.json()
        except Exception as exc:
            raise DatabaseReadError(table, status_code or None, f"Invalid JSON response: {type(exc).__name__}: {exc}") from exc
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

    def _read_source_event_cache(
        self,
        tickers: Sequence[str],
        *,
        event_type: str,
        max_age_days: int,
        payload_kind: str,
        asof_field: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        names = list(dict.fromkeys(str(ticker).upper().strip() for ticker in tickers if str(ticker).strip()))
        audit_columns = [
            "ticker", "provider", "scope", "status", "database_read_state", "age_days",
            "refresh_required", "rows", "asof", "error", "source_family",
        ]
        if not names:
            return pd.DataFrame(), pd.DataFrame(columns=audit_columns)
        if self.settings.mode != "SUPABASE_REST" or not self.settings.read_enabled:
            state = "READ_DISABLED" if not self.settings.read_enabled else self.settings.mode
            return pd.DataFrame(), pd.DataFrame([{
                "ticker": ticker, "provider": "SUPABASE_DATABASE_FIRST", "scope": payload_kind,
                "status": state, "database_read_state": state, "age_days": np.nan,
                "refresh_required": True, "rows": 0, "asof": "", "error": "",
                "source_family": "DATABASE",
            } for ticker in names], columns=audit_columns)
        latest: dict[str, dict[str, Any]] = {}
        error_text = ""
        try:
            for start in range(0, len(names), self.settings.read_batch_size):
                chunk = names[start:start + self.settings.read_batch_size]
                rows = self._get_rows("source_events", {
                    "select": "*",
                    "ticker": f"in.({','.join(chunk)})",
                    "event_type": f"eq.{event_type}",
                    "order": "detected_at.desc",
                    "limit": str(max(20, len(chunk) * 2)),
                })
                for row in rows:
                    ticker = _clean_text(row.get("ticker")).upper()
                    if ticker and ticker not in latest:
                        latest[ticker] = row
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {str(exc)[:500]}"
        if error_text:
            return pd.DataFrame(), pd.DataFrame([{
                "ticker": ticker, "provider": "SUPABASE_DATABASE_FIRST", "scope": payload_kind,
                "status": "DATABASE_READ_FAIL_SOFT", "database_read_state": "DATABASE_READ_FAIL_SOFT",
                "age_days": np.nan, "refresh_required": True, "rows": 0, "asof": "",
                "error": error_text, "source_family": "DATABASE",
            } for ticker in names], columns=audit_columns)

        output: list[dict[str, Any]] = []
        audits: list[dict[str, Any]] = []
        now = _as_utc_timestamp(datetime.now(timezone.utc))
        for ticker in names:
            raw = latest.get(ticker)
            if not raw:
                audits.append({
                    "ticker": ticker, "provider": "SUPABASE_DATABASE_FIRST", "scope": payload_kind,
                    "status": "DATABASE_MISS", "database_read_state": "DATABASE_MISS",
                    "age_days": np.nan, "refresh_required": True, "rows": 0, "asof": "",
                    "error": "", "source_family": "DATABASE",
                })
                continue
            payload = raw.get("payload")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    payload = None
            checked_at = None
            if isinstance(payload, Mapping):
                checked_at = payload.get(asof_field)
            checked_at = checked_at or raw.get("detected_at") or raw.get("last_seen_at")
            stamp = _as_utc_timestamp(checked_at)
            age_days = (now - stamp).total_seconds() / 86400.0 if pd.notna(now) and pd.notna(stamp) else math.inf
            current = math.isfinite(age_days) and age_days <= max(1, int(max_age_days))
            loaded = 0
            if current and isinstance(payload, Mapping):
                local = dict(payload)
                local["ticker"] = ticker
                local["database_source_state"] = "DATABASE_CURRENT"
                local["database_source_checked_at"] = checked_at
                output.append(local)
                loaded = 1
            audits.append({
                "ticker": ticker, "provider": "SUPABASE_DATABASE_FIRST", "scope": payload_kind,
                "status": "DATABASE_CURRENT" if loaded else "DATABASE_STALE",
                "database_read_state": "DATABASE_CURRENT" if loaded else "DATABASE_STALE",
                "age_days": round(age_days, 2) if math.isfinite(age_days) else np.nan,
                "refresh_required": not bool(loaded), "rows": loaded,
                "asof": _clean_text(checked_at), "error": "" if loaded else "Stored evidence expired",
                "source_family": _clean_text(raw.get("source_family")) or "DATABASE",
            })
        return pd.DataFrame(output), pd.DataFrame(audits, columns=audit_columns)


    # ------------------------------------------------------------------
    # Durable resumable job repository (schema v12)
    # ------------------------------------------------------------------
    def _rpc(self, function_name: str, payload: Mapping[str, Any]) -> Any:
        if self.settings.mode != "SUPABASE_REST":
            raise RuntimeError(f"RPC {function_name} requires SUPABASE_REST")
        endpoint = f"{self.settings.supabase_url}/rest/v1/rpc/{function_name}"
        # Only retry RPCs whose repeated execution cannot claim a second batch.
        # claim_scan_job_items is deliberately single-attempt because a timed-out
        # response may already have leased a chunk on the server.
        retry_safe = function_name in {
            "refresh_scan_job_counters", "renew_scan_job_leases", "claim_scan_job_lease",
        }
        response = self._request(
            "post", endpoint, operation=f"RPC {function_name}", retry_safe=retry_safe,
            headers=self._headers(),
            json={key: _json_safe(value) for key, value in payload.items()},
        )
        status_code = int(getattr(response, "status_code", 0) or 0)
        if not bool(getattr(response, "ok", 200 <= status_code < 300)):
            raise DatabaseWriteError(function_name, status_code or None, getattr(response, "text", ""), "")
        try:
            return response.json()
        except Exception:
            return None

    def _patch_rows(self, table: str, filters: Mapping[str, Any], payload: Mapping[str, Any]) -> list[dict[str, Any]]:
        if self.settings.mode != "SUPABASE_REST":
            raise RuntimeError(f"PATCH {table} requires SUPABASE_REST")
        endpoint = f"{self.settings.supabase_url}/rest/v1/{table}"
        headers = self._headers().copy()
        headers["Prefer"] = "return=representation"
        response = self._request(
            "patch", endpoint, operation=f"PATCH {table}", retry_safe=True,
            headers=headers,
            params=dict(filters),
            json={key: _json_safe(value) for key, value in payload.items()},
        )
        status_code = int(getattr(response, "status_code", 0) or 0)
        if not bool(getattr(response, "ok", 200 <= status_code < 300)):
            raise DatabaseWriteError(table, status_code or None, getattr(response, "text", ""), "")
        try:
            body = response.json()
            return body if isinstance(body, list) else []
        except Exception:
            return []

    @staticmethod
    def resumable_job_hash(value: Any) -> str:
        encoded = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def create_or_resume_scan_job(
        self,
        *,
        job_type: str,
        tickers: Sequence[str],
        config_payload: Mapping[str, Any],
        phase: str,
        chunk_size: int = 20,
        max_attempts: int = 2,
        model_version: str = "9.6.0",
    ) -> dict[str, Any]:
        """Create a durable job or return the latest compatible active job.

        Secrets must not be placed in ``config_payload``. The job stores only
        reproducibility-safe settings and the universe ordering.
        """
        if self.settings.mode != "SUPABASE_REST":
            raise RuntimeError("Resumable scan requires SUPABASE_REST and migration v12")
        universe = list(dict.fromkeys(str(t).upper().strip() for t in tickers if str(t).strip()))
        if not universe:
            raise ValueError("Resumable scan universe cannot be empty")
        safe_config = _json_safe(dict(config_payload))
        universe_hash = self.resumable_job_hash(universe)
        config_hash = self.resumable_job_hash(safe_config)
        active_status = "in.(PENDING,RUNNING,PAUSED,FINALIZING)"
        rows = self._get_rows("scan_jobs", {
            "select": "*",
            "job_type": f"eq.{str(job_type)}",
            "universe_hash": f"eq.{universe_hash}",
            "config_hash": f"eq.{config_hash}",
            "status": active_status,
            "order": "updated_at.desc",
            "limit": "1",
        })
        if rows:
            return rows[0]

        now = datetime.now(timezone.utc)
        nonce = now.isoformat(timespec="microseconds")
        job_id = hashlib.sha256(f"{job_type}|{universe_hash}|{config_hash}|{nonce}".encode("utf-8")).hexdigest()
        job_key = hashlib.sha256(f"ACTIVE|{job_type}|{universe_hash}|{config_hash}".encode("utf-8")).hexdigest()
        job = {
            "job_id": job_id,
            "job_type": str(job_type),
            "job_key": job_key,
            "universe_hash": universe_hash,
            "config_hash": config_hash,
            "universe_payload": universe,
            "config_payload": safe_config,
            "status": "PENDING",
            "phase": str(phase),
            "total_items": len(universe),
            "completed_items": 0,
            "failed_items": 0,
            "retry_items": 0,
            "chunk_size": max(1, int(chunk_size)),
            "max_attempts": max(1, int(max_attempts)),
            "result_summary": {},
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "model_version": str(model_version),
            "schema_version": DATABASE_SCHEMA_VERSION,
        }
        try:
            written = self._upsert_supabase("scan_jobs", [job])
        except Exception:
            # A second Streamlit session may have created the same active job
            # between the read and insert. Re-read the durable active row.
            rows = self._get_rows("scan_jobs", {
                "select": "*", "job_key": f"eq.{job_key}",
                "status": active_status, "order": "updated_at.desc", "limit": "1",
            })
            if rows:
                return rows[0]
            raise
        if written != 1:
            rows = self._get_rows("scan_jobs", {
                "select": "*", "job_key": f"eq.{job_key}",
                "status": active_status, "order": "updated_at.desc", "limit": "1",
            })
            if rows:
                return rows[0]
            raise RuntimeError("Failed to create scan_jobs row")
        items = [{
            "item_key": hashlib.sha256(f"{job_id}|{phase}|{ticker}".encode("utf-8")).hexdigest(),
            "job_id": job_id,
            "ticker": ticker,
            "phase": str(phase),
            "sequence_no": sequence,
            "status": "PENDING",
            "attempt_count": 0,
            "max_attempts": max(1, int(max_attempts)),
            "result_payload": {},
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "model_version": str(model_version),
            "schema_version": DATABASE_SCHEMA_VERSION,
        } for sequence, ticker in enumerate(universe)]
        if items and self._upsert_supabase("scan_job_items", items) != len(items):
            self._patch_rows("scan_jobs", {"job_id": f"eq.{job_id}"}, {
                "status": "FAILED", "last_error": "ITEM_INITIALISATION_FAILED", "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            raise RuntimeError("Failed to initialise scan_job_items")
        return job

    def read_scan_job(self, job_id: str) -> dict[str, Any]:
        rows = self._get_rows("scan_jobs", {"select": "*", "job_id": f"eq.{job_id}", "limit": "1"})
        return rows[0] if rows else {}

    def read_latest_scan_job(self, *, job_type: str | None = None, statuses: Sequence[str] | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"select": "*", "order": "updated_at.desc", "limit": "1"}
        if job_type:
            params["job_type"] = f"eq.{job_type}"
        if statuses:
            clean = ",".join(str(value) for value in statuses)
            params["status"] = f"in.({clean})"
        rows = self._get_rows("scan_jobs", params)
        return rows[0] if rows else {}

    def read_scan_job_items(
        self,
        job_id: str,
        *,
        phase: str | None = None,
        statuses: Sequence[str] | None = None,
        include_payload: bool = True,
        limit: int = 5000,
    ) -> pd.DataFrame:
        select = "*" if include_payload else "item_key,job_id,ticker,phase,sequence_no,status,attempt_count,max_attempts,next_attempt_at,lease_owner,lease_expires_at,last_error,updated_at,completed_at"
        params: dict[str, Any] = {
            "select": select,
            "job_id": f"eq.{job_id}",
            "order": "sequence_no.asc",
            "limit": str(max(1, int(limit))),
        }
        if phase:
            params["phase"] = f"eq.{phase}"
        if statuses:
            params["status"] = f"in.({','.join(str(value) for value in statuses)})"
        return pd.DataFrame(self._get_rows("scan_job_items", params))

    def claim_scan_job_lease(self, job_id: str, worker_id: str, *, lease_seconds: int = 300) -> dict[str, Any]:
        payload = self._rpc("claim_scan_job_lease", {
            "p_job_id": job_id,
            "p_worker_id": worker_id,
            "p_lease_seconds": max(60, int(lease_seconds)),
        })
        if isinstance(payload, list):
            return payload[0] if payload else {}
        return payload if isinstance(payload, dict) else {}

    def renew_scan_job_leases(self, job_id: str, worker_id: str, *, lease_seconds: int = 300) -> dict[str, Any]:
        """Renew the job and all RUNNING item leases owned by this worker."""
        payload = self._rpc("renew_scan_job_leases", {
            "p_job_id": job_id,
            "p_worker_id": worker_id,
            "p_lease_seconds": max(60, int(lease_seconds)),
        })
        if isinstance(payload, list):
            return payload[0] if payload else {}
        return payload if isinstance(payload, dict) else {}

    def claim_scan_job_items(
        self,
        job_id: str,
        phase: str,
        *,
        limit: int,
        worker_id: str,
        lease_seconds: int = 300,
    ) -> pd.DataFrame:
        payload = self._rpc("claim_scan_job_items", {
            "p_job_id": job_id,
            "p_phase": phase,
            "p_limit": max(1, int(limit)),
            "p_worker_id": worker_id,
            "p_lease_seconds": max(30, int(lease_seconds)),
        })
        return pd.DataFrame(payload if isinstance(payload, list) else [])

    def complete_scan_job_item(self, item: Mapping[str, Any], result_payload: Mapping[str, Any] | None = None) -> None:
        now = datetime.now(timezone.utc).isoformat()
        record = dict(item)
        record.update({
            "status": "COMPLETE",
            "result_payload": _json_safe(dict(result_payload or {})),
            "lease_owner": None,
            "lease_expires_at": None,
            "next_attempt_at": None,
            "last_error": None,
            "completed_at": now,
            "updated_at": now,
            "model_version": record.get("model_version") or "9.6.0",
            "schema_version": DATABASE_SCHEMA_VERSION,
        })
        self._upsert_supabase("scan_job_items", [record])

    def fail_scan_job_item(
        self,
        item: Mapping[str, Any],
        error: str,
        *,
        retry_delay_seconds: int = 15,
        result_payload: Mapping[str, Any] | None = None,
    ) -> str:
        now = datetime.now(timezone.utc)
        attempts = int(item.get("attempt_count", 0) or 0)
        max_attempts = max(1, int(item.get("max_attempts", 2) or 2))
        terminal = attempts >= max_attempts
        status = "FAILED" if terminal else "RETRY"
        record = dict(item)
        record.update({
            "status": status,
            "result_payload": _json_safe(dict(result_payload or record.get("result_payload") or {})),
            "lease_owner": None,
            "lease_expires_at": None,
            "next_attempt_at": None if terminal else (now + timedelta(seconds=max(1, int(retry_delay_seconds)))).isoformat(),
            "last_error": str(error)[:1200],
            "completed_at": now.isoformat() if terminal else None,
            "updated_at": now.isoformat(),
            "model_version": record.get("model_version") or "9.6.0",
            "schema_version": DATABASE_SCHEMA_VERSION,
        })
        self._upsert_supabase("scan_job_items", [record])
        return status

    def checkpoint_scan_job_items_batch(self, checkpoints: Sequence[Mapping[str, Any]]) -> dict[str, str]:
        """Persist one processed chunk with a single batched UPSERT.

        v9.7.0 checkpointed every ticker through an individual HTTP request.
        With a 400-ticker universe and chunk_size=20 that meant roughly 400
        item-write round trips.  The durable contract is unchanged here; only
        the transport is batched.  A caller can fall back to the legacy
        per-item methods if this batch write raises.
        """
        now = datetime.now(timezone.utc)
        records: list[dict[str, Any]] = []
        states: dict[str, str] = {}
        for checkpoint in checkpoints:
            item = dict(checkpoint.get("item") or {})
            key = _clean_text(item.get("item_key"))
            if not key:
                continue
            success = bool(checkpoint.get("success", False))
            payload = dict(checkpoint.get("payload") or {})
            record = dict(item)
            if success:
                status = "COMPLETE"
                record.update({
                    "status": status,
                    "result_payload": _json_safe(payload),
                    "lease_owner": None,
                    "lease_expires_at": None,
                    "next_attempt_at": None,
                    "last_error": None,
                    "completed_at": now.isoformat(),
                    "updated_at": now.isoformat(),
                    "model_version": record.get("model_version") or "9.8.0",
                    "schema_version": DATABASE_SCHEMA_VERSION,
                })
            else:
                attempts = int(item.get("attempt_count", 0) or 0)
                max_attempts = max(1, int(item.get("max_attempts", 2) or 2))
                terminal = attempts >= max_attempts
                status = "FAILED" if terminal else "RETRY"
                retry_delay = max(1, int(checkpoint.get("retry_delay_seconds", 15) or 15))
                record.update({
                    "status": status,
                    "result_payload": _json_safe(payload or record.get("result_payload") or {}),
                    "lease_owner": None,
                    "lease_expires_at": None,
                    "next_attempt_at": None if terminal else (now + timedelta(seconds=retry_delay)).isoformat(),
                    "last_error": str(checkpoint.get("error") or "ITEM_FAILED")[:1200],
                    "completed_at": now.isoformat() if terminal else None,
                    "updated_at": now.isoformat(),
                    "model_version": record.get("model_version") or "9.8.0",
                    "schema_version": DATABASE_SCHEMA_VERSION,
                })
            records.append(record)
            states[key] = status
        if not records:
            return states
        written = self._upsert_supabase("scan_job_items", records)
        if written != len(records):
            raise RuntimeError(f"Batch checkpoint incomplete: {written}/{len(records)}")
        return states

    def persist_scan_job_artifacts_batch(
        self,
        job_id: str,
        artifacts: Mapping[str, Any],
        *,
        chunk_number: int = 0,
        model_version: str = "9.8.0",
    ) -> dict[str, str]:
        """Publish several deterministic artifacts in one Supabase UPSERT."""
        now = datetime.now(timezone.utc).isoformat()
        records: list[dict[str, Any]] = []
        keys: dict[str, str] = {}
        for artifact_type, payload in dict(artifacts or {}).items():
            artifact_key = hashlib.sha256(
                f"{job_id}|{artifact_type}|{int(chunk_number)}".encode("utf-8")
            ).hexdigest()
            keys[str(artifact_type)] = artifact_key
            records.append({
                "artifact_key": artifact_key,
                "job_id": job_id,
                "artifact_type": str(artifact_type),
                "chunk_number": int(chunk_number),
                "payload": _json_safe(payload),
                "created_at": now,
                "updated_at": now,
                "model_version": str(model_version),
                "schema_version": DATABASE_SCHEMA_VERSION,
            })
        if records:
            written = self._upsert_supabase("scan_job_artifacts", records)
            if written != len(records):
                raise RuntimeError(f"Batch artifact publish incomplete: {written}/{len(records)}")
        return keys

    def requeue_failed_scan_job_items(self, job_id: str) -> int:
        """Reset FAILED items to RETRY inside the same durable job.

        COMPLETE checkpoints and existing artifacts are preserved. The finalizer
        overwrites deterministic artifact keys after the retried items finish.
        """
        failed = self.read_scan_job_items(
            job_id,
            statuses=["FAILED"],
            include_payload=True,
            limit=5000,
        )
        if failed.empty:
            return 0
        now = datetime.now(timezone.utc).isoformat()
        records: list[dict[str, Any]] = []
        for row in failed.to_dict("records"):
            record = dict(row)
            record.update({
                "status": "RETRY",
                "attempt_count": 0,
                "lease_owner": None,
                "lease_expires_at": None,
                "next_attempt_at": now,
                "last_error": "MANUAL_REQUEUE_SAME_JOB",
                "completed_at": None,
                "updated_at": now,
                "model_version": record.get("model_version") or "9.6.0",
                "schema_version": DATABASE_SCHEMA_VERSION,
            })
            records.append(record)
        written = self._upsert_supabase("scan_job_items", records)
        if written != len(records):
            raise RuntimeError(f"Failed to requeue all items: {written}/{len(records)}")
        self.update_scan_job(
            job_id,
            status="RUNNING",
            phase=str(failed.iloc[0].get("phase") or "TECHNICAL"),
            finished_at=None,
            active_worker=None,
            lease_expires_at=None,
            last_error=None,
            result_summary={
                "requeued_failed_items": len(records),
                "requeued_at": now,
                "ranking_state": "STALE_PENDING_RETRY",
            },
        )
        self.refresh_scan_job_counters(job_id)
        return len(records)

    def refresh_scan_job_counters(self, job_id: str) -> dict[str, Any]:
        payload = self._rpc("refresh_scan_job_counters", {"p_job_id": job_id})
        if isinstance(payload, list):
            return payload[0] if payload else {}
        return payload if isinstance(payload, dict) else self.read_scan_job(job_id)

    def update_scan_job(self, job_id: str, **fields: Any) -> dict[str, Any]:
        fields = dict(fields)
        fields["updated_at"] = datetime.now(timezone.utc).isoformat()
        rows = self._patch_rows("scan_jobs", {"job_id": f"eq.{job_id}"}, fields)
        return rows[0] if rows else self.read_scan_job(job_id)

    def persist_scan_job_artifact(
        self,
        job_id: str,
        artifact_type: str,
        payload: Any,
        *,
        chunk_number: int = 0,
        model_version: str = "9.6.0",
    ) -> str:
        artifact_key = hashlib.sha256(f"{job_id}|{artifact_type}|{int(chunk_number)}".encode("utf-8")).hexdigest()
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "artifact_key": artifact_key,
            "job_id": job_id,
            "artifact_type": str(artifact_type),
            "chunk_number": int(chunk_number),
            "payload": _json_safe(payload),
            "created_at": now,
            "updated_at": now,
            "model_version": str(model_version),
            "schema_version": DATABASE_SCHEMA_VERSION,
        }
        self._upsert_supabase("scan_job_artifacts", [record])
        return artifact_key

    def read_scan_job_artifacts(self, job_id: str, artifact_type: str | None = None) -> pd.DataFrame:
        params: dict[str, Any] = {
            "select": "*", "job_id": f"eq.{job_id}", "order": "chunk_number.asc", "limit": "5000",
        }
        if artifact_type:
            params["artifact_type"] = f"eq.{artifact_type}"
        return pd.DataFrame(self._get_rows("scan_job_artifacts", params))

    @staticmethod
    def _normalise_ohlcv_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        out = frame.copy()
        if "Date" in out.columns:
            out.index = pd.to_datetime(out.pop("Date"), errors="coerce")
        else:
            out.index = pd.to_datetime(out.index, errors="coerce")
        out = out.loc[~pd.DatetimeIndex(out.index).isna()].copy()
        out.index = pd.DatetimeIndex(out.index).tz_localize(None).normalize()
        for column in ("Open", "High", "Low", "Close", "Volume"):
            if column not in out.columns:
                out[column] = np.nan
            out[column] = pd.to_numeric(out[column], errors="coerce")
        out = out[["Open", "High", "Low", "Close", "Volume"]]
        out = out.dropna(subset=["Close"]).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        return out

    @classmethod
    def _ohlcv_payload(cls, frame: pd.DataFrame | None, *, max_bars: int = 900) -> list[dict[str, Any]]:
        clean = cls._normalise_ohlcv_frame(frame).tail(max(260, int(max_bars)))
        if clean.empty:
            return []
        local = clean.reset_index().rename(columns={"index": "Date"})
        return [
            {
                "Date": pd.Timestamp(row["Date"]).date().isoformat(),
                "Open": _coerce_numeric(row.get("Open")),
                "High": _coerce_numeric(row.get("High")),
                "Low": _coerce_numeric(row.get("Low")),
                "Close": _coerce_numeric(row.get("Close")),
                "Volume": _coerce_numeric(row.get("Volume")),
            }
            for row in local.to_dict("records")
        ]

    @classmethod
    def _ohlcv_frame_from_payload(cls, payload: Any) -> pd.DataFrame:
        if not isinstance(payload, list):
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        return cls._normalise_ohlcv_frame(pd.DataFrame([row for row in payload if isinstance(row, Mapping)]))

    @classmethod
    def _ohlcv_compact_encode(cls, frame: pd.DataFrame | None, *, max_bars: int = 900) -> tuple[str, str, int]:
        """Return a compressed text representation of OHLCV.

        Supabase JSONB history is durable but expensive to transfer for 400 names.
        v15 keeps the legacy JSON column for rollback compatibility while new
        readers select this zlib/base64 payload instead.
        """
        clean = cls._normalise_ohlcv_frame(frame).tail(max(260, int(max_bars)))
        if clean.empty:
            return "", "ZLIB_CSV_V1", 0
        local = clean.reset_index().rename(columns={"index": "Date"})
        local["Date"] = pd.to_datetime(local["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        raw = local.to_csv(index=False, float_format="%.10g").encode("utf-8")
        compressed = zlib.compress(raw, level=6)
        return base64.b64encode(compressed).decode("ascii"), "ZLIB_CSV_V1", len(clean)

    @classmethod
    def _ohlcv_compact_decode(cls, payload: Any, codec: Any = "ZLIB_CSV_V1") -> pd.DataFrame:
        text = str(payload or "").strip()
        if not text:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        if str(codec or "").upper() != "ZLIB_CSV_V1":
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        try:
            raw = zlib.decompress(base64.b64decode(text.encode("ascii")))
            frame = pd.read_csv(io.BytesIO(raw))
            return cls._normalise_ohlcv_frame(frame)
        except Exception:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    def read_feature_cache(
        self,
        tickers: Sequence[str],
        *,
        expected_session: Any | None = None,
        scanner_version: str = "",
    ) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
        names = list(dict.fromkeys(str(ticker).upper().strip() for ticker in tickers if str(ticker).strip()))
        if not names or self.settings.mode != "SUPABASE_REST" or not self.settings.read_enabled:
            return {}, pd.DataFrame([{"provider": "SUPABASE_FEATURE_CACHE", "status": "DATABASE_UNAVAILABLE", "requested_tickers": len(names)}])
        expected_date = pd.to_datetime(expected_session, errors="coerce")
        expected_text = pd.Timestamp(expected_date).date().isoformat() if pd.notna(expected_date) else ""
        hits: dict[str, dict[str, Any]] = {}
        audits: list[dict[str, Any]] = []
        for start in range(0, len(names), self.settings.read_batch_size):
            chunk = names[start:start + self.settings.read_batch_size]
            try:
                rows = self._get_rows("scanner_feature_cache", {
                    "select": "ticker,last_bar_date,feature_state,source_tier,scanner_version,feature_schema_version,payload,updated_at",
                    "ticker": f"in.({','.join(chunk)})",
                    "limit": str(max(1, len(chunk))),
                })
            except Exception as exc:
                return {}, pd.DataFrame([{
                    "provider": "SUPABASE_FEATURE_CACHE", "status": "READ_FAIL_SOFT",
                    "error": f"{type(exc).__name__}: {str(exc)[:240]}", "requested_tickers": len(names),
                }])
            by_ticker = {str(row.get("ticker", "")).upper().strip(): row for row in rows}
            for ticker in chunk:
                row = by_ticker.get(ticker)
                if not row:
                    audits.append({"ticker": ticker, "provider": "SUPABASE_FEATURE_CACHE", "status": "MISS"})
                    continue
                last_date = str(row.get("last_bar_date") or "")[:10]
                version_ok = (not scanner_version) or str(row.get("scanner_version") or "") == str(scanner_version)
                state_ok = str(row.get("feature_state") or "").upper() == "CURRENT"
                session_ok = (not expected_text) or last_date == expected_text
                payload = row.get("payload") if isinstance(row.get("payload"), Mapping) else {}
                ready = bool(payload) and version_ok and state_ok and session_ok
                if ready:
                    hits[ticker] = dict(payload)
                audits.append({
                    "ticker": ticker, "provider": "SUPABASE_FEATURE_CACHE",
                    "status": "HIT_CURRENT" if ready else "STALE_OR_INCOMPATIBLE",
                    "last_bar_date": last_date or None, "expected_session": expected_text or None,
                    "feature_state": row.get("feature_state"), "source_tier": row.get("source_tier"),
                    "scanner_version": row.get("scanner_version"), "updated_at": row.get("updated_at"),
                })
        return hits, pd.DataFrame(audits)

    def write_feature_cache(
        self,
        payloads: Mapping[str, Mapping[str, Any]],
        *,
        scanner_version: str,
        feature_schema_version: str = "ALL_ELIGIBLE_LITE_V1",
    ) -> pd.DataFrame:
        if not payloads:
            return pd.DataFrame()
        now = datetime.now(timezone.utc).isoformat()
        records: list[dict[str, Any]] = []
        for raw_ticker, raw_payload in payloads.items():
            ticker = str(raw_ticker).upper().strip()
            payload = dict(raw_payload or {})
            if not ticker or not payload:
                continue
            last_bar = pd.to_datetime(payload.get("ohlcv_last_bar_date"), errors="coerce")
            last_text = pd.Timestamp(last_bar).date().isoformat() if pd.notna(last_bar) else None
            records.append({
                "ticker": ticker, "last_bar_date": last_text, "feature_state": "CURRENT" if last_text else "PARTIAL",
                "source_tier": str(payload.get("ohlcv_source_tier") or "DATABASE_OR_PUBLIC"),
                "scanner_version": str(scanner_version), "feature_schema_version": str(feature_schema_version),
                "payload": payload,
                # Feature-cache rows hash their own analytical payload.  The
                # previous v9.8.2 build accidentally referenced OHLCV writer
                # locals (compact_payload/compact_bars/frame) here, which are
                # not in scope and caused cold-cache scans to fail before the
                # first feature-cache write.
                "content_hash": _semantic_hash(payload),
                "updated_at": now,
            })
        if not records:
            return pd.DataFrame()
        try:
            if self.settings.mode == "OUTBOX_ONLY":
                written = self._write_outbox("scanner_feature_cache", records)
            elif self.settings.mode == "SUPABASE_REST":
                written = self._upsert_supabase("scanner_feature_cache", records)
            else:
                written = 0
            return pd.DataFrame([{"provider": "SUPABASE_FEATURE_CACHE", "status": "WRITTEN" if written else "WRITE_SKIPPED", "rows": int(written), "requested": len(records)}])
        except Exception as exc:
            return pd.DataFrame([{"provider": "SUPABASE_FEATURE_CACHE", "status": "WRITE_FAIL_SOFT", "rows": 0, "requested": len(records), "error": f"{type(exc).__name__}: {str(exc)[:240]}"}])

    def read_ohlcv_cache(
        self,
        tickers: Sequence[str],
        *,
        min_bars: int = 60,
    ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
        names = list(dict.fromkeys(str(ticker).upper().strip() for ticker in tickers if str(ticker).strip()))
        if not names or self.settings.mode != "SUPABASE_REST" or not self.settings.read_enabled:
            return {}, pd.DataFrame([{
                "provider": "SUPABASE_OHLCV", "status": "DATABASE_UNAVAILABLE",
                "requested_tickers": len(names), "cache_tickers": 0,
            }])
        histories: dict[str, pd.DataFrame] = {}
        audits: list[dict[str, Any]] = []
        for start in range(0, len(names), self.settings.read_batch_size):
            chunk = names[start:start + self.settings.read_batch_size]
            # v15 selects the compact column only. Legacy JSONB is fetched in a
            # second, bounded query solely for rows that have not yet been
            # converted. This makes every subsequent 400-name warm read much
            # smaller without requiring destructive data migration.
            compact_supported = True
            try:
                rows = self._get_rows("ohlcv_daily_cache", {
                    "select": "ticker,payload_compact,payload_codec,bar_count,first_bar_date,last_bar_date,source_family,source_tier,source_checked_at,refresh_state,last_error,content_hash",
                    "ticker": f"in.({','.join(chunk)})",
                    "limit": str(max(1, len(chunk))),
                })
            except Exception:
                compact_supported = False
                rows = self._get_rows("ohlcv_daily_cache", {
                    "select": "ticker,payload,bar_count,first_bar_date,last_bar_date,source_family,source_tier,source_checked_at,refresh_state,last_error,content_hash",
                    "ticker": f"in.({','.join(chunk)})",
                    "limit": str(max(1, len(chunk))),
                })
            by_ticker = {str(row.get("ticker", "")).upper().strip(): row for row in rows}
            legacy_needed: list[str] = []
            if compact_supported:
                legacy_needed = [
                    ticker for ticker in chunk
                    if ticker in by_ticker and not str(by_ticker[ticker].get("payload_compact") or "").strip()
                ]
                if legacy_needed:
                    try:
                        legacy_rows = self._get_rows("ohlcv_daily_cache", {
                            "select": "ticker,payload",
                            "ticker": f"in.({','.join(legacy_needed)})",
                            "limit": str(max(1, len(legacy_needed))),
                        })
                        for legacy in legacy_rows:
                            key = str(legacy.get("ticker", "")).upper().strip()
                            if key in by_ticker:
                                by_ticker[key]["payload"] = legacy.get("payload")
                    except Exception:
                        pass
            for ticker in chunk:
                row = by_ticker.get(ticker)
                if not row:
                    audits.append({
                        "ticker": ticker, "provider": "SUPABASE_OHLCV",
                        "status": "DATABASE_MISS", "bars": 0,
                        "last_bar_date": None, "refresh_state": "MISSING",
                        "payload_format": "NONE",
                    })
                    continue
                compact = str(row.get("payload_compact") or "").strip()
                if compact:
                    frame = self._ohlcv_compact_decode(compact, row.get("payload_codec"))
                    payload_format = "COMPACT_ZLIB"
                else:
                    frame = self._ohlcv_frame_from_payload(row.get("payload"))
                    payload_format = "LEGACY_JSON" if not frame.empty else "NONE"
                if not frame.empty:
                    histories[ticker] = frame
                bars = int(row.get("bar_count") or len(frame) or 0)
                audits.append({
                    "ticker": ticker, "provider": "SUPABASE_OHLCV",
                    "status": "DATABASE_READY" if bars >= int(min_bars) else "DATABASE_PARTIAL",
                    "bars": bars,
                    "first_bar_date": row.get("first_bar_date"),
                    "last_bar_date": row.get("last_bar_date"),
                    "source_family": row.get("source_family"),
                    "source_tier": row.get("source_tier"),
                    "source_checked_at": row.get("source_checked_at"),
                    "refresh_state": row.get("refresh_state"),
                    "last_error": row.get("last_error"),
                    "payload_format": payload_format,
                    "compact_migration_required": payload_format == "LEGACY_JSON",
                })
        return histories, pd.DataFrame(audits)

    def write_ohlcv_cache(
        self,
        histories: Mapping[str, pd.DataFrame],
        *,
        source_tiers: Mapping[str, str] | None = None,
        refresh_states: Mapping[str, str] | None = None,
        errors: Mapping[str, str] | None = None,
        checked_at: Any | None = None,
        max_bars: int = 900,
    ) -> pd.DataFrame:
        names = list(dict.fromkeys(
            [str(ticker).upper().strip() for ticker in histories]
            + [str(ticker).upper().strip() for ticker in (errors or {})]
        ))
        if not names:
            return pd.DataFrame()
        now = pd.Timestamp(checked_at or datetime.now(timezone.utc))
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        tiers = {str(key).upper().strip(): str(value or "") for key, value in dict(source_tiers or {}).items()}
        states = {str(key).upper().strip(): str(value or "") for key, value in dict(refresh_states or {}).items()}
        failures = {str(key).upper().strip(): str(value or "")[:1200] for key, value in dict(errors or {}).items()}
        records: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        for ticker in names:
            frame = self._normalise_ohlcv_frame(histories.get(ticker)).tail(max(260, int(max_bars)))
            compact_payload, payload_codec, compact_bars = self._ohlcv_compact_encode(frame, max_bars=max_bars)
            tier = tiers.get(ticker, "DATABASE_OR_UNKNOWN")
            upper_tier = tier.upper()
            family = "UNKNOWN"
            for candidate in ("IDX", "ITICK", "TWELVE_DATA", "YAHOO", "STOCKBIT", "GOOGLE_FINANCE"):
                if candidate in upper_tier:
                    family = candidate
                    break
            if family == "UNKNOWN" and tier:
                family = tier.split("_")[0]
            state = states.get(ticker) or ("CURRENT" if compact_bars else "MISSING")
            error = failures.get(ticker, "")
            record = {
                "ticker": ticker,
                # Keep legacy JSON untouched on existing rows; v15 readers do
                # not select it once compact data exists. New rows receive the
                # table default [] while compact history is the primary payload.
                "payload_compact": compact_payload or None,
                "payload_codec": payload_codec,
                "compact_bar_count": int(compact_bars),
                "compact_hash": _semantic_hash(compact_payload) if compact_payload else None,
                "bar_count": int(compact_bars),
                "first_bar_date": pd.Timestamp(frame.index[0]).date().isoformat() if not frame.empty else None,
                "last_bar_date": pd.Timestamp(frame.index[-1]).date().isoformat() if not frame.empty else None,
                "source_family": family,
                "source_tier": tier,
                "source_checked_at": now.isoformat(),
                "refresh_state": state,
                "last_error": error or None,
                "content_hash": _semantic_hash({"compact_hash": _semantic_hash(compact_payload) if compact_payload else None, "bars": int(compact_bars), "last": pd.Timestamp(frame.index[-1]).date().isoformat() if not frame.empty else None}),
                "model_version": "9.6.0",
                "schema_version": DATABASE_SCHEMA_VERSION,
                "created_at": now.isoformat(),
                "updated_at": now.isoformat(),
            }
            records.append(record)
            audit_rows.append({
                "ticker": ticker, "provider": "SUPABASE_OHLCV",
                "status": "WRITE_PENDING", "bars": int(compact_bars),
                "last_bar_date": record["last_bar_date"],
                "refresh_state": state, "error": error,
            })
        if self.settings.mode == "OUTBOX_ONLY":
            written = self._write_outbox("ohlcv_daily_cache", records)
        elif self.settings.mode == "SUPABASE_REST":
            written = self._upsert_supabase("ohlcv_daily_cache", records)
        else:
            written = 0
        for index, row in enumerate(audit_rows):
            row["status"] = "WRITTEN" if index < written else "WRITE_FAILED"
            row["rows_written"] = 1 if index < written else 0
        return pd.DataFrame(audit_rows)

    def read_market_status_cache(self, tickers: Sequence[str], *, max_age_days: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self._read_source_event_cache(
            tickers, event_type="MARKET_STATUS_SNAPSHOT", max_age_days=max_age_days,
            payload_kind="MARKET_STATUS", asof_field="market_status_asof",
        )

    def read_news_review_cache(self, tickers: Sequence[str], *, max_age_days: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self._read_source_event_cache(
            tickers, event_type="NEWS_REVIEW_SNAPSHOT", max_age_days=max_age_days,
            payload_kind="NEWS_REVIEW", asof_field="news_reviewed_at",
        )

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

    def _ensure_scan_id(self, result: Mapping[str, Any]) -> str:
        existing = _clean_text(result.get("scan_id"))
        if existing:
            return existing
        stamp = datetime.now(timezone.utc).isoformat()
        scan_id = hashlib.sha256(
            f"{stamp}|{_clean_text(result.get('scanner_version'))}".encode("utf-8")
        ).hexdigest()[:24]
        if isinstance(result, dict):
            result["scan_id"] = scan_id
        return scan_id

    def build_payloads(self, result: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
        as_of = datetime.now(timezone.utc).isoformat()
        focus = result.get("focus_screens", {}) if isinstance(result.get("focus_screens", {}), Mapping) else {}
        multibagger = focus.get("multibagger", pd.DataFrame())
        fundamentals = result.get("fundamentals", pd.DataFrame())
        fundamental_history = result.get("fundamental_history", pd.DataFrame())
        market_status = result.get("market_status", pd.DataFrame())
        news_review = result.get("news_review", pd.DataFrame())
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
        scan_id = self._ensure_scan_id(result)
        model_version = _clean_text(result.get("scanner_version")) or "8.0.3-ranking-contract-full-universe"
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
                "fundamental_source_families", "fundamental_official_verified", "fundamental_official_source_coverage_pct",
                "fundamental_reconciliation_state", "fundamental_cashflow_statement_coverage_pct", "fundamental_consensus_score", "statement_age_days",
                "fundamental_fetched_at", "database_source_state", "content_hash",
            )),
            "multibagger_snapshots": _frame_records(multibagger, (
                "ticker", "multibagger_status", "multibagger_quality_score", "execution_readiness_score",
                "v9_next_leader_score", "final_score", "real_money_authorization_state",
                "real_money_authorization_pass", "real_money_authorization_blockers", "real_money_manual_checks",
                "real_money_risk_budget_cap_pct", "real_money_risk_budget_idr", "real_money_risk_per_share", "real_money_risk_lots_cap",
                "fundamental_conviction_cap", "fundamental_score_cap_reason",
                "fundamental_data_quality_score", "fundamental_cashflow_state", "fundamental_leverage_risk_state",
                "fundamental_official_state", "fundamental_official_verified", "fundamental_official_source_coverage_pct",
                "fundamental_consensus_score", "fundamental_history_coverage_pct", "fundamental_cashflow_coverage_pct",
                "market_regime", "market_context_score", "market_context_coverage_pct", "market_context_provenance_state",
                "independent_price_verified", "inventory_multi_horizon_score", "distribution_risk_score",
                "reaccumulation_quality_score", "anti_chase_gate",
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
            # V8 removes EOFF/time-cycle from production and does not write
            # placeholder rows to the legacy prediction table.
            "eoff_predictions": [],
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

        if isinstance(market_status, pd.DataFrame) and not market_status.empty and "ticker" in market_status:
            for _, row in market_status.dropna(subset=["ticker"]).drop_duplicates("ticker", keep="last").iterrows():
                payload = {str(key): _json_safe(value) for key, value in row.to_dict().items()}
                ticker = _clean_text(payload.get("ticker")).upper()
                if not ticker:
                    continue
                checked_at = payload.get("market_status_asof") or as_of
                content_hash = _semantic_hash(payload)
                payloads["source_events"].append({
                    "event_key": hashlib.sha256(f"MARKET_STATUS_SNAPSHOT|{ticker}".encode("utf-8")).hexdigest(),
                    "ticker": ticker, "event_type": "MARKET_STATUS_SNAPSHOT",
                    "event_date": _json_safe(pd.to_datetime(checked_at, errors="coerce").date()) if pd.notna(pd.to_datetime(checked_at, errors="coerce")) else None,
                    "source_family": payload.get("market_status_method") or "IDX_MARKET_STATUS",
                    "content_hash": content_hash, "event_fingerprint": content_hash,
                    "refresh_required": not _truthy(payload.get("market_status_score_eligible", payload.get("market_status_verified", False))),
                    "detected_at": checked_at, "last_seen_at": as_of, "payload": payload,
                })

        if isinstance(news_review, pd.DataFrame) and not news_review.empty and "ticker" in news_review:
            for _, row in news_review.dropna(subset=["ticker"]).drop_duplicates("ticker", keep="last").iterrows():
                payload = {str(key): _json_safe(value) for key, value in row.to_dict().items()}
                ticker = _clean_text(payload.get("ticker")).upper()
                if not ticker:
                    continue
                checked_at = payload.get("news_reviewed_at") or as_of
                content_hash = _semantic_hash(payload)
                payloads["source_events"].append({
                    "event_key": hashlib.sha256(f"NEWS_REVIEW_SNAPSHOT|{ticker}".encode("utf-8")).hexdigest(),
                    "ticker": ticker, "event_type": "NEWS_REVIEW_SNAPSHOT",
                    "event_date": _json_safe(pd.to_datetime(checked_at, errors="coerce").date()) if pd.notna(pd.to_datetime(checked_at, errors="coerce")) else None,
                    "source_family": payload.get("news_provider") or payload.get("news_source_family") or "NEWS_REVIEW",
                    "content_hash": content_hash, "event_fingerprint": content_hash,
                    "refresh_required": not _truthy(payload.get("news_score_eligible", False)),
                    "detected_at": checked_at, "last_seen_at": as_of, "payload": payload,
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

    def persist_scan_result(
        self,
        result: Mapping[str, Any],
        *,
        tables: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        scan_id = self._ensure_scan_id(result)
        if self.settings.mode == "DISABLED":
            row = self.status_row("DISABLED_NO_DATABASE", "Scanner tetap berjalan dengan cache lokal; database eksternal belum diaktifkan.")
            row["scan_id"] = scan_id
            return pd.DataFrame([row])
        if self.settings.mode == "CONFIG_INCOMPLETE":
            row = self.status_row("CONFIG_INCOMPLETE", "Aktifkan OUTBOX_ONLY atau isi SUPABASE_URL dan SUPABASE_SECRET_KEY.")
            row["scan_id"] = scan_id
            return pd.DataFrame([row])
        if self.settings.mode == "CONFIG_UNSAFE_KEY":
            row = self.status_row("CONFIG_UNSAFE_KEY", "Reader/writer database membutuhkan backend secret/service-role key; publishable/anon key ditolak.")
            row["scan_id"] = scan_id
            return pd.DataFrame([row])
        payloads = self.build_payloads(result)
        selected_tables = tuple(dict.fromkeys(str(value) for value in (tables or payloads.keys()) if str(value)))
        rows: list[dict[str, Any]] = []
        circuit_reason = ""
        for table in selected_tables:
            records = list(payloads.get(table, []) or [])
            if circuit_reason and self.settings.mode == "SUPABASE_REST":
                row = self.status_row("DATABASE_CIRCUIT_OPEN", circuit_reason)
                row.update({"table": table, "scan_id": scan_id, "rows_attempted": len(records), "rows_written": 0})
                rows.append(row)
                continue
            if not records:
                row = self.status_row("NO_ROWS")
                row.update({"table": table, "scan_id": scan_id})
                rows.append(row)
                continue
            try:
                written = self._write_outbox(table, records) if self.settings.mode == "OUTBOX_ONLY" else self._upsert_supabase(table, records)
                detail = self._write_details.get(table, "")
                state = "OK" if written == len(records) else ("PARTIAL_WRITE" if written > 0 else "DATABASE_FAIL_SOFT")
                row = self.status_row(state, detail)
                row.update({"table": table, "scan_id": scan_id, "rows_attempted": len(records), "rows_written": written})
                if written == 0 and self.settings.mode == "SUPABASE_REST" and self._failure_text_is_systemic(detail):
                    circuit_reason = f"Stopped after systemic write failure on {table}: {detail[:420]}"
            except Exception as exc:
                detail = f"{type(exc).__name__}: {str(exc)[:700]}"
                row = self.status_row("DATABASE_FAIL_SOFT", detail)
                row.update({"table": table, "scan_id": scan_id, "rows_attempted": len(records), "rows_written": 0})
                if self.settings.mode == "SUPABASE_REST" and self._failure_text_is_systemic(detail):
                    circuit_reason = f"Stopped after systemic write exception on {table}: {detail[:420]}"
            rows.append(row)
        return pd.DataFrame(rows)

    def verify_persisted_scan(
        self,
        result: Mapping[str, Any],
        *,
        tables: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Read back persisted keys for the last scan without slowing the scan itself.

        Verification is deliberately user-triggered. It checks exact conflict keys
        for critical decision/evidence tables and reports missing rows explicitly.
        """
        scan_id = self._ensure_scan_id(result)
        if self.settings.mode != "SUPABASE_REST":
            row = self.status_row(
                "VERIFICATION_UNAVAILABLE",
                "Readback verification requires SUPABASE_REST mode.",
            )
            row["scan_id"] = scan_id
            return pd.DataFrame([row])
        payloads = self.build_payloads(result)
        selected = tuple(tables or DATABASE_VERIFICATION_TABLES)
        rows: list[dict[str, Any]] = []
        total_expected = 0
        total_verified = 0
        failed_tables = 0
        for table in selected:
            records = list(payloads.get(table, []) or [])
            conflict_key = TABLE_CONFLICT_TARGETS.get(table, "snapshot_id")
            expected_keys = list(dict.fromkeys(
                _clean_text(record.get(conflict_key))
                for record in records
                if _clean_text(record.get(conflict_key))
            ))
            expected = len(expected_keys)
            total_expected += expected
            if expected == 0:
                row = self.status_row("NO_ROWS", "Tidak ada row yang diharapkan untuk scan ini.")
                row.update({"table": table, "scan_id": scan_id})
                rows.append(row)
                continue
            found: set[str] = set()
            error = ""
            try:
                for start in range(0, expected, self.settings.read_batch_size):
                    chunk = expected_keys[start:start + self.settings.read_batch_size]
                    response_rows = self._get_rows(table, {
                        "select": conflict_key,
                        conflict_key: f"in.({','.join(chunk)})",
                        "limit": str(max(len(chunk), 1)),
                    })
                    found.update(
                        _clean_text(item.get(conflict_key))
                        for item in response_rows
                        if _clean_text(item.get(conflict_key))
                    )
            except Exception as exc:
                error = f"{type(exc).__name__}: {str(exc)[:700]}"
            verified = sum(key in found for key in expected_keys)
            total_verified += verified
            pct = 100.0 * verified / expected if expected else 0.0
            missing = [key for key in expected_keys if key not in found]
            if error:
                state = "READBACK_FAIL_SOFT"
                failed_tables += 1
            elif verified == expected:
                state = "VERIFIED"
            elif verified > 0:
                state = "PARTIAL_VERIFICATION"
                failed_tables += 1
            else:
                state = "NOT_FOUND"
                failed_tables += 1
            detail_parts = []
            if missing:
                detail_parts.append("missing=" + ",".join(missing[:5]))
                if len(missing) > 5:
                    detail_parts.append(f"+{len(missing)-5} more")
            if error:
                detail_parts.append(error)
            row = self.status_row(state, " | ".join(detail_parts) or "Exact persisted keys found.")
            row.update({
                "table": table, "scan_id": scan_id,
                "rows_attempted": expected, "rows_written": expected,
                "rows_verified": verified, "verification_pct": round(pct, 2),
            })
            rows.append(row)
        overall_pct = 100.0 * total_verified / total_expected if total_expected else 0.0
        overall_state = (
            "VERIFIED_ALL_CRITICAL_TABLES"
            if total_expected > 0 and total_verified == total_expected and failed_tables == 0
            else "VERIFICATION_INCOMPLETE"
        )
        summary = self.status_row(
            overall_state,
            f"critical readback {total_verified}/{total_expected} rows across {len(selected)} tables",
        )
        summary.update({
            "table": "__SUMMARY__", "scan_id": scan_id,
            "rows_attempted": total_expected, "rows_written": total_expected,
            "rows_verified": total_verified, "verification_pct": round(overall_pct, 2),
        })
        return pd.DataFrame([summary, *rows])

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
            try:
                self._get_rows("scan_jobs", {"select": "job_id,phase,status", "limit": "1"})
                self._get_rows("scan_job_items", {"select": "item_key,job_id,status", "limit": "1"})
                self._get_rows("scan_job_artifacts", {"select": "artifact_key,job_id,artifact_type", "limit": "1"})
            except DatabaseReadError as exc:
                migration_missing = exc.status_code in {404, 400} and any(
                    token in exc.body.upper()
                    for token in ("PGRST204", "PGRST205", "RELATION", "COLUMN", "SCHEMA CACHE", "NOT FIND")
                )
                if migration_missing:
                    return self.status_row(
                        "MIGRATION_REQUIRED_V12",
                        "Migration v11 tersedia, tetapi migration_v12_resumable_scan_jobs.sql belum diterapkan.",
                    )
                raise
            try:
                self._get_rows("ohlcv_daily_cache", {"select": "ticker,last_bar_date,refresh_state", "limit": "1"})
            except DatabaseReadError as exc:
                migration_missing = exc.status_code in {404, 400} and any(
                    token in exc.body.upper()
                    for token in ("PGRST204", "PGRST205", "RELATION", "COLUMN", "SCHEMA CACHE", "NOT FIND")
                )
                if migration_missing:
                    return self.status_row(
                        "MIGRATION_REQUIRED_V13",
                        "Migration v12 tersedia, tetapi migration_v13_persistent_ohlcv.sql belum diterapkan.",
                    )
                raise
            try:
                self._get_rows(
                    "fundamental_snapshots",
                    {
                        "select": (
                            "snapshot_id,fundamental_official_source_coverage_pct,"
                            "fundamental_reconciliation_state,"
                            "fundamental_cashflow_statement_coverage_pct,"
                            "fundamental_consensus_score"
                        ),
                        "limit": "1",
                    },
                )
                self._get_rows(
                    "multibagger_snapshots",
                    {
                        "select": (
                            "snapshot_id,real_money_authorization_state,"
                            "real_money_authorization_pass,fundamental_conviction_cap,"
                            "market_context_score,independent_price_verified"
                        ),
                        "limit": "1",
                    },
                )
            except DatabaseReadError as exc:
                migration_missing = exc.status_code in {404, 400} and any(
                    token in exc.body.upper()
                    for token in ("PGRST204", "PGRST205", "RELATION", "COLUMN", "SCHEMA CACHE", "NOT FIND")
                )
                if migration_missing:
                    return self.status_row(
                        "MIGRATION_REQUIRED_V14",
                        "Migration v13 tersedia, tetapi migration_v14_guarded_real_money.sql belum diterapkan. "
                        "Jalankan migration v14 sebelum memakai persistence real-money v9.8.0.",
                    )
                raise
            try:
                self._get_rows(
                    "ohlcv_daily_cache",
                    {"select": "ticker,payload_compact,payload_codec,compact_bar_count,compact_hash", "limit": "1"},
                )
                self._get_rows(
                    "scanner_feature_cache",
                    {"select": "ticker,last_bar_date,feature_state,scanner_version,feature_schema_version", "limit": "1"},
                )
            except DatabaseReadError as exc:
                migration_missing = exc.status_code in {404, 400} and any(
                    token in exc.body.upper()
                    for token in ("PGRST204", "PGRST205", "RELATION", "COLUMN", "SCHEMA CACHE", "NOT FIND")
                )
                if migration_missing:
                    return self.status_row(
                        "MIGRATION_REQUIRED_V15",
                        "Migration v14 tersedia, tetapi migration_v15_database_acceleration.sql belum diterapkan. "
                        "Jalankan migration v15 untuk feature cache + compact OHLCV v9.8.2.",
                    )
                raise
            return self.status_row(
                "HEALTHY_V15_DATABASE_ACCELERATION",
                f"HTTP 200; schema {self.settings.schema}; feature cache, compact OHLCV, official reconciliation, and guarded real-money fields ready",
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
        response = self._request(
            "post", endpoint, operation=f"UPSERT {table}", retry_safe=True,
            headers=self._headers(), json=batch,
        )
        status_code = int(getattr(response, "status_code", 0) or 0)
        if bool(getattr(response, "ok", 200 <= status_code < 300)):
            return len(batch)
        hint_record = batch[0] if len(batch) == 1 else {}
        hint = _clean_text(hint_record.get("ticker") or hint_record.get("provider") or hint_record.get("scan_id") or hint_record.get(conflict))
        raise DatabaseWriteError(table, status_code or None, getattr(response, "text", ""), hint)

    @staticmethod
    def _write_error_is_systemic(exc: DatabaseWriteError) -> bool:
        status = int(exc.status_code or 0)
        body = str(exc.body or "").upper()
        # Transport/server/rate/auth/schema failures apply to the whole batch.
        # Recursively bisecting these errors only multiplies network requests and
        # can trap FINALIZING for tens of minutes.
        if status in {401, 403, 404, 408, 425, 429, 500, 502, 503, 504}:
            return True
        systemic_tokens = (
            "PGRST204", "PGRST205", "SCHEMA CACHE", "RELATION",
            "COLUMN", "PERMISSION", "AUTHORIZATION", "JWT",
        )
        return any(token in body for token in systemic_tokens)

    def _write_with_isolation(
        self,
        table: str,
        batch: list[dict[str, Any]],
        failures: list[str],
        *,
        depth: int = 0,
        request_budget: list[int] | None = None,
    ) -> int:
        if not batch:
            return 0
        budget = request_budget if request_budget is not None else [0]
        # A bounded budget guarantees one broken table cannot monopolize the
        # finalizer. 100-row batches normally need one request; isolation is
        # reserved for likely row-level 4xx errors only.
        if budget[0] >= 12:
            failures.append(f"{table}: isolation request budget exhausted; rows={len(batch)}")
            return 0
        budget[0] += 1
        try:
            return self._post_batch(table, batch)
        except DatabaseWriteError as exc:
            if self._write_error_is_systemic(exc):
                failures.append(str(exc))
                return 0
            if len(batch) == 1 or depth >= 4:
                failures.append(str(exc))
                return 0
            # 413 means the payload itself is too large and benefits from
            # splitting. 400/409/422 may be row-specific, so bounded isolation
            # can identify the bad subset without exploding request count.
            if int(exc.status_code or 0) not in {400, 409, 413, 422}:
                failures.append(str(exc))
                return 0
            midpoint = max(1, len(batch) // 2)
            left = self._write_with_isolation(
                table, batch[:midpoint], failures, depth=depth + 1, request_budget=budget
            )
            right = self._write_with_isolation(
                table, batch[midpoint:], failures, depth=depth + 1, request_budget=budget
            )
            return left + right

    @staticmethod
    def _failure_text_is_systemic(detail: str) -> bool:
        text = str(detail or "").upper()
        tokens = (
            "HTTP 401", "HTTP 403", "HTTP 404", "HTTP 408",
            "HTTP 425", "HTTP 429", "HTTP 500", "HTTP 502",
            "HTTP 503", "HTTP 504", "PGRST204", "PGRST205",
            "SCHEMA CACHE", "RELATION", "PERMISSION", "AUTHORIZATION", "JWT",
        )
        return any(token in text for token in tokens)

    def _upsert_supabase(self, table: str, records: list[dict[str, Any]]) -> int:
        written = 0
        failures: list[str] = []
        for start in range(0, len(records), 100):
            before = len(failures)
            written += self._write_with_isolation(table, records[start:start + 100], failures)
            # A provider/auth/schema outage will affect every subsequent 100-row
            # slice too. Open a local circuit breaker after the first systemic
            # failure instead of waiting through the same timeouts repeatedly.
            new_failures = failures[before:]
            if new_failures and any(self._failure_text_is_systemic(value) for value in new_failures):
                break
        if failures:
            preview = " | ".join(failures[:3])
            suffix = f" | +{len(failures) - 3} row errors" if len(failures) > 3 else ""
            self._write_details[table] = f"isolated_failures={len(failures)}; {preview}{suffix}"[:1400]
        else:
            self._write_details[table] = "all_rows_written"
        return written


__all__ = [
    "DATABASE_BRIDGE_VERSION", "DATABASE_SCHEMA_VERSION", "DATABASE_VERIFICATION_TABLES", "DatabaseSettings",
    "DatabaseWriteError", "DatabaseReadError", "DatabaseTransportError", "ScannerDatabaseBridge",
    "TABLE_FIELD_TYPES", "TABLE_CONFLICT_TARGETS", "_normalise_record",
    "_snapshot_id", "_provider_health_frame", "_freshness_state", "_semantic_hash",
]
