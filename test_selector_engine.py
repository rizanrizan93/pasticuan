from __future__ import annotations

import sys
import unittest
from types import ModuleType
from unittest import mock

if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = ModuleType("streamlit")

import numpy as np
import pandas as pd

from ai_engine import LocalAIConfig, update_outcome_memory
from scanner import (
    ScanConfig,
    build_fundamental_history_features,
    prepare_indicators,
)
from scanner_database import ScannerDatabaseBridge
from scanner_focus import scan_multibagger_candidates
from selector_engine import (
    SelectorConfig,
    _bootstrap_mean_ci,
    _spearman,
    _statistical_backend,
    _technical_feature_frame,
    attach_setups_to_selector,
    build_cross_sectional_selector,
    selector_snapshot_frame,
    update_selector_outcomes,
)


def _ohlcv(close: np.ndarray, volume: np.ndarray | None = None) -> pd.DataFrame:
    values = np.asarray(close, dtype=float)
    volume_values = (
        np.asarray(volume, dtype=float)
        if volume is not None else np.full(len(values), 4_000_000.0)
    )
    open_values = np.r_[values[0] * 0.998, values[:-1]]
    return pd.DataFrame(
        {
            "Open": open_values,
            "High": np.maximum(open_values, values) * 1.008,
            "Low": np.minimum(open_values, values) * 0.992,
            "Close": values,
            "Volume": volume_values,
        },
        index=pd.bdate_range("2023-01-02", periods=len(values)),
    )


def _prepared_cross_section(count: int = 8, bars: int = 320) -> dict[str, pd.DataFrame]:
    timeline = np.arange(bars, dtype=float)
    benchmark = _ohlcv(1000.0 * np.exp(0.00025 * timeline))
    prepared: dict[str, pd.DataFrame] = {}
    for index in range(count):
        daily_drift = -0.00015 + index * 0.00013
        seasonal = 0.012 * np.sin(timeline / (15.0 + index))
        close = (700.0 + 25.0 * index) * np.exp(daily_drift * timeline + seasonal)
        volume = np.full(bars, 2_500_000.0 + 350_000.0 * index)
        volume[-40:] *= 1.0 + 0.03 * index
        prepared[f"T{index:02d}.JK"] = prepare_indicators(
            _ohlcv(close, volume), benchmark,
        )
    return prepared


class CrossSectionalSelectorTests(unittest.TestCase):
    def test_spearman_metric_does_not_require_scipy(self):
        left = pd.Series([1.0, 2.0, 3.0])
        right = pd.Series([1.0, 3.0, 2.0])
        with mock.patch("selector_engine._scipy_stats", None):
            result = _spearman(left, right)
        self.assertAlmostEqual(result, 0.5, places=12)

    def test_python312_runtime_uses_scipy_118_bca(self):
        self.assertEqual(_statistical_backend(), "SCIPY_1.18.0")
        low, high, method = _bootstrap_mean_ci(
            [0.4, 0.8, 1.0, 1.2, 1.6, 1.8],
            SelectorConfig(statistical_bootstrap_resamples=299),
        )
        self.assertEqual(method, "SCIPY_BCA")
        self.assertLess(low, high)
        self.assertGreater(high, 0.0)

    def test_gradual_momentum_is_not_treated_like_one_day_jump(self):
        bars = 90
        smooth = np.linspace(100.0, 130.0, bars)
        jump = np.full(bars, 100.0)
        jump[-10:] = 130.0
        smooth_features = _technical_feature_frame(_ohlcv(smooth))
        jump_features = _technical_feature_frame(_ohlcv(jump))
        self.assertGreater(
            float(smooth_features.iloc[-1]["positive_day_ratio20"]),
            float(jump_features.iloc[-1]["positive_day_ratio20"]),
        )
        self.assertLess(
            float(smooth_features.iloc[-1]["jump_concentration20"]),
            float(jump_features.iloc[-1]["jump_concentration20"]),
        )

    def test_selector_outputs_three_excess_return_horizons(self):
        prepared = _prepared_cross_section()
        profiles = {
            ticker: {
                "silent_accumulation_score": 30.0 + 8.0 * index,
                "silent_accumulation_state": "ACCUMULATION",
            }
            for index, ticker in enumerate(prepared)
        }
        selector, audit, panel = build_cross_sectional_selector(
            prepared,
            SelectorConfig(
                min_history_bars=120,
                training_lookback_bars=180,
                anchor_step_bars=10,
                min_cross_section=4,
                min_training_rows=40,
                min_evaluation_rows=12,
                min_evaluation_dates=2,
                min_evaluation_tickers=25,
                logistic_iterations=50,
            ),
            profiles,
        )
        self.assertEqual(len(selector), 8)
        self.assertFalse(panel.empty)
        self.assertEqual(set(audit["model"]), {
            "RULE_ENGINE", "INDEPENDENT_SELECTOR",
            "AI_CHALLENGER", "RELATIVE_STRENGTH",
        })
        for column in (
            "statistical_backend", "brier_skill_ci_low_pct",
            "net_excess_expectancy_ci_low_pct",
            "ai_vs_baseline_advantage_ci_low_pct",
            "ai_vs_baseline_pvalue_adjusted",
            "ai_promotion_gate_reason",
        ):
            self.assertIn(column, audit)
        for horizon in (5, 20, 60):
            self.assertIn(f"selector_expected_excess_return_{horizon}d_pct", selector)
            self.assertIn(f"selector_outperform_probability_{horizon}d_pct", selector)
            self.assertIn(f"selector_score_{horizon}d", selector)
        self.assertEqual(selector.iloc[0]["ticker"], "T07.JK")
        self.assertTrue(selector["selection_rank"].is_monotonic_increasing)

    def test_ten_name_cohort_never_promotes_ai(self):
        prepared = _prepared_cross_section(count=10)
        selector, audit, _ = build_cross_sectional_selector(
            prepared,
            SelectorConfig(
                min_history_bars=120,
                training_lookback_bars=180,
                anchor_step_bars=10,
                min_cross_section=4,
                min_training_rows=40,
                min_evaluation_rows=12,
                min_evaluation_dates=2,
                min_evaluation_tickers=25,
                logistic_iterations=50,
            ),
            {},
        )
        ai_rows = audit[audit["model"].eq("AI_CHALLENGER")]
        self.assertTrue(ai_rows["ai_promotion_state"].eq("INSUFFICIENT_EVIDENCE").all())
        self.assertTrue(ai_rows["ai_can_influence"].eq(False).all())
        self.assertTrue(
            selector["selector_model_state"].eq("INSUFFICIENT_EVIDENCE").all()
        )
        self.assertTrue(
            selector.filter(like="selector_ai_weight_").fillna(0.0).eq(0.0).all().all()
        )

    def test_multibagger_exposes_research_quality_pillars(self):
        prepared = _prepared_cross_section(count=1)
        ticker = next(iter(prepared))
        fundamentals = pd.DataFrame([{
            "ticker": ticker,
            "fundamental_coverage": 88.0,
            "fundamental_score": 84.0,
            "fundamental_score_10": 8.4,
            "fundamental_reliability": "HIGH",
            "fundamental_data_grade": "B",
            "fundamental_source_count": 2,
            "fundamental_history_quarters": 8,
            "fundamental_history_years": 2,
            "fundamental_history_coverage": 85.0,
            "fundamental_consensus_score": 90.0,
            "fundamental_official_reference": True,
            "fundamental_official_verified": False,
            "statement_age_days": 60,
            "revenue_growth": 0.20,
            "earnings_growth": 0.24,
            "roe": 0.19,
            "roa": 0.10,
            "operating_margin": 0.18,
            "net_margin": 0.13,
            "debt_equity": 0.55,
            "current_ratio": 1.8,
            "cash_to_debt": 0.8,
            "free_cash_flow": 1_000_000_000.0,
            "history_cash_conversion": 1.05,
            "history_positive_ocf_ratio": 1.0,
            "history_positive_earnings_ratio": 1.0,
            "history_margin_stability": 0.85,
            "history_share_dilution_yoy": 0.0,
            "history_roic_proxy": 0.16,
            "history_gross_profitability": 0.24,
            "history_gross_margin": 0.35,
            "history_gross_profit_growth": 0.18,
            "history_accruals_to_assets": -0.01,
            "history_leverage_change_yoy": -0.02,
            "history_net_debt_ebitda": 0.8,
            "history_interest_coverage": 9.0,
            "market_cap": 5_000_000_000_000.0,
            "fundamental_model": "GENERAL",
            "sector": "Industrials",
        }])
        result = scan_multibagger_candidates(
            prepared,
            fundamentals,
            config=ScanConfig().replace(time_cycle_enabled=False),
        )
        for column in (
            "growth_persistence_pillar", "profitability_pillar",
            "cash_conversion_pillar", "balance_sheet_safety_pillar",
            "reinvestment_runway_pillar", "quality_pillar_coverage_pct",
            "quality_pillar_gate",
        ):
            self.assertIn(column, result)
        self.assertGreaterEqual(float(result.iloc[0]["quality_pillars_strong"]), 3)

    def test_low_quality_multibagger_defers_full_timecycle(self):
        prepared = _prepared_cross_section(count=1)
        ticker = next(iter(prepared))
        fundamentals = pd.DataFrame([{
            "ticker": ticker,
            "fundamental_coverage": 50.0,
            "fundamental_score": 35.0,
            "fundamental_score_10": 3.5,
            "fundamental_reliability": "LOW",
            "fundamental_data_grade": "D",
            "revenue_growth": -0.10,
            "earnings_growth": -0.20,
            "roe": 0.02,
            "roa": 0.01,
            "net_margin": 0.01,
            "debt_equity": 2.5,
            "current_ratio": 0.7,
            "cash_to_debt": 0.05,
            "free_cash_flow": -1.0,
            "history_cash_conversion": 0.2,
            "history_positive_ocf_ratio": 0.25,
            "history_positive_earnings_ratio": 0.25,
            "history_share_dilution_yoy": 0.15,
            "history_net_debt_ebitda": 5.0,
            "history_interest_coverage": 1.0,
            "fundamental_model": "GENERAL",
        }])
        with mock.patch("scanner_focus.analyze_time_cycle") as timing:
            result = scan_multibagger_candidates(
                prepared,
                fundamentals,
                config=ScanConfig().replace(time_cycle_enabled=True),
            )
        timing.assert_not_called()
        self.assertEqual(
            result.iloc[0]["time_cycle_evaluation_mode"],
            "DEFERRED_LOW_QUALITY",
        )


class FundamentalQualityResearchTests(unittest.TestCase):
    def test_history_features_include_gross_profitability_and_accruals(self):
        periods = pd.date_range("2024-03-31", periods=8, freq="QE")
        rows = []
        for index, period in enumerate(periods):
            revenue = 100.0 + 8.0 * index
            rows.append({
                "ticker": "QUALITY.JK",
                "period_end": period,
                "period_type": f"Q{index % 4 + 1}",
                "source_family": "IDX_OFFICIAL_XBRL",
                "source_verified": True,
                "currency": "IDR",
                "revenue": revenue,
                "gross_profit": 0.36 * revenue,
                "operating_income": 0.20 * revenue,
                "ebit": 0.19 * revenue,
                "ebitda": 0.24 * revenue,
                "net_income": 0.14 * revenue,
                "operating_cash_flow": 0.16 * revenue,
                "capex": -0.04 * revenue,
                "total_assets": 500.0 + 15.0 * index,
                "total_liabilities": 180.0 + 4.0 * index,
                "equity": 320.0 + 11.0 * index,
                "total_debt": 80.0,
                "cash": 60.0,
                "shares_outstanding": 100.0,
                "interest_expense": 2.0,
            })
        features = build_fundamental_history_features(
            pd.DataFrame(rows),
            now=pd.Timestamp("2026-04-15"),
        )
        row = features.iloc[0]
        self.assertGreater(float(row["history_gross_profitability"]), 0.0)
        self.assertGreater(float(row["history_gross_margin"]), 0.30)
        self.assertLess(float(row["history_accruals_to_assets"]), 0.0)

    def test_selection_rank_is_not_replaced_by_setup_readiness(self):
        selector = pd.DataFrame([
            {
                "ticker": "LEADER.JK", "swing_selection_score": 92.0,
                "technical_selection_score": 90.0,
                "silent_accumulation_score": 88.0,
                "selection_risks": "Tidak ada risiko besar",
            },
            {
                "ticker": "SETUP.JK", "swing_selection_score": 55.0,
                "technical_selection_score": 52.0,
                "silent_accumulation_score": 45.0,
                "selection_risks": "Trend lemah",
            },
        ])
        setups = pd.DataFrame([{
            "ticker": "SETUP.JK", "setup": "BREAKOUT_RETEST",
            "status": "EXECUTION_READY", "action": "READY_TRIGGER",
            "entry": 100.0, "stop_loss": 94.0, "tp1": 112.0,
            "rr1": 2.0,
        }])
        radar = attach_setups_to_selector(selector, setups)
        self.assertEqual(radar.iloc[0]["ticker"], "LEADER.JK")
        self.assertEqual(radar.iloc[0]["active_setup"], "NO_SETUP")
        self.assertEqual(radar.iloc[1]["setup_status"], "EXECUTION_READY")
        self.assertIn("tetap di radar", radar.iloc[0]["not_entry_reason"])

    def test_selector_outcomes_are_idempotent_and_resolve(self):
        prepared = _prepared_cross_section(count=8)
        ticker = next(iter(prepared))
        frame = prepared[ticker]
        as_of = frame.index[-8]
        selector = pd.DataFrame([{
            "ticker": ticker, "as_of": as_of, "selection_rank": 1,
            "swing_selection_score": 80.0,
            "multibagger_timing_selector_score": 75.0,
            "technical_selection_score": 82.0,
            "silent_accumulation_score": 70.0,
            "relative_strength_score": 90.0,
            "selector_expected_excess_return_5d_pct": 1.2,
            "selector_outperform_probability_5d_pct": 61.0,
            "selector_score_5d": 85.0,
            "selector_ai_weight_5d_pct": 0.0,
            "selector_model_state_5d": "SHADOW_CHALLENGER",
            "selector_champion_5d": "RULE_ENGINE",
            "selector_version": "test",
        }])
        memory = update_selector_outcomes(
            pd.DataFrame(), selector, prepared,
            SelectorConfig(horizons=(5,)),
        )
        repeated = update_selector_outcomes(
            memory, selector, prepared,
            SelectorConfig(horizons=(5,)),
        )
        self.assertEqual(len(repeated), 1)
        self.assertEqual(repeated.iloc[0]["outcome_status"], "RESOLVED")
        self.assertIn("net_excess_return_pct", repeated)


class ExecutionOutcomePersistenceTests(unittest.TestCase):
    def test_execution_outcome_records_fill_mfe_mae_and_net_cost(self):
        index = pd.bdate_range("2026-01-05", periods=5)
        frame = pd.DataFrame(
            {
                "Open": [101.0, 100.0, 103.0, 110.0, 111.0],
                "High": [102.0, 105.0, 112.0, 113.0, 114.0],
                "Low": [100.0, 99.0, 98.0, 109.0, 110.0],
                "Close": [101.0, 103.0, 111.0, 112.0, 113.0],
                "Volume": [1_000_000.0] * 5,
            },
            index=index,
        )
        memory = pd.DataFrame([{
            "signal_id": "signal-1", "ticker": "TEST.JK",
            "strategy": "BREAKOUT_RETEST", "signal_date": index[0],
            "memory_state": "OPEN", "entry": 100.0, "stop_loss": 95.0,
            "tp1": 110.0, "tp2": 120.0,
        }])
        resolved = update_outcome_memory(
            pd.DataFrame(), {"TEST.JK": frame}, memory,
            LocalAIConfig(memory_entry_window_bars=2, memory_horizon_bars=4),
        )
        row = resolved.iloc[0]
        self.assertEqual(row["memory_state"], "RESOLVED")
        self.assertTrue(bool(row["filled"]))
        self.assertTrue(bool(row["tp1_before_sl"]))
        self.assertGreater(float(row["mfe_r"]), 0.0)
        self.assertLess(float(row["mae_r"]), 0.0)
        self.assertLess(float(row["net_return_pct"]), float(row["gross_return_pct"]))
        self.assertAlmostEqual(float(row["roundtrip_cost_pct"]), 0.65, places=2)

    def test_database_payload_contains_all_new_persistent_layers(self):
        selector = pd.DataFrame([{
            "ticker": "TEST.JK", "as_of": pd.Timestamp("2026-07-28"),
            "selection_rank": 1, "swing_selection_score": 80.0,
            "technical_selection_score": 82.0,
            "silent_accumulation_score": 75.0,
            "relative_strength_score": 90.0,
            "selector_expected_excess_return_5d_pct": 1.5,
            "selector_outperform_probability_5d_pct": 62.0,
            "selector_score_5d": 84.0,
            "selector_ai_weight_5d_pct": 0.0,
            "selector_model_state_5d": "SHADOW_CHALLENGER",
            "selector_champion_5d": "RULE_ENGINE",
            "selector_version": "test-selector",
        }])
        snapshots = selector_snapshot_frame(selector)
        self.assertEqual(len(snapshots), 3)
        execution = pd.DataFrame([{
            "signal_id": "signal-1", "ticker": "TEST.JK",
            "strategy": "BREAKOUT_RETEST",
            "signal_date": pd.Timestamp("2026-07-28"),
            "memory_state": "RESOLVED", "filled": True,
            "tp1_before_sl": True, "r_multiple": 1.2,
            "mfe_r": 1.6, "mae_r": -0.3,
            "custom_training_feature": 77.0,
        }])
        outcomes = pd.DataFrame([{
            "outcome_id": "outcome-1", "snapshot_id": "snapshot-1",
            "ticker": "TEST.JK", "signal_date": pd.Timestamp("2026-07-28"),
            "horizon": "5D", "horizon_bars": 5,
            "outcome_status": "OPEN",
        }])
        audit = pd.DataFrame([{
            "horizon": "5D", "horizon_bars": 5, "model": "RULE_ENGINE",
            "selector_version": "test-selector", "training_rows": 100,
            "model_fit_rows": 100, "calibration_rows": 20,
            "evaluation_rows": 20, "evaluation_dates": 4,
            "evaluation_tickers": 30, "ai_promotion_state": "SHADOW",
            "ai_can_influence": False,
            "statistical_backend": "SCIPY_1.18.0",
            "brier_skill_ci_low_pct": -0.5,
            "ai_promotion_gate_reason": "lower CI Brier skill belum positif",
        }])
        payloads = ScannerDatabaseBridge().build_payloads({
            "scanner_version": "7.3.0-test",
            "focus_screens": {
                "stock_selector": selector,
                "selector_model_audit": audit,
                "selector_outcomes": outcomes,
                "ai_outcome_memory": execution,
            },
            "selector_outcomes": outcomes,
            "ai_outcome_memory": execution,
            "prepared": {},
        })
        for table in (
            "ai_execution_outcomes", "selector_snapshots",
            "selector_outcomes", "selector_model_evaluations",
        ):
            self.assertIn(table, payloads)
            self.assertTrue(payloads[table])
        ai_payload = payloads["ai_execution_outcomes"][0]["payload"]
        self.assertEqual(ai_payload["custom_training_feature"], 77.0)
        evaluation_payload = payloads["selector_model_evaluations"][0]["payload"]
        self.assertEqual(evaluation_payload["statistical_backend"], "SCIPY_1.18.0")
        self.assertEqual(
            evaluation_payload["ai_promotion_gate_reason"],
            "lower CI Brier skill belum positif",
        )


if __name__ == "__main__":
    unittest.main()
