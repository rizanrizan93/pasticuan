from __future__ import annotations

import sys
import unittest
from types import ModuleType

if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = ModuleType("streamlit")

import numpy as np
import pandas as pd

from ai_engine import LocalAIConfig, update_outcome_memory
from scanner import prepare_indicators
from scanner_database import ScannerDatabaseBridge
from selector_engine import (
    SelectorConfig,
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


if __name__ == "__main__":
    unittest.main()
