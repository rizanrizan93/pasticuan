from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

if "requests" not in sys.modules:
    try:
        import requests  # noqa: F401
    except ModuleNotFoundError:
        requests_stub = types.ModuleType("requests")

        class RequestException(Exception):
            pass

        class Timeout(RequestException):
            pass

        class Session:
            pass

        requests_stub.RequestException = RequestException
        requests_stub.Timeout = Timeout
        requests_stub.Session = Session
        requests_stub.get = lambda *args, **kwargs: None
        requests_stub.post = lambda *args, **kwargs: None
        sys.modules["requests"] = requests_stub

if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = types.ModuleType("streamlit")

import dashboard_v660
import scanner_database
from ihsg_direction import (
    IHSGDirectionConfig,
    IHSG_DIRECTION_VERSION,
    analyze_ihsg_direction,
    build_ihsg_feature_frame,
    ihsg_snapshot_frame,
    update_ihsg_outcomes,
)


def make_market(
    *,
    bars: int = 620,
    drift: float = 0.00025,
    volatility: float = 0.008,
    seed: int = 11,
    end: str = "2026-07-27",
    start_price: float = 6500.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end=end, periods=bars)
    returns = drift + rng.normal(0.0, volatility, bars)
    close = start_price * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) * 1.006,
            "Low": np.minimum(open_, close) * 0.994,
            "Close": close,
            "Volume": rng.integers(10_000_000, 40_000_000, bars),
        },
        index=index,
    )


def make_universe(index: pd.DatetimeIndex, count: int = 10) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for number in range(count):
        rng = np.random.default_rng(100 + number)
        returns = 0.0002 + rng.normal(0.0, 0.012, len(index))
        close = (400.0 + 25.0 * number) * np.exp(np.cumsum(returns))
        open_ = np.r_[close[0], close[:-1]]
        result[f"T{number:02d}.JK"] = pd.DataFrame(
            {
                "Open": open_,
                "High": np.maximum(open_, close) * 1.008,
                "Low": np.minimum(open_, close) * 0.992,
                "Close": close,
                "Volume": np.full(len(index), 4_000_000 + number * 100_000),
            },
            index=index,
        )
    return result


class IHSGDirectionEngineTests(unittest.TestCase):
    def small_config(self) -> IHSGDirectionConfig:
        return IHSGDirectionConfig(
            min_history_bars=220,
            min_train_bars=160,
            min_analogues=20,
            max_analogues=48,
            analogue_spacing_bars=4,
            validation_points=24,
            min_validation_predictions=16,
            min_directional_validation_predictions=5,
            min_features=7,
        )

    def test_insufficient_history_fails_closed(self):
        benchmark = make_market(bars=120)
        result = analyze_ihsg_direction(
            benchmark,
            now="2026-07-28 08:00+07:00",
            config=self.small_config(),
        )
        self.assertEqual(result["data_state"], "INSUFFICIENT_HISTORY")
        self.assertTrue(result["horizons"]["prediction_state"].eq("ABSTAIN").all())
        self.assertLessEqual(result["risk_budget_multiplier"], 0.50)

    def test_probabilities_sum_and_risk_cap_never_exceeds_one(self):
        benchmark = make_market()
        universe = make_universe(benchmark.index, 12)
        result = analyze_ihsg_direction(
            benchmark,
            universe,
            now="2026-07-28 08:00+07:00",
            config=self.small_config(),
            eod_final=True,
        )
        self.assertEqual(result["data_state"], "READY")
        self.assertGreaterEqual(result["breadth_member_count"], 8)
        for _, row in result["horizons"].iterrows():
            probabilities = (
                float(row["prob_up_pct"])
                + float(row["prob_sideways_pct"])
                + float(row["prob_down_pct"])
            )
            self.assertAlmostEqual(probabilities, 100.0, delta=0.2)
        self.assertGreaterEqual(result["risk_budget_multiplier"], 0.20)
        self.assertLessEqual(result["risk_budget_multiplier"], 1.00)

    def test_feature_frame_is_causal_for_unchanged_prefix(self):
        benchmark = make_market(bars=500)
        changed = benchmark.copy()
        changed.iloc[-20:, changed.columns.get_loc("Close")] *= np.linspace(1.0, 1.30, 20)
        changed.iloc[-20:, changed.columns.get_loc("High")] = np.maximum(
            changed.iloc[-20:]["High"], changed.iloc[-20:]["Close"]
        )
        prefix_position = -30
        first = build_ihsg_feature_frame(benchmark)
        second = build_ihsg_feature_frame(changed)
        pd.testing.assert_series_equal(
            first.iloc[prefix_position][list(first.columns)],
            second.iloc[prefix_position][list(second.columns)],
            check_names=False,
        )

    def test_bear_structure_never_gets_full_risk_budget(self):
        rng = np.random.default_rng(44)
        bars = 620
        index = pd.bdate_range(end="2026-07-27", periods=bars)
        returns = np.r_[
            rng.normal(0.0001, 0.006, 350),
            rng.normal(-0.0012, 0.009, bars - 350),
        ]
        close = 7000.0 * np.exp(np.cumsum(returns))
        open_ = np.r_[close[0], close[:-1]]
        benchmark = pd.DataFrame(
            {
                "Open": open_,
                "High": np.maximum(open_, close) * 1.005,
                "Low": np.minimum(open_, close) * 0.995,
                "Close": close,
                "Volume": 20_000_000,
            },
            index=index,
        )
        result = analyze_ihsg_direction(
            benchmark,
            now="2026-07-28 08:00+07:00",
            config=self.small_config(),
        )
        self.assertIn(result["regime"], {"BEAR_CONFIRMED", "BEAR_RALLY", "TRANSITION"})
        self.assertLessEqual(result["risk_budget_multiplier"], 0.70)

    def test_predictable_repeating_regime_can_leave_abstain(self):
        bars = 620
        index = pd.bdate_range(end="2026-07-27", periods=bars)
        rng = np.random.default_rng(3)
        state = ((np.arange(bars) // 25) % 2) * 2 - 1
        returns = state * 0.0035 + rng.normal(0.0, 0.0010, bars)
        close = 6500.0 * np.exp(np.cumsum(returns))
        open_ = np.r_[close[0], close[:-1]]
        benchmark = pd.DataFrame(
            {
                "Open": open_,
                "High": np.maximum(open_, close) * 1.003,
                "Low": np.minimum(open_, close) * 0.997,
                "Close": close,
                "Volume": 20_000_000,
            },
            index=index,
        )
        result = analyze_ihsg_direction(
            benchmark,
            now="2026-07-28 08:00+07:00",
            config=self.small_config(),
            eod_final=True,
        )
        self.assertGreaterEqual(int(result["horizons"]["actionable"].sum()), 1)
        self.assertTrue(result["horizons"]["validation_state"].eq("OOS_POSITIVE").any())

    def test_incomplete_eod_forces_abstain(self):
        benchmark = make_market()
        result = analyze_ihsg_direction(
            benchmark,
            now="2026-07-28 08:00+07:00",
            config=self.small_config(),
            eod_final=False,
        )
        self.assertTrue(result["horizons"]["prediction_state"].eq("ABSTAIN").all())
        self.assertTrue(
            result["horizons"]["abstain_reason"].str.contains("WAIT_FINAL_EOD").all()
        )

    def test_ihsg_outcome_resolves_at_exact_horizon(self):
        index = pd.bdate_range("2026-06-01", periods=35)
        close = np.linspace(7000.0, 7350.0, len(index))
        open_ = np.r_[close[0], close[:-1]]
        benchmark = pd.DataFrame(
            {
                "Open": open_,
                "High": np.maximum(open_, close) * 1.003,
                "Low": np.minimum(open_, close) * 0.997,
                "Close": close,
                "Volume": 10_000_000,
            },
            index=index,
        )
        signal_position = 20
        forecast = {
            "version": IHSG_DIRECTION_VERSION,
            "generated_at": pd.Timestamp(index[signal_position], tz="Asia/Jakarta").tz_convert("UTC").isoformat(),
            "as_of": index[signal_position].date().isoformat(),
            "benchmark_close": close[signal_position],
            "regime": "BULL_CONFIRMED",
            "risk_budget_multiplier": 1.0,
            "feature_hash": "abc",
            "horizons": pd.DataFrame(
                [{
                    "horizon": "5D", "horizon_bars": 5,
                    "prediction_state": "UP", "raw_direction": "UP",
                    "prob_up_pct": 60.0, "prob_sideways_pct": 20.0,
                    "prob_down_pct": 20.0, "confidence_pct": 70.0,
                    "neutral_band_pct": 0.2, "validation_state": "OOS_POSITIVE",
                    "brier_skill_pct": 5.0,
                }]
            ),
        }
        outcomes = update_ihsg_outcomes(pd.DataFrame(), forecast, benchmark)
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes.iloc[0]["outcome_status"], "RESOLVED")
        self.assertTrue(bool(outcomes.iloc[0]["hit"]))
        self.assertGreater(float(outcomes.iloc[0]["forward_return_5d"]), 0.0)

    def test_ihsg_outcome_update_is_idempotent(self):
        benchmark = make_market(bars=300)
        forecast = analyze_ihsg_direction(
            benchmark,
            now="2026-07-28 08:00+07:00",
            config=self.small_config(),
        )
        first = update_ihsg_outcomes(pd.DataFrame(), forecast, benchmark)
        second = update_ihsg_outcomes(first, forecast, benchmark)
        self.assertEqual(len(first), len(second))
        self.assertEqual(second["outcome_id"].nunique(), len(second))

    def test_snapshot_frame_contains_risk_only_overlay(self):
        benchmark = make_market()
        forecast = analyze_ihsg_direction(
            benchmark,
            now="2026-07-28 08:00+07:00",
            config=self.small_config(),
        )
        frame = ihsg_snapshot_frame(forecast)
        self.assertEqual(len(frame), 3)
        self.assertTrue(frame["risk_budget_multiplier"].le(1.0).all())
        self.assertTrue(frame["ticker"].eq("^JKSE").all())


class IHSGDatabaseAndDashboardRegressionTests(unittest.TestCase):
    def test_database_hard_expired_payload_is_not_loaded(self):
        settings = scanner_database.DatabaseSettings(
            enabled=True,
            mode="SUPABASE_REST",
            supabase_url="https://example.supabase.co",
            supabase_key="secret",
            supabase_key_type="SECRET",
            read_enabled=True,
            stale_max_age_days=180,
        )
        bridge = scanner_database.ScannerDatabaseBridge(settings)
        row = {
            "ticker": "TEST.JK",
            "payload": {"ticker": "TEST.JK", "fundamental_score": 99.0},
            "source_checked_at": "2020-01-01T00:00:00+00:00",
            "parser_version": "1.1.0",
            "model_version": "7.1.0",
            "refresh_state": "CURRENT",
        }
        with patch.object(bridge, "_get_rows", return_value=[row]):
            data, audit = bridge.read_fundamental_cache(["TEST.JK"])
        self.assertTrue(data.empty)
        self.assertEqual(audit.iloc[0]["database_read_state"], "DATABASE_EXPIRED")
        self.assertEqual(int(audit.iloc[0]["rows"]), 0)

    def test_semantic_hash_ignores_audit_timestamps(self):
        first = {
            "ticker": "TEST.JK",
            "revenue": 100.0,
            "fundamental_fetched_at": "2026-07-27T01:00:00+00:00",
            "database_source_checked_at": "2026-07-27T01:00:00+00:00",
        }
        second = {
            "ticker": "TEST.JK",
            "revenue": 100.0,
            "fundamental_fetched_at": "2026-07-28T02:00:00+00:00",
            "database_source_checked_at": "2026-07-28T02:00:00+00:00",
        }
        self.assertEqual(
            scanner_database._semantic_hash(first),
            scanner_database._semantic_hash(second),
        )
        second["revenue"] = 101.0
        self.assertNotEqual(
            scanner_database._semantic_hash(first),
            scanner_database._semantic_hash(second),
        )

    def test_database_payload_includes_ihsg_snapshots(self):
        forecast = {
            "version": IHSG_DIRECTION_VERSION,
            "as_of": "2026-07-27",
            "data_state": "READY",
            "eod_final": True,
            "benchmark_close": 7500.0,
            "regime": "BULL_CONFIRMED",
            "regime_score": 75.0,
            "consensus_direction": "UP",
            "consensus_confidence": 70.0,
            "risk_budget_multiplier": 1.0,
            "risk_action": "NORMAL_RISK_CAP",
            "feature_coverage_pct": 90.0,
            "breadth_member_count": 200,
            "breadth_ema50_pct": 60.0,
            "feature_hash": "hash",
            "horizons": pd.DataFrame(
                [{
                    "horizon": "5D", "horizon_bars": 5, "raw_direction": "UP",
                    "prediction_state": "UP", "prob_up_pct": 55.0,
                    "prob_sideways_pct": 25.0, "prob_down_pct": 20.0,
                    "confidence_pct": 70.0, "actionable": True,
                }]
            ),
        }
        bridge = scanner_database.ScannerDatabaseBridge(scanner_database.DatabaseSettings())
        payloads = bridge.build_payloads(
            {
                "scanner_version": "7.2.0",
                "ihsg_direction": forecast,
                "fundamentals": pd.DataFrame(),
                "focus_screens": {
                    "multibagger": pd.DataFrame(),
                    "core_swing": pd.DataFrame(),
                },
                "project_management_review": pd.DataFrame(),
            }
        )
        self.assertIn("ihsg_direction_snapshots", payloads)
        self.assertEqual(len(payloads["ihsg_direction_snapshots"]), 1)
        self.assertEqual(
            payloads["ihsg_direction_snapshots"][0]["prediction_state"], "UP"
        )

    def test_duplicate_requested_columns_are_pyarrow_safe(self):
        frame = pd.DataFrame({"ticker": ["TEST.JK"], "score": [88.0]})
        frame.attrs["non_json"] = pd.DataFrame({"x": [1]})
        display = dashboard_v660._safe_display_columns(
            frame, ["ticker", "score", "score", "ticker"]
        )
        self.assertEqual(list(display.columns), ["ticker", "score"])
        self.assertTrue(display.columns.is_unique)
        self.assertEqual(display.attrs, {})

    def test_v5_migration_declares_ihsg_table_and_one_day_outcome(self):
        migration = (
            Path(__file__).resolve().parent
            / "database"
            / "migration_v5_ihsg_direction.sql"
        ).read_text(encoding="utf-8").lower()
        self.assertIn(
            "create table if not exists public.ihsg_direction_snapshots",
            migration,
        )
        self.assertIn("add column if not exists forward_return_1d", migration)
        self.assertIn("enable row level security", migration)


if __name__ == "__main__":
    unittest.main()
