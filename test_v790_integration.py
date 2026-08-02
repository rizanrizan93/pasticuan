from datetime import timedelta
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import scanner
import scanner_focus
from idx_trading_calendar import sanitize_idx_histories
from narrative_engine import build_narrative_profiles
from selector_engine import build_cross_sectional_selector, SelectorConfig


def make_ohlcv(start="2025-01-02", periods=340, drift=0.001):
    idx = pd.bdate_range(start, periods=periods)
    close = 1000.0 * np.cumprod(np.full(periods, 1.0 + drift))
    open_ = np.r_[close[0] * 0.998, close[:-1]]
    high = np.maximum(open_, close) * 1.008
    low = np.minimum(open_, close) * 0.992
    volume = np.full(periods, 5_000_000.0)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)


class CalendarHardeningTests(unittest.TestCase):
    def test_official_holidays_removed_but_single_ticker_zero_volume_kept(self):
        dates = pd.to_datetime(["2026-05-13", "2026-05-14", "2026-05-15", "2026-05-18", "2026-05-19"])
        histories = {}
        for i in range(6):
            volume = [1_000_000, 0, 0, 1_000_000, 1_000_000]
            if i == 0:
                volume[-1] = 0  # legitimate single-name zero-volume day
            histories[f"T{i}.JK"] = pd.DataFrame({
                "Open": [100, 100, 100, 101, 101], "High": [101, 100, 100, 102, 102],
                "Low": [99, 100, 100, 100, 100], "Close": [100, 100, 100, 101, 101],
                "Volume": volume,
            }, index=dates)
        clean, audit = sanitize_idx_histories(histories)
        for frame in clean.values():
            self.assertNotIn(pd.Timestamp("2026-05-14"), frame.index)
            self.assertNotIn(pd.Timestamp("2026-05-15"), frame.index)
        self.assertIn(pd.Timestamp("2026-05-19"), clean["T0.JK"].index)
        self.assertGreaterEqual(len(audit), 12)

    def test_exact_current_session_rejects_one_session_stale_cache(self):
        now = pd.Timestamp("2026-08-01 07:56", tz="Asia/Jakarta")
        self.assertEqual(scanner._expected_last_completed_daily_date(now), pd.Timestamp("2026-07-31"))
        stale = make_ohlcv(start="2026-07-01", periods=22)
        stale = stale.loc[stale.index <= pd.Timestamp("2026-07-30")]
        stale.attrs.update({"bar_state": "FINAL_EOD", "finalized_for_date": "2026-07-30"})
        self.assertFalse(scanner._cache_meta_proves_final(stale, now))
        current = pd.concat([stale, pd.DataFrame({
            "Open": [1100], "High": [1110], "Low": [1090], "Close": [1105], "Volume": [2_000_000]
        }, index=[pd.Timestamp("2026-07-31")])])
        current.attrs.update({"bar_state": "FINAL_EOD", "finalized_for_date": "2026-07-31"})
        self.assertTrue(scanner._cache_meta_proves_final(current, now))


class SelectorConsistencyTests(unittest.TestCase):
    def test_five_stock_universe_uses_absolute_score_only(self):
        prepared = {}
        for i, drift in enumerate([0.0002, 0.0005, 0.0008, 0.0010, 0.0012]):
            prepared[f"S{i}.JK"] = scanner.prepare_indicators(make_ohlcv(drift=drift))
        current, _, _ = build_cross_sectional_selector(
            prepared,
            SelectorConfig(min_training_rows=10_000, min_evaluation_rows=10_000),
        )
        self.assertFalse(current.empty)
        self.assertTrue(current["relative_overlay_weight_pct"].eq(0.0).all())
        self.assertTrue(current["selector_universe_state"].eq("ABSOLUTE_ONLY_SMALL_UNIVERSE").all())
        np.testing.assert_allclose(
            current["swing_selection_score"].to_numpy(float),
            current["absolute_swing_score"].to_numpy(float),
            atol=0.11,
        )


class FundamentalCompletenessTests(unittest.TestCase):
    def _history(self):
        rows = []
        periods = pd.date_range("2024-09-30", periods=8, freq="QE")
        for i, period in enumerate(periods):
            base = 1000.0 + 80.0 * i
            common = {
                "ticker": "TEST.JK", "period_end": period, "period_type": "Q",
                "statement_basis": "STANDALONE_QUARTER", "source_family": "IDX_OFFICIAL_XBRL",
                "source_name": "IDX", "source_url": "https://www.idx.co.id/test.xml",
                "currency": "IDR", "source_verified": True, "validation_flags": "",
                "revenue": base, "gross_profit": base * 0.35, "operating_income": base * 0.18,
                "ebit": base * 0.17, "ebitda": base * 0.21, "net_income": base * 0.12,
                "operating_cash_flow": base * 0.14, "capex": -base * 0.04,
                "interest_expense": base * 0.01, "total_assets": 6000 + 200 * i,
                "equity": 3500 + 150 * i, "total_debt": 900 + 20 * i,
                "cash": 700 + 30 * i, "shares_outstanding": 1_000_000,
            }
            rows.append(common)
        # Complementary duplicate: one row carries a missing fact, another fills it.
        rows[-1]["operating_cash_flow"] = np.nan
        duplicate = rows[-1].copy()
        duplicate["revenue"] = np.nan
        duplicate["operating_cash_flow"] = 300.0
        rows.append(duplicate)
        return pd.DataFrame(rows)

    def test_duplicate_provider_rows_are_fieldwise_coalesced_and_complete(self):
        normalized = scanner.normalize_fundamental_history(self._history())
        latest = normalized.sort_values("period_end").iloc[-1]
        self.assertTrue(np.isfinite(latest["revenue"]))
        self.assertTrue(np.isfinite(latest["operating_cash_flow"]))
        features = scanner.build_fundamental_history_features(normalized, now=pd.Timestamp("2026-08-01", tz="Asia/Jakarta"))
        row = features.iloc[0]
        self.assertTrue(bool(row["fundamental_complete_for_multibagger"]))
        self.assertGreaterEqual(float(row["fundamental_statement_family_coverage_pct"]), 65.0)
        self.assertGreaterEqual(float(row["fundamental_history_period_coverage_pct"]), 75.0)
        self.assertEqual(str(row["fundamental_missing_core_fields"]), "")

    def test_structured_fundamental_evidence_populates_narrative_and_alignment(self):
        history = scanner.normalize_fundamental_history(self._history())
        features = scanner.build_fundamental_history_features(history, now=pd.Timestamp("2026-08-01", tz="Asia/Jakarta"))
        features["revenue_growth"] = features["history_revenue_growth"]
        features["earnings_growth"] = features["history_earnings_growth"]
        features["net_margin"] = features["history_net_margin"]
        prepared = {"TEST.JK": scanner.prepare_indicators(make_ohlcv())}
        profiles = build_narrative_profiles(
            ["TEST.JK"], prepared=prepared, events=pd.DataFrame(), outcomes=pd.DataFrame(),
            fundamentals=features, news_review=pd.DataFrame(), project_management=pd.DataFrame(),
            silent_profiles={}, as_of=pd.Timestamp("2026-08-01", tz="Asia/Jakarta"),
        )
        row = profiles.iloc[0]
        self.assertTrue(np.isfinite(row["narrative_score"]))
        self.assertTrue(np.isfinite(row["issuer_alignment_score"]))
        self.assertIn(row["evidence_acquisition_status"], {"STRUCTURED_FINANCIAL_ACQUIRED", "SOURCE_EVENT_ACQUIRED"})


class ReclaimTriggerTests(unittest.TestCase):
    def test_breakout_retest_exposes_non_chasing_reclaim_trigger(self):
        base = np.linspace(900, 1180, 270)
        tail = np.array([1175, 1180, 1185, 1190, 1195, 1260, 1245, 1225, 1235, 1250])
        close = np.r_[base, tail]
        volume = np.full(len(close), 3_000_000.0)
        volume[-5] = 8_000_000.0
        raw = pd.DataFrame(index=pd.bdate_range("2025-01-01", periods=len(close)))
        raw["Close"] = close
        raw["Open"] = np.r_[close[0] * 0.997, close[:-1]]
        raw["High"] = np.maximum(raw["Open"], raw["Close"]) * 1.006
        raw["Low"] = np.minimum(raw["Open"], raw["Close"]) * 0.994
        raw["Volume"] = volume
        raw.loc[raw.index[-5], ["Open", "High"]] = [1195, 1270]
        raw.loc[raw.index[-3], "Low"] = 1195
        frame = scanner.prepare_indicators(raw)
        plan = scanner.detect_breakout_retest(frame, "TEST.JK")
        self.assertTrue(plan.detected, plan.reason)
        self.assertIsNotNone(plan.reclaim_trigger_price)
        self.assertEqual(plan.trigger, plan.reclaim_trigger_price)
        self.assertIn("RESISTANCE_PLUS_ONE_IDX_TICK", plan.trigger_basis)
        self.assertTrue(plan.trigger_instruction)
        # Trigger should not chase the latest high.
        self.assertLessEqual(float(plan.trigger), float(frame.iloc[-1]["High"]))


class AdaptiveOOSTests(unittest.TestCase):
    def test_adaptive_validation_expands_until_genuine_event_target(self):
        prepared = {f"T{i:03d}.JK": make_ohlcv(periods=230) for i in range(130)}
        calls = []

        def fake_select(mapping, max_tickers):
            names = sorted(mapping)[:max_tickers]
            audit = pd.DataFrame({"ticker": sorted(mapping), "selected": [name in names for name in sorted(mapping)]})
            return {name: mapping[name] for name in names}, audit

        def fake_validate(delta, config):
            calls.append(len(delta))
            iteration = len(calls)
            rows = []
            for setup in scanner.SETUPS:
                rows.append({
                    "ticker": f"X{iteration}.JK", "setup": setup,
                    "signal_date": pd.Timestamp("2026-01-01") + timedelta(days=iteration),
                    "filled": True, "r_multiple": 1.0,
                })
            return pd.DataFrame(), pd.DataFrame(rows)

        def all_oos(events, config):
            out = events.copy()
            out["is_oos"] = True
            out["oos_eligible"] = True
            out["oos_fold"] = 1
            return out

        cfg = scanner.ScanConfig().replace(
            validation_max_tickers=130, min_oos_signal_events=2,
            min_oos_filled_events=2, min_oos_unique_dates=1,
        )
        with (
            patch.object(scanner, "select_walkforward_universe", side_effect=fake_select),
            patch.object(scanner, "run_walkforward_validation", side_effect=fake_validate),
            patch.object(scanner, "_assign_oos_folds", side_effect=all_oos),
            patch.object(scanner, "aggregate_backtest", return_value=pd.DataFrame({"setup": list(scanner.SETUPS)})),
        ):
            _, events, audit = scanner.run_adaptive_walkforward_validation(
                prepared, cfg, initial_tickers=60,
            )
        self.assertEqual(calls, [60, 60])
        self.assertEqual(len(events), len(scanner.SETUPS) * 2)
        expansion = audit.attrs.get("expansion_audit")
        self.assertIsInstance(expansion, pd.DataFrame)
        self.assertEqual(expansion["target_tickers"].tolist(), [60, 120])
        self.assertTrue(bool(expansion.iloc[-1]["evidence_target_met"]))


class MultibaggerPipelineTests(unittest.TestCase):
    def test_completeness_metadata_reaches_multibagger_output(self):
        history = scanner.normalize_fundamental_history(FundamentalCompletenessTests()._history())
        fundamentals = scanner.build_fundamental_history_features(
            history, now=pd.Timestamp("2026-08-01", tz="Asia/Jakarta"),
        )
        aliases = {
            "revenue_growth": "history_revenue_growth",
            "earnings_growth": "history_earnings_growth",
            "roe": "history_roe", "roa": "history_roa",
            "operating_margin": "history_operating_margin",
            "net_margin": "history_net_margin",
            "debt_equity": "history_debt_equity",
            "operating_cash_flow": "history_ocf_ttm",
            "free_cash_flow": "history_fcf_ttm",
        }
        for target, source in aliases.items():
            fundamentals[target] = fundamentals[source]
        fundamentals["fundamental_coverage"] = 100.0
        fundamentals["fundamental_score"] = 80.0
        fundamentals["fundamental_score_10"] = 8.0
        fundamentals["fundamental_reliability"] = "HIGH"
        prepared = {"TEST.JK": scanner.prepare_indicators(make_ohlcv())}
        output = scanner_focus.scan_multibagger_candidates(
            prepared, fundamentals, config=scanner.ScanConfig(),
            silent_profiles=scanner_focus.current_silent_profiles(prepared),
        )
        self.assertEqual(output.loc[0, "fundamental_acquisition_state"], "COMPLETE_FOR_MULTIBAGGER")
        self.assertTrue(bool(output.loc[0, "fundamental_complete_for_multibagger"]))
        self.assertEqual(float(output.loc[0, "fundamental_statement_family_coverage_pct"]), 100.0)
        self.assertEqual(float(output.loc[0, "fundamental_history_period_coverage_pct"]), 100.0)



if __name__ == "__main__":
    unittest.main()
