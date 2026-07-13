from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scanner.backtest import aggregate_backtest, historical_signal_mask  # noqa: E402
from scanner.config import ScanConfig  # noqa: E402
from scanner.data import _extract_batch, normalize_idx_ticker, parse_ticker_csv  # noqa: E402
from scanner.engine import ScanEngine  # noqa: E402
from scanner.fundamentals import score_fundamentals  # noqa: E402
from scanner.indicators import prepare_indicators  # noqa: E402
from scanner.models import MarketContext, SetupPlan  # noqa: E402
from scanner.price_rules import idx_tick_size, round_idx_price  # noqa: E402
from scanner.setups import (  # noqa: E402
    detect_breakout_retest,
    detect_pullback_continuation,
    detect_reversal_accumulation,
    detect_unicorn_sniper,
)


def make_ohlcv(close: np.ndarray, volume: np.ndarray | None = None) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    if volume is None:
        volume = np.full(len(close), 3_000_000.0)
    open_ = np.r_[close[0] * 0.997, close[:-1]]
    high = np.maximum(open_, close) * 1.006
    low = np.minimum(open_, close) * 0.994
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=pd.bdate_range("2024-01-01", periods=len(close)),
    )


class PriceRuleTests(unittest.TestCase):
    def test_idx_tick_ladder_and_boundaries(self):
        self.assertEqual(idx_tick_size(199), 1)
        self.assertEqual(idx_tick_size(200), 2)
        self.assertEqual(idx_tick_size(500), 5)
        self.assertEqual(idx_tick_size(2_000), 10)
        self.assertEqual(idx_tick_size(5_000), 25)
        self.assertEqual(round_idx_price(199.2, "up"), 200)
        self.assertEqual(round_idx_price(499.9, "up"), 500)
        self.assertEqual(round_idx_price(5_001, "up"), 5_025)
        self.assertEqual(round_idx_price(5_001, "down"), 5_000)


class CSVTests(unittest.TestCase):
    def test_ticker_normalization_and_deduplication(self):
        frame = pd.DataFrame({"Kode": ["admr", "ADMR.JK", " IDX:ANTM ", "bad/name", None]})
        self.assertEqual(parse_ticker_csv(frame), ["ADMR.JK", "ANTM.JK"])
        self.assertEqual(normalize_idx_ticker("bbri"), "BBRI.JK")

    def test_yfinance_multiindex_both_orientations(self):
        index = pd.bdate_range("2025-01-01", periods=3)
        values = np.arange(15, dtype=float).reshape(3, 5) + 100
        fields = ["Open", "High", "Low", "Close", "Volume"]
        ticker_first = pd.DataFrame(
            values,
            index=index,
            columns=pd.MultiIndex.from_product([["ADMR.JK"], fields]),
        )
        field_first = pd.DataFrame(
            values,
            index=index,
            columns=pd.MultiIndex.from_product([fields, ["ADMR.JK"]]),
        )
        self.assertEqual(len(_extract_batch(ticker_first, "ADMR.JK", 2)), 3)
        self.assertEqual(len(_extract_batch(field_first, "ADMR.JK", 2)), 3)


class IndicatorTests(unittest.TestCase):
    def test_confirmed_pivot_is_not_emitted_at_raw_peak(self):
        close = np.linspace(100, 130, 240)
        frame = make_ohlcv(close)
        frame.loc[frame.index[220], "High"] = 200
        prepared = prepare_indicators(frame)
        self.assertTrue(pd.isna(prepared["PIVOT_HIGH"].iloc[220]))
        self.assertAlmostEqual(prepared["PIVOT_HIGH"].iloc[223], 200)


class SetupTests(unittest.TestCase):
    def test_pullback_continuation_detects_controlled_retest(self):
        first = np.linspace(900, 1_900, 290)
        tail = np.array([1_885, 1_875, 1_870, 1_875, 1_885, 1_895, 1_905, 1_915, 1_930, 1_950])
        volume = np.full(300, 3_000_000.0)
        volume[-10:-1] = 2_100_000.0
        volume[-1] = 3_200_000.0
        frame = prepare_indicators(make_ohlcv(np.r_[first, tail], volume))
        plan = detect_pullback_continuation(frame, "TEST.JK")
        self.assertTrue(plan.detected, plan.reason)
        self.assertGreater(plan.entry, plan.stop_loss)
        self.assertGreaterEqual(plan.rr2, 2.0)

    def test_breakout_retest_detects_break_and_hold(self):
        base = np.linspace(900, 1_180, 270)
        tail = np.array([1_175, 1_180, 1_185, 1_190, 1_195, 1_260, 1_245, 1_225, 1_235, 1_250])
        close = np.r_[base, tail]
        volume = np.full(len(close), 3_000_000.0)
        volume[-5] = 8_000_000.0
        raw = make_ohlcv(close, volume)
        raw.loc[raw.index[-5], "Open"] = 1_195
        raw.loc[raw.index[-5], "High"] = 1_270
        raw.loc[raw.index[-3], "Low"] = 1_195
        frame = prepare_indicators(raw)
        plan = detect_breakout_retest(frame, "TEST.JK")
        self.assertTrue(plan.detected, plan.reason)
        self.assertIn(plan.action, {"READY_TRIGGER", "WAIT_RETEST"})

    def test_reversal_requires_sweep_and_accumulation_proxy(self):
        pre = np.linspace(2_000, 1_000, 270)
        base = np.array(
            [
                1_010, 1_005, 1_000, 995, 1_000, 1_005, 1_000, 995, 1_000, 1_005,
                1_010, 1_005, 1_000, 995, 1_000, 1_005, 1_010, 1_005, 1_000, 995,
                1_000, 1_005, 1_010, 1_020, 1_030, 1_040, 1_050, 1_060, 1_070, 1_090,
            ]
        )
        volume = np.full(300, 3_000_000.0)
        volume[-10:] = np.linspace(4_000_000, 8_000_000, 10)
        raw = make_ohlcv(np.r_[pre, base], volume)
        raw.loc[raw.index[-10], ["Open", "High", "Low", "Close", "Volume"]] = [1_005, 1_010, 930, 1_000, 8_000_000]
        plan = detect_reversal_accumulation(prepare_indicators(raw), "REV.JK")
        self.assertTrue(plan.detected, plan.reason)
        self.assertIn("Sell-side liquidity sweep", plan.evidence)
        self.assertGreater(plan.entry, plan.stop_loss)

    def test_unicorn_requires_sweep_bos_and_fvg(self):
        raw = make_ohlcv(np.linspace(900, 1_020, 300))
        wave = [
            1_000, 1_005, 1_010, 1_005, 1_000, 995, 1_000, 1_005, 1_010, 1_005,
            1_000, 995, 1_000, 1_005, 1_010, 1_005, 1_000, 995, 1_000, 1_005,
            1_000, 1_000, 1_005, 1_000, 1_005, 1_100, 1_120, 1_090, 1_070, 1_060,
        ]
        for offset, value in enumerate(wave):
            raw.iloc[-30 + offset] = [value - 3, value + 8, value - 8, value, 3_000_000]
        pos = len(raw) - 12
        raw.iloc[pos] = [1_000, 1_010, 930, 1_000, 6_000_000]
        raw.iloc[pos + 1] = [1_010, 1_020, 990, 1_000, 3_000_000]
        raw.iloc[pos + 2] = [1_000, 1_110, 995, 1_100, 9_000_000]
        raw.iloc[pos + 3] = [1_095, 1_130, 1_080, 1_120, 5_000_000]
        for j, value in enumerate([1_110, 1_095, 1_085, 1_075, 1_065, 1_055, 1_060, 1_070], start=pos + 4):
            raw.iloc[j] = [value - 5, value + 10, value - 12, value, 4_000_000]
        plan = detect_unicorn_sniper(prepare_indicators(raw), "ICT.JK")
        self.assertTrue(plan.detected, plan.reason)
        self.assertIn("Bullish FVG valid", plan.evidence)
        self.assertGreater(plan.entry, plan.stop_loss)

    def test_stale_zone_is_never_execution_ready(self):
        frame = prepare_indicators(make_ohlcv(np.linspace(1_000, 1_500, 260)))
        plan = SetupPlan(
            ticker="TEST.JK",
            setup="PULLBACK_CONTINUATION",
            detected=True,
            setup_score=90,
            entry_low=1_450,
            entry_high=1_510,
            entry=1_505,
            stop_loss=1_450,
            tp1=1_590,
            tp2=1_645,
            rr1=1.55,
            rr2=2.55,
            distance_atr=0.1,
            zone_age_bars=31,
            action="READY_TRIGGER",
        )
        metrics = {
            "last_price": 1_500,
            "last_date": "2025-01-01",
            "adtv20_idr": 5e9,
            "atr_pct": 0.03,
            "zero_volume_ratio20": 0,
            "volume_ratio": 1,
            "rsi14": 55,
            "adx14": 25,
            "cmf20": 0.1,
            "roc60": 0.1,
            "distance_52w_high": -0.05,
            "relative_strength60": 0.05,
            "data_lag_days": 0,
        }
        row = ScanEngine()._finalize(plan, frame, MarketContext(regime="RISK_ON"), [], metrics)
        self.assertEqual(row["status"], "WATCHLIST_ENTRY")
        self.assertIn("kedaluwarsa", row["blockers"])

    def test_risk_off_regime_blocks_ready_action(self):
        frame = prepare_indicators(make_ohlcv(np.linspace(1_000, 1_500, 260)))
        plan = SetupPlan(
            ticker="TEST.JK", setup="X", detected=True, setup_score=95,
            entry_low=1_490, entry_high=1_510, entry=1_505, stop_loss=1_450,
            tp1=1_590, tp2=1_645, rr1=1.55, rr2=2.55, distance_atr=0.1,
            zone_age_bars=1, action="READY_TRIGGER",
        )
        metrics = {
            "last_price": 1_500, "last_date": "2025-01-01", "adtv20_idr": 5e9,
            "atr_pct": 0.03, "zero_volume_ratio20": 0, "volume_ratio": 1,
            "rsi14": 55, "adx14": 25, "cmf20": 0.1, "roc60": 0.1,
            "distance_52w_high": -0.05, "relative_strength60": 0.05, "data_lag_days": 0,
        }
        row = ScanEngine()._finalize(plan, frame, MarketContext(regime="RISK_OFF"), [], metrics)
        self.assertEqual(row["status"], "WATCHLIST_ENTRY")
        self.assertIn("RISK_OFF", row["blockers"])


class BacktestTests(unittest.TestCase):
    def test_backtest_mask_has_no_signals_before_long_lookback(self):
        frame = prepare_indicators(make_ohlcv(np.linspace(900, 1_500, 260)))
        mask = historical_signal_mask(frame, "PULLBACK_CONTINUATION")
        self.assertFalse(mask.iloc[:200].any())

    def test_bayesian_probability_is_shrunk(self):
        trades = pd.DataFrame(
            {"setup": ["X"] * 4, "r_multiple": [2.0, 2.0, -1.0, -1.0]}
        )
        stats = aggregate_backtest(trades, ScanConfig())
        self.assertEqual(stats.loc[0, "historical_hit_rate"], 50.0)
        self.assertEqual(stats.loc[0, "bayes_probability"], 50.0)
        self.assertEqual(stats.loc[0, "sample_reliability"], "LOW")


class FundamentalTests(unittest.TestCase):
    def test_quality_company_scores_above_weak_company(self):
        strong = score_fundamentals(
            {
                "revenueGrowth": 0.20,
                "earningsGrowth": 0.25,
                "returnOnEquity": 0.22,
                "returnOnAssets": 0.10,
                "grossMargins": 0.45,
                "operatingMargins": 0.20,
                "profitMargins": 0.15,
                "debtToEquity": 30,
                "currentRatio": 2.0,
                "totalCash": 120,
                "totalDebt": 100,
                "operatingCashflow": 20,
                "freeCashflow": 15,
                "marketCap": 200,
                "pegRatio": 1.2,
            }
        )
        weak = score_fundamentals(
            {
                "revenueGrowth": -0.10,
                "earningsGrowth": -0.20,
                "returnOnEquity": 0.01,
                "returnOnAssets": -0.02,
                "profitMargins": -0.05,
                "debtToEquity": 300,
                "operatingCashflow": -20,
                "freeCashflow": -30,
                "marketCap": 200,
            }
        )
        self.assertGreater(strong["fundamental_score"], weak["fundamental_score"])
        self.assertIn("OCF negatif", weak["fundamental_red_flags"])

    def test_bank_does_not_use_general_der_red_flag(self):
        bank = score_fundamentals(
            {
                "sector": "Financial Services",
                "industry": "Banks - Regional",
                "revenueGrowth": 0.12,
                "earningsGrowth": 0.15,
                "returnOnEquity": 0.18,
                "returnOnAssets": 0.025,
                "profitMargins": 0.25,
                "debtToEquity": 500,
                "operatingCashflow": -100,
            }
        )
        self.assertEqual(bank["fundamental_model"], "FINANCIAL")
        self.assertNotIn("DER tinggi", bank["fundamental_red_flags"])
        self.assertNotIn("OCF negatif", bank["fundamental_red_flags"])


if __name__ == "__main__":
    unittest.main()
