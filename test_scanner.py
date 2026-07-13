from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scanner as scanner_module

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scanner import (  # noqa: E402
    MarketContext,
    ScanConfig,
    ScanEngine,
    SetupPlan,
    _assign_oos_folds,
    _extract_batch,
    _simulate_order,
    _historical_gate_inputs,
    aggregate_backtest,
    apply_fundamental_gate,
    apply_market_status_gate,
    apply_news_gate,
    apply_validation_gate,
    apply_execution_snapshot_gate,
    apply_universe_integrity_gate,
    attach_broker_summary,
    attach_position_sizing,
    enforce_portfolio_execution_budget,
    finalize_execution_integrity,
    parse_portfolio_csv,
    analyze_portfolio_positions,
    detect_breakout_retest,
    detect_pullback_continuation,
    detect_reversal_accumulation,
    detect_unicorn_sniper,
    historical_signal_mask,
    idx_daily_price_band,
    idx_tick_size,
    is_valid_idx_price,
    normalize_idx_ticker,
    ohlcv_quality_issues,
    parse_broker_summary_csv,
    parse_market_status_csv,
    parse_news_review_csv,
    parse_ticker_csv,
    prepare_indicators,
    round_idx_price,
    score_fundamentals,
    size_stockbit_order,
    within_idx_daily_price_band,
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

    def test_default_universe_limit_covers_full_idx_scale(self):
        frame = pd.DataFrame({"ticker": [f"A{i:04d}" for i in range(1_300)]})
        parsed = parse_ticker_csv(frame)
        self.assertEqual(len(parsed), 1_200)
        self.assertEqual(parsed[-1], "A1199.JK")

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


class DeploymentManifestTests(unittest.TestCase):
    def test_every_app_imported_scanner_module_is_packaged(self):
        self.assertTrue((ROOT / "scanner.py").is_file())
        self.assertTrue((ROOT / "test_scanner.py").is_file())
        self.assertFalse((ROOT / "scanner").exists())
        self.assertFalse((ROOT / "tests").exists())


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


def signal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["ADRO.JK"],
            "status": ["EXECUTION_READY"],
            "status_rank": [0],
            "blockers": [""],
            "blocker_count": [0],
            "entry": [1_000.0],
            "stop_loss": [950.0],
        }
    )


def order_bars(highs: list[float], lows: list[float], opens: list[float] | None = None) -> pd.DataFrame:
    if opens is None:
        opens = [100.0] * len(highs)
    close = [(high + low) / 2 for high, low in zip(highs, lows)]
    return pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": close,
            "ATR14": [10.0] * len(highs),
        },
        index=pd.bdate_range("2025-01-01", periods=len(highs)),
    )


def plan() -> SimpleNamespace:
    return SimpleNamespace(
        entry=100.0,
        stop_loss=95.0,
        tp1=109.0,
        tp2=115.0,
        entry_type="BUY_STOP_CONFIRMATION",
    )


class PriceBandTests(unittest.TestCase):
    def test_tick_and_daily_band_are_enforced(self):
        self.assertTrue(is_valid_idx_price(1_000))
        self.assertFalse(is_valid_idx_price(1_001))
        lower, upper = idx_daily_price_band(1_000)
        self.assertEqual((lower, upper), (850, 1_250))
        self.assertTrue(within_idx_daily_price_band(1_250, 1_000))
        self.assertFalse(within_idx_daily_price_band(1_255, 1_000))


class PositionSizingTests(unittest.TestCase):
    def test_rp10m_order_respects_risk_and_position_caps(self):
        cfg = ScanConfig()
        sized = size_stockbit_order(1_000, 950, cfg)
        self.assertEqual(sized["sizing_status"], "OK")
        self.assertLessEqual(sized["max_loss_idr"], 100_000)
        self.assertLessEqual(sized["capital_required_idr"], 4_000_000)
        self.assertGreaterEqual(sized["suggested_lots"], 1)

    def test_invalid_tick_cannot_be_sized(self):
        sized = size_stockbit_order(1_001, 950, ScanConfig())
        self.assertEqual(sized["suggested_lots"], 0)
        self.assertEqual(sized["sizing_status"], "INVALID_TICK")

    def test_account_too_small_downgrades_execution(self):
        cfg = ScanConfig().replace(account_size_idr=100_000, max_position_pct=0.10)
        out = attach_position_sizing(signal_frame(), cfg)
        self.assertEqual(out.loc[0, "status"], "WATCHLIST_ENTRY")
        self.assertIn("kurang dari 1 lot", out.loc[0, "blockers"])


class FailClosedContextTests(unittest.TestCase):
    def test_missing_fundamental_reduces_confidence_without_erasing_setup(self):
        out = apply_fundamental_gate(signal_frame(), ScanConfig())
        self.assertEqual(out.loc[0, "status"], "EXECUTION_READY")
        self.assertEqual(out.loc[0, "fundamental_tier"], "MISSING_NEUTRAL")
        self.assertEqual(out.loc[0, "fundamental_confidence"], 50.0)

    def test_missing_market_status_uses_fallback_confidence(self):
        out = apply_market_status_gate(signal_frame(), pd.DataFrame(), ScanConfig(), "2026-07-13")
        self.assertEqual(out.loc[0, "status"], "EXECUTION_READY")
        self.assertEqual(out.loc[0, "market_status_coverage"], "FALLBACK_REQUIRED")
        self.assertEqual(out.loc[0, "market_status_confidence"], 45.0)

    def test_verified_clean_status_preserves_execution(self):
        status = parse_market_status_csv(
            pd.DataFrame(
                {
                    "ticker": ["ADRO"],
                    "as_of": [pd.Timestamp.now(tz="Asia/Jakarta").date().isoformat()],
                    "suspended": [False],
                    "special_monitoring": [False],
                    "fca": [False],
                    "special_notation": [""],
                    "corporate_action": [False],
                    "source_url": ["https://www.idx.co.id/id/perusahaan-tercatat/daftar-efek-pemantauan-khusus"],
                    "coverage_complete": [True],
                    "verification_method": ["OFFICIAL_IDX_AUTOMATED_SCREEN"],
                }
            )
        )
        out = apply_market_status_gate(signal_frame(), status, ScanConfig(), "2026-07-13")
        self.assertEqual(out.loc[0, "status"], "EXECUTION_READY")
        self.assertEqual(out.loc[0, "market_status_coverage"], "AUTO_VERIFIED")

    def test_fca_is_rejected(self):
        status = parse_market_status_csv(
            pd.DataFrame(
                {
                    "ticker": ["ADRO"],
                    "as_of": ["2026-07-13"],
                    "fca": [True],
                    "source_url": ["https://www.idx.co.id/id/perusahaan-tercatat"],
                }
            )
        )
        out = apply_market_status_gate(signal_frame(), status, ScanConfig(), "2026-07-13")
        self.assertEqual(out.loc[0, "status"], "REJECT")

    def test_missing_ticker_in_partial_status_file_is_watchlist_not_false_suspension(self):
        status = parse_market_status_csv(
            pd.DataFrame(
                {
                    "ticker": ["ANTM"],
                    "as_of": ["2026-07-13"],
                    "source_url": ["https://www.idx.co.id/id/perusahaan-tercatat"],
                }
            )
        )
        out = apply_market_status_gate(signal_frame(), status, ScanConfig(), "2026-07-13")
        self.assertEqual(out.loc[0, "status"], "EXECUTION_READY")
        self.assertEqual(out.loc[0, "market_status_coverage"], "FALLBACK_REQUIRED")
        self.assertNotIn("suspensi", out.loc[0, "blockers"])

    def test_complete_news_review_preserves_execution(self):
        news = parse_news_review_csv(
            pd.DataFrame(
                {
                    "ticker": ["ADRO"],
                    "reviewed_at": [pd.Timestamp.now(tz="Asia/Jakarta").date().isoformat()],
                    "review_status": ["COMPLETE"],
                    "provider_query_ok": [True],
                    "items_reviewed": [0],
                    "coverage_start": [(pd.Timestamp.now(tz="Asia/Jakarta") - pd.Timedelta(days=7)).date().isoformat()],
                    "coverage_end": [pd.Timestamp.now(tz="Asia/Jakarta").date().isoformat()],
                    "provider": ["Yahoo Finance + official IDX disclosure page"],
                }
            )
        )
        out = apply_news_gate(signal_frame(), news, ScanConfig(), "2026-07-13")
        self.assertEqual(out.loc[0, "status"], "EXECUTION_READY")

    def test_verified_severe_negative_news_is_rejected(self):
        news = parse_news_review_csv(
            pd.DataFrame(
                {
                    "ticker": ["ADRO"],
                    "reviewed_at": ["2026-07-13"],
                    "review_status": ["COMPLETE"],
                    "title": ["Material adverse event"],
                    "sentiment": ["NEGATIVE"],
                    "materiality": ["HIGH"],
                    "verified": [True],
                    "source_url": ["https://www.idx.co.id/announcement"],
                }
            )
        )
        out = apply_news_gate(signal_frame(), news, ScanConfig(), "2026-07-13")
        self.assertEqual(out.loc[0, "status"], "REJECT")


class BrokerSummaryTests(unittest.TestCase):
    def test_distribution_proxy_downgrades_but_does_not_claim_owner(self):
        raw = pd.DataFrame(
            {
                "ticker": ["ADRO", "ADRO"],
                "date": ["2026-07-10", "2026-07-13"],
                "broker_code": ["AA", "BB"],
                "buy_value": [10, 10],
                "sell_value": [100, 100],
            }
        )
        parsed = parse_broker_summary_csv(raw)
        out = attach_broker_summary(signal_frame(), parsed)
        self.assertEqual(out.loc[0, "broksum_signal"], "DISTRIBUTION_PROXY")
        self.assertEqual(out.loc[0, "status"], "WATCHLIST_ENTRY")
        self.assertIn("bukan identitas", out.loc[0, "broksum_note"])


class DataQualityTests(unittest.TestCase):
    def test_inconsistent_bar_is_reported(self):
        frame = pd.DataFrame(
            {"Open": [100], "High": [99], "Low": [101], "Close": [100], "Volume": [1_000]},
            index=[pd.Timestamp("2026-07-13")],
        )
        self.assertIn("Bar OHLC tidak konsisten", ohlcv_quality_issues(frame))


class ConditionalFillTests(unittest.TestCase):
    def test_order_never_touched_is_not_a_trade(self):
        bars = order_bars([100, 104, 105, 106], [98, 99, 100, 101])
        untouched = plan()
        untouched.entry = 110.0
        untouched.stop_loss = 100.0
        untouched.tp1 = 128.0
        untouched.tp2 = 140.0
        event = _simulate_order(bars, 0, "X.JK", "X", untouched, 90, "RISK_ON", ScanConfig())
        self.assertFalse(event.filled)
        self.assertEqual(event.result, "NO_FILL")

    def test_trigger_then_tp1_records_fill_and_time(self):
        bars = order_bars([100, 101, 108, 110, 111], [98, 99, 99, 101, 103])
        event = _simulate_order(bars, 0, "X.JK", "X", plan(), 90, "RISK_ON", ScanConfig())
        self.assertTrue(event.filled)
        self.assertTrue(event.tp1_hit)
        self.assertEqual(event.result, "WIN_TP1")
        self.assertGreaterEqual(event.time_to_tp1_bars, 1)

    def test_same_daily_bar_uses_stop_first_convention(self):
        bars = order_bars([100, 110, 111], [98, 94, 100])
        event = _simulate_order(bars, 0, "X.JK", "X", plan(), 90, "RISK_ON", ScanConfig())
        self.assertTrue(event.filled)
        self.assertEqual(event.result, "LOSS")

    def test_gap_that_degrades_rr_is_cancelled(self):
        bars = order_bars([100, 105, 106], [98, 99, 100], opens=[100, 101, 102])
        event = _simulate_order(bars, 0, "X.JK", "X", plan(), 90, "RISK_ON", ScanConfig())
        self.assertFalse(event.filled)
        self.assertIn("RR1", event.no_fill_reason)


class WalkForwardTests(unittest.TestCase):
    def test_initial_window_is_never_marked_oos(self):
        dates = pd.bdate_range("2024-01-01", periods=10)
        events = pd.DataFrame(
            {
                "ticker": ["X.JK"] * 10,
                "setup": ["X"] * 10,
                "signal_date": dates,
            }
        )
        out = _assign_oos_folds(events, ScanConfig().replace(walkforward_min_train_fraction=0.60))
        self.assertFalse(out.iloc[:6]["is_oos"].any())
        self.assertTrue(out.iloc[6:]["is_oos"].all())


class AutonomousHardeningTests(unittest.TestCase):
    def test_spoofed_idx_hostname_never_verifies(self):
        raw = pd.DataFrame(
            {
                "ticker": ["ADRO"],
                "as_of": [pd.Timestamp.now(tz="Asia/Jakarta").date().isoformat()],
                "suspended": [False],
                "special_monitoring": [False],
                "fca": [False],
                "special_notation": [""],
                "corporate_action": [False],
                "source_url": ["https://www.idx.co.id.evil.example/status"],
                "coverage_complete": [True],
                "verification_method": ["AUTOMATED"],
            }
        )
        parsed = parse_market_status_csv(raw)
        self.assertFalse(bool(parsed.loc[0, "market_status_verified"]))

    def test_empty_news_row_cannot_default_to_complete(self):
        parsed = parse_news_review_csv(
            pd.DataFrame(
                {
                    "ticker": ["ADRO"],
                    "reviewed_at": [pd.Timestamp.now(tz="Asia/Jakarta").date().isoformat()],
                    "review_status": ["COMPLETE"],
                }
            )
        )
        self.assertEqual(parsed.loc[0, "news_review_status"], "INCOMPLETE")

    def test_insufficient_oos_is_labeled_honestly(self):
        events = pd.DataFrame(
            {
                "ticker": ["X.JK"] * 4,
                "setup": ["X"] * 4,
                "signal_date": pd.bdate_range("2026-01-01", periods=4),
                "filled": [True] * 4,
                "r_multiple": [2.0, -1.0, 2.0, -1.0],
            }
        )
        folded = _assign_oos_folds(events, ScanConfig())
        stats = aggregate_backtest(folded, ScanConfig())
        self.assertEqual(stats.loc[0, "validation_scope"], "INSUFFICIENT_OOS")
        self.assertEqual(stats.loc[0, "signal_events_oos"], 0)

    def test_validation_gate_keeps_weak_sample_visible(self):
        frame = signal_frame()
        frame["validation_scope"] = "INSUFFICIENT_OOS"
        frame["signal_events_oos"] = 0
        frame["filled_events"] = 0
        out = apply_validation_gate(frame, ScanConfig())
        self.assertEqual(out.loc[0, "status"], "EXECUTION_READY")
        self.assertEqual(out.loc[0, "validation_tier"], "LIMITED")
        self.assertFalse(bool(out.loc[0, "validation_gate_pass"]))

    def test_portfolio_gate_selects_only_available_slot(self):
        frame = pd.concat([signal_frame(), signal_frame().assign(ticker="ANTM.JK")], ignore_index=True)
        frame["quality_score"] = [90, 85]
        frame["bayes_probability"] = [60, 58]
        frame["rr2"] = [3.0, 2.8]
        frame["adtv20_idr"] = [10e9, 10e9]
        frame["max_loss_idr"] = [50_000, 50_000]
        frame["capital_required_idr"] = [2_000_000, 2_000_000]
        out = enforce_portfolio_execution_budget(frame, ScanConfig().replace(max_positions=1))
        self.assertEqual(int(out["portfolio_selected"].sum()), 1)
        self.assertEqual(int(out["status"].eq("EXECUTION_READY").sum()), 1)


    def test_live_and_backtest_use_same_hardened_detectors(self):
        self.assertIs(scanner_module.DETECTORS["PULLBACK_CONTINUATION"], scanner_module.detect_pullback_continuation)
        self.assertIs(scanner_module.DETECTORS["BREAKOUT_RETEST"], scanner_module.detect_breakout_retest)
        self.assertIs(scanner_module.DETECTORS["REVERSAL_ACCUMULATION"], scanner_module.detect_reversal_accumulation)
        self.assertIs(scanner_module.DETECTORS["UNICORN_SNIPER_ICT"], scanner_module.detect_unicorn_sniper)

    def test_quote_snapshot_missing_uses_final_ohlcv_fallback(self):
        frame = signal_frame()
        frame["last_price"] = 1_000.0
        frame["atr_pct"] = 0.03
        frame["absolute_data_age_days"] = 0
        frame["current_bar_incomplete"] = 0
        out = apply_execution_snapshot_gate(frame, pd.DataFrame(), ScanConfig())
        self.assertEqual(out.loc[0, "status"], "EXECUTION_READY")
        self.assertEqual(out.loc[0, "quote_confidence"], 68.0)

    def test_verified_closed_market_quote_preserves_execution(self):
        frame = signal_frame()
        frame["last_price"] = 1_000.0
        frame["atr_pct"] = 0.03
        snap = pd.DataFrame(
            {
                "ticker": ["ADRO.JK"],
                "quote_time": [pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None)],
                "quote_last_price": [1_000.0],
                "quote_volume": [1_000_000],
                "quote_market_state": ["CLOSED"],
                "quote_verified": [True],
            }
        )
        out = apply_execution_snapshot_gate(frame, snap, ScanConfig())
        self.assertEqual(out.loc[0, "status"], "EXECUTION_READY")

    def test_stale_financial_statement_reduces_confidence(self):
        frame = signal_frame()
        frame["fundamental_score"] = 75.0
        frame["fundamental_coverage"] = 85.0
        frame["fundamental_reliability"] = "HIGH"
        frame["fundamental_red_flags"] = ""
        frame["fundamental_error"] = ""
        frame["statement_age_days"] = 300
        out = apply_fundamental_gate(frame, ScanConfig())
        self.assertEqual(out.loc[0, "status"], "EXECUTION_READY")
        self.assertEqual(out.loc[0, "fundamental_tier"], "PARTIAL")

    def test_final_integrity_uses_weighted_evidence_and_critical_gate(self):
        frame = signal_frame()
        frame["quality_score"] = 90.0
        frame["tp1"] = 1_100.0
        frame["tp2"] = 1_150.0
        frame["rr1"] = 2.0
        frame["rr2"] = 3.0
        frame["stop_pct"] = 0.05
        frame["sizing_status"] = "OK"
        frame["suggested_lots"] = 2
        frame["portfolio_selected"] = True
        for col in ("validation_confidence", "fundamental_confidence", "market_status_confidence", "news_confidence", "quote_confidence", "universe_confidence"):
            frame[col] = 100.0
        out = finalize_execution_integrity(frame, ScanConfig())
        self.assertEqual(out.loc[0, "status"], "EXECUTION_READY")
        self.assertGreaterEqual(out.loc[0, "execution_integrity_score"], 90.0)
        self.assertEqual(out.loc[0, "order_instruction"], "BUY_LIMIT")


    def test_small_selected_universe_reduces_confidence_without_erasing_setup(self):
        frame = signal_frame()
        out = apply_universe_integrity_gate(
            frame,
            ["ADRO.JK", "ANTM.JK"],
            ["ADRO.JK", "ANTM.JK"],
            ScanConfig(),
        )
        self.assertFalse(bool(out.loc[0, "universe_gate_pass"]))
        self.assertEqual(out.loc[0, "status"], "EXECUTION_READY")
        self.assertLess(out.loc[0, "universe_confidence"], 70.0)

    def test_broad_high_coverage_universe_passes(self):
        frame = signal_frame()
        requested = [f"X{i:03d}.JK" for i in range(250)]
        prepared = requested[:225]
        out = apply_universe_integrity_gate(frame, requested, prepared, ScanConfig())
        self.assertTrue(bool(out.loc[0, "universe_gate_pass"]))
        self.assertEqual(out.loc[0, "status"], "EXECUTION_READY")

    def test_regular_session_blocks_daily_execution(self):
        frame = signal_frame()
        frame["last_price"] = 1_000.0
        frame["atr_pct"] = 0.03
        snap = pd.DataFrame(
            {
                "ticker": ["ADRO.JK"],
                "quote_time": [pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None)],
                "quote_last_price": [1_000.0],
                "quote_bid": [995.0],
                "quote_ask": [1_000.0],
                "quote_spread_pct": [0.005],
                "quote_volume": [1_000_000],
                "quote_market_state": ["REGULAR"],
                "quote_verified": [True],
            }
        )
        out = apply_execution_snapshot_gate(frame, snap, ScanConfig())
        self.assertEqual(out.loc[0, "status"], "BLOCKED_CONTEXT")
        self.assertIn("Daily candle belum final", out.loc[0, "blockers"])

    def test_smart_money_gate_has_live_historical_parity(self):
        close = np.linspace(1_000, 1_500, 260)
        volume = np.linspace(2_000_000, 5_000_000, 260)
        prepared = prepare_indicators(make_ohlcv(close, volume))
        engine = ScanEngine(ScanConfig().replace(real_money_mode=False))
        _, live_metrics = engine._tradeability(prepared, pd.Timestamp(prepared.index[-1]))
        _, historical_metrics = _historical_gate_inputs(prepared, ScanConfig().replace(real_money_mode=False))
        self.assertEqual(live_metrics["silent_accumulation_score"], historical_metrics["silent_accumulation_score"])


class ResilientEvidenceV4Tests(unittest.TestCase):
    @staticmethod
    def executable_frame() -> pd.DataFrame:
        frame = signal_frame()
        frame["quality_score"] = 90.0
        frame["tp1"] = 1_100.0
        frame["tp2"] = 1_150.0
        frame["rr1"] = 2.0
        frame["rr2"] = 3.0
        frame["stop_pct"] = 0.05
        frame["sizing_status"] = "OK"
        frame["suggested_lots"] = 2
        frame["portfolio_selected"] = True
        return frame

    def test_one_optional_provider_failure_does_not_remove_order(self):
        frame = self.executable_frame()
        frame["market_status_confidence"] = 100.0
        frame["news_confidence"] = 52.0  # news provider unavailable
        frame["fundamental_confidence"] = 100.0
        frame["validation_confidence"] = 100.0
        frame["quote_confidence"] = 100.0
        frame["universe_confidence"] = 100.0
        out = finalize_execution_integrity(frame, ScanConfig())
        self.assertEqual(out.loc[0, "status"], "EXECUTION_READY")
        self.assertEqual(out.loc[0, "order_instruction"], "BUY_LIMIT")

    def test_multiple_missing_layers_produce_pending_data_not_disappearing_signal(self):
        frame = self.executable_frame()
        frame["market_status_confidence"] = 45.0
        frame["news_confidence"] = 52.0
        frame["fundamental_confidence"] = 50.0
        frame["validation_confidence"] = 45.0
        frame["quote_confidence"] = 68.0
        frame["universe_confidence"] = 48.0
        out = finalize_execution_integrity(frame, ScanConfig())
        self.assertEqual(out.loc[0, "status"], "PENDING_DATA")
        self.assertEqual(out.loc[0, "order_instruction"], "DO_NOT_BUY")
        self.assertGreaterEqual(out.loc[0, "execution_confidence_score"], ScanConfig().min_pending_confidence)

    def test_demonstrated_negative_oos_is_critical_blocker(self):
        frame = signal_frame()
        frame["validation_scope"] = "CHRONOLOGICAL_OOS_HOLDOUT"
        frame["signal_events_oos"] = 40
        frame["filled_events"] = 25
        frame["entry_fill_rate_5d"] = 62.5
        frame["bayes_probability"] = 40.0
        frame["tp1_ci_low"] = 25.0
        frame["expectancy_r"] = -0.20
        frame["profit_factor"] = 0.70
        frame["max_losing_streak"] = 7
        out = apply_validation_gate(frame, ScanConfig())
        self.assertEqual(out.loc[0, "status"], "BLOCKED_CONTEXT")
        self.assertTrue(bool(out.loc[0, "validation_critical_blocker"]))


class PortfolioDecisionV4Tests(unittest.TestCase):
    def test_portfolio_parser_aggregates_duplicate_ticker_at_weighted_average(self):
        raw = pd.DataFrame(
            {
                "ticker": ["ADRO", "ADRO.JK"],
                "lots": [2, 3],
                "avg_price": [1_000, 1_200],
            }
        )
        parsed = parse_portfolio_csv(raw)
        self.assertEqual(parsed.loc[0, "lots"], 5)
        self.assertAlmostEqual(parsed.loc[0, "avg_price"], 1_120.0)

    def test_broken_structure_generates_cut_loss(self):
        close = np.linspace(1_500, 850, 280)
        portfolio = pd.DataFrame(
            {
                "ticker": ["TEST.JK"], "lots": [2], "shares": [200],
                "avg_price": [1_200.0], "manual_stop_loss": [1_000.0],
                "manual_tp": [np.nan], "notes": [""],
            }
        )
        result, summary = analyze_portfolio_positions(
            portfolio,
            {"TEST.JK": make_ohlcv(close)},
            account_equity_idr=10_000_000,
            cash_on_hand_idr=5_000_000,
        )
        self.assertEqual(result.loc[0, "position_action"], "CUT_LOSS")
        self.assertGreater(summary["open_risk_idr"], 0)

    def test_average_down_requires_active_setup_and_intact_structure(self):
        base = np.linspace(800, 1_500, 260)
        tail = np.array([
            1_500, 1_490, 1_480, 1_470, 1_460, 1_450, 1_445, 1_440, 1_445, 1_450,
            1_455, 1_460, 1_465, 1_470, 1_465, 1_460, 1_455, 1_460, 1_465, 1_470,
        ])
        close = np.r_[base, tail]
        raw = make_ohlcv(close)
        raw["Volume"] = np.where(raw["Close"] >= raw["Open"], 6_000_000, 2_000_000)
        portfolio = pd.DataFrame(
            {
                "ticker": ["TEST.JK"], "lots": [2], "shares": [200],
                "avg_price": [1_520.0], "manual_stop_loss": [np.nan],
                "manual_tp": [np.nan], "notes": [""],
            }
        )
        signals = pd.DataFrame(
            {
                "ticker": ["TEST.JK"], "status": ["PENDING_DATA"],
                "status_rank": [1], "quality_score": [90.0],
                "setup": ["PULLBACK_CONTINUATION"],
            }
        )
        result, _ = analyze_portfolio_positions(
            portfolio,
            {"TEST.JK": raw},
            signals=signals,
            account_equity_idr=10_000_000,
            cash_on_hand_idr=5_000_000,
        )
        self.assertEqual(result.loc[0, "position_action"], "AVG_DOWN_ALLOWED")
        self.assertGreaterEqual(result.loc[0, "avg_down_lots"], 1)


if __name__ == "__main__":
    unittest.main()
