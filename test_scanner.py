from __future__ import annotations

import sys
import ast
import unittest
from unittest.mock import patch
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
    apply_independent_price_gate,
    apply_universe_integrity_gate,
    attach_broker_summary,
    attach_position_sizing,
    enforce_portfolio_execution_budget,
    apply_analyst_fusion_gate,
    enforce_analyst_portfolio_budget,
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
    specialty_intraday_shortlist,
    scan_sniper_entries,
    scan_bsjp_candidates,
    scan_bpjs_candidates,
    scan_multibagger_candidates,
    scan_ara_hunter_candidates,
    build_specialty_screens,
    build_independent_price_validation,
    fetch_automatic_independent_prices,
    fetch_google_finance_quotes,
    fetch_idx_official_eod_quotes,
    fetch_twelve_data_eod,
    parse_independent_price_file,
    parse_orderbook_snapshot_csv,
    apply_ara_external_confirmation,
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
        self.assertTrue((ROOT / "scanner_specialty.py").is_file())
        self.assertTrue((ROOT / "test_scanner.py").is_file())
        self.assertFalse((ROOT / "scanner").exists())
        self.assertFalse((ROOT / "tests").exists())


class ModularArchitectureV500Tests(unittest.TestCase):
    def test_no_duplicate_top_level_definitions(self):
        for filename in ("scanner.py", "scanner_specialty.py"):
            tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
            names = [
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            ]
            self.assertEqual(len(names), len(set(names)), filename)

    def test_no_duplicate_methods_inside_classes(self):
        for filename in ("scanner.py", "scanner_specialty.py"):
            tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                names = [
                    item.name
                    for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                ]
                self.assertEqual(len(names), len(set(names)), f"{filename}:{node.name}")

    def test_specialty_functions_live_in_second_module(self):
        self.assertEqual(scanner_module.scan_sniper_entries.__module__, "scanner_specialty")
        self.assertEqual(scanner_module.scan_bpjs_candidates.__module__, "scanner_specialty")
        self.assertEqual(scanner_module.scan_bsjp_candidates.__module__, "scanner_specialty")
        self.assertEqual(scanner_module.scan_ara_hunter_candidates.__module__, "scanner_specialty")



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
    def test_default_profile_matches_rp5m_account(self):
        cfg = ScanConfig()
        self.assertEqual(cfg.account_size_idr, 5_000_000.0)
        self.assertEqual(cfg.cash_on_hand_idr, 5_000_000.0)
        self.assertEqual(cfg.max_positions, 3)
        self.assertAlmostEqual(cfg.risk_per_trade_pct, 0.005)

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

    def test_account_too_small_does_not_downgrade_signal_first_execution(self):
        cfg = ScanConfig().replace(account_size_idr=100_000, max_position_pct=0.10)
        out = attach_position_sizing(signal_frame(), cfg)
        self.assertEqual(out.loc[0, "status"], "EXECUTION_READY")
        self.assertEqual(int(out.loc[0, "suggested_lots"]), 0)
        self.assertTrue(bool(out.loc[0, "sizing_is_informational"]))
        self.assertFalse(bool(out.loc[0, "account_risk_gate_applied"]))


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
                    "coverage_start": [(pd.Timestamp.now(tz="Asia/Jakarta") - pd.DateOffset(days=7)).date().isoformat()],
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
        frame["last_price"] = 1_000.0
        frame["entry_low"] = 980.0
        frame["entry_high"] = 1_000.0
        frame["tp1"] = 1_100.0
        frame["tp2"] = 1_150.0
        frame["rr1"] = 2.0
        frame["rr2"] = 3.0
        frame["stop_pct"] = 0.05
        frame["sizing_status"] = "OK"
        frame["suggested_lots"] = 2
        frame["capital_required_idr"] = 200_000.0
        frame["max_loss_idr"] = 10_000.0
        frame["portfolio_selected"] = True
        frame["fundamental_coverage"] = 80.0
        frame["fundamental_score"] = 70.0
        frame["volume_ratio"] = 1.4
        frame["adtv20_idr"] = 5_000_000_000.0
        frame["distance_atr"] = 0.1
        frame["market_regime"] = "RISK_ON"
        frame["independent_price_verified"] = True
        frame["independent_price_state"] = "VERIFIED_INDEPENDENT"
        frame["independent_source_family"] = "IDX_OFFICIAL"
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
        self.assertEqual(out.loc[0, "status"], "EXECUTION_READY")
        self.assertTrue(bool(out.loc[0, "pending_close"]))
        self.assertFalse(bool(out.loc[0, "quote_critical_blocker"]))
        self.assertIn("menunggu candle EOD final", out.loc[0, "evidence_warnings"])

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
        frame["last_price"] = 1_000.0
        frame["entry_low"] = 980.0
        frame["entry_high"] = 1_000.0
        frame["tp1"] = 1_100.0
        frame["tp2"] = 1_150.0
        frame["rr1"] = 2.0
        frame["rr2"] = 3.0
        frame["stop_pct"] = 0.05
        frame["sizing_status"] = "OK"
        frame["suggested_lots"] = 2
        frame["capital_required_idr"] = 200_000.0
        frame["max_loss_idr"] = 10_000.0
        frame["portfolio_selected"] = True
        frame["fundamental_coverage"] = 80.0
        frame["fundamental_score"] = 70.0
        frame["volume_ratio"] = 1.4
        frame["adtv20_idr"] = 5_000_000_000.0
        frame["distance_atr"] = 0.1
        frame["market_regime"] = "RISK_ON"
        frame["independent_price_verified"] = True
        frame["independent_price_state"] = "VERIFIED_INDEPENDENT"
        frame["independent_source_family"] = "IDX_OFFICIAL"
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

    def test_missing_fundamental_snapshot_cannot_issue_direct_order(self):
        frame = self.executable_frame()
        frame["fundamental_coverage"] = 0.0
        frame["fundamental_score"] = np.nan
        frame["market_status_confidence"] = 100.0
        frame["news_confidence"] = 100.0
        frame["fundamental_confidence"] = 100.0
        frame["validation_confidence"] = 100.0
        frame["quote_confidence"] = 100.0
        frame["universe_confidence"] = 100.0
        out = finalize_execution_integrity(frame, ScanConfig())
        self.assertEqual(out.loc[0, "status"], "PENDING_DATA")
        self.assertEqual(out.loc[0, "order_instruction"], "DO_NOT_BUY")
        self.assertIn("Fundamental coverage", out.loc[0, "evidence_warnings"])

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


class PortfolioDecisionV41RegressionTests(unittest.TestCase):
    def test_portfolio_parser_accepts_utf8_bom(self):
        raw = b"\xef\xbb\xbfticker,lots,avg_price,stop_loss,take_profit,notes\nADRO,2,1200,,,test\n"
        parsed = parse_portfolio_csv(raw)
        self.assertEqual(parsed.loc[0, "ticker"], "ADRO.JK")
        self.assertEqual(int(parsed.loc[0, "lots"]), 2)

    def test_weak_trend_above_stop_does_not_force_cut_loss(self):
        close = np.r_[np.linspace(900, 1_500, 279), 1_410]
        portfolio = pd.DataFrame(
            {
                "ticker": ["TEST.JK"], "lots": [1], "shares": [100],
                "avg_price": [1_300.0], "manual_stop_loss": [1_200.0],
                "manual_tp": [np.nan], "notes": [""],
            }
        )
        result, _ = analyze_portfolio_positions(
            portfolio, {"TEST.JK": make_ohlcv(close)},
            account_equity_idr=5_000_000, cash_on_hand_idr=1_000_000,
        )
        self.assertNotEqual(result.loc[0, "position_action"], "CUT_LOSS")
        self.assertFalse(bool(result.loc[0, "stop_breached"]))

    def test_position_weight_uses_selected_equity_basis(self):
        close = np.linspace(1_000, 1_000, 280)
        portfolio = pd.DataFrame(
            {
                "ticker": ["TEST.JK"], "lots": [10], "shares": [1_000],
                "avg_price": [1_000.0], "manual_stop_loss": [900.0],
                "manual_tp": [np.nan], "notes": [""],
            }
        )
        result, summary = analyze_portfolio_positions(
            portfolio, {"TEST.JK": make_ohlcv(close)},
            account_equity_idr=5_000_000, cash_on_hand_idr=1_000_000,
        )
        self.assertAlmostEqual(float(result.loc[0, "position_weight"]), 0.20, places=6)
        self.assertAlmostEqual(float(result.loc[0, "position_weight_pct"]), 20.0, places=6)
        self.assertEqual(summary["equity_source"], "ACCOUNT_EQUITY_INPUT")

    def test_tp2_has_meaningful_separation_from_tp1(self):
        close = np.linspace(800, 1_200, 280)
        portfolio = pd.DataFrame(
            {
                "ticker": ["TEST.JK"], "lots": [1], "shares": [100],
                "avg_price": [1_000.0], "manual_stop_loss": [1_100.0],
                "manual_tp": [np.nan], "notes": [""],
            }
        )
        result, _ = analyze_portfolio_positions(
            portfolio, {"TEST.JK": make_ohlcv(close)},
            account_equity_idr=5_000_000, cash_on_hand_idr=1_000_000,
        )
        tp1 = float(result.loc[0, "suggested_tp1"])
        tp2 = float(result.loc[0, "suggested_tp2"])
        self.assertGreater(tp2, tp1)
        self.assertGreaterEqual(tp2 - tp1, 0.02 * float(result.loc[0, "last_price"]))


class SpecialtyScannerV42Tests(unittest.TestCase):
    @staticmethod
    def _strong_daily_frame() -> pd.DataFrame:
        close = np.r_[np.linspace(700, 980, 279), 1_020]
        volume = np.full(len(close), 8_000_000.0)
        volume[-1] = 18_000_000.0
        frame = prepare_indicators(make_ohlcv(close, volume))
        idx = frame.index[-1]
        frame.at[idx, "ADTV20"] = 8_000_000_000.0
        frame.at[idx, "VOL_RATIO"] = 2.2
        frame.at[idx, "CMF20"] = 0.15
        frame.at[idx, "OBV_SLOPE10"] = 1.0
        frame.at[idx, "CLOSE_LOCATION"] = 0.92
        frame.at[idx, "REL_STRENGTH60"] = 0.12
        frame.at[idx, "RSI14"] = 68.0
        frame.at[idx, "MFI14"] = 70.0
        frame.at[idx, "BODY_ATR"] = 1.10
        frame.at[idx, "HIGH20_PREV"] = 1_000.0
        frame.at[idx, "EMA20"] = 980.0
        frame.at[idx, "EMA50"] = 930.0
        frame.at[idx, "EMA200"] = 800.0
        return frame

    @staticmethod
    def _intraday(mode: str) -> pd.DataFrame:
        index = pd.date_range("2026-07-13 09:00", periods=28, freq="15min")
        close = np.linspace(1_000, 1_030, len(index))
        open_ = np.r_[998.0, close[:-1]]
        high = np.maximum(open_, close) + 2
        low = np.minimum(open_, close) - 2
        if mode == "bsjp":
            volume = np.r_[np.full(22, 500_000.0), np.full(6, 2_000_000.0)]
        else:
            volume = np.r_[2_000_000.0, np.full(len(index) - 1, 500_000.0)]
        return pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=index,
        )

    def test_intraday_shortlist_is_bounded(self):
        prepared = {f"T{i}.JK": self._strong_daily_frame() for i in range(10)}
        result = specialty_intraday_shortlist(prepared, max_candidates=4)
        self.assertEqual(len(result), 4)

    def test_strict_sniper_has_dedicated_ready_state(self):
        frame = self._strong_daily_frame()
        core = pd.DataFrame([{
            "ticker": "TEST.JK", "setup": "UNICORN_SNIPER_ICT",
            "status": "EXECUTION_READY", "quality_score": 92.0,
            "distance_atr": 0.10, "volume_ratio": 1.8,
            "silent_accumulation_score": 85.0, "rr1": 2.2, "rr2": 3.8,
            "stop_pct": 0.04, "last_price": 1020.0, "entry_low": 1005.0,
            "entry_high": 1020.0, "entry": 1015.0, "stop_loss": 975.0,
            "tp1": 1105.0, "tp2": 1170.0, "valid_until": pd.Timestamp("2026-07-20"),
            "blockers": "", "reason": "strict confluence",
        }])
        result = scan_sniper_entries(core, {"TEST.JK": frame})
        self.assertEqual(result.loc[0, "sniper_status"], "SNIPER_READY")
        self.assertGreaterEqual(float(result.loc[0, "sniper_score"]), 82.0)

    def test_strict_sniper_cache_fallback_cannot_be_ready(self):
        frame = self._strong_daily_frame()
        core = pd.DataFrame([{
            "ticker": "TEST.JK", "setup": "UNICORN_SNIPER_ICT",
            "status": "EXECUTION_READY", "quality_score": 92.0,
            "distance_atr": 0.10, "volume_ratio": 1.8,
            "silent_accumulation_score": 85.0, "rr1": 2.2, "rr2": 3.8,
            "stop_pct": 0.04, "last_price": 1020.0, "entry_low": 1005.0,
            "entry_high": 1020.0, "entry": 1015.0, "stop_loss": 975.0,
            "tp1": 1105.0, "tp2": 1170.0, "valid_until": pd.Timestamp("2026-07-20"),
            "blockers": "", "reason": "strict confluence",
            "ohlcv_source_tier": "CACHE_FALLBACK",
        }])
        result = scan_sniper_entries(core, {"TEST.JK": frame})
        self.assertNotEqual(result.loc[0, "sniper_status"], "SNIPER_READY")
        self.assertIn("bukan hasil live", result.loc[0, "blockers"])

    def test_sniper_limit_retrace_can_be_ready_without_core_execution_status(self):
        frame = self._strong_daily_frame()
        core = pd.DataFrame([{
            "ticker": "TEST.JK", "setup": "UNICORN_SNIPER_ICT",
            "status": "WATCHLIST_ENTRY", "action": "WAIT_FVG_RETRACE",
            "detected": True, "invalidated": False,
            "quality_score": 74.0, "distance_atr": 0.55,
            "volume_ratio": 0.80, "silent_accumulation_score": 55.0,
            "rr1": 1.4, "rr2": 2.2, "stop_pct": 0.09,
            "last_price": 1020.0, "entry_low": 1005.0,
            "entry_high": 1020.0, "entry": 1015.0, "stop_loss": 925.0,
            "tp1": 1140.0, "tp2": 1215.0,
            "evidence": "Sell-side liquidity sweep • Bullish BOS dengan displacement • Bullish FVG valid",
            "valid_until": pd.Timestamp("2027-07-20"),
            "ohlcv_source_tier": "LIVE_YAHOO", "blockers": "RR warning",
        }])
        result = scan_sniper_entries(core, {"TEST.JK": frame})
        self.assertEqual(result.loc[0, "sniper_status"], "SNIPER_READY")
        self.assertEqual(result.loc[0, "sniper_entry_mode"], "LIMIT_FVG_RETRACE")
        self.assertIn("HEALTHY_RETRACE_CONTRACTION", result.loc[0, "volume_context"])
        self.assertIn("SL lebar", result.loc[0, "risk_warnings"])

    def test_sniper_does_not_require_current_volume_expansion_on_retrace(self):
        frame = self._strong_daily_frame()
        core = pd.DataFrame([{
            "ticker": "TEST.JK", "setup": "UNICORN_SNIPER_ICT",
            "status": "WATCHLIST_ENTRY", "action": "WAIT_STRICT_UNICORN_CONFLUENCE",
            "detected": True, "invalidated": False,
            "quality_score": 72.0, "distance_atr": 0.75,
            "volume_ratio": 0.65, "silent_accumulation_score": 48.0,
            "rr1": 1.2, "rr2": 1.8, "stop_pct": 0.12,
            "last_price": 1020.0, "entry_low": 1005.0,
            "entry_high": 1020.0, "entry": 1015.0, "stop_loss": 895.0,
            "tp1": 1160.0, "tp2": 1230.0,
            "evidence": "Sell-side liquidity sweep • Bullish BOS dengan displacement • Bullish FVG valid",
            "valid_until": pd.Timestamp("2027-07-20"),
            "ohlcv_source_tier": "LIVE_YAHOO", "blockers": "",
        }])
        result = scan_sniper_entries(core, {"TEST.JK": frame})
        self.assertEqual(result.loc[0, "sniper_status"], "SNIPER_READY")
        self.assertEqual(result.loc[0, "primary_sniper_blocker"], "NONE")

    def test_bsjp_ready_requires_late_session_intraday(self):
        frame = self._strong_daily_frame()
        result = scan_bsjp_candidates(
            {"TEST.JK": frame}, {"TEST.JK": self._intraday("bsjp")},
            now="2026-07-13 15:30:00",
        )
        self.assertFalse(result.empty)
        self.assertEqual(result.loc[0, "bsjp_status"], "BSJP_READY")
        self.assertLessEqual(float(result.loc[0, "risk_pct"]), 0.035)
        self.assertTrue(bool(result.loc[0, "intraday_fresh"]))
        self.assertEqual(result.loc[0, "order_instruction"], "BUY_LIMIT_USER_SIZE")

    def test_bpjs_ready_requires_opening_range_and_vwap(self):
        frame = self._strong_daily_frame()
        result = scan_bpjs_candidates(
            {"TEST.JK": frame}, {"TEST.JK": self._intraday("bpjs")},
            now="2026-07-13 09:45:00",
        )
        self.assertFalse(result.empty)
        self.assertEqual(result.loc[0, "bpjs_status"], "BPJS_READY")
        self.assertEqual(result.loc[0, "mandatory_exit"], "Before regular-market close")
        self.assertEqual(result.loc[0, "order_instruction"], "BUY_LIMIT_USER_SIZE")

    def test_stale_intraday_session_can_never_be_ready(self):
        frame = self._strong_daily_frame()
        result = scan_bpjs_candidates(
            {"TEST.JK": frame}, {"TEST.JK": self._intraday("bpjs")},
            now="2026-07-14 09:45:00",
        )
        self.assertFalse(result.empty)
        self.assertEqual(result.loc[0, "bpjs_status"], "BPJS_STALE_INTRADAY")
        self.assertFalse(bool(result.loc[0, "intraday_fresh"]))

    def test_intraday_future_bars_are_excluded(self):
        metrics = scanner_module._intraday_metrics(
            self._intraday("bpjs"), now="2026-07-13 09:45:00"
        )
        self.assertEqual(int(metrics["intraday_bars"]), 3)
        self.assertLessEqual(pd.Timestamp(metrics["intraday_last_bar_time"]), pd.Timestamp("2026-07-13 09:30"))

    def test_risk_off_regime_is_disclosure_for_bsjp_ready(self):
        frame = self._strong_daily_frame()
        result = scan_bsjp_candidates(
            {"TEST.JK": frame}, {"TEST.JK": self._intraday("bsjp")},
            now="2026-07-13 15:30:00",
            market_context=MarketContext(regime="RISK_OFF"),
        )
        self.assertEqual(result.loc[0, "bsjp_status"], "BSJP_READY")
        self.assertIn("RISK_OFF", result.loc[0, "warnings"])
        self.assertEqual(result.loc[0, "blockers"], "")

    def test_bpjs_5m_waits_until_opening_range_has_post_orb_bar(self):
        frame = self._strong_daily_frame()
        index = pd.date_range("2026-07-13 09:00", periods=3, freq="5min")
        close = np.array([1000.0, 1002.0, 1004.0])
        intraday = pd.DataFrame(
            {
                "Open": np.array([998.0, 1000.0, 1002.0]),
                "High": close + 1.0,
                "Low": np.array([997.0, 999.0, 1001.0]),
                "Close": close,
                "Volume": np.full(3, 2_000_000.0),
            },
            index=index,
        )
        intraday.attrs["interval_minutes"] = 5.0
        result = scan_bpjs_candidates(
            {"TEST.JK": frame}, {"TEST.JK": intraday},
            now="2026-07-13 09:12:00",
        )
        self.assertFalse(result.empty)
        self.assertEqual(result.loc[0, "bpjs_status"], "BPJS_WAIT_OPENING_BARS")
        self.assertEqual(result.loc[0, "intraday_data_state"], "OPENING_RANGE_FORMING")
        self.assertNotEqual(result.loc[0, "action"], "FETCH_15M_DATA")

    def test_bpjs_5m_becomes_measurable_after_first_post_orb_bar(self):
        frame = self._strong_daily_frame()
        index = pd.date_range("2026-07-13 09:00", periods=5, freq="5min")
        close = np.array([1000.0, 1002.0, 1004.0, 1010.0, 1015.0])
        intraday = pd.DataFrame(
            {
                "Open": np.array([998.0, 1000.0, 1002.0, 1004.0, 1010.0]),
                "High": close + 1.0,
                "Low": np.array([997.0, 999.0, 1001.0, 1003.0, 1009.0]),
                "Close": close,
                "Volume": np.array([2_000_000.0, 2_000_000.0, 2_000_000.0, 500_000.0, 500_000.0]),
            },
            index=index,
        )
        intraday.attrs["interval_minutes"] = 5.0
        result = scan_bpjs_candidates(
            {"TEST.JK": frame}, {"TEST.JK": intraday},
            now="2026-07-13 09:25:00",
        )
        self.assertFalse(result.empty)
        self.assertEqual(result.loc[0, "intraday_data_state"], "LIVE_READY")
        self.assertGreaterEqual(int(result.loc[0, "post_orb_bars"]), 1)
        self.assertIn(result.loc[0, "bpjs_status"], {"BPJS_READY", "BPJS_WATCHLIST"})

    def test_bpjs_no_provider_data_is_distinct_from_waiting_bars(self):
        frame = self._strong_daily_frame()
        result = scan_bpjs_candidates(
            {"TEST.JK": frame}, {},
            now="2026-07-13 09:25:00",
        )
        self.assertFalse(result.empty)
        self.assertEqual(result.loc[0, "bpjs_status"], "BPJS_DATA_UNAVAILABLE")
        self.assertEqual(result.loc[0, "action"], "RETRY_INTRADAY_5M")

    def test_multibagger_combines_growth_quality_and_momentum(self):
        frame = self._strong_daily_frame()
        fundamentals = pd.DataFrame([{
            "ticker": "TEST.JK", "fundamental_coverage": 90.0,
            "fundamental_score": 90.0, "revenue_growth": 0.30,
            "earnings_growth": 0.40, "roe": 0.25, "roa": 0.10,
            "net_margin": 0.15, "operating_margin": 0.20,
            "debt_equity": 0.40, "cash_to_debt": 1.0,
            "operating_cash_flow": 1_000_000_000.0,
            "free_cash_flow": 800_000_000.0, "peg_ratio": 1.0,
            "fcf_yield": 0.05, "market_cap": 5_000_000_000_000.0,
            "fundamental_red_flags": "",
        }])
        result = scan_multibagger_candidates({"TEST.JK": frame}, fundamentals)
        self.assertFalse(result.empty)
        self.assertIn(result.loc[0, "multibagger_status"], {"MULTIBAGGER_A_CANDIDATE", "MULTIBAGGER_B_CANDIDATE"})
        self.assertGreaterEqual(float(result.loc[0, "multibagger_score"]), 72.0)

    def test_missing_solvency_metrics_receive_no_free_points(self):
        frame = self._strong_daily_frame()
        fundamentals = pd.DataFrame([{
            "ticker": "TEST.JK", "fundamental_coverage": 75.0,
            "fundamental_score": 85.0, "fundamental_model": "GENERAL",
            "revenue_growth": 0.30, "earnings_growth": 0.40,
            "roe": 0.25, "roa": 0.10, "net_margin": 0.15,
            "operating_margin": 0.20, "debt_equity": np.nan,
            "current_ratio": np.nan, "cash_to_debt": np.nan,
            "operating_cash_flow": 1_000_000_000.0,
            "free_cash_flow": 800_000_000.0, "peg_ratio": 1.0,
            "fcf_yield": 0.05, "market_cap": 5_000_000_000_000.0,
            "fundamental_red_flags": "",
        }])
        result = scan_multibagger_candidates({"TEST.JK": frame}, fundamentals)
        self.assertFalse(result.empty)
        self.assertEqual(float(result.loc[0, "solvency_coverage"]), 0.0)
        self.assertLessEqual(float(result.loc[0, "balance_sheet_score"]), 7.0)
        self.assertNotEqual(result.loc[0, "multibagger_status"], "MULTIBAGGER_A_CANDIDATE")

    def test_ara_hunter_signal_first_is_not_suppressed_by_account_risk(self):
        frame = self._strong_daily_frame()
        result = scan_ara_hunter_candidates(
            {"TEST.JK": frame}, {"TEST.JK": self._intraday("bsjp")},
            now="2026-07-13 20:00:00",
        )
        self.assertFalse(result.empty)
        self.assertIn(result.loc[0, "ara_hunter_status"], {
            "PRE_ARA_READY", "PRE_ARA_CANDIDATE", "ARA_CONTINUATION_READY",
            "ARA_CONTINUATION_CANDIDATE", "ARA_CONFIRMED_ONLY"
        })
        self.assertFalse(bool(result.loc[0, "account_risk_gate_applied"]))
        self.assertNotEqual(result.loc[0, "order_instruction"], "SPECULATIVE_REVIEW_ONLY")

    def test_specialty_builder_returns_all_named_tables(self):
        frame = self._strong_daily_frame()
        result = build_specialty_screens({"TEST.JK": frame})
        self.assertEqual(set(result), {"sniper", "bsjp", "bpjs", "multibagger", "ara_hunter"})


class ResilientCacheHotfixTests(unittest.TestCase):
    def test_timezone_aware_cache_timestamp_does_not_crash(self):
        cached = pd.DataFrame([
            {
                "ticker": "TEST.JK",
                "fundamental_coverage": 80.0,
                "fundamental_error": "",
                "fundamental_fetched_at": pd.Timestamp.now(tz="Asia/Jakarta").isoformat(),
            }
        ])
        result = scanner_module._merge_resilient_rows(
            pd.DataFrame(),
            cached,
            ["TEST.JK"],
            "fundamental_fetched_at",
            lambda row: float(row.get("fundamental_coverage", 0)) >= 45,
            75,
            "CACHE_FALLBACK",
        )
        self.assertEqual(result.loc[0, "evidence_source_tier"], "CACHE_FALLBACK")

    def test_fresh_fundamental_cache_is_used_before_yahoo_request(self):
        cached = pd.DataFrame([
            {
                "ticker": "TEST.JK",
                "fundamental_coverage": 80.0,
                "fundamental_score": 75.0,
                "fundamental_error": "",
                "fundamental_fetched_at": pd.Timestamp.now(tz="Asia/Jakarta").isoformat(),
            }
        ])
        with (
            patch.object(scanner_module, "_load_cache", return_value=cached),
            patch.object(scanner_module, "fetch_fundamentals") as fetch_mock,
            patch.object(scanner_module, "_write_cache"),
        ):
            result = scanner_module.fetch_resilient_fundamentals(["TEST.JK"])
        fetch_mock.assert_not_called()
        self.assertEqual(result.loc[0, "evidence_source_tier"], "CACHE_FALLBACK")
        self.assertEqual(float(result.loc[0, "fundamental_score"]), 75.0)


class DataCompletenessV422Tests(unittest.TestCase):
    def _complete_execution_row(self) -> pd.DataFrame:
        frame = signal_frame()
        frame["quality_score"] = 90.0
        frame["setup"] = "PULLBACK_CONTINUATION"
        frame["last_price"] = 1_000.0
        frame["entry_low"] = 980.0
        frame["entry_high"] = 1_000.0
        frame["tp1"] = 1_100.0
        frame["tp2"] = 1_150.0
        frame["rr1"] = 2.0
        frame["rr2"] = 3.0
        frame["stop_pct"] = 0.05
        frame["volume_ratio"] = 1.4
        frame["adtv20_idr"] = 5_000_000_000.0
        frame["distance_atr"] = 0.1
        frame["market_regime"] = "BULLISH"
        frame["sizing_status"] = "OK"
        frame["suggested_lots"] = 2
        frame["capital_required_idr"] = 200_000.0
        frame["max_loss_idr"] = 10_000.0
        frame["portfolio_selected"] = True
        frame["fundamental_coverage"] = 92.0
        frame["fundamental_confidence"] = 70.0
        frame["validation_scope"] = "CHRONOLOGICAL_OOS_HOLDOUT"
        frame["signal_events_oos"] = 14
        frame["filled_events"] = 14
        frame["entry_fill_rate_5d"] = 50.0
        frame["bayes_probability"] = 50.0
        frame["tp1_ci_low"] = 30.0
        frame["expectancy_r"] = 0.05
        frame["profit_factor"] = 1.05
        frame["max_losing_streak"] = 4
        frame["median_fill_bars"] = 2.0
        frame["median_time_to_tp1_bars"] = 6.0
        frame["validation_confidence"] = 55.6
        frame["market_status_coverage"] = "FALLBACK_REQUIRED"
        frame["market_status_confidence"] = 45.0
        frame["absolute_data_age_days"] = 0
        frame["current_bar_incomplete"] = False
        frame["news_review_status"] = "MISSING"
        frame["news_confidence"] = 52.0
        frame["quote_verified"] = False
        frame["quote_confidence"] = 68.0
        frame["universe_requested_count"] = 250
        frame["universe_prepared_count"] = 240
        frame["universe_coverage_pct"] = 96.0
        frame["universe_confidence"] = 100.0
        return frame

    def test_complete_but_weak_layers_still_exceed_80_data_coverage(self):
        out = finalize_execution_integrity(self._complete_execution_row(), ScanConfig())
        self.assertGreaterEqual(out.loc[0, "data_completeness_score"], 80.0)
        self.assertIn(out.loc[0, "data_completeness_tier"], {"SUFFICIENT", "HIGH"})
        self.assertGreater(out.loc[0, "validation_data_coverage"], out.loc[0, "validation_confidence"])

    def test_data_floor_blocks_order_even_when_confidence_is_high(self):
        frame = ResilientEvidenceV4Tests.executable_frame()
        for column in (
            "market_status_confidence", "fundamental_confidence",
            "validation_confidence", "quote_confidence", "universe_confidence",
        ):
            frame[column] = 100.0
        frame["news_confidence"] = 52.0
        out = finalize_execution_integrity(frame, ScanConfig().replace(min_data_completeness=95.0))
        self.assertEqual(out.loc[0, "status"], "PENDING_DATA")
        self.assertEqual(out.loc[0, "order_instruction"], "DO_NOT_BUY")
        self.assertLess(out.loc[0, "data_completeness_score"], 95.0)


class NewsFallbackV422Tests(unittest.TestCase):
    def test_google_news_rss_parser_returns_dated_items(self):
        xml = b"""<?xml version='1.0' encoding='UTF-8'?>
        <rss><channel><item>
          <title>BBCA catat pertumbuhan laba</title>
          <link>https://example.com/news</link>
          <description>Bank melaporkan growth</description>
          <pubDate>Mon, 13 Jul 2026 08:00:00 GMT</pubDate>
        </item></channel></rss>"""
        response = SimpleNamespace(content=xml)
        response.raise_for_status = lambda: None
        fake_requests = SimpleNamespace(get=lambda *args, **kwargs: response)
        with patch.dict(sys.modules, {"requests": fake_requests}):
            items, ok, error = scanner_module._google_news_rss_items("BBCA.JK", 7)
        self.assertTrue(ok)
        self.assertEqual(error, "")
        self.assertEqual(len(items), 1)
        self.assertIn("pertumbuhan laba", items[0]["title"])
        self.assertTrue(items[0]["pubDate"])


class DailyOhlcvResilienceV424Tests(unittest.TestCase):
    def test_empty_provider_batch_uses_verified_cache_without_retry_storm(self):
        cached = make_ohlcv(np.linspace(900.0, 1_100.0, 260))
        fake_yfinance = SimpleNamespace(download=lambda *args, **kwargs: pd.DataFrame())
        with (
            patch.dict(sys.modules, {"yfinance": fake_yfinance}),
            patch.object(scanner_module, "_load_daily_ohlcv_cache", return_value=cached),
            patch.object(scanner_module, "_write_daily_ohlcv_cache"),
        ):
            histories, report = scanner_module.download_ohlcv(["TEST.JK"], period="2y")
        self.assertIn("TEST.JK", histories)
        self.assertEqual(report.source_tiers["TEST.JK"], "CACHE_FALLBACK")
        self.assertIn("cache fallback", report.warnings["TEST.JK"].lower())
        self.assertNotIn("TEST.JK", report.failed)

    def test_benchmark_uses_cache_when_live_provider_fails(self):
        cached = make_ohlcv(np.linspace(6_500.0, 7_100.0, 260))

        class FailingTicker:
            def __init__(self, ticker):
                self.ticker = ticker

            def history(self, **kwargs):
                raise RuntimeError("rate limited")

        fake_yfinance = SimpleNamespace(Ticker=FailingTicker)
        with (
            patch.dict(sys.modules, {"yfinance": fake_yfinance}),
            patch.object(scanner_module, "_load_daily_ohlcv_cache", return_value=cached),
        ):
            result = scanner_module.download_benchmark(period="2y")
        self.assertEqual(len(result), len(cached))
        self.assertAlmostEqual(float(result["Close"].iloc[-1]), float(cached["Close"].iloc[-1]))

    def test_cache_fallback_can_remain_watchlist_but_never_direct_order(self):
        frame = ResilientEvidenceV4Tests.executable_frame()
        frame["ohlcv_source_tier"] = "CACHE_FALLBACK"
        out = finalize_execution_integrity(frame, ScanConfig())
        self.assertEqual(out.loc[0, "status"], "PENDING_DATA")
        self.assertEqual(out.loc[0, "order_instruction"], "DO_NOT_BUY")
        self.assertIn("live_ohlcv", out.loc[0, "data_missing_layers"])
        self.assertIn("cache", out.loc[0, "evidence_warnings"].lower())


class MultiSourceExecutionV425Tests(unittest.TestCase):
    def test_idx_stock_summary_aliases_are_normalized(self):
        source = pd.DataFrame({
            "Kode Saham": ["ADRO"],
            "Tanggal Perdagangan": ["14/07/2026"],
            "Open Price": [2140],
            "Tertinggi": [2180],
            "Terendah": [2120],
            "Penutupan": [2150],
            "Volume": [10_000_000],
        })
        out = parse_independent_price_file(source, default_source="IDX_OFFICIAL_EOD_UPLOAD")
        self.assertEqual(out.loc[0, "ticker"], "ADRO.JK")
        self.assertEqual(out.loc[0, "independent_source_family"], "IDX_OFFICIAL")
        self.assertEqual(float(out.loc[0, "Close"]), 2150.0)

    def test_manual_independent_price_verifies_matching_primary_close(self):
        dates = pd.bdate_range(end="2026-07-14", periods=30)
        close = np.linspace(1_000.0, 1_100.0, len(dates))
        primary = make_ohlcv(close)
        primary.index = dates
        external = parse_independent_price_file(pd.DataFrame({
            "ticker": ["TEST"], "asof": ["2026-07-14 16:20:00"],
            "last_price": [1_100.0], "source": ["STOCKBIT_MANUAL_QUOTE"],
        }))
        validation = build_independent_price_validation(
            {"TEST.JK": primary}, external, ScanConfig(), now="2026-07-14 17:00:00",
        )
        self.assertTrue(bool(validation.loc[0, "independent_price_verified"]))
        self.assertEqual(validation.loc[0, "independent_price_state"], "VERIFIED_INDEPENDENT")
        same_family = external.copy()
        same_family["independent_source"] = "YAHOO_EXPORT"
        same_family["independent_source_family"] = "YAHOO"
        rejected = build_independent_price_validation(
            {"TEST.JK": primary}, same_family, ScanConfig(), now="2026-07-14 17:00:00",
        )
        self.assertFalse(bool(rejected.loc[0, "independent_price_verified"]))
        self.assertEqual(rejected.loc[0, "independent_price_state"], "SAME_PROVIDER_FAMILY")

    def test_missing_second_source_becomes_ready_for_price_verify(self):
        frame = ResilientEvidenceV4Tests.executable_frame()
        frame["independent_price_verified"] = False
        frame["independent_price_state"] = "MISSING_INDEPENDENT"
        for column in (
            "market_status_confidence", "news_confidence", "fundamental_confidence",
            "validation_confidence", "quote_confidence", "universe_confidence",
        ):
            frame[column] = 100.0
        out = finalize_execution_integrity(frame, ScanConfig())
        self.assertEqual(out.loc[0, "status"], "READY_FOR_PRICE_VERIFY")
        self.assertEqual(out.loc[0, "automation_decision"], "VERIFY_INDEPENDENT_PRICE")
        self.assertIn("INDEPENDENT_PRICE_REQUIRED", out.loc[0, "execution_gate_failures"])

    def test_twelve_data_history_requires_consensus_not_just_one_quote(self):
        dates = pd.bdate_range(end="2026-07-14", periods=30)
        close = np.linspace(900.0, 1_050.0, len(dates))
        primary = make_ohlcv(close)
        primary.index = dates
        external = pd.DataFrame({
            "ticker": ["TEST.JK"] * len(dates), "Date": dates,
            "Open": close, "High": close * 1.01, "Low": close * 0.99,
            "Close": close, "Volume": 3_000_000,
            "independent_source": "TWELVE_DATA_XIDX_EOD",
            "independent_source_family": "TWELVE_DATA",
        })
        validation = build_independent_price_validation(
            {"TEST.JK": primary}, external, ScanConfig(), now="2026-07-14 17:00:00",
        )
        self.assertTrue(bool(validation.loc[0, "independent_price_verified"]))
        self.assertGreaterEqual(int(validation.loc[0, "independent_overlap_bars"]), 20)
        self.assertGreater(float(validation.loc[0, "independent_return_correlation"]), 0.99)

    def test_independent_price_conflict_is_a_critical_block(self):
        frame = ResilientEvidenceV4Tests.executable_frame()
        validation = pd.DataFrame([{
            "ticker": "ADRO.JK", "independent_price_verified": False,
            "independent_price_state": "PRICE_CONFLICT",
            "independent_source": "IDX_OFFICIAL_EOD_UPLOAD",
            "independent_source_family": "IDX_OFFICIAL",
            "independent_price_confidence": 0.0,
        }])
        gated = apply_independent_price_gate(frame, validation, ScanConfig())
        self.assertEqual(gated.loc[0, "status"], "BLOCKED_CONTEXT")
        self.assertTrue(bool(gated.loc[0, "quote_critical_blocker"]))

    def test_twelve_data_adapter_never_exposes_api_key_in_report(self):
        payload = {"values": [{
            "datetime": "2026-07-14", "open": "1000", "high": "1020",
            "low": "990", "close": "1010", "volume": "1000000",
        }]}
        response = SimpleNamespace(json=lambda: payload)
        response.raise_for_status = lambda: None
        fake_requests = SimpleNamespace(get=lambda *args, **kwargs: response)
        with patch.dict(sys.modules, {"requests": fake_requests}):
            data, report = fetch_twelve_data_eod(["TEST.JK"], api_key="SECRET_KEY", max_tickers=1)
        self.assertEqual(report.loc[0, "status"], "OK")
        self.assertEqual(data.loc[0, "independent_source_family"], "TWELVE_DATA")
        self.assertNotIn("SECRET_KEY", report.to_string())
        failing_requests = SimpleNamespace(
            get=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("request apikey=SECRET_KEY failed"))
        )
        with patch.dict(sys.modules, {"requests": failing_requests}):
            _, failed_report = fetch_twelve_data_eod(["TEST.JK"], api_key="SECRET_KEY", max_tickers=1)
        self.assertNotIn("SECRET_KEY", failed_report.to_string())


class ZeroUploadAutomationV426Tests(unittest.TestCase):
    @staticmethod
    def _google_html(ticker: str = "TEST", price: float = 1100.0) -> str:
        timestamp = int(pd.Timestamp("2026-07-14 16:00:00", tz="Asia/Jakarta").timestamp())
        return (
            '<html><div data-exchange="IDX" '
            f'data-symbol="{ticker}" data-last-price="{price}" '
            f'data-last-normal-market-timestamp="{timestamp}" '
            'data-currency-code="IDR"></div></html>'
        )

    def test_google_finance_attribute_parser_is_ticker_and_timestamp_aware(self):
        parsed = scanner_module._google_finance_quote_from_html(self._google_html(), "TEST.JK")
        self.assertEqual(parsed["ticker"], "TEST.JK")
        self.assertEqual(parsed["price"], 1100.0)
        self.assertEqual(parsed["currency"], "IDR")
        self.assertEqual(parsed["timestamp"], pd.Timestamp("2026-07-14 16:00:00"))

    def test_google_finance_adapter_returns_independent_standard_schema(self):
        html = self._google_html()

        def fake_get(url, **kwargs):
            response = SimpleNamespace(text=html, url=url)
            response.raise_for_status = lambda: None
            return response

        with patch.dict(sys.modules, {"requests": SimpleNamespace(get=fake_get)}):
            data, report = fetch_google_finance_quotes(["TEST.JK"], max_tickers=1, retry_count=1)
        self.assertEqual(report.loc[0, "status"], "OK")
        self.assertEqual(data.loc[0, "independent_source_family"], "GOOGLE_FINANCE")
        self.assertEqual(float(data.loc[0, "Close"]), 1100.0)

    def test_official_idx_stock_summary_is_downloaded_without_upload(self):
        payload = {"data": [{
            "StockCode": "TEST", "Date": "2026-07-14T00:00:00",
            "OpenPrice": 1080, "High": 1120, "Low": 1070,
            "Close": 1100, "Volume": 5_000_000,
        }]}

        class FakeSession:
            def get(self, url, **kwargs):
                if "GetStockSummary" in url:
                    response = SimpleNamespace(
                        status_code=200,
                        url="https://www.idx.co.id/primary/TradingSummary/GetStockSummary",
                        json=lambda: payload,
                    )
                    response.raise_for_status = lambda: None
                    return response
                return SimpleNamespace(status_code=200, url=url)

        fake_requests = SimpleNamespace(Session=lambda: FakeSession())
        with patch.dict(sys.modules, {"requests": fake_requests}):
            data, report = fetch_idx_official_eod_quotes(
                ["TEST.JK"], reference_date="2026-07-14", lookback_days=2,
            )
        self.assertEqual(report.loc[0, "status"], "OK")
        self.assertEqual(data.loc[0, "independent_source_family"], "IDX_OFFICIAL")
        self.assertEqual(float(data.loc[0, "Close"]), 1100.0)

    def test_orchestrator_falls_back_to_google_only_for_unresolved_ticker(self):
        empty = pd.DataFrame(columns=[
            "ticker", "Date", "Open", "High", "Low", "Close", "Volume",
            "independent_source", "independent_source_family",
        ])
        official_report = pd.DataFrame([{
            "provider": "IDX_OFFICIAL_STOCK_SUMMARY", "scope": "20260714",
            "status": "FAILED", "rows": 0, "asof": pd.Timestamp("2026-07-14"),
            "error": "HTTP 403",
        }])
        google = pd.DataFrame([{
            "ticker": "TEST.JK", "Date": pd.Timestamp("2026-07-14 16:00:00"),
            "Open": np.nan, "High": np.nan, "Low": np.nan, "Close": 1100.0,
            "Volume": np.nan, "independent_source": "GOOGLE_FINANCE_PUBLIC_QUOTE",
            "independent_source_family": "GOOGLE_FINANCE",
        }])
        google_report = pd.DataFrame([{
            "provider": "GOOGLE_FINANCE", "scope": "TEST.JK", "status": "OK",
            "rows": 1, "asof": pd.Timestamp("2026-07-14 16:00:00"), "error": "",
        }])
        with (
            patch.object(scanner_module, "fetch_idx_official_eod_quotes", return_value=(empty, official_report)),
            patch.object(scanner_module, "fetch_google_finance_quotes", return_value=(google, google_report)) as google_mock,
            patch.object(scanner_module, "fetch_twelve_data_eod") as twelve_mock,
        ):
            data, report = fetch_automatic_independent_prices(
                ["TEST.JK"], reference_date="2026-07-14", twelve_data_api_key="",
            )
        google_mock.assert_called_once()
        twelve_mock.assert_not_called()
        self.assertEqual(data.loc[0, "independent_source_family"], "GOOGLE_FINANCE")
        self.assertIn("OK", report["status"].tolist())

    def test_orchestrator_rejects_conflicting_idx_row_and_continues_to_google(self):
        official = pd.DataFrame([{
            "ticker": "TEST.JK", "Date": pd.Timestamp("2026-07-14"),
            "Open": 880.0, "High": 920.0, "Low": 870.0, "Close": 900.0,
            "Volume": 1_000_000, "independent_source": "IDX_OFFICIAL_STOCK_SUMMARY",
            "independent_source_family": "IDX_OFFICIAL",
        }])
        official_report = pd.DataFrame([{
            "provider": "IDX_OFFICIAL_STOCK_SUMMARY", "scope": "20260714",
            "status": "OK", "rows": 1, "asof": pd.Timestamp("2026-07-14"),
            "error": "",
        }])
        google = pd.DataFrame([{
            "ticker": "TEST.JK", "Date": pd.Timestamp("2026-07-14 16:00:00"),
            "Open": np.nan, "High": np.nan, "Low": np.nan, "Close": 1100.0,
            "Volume": np.nan, "independent_source": "GOOGLE_FINANCE_PUBLIC_QUOTE",
            "independent_source_family": "GOOGLE_FINANCE",
        }])
        google_report = pd.DataFrame([{
            "provider": "GOOGLE_FINANCE", "scope": "TEST.JK", "status": "OK",
            "rows": 1, "asof": pd.Timestamp("2026-07-14 16:00:00"), "error": "",
        }])
        with (
            patch.object(scanner_module, "fetch_idx_official_eod_quotes", return_value=(official, official_report)),
            patch.object(scanner_module, "fetch_google_finance_quotes", return_value=(google, google_report)) as google_mock,
            patch.object(scanner_module, "fetch_twelve_data_eod") as twelve_mock,
        ):
            data, _ = fetch_automatic_independent_prices(
                ["TEST.JK"],
                reference_date="2026-07-14",
                primary_reference={"TEST.JK": (pd.Timestamp("2026-07-14"), 1100.0)},
                twelve_data_api_key="",
            )
        google_mock.assert_called_once()
        twelve_mock.assert_not_called()
        self.assertEqual(
            set(data["independent_source_family"]),
            {"IDX_OFFICIAL", "GOOGLE_FINANCE"},
        )

    def test_automatic_google_quote_can_verify_primary_price(self):
        dates = pd.bdate_range(end="2026-07-14", periods=30)
        primary = make_ohlcv(np.linspace(1_000.0, 1_100.0, len(dates)))
        primary.index = dates
        external = pd.DataFrame([{
            "ticker": "TEST.JK", "Date": pd.Timestamp("2026-07-14 16:00:00"),
            "Open": np.nan, "High": np.nan, "Low": np.nan, "Close": 1100.0,
            "Volume": np.nan, "independent_source": "GOOGLE_FINANCE_PUBLIC_QUOTE",
            "independent_source_family": "GOOGLE_FINANCE",
        }])
        validation = build_independent_price_validation(
            {"TEST.JK": primary}, external, ScanConfig(), now="2026-07-14 17:00:00",
        )
        self.assertTrue(bool(validation.loc[0, "independent_price_verified"]))
        self.assertEqual(validation.loc[0, "independent_price_state"], "VERIFIED_INDEPENDENT")

    def test_streamlit_ui_keeps_core_and_optional_microstructure_uploaders(self):
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertEqual(app_source.count("st.file_uploader("), 4)
        self.assertNotIn("independent_price_upload", app_source)
        self.assertIn("broksum_upload", app_source)
        self.assertIn("orderbook_upload", app_source)


    def test_release_manifest_contains_deployment_and_integration_files(self):
        required = {
            "requirements.txt", "runtime.txt", "README.md",
            "IDX_Scanner_Confirmation_v1.pine", "STOCKBIT_SCREENER_PRESETS.md",
            "AUDIT_REPORT_V4_2_6.md",
        }
        self.assertTrue(required.issubset({path.name for path in ROOT.iterdir()}))

    def test_effective_release_version_and_price_verification_capacity(self):
        self.assertEqual(scanner_module.__version__, "5.0.0-modular-clean")
        self.assertGreaterEqual(ScanConfig().max_automatic_price_candidates, 40)

    def test_streamlit_contains_tradingview_bridge_tab(self):
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("TradingView / Stockbit", app_source)
        self.assertIn("build_tradingview_bridge", app_source)
        self.assertIn("tradingview_scanner_bridge.csv", app_source)


class ARAIntelligenceV460Tests(unittest.TestCase):
    def test_orderbook_parser_calculates_bid_offer_imbalance(self):
        source = pd.DataFrame([
            {"ticker": "ANTM", "timestamp": "2026-07-15 15:55", "level": 1,
             "bid_price": 3200, "bid_lots": 300000, "offer_price": 3210, "offer_lots": 50000},
            {"ticker": "ANTM", "timestamp": "2026-07-15 15:55", "level": 2,
             "bid_price": 3190, "bid_lots": 200000, "offer_price": 3220, "offer_lots": 50000},
        ])
        parsed = parse_orderbook_snapshot_csv(source)
        self.assertEqual(parsed.loc[0, "ticker"], "ANTM.JK")
        self.assertEqual(parsed.loc[0, "orderbook_signal"], "STRONG_BUY_QUEUE")
        self.assertGreater(float(parsed.loc[0, "orderbook_imbalance"]), 0.60)

    def test_pre_ara_and_continuation_are_distinct_models(self):
        pre = SpecialtyScannerV42Tests._strong_daily_frame().copy()
        pre_result = scan_ara_hunter_candidates({"PRE.JK": pre}, now="2026-07-13 20:00:00")
        self.assertFalse(pre_result.empty)
        self.assertEqual(pre_result.loc[0, "ara_model"], "PRE_ARA")
        self.assertIn(pre_result.loc[0, "ara_hunter_status"], {
            "PRE_ARA_READY", "PRE_ARA_CANDIDATE", "PRE_ARA_WATCHLIST"
        })

        cont = SpecialtyScannerV42Tests._strong_daily_frame().copy()
        idx = cont.index[-1]
        prev_close = float(cont.iloc[-2]["Close"])
        ara_price = scanner_module.idx_daily_price_band(prev_close)[1]
        cont.at[idx, "Open"] = prev_close * 1.05
        cont.at[idx, "High"] = ara_price
        cont.at[idx, "Low"] = prev_close * 1.03
        cont.at[idx, "Close"] = ara_price
        cont.at[idx, "VALUE"] = 20_000_000_000.0
        cont.at[idx, "ADTV20"] = 8_000_000_000.0
        cont.at[idx, "VOL_RATIO"] = 3.0
        cont.at[idx, "CLOSE_LOCATION"] = 1.0
        cont.at[idx, "BODY_ATR"] = 1.2
        cont.at[idx, "CMF20"] = 0.15
        cont.at[idx, "OBV_SLOPE10"] = 1.0
        cont_result = scan_ara_hunter_candidates({"CONT.JK": cont}, now="2026-07-13 20:00:00")
        self.assertFalse(cont_result.empty)
        self.assertEqual(cont_result.loc[0, "ara_model"], "ARA_CONTINUATION")
        self.assertIn(cont_result.loc[0, "ara_hunter_status"], {
            "ARA_CONTINUATION_READY", "ARA_CONTINUATION_CANDIDATE", "ARA_CONFIRMED_ONLY"
        })

    def test_optional_observed_flow_can_verify_continuation(self):
        ara = pd.DataFrame([{
            "ticker": "ANTM.JK", "ara_hunter_status": "ARA_CONTINUATION_READY",
            "ara_model_score": 80.0, "ara_hunter_score": 80.0,
            "order_instruction": "PLAN_NEXT_SESSION_LIMIT_NO_MARKET_CHASE",
        }])
        broksum = pd.DataFrame([{
            "ticker": "ANTM.JK", "broksum_signal": "ACCUMULATION_PROXY",
            "broksum_net_ratio": 0.20, "broksum_asof": pd.Timestamp("2026-07-15"),
        }])
        orderbook = pd.DataFrame([{
            "ticker": "ANTM.JK", "orderbook_signal": "STRONG_BUY_QUEUE",
            "orderbook_asof": pd.Timestamp("2026-07-15 15:55"),
            "orderbook_imbalance": 0.70,
        }])
        out = apply_ara_external_confirmation(
            ara, broksum, orderbook, now="2026-07-16 05:00:00"
        )
        self.assertEqual(out.loc[0, "ara_hunter_status"], "ARA_CONTINUATION_VERIFIED_ORDERFLOW")
        self.assertGreater(float(out.loc[0, "ara_final_score"]), 80.0)

    def test_automatic_output_explicitly_labels_proxies(self):
        frame = SpecialtyScannerV42Tests._strong_daily_frame().copy()
        result = scan_ara_hunter_candidates(
            {"TEST.JK": frame}, {"TEST.JK": SpecialtyScannerV42Tests._intraday("bsjp")},
            now="2026-07-13 20:00:00",
        )
        self.assertFalse(result.empty)
        self.assertIn("orderflow_proxy_score", result.columns)
        self.assertIn("queue_proxy_score", result.columns)
        self.assertIn("Orderflow proxy bukan broker summary", result.loc[0, "warnings"])


if __name__ == "__main__":
    unittest.main()


class FreeMultiSourceV431Tests(unittest.TestCase):
    def test_current_cache_skips_yahoo_download(self):
        cached = make_ohlcv(np.linspace(900.0, 1_100.0, 260))
        fake_yfinance = SimpleNamespace(download=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Yahoo should not run")))
        with (
            patch.dict(sys.modules, {"yfinance": fake_yfinance}),
            patch.object(scanner_module, "_load_daily_ohlcv_cache", return_value=cached),
            patch.object(scanner_module, "_daily_cache_is_current", return_value=True),
        ):
            histories, report = scanner_module.download_ohlcv(["TEST.JK"], period="2y")
        self.assertIn("TEST.JK", histories)
        self.assertEqual(report.source_tiers["TEST.JK"], "CACHE_FRESH_VERIFIED_UNKNOWN")

    def test_idx_official_patches_stale_history_after_yahoo_failure(self):
        cached = make_ohlcv(np.linspace(900.0, 1_100.0, 260))
        cached.index = pd.bdate_range(end="2026-07-13", periods=len(cached))
        official = pd.DataFrame({
            "ticker": ["TEST.JK"],
            "Date": [pd.Timestamp("2026-07-14")],
            "Open": [1_105.0], "High": [1_120.0], "Low": [1_100.0],
            "Close": [1_115.0], "Volume": [4_000_000.0],
            "independent_source": ["IDX_OFFICIAL_STOCK_SUMMARY_API"],
            "independent_source_family": ["IDX_OFFICIAL"],
        })
        fake_yfinance = SimpleNamespace(download=lambda *args, **kwargs: pd.DataFrame())
        with (
            patch.dict(sys.modules, {"yfinance": fake_yfinance}),
            patch.object(scanner_module, "_load_daily_ohlcv_cache", return_value=cached),
            patch.object(scanner_module, "_daily_cache_is_current", return_value=False),
            patch.object(scanner_module, "fetch_idx_official_eod_quotes", return_value=(official, pd.DataFrame())),
            patch.object(scanner_module, "_write_daily_ohlcv_cache"),
        ):
            histories, report = scanner_module.download_ohlcv(["TEST.JK"], period="2y")
        self.assertEqual(report.source_tiers["TEST.JK"], "LIVE_IDX_EOD_PATCH")
        self.assertEqual(pd.Timestamp(histories["TEST.JK"].index[-1]), pd.Timestamp("2026-07-14"))
        self.assertAlmostEqual(float(histories["TEST.JK"]["Close"].iloc[-1]), 1_115.0)

    def test_itick_daily_payload_is_normalized_to_idx_ohlcv(self):
        timestamps = [
            int(pd.Timestamp("2026-07-13 00:00:00", tz="UTC").timestamp() * 1000),
            int(pd.Timestamp("2026-07-14 00:00:00", tz="UTC").timestamp() * 1000),
        ]
        payload = {
            "code": 0,
            "msg": None,
            "data": [
                {"t": timestamps[1], "o": 1110, "h": 1130, "l": 1100, "c": 1120, "v": 5000000},
                {"t": timestamps[0], "o": 1090, "h": 1120, "l": 1080, "c": 1100, "v": 4500000},
            ],
        }
        response = SimpleNamespace(json=lambda: payload)
        response.raise_for_status = lambda: None
        fake_requests = SimpleNamespace(get=lambda *args, **kwargs: response)
        with (
            patch.dict(sys.modules, {"requests": fake_requests}),
            patch.object(scanner_module, "_reserve_itick_free_call", return_value=True),
        ):
            histories, report = scanner_module.fetch_itick_ohlcv(
                ["TEST.JK"], api_token="secret", period="2y", interval="1d"
            )
        self.assertIn("TEST.JK", histories)
        self.assertEqual(len(histories["TEST.JK"]), 2)
        self.assertTrue(histories["TEST.JK"].index.is_monotonic_increasing)
        self.assertEqual(report.loc[0, "status"], "OK")
        self.assertAlmostEqual(float(histories["TEST.JK"]["Close"].iloc[-1]), 1120.0)

    def test_itick_rate_budget_failure_is_nonfatal(self):
        with patch.object(scanner_module, "_reserve_itick_free_call", return_value=False):
            histories, report = scanner_module.fetch_itick_ohlcv(
                ["TEST.JK"], api_token="secret", period="2y", interval="1d"
            )
        self.assertEqual(histories, {})
        self.assertEqual(report.loc[0, "status"], "RATE_BUDGET_EXHAUSTED")


class FreeMultiSourceIndependenceV431Tests(unittest.TestCase):
    def test_idx_patch_cannot_verify_itself_as_independent_price(self):
        dates = pd.bdate_range(end="2026-07-14", periods=30)
        primary = make_ohlcv(np.linspace(1_000.0, 1_100.0, len(dates)))
        primary.index = dates
        external = pd.DataFrame({
            "ticker": ["TEST.JK"],
            "Date": [pd.Timestamp("2026-07-14")],
            "Open": [1_095.0], "High": [1_110.0], "Low": [1_090.0],
            "Close": [1_100.0], "Volume": [5_000_000.0],
            "independent_source": ["IDX_OFFICIAL_STOCK_SUMMARY_API"],
            "independent_source_family": ["IDX_OFFICIAL"],
        })
        validation = build_independent_price_validation(
            {"TEST.JK": primary},
            external,
            ScanConfig(),
            now="2026-07-14 17:00:00",
            primary_source_tiers={"TEST.JK": "LIVE_IDX_EOD_PATCH"},
        )
        self.assertFalse(bool(validation.loc[0, "independent_price_verified"]))
        self.assertEqual(validation.loc[0, "independent_price_state"], "SAME_PROVIDER_FAMILY")


class EodIntegrityV432Tests(unittest.TestCase):
    def test_completed_date_uses_single_1620_cutoff(self):
        before = scanner_module._expected_last_completed_daily_date(pd.Timestamp("2026-07-15 16:19", tz="Asia/Jakarta"))
        after = scanner_module._expected_last_completed_daily_date(pd.Timestamp("2026-07-15 16:20", tz="Asia/Jakarta"))
        self.assertEqual(before, pd.Timestamp("2026-07-14"))
        self.assertEqual(after, pd.Timestamp("2026-07-15"))

    def test_completed_daily_frame_drops_same_day_partial_before_cutoff(self):
        frame = make_ohlcv(np.linspace(900.0, 1_100.0, 5))
        frame.index = pd.to_datetime(["2026-07-09", "2026-07-10", "2026-07-13", "2026-07-14", "2026-07-15"])
        out = scanner_module._completed_daily_frame(frame, pd.Timestamp("2026-07-15 14:00", tz="Asia/Jakarta"))
        self.assertEqual(pd.Timestamp(out.index[-1]), pd.Timestamp("2026-07-14"))
        self.assertEqual(out.attrs.get("bar_state"), "FINAL_EOD")

    def test_legacy_same_day_cache_is_not_trusted_before_close(self):
        frame = make_ohlcv(np.linspace(900.0, 1_100.0, 5))
        frame.index = pd.to_datetime(["2026-07-09", "2026-07-10", "2026-07-13", "2026-07-14", "2026-07-15"])
        frame.attrs["written_at"] = "2026-07-15T14:00:00+07:00"
        self.assertFalse(scanner_module._daily_cache_is_current(frame, pd.Timestamp("2026-07-15 14:05", tz="Asia/Jakarta")))

    def test_regular_session_becomes_pending_close_only_at_final_integrity(self):
        frame = ResilientEvidenceV4Tests.executable_frame()
        frame["pending_close"] = True
        out = finalize_execution_integrity(frame, ScanConfig())
        self.assertEqual(out.loc[0, "status"], "PENDING_CLOSE")
        self.assertEqual(out.loc[0, "order_instruction"], "DO_NOT_BUY")
        self.assertEqual(out.loc[0, "primary_execution_blocker"], "DAILY_BAR_NOT_FINAL")


    def test_premarket_0543_does_not_wait_for_close(self):
        waits = scanner_module.idx_core_waits_for_eod(
            now=pd.Timestamp("2026-07-16 05:43", tz="Asia/Jakarta"),
            market_state="PRE",
            quote_time=pd.Timestamp("2026-07-15 16:00"),
            current_bar_incomplete=False,
        )
        self.assertFalse(waits)

    def test_snapshot_gate_preserves_premarket_execution(self):
        frame = signal_frame()
        frame["last_price"] = 1_000.0
        frame["atr_pct"] = 0.03
        frame["absolute_data_age_days"] = 1
        frame["current_bar_incomplete"] = 0
        snap = pd.DataFrame({
            "ticker": ["ADRO.JK"],
            "quote_time": [pd.Timestamp("2026-07-15 16:00")],
            "quote_last_price": [1_000.0],
            "quote_volume": [1_000_000],
            "quote_market_state": ["PRE"],
            "quote_verified": [True],
        })
        with patch.object(
            scanner_module,
            "_jakarta_timestamp",
            return_value=pd.Timestamp("2026-07-16 05:43", tz="Asia/Jakarta"),
        ):
            out = scanner_module.apply_execution_snapshot_gate(frame, snap, ScanConfig())
        self.assertFalse(bool(out.loc[0, "pending_close"]))
        self.assertEqual(out.loc[0, "status"], "EXECUTION_READY")
        self.assertEqual(out.loc[0, "execution_session_phase"], "PREMARKET_PREVIOUS_EOD")

    def test_regular_session_still_waits_for_close(self):
        waits = scanner_module.idx_core_waits_for_eod(
            now=pd.Timestamp("2026-07-16 10:00", tz="Asia/Jakarta"),
            market_state="REGULAR",
            quote_time=pd.Timestamp("2026-07-16 10:00"),
            current_bar_incomplete=True,
        )
        self.assertTrue(waits)

    def test_lunch_break_with_same_day_quote_still_waits(self):
        waits = scanner_module.idx_core_waits_for_eod(
            now=pd.Timestamp("2026-07-16 12:30", tz="Asia/Jakarta"),
            market_state="CLOSED",
            quote_time=pd.Timestamp("2026-07-16 11:30"),
            current_bar_incomplete=False,
        )
        self.assertTrue(waits)

    def test_holiday_closed_with_prior_day_quote_does_not_wait(self):
        waits = scanner_module.idx_core_waits_for_eod(
            now=pd.Timestamp("2026-07-16 10:00", tz="Asia/Jakarta"),
            market_state="CLOSED",
            quote_time=pd.Timestamp("2026-07-15 16:00"),
            current_bar_incomplete=False,
        )
        self.assertFalse(waits)

    def test_weekend_uses_latest_completed_eod(self):
        waits = scanner_module.idx_core_waits_for_eod(
            now=pd.Timestamp("2026-07-18 10:00", tz="Asia/Jakarta"),
            market_state="CLOSED",
            quote_time=pd.Timestamp("2026-07-17 16:00"),
            current_bar_incomplete=False,
        )
        self.assertFalse(waits)

    def test_pullback_does_not_receive_universal_silent_accumulation_blocker(self):
        close = np.linspace(1_000, 1_250, 260)
        prepared = prepare_indicators(make_ohlcv(close, np.full(260, 2_000_000.0)))
        engine = ScanEngine(ScanConfig())
        blockers, metrics = engine._tradeability(prepared, pd.Timestamp(prepared.index[-1]))
        self.assertIn("silent_accumulation_score", metrics)
        self.assertFalse(any("Silent-accumulation" in item for item in blockers))

    def test_ready_distance_is_setup_specific(self):
        cfg = ScanConfig()
        self.assertEqual(scanner_module._ready_distance_atr_for_setup("PULLBACK_CONTINUATION", cfg), 0.45)
        self.assertEqual(scanner_module._ready_distance_atr_for_setup("BREAKOUT_RETEST", cfg), 0.45)
        self.assertEqual(scanner_module._ready_distance_atr_for_setup("UNICORN_SNIPER_ICT", cfg), 0.30)


class AnalystFusionV440Tests(unittest.TestCase):
    @staticmethod
    def pullback_limit_candidate() -> pd.DataFrame:
        frame = pd.DataFrame([{
            "ticker": "TEST.JK",
            "status": "WATCHLIST_ENTRY",
            "status_rank": 5,
            "setup": "PULLBACK_CONTINUATION",
            "detected": True,
            "invalidated": False,
            "action": "WAIT_PULLBACK_CONFIRMATION",
            "evidence": "EMA20 > EMA50 > EMA200 • Momentum 3 bulan positif • Pullback menyentuh value area",
            "quality_score": 85.0,
            "blockers": "Retest/reclaim/entry trigger belum lengkap",
            "critical_blockers": "",
            "last_price": 1_020.0,
            "entry_low": 990.0,
            "entry_high": 1_010.0,
            "entry": 1_000.0,
            "stop_loss": 950.0,
            "tp1": 1_100.0,
            "tp2": 1_150.0,
            "rr1": 2.0,
            "rr2": 3.0,
            "stop_pct": 0.05,
            "distance_atr": 0.30,
            "silent_accumulation_score": 60.0,
            "sizing_status": "OK",
            "suggested_lots": 2,
            "capital_required_idr": 200_000.0,
            "max_loss_idr": 10_000.0,
            "market_regime": "NEUTRAL",
            "market_status_confidence": 80.0,
            "news_confidence": 52.0,
            "fundamental_confidence": 50.0,
            "fundamental_coverage": 0.0,
            "validation_confidence": 45.0,
            "quote_confidence": 68.0,
            "universe_confidence": 100.0,
            "independent_price_verified": False,
            "independent_price_state": "MISSING_INDEPENDENT",
            "ohlcv_source_tier": "LIVE_YAHOO",
            "absolute_data_age_days": 1,
            "current_bar_incomplete": False,
            "pending_close": False,
            "adtv20_idr": 8_000_000_000.0,
        }])
        return frame

    def test_pullback_limit_can_become_analyst_execution_ready(self):
        cfg = ScanConfig().replace(account_size_idr=5_000_000, cash_on_hand_idr=5_000_000)
        frame = apply_analyst_fusion_gate(self.pullback_limit_candidate(), cfg)
        self.assertTrue(bool(frame.loc[0, "analyst_pre_budget_ready"]))
        self.assertEqual(frame.loc[0, "analyst_order_mode"], "LIMIT_PULLBACK_ZONE")
        frame = enforce_analyst_portfolio_budget(frame, cfg, cash_on_hand_idr=5_000_000)
        out = finalize_execution_integrity(frame, cfg)
        self.assertEqual(out.loc[0, "status"], "EXECUTION_READY")
        self.assertEqual(out.loc[0, "execution_mode"], "SIGNAL_FIRST")
        self.assertTrue(bool(out.loc[0, "requires_stockbit_price_check"]))
        self.assertEqual(out.loc[0, "order_instruction"], "BUY_LIMIT_USER_SIZE")
        self.assertFalse(bool(out.loc[0, "account_risk_gate_applied"]))

    def test_price_conflict_still_blocks_analyst_candidate(self):
        cfg = ScanConfig()
        frame = self.pullback_limit_candidate()
        frame["quote_critical_blocker"] = True
        frame = apply_analyst_fusion_gate(frame, cfg)
        self.assertFalse(bool(frame.loc[0, "analyst_pre_budget_ready"]))
        self.assertIn("Konflik quote/candle", frame.loc[0, "analyst_hard_blockers"])

    def test_budget_constraint_does_not_suppress_signal_first_candidate(self):
        cfg = ScanConfig().replace(max_positions=1, cash_on_hand_idr=0)
        frame = apply_analyst_fusion_gate(self.pullback_limit_candidate(), cfg)
        frame = enforce_analyst_portfolio_budget(
            frame, cfg, current_positions=10, cash_on_hand_idr=0
        )
        out = finalize_execution_integrity(frame, cfg)
        self.assertEqual(out.loc[0, "status"], "EXECUTION_READY")
        self.assertEqual(out.loc[0, "primary_execution_blocker"], "NONE")
        self.assertFalse(bool(out.loc[0, "account_risk_gate_applied"]))

    def test_non_actionable_watchlist_is_not_blanket_pending_close(self):
        cfg = ScanConfig()
        frame = self.pullback_limit_candidate()
        frame["quality_score"] = 70.0
        frame["silent_accumulation_score"] = 30.0
        frame["pending_close"] = True
        frame = apply_analyst_fusion_gate(frame, cfg)
        out = finalize_execution_integrity(frame, cfg)
        self.assertEqual(out.loc[0, "status"], "PENDING_CLOSE")

class SignalFirstV450RegressionTests(unittest.TestCase):
    def test_completed_intraday_session_is_usable_for_ara_after_close(self):
        intraday = SpecialtyScannerV42Tests._intraday("bsjp")
        metrics = scanner_module._intraday_metrics(
            intraday, now="2026-07-13 20:00:00"
        )
        self.assertFalse(bool(metrics["intraday_fresh"]))
        self.assertTrue(bool(metrics["intraday_session_complete"]))
        self.assertEqual(metrics["intraday_data_state"], "FINAL_SESSION")

    def test_confirmed_ara_is_visible_instead_of_rejected(self):
        frame = SpecialtyScannerV42Tests._strong_daily_frame().copy()
        idx = frame.index[-1]
        prev_close = float(frame.iloc[-2]["Close"])
        ara_price = scanner_module.idx_daily_price_band(prev_close)[1]
        frame.at[idx, "Open"] = prev_close * 1.05
        frame.at[idx, "High"] = ara_price
        frame.at[idx, "Low"] = prev_close * 1.03
        frame.at[idx, "Close"] = ara_price
        frame.at[idx, "VALUE"] = 20_000_000_000.0
        frame.at[idx, "ADTV20"] = 8_000_000_000.0
        frame.at[idx, "VOL_RATIO"] = 3.0
        frame.at[idx, "CLOSE_LOCATION"] = 1.0
        frame.at[idx, "BODY_ATR"] = 1.2
        frame.at[idx, "CMF20"] = 0.15
        frame.at[idx, "OBV_SLOPE10"] = 1.0
        result = scan_ara_hunter_candidates({"TEST.JK": frame}, now="2026-07-13 20:00:00")
        self.assertFalse(result.empty)
        self.assertIn(result.loc[0, "ara_hunter_status"], {"ARA_CONTINUATION_READY", "ARA_CONTINUATION_CANDIDATE", "ARA_CONFIRMED_ONLY"})
        self.assertTrue(bool(result.loc[0, "signal_valid"]))

    def test_momentum_name_receives_intraday_shortlist_capacity(self):
        prepared = {}
        for i in range(10):
            frame = SpecialtyScannerV42Tests._strong_daily_frame().copy()
            idx = frame.index[-1]
            frame.at[idx, "ADTV20"] = 20_000_000_000.0
            frame.at[idx, "VOL_RATIO"] = 0.8
            frame.at[idx, "CLOSE_LOCATION"] = 0.45
            frame.at[idx, "BODY_ATR"] = 0.2
            frame.at[idx, "HIGH20_PREV"] = frame.at[idx, "Close"] + 100
            prepared[f"L{i}.JK"] = frame
        fast = SpecialtyScannerV42Tests._strong_daily_frame().copy()
        idx = fast.index[-1]
        fast.at[idx, "ADTV20"] = 600_000_000.0
        fast.at[idx, "VOL_RATIO"] = 4.0
        fast.at[idx, "CLOSE_LOCATION"] = 0.98
        fast.at[idx, "BODY_ATR"] = 1.4
        fast.at[idx, "CMF20"] = 0.2
        fast.at[idx, "HIGH20_PREV"] = fast.at[idx, "Close"] - 10
        prepared["FAST.JK"] = fast
        shortlist = specialty_intraday_shortlist(prepared, max_candidates=6)
        self.assertIn("FAST.JK", shortlist)
        self.assertEqual(len(shortlist), 6)


class SignalCalibrationV470Tests(unittest.TestCase):
    def test_missing_independent_price_does_not_block_bpjs(self):
        frame = SpecialtyScannerV42Tests._strong_daily_frame()
        core = pd.DataFrame([{
            "ticker": "TEST.JK", "critical_blockers": "",
            "independent_price_required": True,
            "independent_price_verified": False,
            "ohlcv_source_tier": "LIVE_YAHOO",
            "market_status_critical_blocker": False,
            "news_critical_blocker": False,
            "quote_critical_blocker": False,
        }])
        out = scan_bpjs_candidates(
            {"TEST.JK": frame}, {"TEST.JK": SpecialtyScannerV42Tests._intraday("bpjs")},
            core_signals=core, now="2026-07-13 09:45:00",
        )
        self.assertEqual(out.loc[0, "bpjs_status"], "BPJS_READY")
        self.assertEqual(out.loc[0, "blockers"], "")

    def test_missing_independent_price_does_not_block_bsjp(self):
        frame = SpecialtyScannerV42Tests._strong_daily_frame()
        core = pd.DataFrame([{
            "ticker": "TEST.JK", "critical_blockers": "",
            "independent_price_required": True,
            "independent_price_verified": False,
            "ohlcv_source_tier": "LIVE_YAHOO",
            "market_status_critical_blocker": False,
            "news_critical_blocker": False,
            "quote_critical_blocker": False,
        }])
        out = scan_bsjp_candidates(
            {"TEST.JK": frame}, {"TEST.JK": SpecialtyScannerV42Tests._intraday("bsjp")},
            core_signals=core, now="2026-07-13 15:30:00",
        )
        self.assertEqual(out.loc[0, "bsjp_status"], "BSJP_READY")
        self.assertEqual(out.loc[0, "blockers"], "")

    @staticmethod
    def _core_row(setup: str, action: str, evidence: str, quality: float = 76.0, distance: float = 0.6):
        return pd.DataFrame([{
            "ticker": "TEST.JK", "status": "WATCHLIST_ENTRY", "status_rank": 4,
            "setup": setup, "detected": True, "invalidated": False,
            "action": action, "evidence": evidence, "quality_score": quality,
            "blockers": "Retest/reclaim/entry trigger belum lengkap",
            "last_price": 1000.0, "entry_low": 970.0, "entry_high": 1010.0,
            "entry": 990.0, "stop_loss": 930.0, "tp1": 1100.0, "tp2": 1180.0,
            "rr1": 1.83, "rr2": 3.17, "stop_pct": 0.061,
            "distance_atr": distance, "silent_accumulation_score": 55.0,
            "market_regime": "RISK_OFF", "ohlcv_source_tier": "LIVE_YAHOO",
            "absolute_data_age_days": 1, "pending_close": False,
            "market_status_critical_blocker": False, "quote_critical_blocker": False,
            "independent_price_verified": False, "adtv20_idr": 700_000_000.0,
        }])

    def test_reversal_wait_higher_low_maps_to_limit_ready(self):
        frame = self._core_row(
            "REVERSAL_ACCUMULATION", "WAIT_HIGHER_LOW_AND_FLOW",
            "Sell-side liquidity sweep • CHOCH/BOS bullish terkonfirmasi",
            quality=75.0, distance=0.7,
        )
        out = apply_analyst_fusion_gate(frame)
        self.assertTrue(bool(out.loc[0, "analyst_pre_budget_ready"]))
        self.assertEqual(out.loc[0, "analyst_order_mode"], "LIMIT_CHOCH_RETEST")
        final = finalize_execution_integrity(out)
        self.assertEqual(final.loc[0, "status"], "EXECUTION_READY")

    def test_unicorn_wait_strict_confluence_maps_to_limit_ready(self):
        frame = self._core_row(
            "UNICORN_SNIPER_ICT", "WAIT_STRICT_UNICORN_CONFLUENCE",
            "Sell-side liquidity sweep • Bullish BOS dengan displacement • Bullish FVG valid",
            quality=72.0, distance=0.8,
        )
        out = apply_analyst_fusion_gate(frame)
        self.assertTrue(bool(out.loc[0, "analyst_pre_budget_ready"]))
        self.assertEqual(out.loc[0, "analyst_order_mode"], "LIMIT_FVG_RETRACE")

    def test_wait_choch_remains_watchlist(self):
        frame = self._core_row(
            "REVERSAL_ACCUMULATION", "WAIT_CHOCH",
            "Sell-side liquidity sweep", quality=80.0, distance=0.3,
        )
        out = apply_analyst_fusion_gate(frame)
        self.assertFalse(bool(out.loc[0, "analyst_pre_budget_ready"]))
        self.assertIn("Struktur entry belum valid", out.loc[0, "analyst_hard_blockers"])
