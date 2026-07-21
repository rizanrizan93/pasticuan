from __future__ import annotations

import sys
import ast
import io
import os
import subprocess
import tempfile
import unittest
import zipfile
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
    allocate_multibagger_capital,
    scan_ara_hunter_candidates,
    build_specialty_screens,
    build_daily_opportunity_board,
    build_profit_order_builder,
    build_independent_price_validation,
    fetch_automatic_independent_prices,
    fetch_google_finance_quotes,
    fetch_idx_official_eod_quotes,
    fetch_twelve_data_eod,
    parse_fundamental_history_csv,
    parse_project_management_csv,
    collect_automatic_forward_quality,
    merge_project_management_reviews,
    combine_fundamental_history,
    build_fundamental_history_features,
    enrich_fundamentals_with_history,
    fetch_twelve_data_fundamental_history,
    parse_idx_xbrl_attachment,
    build_source_quorum_audit,
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
        self.assertGreater(plan.tp1, plan.entry)
        self.assertGreater(plan.tp2, plan.tp1)
        self.assertEqual(plan.target_model, 'PRICE_STRUCTURE_ONLY')
        self.assertNotIn('FALLBACK', plan.tp1_basis)
        self.assertNotIn('FALLBACK', plan.tp2_basis)

    def test_structure_target_engine_uses_measured_move_at_new_high(self):
        close = np.linspace(100, 200, 260)
        frame = prepare_indicators(make_ohlcv(close))
        result = scanner_module._price_structure_target_pair(
            frame, 200.0, setup='TEST', explicit_levels=[],
            projection_origin=200.0, projection_height=40.0,
        )
        self.assertTrue(result['target_structure_valid'])
        self.assertGreater(result['tp2'], result['tp1'])
        self.assertIn('MEASURED_MOVE', result['tp1_basis'])
        self.assertNotIn('FALLBACK', result['tp1_basis'])

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
        self.assertIn(plan.action, {"READY_TRIGGER", "WAIT_RETEST", "WAIT_CURRENT_RETEST_CONFIRMATION"})
        self.assertIsNotNone(plan.structural_quality_score)
        self.assertIsNotNone(plan.confirmation_quality_score)
        self.assertIsNotNone(plan.failure_risk_score)

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
        self.assertIsNotNone(plan.structural_quality_score)
        self.assertIsNotNone(plan.supply_demand_score)
        self.assertIsNotNone(plan.confirmation_quality_score)

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

    def test_official_history_ytd_is_converted_to_standalone_quarters(self):
        source = pd.DataFrame({
            "ticker": ["TEST", "TEST"],
            "period_end": ["2026-03-31", "2026-06-30"],
            "period_type": ["Q1", "Q2"],
            "statement_basis": ["YTD_CUMULATIVE", "YTD_CUMULATIVE"],
            "source_url": ["https://www.idx.co.id/report/1", "https://www.idx.co.id/report/2"],
            "revenue": [100.0, 240.0],
            "net_income": [10.0, 27.0],
            "operating_cash_flow": [12.0, 31.0],
        })
        out = parse_fundamental_history_csv(source)
        self.assertEqual(out.loc[1, "source_family"], "IDX_OFFICIAL_REFERENCE")
        self.assertEqual(float(out.loc[1, "revenue"]), 140.0)
        self.assertEqual(float(out.loc[1, "net_income"]), 17.0)
        self.assertEqual(out.loc[1, "statement_basis"], "STANDALONE_QUARTER_FROM_YTD")

    def test_spoofed_idx_domain_is_not_trusted(self):
        source = pd.DataFrame({
            "ticker": ["TEST"], "period_end": ["2026-03-31"],
            "source_url": ["https://idx.co.id.evil.example/report"],
            "revenue": [100.0],
        })
        out = parse_fundamental_history_csv(source)
        self.assertEqual(out.loc[0, "source_family"], "USER_UPLOAD")

    def test_ytd_without_predecessor_is_not_treated_as_one_quarter(self):
        source = pd.DataFrame({
            "ticker": ["TEST"], "period_end": ["2026-06-30"],
            "period_type": ["Q2"], "statement_basis": ["YTD_CUMULATIVE"],
            "source_url": ["https://www.idx.co.id/report/q2"], "revenue": [240.0],
        })
        out = parse_fundamental_history_csv(source)
        self.assertTrue(pd.isna(out.loc[0, "revenue"]))
        self.assertIn("YTD_PREDECESSOR_MISSING", out.loc[0, "validation_flags"])

    def test_matching_two_source_history_can_reach_grade_a(self):
        dates = pd.date_range("2024-03-31", periods=8, freq="QE")
        rows = []
        for family, scale in (("YAHOO", 1.0), ("TWELVE_DATA", 1.01)):
            for i, date in enumerate(dates):
                revenue = 100.0 + 12.0 * i
                net_income = 10.0 + 2.0 * i
                rows.append({
                    "ticker": "TEST.JK", "period_end": date,
                    "period_type": "Q", "statement_basis": "STANDALONE_QUARTER",
                    "source_family": family, "source_name": family, "currency": "IDR",
                    "revenue": revenue * scale, "net_income": net_income * scale,
                    "operating_cash_flow": (net_income + 3.0) * scale,
                    "capex": 2.0 * scale, "total_assets": (500 + 20 * i) * scale,
                    "total_liabilities": (180 + 5 * i) * scale,
                    "equity": (320 + 15 * i) * scale,
                    "total_debt": 80.0 * scale, "cash": (50 + 3 * i) * scale,
                    "shares_outstanding": 100.0, "operating_income": (net_income + 5) * scale,
                    "ebit": (net_income + 5) * scale, "ebitda": (net_income + 8) * scale,
                    "interest_expense": 2.0 * scale,
                })
        history = combine_fundamental_history(pd.DataFrame(rows))
        features = build_fundamental_history_features(history, now="2026-07-01")
        self.assertEqual(features.loc[0, "fundamental_data_grade"], "A")
        self.assertGreaterEqual(float(features.loc[0, "fundamental_consensus_score"]), 75.0)
        base = pd.DataFrame([{
            "ticker": "TEST.JK", "fundamental_score": 82.0,
            "fundamental_coverage": 90.0, "fundamental_model": "GENERAL",
            "fundamental_provider": "Yahoo Finance via yfinance",
        }])
        enriched = enrich_fundamentals_with_history(base, history, now="2026-07-01")
        self.assertEqual(enriched.loc[0, "fundamental_data_grade"], "A")
        self.assertGreaterEqual(float(enriched.loc[0, "fundamental_score_10"]), 8.0)

    def test_accounting_identity_conflict_caps_score(self):
        dates = pd.date_range("2025-03-31", periods=4, freq="QE")
        history = pd.DataFrame([{
            "ticker": "TEST.JK", "period_end": date, "period_type": "Q",
            "source_family": "YAHOO", "currency": "IDR", "revenue": 100 + i * 10,
            "net_income": 15 + i, "operating_cash_flow": 18 + i, "capex": 2,
            "total_assets": 500, "total_liabilities": 400, "equity": 300,
            "total_debt": 80, "cash": 50, "shares_outstanding": 100,
            "ebit": 20, "ebitda": 24, "interest_expense": 2,
        } for i, date in enumerate(dates)])
        base = pd.DataFrame([{"ticker": "TEST.JK", "fundamental_score": 90.0, "fundamental_coverage": 90.0}])
        enriched = enrich_fundamentals_with_history(base, history, now="2026-01-15")
        self.assertIn("ACCOUNTING_IDENTITY", enriched.loc[0, "fundamental_conflicts"])
        self.assertLessEqual(float(enriched.loc[0, "fundamental_score"]), 55.0)


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
        frame["technical_setup_ready"] = True
        frame["action"] = "READY_LIMIT"
        frame["entry_type"] = "LIMIT_ON_PULLBACK_THEN_CONFIRM"
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
        self.assertEqual(out.loc[0, "status"], "BLOCKED_CONTEXT")
        self.assertGreaterEqual(out.loc[0, "execution_integrity_score"], 90.0)
        self.assertEqual(out.loc[0, "order_instruction"], "DO_NOT_BUY")
        self.assertFalse(bool(out.loc[0, "autopilot_verified"]))


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
        frame["technical_setup_ready"] = True
        frame["action"] = "READY_LIMIT"
        frame["entry_type"] = "LIMIT_ON_PULLBACK_THEN_CONFIRM"
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
        frame["independent_last_price"] = 1_000.0
        frame["independent_price_age_days"] = 0
        frame["independent_date_gap_days"] = 0
        frame["ohlcv_source_tier"] = "LIVE_YAHOO"
        frame["absolute_data_age_days"] = 0
        frame["current_bar_incomplete"] = False
        frame["validation_gate_score"] = 80.0
        frame["validation_tier"] = "USABLE"
        frame["zero_volume_ratio20"] = 0.0
        frame["quote_critical_blocker"] = False
        return frame

    def test_one_optional_provider_gap_does_not_remove_verified_order(self):
        frame = self.executable_frame()
        frame["market_status_confidence"] = 100.0
        frame["news_confidence"] = 52.0  # news provider unavailable
        frame["fundamental_confidence"] = 100.0
        frame["validation_confidence"] = 100.0
        frame["quote_confidence"] = 100.0
        frame["universe_confidence"] = 100.0
        cfg = ScanConfig().replace(execution_policy="ACCOUNT_GUARDED", autopilot_enabled=True)
        frame = enforce_analyst_portfolio_budget(frame, cfg)
        out = finalize_execution_integrity(frame, cfg)
        self.assertEqual(out.loc[0, "status"], "EXECUTION_READY")
        self.assertEqual(out.loc[0, "order_instruction"], "BUY_LIMIT_USER_SIZE")
        self.assertTrue(bool(out.loc[0, "autopilot_verified"]))

    def test_multiple_missing_layers_produce_pending_data_not_disappearing_signal(self):
        frame = self.executable_frame()
        frame["market_status_confidence"] = 45.0
        frame["news_confidence"] = 52.0
        frame["fundamental_confidence"] = 50.0
        frame["validation_confidence"] = 45.0
        frame["quote_confidence"] = 68.0
        frame["universe_confidence"] = 48.0
        cfg = ScanConfig().replace(execution_policy="ACCOUNT_GUARDED", autopilot_enabled=True)
        frame = enforce_analyst_portfolio_budget(frame, cfg)
        out = finalize_execution_integrity(frame, cfg)
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
        cfg = ScanConfig().replace(execution_policy="ACCOUNT_GUARDED", autopilot_enabled=True)
        frame = enforce_analyst_portfolio_budget(frame, cfg)
        out = finalize_execution_integrity(frame, cfg)
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
        self.assertEqual(result.loc[0, "sniper_status"], "SNIPER_SIGNAL_READY")
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
        self.assertNotEqual(result.loc[0, "sniper_status"], "SNIPER_SIGNAL_READY")
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
        self.assertEqual(result.loc[0, "sniper_status"], "SNIPER_SIGNAL_READY")
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
        self.assertEqual(result.loc[0, "sniper_status"], "SNIPER_SIGNAL_READY")
        self.assertEqual(result.loc[0, "primary_sniper_blocker"], "NONE")

    def test_bsjp_ready_requires_late_session_intraday(self):
        frame = self._strong_daily_frame()
        result = scan_bsjp_candidates(
            {"TEST.JK": frame}, {"TEST.JK": self._intraday("bsjp")},
            now="2026-07-13 15:30:00",
        )
        self.assertFalse(result.empty)
        self.assertEqual(result.loc[0, "bsjp_status"], "BSJP_SIGNAL_READY")
        self.assertLessEqual(float(result.loc[0, "risk_pct"]), 0.035)
        self.assertTrue(bool(result.loc[0, "intraday_fresh"]))
        self.assertEqual(result.loc[0, "order_instruction"], "WAIT_SHARED_RISK_AND_PRICE_GATE")
        self.assertTrue(bool(result.loc[0, "target_structure_valid"]))
        self.assertNotIn('FALLBACK', str(result.loc[0, "tp1_basis"]))
        self.assertGreater(float(result.loc[0, "rr2"]), float(result.loc[0, "rr1"]))

    def test_bpjs_ready_requires_opening_range_and_vwap(self):
        frame = self._strong_daily_frame()
        result = scan_bpjs_candidates(
            {"TEST.JK": frame}, {"TEST.JK": self._intraday("bpjs")},
            now="2026-07-13 09:45:00",
        )
        self.assertFalse(result.empty)
        self.assertEqual(result.loc[0, "bpjs_status"], "BPJS_SIGNAL_READY")
        self.assertEqual(result.loc[0, "mandatory_exit"], "Before regular-market close")
        self.assertEqual(result.loc[0, "order_instruction"], "WAIT_SHARED_RISK_AND_PRICE_GATE")
        self.assertTrue(bool(result.loc[0, "target_structure_valid"]))
        self.assertNotIn('FALLBACK', str(result.loc[0, "tp1_basis"]))
        self.assertGreater(float(result.loc[0, "rr2"]), float(result.loc[0, "rr1"]))

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

    def test_risk_off_regime_preserves_signal_but_blocks_order(self):
        frame = self._strong_daily_frame()
        result = scan_bsjp_candidates(
            {"TEST.JK": frame}, {"TEST.JK": self._intraday("bsjp")},
            now="2026-07-13 15:30:00",
            market_context=MarketContext(regime="RISK_OFF"),
        )
        self.assertEqual(result.loc[0, "bsjp_status"], "BSJP_SIGNAL_READY")
        self.assertIn("RISK_OFF", result.loc[0, "warnings"])
        self.assertEqual(result.loc[0, "blockers"], "")
        self.assertFalse(bool(result.loc[0, "specialty_prebudget_order_eligible"]))
        self.assertIn("MARKET_REGIME_NOT_RISK_ON", result.loc[0, "specialty_risk_warnings"])

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
        self.assertIn(result.loc[0, "bpjs_status"], {"BPJS_SIGNAL_READY", "BPJS_WATCHLIST"})

    def test_bpjs_no_provider_data_is_distinct_from_waiting_bars(self):
        frame = self._strong_daily_frame()
        result = scan_bpjs_candidates(
            {"TEST.JK": frame}, {},
            now="2026-07-13 09:25:00",
        )
        self.assertFalse(result.empty)
        self.assertEqual(result.loc[0, "bpjs_status"], "BPJS_DATA_UNAVAILABLE")
        self.assertEqual(result.loc[0, "action"], "RETRY_INTRADAY_5M")

    def test_multibagger_snapshot_only_cannot_claim_a_or_b(self):
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
        self.assertEqual(result.loc[0, "multibagger_status"], "MULTIBAGGER_WATCHLIST")
        self.assertEqual(result.loc[0, "fundamental_data_grade"], "D")

    def test_multibagger_a_requires_history_consensus_and_verified_idx(self):
        frame = self._strong_daily_frame()
        fundamentals = pd.DataFrame([{
            "ticker": "TEST.JK", "fundamental_coverage": 92.0,
            "fundamental_score": 91.0, "fundamental_score_10": 9.1,
            "fundamental_data_grade": "A", "fundamental_source_count": 2,
            "fundamental_source_families": "YAHOO • IDX_OFFICIAL_XBRL",
            "fundamental_official_reference": True,
            "fundamental_official_verified": True,
            "fundamental_history_quarters": 8, "fundamental_history_years": 3,
            "fundamental_history_coverage": 90.0, "fundamental_consensus_score": 94.0,
            "fundamental_conflicts": "", "fundamental_reliability": "HIGH",
            "statement_age_days": 30,
            "revenue_growth": 0.30, "earnings_growth": 0.40,
            "roe": 0.25, "roa": 0.10, "net_margin": 0.15,
            "operating_margin": 0.20, "debt_equity": 0.40,
            "current_ratio": 2.0, "cash_to_debt": 1.0,
            "operating_cash_flow": 1_000_000_000.0,
            "free_cash_flow": 800_000_000.0, "peg_ratio": 1.0,
            "fcf_yield": 0.05, "market_cap": 5_000_000_000_000.0,
            "history_cash_conversion": 1.1, "history_positive_ocf_ratio": 1.0,
            "history_positive_earnings_ratio": 1.0, "history_margin_stability": 0.9,
            "history_share_dilution_yoy": 0.0, "history_roic_proxy": 0.20,
            "history_net_debt_ebitda": 0.5, "history_interest_coverage": 10.0,
            "fundamental_model": "GENERAL", "fundamental_red_flags": "",
        }])
        result = scan_multibagger_candidates({"TEST.JK": frame}, fundamentals)
        self.assertFalse(result.empty)
        self.assertEqual(result.loc[0, "multibagger_status"], "MULTIBAGGER_A_CANDIDATE")
        self.assertTrue(bool(result.loc[0, "grade_a_gate"]))

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
            "PRE_ARA_SIGNAL_READY", "PRE_ARA_CANDIDATE", "ARA_CONTINUATION_SIGNAL_READY",
            "ARA_CONTINUATION_CANDIDATE", "ARA_CONFIRMED_ONLY", "PRE_ARA_DAILY_RADAR"
        })
        self.assertFalse(bool(result.loc[0, "account_risk_gate_applied"]))
        self.assertIn(result.loc[0, "setup_state"], {"SETUP_READY", "DAILY_RADAR"})
        self.assertNotEqual(result.loc[0, "order_instruction"], "SPECULATIVE_REVIEW_ONLY")

    def test_specialty_builder_returns_all_named_tables(self):
        frame = self._strong_daily_frame()
        result = build_specialty_screens({"TEST.JK": frame})
        self.assertEqual(set(result), {"sniper", "bsjp", "bpjs", "multibagger", "ara_hunter", "daily_opportunities", "profit_order_builder"})


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
        cfg = ScanConfig().replace(min_data_completeness=95.0, execution_policy="ACCOUNT_GUARDED", autopilot_enabled=True)
        frame = enforce_analyst_portfolio_budget(frame, cfg)
        out = finalize_execution_integrity(frame, cfg)
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
        cfg = ScanConfig().replace(execution_policy="ACCOUNT_GUARDED", autopilot_enabled=True)
        frame = enforce_analyst_portfolio_budget(frame, cfg)
        out = finalize_execution_integrity(frame, cfg)
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
        cfg = ScanConfig().replace(execution_policy="ACCOUNT_GUARDED", autopilot_enabled=True)
        frame = enforce_analyst_portfolio_budget(frame, cfg)
        out = finalize_execution_integrity(frame, cfg)
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

    def test_twelve_fundamental_adapter_merges_three_statements(self):
        payloads = {
            "income_statement": {"income_statement": [{
                "fiscal_date": "2026-03-31", "period": "quarterly",
                "revenue": "1000", "net_income": "120", "ebit": "170",
                "ebitda": "190", "interest_expense": "15",
            }]},
            "balance_sheet": {"balance_sheet": [{
                "fiscal_date": "2026-03-31", "period": "quarterly",
                "total_assets": "5000", "total_liabilities": "2000",
                "equity": "3000", "total_debt": "600", "cash": "500",
                "shares_outstanding": "100",
            }]},
            "cash_flow": {"cash_flow": [{
                "fiscal_date": "2026-03-31", "period": "quarterly",
                "operating_cash_flow": "150", "capex": "30",
            }]},
        }

        def fake_get(url, **kwargs):
            endpoint = url.rsplit("/", 1)[-1]
            response = SimpleNamespace(json=lambda: payloads[endpoint])
            response.raise_for_status = lambda: None
            return response

        with patch.dict(sys.modules, {"requests": SimpleNamespace(get=fake_get)}):
            history, report = fetch_twelve_data_fundamental_history(
                ["TEST.JK"], api_key="SECRET_KEY", max_tickers=1,
            )
        self.assertEqual(report.loc[0, "status"], "OK")
        self.assertEqual(len(history), 1)
        self.assertEqual(float(history.loc[0, "revenue"]), 1000.0)
        self.assertEqual(float(history.loc[0, "operating_cash_flow"]), 150.0)
        self.assertEqual(history.loc[0, "source_family"], "TWELVE_DATA")
        self.assertNotIn("SECRET_KEY", report.to_string())


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
        self.assertEqual(app_source.count("st.file_uploader("), 7)
        self.assertNotIn("independent_price_upload", app_source)
        self.assertIn("broksum_upload", app_source)
        self.assertIn("orderbook_upload", app_source)
        self.assertIn("fundamental_history_upload", app_source)


    def test_release_manifest_contains_deployment_and_integration_files(self):
        root_required = {
            "requirements.txt", "runtime.txt", "README.md",
            "IDX_Scanner_Confirmation_v1.pine", "FIX_REPORT_V6_6_5.md",
            "DASHBOARD_RANKING_GUIDE_V6_6_5.md", "BUILD_VALIDATION_V6_6_5.md",
        }
        docs_required = {"STOCKBIT_SCREENER_PRESETS.md", "DEPLOYMENT_CHECKLIST.md"}
        self.assertTrue(root_required.issubset({path.name for path in ROOT.iterdir()}))
        docs_root = ROOT / "docs" if (ROOT / "docs").is_dir() else ROOT
        self.assertTrue(docs_required.issubset({path.name for path in docs_root.iterdir()}))

    def test_effective_release_version_and_price_verification_capacity(self):
        self.assertEqual(scanner_module.__version__, "6.6.5-best-buy-eoff-top20")
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
            "PRE_ARA_SIGNAL_READY", "PRE_ARA_CANDIDATE", "PRE_ARA_WATCHLIST", "PRE_ARA_DAILY_RADAR"
        })
        self.assertTrue(bool(pre_result.loc[0, "target_structure_valid"]))
        self.assertNotIn('FALLBACK', str(pre_result.loc[0, "tp1_basis"]))

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
            "ARA_CONTINUATION_SIGNAL_READY", "ARA_CONTINUATION_CANDIDATE", "ARA_CONFIRMED_ONLY"
        })
        self.assertTrue(bool(cont_result.loc[0, "target_structure_valid"]))
        self.assertTrue(bool(cont_result.loc[0, "target_recalc_required"]))
        self.assertNotIn('FALLBACK', str(cont_result.loc[0, "tp2_basis"]))

    def test_optional_observed_flow_can_verify_continuation(self):
        ara = pd.DataFrame([{
            "ticker": "ANTM.JK", "ara_hunter_status": "ARA_CONTINUATION_SIGNAL_READY",
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
        self.assertEqual(out.loc[0, "ara_hunter_status"], "ARA_CONTINUATION_FLOW_VERIFIED_SIGNAL")
        self.assertFalse(bool(out.loc[0].get("specialty_order_ready", False)))
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



class ProfitConvictionEngineV58Tests(unittest.TestCase):
    def test_profit_builder_keeps_independent_setups_for_same_ticker(self):
        screens = {
            "bpjs": pd.DataFrame([{
                "ticker": "AAA.JK", "bpjs_status": "BPJS_SIGNAL_READY", "bpjs_score": 92.0,
                "entry": 100.0, "stop_loss": 96.0, "day_tp1": 108.0, "day_tp2": 114.0,
                "rr1": 2.0, "rr2": 3.5, "target_structure_valid": True,
                "adtv20_idr": 25_000_000_000.0, "orb_breakout_confirmed": True,
                "orb_hold_confirmed": True, "session_close_location": 0.92,
                "vwap_slope_pct": 0.002, "breakout_volume_ratio": 2.2,
                "directional_efficiency": 0.85, "intraday_fresh": True,
                "order_instruction": "BUY_LIMIT_USER_SIZE",
            }]),
            "bsjp": pd.DataFrame([{
                "ticker": "AAA.JK", "bsjp_status": "BSJP_SIGNAL_READY", "bsjp_score": 76.0,
                "entry": 100.0, "stop_loss": 96.0, "morning_tp1": 106.0, "morning_tp2": 110.0,
                "rr1": 1.5, "rr2": 2.5, "target_structure_valid": True,
                "adtv20_idr": 10_000_000_000.0, "session_close_location": 0.78,
                "vwap_slope_pct": 0.001, "late_volume_acceleration": 1.4,
                "directional_efficiency": 0.55, "late_volume_share": 0.38,
                "afternoon_breakout": True, "intraday_fresh": True,
            }]),
            "sniper": pd.DataFrame([{
                "ticker": "BBB.JK", "sniper_status": "SNIPER_SIGNAL_READY", "sniper_score": 84.0,
                "sniper_entry": 200.0, "sniper_trigger": 202.0, "sniper_stop": 194.0,
                "sniper_tp1": 216.0, "sniper_tp2": 228.0, "rr1": 2.0, "rr2": 3.5,
                "target_structure_valid": True, "adtv20_idr": 8_000_000_000.0,
                "trigger_state": "CONFIRMED", "silent_accumulation_score": 75.0,
                "primary_sniper_blocker": "NONE",
            }]),
            "ara_hunter": pd.DataFrame(),
        }
        out = build_profit_order_builder(pd.DataFrame(), screens, ScanConfig().replace(profit_conviction_min_score=0.0))
        self.assertEqual(out["ticker"].tolist().count("AAA.JK"), 2)
        self.assertEqual(out.iloc[0]["ticker"], "AAA.JK")
        self.assertEqual(out.iloc[0]["strategy"], "BPJS")
        self.assertIn("BSJP", out.iloc[0]["alternate_strategies"])
        self.assertEqual(set(out.loc[out["ticker"].eq("AAA.JK"), "strategy"]), {"BPJS", "BSJP"})
        self.assertEqual(out.loc[out["ticker"].eq("AAA.JK"), "candidate_id"].nunique(), 2)
        self.assertEqual(out["profit_rank"].tolist(), list(range(1, len(out) + 1)))

    def test_core_probability_in_zero_to_one_scale_is_normalized(self):
        core = pd.DataFrame([{
            "ticker": "CORE.JK", "setup": "PULLBACK_CONTINUATION",
            "status": "EXECUTION_READY", "action": "READY_TRIGGER",
            "quality_score": 90.0, "analyst_fusion_score": 90.0, "distance_atr": 0.1,
            "silent_accumulation_score": 75.0, "adtv20_idr": 20_000_000_000.0,
            "rr1": 2.0, "rr2": 3.0, "target_structure_valid": True,
            "data_completeness_score": 90.0, "execution_confidence_score": 90.0,
            "validation_gate_score": 80.0, "probability_estimate": 0.80,
            "entry": 100.0, "stop_loss": 95.0, "tp1": 110.0, "tp2": 115.0,
        }])
        out = build_profit_order_builder(core, {}, ScanConfig().replace(profit_conviction_min_score=0.0))
        self.assertFalse(out.empty)
        self.assertGreaterEqual(float(out.loc[0, "validation_score"]), 79.0)

    def test_profit_builder_respects_configured_limit(self):
        rows = []
        for idx in range(5):
            rows.append({
                "ticker": f"T{idx:03d}.JK", "bpjs_status": "BPJS_SIGNAL_READY",
                "bpjs_score": 90.0 - idx, "entry": 100.0, "stop_loss": 95.0,
                "day_tp1": 110.0, "day_tp2": 115.0, "rr1": 2.0, "rr2": 3.0,
                "target_structure_valid": True, "adtv20_idr": 10_000_000_000.0,
                "orb_breakout_confirmed": True, "orb_hold_confirmed": True,
                "session_close_location": 0.9, "vwap_slope_pct": 0.001,
                "breakout_volume_ratio": 2.0, "directional_efficiency": 0.8,
                "intraday_fresh": True,
            })
        out = build_profit_order_builder(
            pd.DataFrame(), {"bpjs": pd.DataFrame(rows)},
            ScanConfig().replace(profit_conviction_min_score=0.0, profit_order_builder_limit=3),
        )
        self.assertEqual(len(out), 3)
        self.assertEqual(out["profit_rank"].tolist(), [1, 2, 3])



class LocalHybridAITests(unittest.TestCase):
    def _ranking(self):
        return pd.DataFrame([
            {
                'profit_rank': 1, 'ticker': 'AAA.JK', 'strategy': 'PULLBACK_CONTINUATION',
                'decision_state': 'SETUP_READY', 'profit_conviction_score': 78.0,
                'structure_score': 84.0, 'timing_score': 80.0, 'flow_score': 74.0,
                'liquidity_score': 82.0, 'target_quality_score': 76.0,
                'data_quality_score': 90.0, 'validation_score': 70.0,
                'entry': 100.0, 'stop_loss': 95.0, 'tp1': 110.0, 'tp2': 118.0,
                'rr1': 2.0, 'rr2': 3.6, 'market_regime': 'RISK_ON',
                'atr_pct': 0.025, 'volume_ratio': 1.2, 'rsi14': 58, 'adx14': 24,
                'cmf20': 0.08, 'roc60': 0.12, 'distance_52w_high': -0.08,
                'relative_strength60': 0.06, 'silent_accumulation_score': 78,
                'body_atr': 0.65, 'close_location': 0.76,
            },
            {
                'profit_rank': 2, 'ticker': 'BBB.JK', 'strategy': 'BREAKOUT_RETEST',
                'decision_state': 'ENTRY_PLAN', 'profit_conviction_score': 82.0,
                'structure_score': 75.0, 'timing_score': 55.0, 'flow_score': 45.0,
                'liquidity_score': 55.0, 'target_quality_score': 58.0,
                'data_quality_score': 78.0, 'validation_score': 45.0,
                'entry': 200.0, 'stop_loss': 190.0, 'tp1': 212.0, 'tp2': 225.0,
                'rr1': 1.2, 'rr2': 2.5, 'market_regime': 'RISK_OFF',
                'atr_pct': 0.07, 'volume_ratio': 0.6, 'rsi14': 72, 'adx14': 15,
                'cmf20': -0.07, 'roc60': -0.03, 'distance_52w_high': -0.25,
                'relative_strength60': -0.08, 'silent_accumulation_score': 35,
                'body_atr': 0.25, 'close_location': 0.42,
            },
        ])

    def _history(self, n=90):
        rows = []
        rng = np.random.default_rng(7)
        for i in range(n):
            strong = i % 3 != 0
            filled = bool(strong or i % 6 == 0)
            rows.append({
                'ticker': f'T{i:03d}.JK', 'setup': 'PULLBACK_CONTINUATION' if i % 2 else 'BREAKOUT_RETEST',
                'strategy': 'PULLBACK_CONTINUATION' if i % 2 else 'BREAKOUT_RETEST',
                'signal_date': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i),
                'market_regime': 'RISK_ON' if strong else 'RISK_OFF',
                'quality_score': 85 if strong else 60, 'filled': filled,
                'tp1_hit': bool(strong) if filled else np.nan,
                'r_multiple': (1.4 if strong else -1.0) if filled else 0.0,
                'rr1_plan': 2.0 if strong else 1.1, 'rr2_plan': 3.2 if strong else 1.7,
                'stop_pct': 0.04 if strong else 0.09, 'atr_pct': 0.025 if strong else 0.08,
                'volume_ratio': 1.4 if strong else 0.55, 'rsi14': 58 if strong else 75,
                'adx14': 25 if strong else 13, 'cmf20': 0.09 if strong else -0.08,
                'roc60': 0.13 if strong else -0.05, 'distance_52w_high': -0.07 if strong else -0.3,
                'relative_strength60': 0.08 if strong else -0.1,
                'silent_accumulation_score': 80 if strong else 30,
                'adtv20_idr': 8e9 if strong else 4e8, 'body_atr': 0.7 if strong else 0.2,
                'close_location': 0.8 if strong else 0.35,
            })
        return pd.DataFrame(rows)

    def test_local_ai_enriches_and_reorders_with_guarded_weight(self):
        from ai_engine import LocalAIConfig, enrich_profit_ranking_with_ai
        result, audit = enrich_profit_ranking_with_ai(
            self._ranking(), self._history(240), None,
            LocalAIConfig(min_training_events=20, max_weight=0.35),
        )
        self.assertIn('hybrid_conviction_score', result.columns)
        self.assertIn('ai_tp1_probability_pct', result.columns)
        self.assertFalse(audit.empty)
        self.assertEqual(audit.set_index('metric').loc['Joint validation state', 'value'], 'VALIDATED')
        self.assertEqual(result.iloc[0]['ticker'], 'AAA.JK')
        self.assertLessEqual(float(result['ai_effective_weight_pct'].max()), 35.0)

    def test_shadow_mode_never_changes_rule_score(self):
        from ai_engine import LocalAIConfig, enrich_profit_ranking_with_ai
        ranking = self._ranking()
        result, _ = enrich_profit_ranking_with_ai(
            ranking, self._history(), None,
            LocalAIConfig(mode='SHADOW_ONLY', min_training_events=20),
        )
        mapped = result.set_index('ticker')
        original = ranking.set_index('ticker')
        for ticker in original.index:
            self.assertAlmostEqual(float(mapped.loc[ticker, 'hybrid_conviction_score']), float(original.loc[ticker, 'profit_conviction_score']))

    def test_validation_pending_has_exactly_zero_ai_weight(self):
        from ai_engine import LocalAIConfig, enrich_profit_ranking_with_ai
        result, audit = enrich_profit_ranking_with_ai(
            self._ranking(), self._history(55), None,
            LocalAIConfig(
                min_training_events=20,
                min_calibration_events=40,
                min_evaluation_events=40,
                max_weight=0.35,
            ),
        )
        state = audit.set_index('metric').loc['Joint validation state', 'value']
        self.assertNotEqual(state, 'VALIDATED')
        self.assertTrue(result['ai_effective_weight_pct'].eq(0.0).all())
        self.assertTrue(result['hybrid_conviction_score'].equals(result['profit_conviction_score']))

    def test_outcome_memory_registers_and_resolves(self):
        from ai_engine import LocalAIConfig, update_outcome_memory, resolved_memory_events
        dates = pd.date_range('2026-01-01', periods=30, freq='D')
        frame = pd.DataFrame({
            'Open': [100.0] * 30, 'High': [102.0] * 30, 'Low': [98.0] * 30,
            'Close': [101.0] * 30, 'Volume': [1_000_000] * 30,
        }, index=dates)
        ranking = self._ranking().iloc[[0]].copy()
        memory = update_outcome_memory(ranking, {'AAA.JK': frame.iloc[:10]}, pd.DataFrame(), LocalAIConfig(memory_entry_window_bars=3, memory_horizon_bars=5))
        self.assertEqual(len(memory), 1)
        # Later bars hit TP1 after entry.
        later = frame.copy()
        later.loc[dates[10]:, 'High'] = 112.0
        memory2 = update_outcome_memory(pd.DataFrame(), {'AAA.JK': later}, memory, LocalAIConfig(memory_entry_window_bars=3, memory_horizon_bars=5))
        resolved = resolved_memory_events(memory2)
        self.assertFalse(resolved.empty)
        self.assertTrue(bool(resolved.iloc[0]['filled']))

    def test_no_history_means_no_ai_influence_or_published_probability(self):
        from ai_engine import LocalAIConfig, enrich_profit_ranking_with_ai
        ranking = self._ranking()
        result, _ = enrich_profit_ranking_with_ai(
            ranking, pd.DataFrame(), pd.DataFrame(), LocalAIConfig(),
        )
        mapped = result.set_index('ticker')
        original = ranking.set_index('ticker')
        for ticker in original.index:
            self.assertAlmostEqual(
                float(mapped.loc[ticker, 'hybrid_conviction_score']),
                float(original.loc[ticker, 'profit_conviction_score']),
            )
            self.assertEqual(float(mapped.loc[ticker, 'ai_effective_weight_pct']), 0.0)
            self.assertTrue(pd.isna(mapped.loc[ticker, 'ai_trade_success_probability_pct']))
            self.assertFalse(bool(mapped.loc[ticker, 'ai_can_influence_ranking']))

    def test_duplicate_validation_and_memory_events_count_once(self):
        from ai_engine import LocalAIConfig, enrich_profit_ranking_with_ai
        history = self._history(60)
        history['signal_id'] = [f'event-{idx}' for idx in range(len(history))]
        _, audit = enrich_profit_ranking_with_ai(
            self._ranking(), history, history.copy(),
            LocalAIConfig(min_training_events=20),
        )
        metrics = audit.set_index('metric')['value']
        self.assertEqual(int(metrics['Raw outcome rows']), 120)
        self.assertEqual(int(metrics['Deduplicated outcome rows']), 60)
        self.assertEqual(int(metrics['Duplicate rows removed']), 60)

    def test_missing_outcomes_are_not_fabricated_as_losses(self):
        from ai_engine import LocalAIConfig, enrich_profit_ranking_with_ai
        unresolved = self._history(60)
        unresolved['filled'] = np.nan
        unresolved['tp1_hit'] = np.nan
        result, audit = enrich_profit_ranking_with_ai(
            self._ranking(), unresolved, None, LocalAIConfig(min_training_events=20),
        )
        metrics = audit.set_index('metric')['value']
        self.assertEqual(int(metrics['Filled training events']), 0)
        self.assertTrue(result['ai_trade_success_probability_pct'].isna().all())
        self.assertTrue(result['ai_effective_weight_pct'].eq(0.0).all())

    def test_probability_bounds_refer_to_combined_fill_times_tp1(self):
        from ai_engine import LocalAIConfig, enrich_profit_ranking_with_ai
        result, _ = enrich_profit_ranking_with_ai(
            self._ranking(), self._history(160), None,
            LocalAIConfig(min_training_events=20, min_strategy_events=10),
        )
        available = result.dropna(subset=['ai_trade_success_probability_pct'])
        self.assertFalse(available.empty)
        self.assertTrue((available['ai_probability_lower_pct'] <= available['ai_trade_success_probability_pct']).all())
        self.assertTrue((available['ai_trade_success_probability_pct'] <= available['ai_probability_upper_pct']).all())
        self.assertTrue((available['ai_probability_lower_pct'] <= available['ai_tp1_probability_lower_pct']).all())

    def test_missing_live_features_never_produce_nan_hybrid_score(self):
        from ai_engine import LocalAIConfig, enrich_profit_ranking_with_ai
        ranking = self._ranking().drop(columns=['rsi14', 'adx14', 'cmf20', 'roc60', 'body_atr'])
        result, _ = enrich_profit_ranking_with_ai(
            ranking, self._history(120), None,
            LocalAIConfig(min_training_events=20),
        )
        self.assertTrue(pd.to_numeric(result['hybrid_conviction_score'], errors='coerce').notna().all())
        self.assertTrue(result['ai_feature_coverage_pct'].lt(100.0).all())

    def test_chronological_split_never_divides_same_signal_date(self):
        from ai_engine import _chronological_split
        dates = np.repeat(pd.date_range('2025-01-01', periods=10, freq='D'), 3)
        frame = pd.DataFrame({'signal_date': dates, 'filled': True})
        train, holdout = _chronological_split(frame, 0.30)
        self.assertTrue(set(pd.to_datetime(train['signal_date'])).isdisjoint(set(pd.to_datetime(holdout['signal_date']))))

    def test_memory_cancels_limit_when_gap_opens_below_stop(self):
        from ai_engine import LocalAIConfig, _resolve_one
        row = pd.Series({
            'memory_state': 'OPEN', 'signal_date': pd.Timestamp('2026-01-01'),
            'entry': 100.0, 'stop_loss': 95.0, 'tp1': 110.0, 'tp2': 118.0,
        })
        frame = pd.DataFrame(
            {'Open': [90.0], 'High': [94.0], 'Low': [88.0], 'Close': [92.0]},
            index=[pd.Timestamp('2026-01-02')],
        )
        resolved = _resolve_one(row, frame, LocalAIConfig())
        self.assertEqual(resolved['result'], 'NO_FILL_INVALIDATED_GAP')
        self.assertFalse(bool(resolved['filled']))

    def test_memory_stop_gap_uses_open_and_includes_cost(self):
        from ai_engine import LocalAIConfig, _resolve_one
        row = pd.Series({
            'memory_state': 'OPEN', 'signal_date': pd.Timestamp('2026-01-01'),
            'entry': 100.0, 'stop_loss': 95.0, 'tp1': 110.0, 'tp2': 118.0,
        })
        frame = pd.DataFrame(
            {
                'Open': [100.0, 90.0], 'High': [102.0, 93.0],
                'Low': [98.0, 88.0], 'Close': [101.0, 91.0],
            },
            index=pd.to_datetime(['2026-01-02', '2026-01-03']),
        )
        resolved = _resolve_one(row, frame, LocalAIConfig(memory_horizon_bars=5))
        self.assertEqual(resolved['result'], 'LOSS_GAP')
        self.assertAlmostEqual(float(resolved['exit_price']), 90.0)
        self.assertLess(float(resolved['r_multiple']), -2.0)

    def test_csv_boolean_outcomes_parse_without_turning_open_rows_into_losses(self):
        from ai_engine import _binary_outcome
        parsed = _binary_outcome(pd.Series(['TRUE', 'FALSE', '', 'OPEN', None]))
        self.assertEqual(float(parsed.iloc[0]), 1.0)
        self.assertEqual(float(parsed.iloc[1]), 0.0)
        self.assertTrue(parsed.iloc[2:].isna().all())

    def test_intrabar_fill_exit_ambiguity_is_not_a_win_or_loss_label(self):
        from ai_engine import LocalAIConfig, _resolve_one
        row = pd.Series({
            'memory_state': 'OPEN', 'signal_date': pd.Timestamp('2026-01-01'),
            'entry': 100.0, 'stop_loss': 95.0, 'tp1': 110.0, 'tp2': 118.0,
        })
        frame = pd.DataFrame(
            {'Open': [103.0], 'High': [112.0], 'Low': [99.0], 'Close': [108.0]},
            index=[pd.Timestamp('2026-01-02')],
        )
        resolved = _resolve_one(row, frame, LocalAIConfig())
        self.assertEqual(resolved['result'], 'AMBIGUOUS_FILL_BAR')
        self.assertTrue(bool(resolved['filled']))
        self.assertTrue(bool(resolved['outcome_ambiguous']))
        self.assertTrue(pd.isna(resolved['tp1_hit']))
        self.assertTrue(pd.isna(resolved['r_multiple']))

    def test_memory_summary_excludes_ambiguous_outcomes_from_hit_rate(self):
        from ai_engine import memory_summary
        memory = pd.DataFrame({
            'memory_state': ['RESOLVED', 'RESOLVED', 'RESOLVED', 'RESOLVED'],
            'filled': ['TRUE', 'TRUE', 'TRUE', 'FALSE'],
            'tp1_hit': ['TRUE', 'FALSE', 'OPEN', 'FALSE'],
        })
        metrics = memory_summary(memory).set_index('metric')['value']
        self.assertEqual(int(metrics['Filled']), 3)
        self.assertEqual(int(metrics['TP1-labelled outcomes']), 2)
        self.assertEqual(int(metrics['Ambiguous filled outcomes']), 1)
        self.assertEqual(float(metrics['Empirical hit rate']), 50.0)

    def test_model_without_oos_skill_cannot_change_ranking(self):
        from ai_engine import LocalAIConfig, enrich_profit_ranking_with_ai
        rng = np.random.default_rng(42)
        history = []
        for idx in range(400):
            history.append({
                'ticker': f'R{idx:03d}.JK', 'strategy': 'PULLBACK_CONTINUATION',
                'signal_date': pd.Timestamp('2023-01-01') + pd.Timedelta(days=idx),
                'market_regime': 'NEUTRAL', 'quality_score': rng.normal(70, 10),
                'filled': bool(idx % 4), 'tp1_hit': bool(rng.integers(0, 2)),
                'r_multiple': rng.choice([-1.0, 1.5]),
                'rr1_plan': rng.uniform(0.8, 2.5), 'rr2_plan': rng.uniform(1.5, 4.0),
                'stop_pct': rng.uniform(0.03, 0.10), 'atr_pct': rng.uniform(0.02, 0.08),
                'volume_ratio': rng.uniform(0.5, 2.0), 'rsi14': rng.uniform(35, 75),
                'adx14': rng.uniform(10, 35), 'cmf20': rng.uniform(-0.10, 0.15),
                'roc60': rng.uniform(-0.20, 0.20), 'distance_52w_high': rng.uniform(-0.40, 0),
                'relative_strength60': rng.uniform(-0.20, 0.20),
                'silent_accumulation_score': rng.uniform(20, 90),
                'adtv20_idr': rng.uniform(2e8, 1e10), 'body_atr': rng.uniform(0.1, 1.0),
                'close_location': rng.uniform(0.2, 0.9),
            })
        ranking = self._ranking()
        result, audit = enrich_profit_ranking_with_ai(
            ranking, pd.DataFrame(history), None, LocalAIConfig(),
        )
        skill = float(audit.set_index('metric').loc['OOS Brier skill', 'value'])
        self.assertLessEqual(skill, 0.0)
        self.assertTrue(result['ai_effective_weight_pct'].eq(0.0).all())
        self.assertTrue(result['ai_model_state'].eq('MODEL_REJECTED_NO_OOS_SKILL').all())
        mapped = result.set_index('ticker')
        original = ranking.set_index('ticker')
        for ticker in original.index:
            self.assertAlmostEqual(
                float(mapped.loc[ticker, 'hybrid_conviction_score']),
                float(original.loc[ticker, 'profit_conviction_score']),
            )

    def test_fill_and_tp1_components_must_each_pass_oos_skill(self):
        from ai_engine import LocalAIConfig, enrich_profit_ranking_with_ai
        history = self._history(480)
        rng = np.random.default_rng(123)
        filled = rng.random(len(history)) < 0.75
        history['filled'] = filled
        history['tp1_hit'] = np.where(filled, history['quality_score'].gt(70), np.nan)
        history['r_multiple'] = np.where(
            ~filled, 0.0, np.where(history['tp1_hit'].eq(1), 1.4, -1.0),
        )
        result, audit = enrich_profit_ranking_with_ai(
            self._ranking(), history, None, LocalAIConfig(),
        )
        metrics = audit.set_index('metric')['value']
        self.assertLess(float(metrics['Fill OOS Brier skill']), 0.02)
        self.assertGreater(float(metrics['TP1 OOS Brier skill']), 0.02)
        self.assertEqual(metrics['Joint validation state'], 'REJECTED_NO_SKILL')
        self.assertTrue(result['ai_effective_weight_pct'].eq(0.0).all())
        self.assertTrue(result['ai_gate_reasons'].str.contains('FILL_MODEL_NO_OOS_SKILL').all())



class MultibaggerPeerAITests(unittest.TestCase):
    def test_peer_ai_rewards_quality_cashflow_and_penalizes_outlier(self):
        from ai_engine import enrich_multibagger_with_peer_ai
        frame = pd.DataFrame([
            {
                'ticker': 'GOOD.JK', 'revenue_growth': 0.25, 'earnings_growth': 0.32,
                'roe': 0.24, 'roa': 0.10, 'net_margin': 0.16,
                'cash_conversion_ttm': 1.05, 'positive_ocf_ratio': 1.0,
                'positive_earnings_ratio': 1.0, 'margin_stability': 0.9,
                'fcf_yield': 0.07, 'roic_proxy': 0.18, 'interest_coverage': 12,
                'cash_to_debt': 0.8, 'debt_equity': 0.5, 'net_debt_ebitda': 0.7,
                'share_dilution_yoy': 0.0, 'project_pipeline_score': 78,
                'management_quality_score': 82, 'future_fundamental_impact_score': 76,
                'fundamental_consensus_score': 90, 'fundamental_coverage': 92,
                'silent_accumulation_score': 72, 'future_revenue_uplift_base_pct': 18,
                'future_ebitda_uplift_base_pct': 22, 'future_net_debt_change_pct': 8,
                'project_success_probability_pct': 82,
            },
            {
                'ticker': 'RISK.JK', 'revenue_growth': 0.45, 'earnings_growth': 0.50,
                'roe': 0.10, 'roa': 0.01, 'net_margin': 0.02,
                'cash_conversion_ttm': -0.3, 'positive_ocf_ratio': 0.25,
                'positive_earnings_ratio': 0.5, 'margin_stability': 0.2,
                'fcf_yield': -0.12, 'roic_proxy': 0.01, 'interest_coverage': 0.8,
                'cash_to_debt': 0.05, 'debt_equity': 3.2, 'net_debt_ebitda': 7.0,
                'share_dilution_yoy': 0.25, 'project_pipeline_score': 90,
                'management_quality_score': 35, 'future_fundamental_impact_score': 80,
                'fundamental_consensus_score': 55, 'fundamental_coverage': 70,
                'silent_accumulation_score': 40, 'future_revenue_uplift_base_pct': 35,
                'future_ebitda_uplift_base_pct': 15, 'future_net_debt_change_pct': 180,
                'project_success_probability_pct': 45,
            },
            {'ticker': 'MID.JK', 'revenue_growth': 0.10, 'earnings_growth': 0.12, 'roe': 0.13, 'roa': 0.05,
             'net_margin': 0.08, 'cash_conversion_ttm': 0.8, 'positive_ocf_ratio': 0.75,
             'positive_earnings_ratio': 0.75, 'margin_stability': 0.65, 'fcf_yield': 0.03,
             'roic_proxy': 0.10, 'interest_coverage': 5, 'cash_to_debt': 0.3, 'debt_equity': 1.0,
             'net_debt_ebitda': 2.0, 'share_dilution_yoy': 0.02, 'project_pipeline_score': 55,
             'management_quality_score': 60, 'future_fundamental_impact_score': 55,
             'fundamental_consensus_score': 75, 'fundamental_coverage': 80, 'silent_accumulation_score': 55,
             'future_revenue_uplift_base_pct': 8, 'future_ebitda_uplift_base_pct': 10,
             'future_net_debt_change_pct': 25, 'project_success_probability_pct': 65},
        ])
        result = enrich_multibagger_with_peer_ai(frame)
        scores = result.set_index('ticker')['ai_multibagger_peer_score']
        self.assertGreater(float(scores['GOOD.JK']), float(scores['RISK.JK']))
        self.assertIn('ai_multibagger_outlier_risk', result.columns)
        self.assertTrue(result['ai_multibagger_effective_weight_pct'].eq(0.0).all())
        self.assertTrue(result['ai_multibagger_gate_reasons'].str.contains('FUNDAMENTAL_GRADE_LOW').all())

    def test_peer_ai_uses_sector_and_requires_verified_official_filing(self):
        from ai_engine import enrich_multibagger_with_peer_ai
        rows = []
        for idx in range(10):
            rows.append({
                'ticker': f'P{idx:02d}.JK',
                'sector': 'Technology' if idx < 5 else 'Consumer Cyclical',
                'fundamental_model': 'GENERAL', 'fundamental_data_grade': 'A',
                'fundamental_official_verified': idx != 0,
                'statement_current': True, 'fundamental_source_count': 2,
                'fundamental_conflicts': '', 'severe_fundamental_flags': False,
                'fundamental_score_10': 8.0, 'fundamental_consensus_score': 85.0,
                'fundamental_coverage': 90.0, 'revenue_growth': 0.10 + idx / 100,
                'earnings_growth': 0.12 + idx / 100, 'roe': 0.18, 'roa': 0.08,
                'net_margin': 0.12, 'cash_conversion_ttm': 1.0,
                'positive_ocf_ratio': 1.0, 'positive_earnings_ratio': 1.0,
                'margin_stability': 0.8, 'fcf_yield': 0.05, 'roic_proxy': 0.15,
                'interest_coverage': 8.0, 'cash_to_debt': 0.6, 'debt_equity': 0.7,
                'net_debt_ebitda': 1.2, 'share_dilution_yoy': 0.0,
                'project_pipeline_score': 70.0, 'management_quality_score': 75.0,
                'future_fundamental_impact_score': 70.0,
                'silent_accumulation_score': 65.0,
            })
        result = enrich_multibagger_with_peer_ai(pd.DataFrame(rows)).set_index('ticker')
        self.assertEqual(result.loc['P00.JK', 'ai_multibagger_peer_group'], 'SECTOR::TECHNOLOGY')
        self.assertEqual(int(result.loc['P00.JK', 'ai_multibagger_peer_count']), 5)
        self.assertEqual(result.loc['P09.JK', 'ai_multibagger_peer_group'], 'SECTOR::CONSUMER_CYCLICAL')
        self.assertEqual(float(result.loc['P00.JK', 'ai_multibagger_effective_weight_pct']), 0.0)
        self.assertIn('OFFICIAL_XBRL_NOT_VERIFIED', result.loc['P00.JK', 'ai_multibagger_gate_reasons'])
        self.assertGreater(float(result.loc['P01.JK', 'ai_multibagger_effective_weight_pct']), 0.0)

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
        self.assertEqual(out.loc[0, "status"], "ENTRY_PLAN_READY")
        self.assertEqual(out.loc[0, "execution_mode"], "SIGNAL_FIRST_ENTRY_PLAN")
        self.assertTrue(bool(out.loc[0, "requires_stockbit_price_check"]))
        self.assertEqual(out.loc[0, "order_instruction"], "WAIT_PRICE_AND_CONFIRM")
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
        self.assertEqual(out.loc[0, "status"], "ENTRY_PLAN_READY")
        self.assertEqual(out.loc[0, "primary_execution_blocker"], "TRIGGER_NOT_CONFIRMED")
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

class ExecutionSemanticsV501Tests(unittest.TestCase):
    def test_buy_stop_style_uses_lit_trigger_and_separate_limit(self):
        frame = AnalystFusionV440Tests().pullback_limit_candidate()
        frame.loc[0, "action"] = "READY_TRIGGER"
        frame.loc[0, "entry_type"] = "BUY_STOP_AFTER_RETEST"
        frame.loc[0, "last_price"] = 625.0
        frame.loc[0, "entry"] = 635.0
        frame.loc[0, "trigger"] = 635.0
        frame.loc[0, "entry_low"] = 625.0
        frame.loc[0, "entry_high"] = 645.0
        frame.loc[0, "stop_loss"] = 610.0
        frame.loc[0, "tp1"] = 680.0
        frame.loc[0, "tp2"] = 710.0
        frame.loc[0, "rr1"] = 1.8
        frame.loc[0, "rr2"] = 3.0
        frame.loc[0, "stop_pct"] = 25.0 / 635.0
        frame.loc[0, "distance_atr"] = 0.05
        frame.loc[0, "independent_price_verified"] = True
        frame.loc[0, "independent_price_state"] = "VERIFIED_INDEPENDENT"
        frame.loc[0, "independent_source_family"] = "IDX_OFFICIAL"
        frame.loc[0, "independent_last_price"] = 625.0
        frame.loc[0, "independent_price_age_days"] = 0
        frame.loc[0, "independent_date_gap_days"] = 0
        frame.loc[0, "fundamental_coverage"] = 80.0
        frame.loc[0, "fundamental_score"] = 70.0
        frame.loc[0, "validation_gate_score"] = 80.0
        frame.loc[0, "validation_tier"] = "USABLE"
        frame.loc[0, "ohlcv_source_tier"] = "LIVE_YAHOO"
        frame.loc[0, "absolute_data_age_days"] = 0
        frame.loc[0, "current_bar_incomplete"] = False
        frame.loc[0, "market_regime"] = "RISK_ON"
        frame.loc[0, "adtv20_idr"] = 5_000_000_000.0
        frame.loc[0, "zero_volume_ratio20"] = 0.0
        frame.loc[0, "quote_critical_blocker"] = False
        for column in (
            "validation_confidence", "fundamental_confidence", "market_status_confidence",
            "news_confidence", "quote_confidence", "universe_confidence",
        ):
            frame.loc[0, column] = 100.0
        frame = attach_position_sizing(frame)
        cfg = ScanConfig().replace(execution_policy="ACCOUNT_GUARDED", autopilot_enabled=True)
        frame = apply_analyst_fusion_gate(frame, cfg)
        frame = enforce_analyst_portfolio_budget(frame, cfg)
        out = finalize_execution_integrity(frame, cfg)
        self.assertEqual(out.loc[0, "status"], "EXECUTION_READY")
        self.assertTrue(bool(out.loc[0, "autopilot_verified"]))
        self.assertEqual(out.loc[0, "order_instruction"], "BUY_LIT_USER_SIZE")
        self.assertEqual(float(out.loc[0, "stockbit_trigger_price"]), 635.0)
        self.assertEqual(float(out.loc[0, "stockbit_limit_price"]), 640.0)
        expected = size_stockbit_order(640.0, 610.0, cfg)
        self.assertEqual(int(out.loc[0, "stockbit_order_lots"]), expected["suggested_lots"])
        self.assertEqual(float(out.loc[0, "capital_required_idr"]), expected["capital_required_idr"])

    def test_wait_retest_is_plan_not_execution_ready(self):
        frame = AnalystFusionV440Tests().pullback_limit_candidate()
        frame.loc[0, "setup"] = "BREAKOUT_RETEST"
        frame.loc[0, "action"] = "WAIT_RETEST"
        frame.loc[0, "evidence"] = "Retest level breakout terdeteksi"
        frame.loc[0, "quality_score"] = 90.0
        frame = apply_analyst_fusion_gate(frame)
        out = finalize_execution_integrity(frame)
        self.assertEqual(out.loc[0, "status"], "ENTRY_PLAN_READY")
        self.assertEqual(out.loc[0, "order_instruction"], "WAIT_PRICE_AND_CONFIRM")
        self.assertTrue(pd.isna(out.loc[0, "stockbit_order_price"]))


class AutopilotGuardV510Tests(unittest.TestCase):
    @staticmethod
    def ready_limit_frame() -> pd.DataFrame:
        frame = AnalystFusionV440Tests().pullback_limit_candidate()
        values = {
            "action": "READY_LIMIT",
            "entry_type": "LIMIT_ON_PULLBACK_THEN_CONFIRM",
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
            "distance_atr": 0.10,
            "independent_price_verified": True,
            "independent_price_state": "VERIFIED_INDEPENDENT",
            "independent_source_family": "IDX_OFFICIAL",
            "independent_last_price": 1_020.0,
            "independent_price_age_days": 0,
            "independent_date_gap_days": 0,
            "fundamental_coverage": 80.0,
            "fundamental_score": 70.0,
            "validation_gate_score": 80.0,
            "validation_tier": "USABLE",
            "ohlcv_source_tier": "LIVE_YAHOO",
            "absolute_data_age_days": 0,
            "current_bar_incomplete": False,
            "market_regime": "RISK_ON",
            "adtv20_idr": 5_000_000_000.0,
            "zero_volume_ratio20": 0.0,
            "quote_critical_blocker": False,
        }
        for column, value in values.items():
            frame.loc[0, column] = value
        for column in (
            "validation_confidence", "fundamental_confidence", "market_status_confidence",
            "news_confidence", "quote_confidence", "universe_confidence",
        ):
            frame.loc[0, column] = 100.0
        frame = attach_position_sizing(frame)
        frame = apply_analyst_fusion_gate(frame)
        return frame

    def test_idx_reference_session_is_first_candidate(self):
        dates = scanner_module._candidate_idx_summary_dates("2026-07-16", 3)
        self.assertEqual(dates[0], pd.Timestamp("2026-07-16"))

    def test_specialty_module_can_be_imported_first(self):
        completed = subprocess.run(
            [sys.executable, "-c", "import scanner_specialty; import scanner"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_same_session_cache_survives_transient_provider_outage(self):
        cached = pd.DataFrame([{
            "ticker": "TEST.JK", "Date": pd.Timestamp("2026-07-16"),
            "Open": 1_000.0, "High": 1_030.0, "Low": 990.0, "Close": 1_020.0,
            "Volume": 5_000_000.0, "independent_source": "IDX_OFFICIAL_STOCK_SUMMARY_API",
            "independent_source_family": "IDX_OFFICIAL",
        }])
        empty = scanner_module._empty_automatic_independent_data()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"IDX_SCANNER_CACHE_DIR": directory}):
            scanner_module._write_independent_price_cache(cached)
            with (
                patch.object(scanner_module, "fetch_idx_official_eod_quotes", return_value=(empty, pd.DataFrame())),
                patch.object(scanner_module, "fetch_google_finance_quotes", return_value=(empty, pd.DataFrame())),
            ):
                data, report = fetch_automatic_independent_prices(
                    ["TEST.JK"],
                    reference_date="2026-07-16",
                    primary_reference={"TEST.JK": ("2026-07-16", 1_020.0)},
                    primary_source_tiers={"TEST.JK": "LIVE_YAHOO"},
                )
        self.assertEqual(data.loc[0, "independent_source_family"], "IDX_OFFICIAL")
        self.assertIn("INDEPENDENT_PRICE_CACHE", report["provider"].tolist())

    def test_same_family_cache_does_not_block_independent_google_fallback(self):
        cached = pd.DataFrame([{
            "ticker": "TEST.JK", "Date": pd.Timestamp("2026-07-16"),
            "Open": 1_000.0, "High": 1_030.0, "Low": 990.0, "Close": 1_020.0,
            "Volume": 5_000_000.0, "independent_source": "IDX_OFFICIAL_STOCK_SUMMARY_API",
            "independent_source_family": "IDX_OFFICIAL",
        }])
        google = pd.DataFrame([{
            "ticker": "TEST.JK", "Date": pd.Timestamp("2026-07-16 16:00"),
            "Open": np.nan, "High": np.nan, "Low": np.nan, "Close": 1_020.0,
            "Volume": np.nan, "independent_source": "GOOGLE_FINANCE_PUBLIC_QUOTE",
            "independent_source_family": "GOOGLE_FINANCE",
        }])
        empty = scanner_module._empty_automatic_independent_data()
        with (
            patch.object(scanner_module, "_load_independent_price_cache", return_value=cached),
            patch.object(scanner_module, "fetch_idx_official_eod_quotes", return_value=(empty, pd.DataFrame())),
            patch.object(scanner_module, "fetch_google_finance_quotes", return_value=(google, pd.DataFrame())) as google_mock,
        ):
            data, _ = fetch_automatic_independent_prices(
                ["TEST.JK"],
                reference_date="2026-07-16",
                primary_reference={"TEST.JK": ("2026-07-16", 1_020.0)},
                primary_source_tiers={"TEST.JK": "IDX_OFFICIAL_HTTP"},
            )
        google_mock.assert_called_once()
        self.assertIn("GOOGLE_FINANCE", set(data["independent_source_family"]))

    def test_full_gate_limit_order_becomes_autopilot_verified(self):
        cfg = ScanConfig().replace(execution_policy="ACCOUNT_GUARDED", autopilot_enabled=True)
        frame = enforce_analyst_portfolio_budget(self.ready_limit_frame(), cfg)
        out = finalize_execution_integrity(frame, cfg)
        self.assertEqual(out.loc[0, "status"], "EXECUTION_READY")
        self.assertTrue(bool(out.loc[0, "autopilot_verified"]))
        self.assertEqual(out.loc[0, "autopilot_score"], 100.0)
        self.assertEqual(out.loc[0, "order_instruction"], "BUY_LIMIT_USER_SIZE")
        self.assertGreaterEqual(int(out.loc[0, "stockbit_order_lots"]), 1)
        self.assertEqual(out.loc[0, "stockbit_order_template"], "BRACKET_ORDER_LIMIT")
        self.assertEqual(out.loc[0, "stockbit_time_in_force"], "GFD")
        self.assertEqual(out.loc[0, "broker_submission_mode"], "MANUAL_STOCKBIT")
        self.assertTrue(bool(out.loc[0, "requires_stockbit_price_check"]))
        self.assertTrue(bool(out.loc[0, "opening_gap_recheck_required"]))

    def test_missing_independent_price_never_emits_order(self):
        frame = self.ready_limit_frame()
        frame["independent_price_verified"] = False
        frame["independent_price_state"] = "MISSING_INDEPENDENT"
        cfg = ScanConfig().replace(execution_policy="ACCOUNT_GUARDED", autopilot_enabled=True)
        frame = enforce_analyst_portfolio_budget(frame, cfg)
        out = finalize_execution_integrity(frame, cfg)
        self.assertEqual(out.loc[0, "status"], "READY_FOR_PRICE_VERIFY")
        self.assertFalse(bool(out.loc[0, "autopilot_verified"]))
        self.assertEqual(out.loc[0, "order_instruction"], "WAIT_PRICE_AND_CONFIRM")
        self.assertEqual(int(out.loc[0, "stockbit_order_lots"]), 0)

    def test_risk_off_is_a_hard_autopilot_block_not_a_hidden_filter(self):
        frame = self.ready_limit_frame()
        frame["market_regime"] = "RISK_OFF"
        cfg = ScanConfig().replace(execution_policy="ACCOUNT_GUARDED", autopilot_enabled=True)
        frame = enforce_analyst_portfolio_budget(frame, cfg)
        out = finalize_execution_integrity(frame, cfg)
        self.assertEqual(out.loc[0, "status"], "BLOCKED_CONTEXT")
        self.assertIn("MARKET_REGIME", out.loc[0, "autopilot_blockers"])
        self.assertEqual(out.loc[0, "order_instruction"], "DO_NOT_BUY")

    def test_two_setups_remain_visible_but_only_one_reserves_order(self):
        first = self.ready_limit_frame()
        second = first.copy()
        second["setup"] = "BREAKOUT_RETEST"
        second["analyst_fusion_score"] = pd.to_numeric(second["analyst_fusion_score"]) - 1.0
        frame = pd.concat([first, second], ignore_index=True)
        cfg = ScanConfig().replace(execution_policy="ACCOUNT_GUARDED", autopilot_enabled=True)
        frame = enforce_analyst_portfolio_budget(frame, cfg)
        out = finalize_execution_integrity(frame, cfg)
        self.assertEqual(len(out), 2)
        self.assertEqual(int(out["autopilot_verified"].sum()), 1)
        self.assertEqual(set(out["confluence_setup_count"]), {2})
        self.assertIn("READY_NOT_SELECTED", out["status"].tolist())

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
        self.assertIn(result.loc[0, "ara_hunter_status"], {"ARA_CONTINUATION_SIGNAL_READY", "ARA_CONTINUATION_CANDIDATE", "ARA_CONFIRMED_ONLY"})
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
        self.assertEqual(out.loc[0, "bpjs_status"], "BPJS_SIGNAL_READY")
        self.assertFalse(bool(out.loc[0, "account_risk_gate_applied"]))
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
        self.assertEqual(out.loc[0, "bsjp_status"], "BSJP_SIGNAL_READY")
        self.assertFalse(bool(out.loc[0, "account_risk_gate_applied"]))
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
        self.assertEqual(final.loc[0, "status"], "ENTRY_PLAN_READY")

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


class CompoundingDailyRadarV530Tests(unittest.TestCase):
    def test_live_ara_uses_last_completed_close_as_current_session_reference(self):
        frame = SpecialtyScannerV42Tests._strong_daily_frame()
        intraday = SpecialtyScannerV42Tests._intraday("bpjs")
        out = scan_ara_hunter_candidates(
            {"TEST.JK": frame}, {"TEST.JK": intraday},
            now="2026-07-13 10:30:00", market_context=MarketContext(regime="RISK_ON"),
        )
        completed_close = float(frame.iloc[-1]["Close"])
        self.assertEqual(float(out.loc[0, "price_band_reference"]), completed_close)
        self.assertEqual(float(out.loc[0, "ara_price"]), float(idx_daily_price_band(completed_close)[1]))
        self.assertEqual(out.loc[0, "live_price_source"], "INTRADAY_COMPLETED_BAR")
        self.assertNotEqual(float(out.loc[0, "last_price"]), completed_close)

    def test_invalid_ara_targets_can_never_remain_signal_or_order_ready(self):
        frame = SpecialtyScannerV42Tests._strong_daily_frame()
        invalid_targets = {
            "tp1": np.nan, "tp2": np.nan, "tp1_basis": "UNAVAILABLE",
            "tp2_basis": "UNAVAILABLE", "target_structure": "",
            "target_structure_valid": False,
        }
        with patch("scanner_specialty.price_structure_target_pair", return_value=invalid_targets):
            out = scan_ara_hunter_candidates({"TEST.JK": frame}, now="2026-07-13 20:00:00")
        self.assertFalse(bool(out.loc[0, "target_structure_valid"]))
        self.assertFalse(bool(out.loc[0, "signal_ready"]))
        self.assertFalse(bool(out.loc[0, "specialty_order_ready"]))
        self.assertNotIn(out.loc[0, "ara_hunter_status"], {"PRE_ARA_SIGNAL_READY", "PRE_ARA_ORDER_READY"})
        self.assertIn("PRICE_STRUCTURE_TARGETS_INVALID", out.loc[0, "specialty_execution_blockers"])

    def test_bpjs_and_bsjp_publish_daily_radar_outside_execution_windows(self):
        frame = SpecialtyScannerV42Tests._strong_daily_frame()
        bpjs = scan_bpjs_candidates({"TEST.JK": frame}, now="2026-07-19 12:00:00")
        bsjp = scan_bsjp_candidates({"TEST.JK": frame}, now="2026-07-19 12:00:00")
        self.assertEqual(bpjs.loc[0, "bpjs_status"], "BPJS_DAILY_RADAR")
        self.assertEqual(bsjp.loc[0, "bsjp_status"], "BSJP_DAILY_RADAR")
        self.assertEqual(bpjs.loc[0, "specialty_order_state"], "DAILY_RADAR")
        self.assertEqual(bsjp.loc[0, "specialty_order_state"], "DAILY_RADAR")

    def test_shared_budget_promotes_risk_on_bsjp_to_one_order_ticket(self):
        frame = SpecialtyScannerV42Tests._strong_daily_frame()
        cfg = ScanConfig().replace(execution_policy="ACCOUNT_GUARDED", autopilot_enabled=True)
        screens = build_specialty_screens(
            {"TEST.JK": frame},
            intraday={"TEST.JK": SpecialtyScannerV42Tests._intraday("bsjp")},
            market_context=MarketContext(regime="RISK_ON"),
            now="2026-07-13 15:30:00", config=cfg,
        )
        row = screens["bsjp"].iloc[0]
        self.assertEqual(row["bsjp_status"], "BSJP_ORDER_READY")
        self.assertTrue(bool(row["specialty_order_ready"]))
        self.assertEqual(row["stockbit_order_template"], "BRACKET_ORDER_LIMIT")
        self.assertEqual(row["stockbit_time_in_force"], "GFD")
        self.assertGreaterEqual(int(row["stockbit_order_lots"]), 1)

    def test_risk_off_never_promotes_specialty_order(self):
        frame = SpecialtyScannerV42Tests._strong_daily_frame()
        cfg = ScanConfig().replace(execution_policy="ACCOUNT_GUARDED", autopilot_enabled=True)
        screens = build_specialty_screens(
            {"TEST.JK": frame},
            intraday={"TEST.JK": SpecialtyScannerV42Tests._intraday("bsjp")},
            market_context=MarketContext(regime="RISK_OFF"),
            now="2026-07-13 15:30:00", config=cfg,
        )
        self.assertFalse(bool(screens["bsjp"].loc[0, "specialty_order_ready"]))
        self.assertIn("MARKET_REGIME_NOT_RISK_ON", screens["bsjp"].loc[0, "specialty_execution_blockers"])

    def test_daily_board_separates_radar_from_order_ready(self):
        screens = {
            "bpjs": pd.DataFrame([{"ticker": "A.JK", "bpjs_status": "BPJS_DAILY_RADAR", "bpjs_score": 70.0, "specialty_order_state": "DAILY_RADAR"}]),
            "bsjp": pd.DataFrame([{"ticker": "B.JK", "bsjp_status": "BSJP_ORDER_READY", "bsjp_score": 82.0, "specialty_order_state": "ORDER_READY", "specialty_order_ready": True, "stockbit_order_lots": 2}]),
        }
        board = build_daily_opportunity_board(screens)
        self.assertEqual(board.iloc[0]["decision_state"], "ORDER_READY")
        self.assertTrue(bool(board.iloc[0]["order_ready"]))
        self.assertIn("DAILY_RADAR", board["decision_state"].tolist())


class AutomaticSourceQuorumV550Tests(unittest.TestCase):
    @staticmethod
    def _instance_zip() -> bytes:
        xml = b'''<?xml version="1.0" encoding="UTF-8"?>
<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance"
             xmlns:iso4217="http://www.xbrl.org/2003/iso4217"
             xmlns:idx="http://www.idx.co.id/xbrl/taxonomy/2020">
  <xbrli:context id="CurrentYearDuration">
    <xbrli:entity><xbrli:identifier scheme="IDX">TEST</xbrli:identifier></xbrli:entity>
    <xbrli:period><xbrli:startDate>2026-01-01</xbrli:startDate><xbrli:endDate>2026-03-31</xbrli:endDate></xbrli:period>
  </xbrli:context>
  <xbrli:context id="CurrentYearInstant">
    <xbrli:entity><xbrli:identifier scheme="IDX">TEST</xbrli:identifier></xbrli:entity>
    <xbrli:period><xbrli:instant>2026-03-31</xbrli:instant></xbrli:period>
  </xbrli:context>
  <xbrli:unit id="IDR"><xbrli:measure>iso4217:IDR</xbrli:measure></xbrli:unit>
  <idx:Revenue contextRef="CurrentYearDuration" unitRef="IDR">1250000000</idx:Revenue>
  <idx:ProfitLossAttributableToOwnersOfParentEntity contextRef="CurrentYearDuration" unitRef="IDR">175000000</idx:ProfitLossAttributableToOwnersOfParentEntity>
  <idx:CashFlowsFromUsedInOperatingActivities contextRef="CurrentYearDuration" unitRef="IDR">210000000</idx:CashFlowsFromUsedInOperatingActivities>
  <idx:Assets contextRef="CurrentYearInstant" unitRef="IDR">5000000000</idx:Assets>
  <idx:Liabilities contextRef="CurrentYearInstant" unitRef="IDR">2000000000</idx:Liabilities>
  <idx:Equity contextRef="CurrentYearInstant" unitRef="IDR">3000000000</idx:Equity>
</xbrli:xbrl>'''
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("TEST-2026-Q1.xbrl", xml)
        return buffer.getvalue()

    def test_official_xbrl_zip_is_parsed_without_manual_csv(self):
        out = parse_idx_xbrl_attachment(
            self._instance_zip(), ticker="TEST.JK", period_end="2026-03-31",
            period_type="Q1",
            source_url="https://www.idx.co.id/StaticData/TEST/@instance.zip",
            filename="@instance.zip",
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out.loc[0, "source_family"], "IDX_OFFICIAL_XBRL")
        self.assertTrue(bool(out.loc[0, "source_verified"]))
        self.assertEqual(float(out.loc[0, "revenue"]), 1_250_000_000.0)
        self.assertEqual(float(out.loc[0, "net_income"]), 175_000_000.0)
        self.assertEqual(float(out.loc[0, "total_assets"]), 5_000_000_000.0)
        self.assertEqual(out.loc[0, "currency"], "IDR")

    def test_automatic_idx_xbrl_is_recognized_as_verified_official_source(self):
        idx_history = parse_idx_xbrl_attachment(
            self._instance_zip(), ticker="TEST.JK", period_end="2026-03-31",
            period_type="Q1",
            source_url="https://www.idx.co.id/StaticData/TEST/@instance.zip",
            filename="@instance.zip",
        )
        yahoo_history = idx_history.copy()
        yahoo_history["source_family"] = "YAHOO"
        yahoo_history["source_name"] = "Yahoo Finance statements via yfinance"
        yahoo_history["source_url"] = ""
        yahoo_history["source_verified"] = False
        features = build_fundamental_history_features(
            combine_fundamental_history(idx_history, yahoo_history),
            now="2026-04-30",
        )
        self.assertTrue(bool(features.loc[0, "fundamental_official_reference"]))
        self.assertTrue(bool(features.loc[0, "fundamental_official_verified"]))
        self.assertEqual(int(features.loc[0, "fundamental_source_count"]), 2)

    def test_xbrl_parser_rejects_non_idx_transport(self):
        with self.assertRaises(ValueError):
            parse_idx_xbrl_attachment(
                self._instance_zip(), ticker="TEST.JK", period_end="2026-03-31",
                period_type="Q1", source_url="https://example.com/instance.zip",
            )

    def test_idx_manifest_is_bounded_and_accepts_only_xbrl_attachments(self):
        class Response:
            status_code = 200
            url = "https://www.idx.co.id/primary/ListedCompany/GetFinancialReport"

            @staticmethod
            def json():
                return {
                    "ResultCount": 1,
                    "Results": [{
                        "KodeEmiten": "TEST",
                        "Attachments": [
                            {"File_Name": "@instance.zip", "File_Path": "/StaticData/TEST/@instance.zip"},
                            {"File_Name": "signed.pdf", "File_Path": "/StaticData/TEST/signed.pdf"},
                        ],
                    }],
                }

        calls = []

        def fake_get(url, **kwargs):
            calls.append((url, kwargs))
            return Response()

        manifest, report = scanner_module._idx_manifest_rows(
            ["TEST.JK"], [2025], request_get=fake_get,
        )
        self.assertEqual(len(calls), 5)
        self.assertEqual(len(manifest), 4)
        self.assertTrue(manifest["attachment_url"].str.startswith("https://www.idx.co.id/").all())
        self.assertTrue(manifest["filename"].eq("@instance.zip").all())
        self.assertTrue(report["status"].eq("OK").all())

    def test_idx_history_fetch_downloads_and_caches_official_filing(self):
        manifest = pd.DataFrame([{
            "ticker": "TEST.JK", "period_end": pd.Timestamp("2026-03-31"),
            "period_type": "Q1", "period_code": "tw1", "year": 2026,
            "filename": "@instance.zip",
            "attachment_url": "https://www.idx.co.id/StaticData/TEST/@instance.zip",
            "attachment_rank": 0,
        }])

        class Response:
            status_code = 200
            url = "https://www.idx.co.id/StaticData/TEST/@instance.zip"
            content = AutomaticSourceQuorumV550Tests._instance_zip()

        with patch.object(scanner_module, "_idx_manifest_rows", return_value=(manifest, pd.DataFrame())), patch.object(
            scanner_module, "_load_cache", return_value=pd.DataFrame()
        ), patch.object(scanner_module, "_write_cache") as write_cache:
            history, report = scanner_module.fetch_idx_fundamental_history(
                ["TEST.JK"], max_tickers=1, request_get=lambda *args, **kwargs: Response(),
            )
        self.assertEqual(len(history), 1)
        self.assertEqual(history.loc[0, "source_family"], "IDX_OFFICIAL_XBRL")
        self.assertEqual(report.loc[report["ticker"].eq("TEST.JK"), "status"].iloc[0], "OK")
        write_cache.assert_called_once()

    def test_source_quorum_does_not_confuse_fallback_with_consensus(self):
        history = pd.DataFrame([
            {"ticker": "AAA.JK", "period_end": "2026-03-31", "period_type": "Q1", "source_family": "YAHOO", "revenue": 100.0},
            {"ticker": "AAA.JK", "period_end": "2026-03-31", "period_type": "Q1", "source_family": "IDX_OFFICIAL_XBRL", "source_verified": True, "revenue": 101.0},
            {"ticker": "BBB.JK", "period_end": "2026-03-31", "period_type": "Q1", "source_family": "YAHOO", "revenue": 200.0},
            {"ticker": "BBB.JK", "period_end": "2026-03-31", "period_type": "Q1", "source_family": "TWELVE_DATA", "revenue": 201.0},
        ])
        prices = pd.DataFrame([
            {"ticker": "AAA.JK", "independent_price_verified": True, "independent_overlap_bars": 8, "independent_return_correlation": 0.99},
            {"ticker": "BBB.JK", "independent_price_verified": False, "independent_overlap_bars": 0, "independent_return_correlation": np.nan},
        ])
        status = pd.DataFrame([
            {"ticker": "AAA.JK", "market_status_verified": True},
            {"ticker": "BBB.JK", "market_status_verified": True},
        ])
        news = pd.DataFrame([
            {"ticker": "AAA.JK", "provider_query_ok": True, "idx_disclosure_query_ok": True},
            {"ticker": "BBB.JK", "provider_query_ok": True, "idx_disclosure_query_ok": False},
        ])
        audit = build_source_quorum_audit(
            ["AAA.JK", "BBB.JK"],
            source_tiers={"AAA.JK": "LIVE_YAHOO", "BBB.JK": "LIVE_ITICK_FREE_FALLBACK"},
            price_validation=prices, fundamental_history=history,
            market_status=status, news_review=news,
        ).set_index("data_layer")
        self.assertEqual(int(audit.loc["Execution price / last close", "verified_tickers"]), 1)
        self.assertEqual(int(audit.loc["Technical daily OHLCV", "verified_tickers"]), 1)
        self.assertEqual(int(audit.loc["Fundamental statements", "verified_tickers"]), 1)
        self.assertEqual(audit.loc["IDX restrictions / suspension / FCA", "quorum_state"], "AUTHORITATIVE_VERIFIED")
        self.assertEqual(audit.loc["Intraday 5m", "quorum_state"], "SINGLE_PROVIDER_OR_MANUAL")

class SafeSignalSemanticsV561Tests(unittest.TestCase):
    @staticmethod
    def strong_fundamentals(**changes: object) -> pd.DataFrame:
        row = {
            "ticker": "TEST.JK", "fundamental_coverage": 92.0,
            "fundamental_score": 91.0, "fundamental_score_10": 9.1,
            "fundamental_data_grade": "A", "fundamental_source_count": 2,
            "fundamental_source_families": "YAHOO • IDX_OFFICIAL_XBRL",
            "fundamental_official_reference": True,
            "fundamental_official_verified": True,
            "fundamental_history_quarters": 8, "fundamental_history_years": 3,
            "fundamental_history_coverage": 90.0, "fundamental_consensus_score": 94.0,
            "fundamental_conflicts": "", "fundamental_reliability": "HIGH",
            "statement_age_days": 30,
            "revenue_growth": 0.30, "earnings_growth": 0.40,
            "roe": 0.25, "roa": 0.10, "net_margin": 0.15,
            "operating_margin": 0.20, "debt_equity": 0.40,
            "current_ratio": 2.0, "cash_to_debt": 1.0,
            "operating_cash_flow": 1_000_000_000.0,
            "free_cash_flow": 800_000_000.0, "peg_ratio": 1.0,
            "fcf_yield": 0.05, "market_cap": 5_000_000_000_000.0,
            "history_cash_conversion": 1.1, "history_positive_ocf_ratio": 1.0,
            "history_positive_earnings_ratio": 1.0, "history_margin_stability": 0.9,
            "history_share_dilution_yoy": 0.0, "history_roic_proxy": 0.20,
            "history_net_debt_ebitda": 0.5, "history_interest_coverage": 10.0,
            "fundamental_model": "GENERAL", "fundamental_red_flags": "",
        }
        row.update(changes)
        return pd.DataFrame([row])

    def test_unknown_statement_age_cannot_receive_multibagger_a(self):
        frame = SpecialtyScannerV42Tests._strong_daily_frame()
        result = scan_multibagger_candidates(
            {"TEST.JK": frame}, self.strong_fundamentals(statement_age_days=np.nan)
        )
        self.assertFalse(bool(result.loc[0, "statement_current"]))
        self.assertEqual(result.loc[0, "statement_age_state"], "UNKNOWN")
        self.assertNotEqual(result.loc[0, "multibagger_status"], "MULTIBAGGER_A_CANDIDATE")

    def test_negative_peg_receives_no_valuation_points(self):
        frame = SpecialtyScannerV42Tests._strong_daily_frame()
        result = scan_multibagger_candidates(
            {"TEST.JK": frame}, self.strong_fundamentals(peg_ratio=-3.0)
        )
        self.assertFalse(bool(result.loc[0, "peg_valid_for_valuation"]))
        self.assertEqual(float(result.loc[0, "valuation_score"]), 4.0)  # FCF yield only

    def test_multibagger_a_requires_verified_automatic_idx_xbrl(self):
        frame = SpecialtyScannerV42Tests._strong_daily_frame()
        result = scan_multibagger_candidates(
            {"TEST.JK": frame},
            self.strong_fundamentals(
                fundamental_source_families="YAHOO • TWELVE_DATA",
                fundamental_official_reference=False,
                fundamental_official_verified=False,
            ),
        )
        self.assertFalse(bool(result.loc[0, "grade_a_gate"]))
        self.assertNotEqual(result.loc[0, "multibagger_status"], "MULTIBAGGER_A_CANDIDATE")

    def test_signal_first_missing_price_remains_signal_not_execution_ready(self):
        cfg = ScanConfig().replace(
            execution_policy="SIGNAL_FIRST", autopilot_enabled=False,
            account_size_idr=0.0, cash_on_hand_idr=0.0,
        )
        frame = AutopilotGuardV510Tests.ready_limit_frame()
        frame["independent_price_verified"] = False
        frame["independent_price_state"] = "MISSING_INDEPENDENT"
        frame = enforce_analyst_portfolio_budget(frame, cfg, cash_on_hand_idr=0.0)
        out = finalize_execution_integrity(frame, cfg)
        self.assertEqual(out.loc[0, "status"], "SIGNAL_READY")
        self.assertEqual(out.loc[0, "setup_state"], "SETUP_READY")
        self.assertEqual(out.loc[0, "account_order_state"], "USER_MANAGED")
        self.assertFalse(bool(out.loc[0, "account_risk_gate_applied"]))
        self.assertFalse(bool(out.loc[0, "autopilot_verified"]))
        self.assertFalse(bool(out.loc[0, "manual_execution_candidate"]))
        self.assertEqual(out.loc[0, "order_instruction"], "DO_NOT_BUY_YET")
        self.assertIn("INDEPENDENT_PRICE_REQUIRED", out.loc[0, "signal_execution_blockers"])
        self.assertEqual(int(out.loc[0, "stockbit_order_lots"]), 0)

    def test_signal_first_fully_safe_row_requires_final_stockbit_verify(self):
        cfg = ScanConfig().replace(
            execution_policy="SIGNAL_FIRST", autopilot_enabled=False,
            account_size_idr=0.0, cash_on_hand_idr=0.0,
        )
        frame = AutopilotGuardV510Tests.ready_limit_frame()
        frame = enforce_analyst_portfolio_budget(frame, cfg, cash_on_hand_idr=0.0)
        out = finalize_execution_integrity(frame, cfg)
        self.assertEqual(out.loc[0, "status"], "READY_FOR_STOCKBIT_VERIFY")
        self.assertTrue(bool(out.loc[0, "manual_execution_candidate"]))
        self.assertFalse(bool(out.loc[0, "autopilot_verified"]))
        self.assertFalse(bool(out.loc[0, "strict_execution_ready"]))
        self.assertEqual(out.loc[0, "proposed_order_instruction"], "BUY_LIMIT_USER_SIZE")
        self.assertEqual(out.loc[0, "order_instruction"], "VERIFY_STOCKBIT_THEN_BUY_LIMIT_USER_SIZE")
        self.assertEqual(out.loc[0, "primary_execution_blocker"], "STOCKBIT_BROKER_REVALIDATION_REQUIRED")

    def test_signal_first_critical_fundamental_or_news_never_becomes_manual_ready(self):
        cfg = ScanConfig().replace(execution_policy="SIGNAL_FIRST", autopilot_enabled=False)
        for flag, expected in (
            ("fundamental_critical_blocker", "CRITICAL_CONTEXT"),
            ("news_critical_blocker", "CRITICAL_CONTEXT"),
        ):
            with self.subTest(flag=flag):
                frame = AutopilotGuardV510Tests.ready_limit_frame()
                frame[flag] = True
                frame = enforce_analyst_portfolio_budget(frame, cfg)
                out = finalize_execution_integrity(frame, cfg)
                self.assertEqual(out.loc[0, "status"], "SIGNAL_READY")
                self.assertFalse(bool(out.loc[0, "manual_execution_candidate"]))
                self.assertIn(expected, out.loc[0, "signal_execution_blockers"])
                self.assertEqual(out.loc[0, "order_instruction"], "DO_NOT_BUY_YET")

    def test_signal_first_risk_off_low_rr_and_low_liquidity_stays_radar(self):
        cfg = ScanConfig().replace(execution_policy="SIGNAL_FIRST", autopilot_enabled=False)
        frame = AutopilotGuardV510Tests.ready_limit_frame()
        frame["market_regime"] = "RISK_OFF"
        frame["rr1"] = 0.4
        frame["rr2"] = 0.8
        frame["adtv20_idr"] = 100_000_000.0
        frame["stop_pct"] = 0.12
        frame = enforce_analyst_portfolio_budget(frame, cfg)
        out = finalize_execution_integrity(frame, cfg)
        self.assertEqual(out.loc[0, "status"], "SIGNAL_READY")
        self.assertFalse(bool(out.loc[0, "manual_execution_candidate"]))
        self.assertIn("RISK_LEVELS", out.loc[0, "signal_execution_blockers"])
        self.assertIn("LIQUIDITY", out.loc[0, "signal_execution_blockers"])
        self.assertIn("MARKET_REGIME", out.loc[0, "signal_execution_blockers"])

    def test_signal_first_specialty_keeps_setup_and_moves_risk_to_warnings(self):
        cfg = ScanConfig().replace(
            execution_policy="SIGNAL_FIRST", autopilot_enabled=False,
            account_size_idr=0.0, cash_on_hand_idr=0.0,
        )
        frame = SpecialtyScannerV42Tests._strong_daily_frame()
        result = scan_bsjp_candidates(
            {"TEST.JK": frame}, {"TEST.JK": SpecialtyScannerV42Tests._intraday("bsjp")},
            now="2026-07-13 15:30:00", market_context=MarketContext(regime="RISK_OFF"),
            config=cfg,
        )
        self.assertEqual(result.loc[0, "bsjp_status"], "BSJP_SIGNAL_READY")
        self.assertTrue(bool(result.loc[0, "setup_ready"]))
        self.assertEqual(result.loc[0, "setup_state"], "SETUP_READY")
        self.assertEqual(result.loc[0, "specialty_order_state"], "USER_MANAGED")
        self.assertIn("MARKET_REGIME_NOT_RISK_ON", result.loc[0, "specialty_risk_warnings"])
        self.assertIn("ACCOUNT_SIZE_CANNOT_SUPPORT_ONE_LOT", result.loc[0, "specialty_account_warnings"])
        self.assertEqual(result.loc[0, "specialty_execution_blockers"], "")

    def test_scanengine_has_no_runtime_monkey_patch_assignments(self):
        tree = ast.parse((ROOT / "scanner.py").read_text(encoding="utf-8"))
        offenders: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "ScanEngine":
                        offenders.append(target.attr)
        self.assertEqual(offenders, [])

    def test_release_defaults_and_dashboard_regression_contract(self):
        cfg = ScanConfig()
        self.assertEqual(cfg.execution_policy, "SIGNAL_FIRST")
        self.assertEqual(cfg.fundamental_cache_days, 21)
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn('["SIGNAL_FIRST", "ACCOUNT_GUARDED"]', app_source)
        self.assertIn("@st.fragment", app_source)
        self.assertIn("st.session_state", app_source)

class StructuredProviderErrorsV560Tests(unittest.TestCase):
    def test_provider_error_classifier_distinguishes_operational_failures(self):
        self.assertEqual(scanner_module.classify_provider_error(TimeoutError("read timed out")), "PROVIDER_TIMEOUT")
        self.assertEqual(scanner_module.classify_provider_error(RuntimeError("HTTP 429 too many requests")), "PROVIDER_RATE_LIMIT")
        self.assertEqual(scanner_module.classify_provider_error(KeyError("unexpected")), "PROGRAMMING_ERROR")

    def test_specialty_imports_public_core_contract_only(self):
        tree = ast.parse((ROOT / "scanner_specialty.py").read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "scanner":
                imported.extend(alias.name for alias in node.names)
        private = [name for name in imported if name.startswith("_")]
        self.assertEqual(private, [])


class MultibaggerCapitalAllocatorV570Tests(unittest.TestCase):
    def _candidate(self, ticker: str, conviction_inputs: dict[str, object] | None=None) -> dict[str, object]:
        row: dict[str, object] = {
            'ticker': ticker,
            'multibagger_status': 'MULTIBAGGER_A_CANDIDATE',
            'compounding_state': 'ACCUMULATE_NOW',
            'growth_score': 20.0,
            'profitability_score': 16.0,
            'earnings_quality_score': 17.0,
            'balance_sheet_score': 11.0,
            'valuation_score': 6.0,
            'momentum_score': 9.0,
            'accumulation_score': 9.0,
            'fundamental_data_grade': 'A',
            'fundamental_reliability': 'HIGH',
            'fundamental_consensus_score': 92.0,
            'fundamental_history_coverage': 90.0,
            'fundamental_official_verified': True,
            'fundamental_official_reference': True,
            'fundamental_source_count': 3,
            'solvency_coverage': 100.0,
            'fundamental_model': 'GENERAL',
            'technical_entry_state': 'READY_FOR_STOCKBIT_VERIFY',
            'fundamental_conflicts': '',
            'red_flags': '',
            'entry': 1000.0,
            'last_price': 995.0,
            'multibagger_score': 90.0,
        }
        if conviction_inputs:
            row.update(conviction_inputs)
        return row

    def test_highest_conviction_receives_largest_weight(self):
        frame = pd.DataFrame([
            self._candidate('BEST.JK'),
            self._candidate('SECOND.JK', {'growth_score': 15.0, 'earnings_quality_score': 13.0, 'accumulation_score': 6.0}),
            self._candidate('THIRD.JK', {'growth_score': 12.0, 'profitability_score': 12.0, 'valuation_score': 4.0}),
        ])
        cfg = ScanConfig().replace(multibagger_capital_budget_idr=10_000_000.0, multibagger_max_holdings=3)
        result = allocate_multibagger_capital(frame, cfg)
        ranked = result.sort_values('capital_priority_rank')
        self.assertEqual(ranked.iloc[0]['ticker'], 'BEST.JK')
        self.assertGreater(float(ranked.iloc[0]['strategic_target_weight_pct']), float(ranked.iloc[-1]['strategic_target_weight_pct']))
        self.assertGreater(float(ranked.iloc[0]['recommended_allocation_idr']), float(ranked.iloc[-1]['recommended_allocation_idr']))

    def test_waiting_entry_keeps_target_weight_as_cash_reserve(self):
        frame = pd.DataFrame([
            self._candidate('WAIT.JK', {'compounding_state': 'WAIT_ACCUMULATION_ZONE'}),
            self._candidate('READY.JK', {'growth_score': 15.0, 'earnings_quality_score': 14.0}),
        ])
        cfg = ScanConfig().replace(multibagger_capital_budget_idr=5_000_000.0, multibagger_max_holdings=2)
        result = allocate_multibagger_capital(frame, cfg)
        wait = result.loc[result['ticker'].eq('WAIT.JK')].iloc[0]
        ready = result.loc[result['ticker'].eq('READY.JK')].iloc[0]
        self.assertGreater(float(wait['strategic_target_weight_pct']), 0.0)
        self.assertGreater(float(wait['strategic_target_amount_idr']), 0.0)
        self.assertEqual(float(wait['deploy_now_weight_pct']), 0.0)
        self.assertGreater(float(ready['deploy_now_weight_pct']), 0.0)
        self.assertGreater(float(result['multibagger_cash_reserve_idr'].iloc[0]), 0.0)

    def test_per_name_cap_prevents_single_stock_all_in(self):
        frame = pd.DataFrame([self._candidate('ONLY.JK')])
        cfg = ScanConfig().replace(multibagger_capital_budget_idr=10_000_000.0, multibagger_core_cap_pct=0.35)
        result = allocate_multibagger_capital(frame, cfg)
        self.assertLessEqual(float(result.loc[0, 'strategic_target_weight_pct']), 35.0)
        self.assertGreaterEqual(float(result.loc[0, 'multibagger_cash_reserve_idr']), 6_500_000.0)

    def test_red_flags_receive_no_allocation(self):
        frame = pd.DataFrame([self._candidate('RISK.JK', {'red_flags': 'OCF negatif'})])
        cfg = ScanConfig().replace(multibagger_capital_budget_idr=10_000_000.0)
        result = allocate_multibagger_capital(frame, cfg)
        self.assertFalse(bool(result.loc[0, 'allocation_eligible']))
        self.assertEqual(float(result.loc[0, 'strategic_target_weight_pct']), 0.0)
        self.assertEqual(float(result.loc[0, 'recommended_allocation_idr']), 0.0)


class CrossStrategyForwardQualityV590Tests(unittest.TestCase):
    def _core_row(self, ticker: str, setup: str, action: str, score: float) -> dict[str, object]:
        return {
            'ticker': ticker,
            'setup': setup,
            'status': 'SIGNAL_READY' if action == 'READY_TRIGGER' else 'ENTRY_PLAN_READY',
            'action': action,
            'quality_score': score,
            'analyst_fusion_score': score,
            'structural_quality_score': score,
            'confirmation_quality_score': score - 3,
            'supply_demand_score': score - 2,
            'failure_risk_score': 5.0,
            'distance_atr': 0.2,
            'extension_atr': 0.0,
            'silent_accumulation_score': 75.0,
            'cmf20': 0.08,
            'volume_ratio': 1.4,
            'adtv20_idr': 5_000_000_000.0,
            'rr1': 1.7,
            'rr2': 2.8,
            'target_structure_valid': True,
            'data_completeness_score': 90.0,
            'execution_confidence_score': 88.0,
            'validation_gate_score': 65.0,
            'entry': 1000.0,
            'trigger': 1010.0,
            'stop_loss': 950.0,
            'tp1': 1085.0,
            'tp2': 1140.0,
            'order_instruction': 'WAIT_PRICE_AND_CONFIRM',
            'signal_risk_warnings': '',
            'blockers': '',
        }

    def test_profit_builder_includes_breakout_and_reversal(self):
        core = pd.DataFrame([
            self._core_row('PULL.JK', 'PULLBACK_CONTINUATION', 'WAIT_PULLBACK_CONFIRMATION', 79.0),
            self._core_row('BREAK.JK', 'BREAKOUT_RETEST', 'READY_TRIGGER', 88.0),
            self._core_row('REV.JK', 'REVERSAL_ACCUMULATION', 'WAIT_HIGHER_LOW_AND_FLOW', 82.0),
        ])
        cfg = ScanConfig().replace(profit_conviction_min_score=50.0)
        result = build_profit_order_builder(core, {}, cfg)
        self.assertIn('BREAKOUT_RETEST', set(result['strategy']))
        self.assertIn('REVERSAL_ACCUMULATION', set(result['strategy']))
        self.assertIn('PULLBACK_CONTINUATION', set(result['strategy']))
        audit = result.attrs.get('strategy_audit')
        self.assertIsInstance(audit, pd.DataFrame)
        breakout = audit.loc[audit['strategy'].eq('BREAKOUT_RETEST')].iloc[0]
        self.assertEqual(int(breakout['included_primary']), 1)

    def test_project_management_parser_and_forward_fields(self):
        csv_data = io.BytesIO(
            b'ticker,as_of,source_verified,project_name,project_stage,project_completion_pct,funding_secured_pct,offtake_secured_pct,project_capex_idr,ceo_name,ceo_tenure_years,management_revenue_cagr,management_roic_change_pct,capital_allocation_score,governance_score,audit_clean,management_verified\n'
            b'ANTM,2026-07-01,TRUE,Expansion,CONSTRUCTION,65,100,70,2500000000000,CEO A,4,12,3,82,80,TRUE,TRUE\n'
        )
        parsed = parse_project_management_csv(csv_data)
        self.assertEqual(parsed.loc[0, 'ticker'], 'ANTM.JK')
        self.assertAlmostEqual(float(parsed.loc[0, 'project_completion_pct']), 0.65)
        self.assertTrue(bool(parsed.loc[0, 'management_verified']))

    def test_allocator_uses_verified_forward_quality(self):
        base = {
            'multibagger_status': 'MULTIBAGGER_A_CANDIDATE',
            'compounding_state': 'ACCUMULATE_NOW',
            'growth_score': 18.0,
            'profitability_score': 15.0,
            'earnings_quality_score': 16.0,
            'balance_sheet_score': 10.0,
            'valuation_score': 6.0,
            'momentum_score': 8.0,
            'accumulation_score': 8.0,
            'fundamental_data_grade': 'A',
            'fundamental_reliability': 'HIGH',
            'fundamental_consensus_score': 90.0,
            'fundamental_history_coverage': 90.0,
            'fundamental_official_verified': True,
            'fundamental_official_reference': True,
            'fundamental_source_count': 3,
            'solvency_coverage': 100.0,
            'fundamental_model': 'GENERAL',
            'technical_entry_state': 'SIGNAL_READY',
            'fundamental_conflicts': '',
            'red_flags': '',
            'entry': 1000.0,
            'last_price': 995.0,
            'multibagger_score': 85.0,
            'project_data_coverage_effective': 100.0,
            'management_data_coverage_effective': 100.0,
        }
        strong = dict(base, ticker='STRONG.JK', project_pipeline_score=90.0, management_quality_score=88.0)
        weak = dict(base, ticker='WEAK.JK', project_pipeline_score=35.0, management_quality_score=40.0)
        cfg = ScanConfig().replace(multibagger_capital_budget_idr=10_000_000.0, multibagger_max_holdings=2, multibagger_min_capital_conviction=50.0)
        result = allocate_multibagger_capital(pd.DataFrame([strong, weak]), cfg).sort_values('capital_priority_rank')
        self.assertEqual(result.iloc[0]['ticker'], 'STRONG.JK')
        self.assertGreater(float(result.iloc[0]['capital_conviction_score']), float(result.iloc[1]['capital_conviction_score']))
        self.assertGreater(float(result.iloc[0]['project_capital_weight_pct']), 0.0)
        self.assertGreater(float(result.iloc[0]['management_capital_weight_pct']), 0.0)

class MultibaggerForwardQualityIntegrationV590Tests(unittest.TestCase):
    def test_verified_project_and_management_feed_multibagger_ranking(self):
        frame = SpecialtyScannerV42Tests._strong_daily_frame()
        fundamentals = SafeSignalSemanticsV561Tests.strong_fundamentals(
            history_revenue_ttm=5_000_000_000_000.0,
            history_capex_ttm=800_000_000_000.0,
            history_ocf_ttm=1_000_000_000_000.0,
            history_fcf_ttm=200_000_000_000.0,
            history_revenue_cagr_3y=0.18,
            ceo_name='CEO Automatic',
        )
        review = parse_project_management_csv(pd.DataFrame([{
            'ticker': 'TEST', 'as_of': '2026-07-01', 'source_verified': True,
            'project_name': 'Capacity Expansion', 'project_stage': 'COMMISSIONING',
            'project_completion_pct': 90, 'funding_secured_pct': 100,
            'offtake_secured_pct': 80, 'project_capex_idr': 1_000_000_000_000,
            'strategic_project': True, 'project_risk': 'LOW',
            'ceo_name': 'CEO Verified', 'ceo_tenure_years': 5,
            'management_revenue_cagr': 18, 'management_roic_change_pct': 4,
            'capital_allocation_score': 88, 'governance_score': 90,
            'audit_clean': True, 'management_verified': True,
        }]))
        result = scan_multibagger_candidates(
            {'TEST.JK': frame}, fundamentals, project_management=review,
        )
        self.assertEqual(result.loc[0, 'project_data_source'], 'VERIFIED_PROJECT_PIPELINE')
        self.assertEqual(result.loc[0, 'management_data_source'], 'VERIFIED_MANAGEMENT_REVIEW')
        self.assertEqual(result.loc[0, 'ceo_name'], 'CEO Verified')
        self.assertGreater(float(result.loc[0, 'project_pipeline_score']), 70.0)
        self.assertGreater(float(result.loc[0, 'management_quality_score']), 70.0)
        self.assertGreater(float(result.loc[0, 'forward_quality_coverage']), 50.0)

    def test_forward_project_extraction_and_source_quorum_fields(self):
        from scanner_specialty import _extract_forward_rows
        fund = {
            'ceo_name': 'Budi Santoso', 'ceo_title': 'President Director',
            'history_revenue_cagr_3y': 0.18, 'history_roic_proxy': 0.15,
            'history_cash_conversion': 1.1, 'history_share_dilution_yoy': 0.01,
        }
        document_text = (
            'Perseroan membangun proyek pabrik baru dengan investasi Rp 1,5 triliun. '
            'Progress konstruksi telah mencapai 65% dan ditargetkan commercial operation pada 2027. '
            'Budi Santoso menjabat sebagai Direktur Utama sejak 2022.'
        )
        rows = _extract_forward_rows('TEST.JK', document_text, 'https://www.idx.co.id/test.pdf', 'IDX_OFFICIAL', True, fund)
        frame = pd.DataFrame(rows)
        projects = frame[frame['project_name'].astype(str).str.len().gt(0)]
        self.assertFalse(projects.empty)
        self.assertEqual(projects.iloc[0]['project_stage'], 'CONSTRUCTION')
        self.assertGreater(float(projects.iloc[0]['project_capex_idr']), 1e12)
        self.assertTrue(bool(projects.iloc[0]['source_verified']))

    def test_future_fundamental_impact_uses_disclosed_revenue(self):
        from scanner_specialty import _future_fundamental_impact
        pm = {
            'project_pipeline_score_observed': 80.0, 'project_data_coverage': 90.0,
            'project_completion_pct': 0.7, 'project_funding_secured_pct': 0.8,
            'project_ownership_pct': 1.0, 'project_capex_idr': 500e9,
            'project_expected_revenue_idr': 300e9, 'project_expected_ebitda_idr': 90e9,
        }
        fund = {
            'history_revenue_ttm': 1e12, 'history_ebitda_ttm': 200e9,
            'net_margin': 0.10, 'history_fcf_ttm': 100e9, 'total_debt': 400e9,
        }
        result = _future_fundamental_impact(pm, fund)
        self.assertEqual(result['future_impact_model'], 'DISCLOSED_PROJECT_REVENUE')
        self.assertGreater(result['future_revenue_uplift_base_pct'], 10.0)
        self.assertIn(result['future_impact_confidence'], {'MEDIUM', 'HIGH'})

    def test_manual_project_review_merges_as_override_evidence(self):
        auto = pd.DataFrame([{'ticker': 'ABCD.JK', 'project_name': 'Auto', 'source_verified': True}])
        manual = pd.DataFrame([{'ticker': 'ABCD.JK', 'project_name': 'Manual', 'source_verified': True}])
        merged = merge_project_management_reviews(auto, manual)
        self.assertEqual(len(merged), 2)
        self.assertIn('MANUAL_OVERRIDE', set(merged['review_origin']))

    def test_automatic_forward_collector_builds_evidence_without_upload(self):
        import scanner_specialty as specialty
        fundamentals = pd.DataFrame([{
            'ticker': 'AUTO.JK', 'company_name': 'PT Auto Tbk',
            'company_website': 'https://auto.example.com',
            'fundamental_score_10': 8.5, 'fundamental_coverage': 85.0,
            'revenue_growth': 0.20, 'history_roic_proxy': 0.13,
            'history_cash_conversion': 1.0, 'history_share_dilution_yoy': 0.0,
        }])
        document = (
            'PT Auto Tbk menjalankan proyek ekspansi pabrik dengan investasi Rp 2 triliun. '
            'Progress konstruksi mencapai 70% dan ditargetkan beroperasi pada 2027.'
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'IDX_SCANNER_CACHE_DIR': directory}):
            with patch.object(specialty, '_discover_ir_links', return_value=['https://www.idx.co.id/auto.pdf']), \
                 patch.object(specialty, '_search_forward_links', return_value=[]), \
                 patch.object(specialty, '_fetch_document', return_value=(document, 'https://www.idx.co.id/auto.pdf', 'PDF')):
                evidence, report = specialty.collect_automatic_forward_quality(
                    fundamentals, ['AUTO.JK'],
                    ScanConfig(automatic_forward_quality_top_n=1, automatic_forward_quality_workers=1),
                    force_refresh=True,
                )
        self.assertFalse(evidence.empty)
        self.assertEqual(report.iloc[0]['state'], 'AUTO_SINGLE_SOURCE')
        self.assertTrue(evidence['automatic_discovery'].all())


class TimeCycleV640Tests(unittest.TestCase):
    @staticmethod
    def cyclical_frame(period: int = 21, bars: int = 420) -> pd.DataFrame:
        index = pd.bdate_range("2024-01-02", periods=bars)
        x = np.arange(bars, dtype=float)
        close = 100.0 + 0.05 * x + 8.0 * np.sin(2.0 * np.pi * x / period)
        open_ = close - 0.3 * np.cos(2.0 * np.pi * x / period)
        high = np.maximum(open_, close) + 1.2
        low = np.minimum(open_, close) - 1.2
        volume = 1_000_000.0 * (1.0 + 0.15 * np.cos(2.0 * np.pi * x / period))
        return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=index)

    def test_objective_cycle_detects_known_period(self):
        from time_cycle import analyze_time_cycle
        result = analyze_time_cycle(self.cyclical_frame())
        self.assertIn(result["time_cycle_state"], {"VALIDATED", "LIMITED_EVIDENCE"})
        self.assertLess(abs(float(result["dominant_cycle_bars"]) - 21.0), 5.0)
        self.assertGreaterEqual(int(result["cycle_validation_samples"]), 5)
        self.assertTrue(result["next_reversal_window_start"])

    def test_quick_buy_decision_returns_date_and_structural_levels(self):
        from time_cycle import analyze_time_cycle
        result = analyze_time_cycle(self.cyclical_frame())
        self.assertIn("quick_buy_action", result)
        self.assertIn("best_buy_date", result)
        self.assertIn("best_buy_entry_low", result)
        self.assertIn("best_buy_trigger", result)
        self.assertIn("best_buy_stop_loss", result)
        if result["best_buy_date"]:
            self.assertGreater(float(result["best_buy_trigger"]), float(result["best_buy_stop_loss"]))
            self.assertGreaterEqual(float(result["best_buy_entry_high"]), float(result["best_buy_entry_low"]))

    def test_quick_buy_fails_soft_on_short_history(self):
        from time_cycle import analyze_time_cycle
        result = analyze_time_cycle(self.cyclical_frame(bars=80))
        self.assertEqual(result["quick_buy_state"], "NO_VALID_BUY_DATE")
        self.assertEqual(result["quick_buy_action"], "WAIT")
        self.assertEqual(result["best_buy_date"], "")

    @staticmethod
    def trigger_break_frame() -> pd.DataFrame:
        index = pd.bdate_range("2024-01-02", periods=80)
        close = np.linspace(100.0, 106.0, len(index))
        frame = pd.DataFrame({
            "Open": close - 0.4,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 2_000_000.0,
        }, index=index)
        frame.iloc[-1, frame.columns.get_loc("Open")] = 105.0
        frame.iloc[-1, frame.columns.get_loc("High")] = 122.0
        frame.iloc[-1, frame.columns.get_loc("Low")] = 104.0
        frame.iloc[-1, frame.columns.get_loc("Close")] = 120.0
        frame.attrs["bar_state"] = "FINAL_EOD"
        return frame

    def test_trigger_uses_prior_high_and_can_confirm(self):
        import time_cycle
        frame = self.trigger_break_frame()
        date = frame.index[-1].date().isoformat()
        result = time_cycle._build_quick_buy_decision(
            frame,
            direction="BULLISH_REVERSAL_WINDOW",
            phase="CORRECTION",
            time_state="VALIDATED",
            time_score=90.0,
            confidence=90.0,
            bullish_timing=90.0,
            continuation_timing=75.0,
            price_time=90.0,
            trend_score=80.0,
            window_start=date,
            window_end=date,
            bars_due=0.0,
            eoff={},
        )
        self.assertEqual(result["quick_buy_action"], "BUY_ON_CONFIRMED_TRIGGER")
        self.assertLess(float(result["best_buy_trigger"]), float(frame["Close"].iloc[-1]))
        self.assertGreaterEqual(float(result["best_buy_rr1"]), 1.5)
        self.assertTrue(np.isfinite(float(result["best_buy_tp1"])))

    def test_limited_evidence_never_prepares_an_order(self):
        import time_cycle
        frame = self.trigger_break_frame()
        date = frame.index[-1].date().isoformat()
        result = time_cycle._build_quick_buy_decision(
            frame,
            direction="BULLISH_REVERSAL_WINDOW",
            phase="CORRECTION",
            time_state="LIMITED_EVIDENCE",
            time_score=90.0,
            confidence=90.0,
            bullish_timing=90.0,
            continuation_timing=75.0,
            price_time=90.0,
            trend_score=80.0,
            window_start=date,
            window_end=date,
            bars_due=0.0,
            eoff={},
        )
        self.assertEqual(result["quick_buy_action"], "WAIT_FOR_EVIDENCE")
        self.assertEqual(result["best_buy_order_plan"], "NO_ORDER")

    def test_same_type_cycle_intervals_are_merged_chronologically(self):
        from time_cycle import _chronological_same_type_intervals
        result = _chronological_same_type_intervals([0, 10, 30], [5, 20, 40], 1, 100)
        self.assertEqual(result.tolist(), [10.0, 15.0, 20.0, 20.0])

    def test_limited_evidence_has_zero_core_weight(self):
        import time_cycle
        signals = pd.DataFrame([{"ticker": "TEST.JK", "setup": "PULLBACK_CONTINUATION"}])
        analysis = {
            "time_cycle_state": "LIMITED_EVIDENCE",
            "time_cycle_confidence": 95.0,
            "cycle_validation_samples": 50,
            "bullish_timing_score": 95.0,
            "price_time_confluence_score": 95.0,
        }
        with patch.object(time_cycle, "analyze_time_cycle", return_value=analysis):
            result = time_cycle.enrich_core_signals_with_time_cycle(
                signals, {"TEST.JK": self.cyclical_frame()}, min_confidence=40.0,
            )
        self.assertEqual(float(result.loc[0, "time_cycle_effective_weight_pct"]), 0.0)

    def test_core_enrichment_is_guarded_by_evidence(self):
        from time_cycle import enrich_core_signals_with_time_cycle
        signals = pd.DataFrame([{
            "ticker": "TEST.JK", "setup": "PULLBACK_CONTINUATION",
            "status": "ENTRY_PLAN_READY",
        }])
        enriched = enrich_core_signals_with_time_cycle(
            signals, {"TEST.JK": self.cyclical_frame()}, enabled=True, max_weight=0.10, min_confidence=40.0,
        )
        self.assertIn("time_cycle_alignment_score", enriched.columns)
        self.assertIn("quick_buy_action", enriched.columns)
        self.assertIn("best_buy_date", enriched.columns)
        self.assertGreaterEqual(float(enriched.iloc[0]["time_cycle_effective_weight_pct"]), 0.0)
        self.assertLessEqual(float(enriched.iloc[0]["time_cycle_effective_weight_pct"]), 10.0)

    def test_structural_stop_rejects_micro_stop_and_uses_support(self):
        frame = prepare_indicators(self.cyclical_frame(), None)
        entry = float(frame["Close"].iloc[-1])
        atr = float(frame["ATR14"].iloc[-1])
        raw = entry - idx_tick_size(entry)
        result = scanner_module._structural_stop_selection(frame, entry, raw, atr, "PULLBACK_CONTINUATION")
        self.assertTrue(result["stop_structure_valid"])
        self.assertLess(float(result["stop_loss"]), raw)
        self.assertNotEqual(result["stop_basis"], "STRUCTURE_UNAVAILABLE")

    def test_navigation_text_is_not_project_evidence(self):
        import scanner_specialty as specialty
        noise = "Home Investor Relations Internal Audit Komite Audit Development Management Contact Us"
        self.assertFalse(specialty._is_project_evidence(noise))
        evidence = "Perseroan membangun pabrik baru dengan investasi Rp 2 triliun dan progress konstruksi 70% pada 2026."
        self.assertTrue(specialty._is_project_evidence(evidence))

    def test_specialty_rows_receive_zero_time_cycle_weight(self):
        screens = {
            "bsjp": pd.DataFrame([{
                "ticker": "FAST.JK", "bsjp_status": "BSJP_SIGNAL_READY", "bsjp_score": 82.0,
                "entry": 100.0, "stop_loss": 95.0, "morning_tp1": 110.0, "morning_tp2": 115.0,
                "rr1": 2.0, "rr2": 3.0, "adtv20_idr": 5e9, "target_structure_valid": True,
                "intraday_fresh": True, "session_close_location": 0.85, "vwap_slope_pct": 0.001,
                "late_volume_acceleration": 1.5, "late_volume_share": 0.4, "directional_efficiency": 0.6,
                "afternoon_breakout": True,
            }])
        }
        result = build_profit_order_builder(pd.DataFrame(), screens, ScanConfig(profit_conviction_min_score=0))
        self.assertEqual(float(result.iloc[0]["time_cycle_effective_weight_pct"]), 0.0)


class EyeOfFutureReconstructionV650Tests(unittest.TestCase):
    @staticmethod
    def frame(bars: int = 420) -> pd.DataFrame:
        index = pd.bdate_range("2024-01-02", periods=bars)
        x = np.arange(bars, dtype=float)
        close = 100.0 + 0.04 * x + 7.0 * np.sin(2.0 * np.pi * x / 21.0)
        return pd.DataFrame({
            "Open": close - 0.4,
            "High": close + 1.2,
            "Low": close - 1.2,
            "Close": close,
            "Volume": 1_500_000.0 + 200_000.0 * np.cos(2.0 * np.pi * x / 13.0),
        }, index=index)

    def test_ephemeris_is_offline_and_returns_geocentric_markers(self):
        import eoff_reconstruction as eoff
        from eoff_reconstruction import EOFFConfig, _ephemeris_snapshot
        if eoff.ephem is None:
            self.skipTest("ephem dependency is not installed in the audit environment")
        result = _ephemeris_snapshot(pd.Timestamp("2026-07-20"), EOFFConfig())
        self.assertTrue(result["ephemeris_available"])
        self.assertEqual(result["ephemeris_state"], "READY")
        self.assertTrue(np.isfinite(float(result["moon_declination_deg"])))
        self.assertIn(result["sun_sign"], {
            "ARIES", "TAURUS", "GEMINI", "CANCER", "LEO", "VIRGO",
            "LIBRA", "SCORPIO", "SAGITTARIUS", "CAPRICORN", "AQUARIUS", "PISCES",
        })

    def test_ephemeris_dependency_missing_fails_soft(self):
        import eoff_reconstruction as eoff
        with patch.object(eoff, "ephem", None):
            result = eoff.analyze_eoff_reconstruction(self.frame())
        self.assertEqual(result["eoff_ephemeris_state"], "DEPENDENCY_MISSING")
        self.assertFalse(result["eoff_signal_active"])

    def test_public_style_four_projection_plus_astro_gate_can_activate(self):
        import eoff_reconstruction as eoff
        ephemeris = {
            "ephemeris_available": True, "ephemeris_state": "READY",
            "astro_cluster_score": 90.0, "moon_phase_score": 80.0,
            "moon_declination_extreme_score": 75.0, "aspect_cluster_score": 85.0,
            "sun_annual_cycle_score": 68.0, "sun_sign": "CAPRICORN",
            "sun_annual_cycle_bias": "TRADING_BIAS", "moon_phase_name": "NEW_MOON",
            "moon_phase_angle_deg": 2.0, "moon_declination_deg": 24.0,
            "moon_declination_turning": True, "astro_events": ["NEW_MOON"],
            "active_aspects": [], "retrograde_planets": [], "stationary_planets": ["Mercury"],
            "ingress_events": [],
        }
        cluster = {
            "fib_cluster_count": 8, "fib_unique_anchor_count": 5,
            "fib_cluster_target_bar": len(self.frame()) - 1,
            "fib_cluster_bars_ahead": 0, "fib_cluster_score": 100.0,
            "fib_projection_details": [],
        }
        price_context = {
            "close": 120.0, "atr": 3.0, "ema20": 118.0, "ema50": 112.0, "ema200": 100.0,
            "eoff_direction_bias": "BULLISH", "bullish_pattern_score": 90.0,
            "bearish_pattern_score": 20.0, "bullish_momentum_score": 88.0,
            "bullish_exhaustion_score": 85.0, "bearish_momentum_score": 20.0,
            "bearish_exhaustion_score": 20.0, "support_score": 92.0,
            "resistance_score": 30.0, "fib_price_score": 88.0,
            "trend_bull_score": 90.0, "trend_bear_score": 10.0,
            "swing_low": 100.0, "swing_high": 125.0, "fib_382": 115.45,
            "fib_618": 109.55, "ext_1272": 131.8,
        }
        with patch.object(eoff, "_ephemeris_snapshot", return_value=ephemeris), \
             patch.object(eoff, "_best_projection_cluster", return_value=cluster), \
             patch.object(eoff, "_historical_fib_validation", return_value=(70.0, 20, 6.0, 30.0, 2.33, [200, 240, 280, 320])), \
             patch.object(eoff, "_historical_confluence_validation", return_value=(72.0, 20, 2.4)), \
             patch.object(eoff, "_public_directional_validation", return_value={
                 "events": 20, "reversal_hit_rate": 68.0, "baseline_rate": 30.0,
                 "lift": 2.27, "forward_hit_rate": 65.0,
                 "median_directional_return_pct": 2.4,
                 "validation_state": "CHRONOLOGICAL_FORWARD_TEST",
                 "method": "PUBLIC_FIXED_FAMILY_CHRONOLOGICAL_FORWARD_TEST",
             }), \
             patch.object(eoff, "_price_pattern_momentum", return_value=price_context):
            result = eoff.analyze_eoff_reconstruction(self.frame())
        self.assertTrue(result["eoff_signal_active"])
        self.assertEqual(result["eoff_strength_label"], "VERY_STRONG")
        self.assertGreaterEqual(result["eoff_fib_cluster_count"], 4)
        self.assertGreater(result["eoff_historical_lift"], 1.0)

    def test_low_strength_eoff_has_zero_internal_influence(self):
        import time_cycle
        shadow = {
            "eoff_state": "ACTIVE_PUBLIC_RESEARCH_GUARDED", "eoff_signal_active": True,
            "eoff_strength_label": "LOW", "eoff_direction_bias": "BULLISH",
            "eoff_reconstruction_score": 55.0, "eoff_public_directional_events": 30,
            "eoff_public_forward_hit_rate": 55.0,
        }
        with patch.object(time_cycle, "analyze_eoff_reconstruction", return_value=shadow):
            result = time_cycle.analyze_time_cycle(self.frame())
        self.assertEqual(float(result["eoff_internal_weight_pct"]), 0.0)

    def test_strong_eoff_is_bounded_inside_time_cycle(self):
        import time_cycle
        strong = {
            "eoff_state": "ACTIVE_PUBLIC_RESEARCH_GUARDED", "eoff_signal_active": True,
            "eoff_strength_label": "STRONG", "eoff_direction_bias": "BULLISH",
            "eoff_reconstruction_score": 86.0, "eoff_public_directional_events": 30,
            "eoff_public_forward_hit_rate": 70.0,
        }
        with patch.object(time_cycle, "analyze_eoff_reconstruction", return_value=strong):
            result = time_cycle.analyze_time_cycle(self.frame())
        self.assertGreater(float(result["eoff_internal_weight_pct"]), 0.0)
        self.assertLessEqual(float(result["eoff_internal_weight_pct"]), 35.0)

    def test_future_price_roadmap_is_structured_scenario(self):
        import json
        from eoff_reconstruction import analyze_eoff_reconstruction
        result = analyze_eoff_reconstruction(self.frame())
        roadmap = json.loads(result["eoff_roadmap_json"])
        self.assertGreaterEqual(len(roadmap), 1)
        self.assertIn("phase", roadmap[0])
        self.assertIn("price_zone_low", roadmap[0])
        self.assertIn("invalidation", roadmap[0])

    def test_projection_library_never_uses_future_anchor(self):
        from eoff_reconstruction import EOFFConfig, _projection_candidates
        current = 100
        candidates = _projection_candidates([10, 25, 70, 101, 120], current, EOFFConfig())
        self.assertTrue(all(int(row["anchor"]) <= current for row in candidates))

    def test_public_five_candle_reversal_definition_is_directional(self):
        from eoff_reconstruction import _public_five_candle_reversals
        frame = pd.DataFrame({
            "Low": [5, 4, 3, 4, 5, 5, 5, 5, 5, 5],
            "High": [6, 6, 6, 6, 6, 7, 8, 9, 8, 7],
            "Close": [5.5] * 10,
        }, index=pd.bdate_range("2025-01-01", periods=10))
        bullish, bearish = _public_five_candle_reversals(frame)
        self.assertIn(2, bullish)
        self.assertIn(7, bearish)

    def test_adaptive_factor_walk_forward_requires_oos_skill(self):
        from eoff_reconstruction import EOFFConfig, _walk_forward_factor_metrics
        records = []
        for i in range(30):
            records.append({
                "position": i,
                "reversal_hit": 1.0,
                "baseline_rate": 30.0,
                "forward_hit": 1.0,
                "directional_return_pct": 2.0,
            })
        result = _walk_forward_factor_metrics(records, EOFFConfig(adaptive_min_train_events=10, adaptive_min_oos_events=6), 0.12)
        self.assertTrue(result["validated"])
        self.assertEqual(result["state"], "WALK_FORWARD_VALIDATED")
        self.assertGreater(result["weight_pct"], 0.0)
        self.assertLessEqual(result["weight_pct"], 12.0)

    def test_adaptive_factor_random_or_weak_history_has_zero_weight(self):
        from eoff_reconstruction import EOFFConfig, _walk_forward_factor_metrics
        records = []
        for i in range(32):
            records.append({
                "position": i,
                "reversal_hit": float(i % 4 == 0),
                "baseline_rate": 30.0,
                "forward_hit": float(i % 2 == 0),
                "directional_return_pct": -0.2 if i % 2 else 0.2,
            })
        result = _walk_forward_factor_metrics(records, EOFFConfig(adaptive_min_train_events=10, adaptive_min_oos_events=6), 0.12)
        self.assertFalse(result["validated"])
        self.assertEqual(result["weight_pct"], 0.0)

    def test_declination_keeps_public_prior_but_cannot_activate_alone(self):
        import eoff_reconstruction as eoff
        frame = self.frame()
        ephemeris = {
            "ephemeris_available": True, "ephemeris_state": "READY",
            "moon_phase_score": 0.0, "aspect_cluster_score": 0.0,
            "moon_declination_extreme_score": 95.0, "moon_declination_turning": True,
            "ingress_events": [], "retrograde_transition_events": [], "stationary_planets": [],
            "sun_annual_cycle_bias": "TRADING_BIAS", "sun_annual_cycle_score": 68.0,
            "astro_cluster_score": 90.0, "active_aspects": [], "retrograde_planets": [],
            "moon_phase_name": "FIRST_QUARTER", "moon_phase_angle_deg": 90.0,
            "moon_declination_deg": 28.0, "sun_sign": "CAPRICORN", "astro_events": ["MOON_DECLINATION_EXTREME"],
        }
        cluster = {"fib_cluster_count": 8, "fib_unique_anchor_count": 5, "fib_cluster_target_bar": len(frame)-1, "fib_cluster_bars_ahead": 0, "fib_cluster_score": 100.0, "fib_projection_details": []}
        price_context = {"close":120.0,"atr":3.0,"eoff_direction_bias":"BULLISH","bullish_pattern_score":90.0,"bearish_pattern_score":20.0,"bullish_momentum_score":88.0,"bullish_exhaustion_score":85.0,"support_score":92.0,"resistance_score":30.0,"fib_price_score":88.0,"trend_bull_score":90.0,"trend_bear_score":10.0}
        shadow = {family: {"validated": False, "state": "REJECTED_NO_OOS_SKILL", "weight_fraction": 0.0, "oos_events": 10} for family in eoff.ADAPTIVE_ASTRO_FAMILIES}
        with patch.object(eoff, "_ephemeris_snapshot", return_value=ephemeris), patch.object(eoff, "_best_projection_cluster", return_value=cluster), patch.object(eoff, "_historical_fib_validation", return_value=(70.0,20,6.0,30.0,2.33,[200,240,280,320])), patch.object(eoff, "_historical_confluence_validation", return_value=(np.nan,0,np.nan)), patch.object(eoff, "_public_directional_validation", return_value={"events":0,"reversal_hit_rate":np.nan,"baseline_rate":np.nan,"lift":np.nan,"forward_hit_rate":np.nan,"median_directional_return_pct":np.nan}), patch.object(eoff, "_adaptive_astro_walk_forward", return_value=shadow), patch.object(eoff, "_price_pattern_momentum", return_value=price_context):
            result = eoff.analyze_eoff_reconstruction(frame)
        self.assertFalse(result["eoff_signal_active"])
        self.assertGreater(result["eoff_declination_weight_pct"], 0.0)
        self.assertEqual(result["eoff_validation_path"], "NONE")

    def test_validated_declination_can_supply_guarded_astro_path(self):
        import eoff_reconstruction as eoff
        frame = self.frame()
        ephemeris = {
            "ephemeris_available": True, "ephemeris_state": "READY",
            "moon_phase_score": 0.0, "aspect_cluster_score": 0.0,
            "moon_declination_extreme_score": 95.0, "moon_declination_turning": True,
            "ingress_events": [], "retrograde_transition_events": [], "stationary_planets": [],
            "sun_annual_cycle_bias": "TRADING_BIAS", "sun_annual_cycle_score": 68.0,
            "astro_cluster_score": 90.0, "active_aspects": [], "retrograde_planets": [],
            "moon_phase_name": "FIRST_QUARTER", "moon_phase_angle_deg": 90.0,
            "moon_declination_deg": 28.0, "sun_sign": "CAPRICORN", "astro_events": ["MOON_DECLINATION_EXTREME"],
        }
        cluster = {"fib_cluster_count": 8, "fib_unique_anchor_count": 5, "fib_cluster_target_bar": len(frame)-1, "fib_cluster_bars_ahead": 0, "fib_cluster_score": 100.0, "fib_projection_details": []}
        price_context = {"close":120.0,"atr":3.0,"eoff_direction_bias":"BULLISH","bullish_pattern_score":90.0,"bearish_pattern_score":20.0,"bullish_momentum_score":88.0,"bullish_exhaustion_score":85.0,"support_score":92.0,"resistance_score":30.0,"fib_price_score":88.0,"trend_bull_score":90.0,"trend_bear_score":10.0}
        adaptive = {family: {"validated": False, "state": "SHADOW_INSUFFICIENT_OOS", "weight_fraction": 0.0, "oos_events": 0} for family in eoff.ADAPTIVE_ASTRO_FAMILIES}
        adaptive["MOON_DECLINATION"] = {"validated": True, "state": "WALK_FORWARD_VALIDATED", "weight_fraction": 0.08, "oos_events": 12, "oos_reversal_hit_rate":65.0,"oos_baseline_rate":30.0,"oos_lift":2.17,"oos_forward_hit_rate":65.0,"oos_median_directional_return_pct":2.0}
        with patch.object(eoff, "_ephemeris_snapshot", return_value=ephemeris), patch.object(eoff, "_best_projection_cluster", return_value=cluster), patch.object(eoff, "_historical_fib_validation", return_value=(70.0,20,6.0,30.0,2.33,[200,240,280,320])), patch.object(eoff, "_historical_confluence_validation", return_value=(np.nan,0,np.nan)), patch.object(eoff, "_public_directional_validation", return_value={"events":0,"reversal_hit_rate":np.nan,"baseline_rate":np.nan,"lift":np.nan,"forward_hit_rate":np.nan,"median_directional_return_pct":np.nan}), patch.object(eoff, "_adaptive_astro_walk_forward", return_value=adaptive), patch.object(eoff, "_price_pattern_momentum", return_value=price_context):
            result = eoff.analyze_eoff_reconstruction(frame)
        self.assertTrue(result["eoff_signal_active"])
        self.assertEqual(result["eoff_validation_path"], "OOS_VALIDATED_SECONDARY_WITH_PUBLIC_PRIOR")
        self.assertGreater(result["eoff_declination_weight_pct"], 0.0)
        self.assertIn("MOON_DECLINATION", result["eoff_adaptive_active_factors"])

    def test_public_prior_astro_weights_sum_to_one_and_never_zero(self):
        import eoff_reconstruction as eoff
        weights, multipliers = eoff._normalized_public_astro_weights({}, eoff.EOFFConfig())
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=9)
        self.assertEqual(set(weights), set(eoff.PUBLIC_ASTRO_PRIOR_WEIGHTS))
        self.assertTrue(all(value > 0.0 for value in weights.values()))
        self.assertAlmostEqual(weights["MOON_PHASE"], 0.25, places=6)
        self.assertAlmostEqual(weights["PLANETARY_ASPECT"], 0.25, places=6)
        self.assertAlmostEqual(weights["MOON_DECLINATION"], 0.15, places=6)
        self.assertAlmostEqual(weights["INGRESS"], 0.10, places=6)
        self.assertAlmostEqual(weights["RETROGRADE"], 0.10, places=6)
        self.assertAlmostEqual(weights["SUN_ANNUAL"], 0.15, places=6)
        self.assertTrue(all(value == 1.0 for value in multipliers.values()))

    def test_rejected_factor_is_reduced_not_deleted(self):
        import eoff_reconstruction as eoff
        validation = {family: {"validated": False, "state": "REJECTED_NO_OOS_SKILL"} for family in eoff.ADAPTIVE_ASTRO_FAMILIES}
        weights, multipliers = eoff._normalized_public_astro_weights(validation, eoff.EOFFConfig())
        self.assertTrue(all(weights[family] > 0.0 for family in eoff.ADAPTIVE_ASTRO_FAMILIES))
        self.assertTrue(all(multipliers[family] == 0.75 for family in eoff.ADAPTIVE_ASTRO_FAMILIES))
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=9)



class ThreeTabDecisionDashboardV660Tests(unittest.TestCase):
    def test_top15_keeps_multibagger_and_swing_trade_plans_separate(self):
        from dashboard_v660 import build_top15_ranking

        swing = pd.DataFrame([{
            "ticker": "ANTM.JK",
            "strategy": "PULLBACK_CONTINUATION",
            "hybrid_conviction_score": 78.0,
            "best_buy_score": 82.0,
            "time_cycle_score": 80.0,
            "time_cycle_confidence": 72.0,
            "quick_buy_action": "WAIT_FOR_DATE",
            "best_buy_date": "2026-07-24",
            "best_buy_window_start": "2026-07-23",
            "best_buy_window_end": "2026-07-27",
            "best_buy_entry_low": 3120.0,
            "best_buy_entry_high": 3180.0,
            "best_buy_trigger": 3190.0,
            "best_buy_stop_loss": 3040.0,
            "best_buy_tp1": 3350.0,
            "best_buy_tp2": 3520.0,
            "rr1": 1.5,
            "rr2": 2.6,
        }])
        multibagger = pd.DataFrame([{
            "ticker": "ANTM.JK",
            "capital_conviction_score": 88.0,
            "project_pipeline_score": 80.0,
            "management_quality_score": 75.0,
            "future_fundamental_impact_score": 82.0,
            "forward_quality_coverage": 80.0,
            "multibagger_time_cycle_score": 77.0,
            "allocation_action": "ACCUMULATE_GRADUALLY",
            "capital_tier": "CORE_COMPOUNDING",
        }])
        result = {"specialty_screens": {"profit_order_builder": swing, "multibagger": multibagger}}
        ranking = build_top15_ranking(result)

        self.assertEqual(len(ranking), 1)
        row = ranking.iloc[0]
        self.assertEqual(row["category"], "MULTIBAGGER + SWING/CORE")
        self.assertIn("PULLBACK_CONTINUATION", row["strategy"])
        self.assertEqual(row["decision"], "WAIT_FOR_DATE")
        self.assertEqual(row["best_buy_date"], "2026-07-24")
        self.assertEqual(float(row["trigger"]), 3190.0)
        self.assertEqual(float(row["stop_loss"]), 3040.0)
        self.assertEqual(len(row["all_rows"]), 2)
        self.assertEqual(ranking["candidate_id"].nunique(), 1)

    def test_top15_excludes_intraday_bpjs_bsjp_and_ara(self):
        from dashboard_v660 import build_top15_ranking

        builder = pd.DataFrame([
            {"ticker": "AAA.JK", "strategy": "PULLBACK_CONTINUATION", "hybrid_conviction_score": 75.0},
            {"ticker": "BBB.JK", "strategy": "BPJS", "hybrid_conviction_score": 99.0},
            {"ticker": "CCC.JK", "strategy": "BSJP", "hybrid_conviction_score": 98.0},
            {"ticker": "DDD.JK", "strategy": "PRE_ARA", "hybrid_conviction_score": 97.0},
        ])
        ranking = build_top15_ranking({"specialty_screens": {"profit_order_builder": builder}})
        self.assertEqual(ranking["ticker"].tolist(), ["AAA.JK"])

    def test_top15_limit_and_rank_are_stable(self):
        from dashboard_v660 import build_top15_ranking

        builder = pd.DataFrame([
            {
                "ticker": f"T{i:02d}.JK",
                "strategy": "BREAKOUT_RETEST",
                "hybrid_conviction_score": 90.0 - i,
                "best_buy_score": 70.0,
            }
            for i in range(20)
        ])
        ranking = build_top15_ranking({"specialty_screens": {"profit_order_builder": builder}}, limit=15)
        self.assertEqual(len(ranking), 15)
        self.assertEqual(ranking["rank"].tolist(), list(range(1, 16)))
        self.assertEqual(ranking.iloc[0]["ticker"], "T00.JK")

    def test_action_semantics_rank_buy_above_high_score_avoid(self):
        from dashboard_v660 import build_top15_ranking
        rows = pd.DataFrame([
            {
                "ticker": "BUY.JK", "strategy": "BREAKOUT_RETEST",
                "hybrid_conviction_score": 76.0, "quick_buy_action": "BUY_ON_CONFIRMED_TRIGGER",
                "best_buy_entry_low": 100.0, "best_buy_entry_high": 102.0,
                "best_buy_trigger": 103.0, "best_buy_stop_loss": 98.0,
                "best_buy_tp1": 111.0, "best_buy_rr1": 1.6,
            },
            {
                "ticker": "AVOID.JK", "strategy": "BREAKOUT_RETEST",
                "hybrid_conviction_score": 95.0, "quick_buy_action": "AVOID_NEW_BUY",
            },
        ])
        ranking = build_top15_ranking({"specialty_screens": {"profit_order_builder": rows}})
        self.assertEqual(ranking.iloc[0]["ticker"], "BUY.JK")
        self.assertEqual(ranking.iloc[-1]["decision"], "AVOID_NEW_BUY")

    def test_zero_effective_cycle_weight_means_zero_ranking_influence(self):
        from dashboard_v660 import build_top15_ranking
        rows = pd.DataFrame([
            {
                "ticker": "LOW.JK", "strategy": "BREAKOUT_RETEST",
                "hybrid_conviction_score": 80.0, "time_cycle_state": "LIMITED_EVIDENCE",
                "time_cycle_score": 0.0, "time_cycle_effective_weight_pct": 0.0,
            },
            {
                "ticker": "HIGH.JK", "strategy": "BREAKOUT_RETEST",
                "hybrid_conviction_score": 80.0, "time_cycle_state": "LIMITED_EVIDENCE",
                "time_cycle_score": 100.0, "time_cycle_effective_weight_pct": 0.0,
            },
        ])
        ranking = build_top15_ranking({"specialty_screens": {"profit_order_builder": rows}})
        scores = ranking.set_index("ticker")["combined_score"]
        self.assertEqual(float(scores["LOW.JK"]), float(scores["HIGH.JK"]))

    def test_missing_trade_plan_downgrades_prepare_to_wait(self):
        from dashboard_v660 import build_top15_ranking
        rows = pd.DataFrame([{
            "ticker": "MISS.JK", "strategy": "PULLBACK_CONTINUATION",
            "hybrid_conviction_score": 90.0,
            "quick_buy_action": "PREPARE_BUY_WAIT_TRIGGER",
            "best_buy_entry_low": 100.0, "best_buy_entry_high": 102.0,
            "best_buy_trigger": 103.0, "best_buy_stop_loss": 98.0,
            "best_buy_tp1": np.nan, "best_buy_rr1": np.nan,
        }])
        ranking = build_top15_ranking({"specialty_screens": {"profit_order_builder": rows}})
        self.assertEqual(ranking.loc[0, "decision"], "WAIT_FOR_EVIDENCE")


    def test_top15_unions_signals_when_builder_has_no_core_rows(self):
        from dashboard_v660 import build_top15_ranking

        builder = pd.DataFrame([{
            "ticker": "FAST.JK", "strategy": "BPJS", "hybrid_conviction_score": 99.0,
        }])
        signals = pd.DataFrame([{
            "ticker": "CORE.JK", "strategy": "PULLBACK CONTINUATION",
            "hybrid_conviction_score": 76.0, "quick_buy_action": "WAIT_FOR_EVIDENCE",
            "entry": 100.0, "stop_loss": 95.0, "tp1": 110.0, "rr1": 2.0,
        }])
        result = {"specialty_screens": {"profit_order_builder": builder}, "signals": signals}
        ranking = build_top15_ranking(result)

        self.assertEqual(ranking["ticker"].tolist(), ["CORE.JK"])
        self.assertEqual(ranking.loc[0, "category"], "SWING/CORE")
        self.assertEqual(ranking.loc[0, "strategy"], "PULLBACK_CONTINUATION")
        self.assertEqual(ranking.loc[0, "candidate_source"], "SIGNALS_FALLBACK")

    def test_top15_keeps_global_ranking_without_forced_category_quota(self):
        from dashboard_v660 import build_top15_ranking

        multibagger = pd.DataFrame([
            {
                "ticker": f"M{i:02d}.JK", "capital_conviction_score": 90.0 - i * 0.5,
                "future_fundamental_impact_score": 80.0,
                "project_pipeline_score": 80.0, "management_quality_score": 80.0,
                "forward_quality_coverage": 80.0, "quick_buy_action": "WAIT_FOR_EVIDENCE",
            }
            for i in range(15)
        ])
        signals = pd.DataFrame([{
            "ticker": "SWING.JK", "strategy": "BREAKOUT_RETEST",
            "hybrid_conviction_score": 40.0, "quick_buy_action": "WAIT_FOR_EVIDENCE",
        }])
        result = {"specialty_screens": {"multibagger": multibagger}, "signals": signals}
        ranking = build_top15_ranking(result, limit=15)
        audit = ranking.attrs.get("candidate_pool_audit", {})

        self.assertEqual(int(ranking["category"].eq("MULTIBAGGER").sum()), 15)
        self.assertEqual(int(ranking["category"].eq("SWING/CORE").sum()), 0)
        self.assertEqual(audit.get("eligible_swing_core"), 1)
        self.assertFalse(audit.get("forced_category_quota"))


    def test_top20_prioritizes_best_buy_date_plus_eoff_strong(self):
        from dashboard_v660 import build_top20_ranking

        rows = pd.DataFrame([
            {
                "ticker": "PRIORITY.JK", "strategy": "PULLBACK_CONTINUATION",
                "hybrid_conviction_score": 65.0, "best_buy_date": "2026-07-24",
                "eoff_strength_label": "STRONG", "quick_buy_action": "WAIT_FOR_DATE",
            },
            {
                "ticker": "HIGH.JK", "strategy": "PULLBACK_CONTINUATION",
                "hybrid_conviction_score": 95.0, "eoff_strength_label": "MEDIUM",
                "quick_buy_action": "WAIT_FOR_EVIDENCE",
            },
        ])
        ranking = build_top20_ranking({"signals": rows})
        self.assertEqual(ranking.iloc[0]["ticker"], "PRIORITY.JK")
        self.assertEqual(ranking.iloc[0]["ranking_priority"], "BEST_BUY_DATE + EOFF_STRONG")
        self.assertTrue(bool(ranking.iloc[0]["has_best_buy_date"]))
        self.assertTrue(bool(ranking.iloc[0]["eoff_strong"]))

    def test_top20_priority_order_date_then_strong_then_standard(self):
        from dashboard_v660 import build_top20_ranking

        rows = pd.DataFrame([
            {"ticker": "DATE.JK", "strategy": "BREAKOUT_RETEST", "hybrid_conviction_score": 60.0,
             "best_buy_date": "2026-07-25", "eoff_strength_label": "MEDIUM"},
            {"ticker": "STRONG.JK", "strategy": "BREAKOUT_RETEST", "hybrid_conviction_score": 90.0,
             "eoff_strength_label": "VERY_STRONG"},
            {"ticker": "STANDARD.JK", "strategy": "BREAKOUT_RETEST", "hybrid_conviction_score": 99.0,
             "eoff_strength_label": "MEDIUM"},
        ])
        ranking = build_top20_ranking({"signals": rows})
        self.assertEqual(ranking["ticker"].tolist(), ["DATE.JK", "STRONG.JK", "STANDARD.JK"])
        self.assertEqual(ranking["timing_priority"].tolist(), [1, 2, 3])

    def test_top20_default_limit_and_unique_tickers(self):
        from dashboard_v660 import build_top20_ranking

        signals = []
        for i in range(25):
            signals.append({
                "ticker": f"U{i:02d}.JK", "strategy": "PULLBACK_CONTINUATION",
                "hybrid_conviction_score": 90.0 - i * 0.1,
            })
        signals.append({
            "ticker": "U00.JK", "strategy": "BREAKOUT_RETEST",
            "hybrid_conviction_score": 99.0,
        })
        ranking = build_top20_ranking({"signals": pd.DataFrame(signals)})
        self.assertEqual(len(ranking), 20)
        self.assertEqual(ranking["ticker"].nunique(), 20)
        self.assertEqual(ranking["rank"].tolist(), list(range(1, 21)))
        u00 = ranking[ranking["ticker"].eq("U00.JK")].iloc[0]
        self.assertIn("BREAKOUT_RETEST", u00["strategy"])
        self.assertIn("PULLBACK_CONTINUATION", u00["strategy"])

    def test_unsafe_action_stays_below_safe_even_with_date_and_strong(self):
        from dashboard_v660 import build_top20_ranking

        rows = pd.DataFrame([
            {
                "ticker": "UNSAFE.JK", "strategy": "PULLBACK_CONTINUATION",
                "hybrid_conviction_score": 99.0, "best_buy_date": "2026-07-24",
                "eoff_strength_label": "VERY_STRONG", "quick_buy_action": "AVOID_NEW_BUY",
            },
            {
                "ticker": "SAFE.JK", "strategy": "PULLBACK_CONTINUATION",
                "hybrid_conviction_score": 50.0, "eoff_strength_label": "LOW",
                "quick_buy_action": "WAIT_FOR_EVIDENCE",
            },
        ])
        ranking = build_top20_ranking({"signals": rows})
        self.assertEqual(ranking.iloc[0]["ticker"], "SAFE.JK")
        self.assertEqual(ranking.iloc[-1]["ticker"], "UNSAFE.JK")


if __name__ == "__main__":
    unittest.main()
