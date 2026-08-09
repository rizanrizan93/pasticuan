from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch
import ast
import pathlib
import py_compile

import numpy as np
import pandas as pd

import resumable_app_engine as eng
from macro_engine import MacroRegimeResult

ROOT = pathlib.Path(__file__).resolve().parent


def _tickers(n: int = 400) -> list[str]:
    return [f'T{i:03d}.JK' for i in range(n)]


def _items(n: int = 400, unavailable_last: bool = True) -> pd.DataFrame:
    rows = []
    for i, ticker in enumerate(_tickers(n)):
        ready = not (unavailable_last and i == n - 1)
        if ready:
            payload = {
                'ticker': ticker,
                'completion_state': 'TECHNICAL_READY',
                'technical_ready': True,
                'signal': {
                    'ticker': ticker, 'status': 'WATCHLIST', 'setup_status': 'WATCHLIST',
                    'quality_score': 72.0 + (i % 7), 'momentum_score': 8.0,
                    'rr1': 2.0, 'rr2': 3.0, 'entry': 100.0, 'entry_low': 98.0,
                    'entry_high': 102.0, 'trigger': 103.0, 'trigger_price': 103.0,
                    'stop_loss': 92.0, 'tp1': 116.0, 'tp2': 124.0,
                    'adtv20_idr': 1_500_000_000.0, 'data_quality_score': 80.0,
                    'validation_score': 70.0, 'last_price': 101.0,
                },
                'silent_profile': {
                    'ticker': ticker, 'silent_accumulation_score': 72.0,
                    'silent_accumulation_confidence': 85.0,
                    'inventory_multi_horizon_score': 70.0,
                    'inventory_multi_horizon_coverage_pct': 100.0,
                    'distribution_risk_score': 15.0,
                    'inventory_lifecycle': 'INVENTORY_COLLECTION',
                    'reaccumulation_quality_score': 70.0,
                },
                'narrative_profile': {
                    'ticker': ticker, 'narrative_event_effective_score': 62.0,
                    'narrative_event_coverage_pct': 70.0,
                    'issuer_action_alignment_effective_score': 65.0,
                    'retail_adoption_stage': 'EARLY_AWARENESS',
                },
                'narrative_events': [], 'narrative_outcomes': [],
                'breadth': {'valid': True, 'above_ema50': i % 3 != 0, 'above_ema200': True, 'positive_20d': i % 4 != 0},
                'ohlcv_state': 'CURRENT', 'ohlcv_bars': 800,
                'ohlcv_last_bar_date': '2026-08-07', 'ohlcv_session_lag': 0,
                'ohlcv_source_tier': 'DATABASE_CURRENT_YAHOO',
            }
        else:
            payload = {
                'ticker': ticker, 'completion_state': 'TECHNICAL_UNAVAILABLE',
                'technical_ready': False, 'ohlcv_state': 'MISSING_OR_INSUFFICIENT',
                'ohlcv_bars': 0, 'ohlcv_source_tier': 'UNAVAILABLE',
            }
        rows.append({
            'job_id': 'TEST', 'item_key': f'K{i:04d}', 'ticker': ticker,
            'phase': 'TECHNICAL', 'status': 'COMPLETE', 'attempt_count': 1,
            'max_attempts': 1, 'result_payload': payload,
        })
    return pd.DataFrame(rows)


def _fundamentals(n: int = 118) -> pd.DataFrame:
    rows = []
    for i, ticker in enumerate(_tickers(n)):
        rows.append({
            'ticker': ticker, 'sector': 'ENERGY' if i % 2 == 0 else 'INDUSTRIALS',
            'company_name': f'Company {i}', 'revenue_growth': 0.20 + 0.001 * (i % 10),
            'earnings_growth': 0.28 + 0.001 * (i % 9), 'roe': 0.18,
            'roa': 0.10, 'operating_margin': 0.17, 'net_margin': 0.13,
            'operating_cash_flow': 100.0, 'free_cash_flow': 75.0,
            'debt_equity': 0.35, 'current_ratio': 1.8, 'cash_to_debt': 0.8,
            'fundamental_score': 78.0, 'fundamental_coverage': 88.0,
            'fundamental_source_count': 2, 'statement_age_days': 40.0,
            'fundamental_history_quarters': 3, 'fundamental_history_years': 1,
            'valuation_score': 6.5, 'peg_ratio': 1.1, 'fcf_yield': 0.06,
            'trailing_pe': 14.0, 'price_to_book': 2.0,
            'future_fundamental_impact_score': 72.0, 'project_pipeline_score': 68.0,
            'reinvestment_runway_pillar': 70.0, 'fundamental_inflection_score': 72.0,
            'history_revenue_growth_acceleration': 0.08,
            'history_earnings_growth_acceleration': 0.12,
            'management_quality_score': 72.0, 'capital_allocation_score': 70.0,
            'history_roic_proxy': 0.15, 'history_share_dilution_yoy': 0.0,
            'history_cash_conversion': 0.9, 'history_debt_equity': 0.35,
            'history_ocf_ttm': 100.0, 'history_fcf_ttm': 75.0,
            'latest_statement_date': '2026-06-30',
            'fundamental_history_latest_period': '2026-06-30',
            'fundamental_data_grade': 'B', 'fundamental_history_coverage': 85.0,
            'fundamental_cashflow_statement_coverage_pct': 100.0,
            'fundamental_official_source_coverage_pct': 0.0,
            'fundamental_official_verified': False,
        })
    return pd.DataFrame(rows)


def _history(n: int = 118) -> pd.DataFrame:
    rows = []
    for ticker in _tickers(n):
        for d in ('2025-12-31', '2026-03-31', '2026-06-30'):
            rows.append({'ticker': ticker, 'period_end': d, 'revenue': 100.0, 'net_income': 15.0})
    return pd.DataFrame(rows)


def _benchmark() -> pd.DataFrame:
    idx = pd.bdate_range('2025-01-02', periods=320)
    close = np.linspace(7000.0, 7800.0, len(idx))
    return pd.DataFrame({'Open': close, 'High': close * 1.01, 'Low': close * .99, 'Close': close, 'Volume': 1_000_000}, index=idx)


def _macro_result(universe: list[str]) -> MacroRegimeResult:
    issuer = pd.DataFrame([{
        'ticker': t, 'sector': 'ENERGY', 'issuer_macro_alignment_score': 72.0,
        'issuer_macro_alignment_coverage_pct': 90.0, 'issuer_macro_alignment_basis': 'SYNTHETIC_TEST',
        'market_regime': 'SELECTIVE_RISK_ON', 'market_context_score': 66.0,
        'market_context_coverage_pct': 100.0, 'market_context_provenance_state': 'TEST',
    } for t in universe])
    snapshot = pd.DataFrame([{
        'macro_regime': 'SELECTIVE_RISK_ON', 'macro_regime_score': 66.0,
        'macro_data_coverage_pct': 100.0, 'market_context_score': 66.0,
        'breadth_above_ema50_pct': 65.0, 'ihsg_return_20d': 0.08,
    }])
    return MacroRegimeResult(snapshot=snapshot, factors={}, sector_map=pd.DataFrame(), issuer_map=issuer, source_report=pd.DataFrame())


class FakeBridge:
    settings = SimpleNamespace(mode='FAKE')
    def persist_scan_result(self, *args, **kwargs):
        return pd.DataFrame([{'provider': 'FAKE_DB', 'state': 'WRITTEN'}])


def check_compile_and_imports() -> None:
    for path in ROOT.glob('*.py'):
        py_compile.compile(str(path), doraise=True)
    for name in ('scanner', 'scanner_database', 'simple_focus', 'real_money_guard', 'fundamental_calibration', 'resumable_app_engine', 'fast_scan_engine', 'v9_dashboard'):
        __import__(name)


def check_finalize_no_load_before_fundamental_map_assignment() -> None:
    source = (ROOT / 'resumable_app_engine.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'finalize_daily_scan_job')
    loads = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == 'fundamental_map' and isinstance(n.ctx, ast.Load)]
    stores = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Name) and n.id == 'fundamental_map' and isinstance(n.ctx, ast.Store)]
    assert stores and loads and min(stores) < min(loads), (stores, loads)


def check_partial_400_finalizer_end_to_end() -> None:
    universe = _tickers(400)
    items = _items(400, unavailable_last=True)
    fundamentals, history = _fundamentals(118), _history(118)
    benchmark = _benchmark()
    job = {
        'job_id': 'HOTFIX4-INTEGRATION', 'universe_payload': universe,
        'config_payload': {
            'period': '5y', 'evidence_refresh_cap': 8, 'decision_evidence_cap': 8,
            'evidence_fundamental_cap': 8, 'evidence_official_cap': 4,
            'evidence_snapshot_cap': 6, 'evidence_market_cap': 6,
            'evidence_news_cap': 6, 'execution_verification_cap': 6,
            'macro_external_enabled': True, 'macro_timeout_seconds': 3,
            'lean_persistence': True, 'lean_skip_narrative_history': True,
            'portfolio_records': [],
        },
        'started_at': '2026-08-09T00:00:00+00:00',
    }

    def fake_job_evidence(*args, **kwargs):
        return fundamentals.copy(), history.copy(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    def fake_job_benchmark(*args, **kwargs):
        return benchmark.copy(), pd.DataFrame()
    def fake_macro(*args, **kwargs):
        return _macro_result(universe)
    def fake_refresh(bridge, tickers, f, h, m, n, cfg, config):
        assert len(tickers) <= 8
        return f, h, m, n, pd.DataFrame([{'provider': 'TEST_REFRESH', 'status': 'OK'}])
    def fake_db_ohlcv(*args, **kwargs):
        return {}, None, pd.DataFrame()
    def boom_snapshots(*args, **kwargs):
        raise RuntimeError('simulated snapshot outage')
    def boom_independent(*args, **kwargs):
        raise RuntimeError('simulated independent outage')

    with patch.object(eng, '_job_evidence', side_effect=fake_job_evidence), \
         patch.object(eng, '_job_benchmark', side_effect=fake_job_benchmark), \
         patch.object(eng, 'fetch_macro_series', return_value=({}, pd.DataFrame())), \
         patch.object(eng, 'build_macro_regime', side_effect=fake_macro), \
         patch.object(eng, '_refresh_missing_daily_evidence', side_effect=fake_refresh), \
         patch.object(eng, '_database_first_ohlcv', side_effect=fake_db_ohlcv), \
         patch.object(eng, 'fetch_execution_snapshots', side_effect=boom_snapshots), \
         patch.object(eng, 'fetch_automatic_independent_prices', side_effect=boom_independent):
        out = eng.finalize_daily_scan_job(
            job, FakeBridge(), 'integration', items_override=items,
            durable_updates=False, persist_artifacts=False, return_result=True,
        )
    result = out['result']
    focus = result['focus_screens']
    assert out['completed_tickers'] == 399
    assert len(focus['next_leaders']) > 0, 'Partial 118/400 fundamentals must still produce research leaders'
    assert len(focus['swing_ready']) > 0, 'Technical + partial fundamentals must still produce research swing ranking'
    assert result['ranking_quality_state'] == 'VALID'
    assert isinstance(result['database_sync_report'], pd.DataFrame)


def check_enrichment_exception_is_fail_soft() -> None:
    universe = _tickers(20)
    items = _items(20, unavailable_last=False)
    fundamentals, history = _fundamentals(20), _history(20)
    job = {'job_id': 'FAILSOFT', 'universe_payload': universe, 'config_payload': {'portfolio_records': [], 'lean_persistence': True}}
    with patch.object(eng, '_job_evidence', return_value=(fundamentals, history, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())), \
         patch.object(eng, '_job_benchmark', return_value=(_benchmark(), pd.DataFrame())), \
         patch.object(eng, 'fetch_macro_series', side_effect=RuntimeError('macro down')), \
         patch.object(eng, 'build_macro_regime', side_effect=lambda **kwargs: _macro_result(universe)), \
         patch.object(eng, '_refresh_missing_daily_evidence', side_effect=RuntimeError('enrichment down')), \
         patch.object(eng, '_database_first_ohlcv', return_value=({}, None, pd.DataFrame())), \
         patch.object(eng, 'fetch_execution_snapshots', side_effect=RuntimeError('snapshot down')), \
         patch.object(eng, 'fetch_automatic_independent_prices', side_effect=RuntimeError('independent down')):
        out = eng.finalize_daily_scan_job(job, FakeBridge(), 'failsoft', items_override=items, durable_updates=False, persist_artifacts=False, return_result=True)
    assert 'result' in out and out['result']['ranking_state'] == 'FINAL'




def check_zero_fundamentals_still_yields_swing_research() -> None:
    universe = _tickers(40)
    items = _items(40, unavailable_last=False)
    job = {'job_id': 'ZERO-FUND', 'universe_payload': universe, 'config_payload': {'portfolio_records': [], 'lean_persistence': True}}
    with patch.object(eng, '_job_evidence', return_value=(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())), \
         patch.object(eng, '_job_benchmark', return_value=(_benchmark(), pd.DataFrame())), \
         patch.object(eng, 'fetch_macro_series', return_value=({}, pd.DataFrame())), \
         patch.object(eng, 'build_macro_regime', side_effect=lambda **kwargs: _macro_result(universe)), \
         patch.object(eng, '_refresh_missing_daily_evidence', side_effect=lambda bridge, tickers, f, h, m, n, cfg, config: (f, h, m, n, pd.DataFrame())), \
         patch.object(eng, '_database_first_ohlcv', return_value=({}, None, pd.DataFrame())), \
         patch.object(eng, 'fetch_execution_snapshots', return_value=pd.DataFrame()), \
         patch.object(eng, 'fetch_automatic_independent_prices', return_value=(pd.DataFrame(), pd.DataFrame())):
        out = eng.finalize_daily_scan_job(job, FakeBridge(), 'zero-fund', items_override=items, durable_updates=False, persist_artifacts=False, return_result=True)
    focus = out['result']['focus_screens']
    assert len(focus['swing_ready']) > 0, 'Swing research ranking must survive even when fundamental database is empty'
    # Next Leader is allowed to remain pending because its research contract needs >=35% coverage.



def check_warm_400_fast_path_with_real_finalizer() -> None:
    import fast_scan_engine as fse
    universe = _tickers(400)
    expected = fse._expected_completed_session().date().isoformat()
    payloads = {}
    for i, ticker in enumerate(universe):
        payloads[ticker] = {
            'ticker': ticker, 'completion_state': 'TECHNICAL_READY', 'technical_ready': True,
            'signal': {'ticker': ticker, 'status': 'WATCHLIST', 'setup_status': 'WATCHLIST', 'quality_score': 72, 'momentum_score': 8, 'rr1': 2, 'entry': 100, 'stop_loss': 92, 'tp1': 116, 'adtv20_idr': 1e9, 'data_quality_score': 80, 'validation_score': 70, 'last_price': 101},
            'silent_profile': {'ticker': ticker, 'silent_accumulation_score': 72, 'silent_accumulation_confidence': 85, 'inventory_multi_horizon_score': 70, 'inventory_multi_horizon_coverage_pct': 100, 'distribution_risk_score': 15, 'inventory_lifecycle': 'INVENTORY_COLLECTION'},
            'narrative_profile': {'ticker': ticker, 'narrative_event_effective_score': 62, 'narrative_event_coverage_pct': 70, 'issuer_action_alignment_effective_score': 65},
            'narrative_events': [], 'narrative_outcomes': [],
            'breadth': {'valid': True, 'above_ema50': True, 'above_ema200': True, 'positive_20d': True},
            'ohlcv_state': 'CURRENT', 'ohlcv_bars': 800, 'ohlcv_last_bar_date': expected,
            'ohlcv_session_lag': 0, 'ohlcv_source_tier': 'DATABASE_CURRENT_YAHOO',
        }

    class WarmBridge(FakeBridge):
        read_circuit_open = False; write_circuit_open = False; transport_circuit_open = False; transport_error = ''
        def transport_state(self): return 'FAKE'
        def read_feature_cache(self, tickers, **kwargs):
            return {t: payloads[t] for t in tickers}, pd.DataFrame([{'ticker': t, 'status': 'HIT_CURRENT'} for t in tickers])
        def write_feature_cache(self, *args, **kwargs):
            raise AssertionError('warm path must not recompute/write technical features')

    fund, hist, benchmark = _fundamentals(118), _history(118), _benchmark()
    with patch.object(fse, 'FastDatabaseBridge', WarmBridge), \
         patch.object(eng, '_job_evidence', return_value=(fund, hist, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())), \
         patch.object(eng, '_job_benchmark', return_value=(benchmark, pd.DataFrame())), \
         patch.object(eng, 'fetch_macro_series', return_value=({}, pd.DataFrame())), \
         patch.object(eng, 'build_macro_regime', side_effect=lambda **kwargs: _macro_result(universe)), \
         patch.object(eng, '_refresh_missing_daily_evidence', side_effect=lambda bridge, tickers, f, h, m, n, cfg, config: (f, h, m, n, pd.DataFrame())), \
         patch.object(eng, '_database_first_ohlcv', return_value=({}, None, pd.DataFrame())), \
         patch.object(eng, 'fetch_execution_snapshots', return_value=pd.DataFrame()), \
         patch.object(eng, 'fetch_automatic_independent_prices', return_value=(pd.DataFrame(), pd.DataFrame())):
        result = fse.run_fast_single_scan(universe)
    assert result['feature_cache_hits'] == 400
    assert result['feature_cache_refreshes'] == 0
    assert len(result['focus_screens']['next_leaders']) > 0
    assert len(result['focus_screens']['swing_ready']) > 0

def check_calendar_due_reaches_inner_refresh_filter() -> None:
    # Q1 is <210 days old on Aug 9 but the Q2/H1 reporting window is already open.
    f = pd.DataFrame([{
        'ticker': 'SUNI.JK', 'sector': 'ENERGY', 'fundamental_score_eligible': True,
        'statement_age_days': 131, 'latest_statement_date': '2026-03-31',
        'fundamental_history_latest_period': '2026-03-31',
    }])
    h = pd.DataFrame([
        {'ticker': 'SUNI.JK', 'period_end': '2025-12-31'},
        {'ticker': 'SUNI.JK', 'period_end': '2026-03-31'},
    ])
    class Bridge(FakeBridge):
        pass
    called = {'idx': []}
    def fake_idx(tickers, **kwargs):
        called['idx'].extend(tickers)
        return pd.DataFrame(), pd.DataFrame()
    with patch.object(eng, 'fetch_idx_fundamental_history', side_effect=fake_idx), \
         patch.object(eng, 'select_yahoo_fundamental_tickers', return_value=[]), \
         patch.object(eng, 'fetch_resilient_fundamentals', return_value=pd.DataFrame()), \
         patch.object(eng, 'enrich_fundamentals_with_history', side_effect=lambda x, y: x), \
         patch.object(eng, 'normalize_fundamental_classification', side_effect=lambda x: x), \
         patch.object(eng, '_mark_history_eligible', side_effect=lambda x: x):
        eng._refresh_missing_daily_evidence(
            Bridge(), ['SUNI.JK'], f, h, pd.DataFrame(), pd.DataFrame(), eng.ScanConfig(),
            {'daily_fundamental_refresh_limit': 1, 'daily_official_fundamental_refresh_limit': 1,
             'daily_snapshot_refresh_limit': 1, 'daily_market_refresh_limit': 0, 'daily_news_refresh_limit': 0},
        )
    assert called['idx'] == ['SUNI.JK'], called


def main() -> None:
    checks = [
        check_compile_and_imports,
        check_finalize_no_load_before_fundamental_map_assignment,
        check_partial_400_finalizer_end_to_end,
        check_enrichment_exception_is_fail_soft,
        check_zero_fundamentals_still_yields_swing_research,
        check_warm_400_fast_path_with_real_finalizer,
        check_calendar_due_reaches_inner_refresh_filter,
    ]
    for fn in checks:
        fn(); print('PASS', fn.__name__)
    print('VALIDATION_V9_8_2_HOTFIX4=PASS')


if __name__ == '__main__':
    main()
