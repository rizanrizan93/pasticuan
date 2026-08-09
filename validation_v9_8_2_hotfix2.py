from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch
import json
import pathlib
import py_compile

import numpy as np
import pandas as pd
import requests

import fast_scan_engine as fse
from scanner_database import DatabaseTransportError, ScannerDatabaseBridge
from simple_focus import build_next_leaders, build_swing_ready
from v9_dashboard import render_dashboard_html, select_top_candidates

ROOT = pathlib.Path(__file__).resolve().parent


def check_compile() -> None:
    for path in ROOT.glob('*.py'):
        py_compile.compile(str(path), doraise=True)


def check_independent_db_circuits() -> None:
    bridge = fse.FastDatabaseBridge()
    object.__setattr__(bridge.settings, 'mode', 'SUPABASE_REST')
    object.__setattr__(bridge.settings, 'supabase_url', 'https://example.invalid')
    object.__setattr__(bridge.settings, 'supabase_key', 'secret')
    object.__setattr__(bridge.settings, 'supabase_key_type', 'SERVICE_ROLE')
    calls = {'get': 0, 'post': 0}

    def boom_get(*args, **kwargs):
        calls['get'] += 1
        raise requests.ReadTimeout('simulated read timeout')

    class OkResponse:
        ok = True
        status_code = 201
        text = ''
        def json(self): return []

    def ok_post(*args, **kwargs):
        calls['post'] += 1
        return OkResponse()

    with patch('scanner_database.requests.get', side_effect=boom_get), patch('scanner_database.requests.post', side_effect=ok_post):
        for _ in range(2):
            try:
                bridge._get_rows('scanner_feature_cache', {'select': 'ticker', 'limit': '1'})
            except DatabaseTransportError:
                pass
        assert bridge.read_circuit_open is True
        assert bridge.write_circuit_open is False
        # A read outage must not prevent persistence.
        bridge._upsert_supabase('scanner_feature_cache', [{'ticker': 'A.JK'}])
    assert calls['post'] == 1
    assert bridge.transport_state() == 'READ_DEGRADED_WRITE_AVAILABLE'
    assert bridge.transport_circuit_open is False


def check_feature_cache_metadata_first() -> None:
    bridge = ScannerDatabaseBridge.__new__(ScannerDatabaseBridge)
    bridge.settings = SimpleNamespace(mode='SUPABASE_REST', read_enabled=True, read_batch_size=80)
    calls: list[tuple[str, dict]] = []
    def fake_get(table, params):
        calls.append((table, dict(params)))
        select = params.get('select','')
        if 'payload' not in select:
            return [
                {'ticker':'A.JK','last_bar_date':'2026-08-07','feature_state':'CURRENT','source_tier':'YAHOO','scanner_version':'9.8.2-all-eligible-lite','feature_schema_version':'ALL_ELIGIBLE_LITE_V1','content_hash':'x','updated_at':'2026-08-08T00:00:00Z'},
                {'ticker':'B.JK','last_bar_date':'2026-08-06','feature_state':'CURRENT','source_tier':'YAHOO','scanner_version':'9.8.2-all-eligible-lite','feature_schema_version':'ALL_ELIGIBLE_LITE_V1','content_hash':'y','updated_at':'2026-08-08T00:00:00Z'},
            ]
        return [{'ticker':'A.JK','payload':{'ticker':'A.JK','technical_ready':True,'completion_state':'TECHNICAL_READY','signal':{'status':'WATCHLIST'},'ohlcv_last_bar_date':'2026-08-07'}}]
    bridge._get_rows = fake_get
    hits, audit = bridge.read_feature_cache(['A.JK','B.JK'], expected_session='2026-08-07', scanner_version='9.8.2-all-eligible-lite')
    assert list(hits) == ['A.JK']
    assert 'payload' not in calls[0][1]['select']
    assert any(call[1]['select'] == 'ticker,payload' for call in calls[1:])
    assert audit.loc[audit['ticker'].eq('A.JK'),'status'].iloc[0] == 'HIT_CURRENT'


def partial_universe() -> pd.DataFrame:
    return pd.DataFrame([{
        'ticker':'TEST.JK','fund_sector':'ENERGY','fund_revenue_growth':0.15,'fund_earnings_growth':0.20,'fund_roe':0.14,
        'fund_operating_margin':0.12,'fund_net_margin':0.10,'fund_statement_age_days':150,
        'mac_issuer_macro_alignment_score':70,'mac_issuer_macro_alignment_coverage_pct':80,
        'mac_market_regime':'SELECTIVE_RISK_ON','mac_market_context_score':66,'mac_market_context_coverage_pct':100,
        'sig_setup_status':'WATCHLIST','sig_status':'WATCHLIST','sig_quality_score':65,'sig_momentum_score':8,
        'sig_rr1':1.8,'sig_entry':100,'sig_stop_loss':92,'sig_tp1':115,'sig_adtv20_idr':1e9,
        'sig_data_quality_score':70,'sig_validation_score':60,
        'flow_silent_accumulation_score':68,'flow_silent_accumulation_confidence':75,
        'flow_inventory_lifecycle':'INVENTORY_COLLECTION','flow_distribution_risk_score':20,
        'flow_inventory_multi_horizon_score':70,
    }])


def check_research_ranking_separate_from_production() -> None:
    leaders = build_next_leaders(partial_universe())
    row = leaders.iloc[0]
    assert row['status'] == 'RESEARCH_ONLY'
    assert np.isnan(row['v9_next_leader_score'])
    assert np.isfinite(row['ranking_score'])
    assert bool(row['rank_eligible']) is True
    assert bool(row['production_rank_eligible']) is False
    assert row['real_money_authorization_state'] == 'REAL_MONEY_BLOCKED'

    swings = build_swing_ready(partial_universe(), leaders)
    assert bool(swings.iloc[0]['rank_eligible']) is True


def check_dashboard_research_score_fallback() -> None:
    leaders = build_next_leaders(partial_universe())
    top = select_top_candidates(leaders, model='NEXT_LEADER', limit=3)
    assert len(top) == 1
    html = render_dashboard_html(top, model='NEXT_LEADER', scan_id='hotfix2')
    assert 'TEST' in html
    assert 'RANKING SCORE' in html
    assert 'RESEARCH ONLY' in html


def check_compact_feature_payload() -> None:
    payload = {
        'ticker':'A.JK','technical_ready':True,'completion_state':'TECHNICAL_READY',
        'signal':{'ticker':'A.JK','status':'WATCHLIST','quality_score':70,'nested':[1,2,3]},
        'silent_profile':{'silent_accumulation_score':66,'large_list':[1,2]},
        'narrative_profile':{'narrative_score':55},
        'narrative_events':[{'headline':'x'}]*50,
        'narrative_outcomes':[{'x':1}]*50,
        'breadth':{'valid':True,'above_ema50':True},
        'ohlcv_bars':800,'ohlcv_last_bar_date':'2026-08-07','ohlcv_source_tier':'YAHOO',
    }
    compact = fse._compact_feature_payload(payload)
    assert compact['narrative_events'] == []
    assert compact['narrative_outcomes'] == []
    assert 'nested' not in compact['signal']
    assert len(json.dumps(compact)) < len(json.dumps(payload))


def main() -> None:
    checks = [
        check_compile,
        check_independent_db_circuits,
        check_feature_cache_metadata_first,
        check_research_ranking_separate_from_production,
        check_dashboard_research_score_fallback,
        check_compact_feature_payload,
    ]
    for fn in checks:
        fn(); print('PASS', fn.__name__)
    print('VALIDATION_V9_8_2_HOTFIX2=PASS')


if __name__ == '__main__':
    main()
