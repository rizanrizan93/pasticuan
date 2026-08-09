from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd


def synthetic_ohlcv(rows: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(982)
    idx = pd.bdate_range('2023-01-02', periods=rows)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, rows)))
    return pd.DataFrame({
        'Open': close * (1.0 + rng.normal(0, 0.002, rows)),
        'High': close * 1.012,
        'Low': close * 0.988,
        'Close': close,
        'Volume': rng.integers(10_000, 3_000_000, rows),
    }, index=idx)


def check_compact_ohlcv_roundtrip() -> None:
    from scanner_database import ScannerDatabaseBridge
    frame = synthetic_ohlcv()
    legacy = ScannerDatabaseBridge._ohlcv_payload(frame, max_bars=900)
    compact, codec, bars = ScannerDatabaseBridge._ohlcv_compact_encode(frame, max_bars=900)
    restored = ScannerDatabaseBridge._ohlcv_compact_decode(compact, codec)
    legacy_bytes = len(json.dumps(legacy).encode('utf-8'))
    compact_bytes = len(compact.encode('ascii'))
    assert bars == 900
    assert len(restored) == 900
    assert compact_bytes < legacy_bytes * 0.40, (compact_bytes, legacy_bytes)
    assert abs(float(restored['Close'].iloc[-1]) - float(frame['Close'].iloc[-1])) < 1e-5


def check_feature_cache_contract() -> None:
    source = Path('scanner_database.py').read_text(encoding='utf-8')
    migration = Path('database/migration_v15_database_acceleration.sql').read_text(encoding='utf-8')
    assert 'def read_feature_cache' in source
    assert 'def write_feature_cache' in source
    assert 'scanner_feature_cache' in migration
    assert 'payload_compact' in migration
    assert 'MIGRATION_REQUIRED_V15' in source


def check_all_eligible_lite_orchestration() -> None:
    source = Path('fast_scan_engine.py').read_text(encoding='utf-8')
    assert 'FAST_SCAN_VERSION = "9.8.2-all-eligible-lite"' in source
    assert 'read_feature_cache' in source
    assert 'compute_items' in source
    assert 'feature_cache_hits' in source
    assert '.create_scan_job(' not in source
    assert '.claim_scan_job_items(' not in source


def check_maintenance_reserve() -> None:
    source = Path('resumable_app_engine.py').read_text(encoding='utf-8')
    assert 'maintenance_reserve' in source
    assert 'evidence_cap - maintenance_reserve' in source
    assert 'ENGINE_VERSION = "9.8.2"' in source


def check_ui_contract() -> None:
    source = Path('app.py').read_text(encoding='utf-8')
    assert 'APP_VERSION = "9.8.2-hotfix' in source
    assert 'IDX Super Scanner v9.8.2 Hotfix' in source
    assert 'feature-cache hit' in source
    assert 'st.button("SCAN"' in source
    assert 'Isi Database' not in source


def check_fast_path_cache_hits_without_technical_recompute() -> None:
    import fast_scan_engine as fse

    universe = [f'T{i:03d}.JK' for i in range(20)]
    expected = fse._expected_completed_session().date().isoformat()
    payloads = {
        ticker: {
            'ticker': ticker,
            'completion_state': 'TECHNICAL_READY',
            'technical_ready': True,
            'signal': {'ticker': ticker, 'status': 'WATCHLIST', 'last_price': 100 + i},
            'silent_profile': {'ticker': ticker, 'silent_accumulation_score': 50.0},
            'narrative_profile': {'ticker': ticker, 'narrative_score': 50.0},
            'narrative_events': [],
            'narrative_outcomes': [],
            'breadth': {'valid': True, 'above_ema50': True, 'above_ema200': True, 'positive_20d': True},
            'ohlcv_state': 'CURRENT',
            'ohlcv_bars': 800,
            'ohlcv_last_bar_date': expected,
            'ohlcv_session_lag': 0,
            'ohlcv_source_tier': 'DATABASE_CURRENT_YAHOO',
        }
        for i, ticker in enumerate(universe)
    }

    class FakeBridge:
        transport_circuit_open = False
        transport_error = ''
        settings = SimpleNamespace(mode='FAKE')
        def read_feature_cache(self, tickers, **kwargs):
            return {t: payloads[t] for t in tickers}, pd.DataFrame([{'ticker': t, 'status': 'HIT_CURRENT'} for t in tickers])
        def write_feature_cache(self, *args, **kwargs):
            raise AssertionError('No recompute means no feature write')

    original_bridge = fse.FastDatabaseBridge
    original_process = fse.process_daily_scan_chunk
    original_finalize = fse.finalize_daily_scan_job
    try:
        fse.FastDatabaseBridge = FakeBridge
        def forbidden_process(*args, **kwargs):
            raise AssertionError('Technical engine must not run for current feature-cache hits')
        fse.process_daily_scan_chunk = forbidden_process
        def fake_finalize(job, bridge, worker_id, **kwargs):
            items = kwargs['items_override']
            assert len(items) == len(universe)
            assert items['status'].eq('COMPLETE').all()
            return {'result': {
                'focus_screens': {}, 'prepared': {}, 'stage_timings': pd.DataFrame([{}]),
                'scan_coverage_summary': pd.DataFrame([{'requested_tickers': len(universe), 'ohlcv_ready_tickers': len(universe)}]),
            }}
        fse.finalize_daily_scan_job = fake_finalize
        result = fse.run_fast_single_scan(universe)
        assert result['feature_cache_hits'] == len(universe)
        assert result['feature_cache_refreshes'] == 0
        assert result['all_eligible_state'] == 'ALL_ELIGIBLE_LITE'
    finally:
        fse.FastDatabaseBridge = original_bridge
        fse.process_daily_scan_chunk = original_process
        fse.finalize_daily_scan_job = original_finalize



def check_cold_cache_feature_write_no_nameerror() -> None:
    from scanner_database import ScannerDatabaseBridge

    bridge = ScannerDatabaseBridge.__new__(ScannerDatabaseBridge)
    bridge.settings = SimpleNamespace(mode='SUPABASE_REST')
    captured = {}

    def fake_upsert(table, records):
        captured['table'] = table
        captured['records'] = records
        return len(records)

    bridge._upsert_supabase = fake_upsert
    expected = pd.Timestamp('2026-08-07')
    report = bridge.write_feature_cache({
        'TEST.JK': {
            'ticker': 'TEST.JK',
            'ohlcv_last_bar_date': expected,
            'ohlcv_source_tier': 'DATABASE_CURRENT_YAHOO',
            'technical_ready': True,
            'signal': {'ticker': 'TEST.JK', 'status': 'WATCHLIST'},
        }
    }, scanner_version='9.8.2-all-eligible-lite')
    assert captured['table'] == 'scanner_feature_cache'
    assert len(captured['records']) == 1
    row = captured['records'][0]
    assert row['ticker'] == 'TEST.JK'
    assert row['content_hash']
    assert report.iloc[0]['status'] == 'WRITTEN'

def main() -> None:
    checks = [
        check_compact_ohlcv_roundtrip,
        check_feature_cache_contract,
        check_all_eligible_lite_orchestration,
        check_maintenance_reserve,
        check_ui_contract,
        check_fast_path_cache_hits_without_technical_recompute,
        check_cold_cache_feature_write_no_nameerror,
    ]
    for check in checks:
        check()
        print('PASS', check.__name__)
    print('PASS v9.8.2 database acceleration validation')


if __name__ == '__main__':
    main()
