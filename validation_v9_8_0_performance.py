from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
import py_compile
from unittest.mock import patch

import pandas as pd

from scanner_database import (
    DatabaseSettings,
    DatabaseWriteError,
    ScannerDatabaseBridge,
)
from resumable_app_engine import (
    _expected_completed_session,
    _refresh_missing_daily_evidence,
    finalize_daily_scan_job,
    process_daily_scan_chunk,
)

ROOT = pathlib.Path(__file__).resolve().parent


def check_compile() -> None:
    for path in ROOT.glob('*.py'):
        if path.name.startswith('validation_'):
            continue
        py_compile.compile(str(path), doraise=True)


def check_imports() -> None:
    for name in (
        'scanner_database', 'resumable_scan', 'resumable_app_engine',
        'scanner', 'simple_focus', 'decision_overlay', 'v9_dashboard',
    ):
        importlib.import_module(name)


def check_single_scan_sidebar() -> None:
    source = (ROOT / 'app.py').read_text(encoding='utf-8')
    assert 'operation_mode =' not in source
    assert 'st.radio("Mode kerja"' not in source
    assert 'Mulai / Lanjutkan Isi Database' not in source
    assert 'st.button("SCAN", type="primary", width="stretch")' in source
    assert 'job_type = "DAILY_SCAN"' in source


def check_staged_budget_contract() -> None:
    app_source = (ROOT / 'app.py').read_text(encoding='utf-8')
    # Critical regression guard: deep evidence limits must never equal chunk size.
    for key in (
        'daily_fundamental_refresh_limit',
        'daily_snapshot_refresh_limit',
        'daily_news_refresh_limit',
        'daily_official_fundamental_refresh_limit',
    ):
        assert f'"{key}": int(job_chunk_size)' not in app_source
    assert '"evidence_refresh_cap": 16' in app_source
    assert '"execution_verification_cap": 10' in app_source

    chunk_source = inspect.getsource(process_daily_scan_chunk)
    final_source = inspect.getsource(finalize_daily_scan_job)
    assert '_refresh_missing_daily_evidence(' not in chunk_source
    assert final_source.count('_refresh_missing_daily_evidence(') == 1
    assert 'phase="EVIDENCE_REFRESH"' in final_source
    assert 'execution_verification_cap", 10' in final_source
    assert 'min(\n        12,' in final_source


def check_job_level_evidence_cache_contract() -> None:
    import resumable_app_engine as eng
    chunk_source = inspect.getsource(eng.process_daily_scan_chunk)
    assert '_job_evidence(' in chunk_source
    assert '_load_fundamentals(bridge, tickers' not in chunk_source
    assert '_load_aux(bridge, tickers' not in chunk_source
    assert 'read_narrative_events(tickers' not in chunk_source


def check_official_first_refresh_contract() -> None:
    import resumable_app_engine as eng
    calls: list[str] = []

    class Bridge:
        def persist_scan_result(self, result, **kwargs):
            calls.append('persist')
            assert result.get('mode') == 'daily_delta_refresh'
            assert set(kwargs.get('tables') or ()) == {
                'fundamental_snapshots', 'fundamental_cache',
                'fundamental_history_cache', 'refresh_state',
                'source_events', 'provider_health',
            }
            return pd.DataFrame([{'table': 'fundamental_cache', 'state': 'OK'}])

    base = pd.DataFrame([{
        'ticker': 'TEST.JK', 'sector': 'UNKNOWN',
        'fundamental_score_eligible': False, 'fundamental_score': 50.0,
        'fundamental_coverage': 50.0, 'fundamental_source_count': 1,
        'statement_age_days': 999,
    }])
    idx_hist = pd.DataFrame([
        {'ticker': 'TEST.JK', 'period_end': '2025-12-31', 'source_family': 'IDX_OFFICIAL_XBRL'},
        {'ticker': 'TEST.JK', 'period_end': '2026-03-31', 'source_family': 'IDX_OFFICIAL_XBRL'},
    ])

    def fake_idx(*args, **kwargs):
        calls.append('idx')
        return idx_hist.copy(), pd.DataFrame([{'provider': 'IDX_OFFICIAL_XBRL', 'status': 'OK'}])

    def fake_select(tickers, history, **kwargs):
        calls.append('select_yahoo')
        assert len(history) >= 2
        return []

    def fake_snapshot(*args, **kwargs):
        calls.append('snapshot')
        return base.assign(sector='Technology', fundamental_score_eligible=True).copy()

    with patch.object(eng, 'normalize_fundamental_classification', side_effect=lambda x: x), \
         patch.object(eng, '_mark_history_eligible', side_effect=lambda x: x), \
         patch.object(eng, 'enrich_fundamentals_with_history', side_effect=lambda snapshot, history: snapshot), \
         patch.object(eng, 'fetch_idx_fundamental_history', side_effect=fake_idx), \
         patch.object(eng, 'select_yahoo_fundamental_tickers', side_effect=fake_select), \
         patch.object(eng, 'fetch_resilient_fundamentals', side_effect=fake_snapshot), \
         patch.object(eng, 'fetch_resilient_market_status', return_value=pd.DataFrame([{'ticker': 'TEST.JK'}])), \
         patch.object(eng, 'fetch_resilient_news_review', return_value=pd.DataFrame([{'ticker': 'TEST.JK'}])):
        _refresh_missing_daily_evidence(
            Bridge(), ['TEST.JK'], base, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
            eng.ScanConfig(), {
                'daily_fundamental_refresh_limit': 1,
                'daily_snapshot_refresh_limit': 1,
                'daily_news_refresh_limit': 1,
                'daily_official_fundamental_refresh_limit': 1,
            },
        )
    assert calls.index('idx') < calls.index('select_yahoo')
    assert 'snapshot' in calls and 'persist' in calls


def check_persist_allowlist() -> None:
    class FakeBridge(ScannerDatabaseBridge):
        def __init__(self):
            self.settings = DatabaseSettings(enabled=True, mode='OUTBOX_ONLY')
            self.writes: list[str] = []
            self._write_details = {}
        def _ensure_scan_id(self, result):
            return 'scan-test'
        def build_payloads(self, result):
            return {'alpha': [{'id': 1}], 'beta': [{'id': 2}]}
        def _write_outbox(self, table, records):
            self.writes.append(str(table))
            return len(records)

    bridge = FakeBridge()
    report = bridge.persist_scan_result({}, tables=('beta',))
    assert bridge.writes == ['beta']
    assert list(report['table']) == ['beta']


def check_systemic_write_fails_fast() -> None:
    class FakeBridge(ScannerDatabaseBridge):
        def __init__(self):
            self.calls = 0
        def _post_batch(self, table, batch):
            self.calls += 1
            raise DatabaseWriteError(table, 503, 'service unavailable')

    bridge = FakeBridge()
    failures: list[str] = []
    written = bridge._write_with_isolation('fundamental_snapshots', [{'i': i} for i in range(100)], failures)
    assert written == 0
    assert bridge.calls == 1, f'systemic error recursively retried {bridge.calls} batch requests'
    assert len(failures) == 1


def check_row_level_isolation_is_bounded() -> None:
    class FakeBridge(ScannerDatabaseBridge):
        def __init__(self):
            self.calls = 0
        def _post_batch(self, table, batch):
            self.calls += 1
            if any(row.get('bad') for row in batch):
                raise DatabaseWriteError(table, 422, 'row validation error')
            return len(batch)

    bridge = FakeBridge()
    rows = [{'id': i, 'bad': i == 3} for i in range(8)]
    failures: list[str] = []
    written = bridge._write_with_isolation('technical_snapshots', rows, failures)
    assert 0 < written < len(rows)
    assert bridge.calls <= 12
    assert failures



def check_write_retry_and_table_circuit() -> None:
    class Response:
        status_code = 503
        ok = False
        text = 'service unavailable'

    bridge = ScannerDatabaseBridge.__new__(ScannerDatabaseBridge)
    bridge.settings = DatabaseSettings(
        enabled=True, mode='SUPABASE_REST', supabase_url='https://example.invalid',
        supabase_key='secret', supabase_key_type='SERVICE_ROLE',
        read_attempts=3, write_attempts=2, retry_backoff_seconds=0.001,
    )
    bridge._write_details = {}
    calls = {'post': 0}

    def fake_post(*args, **kwargs):
        calls['post'] += 1
        return Response()

    with patch('scanner_database.requests.post', side_effect=fake_post):
        try:
            bridge._post_batch('scan_runs', [{'snapshot_id': 'x'}])
        except DatabaseWriteError:
            pass
        else:
            raise AssertionError('503 must raise DatabaseWriteError')
    assert calls['post'] == 2, calls

    class CircuitBridge(ScannerDatabaseBridge):
        def __init__(self):
            self.settings = DatabaseSettings(enabled=True, mode='SUPABASE_REST')
            self._write_details = {}
            self.calls: list[str] = []
        def _ensure_scan_id(self, result):
            return 'scan-circuit'
        def build_payloads(self, result):
            return {
                'alpha': [{'id': 1}], 'beta': [{'id': 2}], 'gamma': [{'id': 3}],
            }
        def _upsert_supabase(self, table, records):
            self.calls.append(str(table))
            self._write_details[str(table)] = 'isolated_failures=1; alpha: HTTP 503; service unavailable'
            return 0

    circuit = CircuitBridge()
    report = circuit.persist_scan_result({}, tables=('alpha', 'beta', 'gamma'))
    assert circuit.calls == ['alpha'], circuit.calls
    states = dict(zip(report['table'], report['state']))
    assert states['alpha'] == 'DATABASE_FAIL_SOFT'
    assert states['beta'] == 'DATABASE_CIRCUIT_OPEN'
    assert states['gamma'] == 'DATABASE_CIRCUIT_OPEN'

def check_eod_cutoff_consistency() -> None:
    before = _expected_completed_session(pd.Timestamp('2026-08-07 16:15:00+07:00'))
    after = _expected_completed_session(pd.Timestamp('2026-08-07 16:25:00+07:00'))
    weekend = _expected_completed_session(pd.Timestamp('2026-08-08 16:28:00+07:00'))
    assert before.date().isoformat() == '2026-08-06'
    assert after.date().isoformat() == '2026-08-07'
    assert weekend.date().isoformat() == '2026-08-07'


def check_batch_transport() -> None:
    class FakeBridge(ScannerDatabaseBridge):
        def __init__(self):
            self.calls: list[tuple[str, int]] = []
        def _upsert_supabase(self, table, records):
            self.calls.append((str(table), len(records)))
            return len(records)

    bridge = FakeBridge()
    checkpoints = [
        {'item': {'item_key': f'k{i}', 'ticker': f'T{i}.JK', 'attempt_count': 1, 'max_attempts': 2},
         'success': True, 'payload': {'ticker': f'T{i}.JK'}}
        for i in range(20)
    ]
    states = bridge.checkpoint_scan_job_items_batch(checkpoints)
    assert len(states) == 20
    assert bridge.calls == [('scan_job_items', 20)]

    bridge2 = FakeBridge()
    keys = bridge2.persist_scan_job_artifacts_batch('job', {f'A{i}': {'i': i} for i in range(11)})
    assert len(keys) == 11
    assert bridge2.calls == [('scan_job_artifacts', 11)]


def check_version_contract() -> None:
    import resumable_app_engine as eng
    import resumable_scan as rs
    assert eng.ENGINE_VERSION == '9.8.0'
    assert rs.RESUMABLE_SCAN_VERSION == '9.8.0'
    app_source = (ROOT / 'app.py').read_text(encoding='utf-8')
    assert 'APP_VERSION = "9.8.0-guarded-real-money"' in app_source


def main() -> None:
    checks = [
        check_compile, check_imports, check_single_scan_sidebar,
        check_staged_budget_contract, check_job_level_evidence_cache_contract,
        check_official_first_refresh_contract, check_persist_allowlist,
        check_systemic_write_fails_fast, check_row_level_isolation_is_bounded,
        check_write_retry_and_table_circuit, check_eod_cutoff_consistency,
        check_batch_transport, check_version_contract,
    ]
    for check in checks:
        check()
        print('PASS', check.__name__)
    print('VALIDATION_V9_8_0_PERFORMANCE_REGRESSION=PASS')


if __name__ == '__main__':
    main()
