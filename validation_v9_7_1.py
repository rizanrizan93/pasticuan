from __future__ import annotations

import importlib
import inspect
import pathlib

import pandas as pd

from scanner_database import ScannerDatabaseBridge
from resumable_app_engine import _expected_completed_session, finalize_daily_scan_job

ROOT = pathlib.Path(__file__).resolve().parent


def check_compile() -> None:
    for path in ROOT.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')


def check_imports() -> None:
    for name in (
        'scanner_database', 'resumable_scan', 'resumable_app_engine',
        'scanner', 'simple_focus', 'decision_overlay', 'v9_dashboard',
    ):
        importlib.import_module(name)


def check_batch_checkpoint_transport() -> None:
    class FakeBridge(ScannerDatabaseBridge):
        def __init__(self):
            self.calls: list[tuple[str, int]] = []
        def _upsert_supabase(self, table, records):
            self.calls.append((str(table), len(records)))
            return len(records)

    bridge = FakeBridge()
    checkpoints = [
        {
            'item': {'item_key': f'k{i}', 'ticker': f'T{i}.JK', 'attempt_count': 1, 'max_attempts': 2},
            'success': True,
            'payload': {'ticker': f'T{i}.JK'},
        }
        for i in range(20)
    ]
    states = bridge.checkpoint_scan_job_items_batch(checkpoints)
    assert len(states) == 20
    assert bridge.calls == [('scan_job_items', 20)], bridge.calls


def check_batch_artifact_transport() -> None:
    class FakeBridge(ScannerDatabaseBridge):
        def __init__(self):
            self.calls: list[tuple[str, int]] = []
        def _upsert_supabase(self, table, records):
            self.calls.append((str(table), len(records)))
            return len(records)

    bridge = FakeBridge()
    keys = bridge.persist_scan_job_artifacts_batch('job', {f'A{i}': {'i': i} for i in range(11)})
    assert len(keys) == 11
    assert bridge.calls == [('scan_job_artifacts', 11)], bridge.calls


def check_eod_cutoff_consistency() -> None:
    before = _expected_completed_session(pd.Timestamp('2026-08-07 16:15:00+07:00'))
    after = _expected_completed_session(pd.Timestamp('2026-08-07 16:25:00+07:00'))
    weekend = _expected_completed_session(pd.Timestamp('2026-08-08 16:28:00+07:00'))
    assert before.date().isoformat() == '2026-08-06'
    assert after.date().isoformat() == '2026-08-07'
    assert weekend.date().isoformat() == '2026-08-07'


def check_finalizer_fast_contract() -> None:
    src = inspect.getsource(finalize_daily_scan_job)
    assert 'execution_verification_cap", 24' in src
    assert 'min_bars=260, max_stale_sessions=0, force_refresh=False' in src
    assert 'ThreadPoolExecutor(max_workers=2)' in src
    assert 'persist_scan_job_artifacts_batch' in src
    assert 'phase="DATABASE_SYNC"' in src
    assert 'phase="ARTIFACT_PUBLISH"' in src


def main() -> None:
    checks = [
        check_compile,
        check_imports,
        check_batch_checkpoint_transport,
        check_batch_artifact_transport,
        check_eod_cutoff_consistency,
        check_finalizer_fast_contract,
    ]
    for check in checks:
        check()
        print('PASS', check.__name__)
    print('VALIDATION_V9_7_1=PASS')


if __name__ == '__main__':
    main()
