from __future__ import annotations

import importlib
import inspect
import pathlib
import py_compile
from unittest.mock import patch

import requests

import fast_scan_engine
import resumable_app_engine
import scanner
from scanner_database import DatabaseTransportError

ROOT = pathlib.Path(__file__).resolve().parent


def check_compile_imports() -> None:
    for path in ROOT.glob('*.py'):
        py_compile.compile(str(path), doraise=True)
    for name in (
        'scanner', 'scanner_database', 'macro_engine', 'simple_focus',
        'decision_overlay', 'real_money_guard', 'v9_dashboard',
        'resumable_app_engine', 'fast_scan_engine',
    ):
        importlib.import_module(name)


def check_ui_has_no_durable_repository_dependency() -> None:
    source = (ROOT / 'app.py').read_text(encoding='utf-8')
    assert 'APP_VERSION = "9.8.4-calibration-integrity"' in source
    assert 'run_fast_single_scan' in source
    assert 'create_or_resume_scan_job' not in source
    assert 'read_latest_scan_job' not in source
    assert 'read_scan_job(' not in source
    assert 'run_durable_job_loop' not in source
    assert 'st.button("SCAN", type="primary", width="stretch")' in source


def check_fast_database_circuit() -> None:
    bridge = fast_scan_engine.FastDatabaseBridge()
    # Force REST shape even when local env does not configure Supabase.
    object.__setattr__(bridge.settings, 'mode', 'SUPABASE_REST')
    object.__setattr__(bridge.settings, 'supabase_url', 'https://example.invalid')
    object.__setattr__(bridge.settings, 'supabase_key', 'secret')
    object.__setattr__(bridge.settings, 'supabase_key_type', 'SERVICE_ROLE')
    calls = {'get': 0}
    def boom(*args, **kwargs):
        calls['get'] += 1
        raise requests.ReadTimeout('simulated')
    with patch('scanner_database.requests.get', side_effect=boom):
        for _ in range(2):
            try:
                bridge._get_rows('fundamental_cache', {'select':'ticker','limit':'1'})
            except DatabaseTransportError:
                pass
            else:
                raise AssertionError('transport timeout must raise')
        # Third read fails immediately from the read circuit without another HTTP call.
        try:
            bridge._get_rows('fundamental_cache', {'select':'ticker','limit':'1'})
        except DatabaseTransportError:
            pass
        else:
            raise AssertionError('open read circuit must fail immediately')
    assert calls['get'] == 2, calls
    assert bridge.read_circuit_open is True
    assert bridge.write_circuit_open is False
    assert bridge.transport_circuit_open is False
    assert bridge.settings.read_attempts == 1
    assert bridge.settings.write_attempts == 1
    assert bridge.settings.timeout_seconds <= 8


def check_provider_runtime_is_bounded() -> None:
    source = inspect.getsource(scanner._download_ohlcv_v431)
    public = inspect.signature(scanner.download_ohlcv)
    assert public.parameters['batch_size'].default == 80
    assert "IDX_SCANNER_ENABLE_INDIVIDUAL_YF_RETRY', 'false'" in source
    assert "IDX_SCANNER_YAHOO_RETRY_COUNT', '0'" in source
    assert "IDX_SCANNER_YAHOO_MAX_WORKERS', '12'" in source


def check_single_pass_budget() -> None:
    source = (ROOT / 'fast_scan_engine.py').read_text(encoding='utf-8')
    for token in (
        'cfg.setdefault("evidence_refresh_cap", 8)',
        'cfg.setdefault("evidence_official_cap", 4)',
        'cfg.setdefault("execution_verification_cap", 6)',
        'cfg.setdefault("macro_external_enabled", True)',
        'cfg.setdefault("macro_timeout_seconds", 3)',
        'cfg.setdefault("lean_skip_narrative_history", True)',
    ):
        assert token in source
    final_sig = inspect.signature(resumable_app_engine.finalize_daily_scan_job)
    assert 'items_override' in final_sig.parameters
    assert 'durable_updates' in final_sig.parameters
    assert 'persist_artifacts' in final_sig.parameters
    assert 'return_result' in final_sig.parameters


def check_market_refresh_bounded() -> None:
    source = inspect.getsource(resumable_app_engine._refresh_missing_daily_evidence)
    assert 'daily_market_refresh_limit' in source
    assert 'market_targets = [ticker for ticker in names if ticker not in market_present][:market_limit]' in source


def main() -> None:
    checks = [
        check_compile_imports,
        check_ui_has_no_durable_repository_dependency,
        check_fast_database_circuit,
        check_provider_runtime_is_bounded,
        check_single_pass_budget,
        check_market_refresh_bounded,
    ]
    for check in checks:
        check(); print('PASS', check.__name__)
    print('VALIDATION_V9_8_1=PASS')


if __name__ == '__main__':
    main()
