from __future__ import annotations

import ast
import importlib
import pathlib
import py_compile
from unittest.mock import patch

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent


def check_compile() -> None:
    for path in ROOT.glob('*.py'):
        if path.name.startswith('validation_'):
            continue
        py_compile.compile(str(path), doraise=True)


def check_imports() -> None:
    for name in (
        'resumable_scan', 'resumable_app_engine', 'scanner_database',
        'decision_overlay', 'v9_dashboard', 'scanner', 'simple_focus',
    ):
        importlib.import_module(name)


def check_single_scan_sidebar() -> None:
    source = (ROOT / 'app.py').read_text(encoding='utf-8')
    assert 'operation_mode =' not in source
    assert 'st.radio("Mode kerja"' not in source
    assert 'Mulai / Lanjutkan Isi Database' not in source
    sidebar = source.split('with st.sidebar:', 1)[1].split('# -----------------------------------------------------------------------------\n# One-button durable workflow.', 1)[0]
    assert sidebar.count('st.button(') == 1
    assert 'st.button("SCAN", type="primary", width="stretch")' in source
    assert 'job_type = "DAILY_SCAN"' in source
    # The UI worker must no longer depend on the backfill processors.
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == 'resumable_app_engine'
        for alias in node.names
    }
    assert imported == {'finalize_daily_scan_job', 'process_daily_scan_chunk'}


def check_auto_database_config() -> None:
    source = (ROOT / 'app.py').read_text(encoding='utf-8')
    for key in (
        'daily_fundamental_refresh_limit',
        'daily_snapshot_refresh_limit',
        'daily_news_refresh_limit',
        'daily_official_fundamental_refresh_limit',
    ):
        assert f'"{key}": int(job_chunk_size)' in source


def check_official_first_refresh_contract() -> None:
    import resumable_app_engine as eng

    calls: list[str] = []

    class Bridge:
        def persist_scan_result(self, result):
            calls.append('persist')
            assert result.get('mode') == 'daily_delta_refresh'
            return pd.DataFrame([{'table': 'fundamental_cache', 'state': 'OK'}])

    base = pd.DataFrame([{
        'ticker': 'TEST.JK',
        'sector': 'UNKNOWN',
        'fundamental_score_eligible': False,
        'fundamental_score': 50.0,
        'fundamental_coverage': 50.0,
        'fundamental_source_count': 1,
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
        # Official history is already sufficient; Yahoo should not be selected.
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
        eng._refresh_missing_daily_evidence(
            Bridge(), ['TEST.JK'], base, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
            eng.ScanConfig(), {
                'daily_fundamental_refresh_limit': 1,
                'daily_snapshot_refresh_limit': 1,
                'daily_news_refresh_limit': 1,
                'daily_official_fundamental_refresh_limit': 1,
            },
        )

    assert calls.index('idx') < calls.index('select_yahoo')
    assert 'snapshot' in calls
    assert 'persist' in calls


def check_version_contract() -> None:
    import resumable_app_engine as eng
    import resumable_scan as rs
    assert eng.ENGINE_VERSION == '9.7.2'
    assert rs.RESUMABLE_SCAN_VERSION == '9.7.2'
    app_source = (ROOT / 'app.py').read_text(encoding='utf-8')
    assert 'APP_VERSION = "9.7.2-single-scan-database-first"' in app_source


if __name__ == '__main__':
    checks = [
        check_compile,
        check_imports,
        check_single_scan_sidebar,
        check_auto_database_config,
        check_official_first_refresh_contract,
        check_version_contract,
    ]
    for check in checks:
        check()
        print('PASS', check.__name__)
    print('VALIDATION_V9_7_2=PASS')
