from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import patch
import pathlib, py_compile
import pandas as pd

import resumable_app_engine as eng
from scanner_database import ScannerDatabaseBridge, DatabaseSettings

ROOT=pathlib.Path(__file__).resolve().parent


def check_compile() -> None:
    for p in ROOT.glob('*.py'):
        py_compile.compile(str(p), doraise=True)


def check_zero_budget_respected() -> None:
    source=(ROOT/'resumable_app_engine.py').read_text(encoding='utf-8')
    assert 'requested_evidence_cap = max(0, _int_config(config, "evidence_refresh_cap", 16))' in source
    assert 'requested_decision_cap = max(0, _int_config(config, "decision_evidence_cap", 12))' in source
    assert 'requested_verify_cap = max(0, _int_config(config, "execution_verification_cap", 10))' in source
    assert 'config.get("evidence_refresh_cap", 16) or 16' not in source
    assert 'config.get("execution_verification_cap", 10) or 10' not in source


def check_malformed_feature_cache_rejected() -> None:
    class Bridge(ScannerDatabaseBridge):
        def __init__(self, malformed: bool, schema='ALL_ELIGIBLE_LITE_V1'):
            super().__init__(DatabaseSettings(enabled=True, mode='SUPABASE_REST', supabase_url='x', supabase_key='k', supabase_key_type='SERVICE_ROLE', read_enabled=True))
            self.malformed=malformed; self.schema=schema
        def _get_rows(self, table, params):
            sel=params['select']
            if 'payload' not in sel:
                return [{'ticker':'TEST.JK','last_bar_date':'2026-08-07','feature_state':'CURRENT','source_tier':'X','scanner_version':'9.8.2-all-eligible-lite','feature_schema_version':self.schema,'content_hash':'x','updated_at':'now'}]
            if self.malformed:
                return [{'ticker':'TEST.JK','payload':{'ticker':'TEST.JK','technical_ready':False,'completion_state':'TECHNICAL_UNAVAILABLE','ohlcv_last_bar_date':'2026-08-07'}}]
            return [{'ticker':'TEST.JK','payload':{'ticker':'TEST.JK','technical_ready':True,'completion_state':'TECHNICAL_READY','signal':{'ticker':'TEST.JK','status':'WATCHLIST'},'ohlcv_last_bar_date':'2026-08-07'}}]
    hits,audit=Bridge(True).read_feature_cache(['TEST.JK'], expected_session='2026-08-07', scanner_version='9.8.2-all-eligible-lite')
    assert not hits, hits
    assert 'PAYLOAD_READ_FAIL_SOFT' in set(audit['status'].astype(str)), audit
    hits2,audit2=Bridge(False, schema='WRONG_V0').read_feature_cache(['TEST.JK'], expected_session='2026-08-07', scanner_version='9.8.2-all-eligible-lite')
    assert not hits2, hits2
    hits3,audit3=Bridge(False).read_feature_cache(['TEST.JK'], expected_session='2026-08-07', scanner_version='9.8.2-all-eligible-lite')
    assert 'TEST.JK' in hits3, (hits3,audit3)


def main() -> None:
    for fn in [check_compile, check_zero_budget_respected, check_malformed_feature_cache_rejected]:
        fn(); print('PASS',fn.__name__)
    print('VALIDATION_V9_8_2_HOTFIX5=PASS')

if __name__=='__main__': main()
