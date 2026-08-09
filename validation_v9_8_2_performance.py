from __future__ import annotations

import json
import time
import numpy as np
import pandas as pd

import validation_v9_8_0_performance as legacy
from scanner_database import ScannerDatabaseBridge


def check_v981_performance_regressions() -> None:
    checks = [
        legacy.check_compile,
        legacy.check_imports,
        legacy.check_single_scan_sidebar,
        legacy.check_staged_budget_contract,
        legacy.check_job_level_evidence_cache_contract,
        legacy.check_official_first_refresh_contract,
        legacy.check_persist_allowlist,
        legacy.check_systemic_write_fails_fast,
        legacy.check_row_level_isolation_is_bounded,
        legacy.check_write_retry_and_table_circuit,
        legacy.check_eod_cutoff_consistency,
        legacy.check_batch_transport,
    ]
    for check in checks:
        check()


def benchmark_compact_ohlcv() -> None:
    rng = np.random.default_rng(9820)
    idx = pd.bdate_range('2023-01-02', periods=900)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, len(idx))))
    frame = pd.DataFrame({
        'Open': close * (1 + rng.normal(0, .002, len(idx))),
        'High': close * 1.01,
        'Low': close * .99,
        'Close': close,
        'Volume': rng.integers(10000, 2500000, len(idx)),
    }, index=idx)
    start = time.perf_counter()
    legacy_payload = ScannerDatabaseBridge._ohlcv_payload(frame, max_bars=900)
    compact, codec, bars = ScannerDatabaseBridge._ohlcv_compact_encode(frame, max_bars=900)
    restored = ScannerDatabaseBridge._ohlcv_compact_decode(compact, codec)
    elapsed = time.perf_counter() - start
    legacy_bytes = len(json.dumps(legacy_payload).encode('utf-8'))
    compact_bytes = len(compact.encode('ascii'))
    ratio = compact_bytes / legacy_bytes
    assert bars == 900 and len(restored) == 900
    assert ratio < .40
    assert elapsed < 1.0
    print(f'BENCH compact_900bars ratio={ratio:.3f} encode_decode={elapsed:.3f}s legacy_bytes={legacy_bytes} compact_bytes={compact_bytes}')


def main() -> None:
    check_v981_performance_regressions()
    print('PASS check_v981_performance_regressions')
    benchmark_compact_ohlcv()
    print('PASS benchmark_compact_ohlcv')
    print('VALIDATION_V9_8_2_PERFORMANCE=PASS')


if __name__ == '__main__':
    main()
