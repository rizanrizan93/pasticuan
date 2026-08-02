from __future__ import annotations

import os
from unittest import mock

import numpy as np
import pandas as pd

import scanner
from scanner import ScanConfig, ScanEngine


def _ohlcv(seed: int, bars: int = 280) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = (180.0 + seed) * np.exp(np.cumsum(rng.normal(0.00035, 0.011, bars)))
    open_ = close * (1.0 + rng.normal(0.0, 0.003, bars))
    return pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) * 1.008,
            "Low": np.minimum(open_, close) * 0.992,
            "Close": close,
            "Adj Close": close,
            "Volume": rng.integers(500_000, 8_000_000, bars),
        },
        index=pd.bdate_range("2025-01-02", periods=bars),
    )


def _fixture(count: int = 20):
    histories = {f"T{index:02d}.JK": _ohlcv(index + 1) for index in range(count)}
    benchmark = _ohlcv(900)
    cfg = ScanConfig(
        min_bars=120,
        narrative_enabled=False,
        incremental_cache_enabled=True,
        core_incremental_cache_enabled=True,
        core_incremental_cache_partitions=8,
    )
    return histories, benchmark, cfg


def _use_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("IDX_SCANNER_INCREMENTAL_CACHE_ENABLED", "1")
    monkeypatch.setenv("IDX_SCANNER_INCREMENTAL_DB", str(tmp_path / "core.sqlite3"))
    monkeypatch.setenv("IDX_SCANNER_CACHE_DIR", str(tmp_path))


def test_identical_core_scan_hits_summary_and_skips_indicator_rebuild(tmp_path, monkeypatch):
    _use_cache(monkeypatch, tmp_path)
    histories, benchmark, cfg = _fixture()
    first = ScanEngine(cfg).scan(histories, benchmark)
    with mock.patch("scanner.prepare_indicators", wraps=scanner.prepare_indicators) as wrapped:
        second = ScanEngine(cfg).scan(histories, benchmark)
    assert wrapped.call_count == 0
    states = second["core_incremental_cache_report"]
    assert "HIT" in set(states.loc[states["stage"].eq("core_summary"), "cache_state"])
    pd.testing.assert_frame_equal(first["signals"], second["signals"])
    pd.testing.assert_frame_equal(first["universe"], second["universe"])
    assert all(
        isinstance(frame.attrs.get("_silent_accumulation_profile"), dict)
        for frame in second["prepared"].values()
    )


def test_one_ticker_change_rebuilds_only_one_prepared_partition(tmp_path, monkeypatch):
    _use_cache(monkeypatch, tmp_path)
    histories, benchmark, cfg = _fixture(32)
    ScanEngine(cfg).scan(histories, benchmark)
    changed = {ticker: frame.copy() for ticker, frame in histories.items()}
    changed["T00.JK"].iloc[-1, changed["T00.JK"].columns.get_loc("Close")] += 1.0
    with mock.patch("scanner.prepare_indicators", wraps=scanner.prepare_indicators) as wrapped:
        result = ScanEngine(cfg).scan(changed, benchmark)
    assert 1 <= wrapped.call_count < len(histories)
    prepared_rows = result["core_incremental_cache_report"].loc[
        result["core_incremental_cache_report"]["stage"].str.startswith("core_prepared_partition_")
    ]
    assert int(prepared_rows["cache_state"].str.endswith("REBUILT").sum()) == 1
    assert int(prepared_rows["cache_state"].eq("HIT").sum()) >= 1


def test_benchmark_change_invalidates_every_prepared_partition(tmp_path, monkeypatch):
    _use_cache(monkeypatch, tmp_path)
    histories, benchmark, cfg = _fixture(24)
    ScanEngine(cfg).scan(histories, benchmark)
    changed_benchmark = benchmark.copy()
    changed_benchmark.iloc[-1, changed_benchmark.columns.get_loc("Close")] += 1.0
    with mock.patch("scanner.prepare_indicators", wraps=scanner.prepare_indicators) as wrapped:
        result = ScanEngine(cfg).scan(histories, changed_benchmark)
    assert wrapped.call_count == len(histories)
    prepared_rows = result["core_incremental_cache_report"].loc[
        result["core_incremental_cache_report"]["stage"].str.startswith("core_prepared_partition_")
    ]
    assert prepared_rows["cache_state"].str.endswith("REBUILT").all()


def test_cached_and_disabled_paths_are_decision_identical(tmp_path, monkeypatch):
    histories, benchmark, cfg = _fixture(12)
    monkeypatch.setenv("IDX_SCANNER_INCREMENTAL_CACHE_ENABLED", "0")
    uncached = ScanEngine(cfg).scan(histories, benchmark)
    _use_cache(monkeypatch, tmp_path)
    cached = ScanEngine(cfg).scan(histories, benchmark)
    pd.testing.assert_frame_equal(uncached["signals"], cached["signals"])
    pd.testing.assert_frame_equal(uncached["universe"], cached["universe"])
    assert uncached["market_context"] == cached["market_context"]
    assert uncached["asof"] == cached["asof"]
    for ticker in uncached["prepared"]:
        pd.testing.assert_frame_equal(uncached["prepared"][ticker], cached["prepared"][ticker])
        assert uncached["prepared"][ticker].attrs == cached["prepared"][ticker].attrs


def test_all_missing_volume_ticker_is_excluded_without_index_crash(tmp_path, monkeypatch):
    monkeypatch.setenv("IDX_SCANNER_INCREMENTAL_CACHE_ENABLED", "0")
    histories, benchmark, cfg = _fixture(2)
    histories["BAD.JK"] = _ohlcv(333)
    histories["BAD.JK"]["Volume"] = np.nan
    result = ScanEngine(cfg).scan(histories, benchmark)
    assert "BAD.JK" not in result["prepared"]
    assert set(result["universe"]["ticker"]) == {"T00.JK", "T01.JK"}


def test_core_cache_config_change_invalidates_summary(tmp_path, monkeypatch):
    _use_cache(monkeypatch, tmp_path)
    histories, benchmark, cfg = _fixture(10)
    ScanEngine(cfg).scan(histories, benchmark)
    changed_cfg = cfg.replace(min_score=cfg.min_score + 1.0)
    result = ScanEngine(changed_cfg).scan(histories, benchmark)
    summary_states = result["core_incremental_cache_report"].loc[
        result["core_incremental_cache_report"]["stage"].eq("core_summary"), "cache_state"
    ]
    assert "HIT" not in set(summary_states)
