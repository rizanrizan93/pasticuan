from __future__ import annotations

import pytest

from fast_scan_engine import run_fast_single_scan


def test_fast_scan_refuses_silent_truncation_above_400():
    tickers = [f"T{i:03d}.JK" for i in range(401)]
    with pytest.raises(ValueError, match="UNIVERSE_EXCEEDS_400:401"):
        run_fast_single_scan(tickers)