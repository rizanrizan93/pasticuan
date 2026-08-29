from __future__ import annotations

import pytest

from fast_scan_engine import reconcile_requested_universe, run_fast_single_scan


def test_fast_scan_refuses_silent_truncation_above_400():
    tickers = [f"T{i:03d}.JK" for i in range(401)]
    with pytest.raises(ValueError, match="UNIVERSE_EXCEEDS_400:401"):
        run_fast_single_scan(tickers)


def test_portfolio_extras_never_displace_uploaded_400_universe():
    uploaded = [f"T{i:03d}.JK" for i in range(400)]
    universe, extras = reconcile_requested_universe(
        uploaded,
        ["PORT.JK", uploaded[0]],
        max_tickers=400,
    )

    assert universe == uploaded
    assert len(universe) == 400
    assert extras == ["PORT.JK"]
    assert "PORT.JK" not in universe
