from pathlib import Path


def test_valuation_accepts_verified_ksei_source_as_share_count_provenance():
    source = Path("scanner.py").read_text(encoding="utf-8")
    assert "_truthy(row.get('ksei_source_verified', False))" in source
    assert "row.get('source_checked_at')" in source
    assert "ksei_shares_current = bool(" in source
    assert "derived_market_cap = (" in source
    assert "price * shares_adjusted" in source
    assert "fcf_yield_derived = (" in source


def test_valuation_keeps_split_adjustment_and_market_cap_bounds():
    source = Path("scanner.py").read_text(encoding="utf-8")
    assert "_valuation_split_adjustment(" in source
    assert "10_000_000_000.0 <= derived_market_cap <= 20_000_000_000_000_000.0" in source
