from __future__ import annotations

import pytest

from resumable_app_engine import _universe_metadata


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        ("false", False),
        ("", False),
        (None, False),
    ],
)
def test_universe_metadata_uses_public_truthy_and_preserves_uploaded_fields(raw_value, expected):
    records = [
        {
            "ticker": "TEST",
            "idx_sector": "Technology",
            "rank_universe": 7,
            "universe_role": "PORTFOLIO",
            "priority": "HIGH",
            "active_scan": "yes",
            "portfolio_held": raw_value,
        },
        {
            "ticker": "KEEP",
            "idx_sector": "Energy",
            "rank_universe": 8,
            "universe_role": "PORTFOLIO",
            "priority": "NORMAL",
            "active_scan": "yes",
            "portfolio_held": True,
        },
        {
            "ticker": "OUTSIDE",
            "portfolio_held": True,
        },
    ]

    frame = _universe_metadata(records, ["TEST", "KEEP"])

    assert frame["ticker"].tolist() == ["TEST.JK", "KEEP.JK"]
    assert frame["portfolio_held"].map(type).eq(bool).all()
    assert frame.set_index("ticker").loc["TEST.JK", "portfolio_held"] == expected
    assert frame.set_index("ticker").loc["KEEP.JK", "portfolio_held"] == True  # noqa: E712
    test_row = frame.set_index("ticker").loc["TEST.JK"]
    assert test_row["universe_rank"] == 7
    assert test_row["universe_role"] == "PORTFOLIO"
    assert test_row["universe_priority"] == "HIGH"
    assert test_row["universe_active_scan"] == "yes"
    assert test_row["universe_metadata_source"] == "UPLOADED_UNIVERSE"
