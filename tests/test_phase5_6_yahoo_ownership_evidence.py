from __future__ import annotations

from datetime import date

import pandas as pd

from shared_yahoo_ownership_evidence import (
    INSTITUTIONAL_PROVIDER,
    LINEAGE_STATE,
    normalize_major_holders,
    normalize_named_holders,
    validate_concentration_rows,
)


def test_major_holders_are_concentration_facts_not_free_float() -> None:
    frame = pd.DataFrame(
        {"Value": [0.60814, 0.19018, 0.48531, 374.0]},
        index=["insidersPercentHeld", "institutionsPercentHeld", "institutionsFloatPercentHeld", "institutionsCount"],
    )
    rows = normalize_major_holders("BBCA.JK", frame, observed_on=date(2026, 9, 4))
    assert len(rows) == 4
    metrics = {row["metric_name"]: row for row in rows}
    assert metrics["insiders_held_pct"]["metric_value"] == 60.814
    assert metrics["institutions_held_pct"]["metric_value"] == 19.018
    assert metrics["institutions_float_held_pct"]["metric_value"] == 48.531
    assert metrics["institutions_count"]["metric_value"] == 374.0
    assert all(row["official_verified"] is False for row in rows)
    assert all(row["source_authority"] == "PUBLIC_PROVIDER" for row in rows)
    assert all(row["lineage_state"] == LINEAGE_STATE for row in rows)
    assert not any("free_float" in row["metric_name"] for row in rows)
    assert validate_concentration_rows(rows) == (True, "VALID")


def test_named_holder_preserves_report_date_and_percentage() -> None:
    frame = pd.DataFrame([{
        "Date Reported": pd.Timestamp("2026-07-31"),
        "Holder": "Example Emerging Markets Fund",
        "pctHeld": 0.0053,
        "Shares": 641453900,
        "Value": 4345850172500,
        "pctChange": 0.0,
    }])
    rows = normalize_named_holders(
        "BBCA",
        frame,
        provider=INSTITUTIONAL_PROVIDER,
        holder_category="INSTITUTIONAL_DISCLOSURE",
        observed_on=date(2026, 9, 4),
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["source_period"] == "2026-07-31"
    assert row["holder_name"] == "Example Emerging Markets Fund"
    assert row["shares_held"] == 641453900.0
    assert row["ownership_percentage"] == 0.53
    assert row["provider"] == INSTITUTIONAL_PROVIDER


def test_unknown_major_labels_are_not_fabricated() -> None:
    frame = pd.DataFrame({"Value": [0.25]}, index=["unknownYahooMetric"])
    rows = normalize_major_holders("BBCA", frame, observed_on=date(2026, 9, 4))
    assert rows == []
