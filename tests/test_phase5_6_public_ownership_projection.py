from __future__ import annotations

from shared_public_ownership_projection import (
    PROVENANCE_STATE,
    SOURCE_PROVIDER,
    canonicalize_ownership_concentration,
)


def _row(metric: str, value: float, *, period: str = "2026-09-01", observed: str = "2026-09-04", fetched: str = "2026-09-04T00:00:00+00:00") -> dict[str, object]:
    return {
        "provider": SOURCE_PROVIDER,
        "ticker": "BBCA",
        "source_period": period,
        "observed_on": observed,
        "metric_name": metric,
        "metric_value": value,
        "metric_unit": "COUNT" if metric == "institutions_count" else "PERCENT",
        "source_authority": "PUBLIC_PROVIDER",
        "official_verified": False,
        "lineage_state": "PUBLIC_PROVIDER_OBSERVED_NOT_IDX_KSEI",
        "validation_state": "VALID",
        "fetched_at": fetched,
    }


def test_projection_is_context_only_and_four_fields_equal_full_coverage() -> None:
    rows = [
        _row("insiders_held_pct", 60.8),
        _row("institutions_held_pct", 19.0),
        _row("institutions_float_held_pct", 48.5),
        _row("institutions_count", 374.0),
    ]
    item = canonicalize_ownership_concentration(rows)["BBCA"]
    assert item["coverage_pct"] == 100.0
    assert item["official_verified"] is False
    assert item["provenance_state"] == PROVENANCE_STATE
    assert "free_float" not in item
    assert "ownership_score" not in item


def test_latest_row_wins_per_metric_without_inference() -> None:
    rows = [
        _row("institutions_held_pct", 18.0, period="2026-08-01", fetched="2026-08-02T00:00:00+00:00"),
        _row("institutions_held_pct", 19.0, period="2026-09-01", fetched="2026-09-04T00:00:00+00:00"),
        _row("institutions_count", 374.0),
    ]
    item = canonicalize_ownership_concentration(rows)["BBCA"]
    assert item["institutions_held_pct"] == 19.0
    assert item["coverage_pct"] == 50.0
    assert item["insiders_held_pct"] is None


def test_official_or_wrong_authority_rows_are_rejected() -> None:
    official = _row("insiders_held_pct", 60.0)
    official["official_verified"] = True
    wrong = _row("institutions_held_pct", 20.0)
    wrong["source_authority"] = "OFFICIAL"
    assert canonicalize_ownership_concentration([official, wrong]) == {}
