from __future__ import annotations

import pandas as pd
import pytest

from evidence_independence_audit import (
    COMPONENT_ALIASES,
    PARTICIPANT_COMPONENT,
    audit_evidence_independence,
    build_component_matrix,
)


def _row(ticker: str, value: bool = True, **changes):
    row = {"ticker": ticker, **{component: value for component in COMPONENT_ALIASES}}
    row.update(changes)
    return row


def test_nine_components_remain_separate() -> None:
    matrix = build_component_matrix(pd.DataFrame([_row("BBCA")]))
    assert list(matrix.columns[:10]) == ["ticker", *COMPONENT_ALIASES]
    assert matrix.iloc[0]["available_component_count"] == 9


def test_missing_column_is_unknown_not_false() -> None:
    matrix, summary = audit_evidence_independence(
        pd.DataFrame([{"ticker": "BBCA", "fundamental_valid": True}])
    )
    assert pd.isna(matrix.iloc[0]["reported_ownership_changes"])
    coverage = summary["component_coverage"]["reported_ownership_changes"]
    assert coverage == {
        "known": 0, "available": 0, "unavailable": 0, "unknown": 1,
        "available_percentage_of_known": 0.0, "available_percentage_of_universe": 0.0,
    }


def test_aliases_accept_existing_coverage_output_without_combining_components() -> None:
    matrix = build_component_matrix(pd.DataFrame([{
        "ticker": "BBCA", "foreign_20_session_sufficient": True, "ohlcv_valid": True,
        "forward_evidence_available": False, "broker_sufficient": False, "fundamental_valid": True,
    }]))
    row = matrix.iloc[0]
    assert bool(row["foreign_flow_persistence"]) and bool(row["volume_value_turnover"])
    assert not bool(row["official_corporate_catalysts"])
    assert not bool(row[PARTICIPANT_COMPONENT]) and bool(row["fundamental_growth_quality"])
    assert row["known_component_count"] == 5


def test_pairwise_correlation_uses_only_known_pairs() -> None:
    rows = []
    for index in range(40):
        rows.append({
            "ticker": f"T{index:03d}",
            "reported_ownership_changes": index % 2 == 0,
            "foreign_flow_persistence": index % 2 == 0,
            "volume_value_turnover": pd.NA if index < 20 else index % 3 == 0,
        })
    _, summary = audit_evidence_independence(pd.DataFrame(rows), minimum_paired_rows=20)
    duplicate = summary["possible_duplicate_pairs"]
    assert duplicate == [{
        "left": "reported_ownership_changes", "right": "foreign_flow_persistence",
        "paired_rows": 40, "correlation": 1.0,
    }]
    volume_pair = next(item for item in summary["pairwise_overlap"] if item["left"] == "reported_ownership_changes" and item["right"] == "volume_value_turnover")
    assert volume_pair["paired_rows"] == 20


def test_participant_context_reports_counts_without_readiness_gate() -> None:
    rows = [
        _row("WITH", participant_broker_flow=True),
        _row("WITHOUT", participant_broker_flow=False),
        _row("ONLY", False, participant_broker_flow=True),
        {"ticker": "UNKNOWN", "fundamental_growth_quality": True},
    ]
    _, summary = audit_evidence_independence(pd.DataFrame(rows), minimum_paired_rows=2)
    context = summary["participant_context"]
    assert context["participant_known"] == 3 and context["participant_unknown"] == 1
    assert context["participant_available"] == 2 and context["participant_unavailable"] == 1
    assert context["participant_only_rows"] == 1
    assert summary["policy_effect"] == "MEASUREMENT_ONLY_NO_PRODUCTION_CHANGE"


def test_exact_400_ticker_fixture_counts_are_not_invented() -> None:
    universe = [f"T{index:03d}" for index in range(400)]
    rows = [
        {"ticker": ticker, "foreign_flow_persistence": index < 300, "participant_broker_flow": index < 100}
        for index, ticker in enumerate(universe)
    ]
    matrix, summary = audit_evidence_independence(pd.DataFrame(rows), universe=universe)
    assert len(matrix) == summary["total_universe"] == 400
    assert summary["component_coverage"]["foreign_flow_persistence"]["available"] == 300
    assert summary["component_coverage"][PARTICIPANT_COMPONENT]["available"] == 100
    assert summary["component_coverage"]["fundamental_growth_quality"]["unknown"] == 400


def test_requested_universe_retains_missing_tickers_as_unknown() -> None:
    matrix, summary = audit_evidence_independence(
        pd.DataFrame([_row("BBCA")]), universe=["BBCA", "BBRI"]
    )
    missing = matrix.set_index("ticker").loc["BBRI"]
    assert missing["known_component_count"] == 0
    assert summary["component_coverage"][PARTICIPANT_COMPONENT]["unknown"] == 1


def test_duplicate_ticker_rejected_to_prevent_double_counting() -> None:
    with pytest.raises(ValueError, match="EVIDENCE_TICKER_DUPLICATE"):
        build_component_matrix(pd.DataFrame([_row("BBCA"), _row("BBCA")]))


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"duplicate_correlation_threshold": -0.1}, "INVALID_CORRELATION_THRESHOLD"),
        ({"duplicate_correlation_threshold": 1.1}, "INVALID_CORRELATION_THRESHOLD"),
        ({"minimum_paired_rows": 1}, "INVALID_MINIMUM_PAIRED_ROWS"),
    ],
)
def test_invalid_audit_configuration_fails_closed(kwargs, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        audit_evidence_independence(pd.DataFrame([_row("BBCA")]), **kwargs)


def test_audit_module_does_not_import_scanner_decisions() -> None:
    source = __import__("pathlib").Path(__file__).resolve().parents[1].joinpath(
        "evidence_independence_audit.py"
    ).read_text().lower()
    assert "entry_price" not in source and "take_profit" not in source and "stop_loss" not in source
    assert "mandatory" not in source and "recommendation" not in source
