from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from shared_evidence_coverage_metrics import (
    CACHE_COUNTERS,
    FULLY_EVIDENCE_READY_COMPONENTS,
    METRIC_SPECS,
    build_shared_coverage_report,
)


AS_OF = "2026-08-31"


def _complete(ticker: str, *, value: bool = True) -> dict:
    row = {"ticker": ticker, **{metric: value for metric in METRIC_SPECS}}
    row.update({
        "market_latest_session": AS_OF,
        "fundamental_latest_publication": "2026-08-20",
        "foreign_latest_session": AS_OF,
        "ownership_latest_publication": "2026-08-25",
        "forward_latest_evidence_date": "2026-08-26",
        "capital_latest_evidence_date": "2026-08-27",
        "risk_latest_evidence_date": "2026-08-28",
        "participant_latest_session": AS_OF,
        "forward_source": "IDX_ANNOUNCEMENT",
        "participant_ids": "YP|CC",
        "participant_history_depth": 20,
    })
    return row


def _consumption(scanner: str, ticker: str, metrics=FULLY_EVIDENCE_READY_COMPONENTS) -> pd.DataFrame:
    return pd.DataFrame([
        {"scanner": scanner, "ticker": ticker, "metric": metric, "consumed": True}
        for metric in metrics
    ])


def test_all_task36_families_metrics_dates_and_cache_counters_are_reported() -> None:
    detail, report = build_shared_coverage_report(
        ["AAA.JK", "BBB"],
        pd.DataFrame([_complete("AAA"), _complete("BBB", value=False)]),
        scanner_consumption=pd.concat([_consumption("EMIR", "AAA"), _consumption("PASTICUAN", "AAA")]),
        cache_counters={name: index + 1 for index, name in enumerate(CACHE_COUNTERS)},
        as_of=AS_OF,
    )
    assert set(report["shared_factual_coverage"]) == {
        "MARKET", "FUNDAMENTALS", "FOREIGN", "OWNERSHIP", "FORWARD",
        "CAPITAL_STRUCTURE", "RISK", "PARTICIPANT",
    }
    assert len(METRIC_SPECS) == 36
    market = report["shared_factual_coverage"]["MARKET"]["market_ohlcv"]
    assert market["numerator"] == 1 and market["denominator"] == 2
    assert market["percentage"] == 50.0 and market["latest_source_date"] == AS_OF
    assert report["shared_cache"] == {name: index + 1 for index, name in enumerate(CACHE_COUNTERS)}
    assert list(detail["ticker"]) == ["AAA", "BBB"]


def test_shared_fact_coverage_and_each_scanner_consumption_remain_separate() -> None:
    facts = pd.DataFrame([_complete("AAA"), _complete("BBB")])
    consumption = pd.concat([
        _consumption("EMIR", "AAA", ["market_ohlcv"]),
        _consumption("PASTICUAN", "BBB", ["market_stock_summary"]),
    ])
    _, report = build_shared_coverage_report(
        ["AAA", "BBB"], facts, scanner_consumption=consumption, as_of=AS_OF,
    )
    shared = report["shared_factual_coverage"]["MARKET"]
    assert shared["market_ohlcv"]["numerator"] == 2
    assert shared["market_stock_summary"]["numerator"] == 2
    assert report["scanner_consumption"]["EMIR"]["market_ohlcv"]["consumed_count"] == 1
    assert report["scanner_consumption"]["EMIR"]["market_stock_summary"]["consumed_count"] == 0
    assert report["scanner_consumption"]["PASTICUAN"]["market_stock_summary"]["consumed_count"] == 1


def test_fully_evidence_ready_has_exact_definition_count_and_percentage() -> None:
    partial = _complete("BBB")
    partial["ownership_historical_delta"] = False
    detail, report = build_shared_coverage_report(
        ["AAA", "BBB"], pd.DataFrame([_complete("AAA"), partial]), as_of=AS_OF,
    )
    assert detail.set_index("ticker")["fully_evidence_ready"].to_dict() == {"AAA": True, "BBB": False}
    ready = report["fully_evidence_ready"]
    assert ready["required_metrics"] == list(FULLY_EVIDENCE_READY_COMPONENTS)
    assert ready["count"] == 1 and ready["denominator"] == 2 and ready["percentage"] == 50.0
    assert report["policy_effect"] == "MEASUREMENT_ONLY_NO_SCORING_OR_GATE_CHANGE"


def test_observation_metrics_preserve_breadth_depth_source_distribution_and_latest_dates() -> None:
    second = _complete("BBB")
    second.update({
        "forward_source": "IDX_PRESS_RELEASE", "participant_ids": "CC|PD",
        "participant_history_depth": 7, "ownership_latest_publication": "2026-08-30",
    })
    _, report = build_shared_coverage_report(
        ["AAA", "BBB"], pd.DataFrame([_complete("AAA"), second]), as_of=AS_OF,
    )
    observed = report["observations"]
    assert observed["ownership_latest_publication"] == "2026-08-30"
    assert observed["forward_source_distribution"] == {"IDX_ANNOUNCEMENT": 1, "IDX_PRESS_RELEASE": 1}
    assert observed["participant_breadth_distinct"] == 3
    assert observed["participant_historical_session_depth_min"] == 7
    assert observed["participant_historical_session_depth_max"] == 20


def test_future_dated_fact_is_not_available_at_cutoff() -> None:
    row = _complete("AAA")
    row["market_latest_session"] = "2026-09-01"
    detail, report = build_shared_coverage_report(["AAA"], pd.DataFrame([row]), as_of=AS_OF)
    assert not bool(detail.iloc[0]["market_ohlcv"])
    assert report["shared_factual_coverage"]["MARKET"]["market_ohlcv"]["numerator"] == 0
    assert report["observations"]["market_latest_session"] == ""


def test_unknown_is_counted_and_never_promoted_to_false_known_evidence() -> None:
    detail, report = build_shared_coverage_report(["AAA", "MISS"], pd.DataFrame([_complete("AAA")]), as_of=AS_OF)
    metric = report["shared_factual_coverage"]["MARKET"]["market_ohlcv"]
    assert metric["known"] == 1 and metric["unknown"] == 1
    assert pd.isna(detail.set_index("ticker").loc["MISS", "market_ohlcv"])


def test_phantom_consumption_duplicate_identity_and_bad_cache_counter_fail_closed() -> None:
    false_row = _complete("AAA", value=False)
    phantom = _consumption("EMIR", "AAA", ["market_ohlcv"])
    with pytest.raises(ValueError, match="CONSUMPTION_WITHOUT_SHARED_FACT"):
        build_shared_coverage_report(["AAA"], pd.DataFrame([false_row]), scanner_consumption=phantom, as_of=AS_OF)
    duplicate = pd.concat([phantom, phantom])
    with pytest.raises(ValueError, match="CONSUMPTION_IDENTITY_DUPLICATE"):
        build_shared_coverage_report(["AAA"], pd.DataFrame([_complete("AAA")]), scanner_consumption=duplicate, as_of=AS_OF)
    with pytest.raises(ValueError, match="CACHE_COUNTER_INVALID"):
        build_shared_coverage_report(["AAA"], pd.DataFrame([_complete("AAA")]), cache_counters={"provider_calls": -1}, as_of=AS_OF)


def test_exact_400_fixture_reports_measured_counts_without_live_calls() -> None:
    universe = [f"T{index:03d}" for index in range(400)]
    facts = pd.DataFrame([_complete(ticker, value=index < 300) for index, ticker in enumerate(universe)])
    _, report = build_shared_coverage_report(universe, facts, as_of=AS_OF)
    metric = report["shared_factual_coverage"]["FOREIGN"]["foreign_20_session_sufficient"]
    assert report["universe_count"] == 400
    assert metric["numerator"] == 300 and metric["denominator"] == 400 and metric["percentage"] == 75.0


def test_coverage_module_is_scanner_neutral_and_contains_no_decision_fields() -> None:
    source = Path(__file__).resolve().parents[1].joinpath("shared_evidence_coverage_metrics.py").read_text().lower()
    for forbidden in ("entry_price", "take_profit", "stop_loss", "recommendation", "ranking"):
        assert forbidden not in source
