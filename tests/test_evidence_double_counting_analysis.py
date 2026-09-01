from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from evidence_double_counting_analysis import SIGNALS, analyze_double_counting


def _rows(count: int = 40) -> list[dict]:
    rows = []
    for index in range(count):
        common = float(index)
        rows.append({
            "ticker": f"T{index:03d}",
            "foreign_accumulation": common,
            "ownership_accumulation": float((index * 7) % 13),
            "participant_accumulation": float(index % 5),
            "volume_turnover": float(index % 9),
            "existing_silent_accumulation": common,
            "inventory_proxies": common * 2,
            "trend_momentum": float(index % 4),
            "smc_ict_structure": float((index % 4) + (0.01 if index % 3 else 0)),
            "fundamental_growth": float((index * 11) % 17),
            "corporate_catalysts": float(index % 2),
            "dilution": float(index % 2),
        })
    return rows


def _pair(summary: dict, left: str, right: str) -> dict:
    return next(
        item for item in summary["pairwise_analysis"]
        if {item["left"], item["right"]} == {left, right}
    )


def test_all_eleven_required_signals_are_analyzed_without_combining_them() -> None:
    matrix, summary = analyze_double_counting(pd.DataFrame(_rows()), minimum_paired_rows=20)
    assert list(matrix.columns) == ["ticker", *SIGNALS]
    assert len(summary["signals"]) == 11
    assert len(summary["pairwise_analysis"]) == 55


def test_semantically_overlapping_near_identical_signals_are_possible_double_counting() -> None:
    _, summary = analyze_double_counting(pd.DataFrame(_rows()), minimum_paired_rows=20)
    pair = _pair(summary, "foreign_accumulation", "existing_silent_accumulation")
    assert pair["spearman_correlation"] == 1.0
    assert pair["semantic_overlap"]
    assert pair["classification"] == "POSSIBLE_DOUBLE_COUNTING"
    assert summary["possible_double_counting"]


def test_high_correlation_without_declared_semantic_overlap_is_not_auto_double_counted() -> None:
    _, summary = analyze_double_counting(pd.DataFrame(_rows()), minimum_paired_rows=20)
    pair = _pair(summary, "foreign_accumulation", "fundamental_growth")
    if pair["spearman_correlation"] is not None and abs(pair["spearman_correlation"]) >= 0.70:
        assert pair["classification"] == "HIGHLY_CORRELATED_EVIDENCE"
    assert pair["classification"] != "POSSIBLE_DOUBLE_COUNTING"


def test_unknown_values_use_known_pairs_only_and_small_samples_are_insufficient() -> None:
    rows = _rows(10)
    for row in rows[:5]:
        row.pop("dilution")
    matrix, summary = analyze_double_counting(pd.DataFrame(rows), minimum_paired_rows=10)
    assert summary["signal_observability"]["dilution"] == {"known": 5, "unknown": 5}
    pair = _pair(summary, "corporate_catalysts", "dilution")
    assert pair["paired_rows"] == 5 and pair["classification"] == "INSUFFICIENT_DATA"
    assert matrix["dilution"].isna().sum() == 5


def test_aliases_are_supported_but_duplicate_tickers_fail_closed() -> None:
    alias = pd.DataFrame([{"ticker": "BBCA", "foreign_flow_persistence": 1, "reported_ownership_changes": 0}])
    matrix, _ = analyze_double_counting(alias, minimum_paired_rows=3)
    assert matrix.iloc[0]["foreign_accumulation"] == 1
    with pytest.raises(ValueError, match="CORRELATION_TICKER_DUPLICATE"):
        analyze_double_counting(pd.concat([alias, alias]), minimum_paired_rows=3)


def test_analysis_is_measurement_only_and_does_not_change_scoring() -> None:
    _, summary = analyze_double_counting(pd.DataFrame(_rows()), minimum_paired_rows=20)
    assert summary["policy_effect"] == "ANALYSIS_ONLY_NO_PHASE5_6_SCORING_WEIGHT_CHANGE"
    source = Path(__file__).resolve().parents[1].joinpath("evidence_double_counting_analysis.py").read_text().lower()
    for forbidden in ("entry_price", "take_profit", "stop_loss", "recommendation"):
        assert forbidden not in source


@pytest.mark.parametrize("minimum", [0, 1, 2, True])
def test_invalid_minimum_paired_rows_fails_closed(minimum) -> None:
    with pytest.raises(ValueError, match="INVALID_MINIMUM_PAIRED_ROWS"):
        analyze_double_counting(pd.DataFrame(_rows()), minimum_paired_rows=minimum)
