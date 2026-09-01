from __future__ import annotations

"""Measurement-only correlation and double-counting analysis for Phase 5.6."""

from collections import OrderedDict
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


ANALYSIS_VERSION = "1.0.0-phase5.6-task37"
SIGNALS: Mapping[str, tuple[str, ...]] = OrderedDict({
    "foreign_accumulation": ("foreign_accumulation", "foreign_flow_persistence"),
    "ownership_accumulation": ("ownership_accumulation", "reported_ownership_changes"),
    "participant_accumulation": ("participant_accumulation", "participant_broker_flow"),
    "volume_turnover": ("volume_turnover", "volume_value_turnover"),
    "existing_silent_accumulation": ("existing_silent_accumulation", "silent_accumulation"),
    "inventory_proxies": ("inventory_proxies", "inventory_proxy"),
    "trend_momentum": ("trend_momentum", "momentum"),
    "smc_ict_structure": ("smc_ict_structure", "smc_ict_market_structure"),
    "fundamental_growth": ("fundamental_growth", "fundamental_growth_quality"),
    "corporate_catalysts": ("corporate_catalysts", "official_corporate_catalysts"),
    "dilution": ("dilution", "dilution_capital_action"),
})

# These pairs have semantic overlap even before empirical correlation is measured.
SEMANTIC_OVERLAP_PAIRS = {
    frozenset(pair)
    for pair in (
        ("foreign_accumulation", "existing_silent_accumulation"),
        ("foreign_accumulation", "inventory_proxies"),
        ("ownership_accumulation", "existing_silent_accumulation"),
        ("ownership_accumulation", "inventory_proxies"),
        ("participant_accumulation", "existing_silent_accumulation"),
        ("participant_accumulation", "inventory_proxies"),
        ("participant_accumulation", "volume_turnover"),
        ("volume_turnover", "existing_silent_accumulation"),
        ("volume_turnover", "inventory_proxies"),
        ("trend_momentum", "smc_ict_structure"),
        ("corporate_catalysts", "dilution"),
    )
}


def _ticker(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text[:-3] if text.endswith(".JK") else text


def _matrix(detail: pd.DataFrame, universe: Iterable[Any] | None) -> pd.DataFrame:
    source = detail.copy() if isinstance(detail, pd.DataFrame) else pd.DataFrame()
    if not source.empty and "ticker" not in source:
        raise ValueError("CORRELATION_TICKER_COLUMN_MISSING")
    if "ticker" in source:
        source["ticker"] = source["ticker"].map(_ticker)
        source = source[source["ticker"] != ""]
        if source["ticker"].duplicated().any():
            raise ValueError("CORRELATION_TICKER_DUPLICATE")
        source = source.set_index("ticker", drop=False)
    requested = list(dict.fromkeys(_ticker(value) for value in (universe or ()) if _ticker(value)))
    tickers = requested or (source["ticker"].tolist() if not source.empty else [])
    matrix = pd.DataFrame({"ticker": tickers})
    for signal, aliases in SIGNALS.items():
        column = next((alias for alias in aliases if alias in source), "")
        values = []
        for ticker in tickers:
            raw = source.at[ticker, column] if column and ticker in source.index else np.nan
            value = pd.to_numeric(raw, errors="coerce")
            values.append(float(value) if pd.notna(value) and np.isfinite(float(value)) else np.nan)
        matrix[signal] = values
    return matrix


def _classification(left: str, right: str, correlation: float | None) -> str:
    if correlation is None:
        return "INSUFFICIENT_DATA"
    magnitude = abs(correlation)
    if magnitude >= 0.90 and frozenset((left, right)) in SEMANTIC_OVERLAP_PAIRS:
        return "POSSIBLE_DOUBLE_COUNTING"
    if magnitude >= 0.70:
        return "HIGHLY_CORRELATED_EVIDENCE"
    if magnitude >= 0.30:
        return "PARTIALLY_REDUNDANT_EVIDENCE"
    return "INDEPENDENT_EVIDENCE"


def analyze_double_counting(
    detail: pd.DataFrame,
    *,
    universe: Iterable[Any] | None = None,
    minimum_paired_rows: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if isinstance(minimum_paired_rows, bool) or int(minimum_paired_rows) < 3:
        raise ValueError("INVALID_MINIMUM_PAIRED_ROWS")
    matrix = _matrix(detail, universe)
    pairs: list[dict[str, Any]] = []
    names = list(SIGNALS)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            paired = matrix[[left, right]].dropna()
            correlation: float | None = None
            if (
                len(paired) >= int(minimum_paired_rows)
                and paired[left].nunique() > 1
                and paired[right].nunique() > 1
            ):
                observed = float(paired[left].corr(paired[right], method="spearman"))
                if np.isfinite(observed):
                    correlation = round(observed, 6)
            classification = _classification(left, right, correlation)
            pairs.append({
                "left": left,
                "right": right,
                "paired_rows": len(paired),
                "spearman_correlation": correlation,
                "semantic_overlap": frozenset((left, right)) in SEMANTIC_OVERLAP_PAIRS,
                "classification": classification,
            })

    distribution: dict[str, int] = {}
    for item in pairs:
        label = item["classification"]
        distribution[label] = distribution.get(label, 0) + 1
    known = {
        signal: {
            "known": int(matrix[signal].notna().sum()),
            "unknown": int(matrix[signal].isna().sum()),
        }
        for signal in SIGNALS
    }
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "total_universe": len(matrix),
        "signals": list(SIGNALS),
        "signal_observability": known,
        "pairwise_analysis": pairs,
        "classification_distribution": dict(sorted(distribution.items())),
        "possible_double_counting": [
            item for item in pairs if item["classification"] == "POSSIBLE_DOUBLE_COUNTING"
        ],
        "thresholds": {
            "partially_redundant_absolute_correlation": 0.30,
            "highly_correlated_absolute_correlation": 0.70,
            "possible_double_counting_absolute_correlation": 0.90,
            "minimum_paired_rows": int(minimum_paired_rows),
        },
        "policy_effect": "ANALYSIS_ONLY_NO_PHASE5_6_SCORING_WEIGHT_CHANGE",
    }
    return matrix, summary


__all__ = [
    "ANALYSIS_VERSION", "SEMANTIC_OVERLAP_PAIRS", "SIGNALS", "analyze_double_counting",
]
