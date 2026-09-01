from __future__ import annotations

"""Scoring-independent measurement of evidence coverage and overlap."""

from collections import OrderedDict
import math
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


AUDIT_VERSION = "1.0.0-phase5.6-task26"
PARTICIPANT_COMPONENT = "participant_broker_flow"
COMPONENT_ALIASES: Mapping[str, tuple[str, ...]] = OrderedDict({
    "reported_ownership_changes": (
        "reported_ownership_changes", "ownership_change_available", "ownership_changes_available",
    ),
    "foreign_flow_persistence": (
        "foreign_flow_persistence", "foreign_20_session_sufficient", "foreign_sufficient",
    ),
    "volume_value_turnover": (
        "volume_value_turnover", "market_activity_available", "ohlcv_valid",
    ),
    "smc_ict_market_structure": (
        "smc_ict_market_structure", "market_structure_available", "smc_ict_available",
    ),
    "dilution_capital_action": (
        "dilution_capital_action", "capital_action_available", "dilution_context_available",
    ),
    "official_corporate_catalysts": (
        "official_corporate_catalysts", "official_catalyst_available", "forward_evidence_available",
    ),
    PARTICIPANT_COMPONENT: (
        PARTICIPANT_COMPONENT, "participant_flow_available", "broker_sufficient", "broker_direct",
    ),
    "liquidity_risk_context": (
        "liquidity_risk_context", "liquidity_risk_available", "risk_context_available",
    ),
    "fundamental_growth_quality": (
        "fundamental_growth_quality", "fundamental_valid", "fundamental_official",
    ),
})


def _ticker(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text[:-3] if text.endswith(".JK") else text


def _boolean(value: Any) -> bool | pd._libs.missing.NAType:
    if value is None or value is pd.NA:
        return pd.NA
    try:
        if pd.isna(value):
            return pd.NA
    except (TypeError, ValueError):
        return pd.NA
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().upper()
    if text in {"1", "TRUE", "YES", "VALID", "AVAILABLE", "PRESENT", "PASS", "VERIFIED"}:
        return True
    if text in {"0", "FALSE", "NO", "INVALID", "UNAVAILABLE", "ABSENT", "FAIL"}:
        return False
    return pd.NA


def build_component_matrix(
    detail: pd.DataFrame,
    *,
    universe: Iterable[Any] | None = None,
) -> pd.DataFrame:
    source = detail.copy() if isinstance(detail, pd.DataFrame) else pd.DataFrame()
    ticker_col = next((name for name in ("ticker", "symbol", "code", "stock_code") if name in source), "")
    if not ticker_col and not source.empty:
        raise ValueError("EVIDENCE_TICKER_COLUMN_MISSING")
    if ticker_col:
        source["ticker"] = source[ticker_col].map(_ticker)
        source = source[source["ticker"] != ""]
        if source["ticker"].duplicated().any():
            raise ValueError("EVIDENCE_TICKER_DUPLICATE")
        source = source.set_index("ticker", drop=False)
    requested = list(dict.fromkeys(_ticker(value) for value in (universe or ()) if _ticker(value)))
    tickers = requested or (source["ticker"].tolist() if not source.empty else [])
    matrix = pd.DataFrame({"ticker": tickers})
    for component, aliases in COMPONENT_ALIASES.items():
        column = next((alias for alias in aliases if alias in source), "")
        if not column:
            matrix[component] = pd.array([pd.NA] * len(matrix), dtype="boolean")
            continue
        values = []
        for ticker in tickers:
            value = source.at[ticker, column] if ticker in source.index else pd.NA
            values.append(_boolean(value))
        matrix[component] = pd.array(values, dtype="boolean")
    component_columns = list(COMPONENT_ALIASES)
    matrix["known_component_count"] = matrix[component_columns].notna().sum(axis=1).astype(int)
    matrix["available_component_count"] = matrix[component_columns].fillna(False).sum(axis=1).astype(int)
    return matrix


def _correlation(left: pd.Series, right: pd.Series) -> float | None:
    if len(left) < 2 or left.nunique() < 2 or right.nunique() < 2:
        return None
    value = float(np.corrcoef(left.astype(int), right.astype(int))[0, 1])
    return round(value, 6) if math.isfinite(value) else None


def audit_evidence_independence(
    detail: pd.DataFrame,
    *,
    universe: Iterable[Any] | None = None,
    duplicate_correlation_threshold: float = 0.90,
    minimum_paired_rows: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not 0 <= float(duplicate_correlation_threshold) <= 1:
        raise ValueError("INVALID_CORRELATION_THRESHOLD")
    if int(minimum_paired_rows) < 2:
        raise ValueError("INVALID_MINIMUM_PAIRED_ROWS")
    matrix = build_component_matrix(detail, universe=universe)
    total = len(matrix)
    coverage: dict[str, dict[str, int | float]] = {}
    for component in COMPONENT_ALIASES:
        series = matrix[component]
        known = int(series.notna().sum())
        available = int(series.fillna(False).sum())
        unavailable = known - available
        unknown = total - known
        coverage[component] = {
            "known": known,
            "available": available,
            "unavailable": unavailable,
            "unknown": unknown,
            "available_percentage_of_known": round(100.0 * available / known, 2) if known else 0.0,
            "available_percentage_of_universe": round(100.0 * available / total, 2) if total else 0.0,
        }

    pairwise: list[dict[str, Any]] = []
    components = list(COMPONENT_ALIASES)
    for index, left_name in enumerate(components):
        for right_name in components[index + 1:]:
            paired = matrix[[left_name, right_name]].dropna()
            left = paired[left_name].astype(bool)
            right = paired[right_name].astype(bool)
            correlation = _correlation(left, right)
            both = int((left & right).sum())
            entry = {
                "left": left_name,
                "right": right_name,
                "paired_rows": len(paired),
                "both_available": both,
                "correlation": correlation,
                "possible_duplicate": bool(
                    correlation is not None
                    and len(paired) >= int(minimum_paired_rows)
                    and abs(correlation) >= float(duplicate_correlation_threshold)
                ),
            }
            pairwise.append(entry)

    other_components = [name for name in components if name != PARTICIPANT_COMPONENT]
    participant = matrix[PARTICIPANT_COMPONENT]
    other_known = matrix[other_components].notna().sum(axis=1)
    other_available = matrix[other_components].fillna(False).sum(axis=1)
    participant_known = participant.notna()
    with_participant = participant_known & participant.fillna(False)
    without_participant = participant_known & ~participant.fillna(False)

    def _mean(values: pd.Series) -> float | None:
        return round(float(values.mean()), 4) if len(values) else None

    participant_context = {
        "participant_known": int(participant_known.sum()),
        "participant_available": int(with_participant.sum()),
        "participant_unavailable": int(without_participant.sum()),
        "participant_unknown": int((~participant_known).sum()),
        "mean_other_available_when_participant_available": _mean(other_available[with_participant]),
        "mean_other_available_when_participant_unavailable": _mean(other_available[without_participant]),
        "participant_only_rows": int((with_participant & (other_available == 0) & (other_known > 0)).sum()),
        "all_other_known_and_available_with_participant": int(
            (with_participant & (other_known == len(other_components)) & (other_available == len(other_components))).sum()
        ),
        "all_other_known_and_available_without_participant": int(
            (without_participant & (other_known == len(other_components)) & (other_available == len(other_components))).sum()
        ),
    }
    summary = {
        "audit_version": AUDIT_VERSION,
        "total_universe": total,
        "component_coverage": coverage,
        "pairwise_overlap": pairwise,
        "possible_duplicate_pairs": [
            {key: item[key] for key in ("left", "right", "paired_rows", "correlation")}
            for item in pairwise if item["possible_duplicate"]
        ],
        "participant_context": participant_context,
        "policy_effect": "MEASUREMENT_ONLY_NO_PRODUCTION_CHANGE",
    }
    return matrix, summary


__all__ = [
    "AUDIT_VERSION", "COMPONENT_ALIASES", "PARTICIPANT_COMPONENT",
    "audit_evidence_independence", "build_component_matrix",
]
