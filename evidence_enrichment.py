from __future__ import annotations

"""Derived evidence that remains explicitly separate from direct disclosure.

The scanner has rich period-aligned financial history, but the resumable path
previously exposed only a few raw fields to the Next Leader model.  These
helpers turn observed financial outcomes into forward-capacity and management-
execution *proxies*.  They never manufacture project, guidance, biography, or
governance evidence and never overwrite a direct score supplied by a source.
"""

from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd


EVIDENCE_ENRICHMENT_VERSION = "1.0.0-financial-outcome-proxies"


def _num(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _fraction(value: Any) -> float:
    number = _num(value)
    if not np.isfinite(number):
        return np.nan
    return number / 100.0 if abs(number) > 2.0 else number


def _clip(value: Any) -> float:
    number = _num(value)
    return float(np.clip(number, 0.0, 100.0)) if np.isfinite(number) else np.nan


def _linear(value: Any, weak: float, strong: float) -> float:
    number = _num(value)
    if not np.isfinite(number) or strong == weak:
        return np.nan
    return _clip(100.0 * (number - weak) / (strong - weak))


def _inverse(value: Any, good: float, bad: float) -> float:
    score = _linear(value, good, bad)
    return 100.0 - score if np.isfinite(score) else np.nan


def _positive(value: Any) -> float:
    number = _num(value)
    if not np.isfinite(number):
        return np.nan
    return 82.0 if number > 0 else 18.0


def _mean_available(*values: Any) -> float:
    finite = [_num(value) for value in values]
    finite = [value for value in finite if np.isfinite(value)]
    return float(np.mean(finite)) if finite else np.nan


def _first(row: Mapping[str, Any], names: Sequence[str], *, fraction: bool = False) -> float:
    for name in names:
        value = _fraction(row.get(name)) if fraction else _num(row.get(name))
        if np.isfinite(value):
            return value
    return np.nan


def _weighted(
    items: Sequence[tuple[str, float, float | Callable[[], float]]],
) -> tuple[float, float, str]:
    observed: list[tuple[str, float, float]] = []
    possible = sum(weight for _, weight, _ in items)
    for name, weight, raw in items:
        value = raw() if callable(raw) else raw
        score = _clip(value)
        if np.isfinite(score):
            observed.append((name, weight, score))
    observed_weight = sum(weight for _, weight, _ in observed)
    if observed_weight <= 0 or possible <= 0:
        return np.nan, 0.0, ""
    score = sum(weight * value for _, weight, value in observed) / observed_weight
    coverage = 100.0 * observed_weight / possible
    return round(score, 1), round(coverage, 1), " | ".join(name for name, _, _ in observed)


def _row_proxies(row: Mapping[str, Any]) -> dict[str, Any]:
    revenue_growth = _first(row, ("history_revenue_growth", "revenue_growth"), fraction=True)
    earnings_growth = _first(row, ("history_earnings_growth", "earnings_growth"), fraction=True)
    revenue_acceleration = _first(row, ("history_revenue_growth_acceleration",), fraction=True)
    earnings_acceleration = _first(row, ("history_earnings_growth_acceleration",), fraction=True)
    roe = _first(row, ("history_roe", "roe"), fraction=True)
    roic = _first(row, ("history_roic_proxy", "roic_proxy"), fraction=True)
    operating_margin = _first(row, ("history_operating_margin", "operating_margin"), fraction=True)
    cash_conversion = _first(row, ("history_cash_conversion", "cash_conversion_ttm"))
    fcf = _first(row, ("history_fcf_ttm", "free_cash_flow"))
    dilution = _first(row, ("history_share_dilution_yoy", "share_dilution_yoy"), fraction=True)
    debt_equity = _first(row, ("history_debt_equity", "debt_equity"))
    cash_to_debt = _first(row, ("cash_to_debt",))
    positive_earnings = _first(row, ("history_positive_earnings_ratio",))
    positive_ocf = _first(row, ("history_positive_ocf_ratio",))
    margin_stability = _first(row, ("history_margin_stability",))
    accruals = _first(row, ("history_accruals_to_assets",), fraction=True)
    leverage_change = _first(row, ("history_leverage_change_yoy",), fraction=True)

    growth_persistence, growth_coverage, growth_basis = _weighted((
        ("REVENUE_GROWTH", 0.22, _linear(revenue_growth, -0.05, 0.20)),
        ("EARNINGS_GROWTH", 0.22, _linear(earnings_growth, -0.10, 0.28)),
        ("REVENUE_ACCELERATION", 0.14, _linear(revenue_acceleration, -0.10, 0.10)),
        ("EARNINGS_ACCELERATION", 0.14, _linear(earnings_acceleration, -0.20, 0.20)),
        ("POSITIVE_EARNINGS_PERIODS", 0.14, _linear(positive_earnings, 0.50, 1.00)),
        ("MARGIN_STABILITY", 0.14, _linear(margin_stability, 0.20, 0.85)),
    ))
    reinvestment, reinvestment_coverage, reinvestment_basis = _weighted((
        ("ROIC", 0.25, _linear(roic, 0.03, 0.18)),
        ("ROE", 0.15, _linear(roe, 0.05, 0.20)),
        ("REVENUE_GROWTH", 0.15, _linear(revenue_growth, -0.03, 0.18)),
        ("EARNINGS_GROWTH", 0.15, _linear(earnings_growth, -0.05, 0.22)),
        ("CASH_CONVERSION", 0.12, _linear(cash_conversion, 0.45, 1.20)),
        ("FREE_CASH_FLOW", 0.08, _positive(fcf)),
        ("LEVERAGE", 0.10, _inverse(debt_equity, 0.25, 2.00)),
    ))
    forward_capacity, forward_coverage, forward_basis = _weighted((
        ("GROWTH_PERSISTENCE", 0.22, growth_persistence),
        ("REINVESTMENT", 0.20, reinvestment),
        ("ROIC", 0.13, _linear(roic, 0.03, 0.18)),
        ("CASH_CONVERSION", 0.12, _linear(cash_conversion, 0.45, 1.20)),
        ("FREE_CASH_FLOW", 0.10, _positive(fcf)),
        ("BALANCE_SHEET", 0.10, _inverse(debt_equity, 0.25, 2.00)),
        ("POSITIVE_OCF_PERIODS", 0.08, _linear(positive_ocf, 0.50, 1.00)),
        ("CASH_TO_DEBT", 0.05, _linear(cash_to_debt, 0.10, 1.20)),
    ))
    management_execution, management_coverage, management_basis = _weighted((
        ("ROIC_OUTCOME", 0.22, _linear(roic, 0.03, 0.18)),
        ("ROE_OUTCOME", 0.12, _linear(roe, 0.05, 0.20)),
        ("CASH_CONVERSION", 0.18, _linear(cash_conversion, 0.45, 1.20)),
        ("DILUTION_DISCIPLINE", 0.16, _inverse(dilution, 0.00, 0.15)),
        ("OPERATING_MARGIN", 0.10, _linear(operating_margin, 0.02, 0.18)),
        ("BALANCE_SHEET", 0.10, _inverse(debt_equity, 0.25, 2.00)),
        ("MARGIN_STABILITY", 0.06, _linear(margin_stability, 0.20, 0.85)),
        ("ACCRUAL_DISCIPLINE", 0.06, _inverse(accruals, -0.02, 0.10)),
    ))
    capital_allocation, capital_coverage, capital_basis = _weighted((
        ("ROIC", 0.24, _linear(roic, 0.03, 0.18)),
        ("FREE_CASH_FLOW", 0.14, _positive(fcf)),
        ("DILUTION_DISCIPLINE", 0.18, _inverse(dilution, 0.00, 0.15)),
        ("LEVERAGE", 0.14, _inverse(debt_equity, 0.25, 2.00)),
        ("CASH_CONVERSION", 0.12, _linear(cash_conversion, 0.45, 1.20)),
        ("PROFITABLE_GROWTH", 0.12, _mean_available(
            _linear(revenue_growth, -0.03, 0.18),
            _linear(earnings_growth, -0.05, 0.22),
        )),
        ("LEVERAGE_TREND", 0.06, _inverse(leverage_change, -0.04, 0.08)),
    ))
    quality_coverage = round(float(np.nanmean([
        value for value in (growth_coverage, reinvestment_coverage, management_coverage, capital_coverage)
        if np.isfinite(value)
    ])), 1) if any(np.isfinite(value) for value in (
        growth_coverage, reinvestment_coverage, management_coverage, capital_coverage
    )) else 0.0
    return {
        "forward_growth_persistence_score": growth_persistence,
        "forward_growth_persistence_coverage_pct": growth_coverage,
        "forward_growth_persistence_basis": growth_basis,
        "reinvestment_runway_pillar": reinvestment,
        "reinvestment_runway_coverage_pct": reinvestment_coverage,
        "reinvestment_runway_basis": reinvestment_basis,
        "forward_financial_capacity_score": forward_capacity,
        "forward_financial_capacity_coverage_pct": forward_coverage,
        "forward_financial_capacity_basis": forward_basis,
        "management_execution_proxy_score": management_execution,
        "management_execution_proxy_coverage_pct": management_coverage,
        "management_execution_proxy_basis": management_basis,
        "capital_allocation_proxy_score": capital_allocation,
        "capital_allocation_proxy_coverage_pct": capital_coverage,
        "capital_allocation_proxy_basis": capital_basis,
        "quality_pillar_coverage_pct": quality_coverage,
        "derived_evidence_provenance_state": "PERIOD_ALIGNED_FINANCIAL_OUTCOME_PROXY_NOT_DIRECT_GUIDANCE_OR_BIOGRAPHY",
        "derived_evidence_version": EVIDENCE_ENRICHMENT_VERSION,
    }


def enrich_fundamental_evidence(fundamentals: pd.DataFrame | None) -> pd.DataFrame:
    """Attach auditable financial-outcome proxies to each fundamental row."""
    if fundamentals is None or fundamentals.empty:
        return fundamentals.copy() if isinstance(fundamentals, pd.DataFrame) else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    direct_fields = {
        "reinvestment_runway_pillar",
        "management_quality_score",
        "capital_allocation_score",
        "future_fundamental_impact_score",
        "project_pipeline_score",
    }
    for raw in fundamentals.to_dict("records"):
        row = dict(raw)
        derived = _row_proxies(row)
        for key, value in derived.items():
            if key in direct_fields and np.isfinite(_num(row.get(key))):
                continue
            row[key] = value
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["EVIDENCE_ENRICHMENT_VERSION", "enrich_fundamental_evidence"]
