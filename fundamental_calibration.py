from __future__ import annotations

"""Pure calibration helpers for reporting cadence and thesis classification.

This module has no provider/database/UI imports.  It exists so statement
freshness, latest-growth provenance and thesis labels are consistent across
ranking, bounded enrichment and real-money authorization.
"""

from typing import Any, Mapping

import numpy as np
import pandas as pd

CALIBRATION_VERSION = "1.0.0-v9.8.2-hotfix3"
REPORT_REFRESH_GRACE_DAYS = 20
REPORT_HARD_STALE_DAYS = 365


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _text(value: Any) -> str:
    return str(value or "").strip()


def _first(row: Mapping[str, Any], *names: str, default: Any = np.nan) -> Any:
    for name in names:
        value = row.get(name, np.nan)
        if isinstance(value, str):
            if value.strip():
                return value
        elif value is not None and not (isinstance(value, float) and np.isnan(value)):
            return value
    return default


def _first_num(row: Mapping[str, Any], *names: str) -> float:
    for name in names:
        value = _finite(row.get(name), np.nan)
        if np.isfinite(value):
            return value
    return np.nan


def _timestamp(value: Any) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return pd.NaT
    try:
        stamp = pd.Timestamp(value)
    except Exception:
        return pd.NaT
    if pd.isna(stamp):
        return pd.NaT
    if stamp.tzinfo is not None:
        try:
            stamp = stamp.tz_convert("Asia/Jakarta").tz_localize(None)
        except Exception:
            stamp = stamp.tz_localize(None)
    return stamp.normalize()


def _now(value: Any | None = None) -> pd.Timestamp:
    stamp = pd.Timestamp(value) if value is not None else pd.Timestamp.now(tz="Asia/Jakarta")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("Asia/Jakarta").tz_localize(None)
    return stamp.normalize()


def reporting_refresh_profile(
    row: Mapping[str, Any],
    *,
    now: Any | None = None,
    grace_days: int = REPORT_REFRESH_GRACE_DAYS,
) -> dict[str, Any]:
    """Calendar-aware statement freshness.

    A statement can be younger than an arbitrary age threshold yet already sit
    behind the next quarterly reporting window.  That state should trigger
    bounded refresh, not disappear from research ranking and not authorize real
    money as if the period were fully current.
    """
    latest = _timestamp(_first(
        row,
        "fund_fundamental_history_latest_period", "fundamental_history_latest_period",
        "fund_latest_statement_date", "latest_statement_date",
        "fund_fundamental_latest_period", "fundamental_latest_period",
        default=pd.NaT,
    ))
    current = _now(now)
    if pd.isna(latest):
        age_hint = _first_num(row, "fund_statement_age_days", "statement_age_days", "fund_fundamental_history_age_days", "fundamental_history_age_days")
        if np.isfinite(age_hint) and age_hint >= 0:
            latest = (current - pd.Timedelta(days=int(age_hint))).normalize()
        else:
            return {
                "fundamental_refresh_state": "MISSING_DATE",
                "fundamental_refresh_due": True,
                "fundamental_latest_period": pd.NaT,
                "fundamental_statement_age_days": np.nan,
                "fundamental_next_period_end": pd.NaT,
                "fundamental_refresh_open_at": pd.NaT,
                "fundamental_calibration_version": CALIBRATION_VERSION,
            }
    age = int((current - latest).days)
    next_period_end = latest + pd.offsets.QuarterEnd(1)
    refresh_open_at = (next_period_end + pd.Timedelta(days=max(0, int(grace_days)))).normalize()
    if age > REPORT_HARD_STALE_DAYS:
        state = "STALE"
        due = True
    elif current >= refresh_open_at:
        state = "REFRESH_WINDOW"
        due = True
    else:
        state = "CURRENT"
        due = False
    return {
        "fundamental_refresh_state": state,
        "fundamental_refresh_due": bool(due),
        "fundamental_latest_period": latest,
        "fundamental_statement_age_days": age,
        "fundamental_next_period_end": next_period_end,
        "fundamental_refresh_open_at": refresh_open_at,
        "fundamental_calibration_version": CALIBRATION_VERSION,
    }


def latest_growth_profile(row: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve latest-history growth before point-in-time proxy growth."""
    history_revenue = _first_num(row, "fund_history_revenue_growth", "history_revenue_growth")
    history_earnings = _first_num(row, "fund_history_earnings_growth", "history_earnings_growth")
    snapshot_revenue = _first_num(row, "fund_revenue_growth_snapshot", "revenue_growth_snapshot", "fund_revenue_growth", "revenue_growth")
    snapshot_earnings = _first_num(row, "fund_earnings_growth_snapshot", "earnings_growth_snapshot", "fund_earnings_growth", "earnings_growth")
    prior_revenue = _first_num(row, "fund_history_prior_revenue_growth", "history_prior_revenue_growth")
    prior_earnings = _first_num(row, "fund_history_prior_earnings_growth", "history_prior_earnings_growth")

    revenue = history_revenue if np.isfinite(history_revenue) else snapshot_revenue
    earnings = history_earnings if np.isfinite(history_earnings) else snapshot_earnings
    basis = "LATEST_HISTORY_PERIOD" if np.isfinite(history_revenue) or np.isfinite(history_earnings) else "SNAPSHOT_PROXY"

    conflicts: list[str] = []
    if np.isfinite(history_revenue) and np.isfinite(snapshot_revenue):
        if np.sign(history_revenue) != np.sign(snapshot_revenue) and abs(history_revenue - snapshot_revenue) >= 0.15:
            conflicts.append("REVENUE_GROWTH_SIGN_CONFLICT")
    if np.isfinite(history_earnings) and np.isfinite(snapshot_earnings):
        if np.sign(history_earnings) != np.sign(snapshot_earnings) and abs(history_earnings - snapshot_earnings) >= 0.20:
            conflicts.append("EARNINGS_GROWTH_SIGN_CONFLICT")

    if np.isfinite(revenue) and np.isfinite(earnings) and revenue <= -0.05 and earnings <= -0.05:
        trend = "FUNDAMENTAL_DETERIORATION"
    elif (
        np.isfinite(revenue) and revenue >= 0.10
        and np.isfinite(earnings) and earnings >= 0.15
        and ((np.isfinite(prior_revenue) and prior_revenue <= 0.0) or (np.isfinite(prior_earnings) and prior_earnings <= 0.0))
    ):
        trend = "TURNAROUND_RECOVERY"
    elif np.isfinite(revenue) and np.isfinite(earnings) and revenue >= 0.10 and earnings >= 0.10:
        trend = "GROWTH_ACCELERATION"
    elif np.isfinite(revenue) and np.isfinite(earnings) and (revenue < 0 or earnings < 0):
        trend = "MIXED_OR_DECELERATING"
    else:
        trend = "NEUTRAL_OR_PARTIAL"

    extreme_earnings = bool(np.isfinite(earnings) and abs(earnings) >= 3.0)
    return {
        "fundamental_latest_revenue_growth": revenue,
        "fundamental_latest_earnings_growth": earnings,
        "fundamental_prior_revenue_growth": prior_revenue,
        "fundamental_prior_earnings_growth": prior_earnings,
        "fundamental_growth_basis_state": basis,
        "fundamental_growth_conflict_state": " | ".join(conflicts) if conflicts else "ALIGNED_OR_SINGLE_SOURCE",
        "fundamental_trend_state": trend,
        "fundamental_extreme_earnings_base_review": extreme_earnings,
    }


def classify_thesis_archetype(
    row: Mapping[str, Any],
    *,
    business_score: float = np.nan,
    future_score: float = np.nan,
) -> str:
    growth = latest_growth_profile(row)
    trend = growth["fundamental_trend_state"]
    revenue = _finite(growth["fundamental_latest_revenue_growth"], np.nan)
    earnings = _finite(growth["fundamental_latest_earnings_growth"], np.nan)
    roe = _first_num(row, "fund_roe", "fund_history_roe", "roe", "history_roe")
    debt_equity = _first_num(row, "fund_debt_equity", "fund_history_debt_equity", "debt_equity", "history_debt_equity")
    cash_conversion = _first_num(row, "fund_history_cash_conversion", "fund_cash_conversion_ttm", "history_cash_conversion", "cash_conversion_ttm")
    sector = _text(_first(row, "fund_sector", "mac_sector", "sector", default="UNKNOWN")).upper()

    if trend == "FUNDAMENTAL_DETERIORATION":
        return "FUNDAMENTAL_DETERIORATION"
    if bool(growth["fundamental_extreme_earnings_base_review"]):
        return "BASE_EFFECT_REVIEW"
    if trend == "TURNAROUND_RECOVERY":
        return "TURNAROUND_RECOVERY"
    if sector in {"ENERGY", "BASIC MATERIALS", "BASIC MATERIAL", "MATERIALS"} and np.isfinite(future_score) and future_score >= 50:
        return "CYCLICAL_RECOVERY"
    quality_ok = np.isfinite(business_score) and business_score >= 65.0
    leverage_ok = not np.isfinite(debt_equity) or debt_equity <= 1.20
    cash_ok = not np.isfinite(cash_conversion) or cash_conversion >= 0.60
    if (
        np.isfinite(revenue) and revenue >= 0.12
        and np.isfinite(earnings) and earnings >= 0.10
        and np.isfinite(roe) and roe >= 0.12
        and quality_ok and leverage_ok and cash_ok
    ):
        return "GROWTH_COMPOUNDER"
    if np.isfinite(future_score) and future_score >= 65:
        return "EVENT_DRIVEN_RERATING"
    if np.isfinite(revenue) and revenue >= 0.12:
        return "EMERGING_GROWTH"
    return "GROWTH_VALUE_CANDIDATE"


def maintenance_refresh_priority(row: Mapping[str, Any], *, history_count: int = 0, now: Any | None = None) -> tuple[int, float]:
    """Small deterministic priority used only inside bounded enrichment."""
    freshness = reporting_refresh_profile(row, now=now)
    age = _finite(freshness.get("fundamental_statement_age_days"), np.nan)
    sector = _text(_first(row, "sector", "fund_sector", default="UNKNOWN")).upper()
    eligible_raw = _first(row, "fundamental_score_eligible", default=False)
    eligible = str(eligible_raw).strip().upper() in {"1", "TRUE", "YES", "PASS", "VALID"} if not isinstance(eligible_raw, bool) else eligible_raw
    if not row:
        bucket = 0
    elif freshness.get("fundamental_refresh_state") in {"MISSING_DATE", "STALE", "REFRESH_WINDOW"}:
        bucket = 0
    elif sector == "UNKNOWN":
        bucket = 1
    elif not eligible:
        bucket = 2
    elif int(history_count) < 2:
        bucket = 3
    else:
        bucket = 9
    return bucket, -(age if np.isfinite(age) else 9999.0)


__all__ = [
    "CALIBRATION_VERSION", "REPORT_REFRESH_GRACE_DAYS", "REPORT_HARD_STALE_DAYS",
    "reporting_refresh_profile", "latest_growth_profile", "classify_thesis_archetype",
    "maintenance_refresh_priority",
]
