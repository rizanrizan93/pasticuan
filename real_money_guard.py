from __future__ import annotations

"""Independent real-money authorization and fundamental conviction caps.

This module is deliberately pure: no provider, database, Streamlit, or scanner
imports. Ranking is allowed to describe research conviction; authorization is a
separate fail-closed decision about whether a plan may be sent to the order
builder with real capital.
"""

from typing import Any, Mapping

import numpy as np
import pandas as pd

from fundamental_calibration import reporting_refresh_profile, latest_growth_profile

REAL_MONEY_GUARD_VERSION = "1.1.0-v9.8.2-hotfix3"
MAX_REAL_MONEY_RISK_BUDGET_PCT = 0.75


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().upper() in {"1", "TRUE", "YES", "Y", "PASS", "VALID", "VERIFIED"}


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


def fundamental_conviction_profile(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return an auditable score ceiling from statement quality/cash-flow/leverage.

    The cap is intentionally not a new additive score. It only prevents weak
    evidence from receiving research conviction that the statement lineage does
    not support.
    """
    grade = _text(_first(row, "fund_fundamental_data_grade", "fundamental_data_grade", default="D")).upper() or "D"
    history_cov = _first_num(row, "fund_fundamental_history_coverage", "fundamental_history_coverage")
    cashflow_cov = _first_num(row, "fund_fundamental_cashflow_statement_coverage_pct", "fundamental_cashflow_statement_coverage_pct")
    official_cov = _first_num(row, "fund_fundamental_official_source_coverage_pct", "fundamental_official_source_coverage_pct")
    consensus = _first_num(row, "fund_fundamental_consensus_score", "fundamental_consensus_score")
    ocf = _first_num(row, "fund_history_ocf_ttm", "history_ocf_ttm", "fund_operating_cash_flow", "operating_cash_flow")
    fcf = _first_num(row, "fund_history_fcf_ttm", "history_fcf_ttm", "fund_free_cash_flow", "free_cash_flow")
    debt_equity = _first_num(row, "fund_history_debt_equity", "history_debt_equity", "fund_debt_equity", "debt_equity")
    official_verified = _truthy(_first(row, "fund_fundamental_official_verified", "fundamental_official_verified", default=False))
    conflicts = _text(_first(row, "fund_fundamental_conflicts", "fundamental_conflicts", default=""))
    sector = _text(_first(row, "fund_sector", "sector", default="UNKNOWN")).upper()
    is_financial = sector in {"FINANCIALS", "FINANCIAL", "BANKS", "BANKING"}
    freshness = reporting_refresh_profile(row)
    growth = latest_growth_profile(row)

    cap = {"A": 100.0, "B": 90.0, "C": 78.0, "D": 68.0}.get(grade, 72.0)
    reasons: list[str] = [f"GRADE_{grade}"]

    if np.isfinite(history_cov):
        if history_cov < 40:
            cap = min(cap, 65.0); reasons.append("HISTORY_COVERAGE<40")
        elif history_cov < 55:
            cap = min(cap, 75.0); reasons.append("HISTORY_COVERAGE<55")

    # Bank/financial cash-flow statements are not comparable to industrial OCF/FCF.
    # Do not punish a bank merely because OCF is absent/volatile; authorization for
    # financials relies on statement quality, official lineage, consensus, and the
    # sector-specific fundamental engine instead.
    if is_financial:
        cashflow_state = "SECTOR_SPECIFIC_FINANCIAL_CASHFLOW_NOT_PRIMARY"
    else:
        if not np.isfinite(ocf):
            cap = min(cap, 74.0); reasons.append("OCF_MISSING")
            cashflow_state = "OCF_MISSING"
        elif ocf <= 0:
            cap = min(cap, 78.0); reasons.append("OCF_NONPOSITIVE")
            cashflow_state = "OCF_NONPOSITIVE"
        elif np.isfinite(fcf) and fcf > 0:
            cashflow_state = "OCF_FCF_POSITIVE"
        elif np.isfinite(fcf):
            cap = min(cap, 84.0); reasons.append("FCF_NONPOSITIVE")
            cashflow_state = "OCF_POSITIVE_FCF_NONPOSITIVE"
        else:
            cap = min(cap, 86.0); reasons.append("FCF_MISSING")
            cashflow_state = "OCF_POSITIVE_FCF_MISSING"

        if np.isfinite(cashflow_cov):
            if cashflow_cov < 34:
                cap = min(cap, 72.0); reasons.append("CASHFLOW_COVERAGE<34")
            elif cashflow_cov < 66:
                cap = min(cap, 84.0); reasons.append("CASHFLOW_COVERAGE<66")

    leverage_state = "LEVERAGE_UNKNOWN"
    if is_financial:
        leverage_state = "SECTOR_SPECIFIC_FINANCIAL_LEVERAGE"
    elif np.isfinite(debt_equity):
        if debt_equity >= 2.0:
            cap = min(cap, 68.0); reasons.append("DER>=2.0"); leverage_state = "LEVERAGE_HIGH"
        elif debt_equity >= 1.5:
            cap = min(cap, 76.0); reasons.append("DER>=1.5"); leverage_state = "LEVERAGE_ELEVATED"
        elif debt_equity >= 1.2:
            cap = min(cap, 84.0); reasons.append("DER>=1.2"); leverage_state = "LEVERAGE_WATCH"
        else:
            leverage_state = "BALANCE_SHEET_CAPACITY_OK"

    if np.isfinite(consensus):
        if consensus < 50:
            cap = min(cap, 68.0); reasons.append("SOURCE_CONSENSUS<50")
        elif consensus < 70:
            cap = min(cap, 82.0); reasons.append("SOURCE_CONSENSUS<70")
    if conflicts:
        cap = min(cap, 78.0); reasons.append("CROSS_SOURCE_CONFLICT")

    refresh_state = str(freshness.get("fundamental_refresh_state", "")).upper()
    if refresh_state == "REFRESH_WINDOW":
        cap = min(cap, 84.0); reasons.append("LATEST_REPORT_REFRESH_WINDOW")
    elif refresh_state == "STALE":
        cap = min(cap, 70.0); reasons.append("LATEST_REPORT_STALE")
    elif refresh_state == "MISSING_DATE":
        # Missing period lineage remains a real-money blocker below.  Do not
        # retroactively destroy a high-quality synthetic/legacy research score
        # solely because an older cache schema omitted the date field.
        reasons.append("LATEST_REPORT_DATE_MISSING")
    growth_conflict = str(growth.get("fundamental_growth_conflict_state", "")).upper()
    if "SIGN_CONFLICT" in growth_conflict:
        cap = min(cap, 80.0); reasons.append("LATEST_GROWTH_SIGN_CONFLICT")
    if bool(growth.get("fundamental_extreme_earnings_base_review", False)):
        cap = min(cap, 82.0); reasons.append("EXTREME_EARNINGS_BASE_REVIEW")

    if official_verified:
        if np.isfinite(official_cov) and official_cov < 50:
            cap = min(cap, 86.0); reasons.append("OFFICIAL_CRITICAL_COVERAGE<50")
        official_state = "IDX_OFFICIAL_VERIFIED"
    else:
        cap = min(cap, 90.0); reasons.append("OFFICIAL_NOT_VERIFIED")
        official_state = "PROXY_OR_CROSSCHECK_ONLY"

    data_quality = 0.0
    weights = 0.0
    quality_components = [
        ({"A": 95.0, "B": 82.0, "C": 68.0, "D": 45.0}.get(grade, 55.0), 0.25),
        (history_cov, 0.20),
        (consensus, 0.15),
        (official_cov if official_verified else 0.0, 0.20),
    ]
    if not is_financial:
        quality_components.append((cashflow_cov, 0.20))
    for value, weight in quality_components:
        if np.isfinite(value):
            data_quality += float(np.clip(value, 0, 100)) * weight
            weights += weight
    data_quality = data_quality / weights if weights else 0.0

    return {
        "fundamental_score_pre_cap": np.nan,
        "fundamental_conviction_cap": round(float(np.clip(cap, 0.0, 100.0)), 1),
        "fundamental_score_cap_reason": " | ".join(dict.fromkeys(reasons)),
        "fundamental_data_quality_score": round(float(np.clip(data_quality, 0.0, 100.0)), 1),
        "fundamental_cashflow_state": cashflow_state,
        "fundamental_leverage_risk_state": leverage_state,
        "fundamental_official_state": official_state,
        "fundamental_official_source_coverage_pct": round(official_cov, 1) if np.isfinite(official_cov) else 0.0,
        "fundamental_official_verified": bool(official_verified),
        "fundamental_consensus_score": round(consensus, 1) if np.isfinite(consensus) else np.nan,
        "fundamental_history_coverage_pct": round(history_cov, 1) if np.isfinite(history_cov) else 0.0,
        "fundamental_cashflow_coverage_pct": round(cashflow_cov, 1) if np.isfinite(cashflow_cov) else 0.0,
        "fundamental_sector_specific_financial": bool(is_financial),
        "fundamental_refresh_state": freshness.get("fundamental_refresh_state"),
        "fundamental_refresh_due": bool(freshness.get("fundamental_refresh_due", False)),
        "fundamental_latest_period": freshness.get("fundamental_latest_period"),
        "fundamental_growth_basis_state": growth.get("fundamental_growth_basis_state"),
        "fundamental_growth_conflict_state": growth.get("fundamental_growth_conflict_state"),
        "fundamental_trend_state": growth.get("fundamental_trend_state"),
        "fundamental_extreme_earnings_base_review": bool(growth.get("fundamental_extreme_earnings_base_review", False)),
        "history_ocf_ttm": ocf,
        "history_fcf_ttm": fcf,
        "history_debt_equity": debt_equity,
    }


def _market_risk_off(row: Mapping[str, Any]) -> bool:
    regime = _text(_first(row, "market_regime", "macro_regime", default="")).upper()
    return regime in {"RISK_OFF", "RISK_OFF_CONTRACTION", "SELECTIVE_RISK_OFF"}


def _authorization_for_row(row: Mapping[str, Any], model: str, account_size_idr: float | None = None, requested_risk_budget_pct: float | None = None) -> dict[str, Any]:
    model_name = model.upper()
    status = _text(row.get("status")).upper()
    production = _truthy(row.get("production_gate_pass", False))
    methodology = _truthy(row.get("methodology_gate_pass", True))
    distribution = _first_num(row, "distribution_risk_score")
    independent_verified = _truthy(_first(row, "independent_price_verified", default=False))
    official_verified = _truthy(_first(row, "fundamental_official_verified", default=False))
    official_cov = _first_num(row, "fundamental_official_source_coverage_pct")
    cashflow_state = _text(row.get("fundamental_cashflow_state")).upper()
    leverage_state = _text(row.get("fundamental_leverage_risk_state")).upper()
    data_quality = _first_num(row, "fundamental_data_quality_score")
    refresh_state = _text(_first(row, "fundamental_refresh_state", "fundamental_freshness_state", default="")).upper()
    trend_state = _text(row.get("fundamental_trend_state")).upper()
    growth_conflict_state = _text(row.get("fundamental_growth_conflict_state")).upper()
    extreme_base_review = _truthy(row.get("fundamental_extreme_earnings_base_review", False))
    rr = _first_num(row, "rr1")
    entry = _first_num(row, "entry", "entry_low")
    stop = _first_num(row, "stop_loss")
    account = _finite(account_size_idr, np.nan)
    requested_risk = _finite(requested_risk_budget_pct, MAX_REAL_MONEY_RISK_BUDGET_PCT)
    effective_risk_pct = min(MAX_REAL_MONEY_RISK_BUDGET_PCT, max(0.0, requested_risk))
    risk_budget_idr = account * (effective_risk_pct / 100.0) if np.isfinite(account) and account > 0 else np.nan
    risk_per_share = entry - stop if np.isfinite(entry) and np.isfinite(stop) and entry > stop > 0 else np.nan
    risk_lots_cap = int(risk_budget_idr // (risk_per_share * 100.0)) if np.isfinite(risk_budget_idr) and np.isfinite(risk_per_share) and risk_per_share > 0 else 0
    blockers: list[str] = []
    manual: list[str] = []

    if not production:
        blockers.append("PRODUCTION_GATE")
    if not methodology:
        blockers.append("METHODOLOGY_GATE")
    if np.isfinite(distribution) and distribution >= 50:
        blockers.append("DISTRIBUTION>=50")
    market_regime = _text(_first(row, "market_regime", "macro_regime", default="")).upper()
    market_coverage = _first_num(row, "market_context_coverage_pct")
    if _market_risk_off(row):
        blockers.append("MARKET_RISK_OFF")
    if market_regime in {"", "DATA_PENDING", "MARKET_CONTEXT_UNAVAILABLE"}:
        blockers.append("MARKET_CONTEXT_PENDING")
    if np.isfinite(market_coverage) and market_coverage < 50:
        blockers.append("MARKET_CONTEXT_COVERAGE<50")
    if leverage_state in {"LEVERAGE_HIGH", "LEVERAGE_ELEVATED"}:
        blockers.append(leverage_state)
    if cashflow_state in {"OCF_MISSING", "OCF_NONPOSITIVE"}:
        blockers.append(cashflow_state)
    if not np.isfinite(risk_per_share) or risk_lots_cap < 1:
        blockers.append("RISK_GEOMETRY_OR_BUDGET_INVALID")
    if refresh_state in {"STALE", "MISSING_DATE"}:
        blockers.append(f"FUNDAMENTAL_{refresh_state}")
    elif refresh_state == "REFRESH_WINDOW":
        manual.append("LATEST_REPORT_REFRESH_REQUIRED")
    if "SIGN_CONFLICT" in growth_conflict_state:
        manual.append("LATEST_GROWTH_SIGN_CONFLICT_REVIEW")
    if extreme_base_review:
        manual.append("EXTREME_EARNINGS_BASE_REVIEW")

    if model_name == "NEXT_LEADER":
        score = _first_num(row, "v9_next_leader_score", "final_score")
        coverage = _first_num(row, "score_coverage_pct")
        business = _first_num(row, "business_quality_score")
        future = _first_num(row, "future_fundamental_score")
        technical = _first_num(row, "technical_readiness_score")
        adtv = _first_num(row, "adtv20_idr")
        if status != "BUY_ZONE": blockers.append("NOT_BUY_ZONE")
        if not np.isfinite(score) or score < 75: blockers.append("FINAL_SCORE<75")
        if not np.isfinite(coverage) or coverage < 70: blockers.append("COVERAGE<70")
        if not np.isfinite(business) or business < 60: blockers.append("BUSINESS<60")
        if not np.isfinite(future) or future < 45: blockers.append("FUTURE<45")
        if np.isfinite(technical) and technical < 68: blockers.append("TECHNICAL<68")
        if np.isfinite(adtv) and adtv < 250_000_000: blockers.append("LIQUIDITY<250M_ADTV")
        if np.isfinite(rr) and rr < 1.5: blockers.append("RR<1.5")
        if np.isfinite(data_quality) and data_quality < 70: blockers.append("FUND_DATA_QUALITY<70")
        if trend_state == "FUNDAMENTAL_DETERIORATION": blockers.append("FUNDAMENTAL_DETERIORATION")
        if not official_verified or not np.isfinite(official_cov) or official_cov < 50:
            manual.append("OFFICIAL_FILING_CONFIRMATION")
    else:
        score = _first_num(row, "v9_swing_score", "final_score")
        technical = _first_num(row, "technical_execution_score")
        risk_data = _first_num(row, "risk_data_score")
        if status not in {"EXECUTION_READY", "ENTRY_PLAN_READY"}: blockers.append("NOT_EXECUTION_CANDIDATE")
        if not np.isfinite(score) or score < 65: blockers.append("SWING_SCORE<65")
        if not np.isfinite(technical) or technical < 60: blockers.append("TECHNICAL<60")
        if np.isfinite(risk_data) and risk_data < 55: blockers.append("RISK_DATA<55")
        if trend_state == "FUNDAMENTAL_DETERIORATION": manual.append("FUNDAMENTAL_DETERIORATION_SWING_REVIEW")
        if np.isfinite(rr) and rr < 1.5: blockers.append("RR<1.5")
        if not official_verified:
            manual.append("OFFICIAL_FILING_NOT_VERIFIED")

    if not independent_verified:
        manual.append("INDEPENDENT_PRICE_CONFIRMATION")

    blockers = list(dict.fromkeys(blockers))
    manual = list(dict.fromkeys(manual))
    if blockers:
        state = "REAL_MONEY_BLOCKED"
    elif manual:
        state = "REAL_MONEY_MANUAL_CONFIRMATION_REQUIRED"
    else:
        state = "REAL_MONEY_DIRECT_VERIFIED_READY"
    return {
        "real_money_authorization_state": state,
        "real_money_authorization_pass": state == "REAL_MONEY_DIRECT_VERIFIED_READY",
        "real_money_authorization_blockers": " | ".join(blockers),
        "real_money_manual_checks": " | ".join(manual),
        "real_money_risk_budget_cap_pct": round(effective_risk_pct, 4),
        "real_money_risk_budget_idr": round(risk_budget_idr, 0) if np.isfinite(risk_budget_idr) else np.nan,
        "real_money_risk_per_share": round(risk_per_share, 4) if np.isfinite(risk_per_share) else np.nan,
        "real_money_risk_lots_cap": int(risk_lots_cap),
        "independent_price_verified": bool(independent_verified),
        "real_money_guard_version": REAL_MONEY_GUARD_VERSION,
    }


def apply_real_money_authorization(frame: pd.DataFrame, *, model: str, account_size_idr: float | None = None, requested_risk_budget_pct: float | None = None) -> pd.DataFrame:
    """Attach fail-closed authorization without removing research candidates."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    out = frame.copy()
    auth = pd.DataFrame([_authorization_for_row(row.to_dict(), model, account_size_idr=account_size_idr, requested_risk_budget_pct=requested_risk_budget_pct) for _, row in out.iterrows()], index=out.index)
    for column in auth.columns:
        out[column] = auth[column]

    direct = out["real_money_authorization_pass"].fillna(False).astype(bool)
    risk_cap_lots = pd.to_numeric(out.get("real_money_risk_lots_cap", pd.Series(0, index=out.index)), errors="coerce").fillna(0).astype(int)
    if str(model).upper() == "NEXT_LEADER":
        if "recommended_lots" in out.columns:
            existing_lots = pd.to_numeric(out["recommended_lots"], errors="coerce").fillna(0).astype(int)
            out["recommended_lots"] = np.where(direct, np.minimum(existing_lots, risk_cap_lots), 0).astype(int)
        if "recommended_allocation_idr" in out.columns:
            entry_series = pd.to_numeric(out.get("entry", out.get("entry_low", pd.Series(np.nan, index=out.index))), errors="coerce")
            lots_series = pd.to_numeric(out.get("recommended_lots", pd.Series(0, index=out.index)), errors="coerce").fillna(0)
            risk_capped_allocation = lots_series * 100.0 * entry_series
            existing = pd.to_numeric(out["recommended_allocation_idr"], errors="coerce").fillna(0.0)
            out["recommended_allocation_idr"] = np.where(direct, np.minimum(existing, risk_capped_allocation.fillna(0.0)), 0.0)
    else:
        if "stockbit_order_lots" in out.columns:
            existing_lots = pd.to_numeric(out["stockbit_order_lots"], errors="coerce").fillna(0).astype(int)
            out["stockbit_order_lots"] = np.where(direct, np.minimum(existing_lots, risk_cap_lots), 0).astype(int)
        if "order_builder_eligible" in out.columns:
            out["order_builder_eligible"] = out["order_builder_eligible"].fillna(False).astype(bool) & direct
        if "order_ready" in out.columns:
            out["order_ready"] = out["order_ready"].fillna(False).astype(bool) & direct
    return out


__all__ = [
    "REAL_MONEY_GUARD_VERSION",
    "MAX_REAL_MONEY_RISK_BUDGET_PCT",
    "fundamental_conviction_profile",
    "apply_real_money_authorization",
]
