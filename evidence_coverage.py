from __future__ import annotations

"""Fail-closed, scoring-independent evidence coverage measurement for Pasticuan."""

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from idx_trading_calendar import latest_expected_completed_session, previous_idx_session, trading_session_age

ENGINE_VERSION = "1.0.0-pasticuan-independent-evidence-coverage"
OFFICIAL_TOKENS = ("IDX", "XBRL", "OFFICIAL", "ISSUER")
CORE_FIELDS: Mapping[str, tuple[str, ...]] = {
    "revenue": ("revenue", "total_revenue"),
    "net_income": ("net_income", "net_profit", "earnings"),
    "assets": ("assets", "total_assets"),
    "liabilities": ("liabilities", "total_liabilities"),
    "equity": ("equity", "total_equity"),
    "cash": ("cash", "cash_and_equivalents"),
    "short_term_debt": ("short_term_debt", "current_debt"),
    "long_term_debt": ("long_term_debt", "noncurrent_debt"),
    "debt": ("total_debt", "debt", "short_term_debt", "long_term_debt"),
    "ocf": ("ocf", "operating_cash_flow", "cash_flow_from_operations"),
    "capex": ("capex", "capital_expenditure"),
}


@dataclass(frozen=True)
class CoveragePolicy:
    require_forward: bool = True
    require_foreign: bool = True
    require_broker: bool = False
    fundamental_max_age_days: int = 550
    forward_max_age_days: int = 365


def _ticker(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text[:-3] if text.endswith(".JK") else text


def _frame(value: Any) -> pd.DataFrame:
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _truth(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value or "").strip().upper() in {"1", "TRUE", "YES", "VERIFIED", "MATCH", "PASS", "VALID"}


def _number(row: Mapping[str, Any], aliases: Iterable[str]) -> float:
    for name in aliases:
        value = pd.to_numeric(row.get(name), errors="coerce")
        if pd.notna(value) and np.isfinite(float(value)):
            return float(value)
    return np.nan


def _date(row: Mapping[str, Any], aliases: Iterable[str]) -> pd.Timestamp | None:
    for name in aliases:
        value = pd.to_datetime(row.get(name), errors="coerce", utc=True)
        if pd.notna(value):
            return pd.Timestamp(value).tz_convert("Asia/Jakarta").tz_localize(None).normalize()
    return None


def _expected_sessions(as_of: Any, count: int = 20) -> list[pd.Timestamp]:
    current = latest_expected_completed_session(as_of)
    result: list[pd.Timestamp] = []
    while len(result) < count:
        result.append(current)
        current = previous_idx_session(current, include_date=False)
    return result


def _rows_by_ticker(frame: pd.DataFrame, universe: set[str]) -> dict[str, pd.DataFrame]:
    if frame.empty:
        return {}
    ticker_col = next((name for name in ("ticker", "symbol", "code", "stock_code") if name in frame), "")
    if not ticker_col:
        return {}
    work = frame.copy()
    work["_ticker"] = work[ticker_col].map(_ticker)
    work = work[work["_ticker"].isin(universe)]
    return {name: rows.copy() for name, rows in work.groupby("_ticker", sort=False)}


def build_evidence_coverage(
    universe: Iterable[Any],
    *,
    ohlcv: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    forward: pd.DataFrame | None = None,
    foreign: pd.DataFrame | None = None,
    broker: pd.DataFrame | None = None,
    as_of: Any = None,
    policy: CoveragePolicy | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    policy = policy or CoveragePolicy()
    names = list(dict.fromkeys(_ticker(value) for value in universe if _ticker(value)))
    universe_set = set(names)
    expected = _expected_sessions(as_of, 20)
    expected_set = set(expected)
    latest_expected = expected[0]
    as_of_day = pd.Timestamp(as_of or pd.Timestamp.now(tz="Asia/Jakarta"))
    if as_of_day.tzinfo is not None:
        as_of_day = as_of_day.tz_convert("Asia/Jakarta").tz_localize(None)
    as_of_day = as_of_day.normalize()
    groups = {
        "ohlcv": _rows_by_ticker(_frame(ohlcv), universe_set),
        "fundamental": _rows_by_ticker(_frame(fundamentals), universe_set),
        "forward": _rows_by_ticker(_frame(forward), universe_set),
        "foreign": _rows_by_ticker(_frame(foreign), universe_set),
        "broker": _rows_by_ticker(_frame(broker), universe_set),
    }
    rows: list[dict[str, Any]] = []
    for name in names:
        reasons: list[str] = []
        result: dict[str, Any] = {"ticker": name}

        price_rows = groups["ohlcv"].get(name, pd.DataFrame())
        if price_rows.empty:
            ohlcv_state, latest_price = "NO_DATA", None
        else:
            date_col = next((col for col in ("trade_date", "date", "timestamp") if col in price_rows), "")
            dates = pd.to_datetime(price_rows.get(date_col), errors="coerce").dt.tz_localize(None).dt.normalize() if date_col else pd.Series(pd.NaT, index=price_rows.index)
            local = price_rows.assign(_date=dates).dropna(subset=["_date"])
            local = local[local["_date"] <= latest_expected].sort_values("_date")
            latest_price = local.iloc[-1] if not local.empty else None
            if latest_price is None:
                ohlcv_state = "INVALID"
            elif latest_price["_date"] != latest_expected:
                ohlcv_state = "STALE"
            elif _number(latest_price, ("close", "Close", "adj_close", "Adj Close")) <= 0:
                ohlcv_state = "INVALID"
            elif _number(latest_price, ("volume", "Volume")) <= 0:
                ohlcv_state = "ZERO_VOLUME"
            else:
                ohlcv_state = "VALID"
        result.update({"ohlcv_state": ohlcv_state, "ohlcv_valid": ohlcv_state == "VALID", "ohlcv_latest_session": latest_price["_date"].date().isoformat() if latest_price is not None else ""})
        if ohlcv_state != "VALID":
            reasons.append({"NO_DATA": "NO_OHLCV", "STALE": "STALE", "ZERO_VOLUME": "ZERO_VOLUME", "INVALID": "INVALID_OHLCV"}[ohlcv_state])

        fund_rows = groups["fundamental"].get(name, pd.DataFrame())
        fund_row: Mapping[str, Any] = {}
        if not fund_rows.empty:
            dated = fund_rows.copy()
            dated["_report_date"] = dated.apply(lambda item: _date(item, ("report_date", "as_of", "period_end", "reporting_period")), axis=1)
            dated["_publication_date"] = dated.apply(
                lambda item: _date(item, ("publication_date", "published_at", "filing_date", "available_at")), axis=1
            )
            dated["_quality"] = dated.apply(
                lambda item: 3 if (_truth(item.get("source_verified")) and str(item.get("source_url") or "").strip() and any(token in str(item.get("source") or item.get("provider") or "").upper() for token in OFFICIAL_TOKENS))
                else (2 if (_truth(item.get("source_verified")) and (_truth(item.get("issuer_match")) or _truth(item.get("identity_verified")))) else (1 if _truth(item.get("source_verified")) else 0)),
                axis=1,
            )
            dated = dated.dropna(subset=["_report_date", "_publication_date"])
            dated = dated[
                (dated["_report_date"] <= dated["_publication_date"])
                & (dated["_publication_date"] <= as_of_day)
                & (dated["_report_date"] <= as_of_day)
            ].sort_values(["_quality", "_report_date", "_publication_date"])
            if not dated.empty:
                fund_row = dated.iloc[-1]
        has_fund = bool(len(fund_row))
        values = {field: _number(fund_row, aliases) for field, aliases in CORE_FIELDS.items()}
        period = str(fund_row.get("period_type") or fund_row.get("statement_basis") or "").strip().upper()
        period_valid = period in {"FY", "ANNUAL", "YTD", "YTD_CUMULATIVE", "Q1", "Q2", "Q3", "Q4", "QUARTER", "QUARTERLY", "TTM"}
        identity_valid = _truth(fund_row.get("issuer_match")) or _truth(fund_row.get("identity_verified"))
        source_verified = _truth(fund_row.get("source_verified"))
        source = str(fund_row.get("source") or fund_row.get("source_family") or fund_row.get("provider") or "").strip()
        source_url = str(fund_row.get("source_url") or "").strip()
        report_date = fund_row.get("_report_date") if has_fund else None
        publication_date = fund_row.get("_publication_date") if has_fund else None
        age = (as_of_day - report_date).days if isinstance(report_date, pd.Timestamp) else None
        fresh = age is not None and 0 <= age <= policy.fundamental_max_age_days
        core_available = all(np.isfinite(values[field]) for field in ("revenue", "net_income", "assets", "liabilities", "equity"))
        official = bool(source_verified and source_url and any(token in source.upper() for token in OFFICIAL_TOKENS))
        ocf_period = str(fund_row.get("ocf_period_type") or period).upper()
        capex_period = str(fund_row.get("capex_period_type") or period).upper()
        fcf_available = bool(np.isfinite(values["ocf"]) and np.isfinite(values["capex"]) and period_valid and ocf_period == capex_period)
        fundamental_valid = bool(core_available and identity_valid and source_verified and period_valid and fresh)
        result.update({"fundamental_valid": fundamental_valid, "fundamental_official": official and fundamental_valid, "fundamental_source": source, "fundamental_report_date": report_date.date().isoformat() if isinstance(report_date, pd.Timestamp) else "", "fundamental_publication_date": publication_date.date().isoformat() if isinstance(publication_date, pd.Timestamp) else "", "fundamental_freshness_state": "FRESH" if fresh else ("STALE" if report_date is not None else "MISSING"), "ocf_available": bool(np.isfinite(values["ocf"])), "capex_available": bool(np.isfinite(values["capex"])), "fcf_available": fcf_available})
        for field, value in values.items():
            result[f"fundamental_{field}_available"] = bool(np.isfinite(value))
        if not has_fund:
            reasons.append("NO_REPORT")
        else:
            if not identity_valid:
                reasons.append("IDENTITY_MISMATCH")
            if not period_valid:
                reasons.append("WRONG_REPORTING_PERIOD")
            if not fresh:
                reasons.append("STALE")
            if not core_available:
                reasons.append("MISSING_CORE_FUNDAMENTAL")
            if not np.isfinite(values["ocf"]):
                reasons.append("MISSING_CASHFLOW")
            if np.isfinite(values["ocf"]) and not fcf_available:
                reasons.append("MISSING_OR_INCOMPATIBLE_CAPEX")

        forward_rows = groups["forward"].get(name, pd.DataFrame())
        forward_valid = 0
        if not forward_rows.empty:
            work = forward_rows.copy()
            work["_evidence_date"] = work.apply(lambda item: _date(item, ("evidence_date", "date", "as_of")), axis=1)
            work["_publication_date"] = work.apply(
                lambda item: _date(item, ("publication_date", "published_at", "evidence_date", "date", "as_of")), axis=1
            )
            work = work.drop_duplicates(subset=[col for col in ("_ticker", "evidence_type", "_evidence_date", "source_url") if col in work])
            for _, item in work.iterrows():
                item_date = item.get("_evidence_date")
                publication = item.get("_publication_date")
                item_fresh = (
                    isinstance(item_date, pd.Timestamp)
                    and isinstance(publication, pd.Timestamp)
                    and item_date <= publication <= as_of_day
                    and 0 <= (as_of_day - item_date).days <= policy.forward_max_age_days
                )
                if item_fresh and _truth(item.get("issuer_match")) and _truth(item.get("source_verified")) and str(item.get("source_url") or "").startswith("http") and str(item.get("evidence_type") or "").strip():
                    forward_valid += 1
        result.update({"forward_evidence_available": forward_valid > 0, "forward_evidence_count": forward_valid})
        if policy.require_forward and forward_valid == 0:
            reasons.append("NO_FORWARD_EVIDENCE")

        foreign_rows = groups["foreign"].get(name, pd.DataFrame())
        observed_sessions = 0
        latest_foreign = None
        foreign_source = ""
        if not foreign_rows.empty:
            work = foreign_rows.copy()
            date_col = next((col for col in ("trade_date", "date", "foreign_latest_session") if col in work), "")
            work["_date"] = pd.to_datetime(work.get(date_col), errors="coerce").dt.tz_localize(None).dt.normalize() if date_col else pd.NaT
            work = work[work["_date"].isin(expected_set)]
            observed_sessions = int(work["_date"].nunique())
            latest_foreign = work["_date"].max() if not work.empty else None
            source_col = next((col for col in ("source", "foreign_source", "zapi_flow_source") if col in work), "")
            foreign_source = " | ".join(sorted(set(str(value) for value in work.get(source_col, pd.Series(dtype=str)).dropna() if str(value).strip())))
        foreign_ratio = observed_sessions / 20.0
        foreign_fresh = isinstance(latest_foreign, pd.Timestamp) and trading_session_age(latest_foreign, latest_expected) == 0
        foreign_sufficient = bool(foreign_ratio >= 0.90 and foreign_fresh and foreign_source)
        result.update({"foreign_source": foreign_source, "foreign_latest_session": latest_foreign.date().isoformat() if isinstance(latest_foreign, pd.Timestamp) else "", "foreign_expected_sessions": 20, "foreign_observed_sessions": observed_sessions, "foreign_coverage_ratio": foreign_ratio, "foreign_20_session_sufficient": foreign_sufficient, "foreign_freshness_state": "FRESH" if foreign_fresh else ("STALE" if latest_foreign is not None else "MISSING"), "foreign_window_state": "SUFFICIENT" if foreign_sufficient else ("PARTIAL" if observed_sessions else "MISSING")})
        if policy.require_foreign and not foreign_sufficient:
            reasons.append("INSUFFICIENT_FOREIGN_HISTORY" if observed_sessions else "PROVIDER_UNAVAILABLE")

        broker_rows = groups["broker"].get(name, pd.DataFrame())
        broker_days = broker_count = 0
        broker_sufficient = broker_direct = False
        if not broker_rows.empty:
            date_col = next((col for col in ("trade_date", "date") if col in broker_rows), "")
            dates = pd.to_datetime(broker_rows.get(date_col), errors="coerce").dt.tz_localize(None).dt.normalize() if date_col else pd.Series(pd.NaT, index=broker_rows.index)
            work = broker_rows.assign(_date=dates)
            work = work[work["_date"].isin(expected_set)]
            broker_days = int(work["_date"].nunique())
            broker_col = next((col for col in ("broker", "broker_code", "participant") if col in work), "")
            broker_count = int(work[broker_col].dropna().astype(str).nunique()) if broker_col else 0
            broker_direct = bool(work.apply(lambda item: _truth(item.get("source_verified")) and str(item.get("evidence_status") or "").upper() == "BROKER_DIRECT", axis=1).any())
            broker_sufficient = bool(work.get("broker_sufficient", pd.Series(False, index=work.index)).map(_truth).any())
        result.update({"broker_distinct_trading_days": broker_days, "broker_distinct_brokers": broker_count, "broker_direct": broker_direct, "broker_sufficient": broker_sufficient, "broker_status": "BROKER_DIRECT" if broker_direct else ("CONFIRMATION_ONLY" if broker_days else "MISSING")})
        if policy.require_broker and not broker_sufficient:
            reasons.append("INSUFFICIENT_BROKER_EVIDENCE")

        ready = bool(result["ohlcv_valid"] and result["fundamental_valid"] and (result["forward_evidence_available"] or not policy.require_forward) and (foreign_sufficient or not policy.require_foreign) and (broker_sufficient or not policy.require_broker))
        result["fully_evidence_ready"] = ready
        result["missing_reasons"] = "|".join(dict.fromkeys(reasons))
        rows.append(result)

    detail = pd.DataFrame(rows)
    total = len(detail)
    metrics = ("ohlcv_valid", "fundamental_valid", "fundamental_official", "ocf_available", "fcf_available", "forward_evidence_available", "foreign_20_session_sufficient", "broker_sufficient", "fully_evidence_ready")
    summary: dict[str, Any] = {"engine_version": ENGINE_VERSION, "scanner": "PASTICUAN", "total_universe": total, "as_of": as_of_day.date().isoformat(), "latest_expected_idx_session": latest_expected.date().isoformat()}
    for metric in metrics:
        count = int(detail.get(metric, pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        summary[metric] = {"count": count, "percentage": round(100.0 * count / total, 2) if total else 0.0}
    field_counts = {field: int(detail.get(f"fundamental_{field}_available", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) for field in CORE_FIELDS}
    summary["fundamental_field_coverage"] = {field: {"count": count, "percentage": round(100.0 * count / total, 2) if total else 0.0} for field, count in field_counts.items()}
    reason_counts: dict[str, int] = {}
    for value in detail.get("missing_reasons", pd.Series(dtype=str)):
        for reason in str(value).split("|"):
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
    summary["missing_reason_distribution"] = dict(sorted(reason_counts.items()))
    summary["ohlcv_state_distribution"] = detail.get("ohlcv_state", pd.Series(dtype=str)).value_counts().sort_index().to_dict()
    return detail, summary


__all__ = ["CORE_FIELDS", "CoveragePolicy", "ENGINE_VERSION", "build_evidence_coverage"]
