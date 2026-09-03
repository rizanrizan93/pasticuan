from __future__ import annotations

"""Runtime bridge for scanner-neutral fundamental facts.

The Shared Evidence Hub stores facts only. This module reads those facts into a
canonical bundle, publishes source-backed runtime metrics, and provides a bounded
structured-provider collector. It intentionally contains no scanner score, rank,
entry, gate, recommendation, or future-fundamental logic.
"""

from datetime import date, datetime, timezone
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence

import requests

from shared_evidence_hub import HubConfig, MissingReason, SupabaseEvidenceBackend, normalize_failure_reason


TABLE = "evidence_fundamental_metrics"
RUNTIME_VERSION = "1.1.0-phase5.6-runtime"
PLUANG_RESOLVE_URL = "https://api.zpi.web.id/v1/finance:pluang/resolve"
PLUANG_FUNDAMENTALS_URL = "https://api.zpi.web.id/v1/finance:pluang/fundamentals"
PLUANG_FINANCIALS_URL = "https://api.zpi.web.id/v1/finance:pluang/financials"
YAHOO_SUMMARY_URL = "https://api.zpi.web.id/v1/finance:yahoo-finance/summary"
YAHOO_FINANCIALS_URL = "https://api.zpi.web.id/v1/finance:yahoo-finance/financials"
REQUEST_TIMEOUT_SECONDS = 18

# These fields are stored as decimal ratios in PASTICUAN's operational snapshot,
# but the shared contract names them *_pct and therefore stores percentage points.
_OPERATIONAL_RATIO_TO_PERCENT = {
    "revenue_growth": "revenue_growth_pct",
    "earnings_growth": "earnings_growth_pct",
    "roe": "roe_pct",
    "roa": "roa_pct",
    "roic_proxy": "roic_proxy_pct",
    "net_margin": "net_margin_pct",
    "operating_margin": "operating_margin_pct",
}
_OPERATIONAL_DIRECT = {
    "operating_cash_flow": ("operating_cash_flow", "CURRENCY_NATIVE"),
    "free_cash_flow": ("free_cash_flow", "CURRENCY_NATIVE"),
    "cash_conversion_ttm": ("cash_conversion_ttm", "RATIO"),
    "debt_equity": ("debt_equity", "RATIO"),
    "net_debt_ebitda": ("net_debt_ebitda", "RATIO"),
    "interest_coverage": ("interest_coverage", "RATIO"),
    "market_cap": ("market_cap", "CURRENCY_NATIVE"),
    "fundamental_coverage": ("fundamental_coverage_pct", "PERCENT"),
}
_PERCENT_METRICS = {
    "revenue_growth_pct", "earnings_growth_pct", "roe_pct", "roa_pct",
    "roic_proxy_pct", "net_margin_pct", "operating_margin_pct", "gross_margin_pct",
}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def bare_ticker(value: Any) -> str:
    text = _clean(value).upper()
    return text[:-3] if text.endswith(".JK") else text


def jk_ticker(value: Any) -> str:
    text = bare_ticker(value)
    return f"{text}.JK" if text else ""


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _iso_date(value: Any) -> str | None:
    text = _clean(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        return None


def _iso_stamp(value: Any) -> str | None:
    text = _clean(value)
    if not text:
        return None
    try:
        stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).isoformat()


def _stamp_ord(value: Any) -> float:
    text = _iso_stamp(value)
    if not text:
        return 0.0
    return datetime.fromisoformat(text).timestamp()


def _date_ord(value: Any) -> int:
    text = _iso_date(value)
    return date.fromisoformat(text).toordinal() if text else 0


def _hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _scaled_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return _finite(value)
    text = _clean(value).replace("\u00a0", "").replace(" ", "")
    if not text or text.lower() in {"-", "—", "na", "n/a", "nan", "none", "null"}:
        return None
    text = re.sub(r"^(rp|idr|usd)", "", text, flags=re.IGNORECASE).replace(",", "")
    if text.endswith("%") or text.lower().endswith("x"):
        text = text[:-1]
    multiplier = 1.0
    if text and text[-1:].upper() in {"K", "M", "B", "T"}:
        multiplier = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[text[-1].upper()]
        text = text[:-1]
    try:
        number = float(text) * multiplier
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _quarter_end(value: Any) -> str | None:
    text = _clean(value).replace("’", "'").replace("‘", "'")
    match = re.search(r"Q([1-4]).*?(\d{2,4})", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return _iso_date(value)
    q = int(match.group(1))
    year_text = match.group(2)
    year = int(year_text) if len(year_text) == 4 else 2000 + int(year_text)
    month = q * 3
    day = 31 if month in {3, 12} else 30
    return date(year, month, day).isoformat()


def normalize_operational_snapshot_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert PASTICUAN operational snapshot units into canonical shared units."""
    output: list[dict[str, Any]] = []
    fetched_at = datetime.now(timezone.utc).isoformat()
    for raw in rows:
        source = dict(raw)
        ticker = bare_ticker(source.get("ticker"))
        families = _clean(source.get("fundamental_source_families"))
        observed_at = _iso_stamp(source.get("fundamental_fetched_at") or source.get("as_of") or source.get("updated_at"))
        if not ticker or not families or not observed_at:
            continue
        period_end = _iso_date(source.get("period_end"))
        statement_date = _iso_date(source.get("statement_date"))
        record_hash = _clean(source.get("content_hash")) or _hash({
            "ticker": ticker,
            "period_end": period_end,
            "statement_date": statement_date,
            "families": families,
            "snapshot": source,
        })
        for field, metric in _OPERATIONAL_RATIO_TO_PERCENT.items():
            value = _finite(source.get(field))
            if value is None:
                continue
            output.append({
                "provider": "OPERATIONAL_FUNDAMENTAL_BRIDGE",
                "ticker": ticker,
                "period_end": period_end,
                "statement_date": statement_date,
                "metric_name": metric,
                "metric_value": value * 100.0,
                "metric_unit": "PERCENT",
                "source_families": families,
                "official_verified": False,
                "source_record_hash": record_hash,
                "lineage_state": "BRIDGED_AGGREGATED_OPERATIONAL_METRIC_PERCENT_CANONICAL_V2",
                "observed_at": observed_at,
                "validation_state": "VALID",
                "fetched_at": fetched_at,
            })
        for field, (metric, unit) in _OPERATIONAL_DIRECT.items():
            value = _finite(source.get(field))
            if value is None:
                continue
            output.append({
                "provider": "OPERATIONAL_FUNDAMENTAL_BRIDGE",
                "ticker": ticker,
                "period_end": period_end,
                "statement_date": statement_date,
                "metric_name": metric,
                "metric_value": value,
                "metric_unit": unit,
                "source_families": families,
                "official_verified": False,
                "source_record_hash": record_hash,
                "lineage_state": "BRIDGED_AGGREGATED_OPERATIONAL_METRIC_PERCENT_CANONICAL_V2",
                "observed_at": observed_at,
                "validation_state": "VALID",
                "fetched_at": fetched_at,
            })
    return output


def _growth_pct(current: float | None, prior: float | None, *, earnings: bool = False) -> float | None:
    if current is None or prior is None or prior == 0:
        return None
    if earnings and ((prior < 0 < current) or (prior > 0 > current)):
        return None
    if earnings and prior < 0 and current < 0:
        return 100.0 * (current - prior) / abs(prior)
    return 100.0 * (current / prior - 1.0)


def _same_quarter_prior(rows: Sequence[Mapping[str, Any]], metric: str, latest_period: str) -> float | None:
    latest = date.fromisoformat(latest_period)
    candidates: list[tuple[int, float]] = []
    for row in rows:
        if _clean(row.get("metric_name")) != metric:
            continue
        period = _iso_date(row.get("period_end"))
        value = _finite(row.get("metric_value"))
        if not period or value is None:
            continue
        days = (latest - date.fromisoformat(period)).days
        if 300 <= days <= 430:
            candidates.append((abs(days - 365), value))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def canonicalize_metric_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build one canonical proxy + official bundle per ticker without scanner conclusions."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in rows:
        row = dict(raw)
        ticker = bare_ticker(row.get("ticker"))
        value = _finite(row.get("metric_value"))
        metric = _clean(row.get("metric_name")).lower()
        if not ticker or not metric or value is None or _clean(row.get("validation_state")).upper() != "VALID":
            continue
        # Compatibility for the pre-v2 aggregate bridge. Corrected rows carry an
        # explicit canonical-v2 lineage marker and therefore are never rescaled.
        if (
            row.get("provider") == "OPERATIONAL_FUNDAMENTAL_BRIDGE"
            and metric in _PERCENT_METRICS
            and "PERCENT_CANONICAL_V2" not in _clean(row.get("lineage_state"))
        ):
            value *= 100.0
        row["metric_value"] = value
        row["ticker"] = ticker
        row["metric_name"] = metric
        grouped.setdefault(ticker, []).append(row)

    result: dict[str, dict[str, Any]] = {}
    for ticker, items in grouped.items():
        proxy_rows = [row for row in items if not bool(row.get("official_verified"))]
        official_rows = [row for row in items if bool(row.get("official_verified")) and _iso_date(row.get("period_end"))]

        proxy: dict[str, float] = {}
        proxy_meta: dict[str, dict[str, Any]] = {}
        for row in proxy_rows:
            metric = row["metric_name"]
            current = proxy_meta.get(metric)
            key = (_stamp_ord(row.get("observed_at")), _date_ord(row.get("period_end")), _clean(row.get("provider")))
            old_key = (
                _stamp_ord(current.get("observed_at")), _date_ord(current.get("period_end")), _clean(current.get("provider"))
            ) if current else (-1.0, -1, "")
            if key >= old_key:
                proxy[metric] = float(row["metric_value"])
                proxy_meta[metric] = row

        official_period = max((_iso_date(row.get("period_end")) for row in official_rows), default=None)
        official_period_rows = [row for row in official_rows if _iso_date(row.get("period_end")) == official_period] if official_period else []
        official: dict[str, float] = {}
        for row in sorted(official_period_rows, key=lambda r: _stamp_ord(r.get("observed_at"))):
            official[row["metric_name"]] = float(row["metric_value"])

        if official_period:
            revenue = official.get("revenue")
            net_income = official.get("net_income")
            equity = official.get("equity")
            liabilities = official.get("total_liabilities")
            debt = official.get("total_debt")
            cash = official.get("cash")
            ocf = official.get("operating_cash_flow")
            if "net_margin_pct" not in official and revenue not in (None, 0) and net_income is not None:
                official["net_margin_pct"] = 100.0 * net_income / revenue
            if "total_liabilities_to_equity" not in official and equity not in (None, 0) and liabilities is not None:
                official["total_liabilities_to_equity"] = liabilities / equity
            if "interest_bearing_debt_to_equity" not in official and equity not in (None, 0) and debt is not None:
                official["interest_bearing_debt_to_equity"] = debt / equity
            if "cash_to_debt_ratio" not in official and debt not in (None, 0) and cash is not None:
                official["cash_to_debt_ratio"] = cash / debt
            if "net_debt_to_equity" not in official and equity not in (None, 0) and debt is not None and cash is not None:
                official["net_debt_to_equity"] = (debt - cash) / equity
            if "ocf_conversion_ratio" not in official and net_income not in (None, 0) and ocf is not None:
                official["ocf_conversion_ratio"] = ocf / net_income
            prior_revenue = _same_quarter_prior(official_rows, "revenue", official_period)
            prior_income = _same_quarter_prior(official_rows, "net_income", official_period)
            rev_growth = _growth_pct(revenue, prior_revenue)
            earn_growth = _growth_pct(net_income, prior_income, earnings=True)
            if rev_growth is not None:
                official["revenue_growth_yoy_pct"] = rev_growth
            if earn_growth is not None:
                official["earnings_growth_yoy_pct"] = earn_growth

        # Exact current-period facts are also valid field-level fallbacks when a
        # normalized proxy metric is absent. They do not replace newer proxy facts.
        for metric, value in official.items():
            proxy.setdefault(metric, value)

        proxy_observed = max((_iso_stamp(row.get("observed_at")) for row in proxy_rows if _iso_stamp(row.get("observed_at"))), default=None)
        official_observed = max((_iso_stamp(row.get("observed_at")) for row in official_period_rows if _iso_stamp(row.get("observed_at"))), default=None)
        proxy_period = max((_iso_date(row.get("period_end")) for row in proxy_rows if _iso_date(row.get("period_end"))), default=None) or official_period
        source_families = sorted({_clean(row.get("source_families")) for row in items if _clean(row.get("source_families"))})
        official_expected = {
            "revenue", "net_income", "operating_cash_flow", "cash", "equity",
            "total_assets", "total_liabilities", "current_ratio",
            "interest_bearing_debt_to_equity", "cash_to_debt_ratio",
        }
        official_coverage = 100.0 * sum(metric in official for metric in official_expected) / len(official_expected) if official else 0.0
        result[ticker] = {
            "ticker": ticker,
            "proxy_metrics": proxy,
            "proxy_period_end": proxy_period,
            "proxy_observed_at": proxy_observed or official_observed,
            "official_metrics": official,
            "official_period_end": official_period,
            "official_observed_at": official_observed,
            "official_coverage_pct": round(official_coverage, 1),
            "source_families": source_families,
            "runtime_version": RUNTIME_VERSION,
        }
    return result


def _metric_row(
    *, ticker: str, provider: str, source_families: str, metric_name: str,
    metric_value: float, metric_unit: str, observed_at: str, period_end: str | None,
    payload_hash: str,
) -> dict[str, Any]:
    return {
        "provider": provider,
        "ticker": bare_ticker(ticker),
        "period_end": period_end,
        "statement_date": period_end,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "metric_unit": metric_unit,
        "source_families": source_families,
        "official_verified": False,
        "source_record_hash": payload_hash,
        "lineage_state": "STRUCTURED_PROVIDER_OBSERVED_NOT_IDX_OFFICIAL",
        "observed_at": observed_at,
        "validation_state": "VALID",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def _latest_item(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    records = [item for item in items if isinstance(item, Mapping)]
    records.sort(key=lambda item: _date_ord(item.get("date")), reverse=True)
    return records[0] if records else {}


def _value(item: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        number = _scaled_number(item.get(key))
        if number is not None:
            return number
    return None


def _add(rows: list[dict[str, Any]], *, ticker: str, provider: str, family: str, metric: str,
         value: Any, unit: str, observed_at: str, period_end: str | None, payload_hash: str) -> None:
    number = _scaled_number(value)
    if number is None:
        return
    rows.append(_metric_row(
        ticker=ticker, provider=provider, source_families=family, metric_name=metric,
        metric_value=number, metric_unit=unit, observed_at=observed_at,
        period_end=period_end, payload_hash=payload_hash,
    ))


def normalize_pluang_payloads(ticker: str, fundamentals: Mapping[str, Any], financials: Mapping[str, Any], *, observed_at: str) -> list[dict[str, Any]]:
    code = bare_ticker(fundamentals.get("code") or financials.get("code"))
    if not code or code != bare_ticker(ticker):
        raise RuntimeError(MissingReason.ISSUER_MISMATCH.value)
    rows: list[dict[str, Any]] = []
    fundamentals_hash = _hash({"ticker": code, "payload": fundamentals})
    financials_hash = _hash({"ticker": code, "payload": financials})
    ratios = fundamentals.get("ratios") if isinstance(fundamentals.get("ratios"), Mapping) else {}
    profitability = ratios.get("profitability") if isinstance(ratios.get("profitability"), Mapping) else {}
    solvency = ratios.get("solvency") if isinstance(ratios.get("solvency"), Mapping) else {}
    overview = fundamentals.get("overview") if isinstance(fundamentals.get("overview"), Mapping) else {}
    earnings = fundamentals.get("earnings") if isinstance(fundamentals.get("earnings"), list) else []
    latest_period = next((_quarter_end(item.get("quarter")) for item in reversed(earnings) if isinstance(item, Mapping) and _quarter_end(item.get("quarter"))), None)
    for key, metric in (("roe", "roe_pct"), ("roa", "roa_pct"), ("npm", "net_margin_pct"), ("opm", "operating_margin_pct"), ("gpm", "gross_margin_pct")):
        _add(rows, ticker=code, provider="PLUANG_STRUCTURED_VIA_ZAPI", family="PLUANG_FUNDAMENTALS_VIA_ZAPI", metric=metric,
             value=profitability.get(key), unit="PERCENT", observed_at=observed_at, period_end=latest_period, payload_hash=fundamentals_hash)
    _add(rows, ticker=code, provider="PLUANG_STRUCTURED_VIA_ZAPI", family="PLUANG_FUNDAMENTALS_VIA_ZAPI", metric="current_ratio",
         value=solvency.get("cr"), unit="RATIO", observed_at=observed_at, period_end=latest_period, payload_hash=fundamentals_hash)
    de = _scaled_number(solvency.get("de"))
    if de is not None:
        _add(rows, ticker=code, provider="PLUANG_STRUCTURED_VIA_ZAPI", family="PLUANG_FUNDAMENTALS_VIA_ZAPI", metric="debt_equity",
             value=de / 100.0, unit="RATIO", observed_at=observed_at, period_end=latest_period, payload_hash=fundamentals_hash)
    for key, metric, unit in (("eps", "eps", "CURRENCY_PER_SHARE"), ("bvps", "book_value_per_share", "CURRENCY_PER_SHARE"), ("revenue", "revenue", "CURRENCY_NATIVE"), ("net_income", "net_income", "CURRENCY_NATIVE"), ("gross_profit", "gross_profit", "CURRENCY_NATIVE")):
        _add(rows, ticker=code, provider="PLUANG_STRUCTURED_VIA_ZAPI", family="PLUANG_FUNDAMENTALS_VIA_ZAPI", metric=metric,
             value=overview.get(key), unit=unit, observed_at=observed_at, period_end=latest_period, payload_hash=fundamentals_hash)

    quarterly = financials.get("quarterly") if isinstance(financials.get("quarterly"), Mapping) else {}
    blocks = {
        "incomeStatement": {"revenue": ("revenue", "CURRENCY_NATIVE"), "netProfitLoss": ("net_income", "CURRENCY_NATIVE"), "profitMargin": ("net_margin_ratio", "RATIO")},
        "balanceSheet": {"assets": ("total_assets", "CURRENCY_NATIVE"), "liabilities": ("total_liabilities", "CURRENCY_NATIVE"), "debtToAsset": ("debt_to_asset", "RATIO")},
        "cashFlow": {"operating": ("operating_cash_flow", "CURRENCY_NATIVE"), "investing": ("investing_cash_flow", "CURRENCY_NATIVE"), "finance": ("financing_cash_flow", "CURRENCY_NATIVE"), "netCF": ("net_cash_flow", "CURRENCY_NATIVE")},
    }
    for block_name, mapping in blocks.items():
        block = quarterly.get(block_name) if isinstance(quarterly.get(block_name), Mapping) else {}
        chart = block.get("chart") if isinstance(block.get("chart"), list) else []
        candidates = [item for item in chart if isinstance(item, Mapping) and _quarter_end(item.get("timeframe"))]
        candidates.sort(key=lambda item: _date_ord(_quarter_end(item.get("timeframe"))), reverse=True)
        if not candidates:
            continue
        latest = candidates[0]
        period = _quarter_end(latest.get("timeframe"))
        for key, (metric, unit) in mapping.items():
            _add(rows, ticker=code, provider="PLUANG_STRUCTURED_VIA_ZAPI", family="PLUANG_FINANCIALS_VIA_ZAPI", metric=metric,
                 value=latest.get(key), unit=unit, observed_at=observed_at, period_end=period, payload_hash=financials_hash)
    return rows


def normalize_yahoo_payloads(ticker: str, summary: Mapping[str, Any], statements: Mapping[str, Mapping[str, Any]], *, observed_at: str) -> list[dict[str, Any]]:
    symbol = jk_ticker(summary.get("symbol") or ticker)
    if not symbol or symbol != jk_ticker(ticker):
        raise RuntimeError(MissingReason.ISSUER_MISMATCH.value)
    rows: list[dict[str, Any]] = []
    provider = "YAHOO_STRUCTURED_VIA_ZAPI"
    family = "YAHOO_FINANCIALS_VIA_ZAPI"
    latest_income = _latest_item(statements.get("income", {}))
    latest_balance = _latest_item(statements.get("balance", {}))
    latest_cash = _latest_item(statements.get("cashflow", {}))
    period_end = _iso_date(latest_income.get("date") or latest_balance.get("date") or latest_cash.get("date"))
    payload_hash = _hash({"ticker": symbol, "summary": summary, "statements": statements})
    for key, metric in (("revenueGrowthPercent", "revenue_growth_pct"), ("returnOnEquityPercent", "roe_pct"), ("profitMarginPercent", "net_margin_pct"), ("operatingMarginPercent", "operating_margin_pct"), ("grossMarginPercent", "gross_margin_pct")):
        _add(rows, ticker=symbol, provider=provider, family=family, metric=metric, value=summary.get(key), unit="PERCENT", observed_at=observed_at, period_end=period_end, payload_hash=payload_hash)
    for key, metric in (("revenue", "revenue"), ("totalCash", "cash"), ("totalDebt", "total_debt"), ("marketCap", "market_cap")):
        _add(rows, ticker=symbol, provider=provider, family=family, metric=metric, value=summary.get(key), unit="CURRENCY_NATIVE", observed_at=observed_at, period_end=period_end, payload_hash=payload_hash)
    income_aliases = {
        "revenue": ("revenue", "totalRevenue"), "net_income": ("netIncome", "netIncomeCommonStockholders"),
        "gross_profit": ("grossProfit",), "operating_income": ("operatingIncome",), "ebitda": ("ebitda",),
    }
    balance_aliases = {
        "total_assets": ("totalAssets", "assets"), "total_liabilities": ("totalLiabilities", "liabilities", "totalLiabilitiesNetMinorityInterest"),
        "equity": ("stockholdersEquity", "totalEquity", "totalEquityGrossMinorityInterest"), "total_debt": ("totalDebt",),
        "cash": ("cashCashEquivalentsAndShortTermInvestments", "cashAndCashEquivalents", "cash"),
        "current_assets": ("currentAssets", "totalCurrentAssets"), "current_liabilities": ("currentLiabilities", "totalCurrentLiabilities"),
    }
    cash_aliases = {
        "operating_cash_flow": ("operatingCashFlow", "cashFlowFromContinuingOperatingActivities", "operating"),
        "capex": ("capitalExpenditure", "capitalExpenditures", "capex"), "free_cash_flow": ("freeCashFlow",),
    }
    for metric, keys in income_aliases.items():
        _add(rows, ticker=symbol, provider=provider, family=family, metric=metric, value=_value(latest_income, *keys), unit="CURRENCY_NATIVE", observed_at=observed_at, period_end=period_end, payload_hash=payload_hash)
    for metric, keys in balance_aliases.items():
        _add(rows, ticker=symbol, provider=provider, family=family, metric=metric, value=_value(latest_balance, *keys), unit="CURRENCY_NATIVE", observed_at=observed_at, period_end=period_end, payload_hash=payload_hash)
    for metric, keys in cash_aliases.items():
        _add(rows, ticker=symbol, provider=provider, family=family, metric=metric, value=_value(latest_cash, *keys), unit="CURRENCY_NATIVE", observed_at=observed_at, period_end=period_end, payload_hash=payload_hash)

    values = {row["metric_name"]: float(row["metric_value"]) for row in rows}
    equity, debt, cash = values.get("equity"), values.get("total_debt"), values.get("cash")
    assets, liabilities = values.get("total_assets"), values.get("total_liabilities")
    ca, cl = values.get("current_assets"), values.get("current_liabilities")
    ni, rev, ocf = values.get("net_income"), values.get("revenue"), values.get("operating_cash_flow")
    derived = {
        "debt_equity": (debt / equity if debt is not None and equity not in (None, 0) else None, "RATIO"),
        "total_liabilities_to_equity": (liabilities / equity if liabilities is not None and equity not in (None, 0) else None, "RATIO"),
        "cash_to_debt_ratio": (cash / debt if cash is not None and debt not in (None, 0) else None, "RATIO"),
        "current_ratio": (ca / cl if ca is not None and cl not in (None, 0) else None, "RATIO"),
        "ocf_conversion_ratio": (ocf / ni if ocf is not None and ni not in (None, 0) else None, "RATIO"),
        "net_margin_pct": (100.0 * ni / rev if ni is not None and rev not in (None, 0) else None, "PERCENT"),
        "roa_pct": (100.0 * ni / assets if ni is not None and assets not in (None, 0) else None, "PERCENT"),
    }
    for metric, (value, unit) in derived.items():
        if metric not in values and value is not None:
            _add(rows, ticker=symbol, provider=provider, family=family, metric=metric, value=value, unit=unit, observed_at=observed_at, period_end=period_end, payload_hash=payload_hash)

    income_items = statements.get("income", {}).get("items") if isinstance(statements.get("income", {}).get("items"), list) else []
    if period_end and income_items:
        prior_rev = _same_quarter_prior([
            {"metric_name": "revenue", "period_end": item.get("date"), "metric_value": _value(item, "revenue", "totalRevenue")}
            for item in income_items if isinstance(item, Mapping)
        ], "revenue", period_end)
        prior_ni = _same_quarter_prior([
            {"metric_name": "net_income", "period_end": item.get("date"), "metric_value": _value(item, "netIncome", "netIncomeCommonStockholders")}
            for item in income_items if isinstance(item, Mapping)
        ], "net_income", period_end)
        rev_growth = _growth_pct(values.get("revenue"), prior_rev)
        earn_growth = _growth_pct(values.get("net_income"), prior_ni, earnings=True)
        if rev_growth is not None and "revenue_growth_pct" not in values:
            _add(rows, ticker=symbol, provider=provider, family=family, metric="revenue_growth_pct", value=rev_growth, unit="PERCENT", observed_at=observed_at, period_end=period_end, payload_hash=payload_hash)
        if earn_growth is not None:
            _add(rows, ticker=symbol, provider=provider, family=family, metric="earnings_growth_pct", value=earn_growth, unit="PERCENT", observed_at=observed_at, period_end=period_end, payload_hash=payload_hash)
    return rows


class SharedFundamentalRuntime:
    def __init__(self, client_id: str, *, config: HubConfig | None = None, backend: SupabaseEvidenceBackend | None = None,
                 api_key: str | None = None, session: requests.Session | None = None):
        self.config = config or HubConfig.from_environment(client_id=client_id)
        self.backend = backend or SupabaseEvidenceBackend(self.config)
        self.api_key = _clean(api_key) if api_key is not None else self._secret("ZAPI_KEY")
        self.session = session or requests.Session()

    @staticmethod
    def _secret(name: str) -> str:
        import os
        value = _clean(os.getenv(name, ""))
        if value:
            return value
        try:
            import streamlit as st
            return _clean(st.secrets.get(name, ""))
        except Exception:
            return ""

    @property
    def ready(self) -> bool:
        return bool(self.config.ready)

    def _read_chunk(self, tickers: Sequence[str], *, provider: str | None = None) -> list[dict[str, Any]]:
        if not tickers or not self.ready:
            return []
        quoted = ",".join(f'"{bare_ticker(ticker).replace(chr(34), "")}"' for ticker in tickers if bare_ticker(ticker))
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            params: dict[str, Any] = {
                "select": "provider,ticker,period_end,statement_date,metric_name,metric_value,metric_unit,source_families,official_verified,source_record_hash,lineage_state,observed_at,validation_state,fetched_at",
                "ticker": f"in.({quoted})",
                "validation_state": "eq.VALID",
                "limit": 1000,
                "offset": offset,
            }
            if provider:
                params["provider"] = f"eq.{provider}"
            payload = self.backend._request("GET", TABLE, params=params)  # shared internal PostgREST client
            page = [dict(row) for row in payload] if isinstance(payload, list) else []
            rows.extend(page)
            if len(page) < 1000:
                break
            offset += len(page)
        return rows

    def read_bundle(self, tickers: Iterable[str]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        symbols = list(dict.fromkeys(bare_ticker(ticker) for ticker in tickers if bare_ticker(ticker)))
        if not symbols or not self.ready:
            return {}, {"state": MissingReason.ENVIRONMENT_BLOCKED.value, "rows": 0, "tickers": 0}
        rows: list[dict[str, Any]] = []
        try:
            for start in range(0, len(symbols), 80):
                rows.extend(self._read_chunk(symbols[start:start + 80]))
        except Exception as exc:
            return {}, {"state": normalize_failure_reason(exc), "rows": len(rows), "tickers": 0}
        bundle = canonicalize_metric_rows(rows)
        return bundle, {"state": "SHARED_HUB_LOADED", "rows": len(rows), "tickers": len(bundle)}

    def _zapi_get(self, url: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self.api_key:
            raise RuntimeError(MissingReason.ENVIRONMENT_BLOCKED.value)
        try:
            response = self.session.get(url, params=dict(params), headers={"x-api-key": self.api_key, "Accept": "application/json"}, timeout=REQUEST_TIMEOUT_SECONDS)
        except requests.Timeout as exc:
            raise RuntimeError(MissingReason.TIMEOUT.value) from exc
        except requests.ConnectionError as exc:
            raise RuntimeError(MissingReason.CONNECTION_ERROR.value) from exc
        if response.status_code in {401, 403, 404, 429}:
            raise RuntimeError(f"HTTP_{response.status_code}")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise RuntimeError(MissingReason.PARSE_FAILURE.value)
        return payload

    def _persist(self, rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        records = [dict(row) for row in rows]
        if not records:
            return []
        output: list[dict[str, Any]] = []
        for start in range(0, len(records), 250):
            batch = records[start:start + 250]
            written = self.backend.upsert_rows(TABLE, batch, conflict=("provider", "ticker", "metric_name", "source_record_hash"))
            if len(written) != len(batch):
                raise RuntimeError(MissingReason.PERSIST_FAILURE.value)
            output.extend(written)
        return output

    def bridge_operational_snapshots(self, *, limit: int = 5000) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not self.ready:
            return [], {"state": MissingReason.ENVIRONMENT_BLOCKED.value, "rows": 0}
        source = self.backend.read_rows(
            "latest_fundamental_snapshots", {},
            select=("ticker,period_end,statement_date,revenue_growth,earnings_growth,roe,roa,roic_proxy,net_margin,operating_margin,"
                    "operating_cash_flow,free_cash_flow,cash_conversion_ttm,debt_equity,net_debt_ebitda,interest_coverage,market_cap,"
                    "fundamental_source_families,fundamental_coverage,fundamental_fetched_at,as_of,updated_at,content_hash"),
            limit=max(1, min(int(limit), 50000)),
        )
        rows = normalize_operational_snapshot_rows(source)
        written = self._persist(rows) if rows else []
        return written, {"state": "BRIDGED_CANONICAL_V2" if rows else MissingReason.EMPTY_RESPONSE.value, "source_rows": len(source), "rows": len(written), "tickers": len({row["ticker"] for row in rows})}

    def refresh_pluang(self, ticker: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        code = bare_ticker(ticker)
        observed_at = datetime.now(timezone.utc).isoformat()
        resolve = self._zapi_get(PLUANG_RESOLVE_URL, {"code": code})
        stock_id = resolve.get("stockId")
        if not stock_id:
            raise RuntimeError(MissingReason.NO_MATCH.value)
        params = {"code": code, "stockId": stock_id}
        fundamentals = self._zapi_get(PLUANG_FUNDAMENTALS_URL, params)
        financials = self._zapi_get(PLUANG_FINANCIALS_URL, {**params, "period": "quarterly"})
        rows = normalize_pluang_payloads(code, fundamentals, financials, observed_at=observed_at)
        written = self._persist(rows) if rows else []
        return written, {"state": "REFRESHED" if written else MissingReason.EMPTY_RESPONSE.value, "provider": "PLUANG", "api_calls": 3, "rows": len(written), "ticker": code}

    def refresh_yahoo(self, ticker: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        symbol = jk_ticker(ticker)
        observed_at = datetime.now(timezone.utc).isoformat()
        summary = self._zapi_get(YAHOO_SUMMARY_URL, {"symbol": symbol})
        statements: dict[str, Mapping[str, Any]] = {}
        for family in ("income", "balance", "cashflow"):
            statements[family] = self._zapi_get(YAHOO_FINANCIALS_URL, {"symbol": symbol, "statement": family, "period": "quarterly"})
        rows = normalize_yahoo_payloads(symbol, summary, statements, observed_at=observed_at)
        written = self._persist(rows) if rows else []
        return written, {"state": "REFRESHED" if written else MissingReason.EMPTY_RESPONSE.value, "provider": "YAHOO_ZAPI", "api_calls": 4, "rows": len(written), "ticker": bare_ticker(symbol)}

    def refresh_structured(self, ticker: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        attempts: list[dict[str, Any]] = []
        for provider, fn in (("PLUANG", self.refresh_pluang), ("YAHOO_ZAPI", self.refresh_yahoo)):
            try:
                rows, meta = fn(ticker)
                attempts.append({"provider": provider, "state": meta.get("state"), "rows": len(rows)})
                if rows:
                    return rows, {**meta, "attempts": attempts}
            except Exception as exc:
                attempts.append({"provider": provider, "state": normalize_failure_reason(exc), "detail": str(exc)[:120]})
        return [], {"state": "STRUCTURED_PROVIDERS_EXHAUSTED", "ticker": bare_ticker(ticker), "rows": 0, "attempts": attempts}

    def publish_metrics(self, ticker: str, metrics: Mapping[str, Any], *, provider: str, source_families: str,
                        observed_at: Any, period_end: Any = None, units: Mapping[str, str] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not self.ready:
            return [], {"state": MissingReason.ENVIRONMENT_BLOCKED.value, "rows": 0}
        code = bare_ticker(ticker)
        stamp = _iso_stamp(observed_at) or datetime.now(timezone.utc).isoformat()
        period = _iso_date(period_end)
        finite_metrics = {str(name): float(value) for name, value in metrics.items() if (value := _finite(value)) is not None}
        if not code or not finite_metrics or not _clean(source_families):
            return [], {"state": MissingReason.EMPTY_RESPONSE.value, "rows": 0}
        record_hash = _hash({"ticker": code, "provider": provider, "period_end": period, "metrics": finite_metrics})
        unit_map = dict(units or {})
        rows = [_metric_row(
            ticker=code, provider=_clean(provider).upper(), source_families=_clean(source_families), metric_name=name,
            metric_value=value, metric_unit=unit_map.get(name, "NORMALIZED"), observed_at=stamp,
            period_end=period, payload_hash=record_hash,
        ) for name, value in finite_metrics.items()]
        written = self._persist(rows)
        return written, {"state": "PUBLISHED", "rows": len(written), "ticker": code, "provider": _clean(provider).upper()}


__all__ = [
    "PLUANG_FINANCIALS_URL", "PLUANG_FUNDAMENTALS_URL", "PLUANG_RESOLVE_URL",
    "RUNTIME_VERSION", "SharedFundamentalRuntime", "TABLE", "YAHOO_FINANCIALS_URL",
    "YAHOO_SUMMARY_URL", "bare_ticker", "canonicalize_metric_rows", "jk_ticker",
    "normalize_operational_snapshot_rows", "normalize_pluang_payloads", "normalize_yahoo_payloads",
]
