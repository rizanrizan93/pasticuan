from __future__ import annotations

"""Scanner-neutral runtime bridge for shared fundamental facts.

Facts only: no scanner score, rank, recommendation, gate, entry, stop, target,
or Future Fundamental conclusion is stored here.
"""

from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
import re
from typing import Any, Iterable, Mapping, Sequence

import requests

from shared_evidence_hub import HubConfig, MissingReason, SupabaseEvidenceBackend, normalize_failure_reason

TABLE = "evidence_fundamental_metrics"
RUNTIME_VERSION = "1.1.1-phase5.6-runtime"
PLUANG_RESOLVE_URL = "https://api.zpi.web.id/v1/finance:pluang/resolve"
PLUANG_FUNDAMENTALS_URL = "https://api.zpi.web.id/v1/finance:pluang/fundamentals"
PLUANG_FINANCIALS_URL = "https://api.zpi.web.id/v1/finance:pluang/financials"
YAHOO_SUMMARY_URL = "https://api.zpi.web.id/v1/finance:yahoo-finance/summary"
YAHOO_FINANCIALS_URL = "https://api.zpi.web.id/v1/finance:yahoo-finance/financials"
REQUEST_TIMEOUT_SECONDS = 18

_RATIO_TO_PERCENT = {
    "revenue_growth": "revenue_growth_pct",
    "earnings_growth": "earnings_growth_pct",
    "roe": "roe_pct",
    "roa": "roa_pct",
    "roic_proxy": "roic_proxy_pct",
    "net_margin": "net_margin_pct",
    "operating_margin": "operating_margin_pct",
}
_DIRECT_FIELDS = {
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
    return datetime.fromisoformat(text).timestamp() if text else 0.0


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
    quarter = int(match.group(1))
    year_text = match.group(2)
    year = int(year_text) if len(year_text) == 4 else 2000 + int(year_text)
    month = quarter * 3
    day = 31 if month in {3, 12} else 30
    return date(year, month, day).isoformat()


def normalize_operational_snapshot_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
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
        record_hash = _clean(source.get("content_hash")) or _hash({"ticker": ticker, "period_end": period_end, "families": families, "source": source})
        for field, metric in _RATIO_TO_PERCENT.items():
            value = _finite(source.get(field))
            if value is None:
                continue
            output.append(_row(
                provider="OPERATIONAL_FUNDAMENTAL_BRIDGE", ticker=ticker, period_end=period_end,
                statement_date=statement_date, metric_name=metric, metric_value=value * 100.0,
                metric_unit="PERCENT", source_families=families, official_verified=False,
                source_record_hash=record_hash,
                lineage_state="BRIDGED_AGGREGATED_OPERATIONAL_METRIC_PERCENT_CANONICAL_V2",
                observed_at=observed_at, fetched_at=fetched_at,
            ))
        for field, (metric, unit) in _DIRECT_FIELDS.items():
            value = _finite(source.get(field))
            if value is None:
                continue
            output.append(_row(
                provider="OPERATIONAL_FUNDAMENTAL_BRIDGE", ticker=ticker, period_end=period_end,
                statement_date=statement_date, metric_name=metric, metric_value=value,
                metric_unit=unit, source_families=families, official_verified=False,
                source_record_hash=record_hash,
                lineage_state="BRIDGED_AGGREGATED_OPERATIONAL_METRIC_PERCENT_CANONICAL_V2",
                observed_at=observed_at, fetched_at=fetched_at,
            ))
    return output


def _row(*, provider: str, ticker: str, period_end: str | None, statement_date: str | None,
         metric_name: str, metric_value: float, metric_unit: str, source_families: str,
         official_verified: bool, source_record_hash: str, lineage_state: str,
         observed_at: str, fetched_at: str | None = None) -> dict[str, Any]:
    return {
        "provider": provider, "ticker": bare_ticker(ticker), "period_end": period_end,
        "statement_date": statement_date, "metric_name": metric_name,
        "metric_value": metric_value, "metric_unit": metric_unit,
        "source_families": source_families, "official_verified": official_verified,
        "source_record_hash": source_record_hash, "lineage_state": lineage_state,
        "observed_at": observed_at, "validation_state": "VALID",
        "fetched_at": fetched_at or datetime.now(timezone.utc).isoformat(),
    }


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
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in rows:
        row = dict(raw)
        ticker = bare_ticker(row.get("ticker"))
        metric = _clean(row.get("metric_name")).lower()
        value = _finite(row.get("metric_value"))
        if not ticker or not metric or value is None or _clean(row.get("validation_state")).upper() != "VALID":
            continue
        if row.get("provider") == "OPERATIONAL_FUNDAMENTAL_BRIDGE" and metric in _PERCENT_METRICS and "PERCENT_CANONICAL_V2" not in _clean(row.get("lineage_state")):
            value *= 100.0
        row["ticker"] = ticker
        row["metric_name"] = metric
        row["metric_value"] = value
        grouped.setdefault(ticker, []).append(row)

    output: dict[str, dict[str, Any]] = {}
    for ticker, items in grouped.items():
        proxy_rows = [r for r in items if not bool(r.get("official_verified"))]
        official_rows = [r for r in items if bool(r.get("official_verified")) and _iso_date(r.get("period_end"))]
        proxy: dict[str, float] = {}
        proxy_meta: dict[str, dict[str, Any]] = {}
        for row in proxy_rows:
            metric = row["metric_name"]
            previous = proxy_meta.get(metric)
            new_key = (_stamp_ord(row.get("observed_at")), _date_ord(row.get("period_end")), _clean(row.get("provider")))
            old_key = (_stamp_ord(previous.get("observed_at")), _date_ord(previous.get("period_end")), _clean(previous.get("provider"))) if previous else (-1.0, -1, "")
            if new_key >= old_key:
                proxy[metric] = float(row["metric_value"])
                proxy_meta[metric] = row

        official_period = max((_iso_date(r.get("period_end")) for r in official_rows), default=None)
        current_official = [r for r in official_rows if _iso_date(r.get("period_end")) == official_period] if official_period else []
        official: dict[str, float] = {}
        for row in sorted(current_official, key=lambda r: _stamp_ord(r.get("observed_at"))):
            official[row["metric_name"]] = float(row["metric_value"])
        if official_period:
            revenue = official.get("revenue")
            income = official.get("net_income")
            equity = official.get("equity")
            debt = official.get("total_debt")
            cash = official.get("cash")
            liabilities = official.get("total_liabilities")
            ocf = official.get("operating_cash_flow")
            if revenue not in (None, 0) and income is not None:
                official.setdefault("net_margin_pct", 100.0 * income / revenue)
            if equity not in (None, 0) and liabilities is not None:
                official.setdefault("total_liabilities_to_equity", liabilities / equity)
            if equity not in (None, 0) and debt is not None:
                official.setdefault("interest_bearing_debt_to_equity", debt / equity)
            if debt not in (None, 0) and cash is not None:
                official.setdefault("cash_to_debt_ratio", cash / debt)
            if equity not in (None, 0) and debt is not None and cash is not None:
                official.setdefault("net_debt_to_equity", (debt - cash) / equity)
            if income not in (None, 0) and ocf is not None:
                official.setdefault("ocf_conversion_ratio", ocf / income)
            rev_growth = _growth_pct(revenue, _same_quarter_prior(official_rows, "revenue", official_period))
            earn_growth = _growth_pct(income, _same_quarter_prior(official_rows, "net_income", official_period), earnings=True)
            if rev_growth is not None:
                official["revenue_growth_yoy_pct"] = rev_growth
            if earn_growth is not None:
                official["earnings_growth_yoy_pct"] = earn_growth
        for metric, value in official.items():
            proxy.setdefault(metric, value)
        proxy_period = max((_iso_date(r.get("period_end")) for r in proxy_rows if _iso_date(r.get("period_end"))), default=None) or official_period
        proxy_observed = max((_iso_stamp(r.get("observed_at")) for r in proxy_rows if _iso_stamp(r.get("observed_at"))), default=None)
        official_observed = max((_iso_stamp(r.get("observed_at")) for r in current_official if _iso_stamp(r.get("observed_at"))), default=None)
        families = sorted({_clean(r.get("source_families")) for r in items if _clean(r.get("source_families"))})
        expected = {"revenue", "net_income", "operating_cash_flow", "cash", "equity", "total_assets", "total_liabilities", "current_ratio", "interest_bearing_debt_to_equity", "cash_to_debt_ratio"}
        official_coverage = 100.0 * sum(name in official for name in expected) / len(expected) if official else 0.0
        output[ticker] = {
            "ticker": ticker, "proxy_metrics": proxy, "proxy_period_end": proxy_period,
            "proxy_observed_at": proxy_observed or official_observed,
            "official_metrics": official, "official_period_end": official_period,
            "official_observed_at": official_observed,
            "official_coverage_pct": round(official_coverage, 1),
            "source_families": families, "runtime_version": RUNTIME_VERSION,
        }
    return output


def _latest_item(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    items = payload.get("items") if isinstance(payload.get("items"), list) else []
    rows = [item for item in items if isinstance(item, Mapping)]
    rows.sort(key=lambda item: _date_ord(item.get("date")), reverse=True)
    return rows[0] if rows else {}


def _value(item: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        number = _scaled_number(item.get(key))
        if number is not None:
            return number
    return None


def _provider_row(rows: list[dict[str, Any]], *, ticker: str, provider: str, family: str,
                  metric: str, value: Any, unit: str, observed_at: str,
                  period_end: str | None, payload_hash: str) -> None:
    number = _scaled_number(value)
    if number is None:
        return
    rows.append(_row(
        provider=provider, ticker=ticker, period_end=period_end, statement_date=period_end,
        metric_name=metric, metric_value=number, metric_unit=unit, source_families=family,
        official_verified=False, source_record_hash=payload_hash,
        lineage_state="STRUCTURED_PROVIDER_OBSERVED_NOT_IDX_OFFICIAL", observed_at=observed_at,
    ))


def normalize_pluang_payloads(ticker: str, fundamentals: Mapping[str, Any], financials: Mapping[str, Any], *, observed_at: str) -> list[dict[str, Any]]:
    code = bare_ticker(fundamentals.get("code") or financials.get("code"))
    if not code or code != bare_ticker(ticker):
        raise RuntimeError(MissingReason.ISSUER_MISMATCH.value)
    rows: list[dict[str, Any]] = []
    fh = _hash({"ticker": code, "payload": fundamentals})
    sh = _hash({"ticker": code, "payload": financials})
    ratios = fundamentals.get("ratios") if isinstance(fundamentals.get("ratios"), Mapping) else {}
    profitability = ratios.get("profitability") if isinstance(ratios.get("profitability"), Mapping) else {}
    solvency = ratios.get("solvency") if isinstance(ratios.get("solvency"), Mapping) else {}
    overview = fundamentals.get("overview") if isinstance(fundamentals.get("overview"), Mapping) else {}
    earnings = fundamentals.get("earnings") if isinstance(fundamentals.get("earnings"), list) else []
    period = next((_quarter_end(item.get("quarter")) for item in reversed(earnings) if isinstance(item, Mapping) and _quarter_end(item.get("quarter"))), None)
    for key, metric in (("roe", "roe_pct"), ("roa", "roa_pct"), ("npm", "net_margin_pct"), ("opm", "operating_margin_pct"), ("gpm", "gross_margin_pct")):
        _provider_row(rows, ticker=code, provider="PLUANG_STRUCTURED_VIA_ZAPI", family="PLUANG_FUNDAMENTALS_VIA_ZAPI", metric=metric, value=profitability.get(key), unit="PERCENT", observed_at=observed_at, period_end=period, payload_hash=fh)
    _provider_row(rows, ticker=code, provider="PLUANG_STRUCTURED_VIA_ZAPI", family="PLUANG_FUNDAMENTALS_VIA_ZAPI", metric="current_ratio", value=solvency.get("cr"), unit="RATIO", observed_at=observed_at, period_end=period, payload_hash=fh)
    de = _scaled_number(solvency.get("de"))
    if de is not None:
        _provider_row(rows, ticker=code, provider="PLUANG_STRUCTURED_VIA_ZAPI", family="PLUANG_FUNDAMENTALS_VIA_ZAPI", metric="debt_equity", value=de / 100.0, unit="RATIO", observed_at=observed_at, period_end=period, payload_hash=fh)
    for key, metric, unit in (("eps", "eps", "CURRENCY_PER_SHARE"), ("bvps", "book_value_per_share", "CURRENCY_PER_SHARE"), ("revenue", "revenue", "CURRENCY_NATIVE"), ("net_income", "net_income", "CURRENCY_NATIVE"), ("gross_profit", "gross_profit", "CURRENCY_NATIVE")):
        _provider_row(rows, ticker=code, provider="PLUANG_STRUCTURED_VIA_ZAPI", family="PLUANG_FUNDAMENTALS_VIA_ZAPI", metric=metric, value=overview.get(key), unit=unit, observed_at=observed_at, period_end=period, payload_hash=fh)
    quarterly = financials.get("quarterly") if isinstance(financials.get("quarterly"), Mapping) else {}
    blocks = {
        "incomeStatement": {"revenue": ("revenue", "CURRENCY_NATIVE"), "netProfitLoss": ("net_income", "CURRENCY_NATIVE")},
        "balanceSheet": {"assets": ("total_assets", "CURRENCY_NATIVE"), "liabilities": ("total_liabilities", "CURRENCY_NATIVE")},
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
        latest_period = _quarter_end(latest.get("timeframe"))
        for key, (metric, unit) in mapping.items():
            _provider_row(rows, ticker=code, provider="PLUANG_STRUCTURED_VIA_ZAPI", family="PLUANG_FINANCIALS_VIA_ZAPI", metric=metric, value=latest.get(key), unit=unit, observed_at=observed_at, period_end=latest_period, payload_hash=sh)
    return rows


def normalize_yahoo_payloads(ticker: str, summary: Mapping[str, Any], statements: Mapping[str, Mapping[str, Any]], *, observed_at: str) -> list[dict[str, Any]]:
    symbol = jk_ticker(summary.get("symbol") or ticker)
    if not symbol or symbol != jk_ticker(ticker):
        raise RuntimeError(MissingReason.ISSUER_MISMATCH.value)
    rows: list[dict[str, Any]] = []
    provider, family = "YAHOO_STRUCTURED_VIA_ZAPI", "YAHOO_FINANCIALS_VIA_ZAPI"
    income = _latest_item(statements.get("income", {}))
    balance = _latest_item(statements.get("balance", {}))
    cashflow = _latest_item(statements.get("cashflow", {}))
    period = _iso_date(income.get("date") or balance.get("date") or cashflow.get("date"))
    ph = _hash({"ticker": symbol, "summary": summary, "statements": statements})
    for key, metric in (("revenueGrowthPercent", "revenue_growth_pct"), ("returnOnEquityPercent", "roe_pct"), ("profitMarginPercent", "net_margin_pct"), ("operatingMarginPercent", "operating_margin_pct"), ("grossMarginPercent", "gross_margin_pct")):
        _provider_row(rows, ticker=symbol, provider=provider, family=family, metric=metric, value=summary.get(key), unit="PERCENT", observed_at=observed_at, period_end=period, payload_hash=ph)
    aliases = {
        "revenue": (income, ("revenue", "totalRevenue")),
        "net_income": (income, ("netIncome", "netIncomeCommonStockholders")),
        "gross_profit": (income, ("grossProfit",)),
        "ebitda": (income, ("ebitda",)),
        "total_assets": (balance, ("totalAssets", "assets")),
        "total_liabilities": (balance, ("totalLiabilities", "liabilities", "totalLiabilitiesNetMinorityInterest")),
        "equity": (balance, ("stockholdersEquity", "totalEquity", "totalEquityGrossMinorityInterest")),
        "total_debt": (balance, ("totalDebt",)),
        "cash": (balance, ("cashCashEquivalentsAndShortTermInvestments", "cashAndCashEquivalents", "cash")),
        "current_assets": (balance, ("currentAssets", "totalCurrentAssets")),
        "current_liabilities": (balance, ("currentLiabilities", "totalCurrentLiabilities")),
        "operating_cash_flow": (cashflow, ("operatingCashFlow", "cashFlowFromContinuingOperatingActivities", "operating")),
        "free_cash_flow": (cashflow, ("freeCashFlow",)),
    }
    for metric, (item, keys) in aliases.items():
        _provider_row(rows, ticker=symbol, provider=provider, family=family, metric=metric, value=_value(item, *keys), unit="CURRENCY_NATIVE", observed_at=observed_at, period_end=period, payload_hash=ph)
    values = {r["metric_name"]: float(r["metric_value"]) for r in rows}
    equity, debt, cash = values.get("equity"), values.get("total_debt"), values.get("cash")
    assets, liabilities = values.get("total_assets"), values.get("total_liabilities")
    ca, cl = values.get("current_assets"), values.get("current_liabilities")
    ni, revenue, ocf = values.get("net_income"), values.get("revenue"), values.get("operating_cash_flow")
    derived = {
        "debt_equity": debt / equity if debt is not None and equity not in (None, 0) else None,
        "total_liabilities_to_equity": liabilities / equity if liabilities is not None and equity not in (None, 0) else None,
        "cash_to_debt_ratio": cash / debt if cash is not None and debt not in (None, 0) else None,
        "current_ratio": ca / cl if ca is not None and cl not in (None, 0) else None,
        "ocf_conversion_ratio": ocf / ni if ocf is not None and ni not in (None, 0) else None,
        "net_margin_pct": 100.0 * ni / revenue if ni is not None and revenue not in (None, 0) else None,
        "roa_pct": 100.0 * ni / assets if ni is not None and assets not in (None, 0) else None,
    }
    for metric, value in derived.items():
        if metric in values or value is None:
            continue
        unit = "PERCENT" if metric.endswith("_pct") else "RATIO"
        _provider_row(rows, ticker=symbol, provider=provider, family=family, metric=metric, value=value, unit=unit, observed_at=observed_at, period_end=period, payload_hash=ph)
    income_items = statements.get("income", {}).get("items") if isinstance(statements.get("income", {}).get("items"), list) else []
    if period and income_items:
        prior_revenue = _same_quarter_prior([{"metric_name": "revenue", "period_end": item.get("date"), "metric_value": _value(item, "revenue", "totalRevenue")} for item in income_items if isinstance(item, Mapping)], "revenue", period)
        prior_income = _same_quarter_prior([{"metric_name": "net_income", "period_end": item.get("date"), "metric_value": _value(item, "netIncome", "netIncomeCommonStockholders")} for item in income_items if isinstance(item, Mapping)], "net_income", period)
        rev_growth = _growth_pct(revenue, prior_revenue)
        earn_growth = _growth_pct(ni, prior_income, earnings=True)
        if rev_growth is not None and "revenue_growth_pct" not in values:
            _provider_row(rows, ticker=symbol, provider=provider, family=family, metric="revenue_growth_pct", value=rev_growth, unit="PERCENT", observed_at=observed_at, period_end=period, payload_hash=ph)
        if earn_growth is not None:
            _provider_row(rows, ticker=symbol, provider=provider, family=family, metric="earnings_growth_pct", value=earn_growth, unit="PERCENT", observed_at=observed_at, period_end=period, payload_hash=ph)
    return rows


class SharedFundamentalRuntime:
    def __init__(self, client_id: str, *, config: HubConfig | None = None, backend: SupabaseEvidenceBackend | None = None, api_key: str | None = None, session: requests.Session | None = None):
        self.config = config or HubConfig.from_environment(client_id=client_id)
        self.backend = backend or SupabaseEvidenceBackend(self.config)
        self.api_key = _clean(api_key) if api_key is not None else self._secret("ZAPI_KEY")
        self.session = session or requests.Session()

    @staticmethod
    def _secret(name: str) -> str:
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

    def _read_chunk(self, tickers: Sequence[str]) -> list[dict[str, Any]]:
        quoted = ",".join(f'"{bare_ticker(t)}"' for t in tickers if bare_ticker(t))
        if not quoted or not self.ready:
            return []
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            params = {
                "select": "provider,ticker,period_end,statement_date,metric_name,metric_value,metric_unit,source_families,official_verified,source_record_hash,lineage_state,observed_at,validation_state,fetched_at",
                "ticker": f"in.({quoted})", "validation_state": "eq.VALID", "limit": 1000, "offset": offset,
            }
            payload = self.backend._request("GET", TABLE, params=params)
            page = [dict(row) for row in payload] if isinstance(payload, list) else []
            rows.extend(page)
            if len(page) < 1000:
                break
            offset += len(page)
        return rows

    def read_bundle(self, tickers: Iterable[str]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        symbols = list(dict.fromkeys(bare_ticker(t) for t in tickers if bare_ticker(t)))
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
        output: list[dict[str, Any]] = []
        for start in range(0, len(records), 250):
            batch = records[start:start + 250]
            written = self.backend.upsert_rows(TABLE, batch, conflict=("provider", "ticker", "metric_name", "source_record_hash"))
            if len(written) != len(batch):
                raise RuntimeError(MissingReason.PERSIST_FAILURE.value)
            output.extend(dict(row) for row in written)
        return output

    def bridge_operational_snapshots(self, *, limit: int = 5000) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not self.ready:
            return [], {"state": MissingReason.ENVIRONMENT_BLOCKED.value, "rows": 0}
        source = self.backend.read_rows("latest_fundamental_snapshots", {}, select=("ticker,period_end,statement_date,revenue_growth,earnings_growth,roe,roa,roic_proxy,net_margin,operating_margin,operating_cash_flow,free_cash_flow,cash_conversion_ttm,debt_equity,net_debt_ebitda,interest_coverage,market_cap,fundamental_source_families,fundamental_coverage,fundamental_fetched_at,as_of,updated_at,content_hash"), limit=max(1, min(int(limit), 50000)))
        rows = normalize_operational_snapshot_rows(source)
        written = self._persist(rows) if rows else []
        return written, {"state": "BRIDGED_CANONICAL_V2" if rows else MissingReason.EMPTY_RESPONSE.value, "source_rows": len(source), "rows": len(written), "tickers": len({r["ticker"] for r in rows})}

    def refresh_pluang(self, ticker: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        code = bare_ticker(ticker)
        observed = datetime.now(timezone.utc).isoformat()
        resolved = self._zapi_get(PLUANG_RESOLVE_URL, {"code": code})
        stock_id = resolved.get("stockId")
        if not stock_id:
            raise RuntimeError(MissingReason.NO_MATCH.value)
        params = {"code": code, "stockId": stock_id}
        fundamentals = self._zapi_get(PLUANG_FUNDAMENTALS_URL, params)
        financials = self._zapi_get(PLUANG_FINANCIALS_URL, {**params, "period": "quarterly"})
        rows = normalize_pluang_payloads(code, fundamentals, financials, observed_at=observed)
        written = self._persist(rows) if rows else []
        return written, {"state": "REFRESHED" if written else MissingReason.EMPTY_RESPONSE.value, "provider": "PLUANG", "api_calls": 3, "rows": len(written), "ticker": code}

    def refresh_yahoo(self, ticker: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        symbol = jk_ticker(ticker)
        observed = datetime.now(timezone.utc).isoformat()
        summary = self._zapi_get(YAHOO_SUMMARY_URL, {"symbol": symbol})
        statements = {family: self._zapi_get(YAHOO_FINANCIALS_URL, {"symbol": symbol, "statement": family, "period": "quarterly"}) for family in ("income", "balance", "cashflow")}
        rows = normalize_yahoo_payloads(symbol, summary, statements, observed_at=observed)
        written = self._persist(rows) if rows else []
        return written, {"state": "REFRESHED" if written else MissingReason.EMPTY_RESPONSE.value, "provider": "YAHOO_ZAPI", "api_calls": 4, "rows": len(written), "ticker": bare_ticker(symbol)}

    def refresh_structured(self, ticker: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        attempts: list[dict[str, Any]] = []
        for provider, function in (("PLUANG", self.refresh_pluang), ("YAHOO_ZAPI", self.refresh_yahoo)):
            try:
                rows, meta = function(ticker)
                attempts.append({"provider": provider, "state": meta.get("state"), "rows": len(rows)})
                if rows:
                    return rows, {**meta, "attempts": attempts}
            except Exception as exc:
                attempts.append({"provider": provider, "state": normalize_failure_reason(exc), "detail": str(exc)[:120]})
        return [], {"state": "STRUCTURED_PROVIDERS_EXHAUSTED", "ticker": bare_ticker(ticker), "rows": 0, "attempts": attempts}

    def publish_metrics(self, ticker: str, metrics: Mapping[str, Any], *, provider: str, source_families: str, observed_at: Any, period_end: Any = None, units: Mapping[str, str] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not self.ready:
            return [], {"state": MissingReason.ENVIRONMENT_BLOCKED.value, "rows": 0}
        code = bare_ticker(ticker)
        finite_metrics: dict[str, float] = {}
        for name, raw_value in metrics.items():
            number = _finite(raw_value)
            if number is not None:
                finite_metrics[str(name)] = number
        if not code or not finite_metrics or not _clean(source_families):
            return [], {"state": MissingReason.EMPTY_RESPONSE.value, "rows": 0}
        stamp = _iso_stamp(observed_at) or datetime.now(timezone.utc).isoformat()
        period = _iso_date(period_end)
        record_hash = _hash({"ticker": code, "provider": provider, "period_end": period, "metrics": finite_metrics})
        unit_map = dict(units or {})
        rows = [_row(provider=_clean(provider).upper(), ticker=code, period_end=period, statement_date=period, metric_name=name, metric_value=value, metric_unit=unit_map.get(name, "NORMALIZED"), source_families=_clean(source_families), official_verified=False, source_record_hash=record_hash, lineage_state="STRUCTURED_PROVIDER_OBSERVED_NOT_IDX_OFFICIAL", observed_at=stamp) for name, value in finite_metrics.items()]
        written = self._persist(rows)
        return written, {"state": "PUBLISHED", "rows": len(written), "ticker": code, "provider": _clean(provider).upper()}


__all__ = ["PLUANG_FINANCIALS_URL", "PLUANG_FUNDAMENTALS_URL", "PLUANG_RESOLVE_URL", "RUNTIME_VERSION", "SharedFundamentalRuntime", "TABLE", "YAHOO_FINANCIALS_URL", "YAHOO_SUMMARY_URL", "bare_ticker", "canonicalize_metric_rows", "jk_ticker", "normalize_operational_snapshot_rows", "normalize_pluang_payloads", "normalize_yahoo_payloads"]
