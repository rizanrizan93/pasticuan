from __future__ import annotations

"""Phase 5.6 scanner-neutral structured fundamental evidence.

This module has two responsibilities:
1. Promote already-persisted, source-backed operational fundamental metrics into
   the shared hub without re-downloading the original filing.
2. Use structured ZAPI/Pluang fundamentals + financial statements only to fill
   missing fields. XBRL remains the official corroboration layer, not a runtime
   prerequisite.

No scanner score, rank, gate, recommendation, or future-fundamental conclusion
is persisted here.
"""

from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
import re
from typing import Any, Iterable, Mapping

import requests

from shared_evidence_hub import HubConfig, MissingReason, SupabaseEvidenceBackend


TABLE = "evidence_fundamental_metrics"
OPERATIONAL_TABLE = "latest_fundamental_snapshots"
PLUANG_FUNDAMENTALS_URL = "https://api.zpi.web.id/v1/finance:pluang/fundamentals"
PLUANG_FINANCIALS_URL = "https://api.zpi.web.id/v1/finance:pluang/financials"
REQUEST_TIMEOUT_SECONDS = 20
STRUCTURED_TTL_DAYS = 14

OPERATIONAL_METRICS: Mapping[str, tuple[str, str]] = {
    "revenue_growth": ("revenue_growth_pct", "PERCENT"),
    "earnings_growth": ("earnings_growth_pct", "PERCENT"),
    "roe": ("roe_pct", "PERCENT"),
    "roa": ("roa_pct", "PERCENT"),
    "roic_proxy": ("roic_proxy_pct", "PERCENT"),
    "net_margin": ("net_margin_pct", "PERCENT"),
    "operating_margin": ("operating_margin_pct", "PERCENT"),
    "operating_cash_flow": ("operating_cash_flow", "CURRENCY_NATIVE"),
    "free_cash_flow": ("free_cash_flow", "CURRENCY_NATIVE"),
    "cash_conversion_ttm": ("cash_conversion_ttm", "RATIO"),
    "debt_equity": ("debt_equity", "RATIO"),
    "net_debt_ebitda": ("net_debt_ebitda", "RATIO"),
    "interest_coverage": ("interest_coverage", "RATIO"),
    "market_cap": ("market_cap", "CURRENCY_NATIVE"),
    "fundamental_coverage": ("fundamental_coverage_pct", "PERCENT"),
}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _ticker(value: Any) -> str:
    text = _clean(value).upper()
    return text[:-3] if text.endswith(".JK") else text


def _secret(name: str) -> str:
    value = _clean(os.getenv(name, ""))
    if value:
        return value
    try:
        import streamlit as st

        return _clean(st.secrets.get(name, ""))
    except Exception:
        return ""


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _scaled_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return _finite(value)
    text = _clean(value).replace("\u00a0", "").replace(" ", "")
    if not text or text.lower() in {"-", "—", "na", "n/a", "nan", "none"}:
        return None
    text = re.sub(r"^(rp|idr|usd)", "", text, flags=re.IGNORECASE)
    text = text.replace(",", "")
    percent = text.endswith("%")
    multiple = text.lower().endswith("x")
    if percent or multiple:
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


def _hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _quarter_end(timeframe: Any) -> str | None:
    text = _clean(timeframe).replace("’", "'").replace("‘", "'")
    match = re.search(r"Q([1-4]).*?(\d{2,4})", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    quarter = int(match.group(1))
    year_raw = match.group(2)
    year = int(year_raw) if len(year_raw) == 4 else 2000 + int(year_raw)
    month = quarter * 3
    day = 31 if month in {3, 12} else 30
    return date(year, month, day).isoformat()


def normalize_operational_snapshots(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        ticker = _ticker(source.get("ticker"))
        families = _clean(source.get("fundamental_source_families"))
        if not ticker or not families:
            continue
        observed_at = (
            _iso_stamp(source.get("fundamental_fetched_at"))
            or _iso_stamp(source.get("as_of"))
            or _iso_stamp(source.get("updated_at"))
        )
        if not observed_at:
            continue
        period_end = _iso_date(source.get("period_end"))
        statement_date = _iso_date(source.get("statement_date"))
        values = {
            field: _finite(source.get(field))
            for field in OPERATIONAL_METRICS
            if _finite(source.get(field)) is not None
        }
        if not values:
            continue
        record_hash = _clean(source.get("content_hash")) or _hash({
            "ticker": ticker,
            "period_end": period_end,
            "statement_date": statement_date,
            "source_families": families,
            "values": values,
        })
        # This snapshot may blend IDX and public/vendor sources. Aggregate
        # source verification must never be promoted to field-level official
        # verification. Exact operational financial facts are bridged separately.
        official = False
        lineage = "BRIDGED_AGGREGATED_OPERATIONAL_METRIC_NOT_FIELD_OFFICIAL"
        for field, value in values.items():
            metric_name, unit = OPERATIONAL_METRICS[field]
            output.append({
                "provider": "OPERATIONAL_FUNDAMENTAL_BRIDGE",
                "ticker": ticker,
                "period_end": period_end,
                "statement_date": statement_date,
                "metric_name": metric_name,
                "metric_value": value,
                "metric_unit": unit,
                "source_families": families,
                "official_verified": official,
                "source_record_hash": record_hash,
                "lineage_state": lineage,
                "observed_at": observed_at,
                "validation_state": "VALID",
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            })
    return output


def normalize_operational_financial_facts(
    periods: Iterable[Mapping[str, Any]],
    facts: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Promote exact source-lineaged operational financial facts into the hub.

    Unlike the aggregate snapshot bridge, this path can retain field-level
    official verification because financial_facts.source_lineage identifies
    the source family for each metric.
    """
    period_map: dict[str, dict[str, Any]] = {}
    for raw in periods:
        row = dict(raw)
        period_id = _clean(row.get("financial_period_id"))
        ticker = _ticker(row.get("ticker"))
        if not period_id or not ticker or not bool(row.get("is_current")):
            continue
        row["ticker"] = ticker
        period_map[period_id] = row

    output: list[dict[str, Any]] = []
    fetched_at = datetime.now(timezone.utc).isoformat()
    for raw in facts:
        fact = dict(raw)
        period_id = _clean(fact.get("financial_period_id"))
        period = period_map.get(period_id)
        if period is None:
            continue
        ticker = _ticker(fact.get("ticker") or period.get("ticker"))
        metric_code = _clean(fact.get("metric_code")).upper()
        value = _finite(fact.get("normalized_value"))
        if value is None:
            value = _finite(fact.get("reported_value"))
        if not ticker or not metric_code or value is None:
            continue
        lineage = fact.get("source_lineage") if isinstance(fact.get("source_lineage"), Mapping) else {}
        source_family = _clean(period.get("source_family"))
        source_verified = bool(lineage.get("source_verified")) and source_family.startswith("IDX_OFFICIAL_XBRL")
        observed_at = (
            _iso_stamp(fact.get("created_at"))
            or _iso_stamp(period.get("updated_at"))
            or _iso_stamp(period.get("created_at"))
        )
        if not observed_at or not source_family:
            continue
        fact_id = _clean(fact.get("financial_fact_id"))
        source_record_hash = _hash({
            "financial_fact_id": fact_id,
            "financial_period_id": period_id,
            "document_id": _clean(period.get("document_id")),
            "document_hash": _clean(period.get("document_hash")),
            "metric_code": metric_code,
            "source_family": source_family,
            "source_lineage": lineage,
        })
        currency = _clean(fact.get("currency") or period.get("currency"))
        output.append({
            "provider": "OPERATIONAL_FINANCIAL_FACT_BRIDGE",
            "ticker": ticker,
            "period_end": _iso_date(period.get("period_end")),
            "statement_date": _iso_date(period.get("filing_date") or period.get("period_end")),
            "metric_name": metric_code.lower(),
            "metric_value": value,
            "metric_unit": currency or "NORMALIZED_NATIVE_OR_RATIO",
            "source_families": source_family,
            "official_verified": source_verified,
            "source_record_hash": source_record_hash,
            "lineage_state": "OPERATIONAL_FINANCIAL_FACT_EXACT_LINEAGE",
            "observed_at": observed_at,
            "validation_state": "VALID",
            "fetched_at": fetched_at,
        })
    return output


def _pluang_metric_rows(
    ticker: str,
    fundamentals: Mapping[str, Any],
    financials: Mapping[str, Any],
    *,
    observed_at: datetime,
) -> list[dict[str, Any]]:
    code = _ticker(fundamentals.get("code") or financials.get("code"))
    if not code or code != _ticker(ticker):
        raise RuntimeError(MissingReason.ISSUER_MISMATCH.value)
    if _clean(fundamentals.get("source")).lower() not in {"", "pluang"}:
        raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
    if _clean(financials.get("source")).lower() not in {"", "pluang"}:
        raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)

    rows: list[dict[str, Any]] = []
    stamp = observed_at.astimezone(timezone.utc).isoformat()

    def add(
        metric_name: str,
        value: Any,
        unit: str,
        *,
        period_end: str | None,
        family: str,
        source_payload: Mapping[str, Any],
    ) -> None:
        number = _scaled_number(value)
        if number is None:
            return
        record_hash = _hash({
            "ticker": code,
            "period_end": period_end,
            "family": family,
            "payload": source_payload,
        })
        rows.append({
            "provider": "PLUANG_STRUCTURED_VIA_ZAPI",
            "ticker": code,
            "period_end": period_end,
            "statement_date": None,
            "metric_name": metric_name,
            "metric_value": number,
            "metric_unit": unit,
            "source_families": family,
            "official_verified": False,
            "source_record_hash": record_hash,
            "lineage_state": "STRUCTURED_PROVIDER_OBSERVED_NOT_IDX_OFFICIAL",
            "observed_at": stamp,
            "validation_state": "VALID",
            "fetched_at": stamp,
        })

    ratios = fundamentals.get("ratios") if isinstance(fundamentals.get("ratios"), Mapping) else {}
    profitability = ratios.get("profitability") if isinstance(ratios.get("profitability"), Mapping) else {}
    solvency = ratios.get("solvency") if isinstance(ratios.get("solvency"), Mapping) else {}
    overview = fundamentals.get("overview") if isinstance(fundamentals.get("overview"), Mapping) else {}
    earnings = fundamentals.get("earnings") if isinstance(fundamentals.get("earnings"), list) else []
    latest_period = None
    for item in reversed(earnings):
        if isinstance(item, Mapping):
            latest_period = _quarter_end(item.get("quarter"))
            if latest_period:
                break
    for key, metric in (
        ("roe", "roe_pct"), ("roa", "roa_pct"), ("npm", "net_margin_pct"),
        ("opm", "operating_margin_pct"), ("gpm", "gross_margin_pct"),
    ):
        add(metric, profitability.get(key), "PERCENT", period_end=latest_period,
            family="PLUANG_FUNDAMENTALS_VIA_ZAPI", source_payload=fundamentals)
    add("current_ratio", solvency.get("cr"), "RATIO", period_end=latest_period,
        family="PLUANG_FUNDAMENTALS_VIA_ZAPI", source_payload=fundamentals)
    add("debt_equity_pct", solvency.get("de"), "PERCENT", period_end=latest_period,
        family="PLUANG_FUNDAMENTALS_VIA_ZAPI", source_payload=fundamentals)
    for key, metric, unit in (
        ("eps", "eps", "CURRENCY_PER_SHARE"),
        ("bvps", "book_value_per_share", "CURRENCY_PER_SHARE"),
        ("revenue", "revenue", "CURRENCY_NATIVE"),
        ("net_income", "net_income", "CURRENCY_NATIVE"),
        ("gross_profit", "gross_profit", "CURRENCY_NATIVE"),
    ):
        add(metric, overview.get(key), unit, period_end=latest_period,
            family="PLUANG_FUNDAMENTALS_VIA_ZAPI", source_payload=fundamentals)

    quarterly = financials.get("quarterly") if isinstance(financials.get("quarterly"), Mapping) else {}
    blocks = {
        "incomeStatement": {
            "revenue": ("revenue", "CURRENCY_NATIVE"),
            "netProfitLoss": ("net_income", "CURRENCY_NATIVE"),
            "profitMargin": ("net_margin_ratio", "RATIO"),
        },
        "balanceSheet": {
            "assets": ("assets", "CURRENCY_NATIVE"),
            "liabilities": ("liabilities", "CURRENCY_NATIVE"),
            "debtToAsset": ("debt_to_asset", "RATIO"),
        },
        "cashFlow": {
            "operating": ("operating_cash_flow", "CURRENCY_NATIVE"),
            "investing": ("investing_cash_flow", "CURRENCY_NATIVE"),
            "finance": ("financing_cash_flow", "CURRENCY_NATIVE"),
            "netCF": ("net_cash_flow", "CURRENCY_NATIVE"),
        },
    }
    for block_name, mappings in blocks.items():
        block = quarterly.get(block_name) if isinstance(quarterly.get(block_name), Mapping) else {}
        chart = block.get("chart") if isinstance(block.get("chart"), list) else []
        candidates = [item for item in chart if isinstance(item, Mapping) and _quarter_end(item.get("timeframe"))]
        if not candidates:
            continue
        candidates.sort(key=lambda item: _quarter_end(item.get("timeframe")) or "")
        latest = candidates[-1]
        period_end = _quarter_end(latest.get("timeframe"))
        for key, (metric_name, unit) in mappings.items():
            add(metric_name, latest.get(key), unit, period_end=period_end,
                family="PLUANG_FINANCIALS_VIA_ZAPI", source_payload=financials)

    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["metric_name"], row["source_record_hash"], row["source_families"])
        unique[key] = row
    return list(unique.values())


def _upsert_metric_batches(
    backend: SupabaseEvidenceBackend,
    rows: Iterable[Mapping[str, Any]],
    *,
    batch_size: int = 500,
) -> list[dict[str, Any]]:
    records = [dict(row) for row in rows]
    output: list[dict[str, Any]] = []
    size = max(50, min(int(batch_size), 1000))
    for start in range(0, len(records), size):
        batch = records[start:start + size]
        written = backend.upsert_rows(
            TABLE,
            batch,
            conflict=("provider", "ticker", "metric_name", "source_record_hash"),
        )
        if len(written) != len(batch):
            raise RuntimeError(MissingReason.PERSIST_FAILURE.value)
        output.extend(dict(row) for row in written)
    return output


def validate_structured_metrics(rows: Iterable[Mapping[str, Any]]) -> tuple[bool, str]:
    records = [dict(row) for row in rows]
    if not records:
        return False, MissingReason.EMPTY_RESPONSE.value
    for row in records:
        if not _ticker(row.get("ticker")) or not _clean(row.get("metric_name")):
            return False, MissingReason.PARSE_FAILURE.value
        if _finite(row.get("metric_value")) is None:
            return False, MissingReason.PARSE_FAILURE.value
        if not _clean(row.get("source_families")) or not _clean(row.get("source_record_hash")):
            return False, MissingReason.PARSE_FAILURE.value
        if not _iso_stamp(row.get("observed_at")):
            return False, MissingReason.PARSE_FAILURE.value
        if _clean(row.get("validation_state")) not in {"VALID", "STALE"}:
            return False, MissingReason.PARSE_FAILURE.value
    return True, "VALID"


class SharedStructuredFundamentalEvidence:
    def __init__(
        self,
        client_id: str,
        *,
        config: HubConfig | None = None,
        backend: SupabaseEvidenceBackend | None = None,
        api_key: str | None = None,
        session: requests.Session | None = None,
    ):
        self.config = config or HubConfig.from_environment(client_id=client_id)
        self.backend = backend or SupabaseEvidenceBackend(self.config)
        self.api_key = _secret("ZAPI_KEY") if api_key is None else _clean(api_key)
        self.session = session or requests.Session()

    @property
    def ready(self) -> bool:
        return bool(self.config.ready)

    def bridge_operational_financial_facts(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not self.ready:
            return [], {"state": MissingReason.ENVIRONMENT_BLOCKED.value, "rows": 0}
        periods = self.backend.read_rows(
            "financial_periods",
            {"is_current": "true"},
            select=(
                "financial_period_id,ticker,period_end,period_type,filing_date,currency,unit_multiplier,"
                "source_family,source_url,document_id,document_hash,is_current,created_at,updated_at"
            ),
            limit=5000,
        )
        facts = self.backend.read_rows(
            "financial_facts",
            {},
            select=(
                "financial_fact_id,financial_period_id,ticker,metric_code,reported_value,normalized_value,"
                "currency,unit_multiplier,fact_context,source_lineage,created_at"
            ),
            limit=50000,
        )
        rows = normalize_operational_financial_facts(periods, facts)
        valid, reason = validate_structured_metrics(rows)
        if not valid:
            return [], {"state": reason, "rows": 0, "period_rows": len(periods), "fact_rows": len(facts)}
        written = _upsert_metric_batches(self.backend, rows)
        return [dict(row) for row in written], {
            "state": "BRIDGED",
            "period_rows": len(periods),
            "fact_rows": len(facts),
            "rows": len(written),
            "tickers": len({row["ticker"] for row in rows}),
            "official_rows": sum(bool(row.get("official_verified")) for row in rows),
        }

    def bridge_operational(self, *, limit: int = 5000) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not self.ready:
            return [], {"state": MissingReason.ENVIRONMENT_BLOCKED.value, "rows": 0}
        source = self.backend.read_rows(
            OPERATIONAL_TABLE,
            {},
            select=(
                "ticker,period_end,statement_date,revenue_growth,earnings_growth,roe,roa,roic_proxy,"
                "net_margin,operating_margin,operating_cash_flow,free_cash_flow,cash_conversion_ttm,"
                "debt_equity,net_debt_ebitda,interest_coverage,market_cap,fundamental_source_families,"
                "fundamental_coverage,fundamental_fetched_at,as_of,updated_at,content_hash"
            ),
            limit=max(1, min(int(limit), 50000)),
        )
        rows = normalize_operational_snapshots(source)
        valid, reason = validate_structured_metrics(rows)
        if not valid:
            return [], {"state": reason, "rows": 0, "source_rows": len(source)}
        written = self.backend.upsert_rows(
            TABLE, rows, conflict=("provider", "ticker", "metric_name", "source_record_hash")
        )
        return [dict(row) for row in written], {
            "state": "BRIDGED",
            "source_rows": len(source),
            "rows": len(written),
            "tickers": len({row["ticker"] for row in rows}),
        }

    def _zapi_get(self, url: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self.api_key:
            raise RuntimeError(MissingReason.ENVIRONMENT_BLOCKED.value)
        try:
            response = self.session.get(
                url,
                params=dict(params),
                headers={"x-api-key": self.api_key, "Accept": "application/json"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
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

    def refresh_pluang(self, ticker: str, *, observed_at: datetime | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        code = _ticker(ticker)
        if not code or not self.ready:
            return [], {"state": MissingReason.ENVIRONMENT_BLOCKED.value, "api_calls": 0}
        observed = observed_at or datetime.now(timezone.utc)
        fundamentals = self._zapi_get(PLUANG_FUNDAMENTALS_URL, {"code": code})
        financials = self._zapi_get(PLUANG_FINANCIALS_URL, {"code": code, "period": "quarterly"})
        rows = _pluang_metric_rows(code, fundamentals, financials, observed_at=observed)
        valid, reason = validate_structured_metrics(rows)
        if not valid:
            return [], {"state": reason, "api_calls": 2, "rows": 0}
        written = self.backend.upsert_rows(
            TABLE, rows, conflict=("provider", "ticker", "metric_name", "source_record_hash")
        )
        return [dict(row) for row in written], {
            "state": "REFRESHED",
            "ticker": code,
            "api_calls": 2,
            "rows": len(written),
        }


__all__ = [
    "OPERATIONAL_METRICS",
    "OPERATIONAL_TABLE",
    "PLUANG_FINANCIALS_URL",
    "PLUANG_FUNDAMENTALS_URL",
    "SharedStructuredFundamentalEvidence",
    "TABLE",
    "normalize_operational_financial_facts",
    "normalize_operational_snapshots",
    "validate_structured_metrics",
]
