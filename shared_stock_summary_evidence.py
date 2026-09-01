from __future__ import annotations

"""Shared, scanner-neutral ZAPI IDX stock-summary evidence producer.

The producer stores factual daily market rows only.  It has no scanner score,
ranking, recommendation, or gate semantics, and it never substitutes these rows
for a scanner's own price-history truth.
"""

from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from typing import Any, Iterable, Mapping

import requests

from shared_evidence_hub import (
    EvidenceKey,
    HubConfig,
    MissingReason,
    SharedEvidenceCoordinator,
    SupabaseEvidenceBackend,
)


ZAPI_STOCK_SUMMARY_URL = "https://api.zpi.web.id/v1/finance:idx/stock-summary"
PROVIDER = "ZAPI"
EVIDENCE_FAMILY = "STOCK_SUMMARY"
SCOPE = "IDX_ALL"
TABLE = "evidence_market_daily"
BULK_LENGTH = 5000


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _canonical_ticker(value: Any) -> str:
    ticker = _clean(value).upper()
    return ticker[:-3] if ticker.endswith(".JK") else ticker


def _secret(name: str) -> str:
    value = _clean(os.getenv(name, ""))
    if value:
        return value
    try:
        import streamlit as st

        return _clean(st.secrets.get(name, ""))
    except Exception:
        return ""


def _number(value: Any) -> int | float | None:
    if value is None or isinstance(value, bool) or _clean(value) == "":
        return None
    try:
        number = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return int(number) if number.is_integer() else number


def _first(item: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in item:
            return item[name]
    return None


def _day(value: Any) -> date | None:
    text = _clean(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _unwrap(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise RuntimeError(MissingReason.PARSE_FAILURE.value)
    current: Mapping[str, Any] = payload
    for _ in range(3):
        nested = current.get("data")
        if isinstance(nested, Mapping) and any(
            name in nested for name in ("data", "recordsTotal", "recordsFiltered", "total", "date")
        ):
            current = nested
        else:
            break
    return current


def normalize_stock_summary(
    payload: Any,
    *,
    trade_date: date,
    universe: Iterable[str],
    fetched_at: datetime | None = None,
) -> list[dict[str, Any]]:
    """Normalize one bulk response and reject cross-date or duplicate evidence."""

    root = _unwrap(payload)
    raw_rows = root.get("data")
    if not isinstance(raw_rows, list):
        raise RuntimeError(MissingReason.PARSE_FAILURE.value)
    if not raw_rows:
        raise RuntimeError(MissingReason.EMPTY_RESPONSE.value)

    wanted = {_canonical_ticker(value) for value in universe if _canonical_ticker(value)}
    root_date = _day(root.get("date") or root.get("Date"))
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    stamp = (fetched_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()

    for item in raw_rows:
        if not isinstance(item, Mapping):
            continue
        ticker = _canonical_ticker(_first(item, "StockCode", "stockCode", "code", "ticker"))
        if not ticker or (wanted and ticker not in wanted):
            continue
        source_date = _day(_first(item, "Date", "date")) or root_date
        if source_date != trade_date:
            raise RuntimeError(MissingReason.WRONG_PERIOD.value)
        if ticker in seen:
            raise RuntimeError(MissingReason.PARSE_FAILURE.value)
        seen.add(ticker)
        row = {
            "provider": PROVIDER,
            "trade_date": trade_date.isoformat(),
            "ticker": ticker,
            "open": _number(_first(item, "Open", "OpenPrice", "open")),
            "high": _number(_first(item, "High", "HighPrice", "high")),
            "low": _number(_first(item, "Low", "LowPrice", "low")),
            "close": _number(_first(item, "Close", "ClosePrice", "close")),
            "previous": _number(_first(item, "Previous", "PreviousPrice", "previous")),
            "volume": _number(_first(item, "Volume", "volume")),
            "value": _number(_first(item, "Value", "value")),
            "frequency": _number(_first(item, "Frequency", "frequency")),
            "bid": _number(_first(item, "Bid", "bid")),
            "offer": _number(_first(item, "Offer", "offer")),
            "bid_volume": _number(_first(item, "BidVolume", "bidVolume")),
            "offer_volume": _number(_first(item, "OfferVolume", "offerVolume")),
            "listed_shares": _number(_first(item, "ListedShares", "listedShares")),
            # ZAPI's official response currently spells this field TradebleShares.
            "tradeable_shares": _number(_first(item, "TradebleShares", "TradeableShares", "tradeableShares")),
            "foreign_buy": _number(_first(item, "ForeignBuy", "foreignBuy")),
            "foreign_sell": _number(_first(item, "ForeignSell", "foreignSell")),
            "non_regular_volume": _number(_first(item, "NonRegularVolume", "nonRegularVolume")),
            "non_regular_value": _number(_first(item, "NonRegularValue", "nonRegularValue")),
            "non_regular_frequency": _number(_first(item, "NonRegularFrequency", "nonRegularFrequency")),
            "source_url": ZAPI_STOCK_SUMMARY_URL,
            "fetched_at": stamp,
            "freshness_state": "CURRENT",
            "validation_state": "VALID",
        }
        normalized.append(row)

    canonical = [
        {name: value for name, value in row.items() if name not in {"payload_hash", "fetched_at"}}
        for row in sorted(normalized, key=lambda value: value["ticker"])
    ]
    payload_hash = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    for row in normalized:
        row["payload_hash"] = payload_hash
    return normalized


def validate_stock_summary(
    rows: Iterable[Mapping[str, Any]], *, trade_date: date, minimum_ticker_breadth: int
) -> tuple[bool, str]:
    records = [dict(row) for row in rows]
    if not records:
        return False, MissingReason.EMPTY_RESPONSE.value
    tickers = [_canonical_ticker(row.get("ticker")) for row in records]
    if len(set(tickers)) != len(tickers) or any(not ticker for ticker in tickers):
        return False, MissingReason.PARSE_FAILURE.value
    if any(_day(row.get("trade_date")) != trade_date for row in records):
        return False, MissingReason.WRONG_PERIOD.value
    numeric_fields = (
        "open", "high", "low", "close", "previous", "volume", "value", "frequency",
        "bid", "offer", "bid_volume", "offer_volume", "listed_shares", "tradeable_shares",
        "foreign_buy", "foreign_sell", "non_regular_volume", "non_regular_value",
        "non_regular_frequency",
    )
    if any(
        value is not None and (not isinstance(value, (int, float)) or value < 0)
        for row in records for value in (row.get(name) for name in numeric_fields)
    ):
        return False, MissingReason.CONTEXT_REJECTED.value
    if len(tickers) < max(1, int(minimum_ticker_breadth)):
        return False, MissingReason.INSUFFICIENT_HISTORY.value
    return True, "VALID"


class SharedStockSummaryEvidence:
    def __init__(
        self,
        client_id: str,
        *,
        backend: Any | None = None,
        coordinator: SharedEvidenceCoordinator | None = None,
        session: Any | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 20.0,
    ):
        self.client_id = _clean(client_id).upper() or "UNKNOWN"
        self.config = HubConfig.from_environment(client_id=self.client_id)
        self.backend = backend or (SupabaseEvidenceBackend(self.config) if self.config.ready else None)
        self.coordinator = coordinator or (
            SharedEvidenceCoordinator(self.backend, client_id=self.client_id) if self.backend is not None else None
        )
        self.session = session or requests.Session()
        self.api_key = _secret("ZAPI_KEY") if api_key is None else _clean(api_key)
        self.timeout_seconds = max(2.0, min(float(timeout_seconds), 60.0))

    @property
    def ready(self) -> bool:
        return self.backend is not None and self.coordinator is not None

    def status(self) -> dict[str, str]:
        return {
            "hub_state": "CONFIGURED" if self.ready else "MISSING",
            "zapi_key_state": "CONFIGURED" if self.api_key else "MISSING",
            "client_id": self.client_id,
        }

    @staticmethod
    def _quota_remaining(headers: Mapping[str, Any]) -> int | None:
        lowered = {str(name).lower(): value for name, value in headers.items()}
        for name in ("x-ratelimit-remaining", "ratelimit-remaining", "x-rate-limit-remaining"):
            if name in lowered:
                try:
                    return int(lowered[name])
                except (TypeError, ValueError):
                    return None
        return None

    def _fetch_bulk(
        self, trade_date: date, universe: set[str], fetch_meta: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError(MissingReason.ENVIRONMENT_BLOCKED.value)
        fetch_meta["http_calls"] = int(fetch_meta.get("http_calls", 0)) + 1
        try:
            response = self.session.request(
                "GET",
                ZAPI_STOCK_SUMMARY_URL,
                params={"date": trade_date.isoformat(), "length": BULK_LENGTH, "start": 0},
                headers={
                    "Accept": "application/json",
                    "User-Agent": "Shared-IDX-Evidence-Hub/stock-summary",
                    "x-api-key": self.api_key,
                },
                timeout=self.timeout_seconds,
            )
        except requests.Timeout as exc:
            raise RuntimeError(MissingReason.TIMEOUT.value) from exc
        except requests.ConnectionError as exc:
            raise RuntimeError(MissingReason.CONNECTION_ERROR.value) from exc

        status = int(getattr(response, "status_code", 0) or 0)
        headers = getattr(response, "headers", {}) or {}
        quota = self._quota_remaining(headers)
        if quota is not None:
            fetch_meta["quota_remaining"] = quota
        fetch_meta["http_status"] = status
        if status == 429:
            message = _clean(getattr(response, "text", "")).lower()
            if quota == 0 or "quota" in message:
                raise RuntimeError(MissingReason.QUOTA_EXHAUSTED.value)
            raise RuntimeError(MissingReason.HTTP_429.value)
        if status in {401, 403, 404}:
            raise RuntimeError(f"HTTP_{status}")
        if not 200 <= status < 300:
            raise RuntimeError(f"HTTP_{status}")
        if not getattr(response, "content", b""):
            raise RuntimeError(MissingReason.EMPTY_RESPONSE.value)
        try:
            payload = response.json()
        except (TypeError, ValueError) as exc:
            raise RuntimeError(MissingReason.PARSE_FAILURE.value) from exc
        return normalize_stock_summary(payload, trade_date=trade_date, universe=universe)

    def get_day(
        self,
        trade_date: date,
        universe: Iterable[str],
        *,
        minimum_ticker_breadth: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        names = {_canonical_ticker(value) for value in universe if _canonical_ticker(value)}
        minimum = (
            max(1, math.ceil(len(names) * 0.8))
            if minimum_ticker_breadth is None
            else max(1, int(minimum_ticker_breadth))
        )
        if not self.ready:
            return [], {**self.status(), "state": MissingReason.ENVIRONMENT_BLOCKED.value, "http_calls": 0}
        fetch_meta: dict[str, Any] = {"http_calls": 0}

        def read_current() -> list[dict[str, Any]]:
            return self.backend.read_rows(
                TABLE,
                {"provider": PROVIDER, "trade_date": trade_date.isoformat()},
                limit=BULK_LENGTH,
            )

        def fetch() -> list[dict[str, Any]]:
            return self._fetch_bulk(trade_date, names, fetch_meta)

        def persist(rows: list[Mapping[str, Any]]) -> int:
            written = self.backend.upsert_rows(
                TABLE, rows, conflict=("provider", "trade_date", "ticker")
            )
            return len(written)

        result = self.coordinator.get_or_refresh(
            EvidenceKey(PROVIDER, EVIDENCE_FAMILY, SCOPE, trade_date),
            read_current=read_current,
            fetch=fetch,
            persist=persist,
            validate=lambda rows: validate_stock_summary(
                rows, trade_date=trade_date, minimum_ticker_breadth=minimum
            ),
            minimum_rows=minimum,
            lease_seconds=300,
        )
        rows = [dict(row) for row in result.rows]
        meta: dict[str, Any] = {
            **self.status(),
            "state": result.reason,
            "rows": len(rows),
            "ticker_breadth": len({_canonical_ticker(row.get("ticker")) for row in rows}),
            "provider_called": bool(fetch_meta["http_calls"]),
            "request_avoided": result.request_avoided,
            "cache_hit": result.cache_hit,
            "lease_state": result.lease_state,
            **fetch_meta,
        }
        return rows, meta

    def metrics(self) -> dict[str, int]:
        return self.coordinator.metrics() if self.coordinator is not None else {}


__all__ = [
    "BULK_LENGTH",
    "EVIDENCE_FAMILY",
    "PROVIDER",
    "SCOPE",
    "SharedStockSummaryEvidence",
    "TABLE",
    "ZAPI_STOCK_SUMMARY_URL",
    "normalize_stock_summary",
    "validate_stock_summary",
]
