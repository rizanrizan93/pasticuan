from __future__ import annotations

from datetime import date, datetime, timezone
import threading
from typing import Any, Mapping

import pytest
import requests

from shared_evidence_hub import EvidenceKey, SharedEvidenceCoordinator
from shared_stock_summary_evidence import (
    BULK_LENGTH,
    SharedStockSummaryEvidence,
    ZAPI_STOCK_SUMMARY_URL,
    normalize_stock_summary,
    validate_stock_summary,
)


DAY = date(2026, 8, 31)
UNIVERSE = {"BBCA", "BBRI"}


def _raw(ticker: str, *, day: str = "2026-08-31", close: Any = 9000) -> dict[str, Any]:
    return {
        "Date": day,
        "StockCode": ticker,
        "Open": 8900,
        "High": 9100,
        "Low": 8850,
        "Close": close,
        "Previous": 8875,
        "Volume": 10_000,
        "Value": 90_000_000,
        "Frequency": 500,
        "Bid": 8975,
        "Offer": 9000,
        "BidVolume": 25,
        "OfferVolume": 30,
        "ListedShares": 100_000,
        "TradebleShares": 80_000,
        "ForeignBuy": 6000,
        "ForeignSell": 4500,
        "NonRegularVolume": 12,
        "NonRegularValue": 108_000,
        "NonRegularFrequency": 2,
    }


def _payload(*rows: Mapping[str, Any]) -> dict[str, Any]:
    return {"data": {"recordsTotal": len(rows), "data": list(rows)}}


class Response:
    def __init__(
        self,
        payload: Any = None,
        *,
        status: int = 200,
        headers: Mapping[str, Any] | None = None,
        content: bytes = b"json",
        text: str = "",
        malformed: bool = False,
    ):
        self.payload = payload
        self.status_code = status
        self.headers = dict(headers or {})
        self.content = content
        self.text = text
        self.malformed = malformed

    def json(self) -> Any:
        if self.malformed:
            raise ValueError("bad json")
        return self.payload


class Session:
    def __init__(self, outcomes: list[Any]):
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, url: str, **kwargs: Any) -> Response:
        self.calls.append({"method": method, "url": url, **kwargs})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class MemoryBackend:
    def __init__(self):
        self.rows: list[dict[str, Any]] = []
        self.leases: dict[tuple[str, str, str, date], dict[str, Any]] = {}
        self.provider_states: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    @staticmethod
    def _identity(key: EvidenceKey) -> tuple[str, str, str, date]:
        normalized = key.normalized()
        return normalized.provider, normalized.family, normalized.scope, normalized.target_date

    def acquire_lease(self, key: EvidenceKey, holder: str, lease_seconds: int) -> Mapping[str, Any]:
        with self.lock:
            identity = self._identity(key)
            current = self.leases.get(identity)
            if current and current["state"] == "HELD" and current["holder"] != holder:
                return {"acquired": False, "lease_state": "HELD"}
            self.leases[identity] = {"state": "HELD", "holder": holder}
            return {"acquired": True, "lease_state": "HELD"}

    def complete_lease(self, key: EvidenceKey, holder: str, state: str) -> bool:
        with self.lock:
            self.leases[self._identity(key)].update({"state": "COMPLETED", "result": state})
        return True

    def fail_lease(self, key: EvidenceKey, holder: str, reason: str) -> bool:
        with self.lock:
            self.leases[self._identity(key)].update({"state": "FAILED", "reason": reason})
        return True

    def record_provider_state(self, row: Mapping[str, Any]) -> None:
        self.provider_states.append(dict(row))

    def read_rows(self, table: str, filters: Mapping[str, Any], *, limit: int, **_: Any) -> list[dict[str, Any]]:
        assert table == "evidence_market_daily"
        return [
            dict(row) for row in self.rows
            if all(str(row.get(name)) == str(value) for name, value in filters.items())
        ][:limit]

    def upsert_rows(
        self, table: str, rows: list[Mapping[str, Any]], *, conflict: tuple[str, ...]
    ) -> list[dict[str, Any]]:
        assert table == "evidence_market_daily"
        assert conflict == ("provider", "trade_date", "ticker")
        with self.lock:
            keyed = {
                tuple(row.get(name) for name in conflict): dict(row)
                for row in self.rows
            }
            for row in rows:
                keyed[tuple(row.get(name) for name in conflict)] = dict(row)
            self.rows = list(keyed.values())
        return [dict(row) for row in rows]


def _producer(
    backend: MemoryBackend, session: Session, *, client: str = "PASTICUAN", api_key: str = "test-key"
) -> SharedStockSummaryEvidence:
    coordinator = SharedEvidenceCoordinator(backend, client_id=client, worker_id=f"{client}-worker")
    return SharedStockSummaryEvidence(
        client,
        backend=backend,
        coordinator=coordinator,
        session=session,
        api_key=api_key,
    )


def test_normalizes_every_market_daily_field_and_hash_is_deterministic() -> None:
    stamp = datetime(2026, 8, 31, 10, tzinfo=timezone.utc)
    rows = normalize_stock_summary(
        _payload(_raw("BBCA")), trade_date=DAY, universe=UNIVERSE, fetched_at=stamp
    )
    assert rows == [{
        "provider": "ZAPI", "trade_date": "2026-08-31", "ticker": "BBCA",
        "open": 8900, "high": 9100, "low": 8850, "close": 9000, "previous": 8875,
        "volume": 10000, "value": 90000000, "frequency": 500, "bid": 8975,
        "offer": 9000, "bid_volume": 25, "offer_volume": 30,
        "listed_shares": 100000, "tradeable_shares": 80000,
        "foreign_buy": 6000, "foreign_sell": 4500, "non_regular_volume": 12,
        "non_regular_value": 108000, "non_regular_frequency": 2,
        "source_url": ZAPI_STOCK_SUMMARY_URL, "fetched_at": stamp.isoformat(),
        "freshness_state": "CURRENT", "validation_state": "VALID",
        "payload_hash": rows[0]["payload_hash"],
    }]
    assert len(rows[0]["payload_hash"]) == 64
    again = normalize_stock_summary(
        _payload(_raw("BBCA")), trade_date=DAY, universe=UNIVERSE,
        fetched_at=datetime(2026, 8, 31, 11, tzinfo=timezone.utc),
    )
    assert again[0]["payload_hash"] == rows[0]["payload_hash"]


def test_bulk_request_persists_filtered_universe_in_one_call() -> None:
    backend = MemoryBackend()
    session = Session([Response(_payload(_raw("BBCA"), _raw("BBRI"), _raw("OUTS")))])
    rows, meta = _producer(backend, session).get_day(DAY, UNIVERSE, minimum_ticker_breadth=2)
    assert {row["ticker"] for row in rows} == UNIVERSE
    assert len(session.calls) == 1
    assert session.calls[0]["params"] == {"date": "2026-08-31", "length": BULK_LENGTH, "start": 0}
    assert meta["state"] == "REFRESHED" and meta["http_calls"] == 1
    assert "test-key" not in str(meta)


def test_valid_daily_cache_is_reused_without_api_key_or_http_call() -> None:
    backend = MemoryBackend()
    backend.rows = normalize_stock_summary(_payload(_raw("BBCA")), trade_date=DAY, universe=UNIVERSE)
    session = Session([])
    rows, meta = _producer(backend, session, api_key="").get_day(
        DAY, UNIVERSE, minimum_ticker_breadth=1
    )
    assert len(rows) == 1 and not session.calls
    assert meta["state"] == "CACHE_HIT" and meta["request_avoided"]
    assert meta["zapi_key_state"] == "MISSING"


def test_stale_validation_state_refreshes_once() -> None:
    backend = MemoryBackend()
    backend.rows = normalize_stock_summary(_payload(_raw("BBCA")), trade_date=DAY, universe=UNIVERSE)
    backend.rows[0]["validation_state"] = "STALE"
    session = Session([Response(_payload(_raw("BBCA", close=9050)))])
    rows, meta = _producer(backend, session).get_day(DAY, UNIVERSE, minimum_ticker_breadth=1)
    assert len(session.calls) == 1 and rows[0]["close"] == 9050
    assert meta["state"] == "REFRESHED"


@pytest.mark.parametrize("first,second", [("PASTICUAN", "EMIR"), ("EMIR", "PASTICUAN")])
def test_second_scanner_reuses_same_daily_key(first: str, second: str) -> None:
    backend = MemoryBackend()
    first_session = Session([Response(_payload(_raw("BBCA")))])
    second_session = Session([])
    first_rows, _ = _producer(backend, first_session, client=first).get_day(
        DAY, UNIVERSE, minimum_ticker_breadth=1
    )
    second_rows, meta = _producer(backend, second_session, client=second, api_key="").get_day(
        DAY, UNIVERSE, minimum_ticker_breadth=1
    )
    assert first_rows == second_rows and not second_session.calls
    assert meta["cache_hit"] and meta["request_avoided"]


def test_upsert_readback_prevents_duplicate_daily_rows() -> None:
    backend = MemoryBackend()
    session = Session([Response(_payload(_raw("BBCA"), _raw("BBRI")))])
    producer = _producer(backend, session)
    producer.get_day(DAY, UNIVERSE, minimum_ticker_breadth=2)
    producer.get_day(DAY, UNIVERSE, minimum_ticker_breadth=2)
    assert len(backend.rows) == 2 and len(session.calls) == 1


def test_missing_key_on_cache_miss_makes_no_http_request() -> None:
    backend = MemoryBackend()
    session = Session([])
    rows, meta = _producer(backend, session, api_key="").get_day(
        DAY, UNIVERSE, minimum_ticker_breadth=1
    )
    assert not rows and not session.calls
    assert meta["state"] == "ENVIRONMENT_BLOCKED" and not meta["provider_called"]


@pytest.mark.parametrize(
    "response,reason",
    [
        (Response({}, status=401), "HTTP_401"),
        (Response({}, status=403), "HTTP_403"),
        (Response({}, status=404), "HTTP_404"),
        (Response({}, status=429, text="rate limited"), "HTTP_429"),
        (Response({}, status=429, headers={"x-ratelimit-remaining": "0"}), "QUOTA_EXHAUSTED"),
    ],
)
def test_http_and_quota_failures_are_explicit(response: Response, reason: str) -> None:
    rows, meta = _producer(MemoryBackend(), Session([response])).get_day(
        DAY, UNIVERSE, minimum_ticker_breadth=1
    )
    assert not rows and meta["state"] == reason and meta["http_calls"] == 1


@pytest.mark.parametrize(
    "outcome,reason",
    [(requests.Timeout("slow"), "TIMEOUT"), (requests.ConnectionError("offline"), "CONNECTION_ERROR")],
)
def test_network_failures_are_explicit(outcome: Exception, reason: str) -> None:
    rows, meta = _producer(MemoryBackend(), Session([outcome])).get_day(
        DAY, UNIVERSE, minimum_ticker_breadth=1
    )
    assert not rows and meta["state"] == reason


@pytest.mark.parametrize(
    "response,reason",
    [
        (Response(content=b""), "EMPTY_RESPONSE"),
        (Response(malformed=True), "PARSE_FAILURE"),
        (Response({"unexpected": []}), "PARSE_FAILURE"),
        (Response(_payload()), "EMPTY_RESPONSE"),
        (Response(_payload(_raw("BBCA", day="2026-08-29"))), "WRONG_PERIOD"),
    ],
)
def test_empty_malformed_and_wrong_date_payloads_fail_closed(response: Response, reason: str) -> None:
    rows, meta = _producer(MemoryBackend(), Session([response])).get_day(
        DAY, UNIVERSE, minimum_ticker_breadth=1
    )
    assert not rows and meta["state"] == reason


def test_quota_is_reported_only_when_provider_sends_header() -> None:
    _, with_quota = _producer(
        MemoryBackend(),
        Session([Response(_payload(_raw("BBCA")), headers={"X-RateLimit-Remaining": "17"})]),
    ).get_day(DAY, UNIVERSE, minimum_ticker_breadth=1)
    _, without_quota = _producer(
        MemoryBackend(), Session([Response(_payload(_raw("BBCA")))])
    ).get_day(DAY, UNIVERSE, minimum_ticker_breadth=1)
    assert with_quota["quota_remaining"] == 17
    assert "quota_remaining" not in without_quota


def test_validation_rejects_negative_fact_and_insufficient_breadth() -> None:
    rows = normalize_stock_summary(
        _payload(_raw("BBCA", close=-1)), trade_date=DAY, universe=UNIVERSE
    )
    assert validate_stock_summary(rows, trade_date=DAY, minimum_ticker_breadth=1) == (
        False, "CONTEXT_REJECTED"
    )
    valid = normalize_stock_summary(_payload(_raw("BBCA")), trade_date=DAY, universe=UNIVERSE)
    assert validate_stock_summary(valid, trade_date=DAY, minimum_ticker_breadth=2) == (
        False, "INSUFFICIENT_HISTORY"
    )


def test_rows_have_no_scanner_conclusion_fields() -> None:
    row = normalize_stock_summary(_payload(_raw("BBCA")), trade_date=DAY, universe=UNIVERSE)[0]
    forbidden = {"score", "rank", "gate", "recommendation", "signal", "bandar"}
    assert forbidden.isdisjoint(row)
