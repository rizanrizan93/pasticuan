from __future__ import annotations

from datetime import date
from pathlib import Path
import threading
from typing import Any, Mapping

import pytest
import requests

from shared_evidence_hub import EvidenceKey, SharedEvidenceCoordinator
from shared_risk_event_evidence import (
    FEEDS,
    MARGIN_LENGTH,
    NOTICE_LENGTH,
    SharedRiskEventEvidence,
    derive_recent_dilution_events,
    normalize_risk_rows,
    validate_risk_rows,
)


START = date(2026, 8, 1)
END = date(2026, 8, 31)
OBSERVED = date(2026, 9, 1)
ATTACHMENT = "https://www.idx.co.id/Portals/0/StaticData/NewsAndAnnouncement/UMA/test.pdf"


def _uma(**changes: Any) -> dict[str, Any]:
    row = {
        "id": "20260811081324", "code": "FUTR", "date": "2026-08-11",
        "name": "Futura Energi Global Tbk.", "title": "UMA atas Saham FUTR",
        "status": "A", "attachment": ATTACHMENT,
        "announcementNo": "Peng-UMA-00237/BEI.WAS/08-2026",
    }
    row.update(changes)
    return row


def _suspension(**changes: Any) -> dict[str, Any]:
    row = {
        "code": "TRUK", "date": "2026-08-11",
        "title": "Penghentian Sementara Perdagangan Saham TRUK",
        "infoType": "SPT", "attachment": ATTACHMENT,
    }
    row.update(changes)
    return row


def _notice_payload(feed: str, items: list[Mapping[str, Any]], *, page: int = 1, has_more: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "page": page, "count": len(items), "items": list(items), "total": len(items),
        "dateFrom": START.strftime("%Y%m%d"), "dateTo": END.strftime("%Y%m%d"),
        "dataset": feed, "hasMore": has_more, "nextPage": page + 1 if has_more else None,
        "provider": "idx",
    }
    if feed == "suspension":
        payload.update({"type": "SPT", "locale": "id"})
    return payload


def _margin_payload(items: list[Mapping[str, Any]], *, start: int = 0, total: int | None = None, response_date: str = "2026-08-11T00:00:00") -> dict[str, Any]:
    return {
        "data": list(items), "date": response_date, "start": start,
        "total": len(items) if total is None else total, "length": MARGIN_LENGTH,
        "dataset": "margin-summary", "provider": "idx",
    }


def _margin(code: str = "BBCA") -> dict[str, Any]:
    return {"low": 8000, "code": code, "high": 8500, "close": 8400, "value": 4_000_000, "change": 25, "volume": 474_200, "frequency": 178}


def _lendable(code: str = "BBCA") -> dict[str, Any]:
    return {"code": code, "volume": 4_293_090, "regularBorrowFee": "18%", "frontEndBorrowFee": "5%-20%"}


def _lendable_payload(items: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {"view": "list", "count": len(items), "items": list(items), "total": len(items), "dataset": "lendable-stock", "provider": "idx"}


class Response:
    def __init__(self, payload: Any = None, *, status: int = 200, content: bytes = b"json", malformed: bool = False):
        self.payload = payload
        self.status_code = status
        self.content = content
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
                return {"acquired": False}
            self.leases[identity] = {"state": "HELD", "holder": holder}
            return {"acquired": True}

    def complete_lease(self, key: EvidenceKey, holder: str, state: str) -> bool:
        self.leases[self._identity(key)]["state"] = "COMPLETED"
        return True

    def fail_lease(self, key: EvidenceKey, holder: str, reason: str) -> bool:
        self.leases[self._identity(key)].update({"state": "FAILED", "reason": reason})
        return True

    def record_provider_state(self, row: Mapping[str, Any]) -> None:
        self.provider_states.append(dict(row))

    def read_rows(self, table: str, filters: Mapping[str, Any], *, limit: int, **_: Any) -> list[dict[str, Any]]:
        assert table == "evidence_risk_events"
        return [dict(row) for row in self.rows if all(str(row.get(key)) == str(value) for key, value in filters.items())][:limit]

    def upsert_rows(self, table: str, rows: list[Mapping[str, Any]], *, conflict: tuple[str, ...]) -> list[dict[str, Any]]:
        assert table == "evidence_risk_events"
        assert conflict == ("provider", "event_type", "event_date", "ticker", "source_id")
        keyed = {tuple(row.get(key) for key in conflict): dict(row) for row in self.rows}
        for row in rows:
            keyed[tuple(row.get(key) for key in conflict)] = dict(row)
        self.rows = list(keyed.values())
        return [dict(row) for row in rows]


def _producer(backend: MemoryBackend, session: Session, *, client: str = "PASTICUAN", api_key: str = "fixture-key") -> SharedRiskEventEvidence:
    coordinator = SharedEvidenceCoordinator(backend, client_id=client, worker_id=f"{client}-worker")
    return SharedRiskEventEvidence(client, backend=backend, coordinator=coordinator, session=session, api_key=api_key)


def test_uma_notice_preserves_exact_publication_metadata_without_attachment_download() -> None:
    row = normalize_risk_rows([_uma()], feed="uma", source_period=START, window_end_date=END, observed_on=OBSERVED)[0]
    assert row["event_type"] == "UMA_NOTICE" and row["active_state"] == "UMA_ACTIVE_OR_RECENT"
    assert row["event_date"] == "2026-08-11" and row["publication_date"] == "2026-08-11"
    assert row["date_semantics"] == "NOTICE_PUBLICATION_DATE"
    assert row["source_url"] == ATTACHMENT and row["details"]["attachment_downloaded"] is False


def test_suspension_title_does_not_assert_current_suspension_or_release() -> None:
    item = _suspension(title="Pembukaan Kembali Perdagangan Saham TRUK")
    row = normalize_risk_rows([item], feed="suspension", source_period=START, window_end_date=END, observed_on=OBSERVED)[0]
    assert row["event_type"] == "SUSPENSION_NOTICE"
    assert row["active_state"] == "SUSPENSION_ACTIVE_OR_RECENT"
    assert row["title"] == item["title"]
    assert "RELEASED" not in row["active_state"] and "CURRENTLY_SUSPENDED" not in row["active_state"]


def test_notice_outside_window_or_invalid_ticker_is_skipped() -> None:
    assert normalize_risk_rows(
        [_uma(date="2026-07-31"), _uma(id="two", code="BAD!")],
        feed="uma", source_period=START, window_end_date=END, observed_on=OBSERVED,
    ) == []


def test_nonofficial_notice_attachment_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="CONTEXT_REJECTED"):
        normalize_risk_rows(
            [_uma(attachment="https://evil.example/file.pdf")],
            feed="uma", source_period=START, window_end_date=END, observed_on=OBSERVED,
        )


def test_margin_eligibility_uses_provider_period_not_observation_date() -> None:
    period = date(2026, 8, 11)
    row = normalize_risk_rows(
        [_margin()], feed="margin-summary", source_period=period,
        window_end_date=period, observed_on=OBSERVED, provider_date=period,
    )[0]
    assert row["event_date"] == "2026-08-11" and row["publication_date"] is None
    assert row["active_state"] == "MARGIN_ELIGIBLE" and row["date_semantics"] == "PROVIDER_PERIOD_DATE"
    assert row["details"]["volume"] == 474_200 and row["details"]["close"] == 8400


def test_margin_wrong_provider_period_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="WRONG_PERIOD"):
        normalize_risk_rows(
            [_margin()], feed="margin-summary", source_period=date(2026, 8, 11),
            window_end_date=date(2026, 8, 11), observed_on=OBSERVED,
            provider_date=date(2026, 8, 10),
        )


def test_lendable_is_explicit_observation_not_historical_event_date() -> None:
    row = normalize_risk_rows(
        [_lendable()], feed="lendable-stock", source_period=OBSERVED,
        window_end_date=OBSERVED, observed_on=OBSERVED,
    )[0]
    assert row["event_date"] == OBSERVED.isoformat() and row["date_semantics"] == "OBSERVED_ON"
    assert row["active_state"] == "LENDABLE" and row["details"]["regular_borrow_fee"] == "18%"


def test_deduplication_is_deterministic_and_conflicts_fail_closed() -> None:
    item = _uma()
    rows = normalize_risk_rows([item, dict(item)], feed="uma", source_period=START, window_end_date=END, observed_on=OBSERVED)
    assert len(rows) == 1
    with pytest.raises(RuntimeError, match="PARSE_FAILURE"):
        normalize_risk_rows([item, dict(item, title="Conflict")], feed="uma", source_period=START, window_end_date=END, observed_on=OBSERVED)


def _capital(**changes: Any) -> dict[str, Any]:
    row = {
        "ticker": "BBCA", "event_type": "RIGHTS_ISSUE", "event_date": "2026-08-10",
        "publication_date": "2026-08-01", "pre_shares": 1000, "post_shares": 1200,
        "delta_shares": 200, "delta_percent": 20, "source": "IDX_ISSUED_HISTORY_VIA_ZAPI",
        "source_url": FEEDS["uma"]["url"], "source_id": "rights-1",
        "source_verified": True, "validation_state": "VALID",
    }
    row.update(changes)
    return row


def test_recent_dilution_context_requires_explicit_positive_eligible_delta() -> None:
    rows = derive_recent_dilution_events([
        _capital(),
        _capital(source_id="split", event_type="STOCK_SPLIT"),
        _capital(source_id="bonus", event_type="BONUS_SHARES"),
        _capital(source_id="negative", delta_shares=-10),
        _capital(source_id="old", event_date="2020-01-01"),
        _capital(source_id="unverified", source_verified=False),
    ], observed_on=OBSERVED)
    assert len(rows) == 1 and rows[0]["active_state"] == "RECENT_DILUTION_EVENT"
    assert rows[0]["details"]["delta_shares"] == 200
    assert rows[0]["date_semantics"] == "CAPITAL_ACTION_EVENT_DATE"


def test_derived_dilution_persists_without_provider_call() -> None:
    backend = MemoryBackend()
    session = Session([])
    rows, meta = _producer(backend, session, api_key="").persist_dilution_context([_capital()], observed_on=OBSERVED)
    assert len(rows) == 1 and meta == {"state": "PERSISTED", "rows": 1, "api_calls": 0}
    assert not session.calls and backend.rows == rows


@pytest.mark.parametrize("feed,item", [("uma", _uma()), ("suspension", _suspension())])
def test_notice_window_uses_global_filtered_pages_without_keyword(feed: str, item: dict[str, Any]) -> None:
    session = Session([Response(_notice_payload(feed, [item]))])
    rows, meta = _producer(MemoryBackend(), session).get_notice_window(START, END, feed=feed, observed_on=OBSERVED)
    assert len(rows) == 1 and meta["api_calls"] == 1 and meta["attachment_calls"] == 0
    expected = {"page": 1, "length": NOTICE_LENGTH, "dateFrom": "20260801", "dateTo": "20260831"}
    if feed == "suspension":
        expected.update({"type": "SPT", "locale": "id"})
    assert session.calls[0]["url"] == FEEDS[feed]["url"] and session.calls[0]["params"] == expected
    assert "keyword" not in session.calls[0]["params"]


def test_notice_pagination_follows_has_more() -> None:
    session = Session([
        Response(_notice_payload("uma", [_uma()], has_more=True)),
        Response(_notice_payload("uma", [_uma(id="two", code="BBRI")], page=2)),
    ])
    rows, meta = _producer(MemoryBackend(), session).get_notice_window(START, END, observed_on=OBSERVED)
    assert {row["ticker"] for row in rows} == {"FUTR", "BBRI"}
    assert meta["api_calls"] == 2 and session.calls[1]["params"]["page"] == 2


def test_notice_never_persists_truncated_max_page_result() -> None:
    session = Session([Response(_notice_payload("uma", [_uma()], has_more=True))])
    rows, meta = _producer(MemoryBackend(), session).get_notice_window(
        START, END, observed_on=OBSERVED, max_pages=1
    )
    assert not rows and meta["state"] == "INSUFFICIENT_HISTORY" and meta["api_calls"] == 1


def test_margin_uses_global_offset_pagination_and_exact_period() -> None:
    period = date(2026, 8, 11)
    session = Session([
        Response(_margin_payload([_margin()], total=301)),
        Response(_margin_payload([_margin("BBRI")], start=300, total=301)),
    ])
    rows, meta = _producer(MemoryBackend(), session).get_margin(period, observed_on=OBSERVED)
    assert {row["ticker"] for row in rows} == {"BBCA", "BBRI"} and meta["api_calls"] == 2
    assert [call["params"] for call in session.calls] == [
        {"date": "2026-08-11", "length": MARGIN_LENGTH, "start": 0},
        {"date": "2026-08-11", "length": MARGIN_LENGTH, "start": MARGIN_LENGTH},
    ]


def test_lendable_uses_one_market_wide_list_call() -> None:
    session = Session([Response(_lendable_payload([_lendable(), _lendable("BBRI")]))])
    rows, meta = _producer(MemoryBackend(), session).get_lendable(OBSERVED)
    assert {row["ticker"] for row in rows} == {"BBCA", "BBRI"} and meta["api_calls"] == 1
    assert session.calls[0]["params"] == {"sort": "code", "view": "list"}
    assert "code" not in session.calls[0]["params"]


@pytest.mark.parametrize("first,second", [("PASTICUAN", "EMIR"), ("EMIR", "PASTICUAN")])
@pytest.mark.parametrize("feed", ["uma", "suspension", "margin-summary", "lendable-stock"])
def test_second_scanner_reuses_risk_evidence_without_key(first: str, second: str, feed: str) -> None:
    backend = MemoryBackend()
    if feed in {"uma", "suspension"}:
        item = _uma() if feed == "uma" else _suspension()
        first_rows, _ = _producer(backend, Session([Response(_notice_payload(feed, [item]))]), client=first).get_notice_window(
            START, END, feed=feed, observed_on=OBSERVED
        )
        second_session = Session([])
        second_rows, meta = _producer(backend, second_session, client=second, api_key="").get_notice_window(
            START, END, feed=feed, observed_on=OBSERVED
        )
    elif feed == "margin-summary":
        period = date(2026, 8, 11)
        first_rows, _ = _producer(backend, Session([Response(_margin_payload([_margin()]))]), client=first).get_margin(period, observed_on=OBSERVED)
        second_session = Session([])
        second_rows, meta = _producer(backend, second_session, client=second, api_key="").get_margin(period, observed_on=OBSERVED)
    else:
        first_rows, _ = _producer(backend, Session([Response(_lendable_payload([_lendable()]))]), client=first).get_lendable(OBSERVED)
        second_session = Session([])
        second_rows, meta = _producer(backend, second_session, client=second, api_key="").get_lendable(OBSERVED)
    assert first_rows == second_rows and not second_session.calls
    assert meta["cache_hit"] and meta["request_avoided"] and meta["api_calls"] == 0


def test_missing_key_invalid_feed_and_reversed_window_make_no_request() -> None:
    session = Session([])
    producer = _producer(MemoryBackend(), session, api_key="")
    assert producer.get_lendable(OBSERVED)[1]["state"] == "ENVIRONMENT_BLOCKED"
    assert producer.get_notice_window(START, END, feed="margin-summary")[1]["state"] == "CONTEXT_REJECTED"
    assert producer.get_notice_window(END, START)[1]["state"] == "WRONG_PERIOD"
    assert not session.calls


@pytest.mark.parametrize("status", [401, 403, 404, 429])
def test_http_failures_are_explicit(status: int) -> None:
    rows, meta = _producer(MemoryBackend(), Session([Response(status=status)])).get_lendable(OBSERVED)
    assert not rows and meta["state"] == f"HTTP_{status}"


@pytest.mark.parametrize("outcome,reason", [(requests.Timeout(), "TIMEOUT"), (requests.ConnectionError(), "CONNECTION_ERROR")])
def test_network_failures_are_explicit(outcome: Exception, reason: str) -> None:
    rows, meta = _producer(MemoryBackend(), Session([outcome])).get_lendable(OBSERVED)
    assert not rows and meta["state"] == reason


@pytest.mark.parametrize(
    "response,reason",
    [
        (Response(content=b""), "EMPTY_RESPONSE"),
        (Response(malformed=True), "PARSE_FAILURE"),
        (Response({"items": [], "dataset": "wrong", "provider": "idx"}), "CONTEXT_REJECTED"),
        (Response({"items": {}, "dataset": "lendable-stock", "provider": "idx"}), "PARSE_FAILURE"),
    ],
)
def test_bad_responses_fail_closed(response: Response, reason: str) -> None:
    rows, meta = _producer(MemoryBackend(), Session([response])).get_lendable(OBSERVED)
    assert not rows and meta["state"] == reason


def test_empty_feed_is_explicit_no_report() -> None:
    rows, meta = _producer(MemoryBackend(), Session([Response(_lendable_payload([]))])).get_lendable(OBSERVED)
    assert not rows and meta["state"] == "NO_REPORT"


def test_validation_and_migration_remain_factual_without_new_gates() -> None:
    row = normalize_risk_rows([_uma()], feed="uma", source_period=START, window_end_date=END, observed_on=OBSERVED)[0]
    assert validate_risk_rows([row], feed="uma", source_period=START, window_end_date=END, observed_on=OBSERVED) == (True, "VALID")
    assert validate_risk_rows([dict(row, observed_on="2026-08-31")], feed="uma", source_period=START, window_end_date=END, observed_on=OBSERVED)[1] == "WRONG_PERIOD"
    root = Path(__file__).resolve().parents[1]
    migration = next((root / "database").glob("migration_v*_shared_evidence_hub.sql")).read_text()
    module = (root / "shared_risk_event_evidence.py").read_text().lower()
    for fragment in (
        "evidence_risk_events_feed_window_idx", "details jsonb", "date_semantics text",
        "'uma_active_or_recent', 'suspension_active_or_recent', 'recent_dilution_event'",
        "grant select, insert, update on table public.%i to service_role", "enable row level security",
    ):
        assert fragment in migration.lower()
    for forbidden in ("production_gate", "rejection_gate", "scanner_score", "recommendation"):
        assert forbidden not in module
