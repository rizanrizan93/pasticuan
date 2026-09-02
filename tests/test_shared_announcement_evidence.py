from __future__ import annotations

from datetime import date
from pathlib import Path
import threading
from typing import Any, Mapping

import pytest
import requests

from shared_announcement_evidence import (
    CONFIRMATION_STATE,
    MAX_PAGES_PER_RUN,
    PAGE_LENGTH,
    REQUEST_TIMEOUT_SECONDS,
    SharedAnnouncementEvidence,
    ZAPI_ANNOUNCEMENTS_URL,
    ZAPI_PRESS_RELEASE_URL,
    normalize_feed_items,
    validate_announcement_rows,
)
from shared_evidence_hub import EvidenceKey, SharedEvidenceCoordinator


DAY = date(2026, 8, 2)
ATTACHMENT = "https://www.idx.co.id/StaticData/NewsAndAnnouncement/ANNOUNCEMENTSTOCK/doc.pdf"


def _announcement(
    *,
    identity: str = "event-1",
    code: str = "BBCA",
    title: str = "Rencana Rights Issue dan Perubahan Pengendalian",
    published: str = "2026-08-02T16:37:12",
    event_date: str | None = None,
    attachment: str = ATTACHMENT,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": identity, "code": code, "type": "STOCK", "title": title,
        "subject": "Keterbukaan informasi", "formId": "11000",
        "createdAt": "2026-08-02T18:00:00",
        "publishedAt": published, "announcementNo": "001/TEST/VIII/2026",
        "attachments": [{"url": attachment, "fileName": "doc.pdf"}],
    }
    if event_date is not None:
        row["eventDate"] = event_date
    return row


def _press(*, identity: int = 2680, published: str = "2026-08-02T11:00:00") -> dict[str, Any]:
    return {
        "id": identity, "title": "BEI meluncurkan layanan baru", "summary": "Ringkasan resmi BEI",
        "locale": "id-id", "publishedAt": published,
    }


def _announcement_payload(items: list[Mapping[str, Any]], *, page: int = 1, total: int | None = None) -> dict[str, Any]:
    return {
        "data": list(items), "page": page, "total": len(items) if total is None else total,
        "length": PAGE_LENGTH, "locale": "id", "dataset": "announcements", "provider": "idx",
    }


def _press_payload(items: list[Mapping[str, Any]], *, page: int = 1, has_more: bool = False) -> dict[str, Any]:
    return {
        "items": list(items), "page": page, "total": len(items), "length": PAGE_LENGTH,
        "hasMore": has_more, "dataset": "press-release", "provider": "idx",
    }


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
                return {"acquired": False, "lease_state": "HELD"}
            self.leases[identity] = {"state": "HELD", "holder": holder}
            return {"acquired": True, "lease_state": "HELD"}

    def complete_lease(self, key: EvidenceKey, holder: str, state: str) -> bool:
        self.leases[self._identity(key)]["state"] = "COMPLETED"
        return True

    def fail_lease(self, key: EvidenceKey, holder: str, reason: str) -> bool:
        self.leases[self._identity(key)].update({"state": "FAILED", "reason": reason})
        return True

    def record_provider_state(self, row: Mapping[str, Any]) -> None:
        self.provider_states.append(dict(row))

    def read_rows(self, table: str, filters: Mapping[str, Any], *, limit: int, **_: Any) -> list[dict[str, Any]]:
        assert table == "evidence_announcements"
        return [dict(row) for row in self.rows if all(str(row.get(k)) == str(v) for k, v in filters.items())][:limit]

    def upsert_rows(self, table: str, rows: list[Mapping[str, Any]], *, conflict: tuple[str, ...]) -> list[dict[str, Any]]:
        assert table == "evidence_announcements" and conflict == ("source_event_id",)
        keyed = {row["source_event_id"]: dict(row) for row in self.rows}
        for row in rows:
            keyed[str(row["source_event_id"])] = dict(row)
        self.rows = list(keyed.values())
        return [dict(row) for row in rows]


def _producer(backend: MemoryBackend, session: Session, *, client: str = "PASTICUAN", api_key: str = "test-key") -> SharedAnnouncementEvidence:
    coordinator = SharedEvidenceCoordinator(backend, client_id=client, worker_id=f"{client}-worker")
    return SharedAnnouncementEvidence(client, backend=backend, coordinator=coordinator, session=session, api_key=api_key)


def test_announcement_metadata_never_confirms_title_only_material_event() -> None:
    row = normalize_feed_items([_announcement()], feed="announcements", publication_date=DAY)[0]
    assert row["event_type"] == "IDX_DISCLOSURE_FORM_11000"
    assert row["event_confirmation_state"] == CONFIRMATION_STATE
    assert row["source_document_hash"] is None
    assert row["event_date"] is None
    assert row["attachment_urls"] == [ATTACHMENT] and row["attachment_count"] == 1
    assert "RIGHTS" not in row["event_type"] and "CONTROL" not in row["event_type"]


def test_explicit_event_date_is_separate_from_publication_and_created_dates() -> None:
    row = normalize_feed_items(
        [_announcement(event_date="2026-08-01T09:00:00")], feed="announcements", publication_date=DAY
    )[0]
    assert row["event_date"] == "2026-08-01"
    assert row["publication_date"] == "2026-08-02"
    assert row["event_at"] != row["published_at"]


def test_press_release_is_global_exchange_metadata_without_ticker_inference() -> None:
    row = normalize_feed_items([_press()], feed="press-release", publication_date=DAY)[0]
    assert row["ticker"] is None and row["event_type"] == "IDX_PRESS_RELEASE"
    assert row["source"] == "IDX_GLOBAL_PRESS_RELEASE_VIA_ZAPI"
    assert row["event_date"] is None and row["attachment_count"] == 0


def test_deduplication_is_deterministic_and_conflicts_fail_closed() -> None:
    item = _announcement()
    rows = normalize_feed_items([item, dict(item)], feed="announcements", publication_date=DAY)
    assert len(rows) == 1
    changed = dict(item)
    changed["title"] = "Conflicting payload"
    with pytest.raises(RuntimeError, match="PARSE_FAILURE"):
        normalize_feed_items([item, changed], feed="announcements", publication_date=DAY)


def test_missing_provider_id_uses_stable_payload_hash_identity() -> None:
    item = _announcement(identity="")
    first = normalize_feed_items([item], feed="announcements", publication_date=DAY)[0]
    second = normalize_feed_items([dict(item)], feed="announcements", publication_date=DAY)[0]
    assert first["source_event_id"] == second["source_event_id"]
    assert first["source_event_id"].startswith("IDX_ANNOUNCEMENTS:")


@pytest.mark.parametrize(
    "item,reason",
    [
        (_announcement(code="BAD!"), "NO_ROWS"),
        (_announcement(attachment="https://evil.example/doc.pdf"), "CONTEXT_REJECTED"),
    ],
)
def test_invalid_ticker_is_skipped_and_nonofficial_attachment_is_rejected(item: dict[str, Any], reason: str) -> None:
    if reason == "NO_ROWS":
        assert normalize_feed_items([item], feed="announcements", publication_date=DAY) == []
    else:
        with pytest.raises(RuntimeError, match=reason):
            normalize_feed_items([item], feed="announcements", publication_date=DAY)


def test_global_pipeline_uses_one_feed_call_not_per_ticker() -> None:
    backend = MemoryBackend()
    payload = _announcement_payload([_announcement(identity="a", code="BBCA"), _announcement(identity="b", code="BBRI")])
    session = Session([Response(payload)])
    rows, meta = _producer(backend, session).get_day(DAY)
    assert {row["ticker"] for row in rows} == {"BBCA", "BBRI"}
    assert len(session.calls) == 1 and meta["api_calls"] == 1 and meta["attachment_calls"] == 0
    assert session.calls[0]["url"] == ZAPI_ANNOUNCEMENTS_URL
    assert session.calls[0]["params"] == {"page": 1, "length": PAGE_LENGTH, "locale": "id"}
    assert session.calls[0]["allow_redirects"] is False
    assert session.calls[0]["timeout"] == REQUEST_TIMEOUT_SECONDS
    assert session.calls[0]["headers"]["User-Agent"] == "Shared-IDX-Evidence-Hub/announcement-metadata"
    assert meta["page_budget"] == MAX_PAGES_PER_RUN and meta["bounded_complete"] is True
    assert "company-announcements" not in str(session.calls)


def test_incremental_pagination_stops_after_crossing_target_day() -> None:
    page_one = _announcement_payload([_announcement(identity="a"), _announcement(identity="b")], page=1, total=300)
    page_two = _announcement_payload([
        _announcement(identity="c"),
        _announcement(identity="old", published="2026-08-01T23:00:00"),
    ], page=2, total=300)
    session = Session([Response(page_one), Response(page_two)])
    rows, meta = _producer(MemoryBackend(), session).get_day(DAY, max_pages=5)
    assert {row["source_event_id"] for row in rows} == {
        "IDX_ANNOUNCEMENTS:a", "IDX_ANNOUNCEMENTS:b", "IDX_ANNOUNCEMENTS:c",
    }
    assert meta["pages"] == 2 and meta["api_calls"] == 2


def test_publication_day_is_normalized_to_idx_local_date() -> None:
    row = normalize_feed_items(
        [_announcement(published="2026-08-01T18:30:00Z")],
        feed="announcements",
        publication_date=DAY,
    )[0]
    assert row["publication_date"] == "2026-08-02"
    assert row["published_at"].startswith("2026-08-01T18:30:00")


def test_page_budget_exhaustion_rejects_partial_day() -> None:
    page_one = _announcement_payload([_announcement(identity="a")], page=1, total=999)
    page_two = _announcement_payload([_announcement(identity="b")], page=2, total=999)
    backend = MemoryBackend()
    rows, meta = _producer(
        backend, Session([Response(page_one), Response(page_two)])
    ).get_day(DAY, max_pages=2)
    assert rows == []
    assert backend.rows == []
    assert meta["state"] == "CONTEXT_REJECTED"
    assert meta["api_calls"] == 2
    assert meta["bounded_complete"] is False


def test_redirect_is_blocked_and_counted_as_attempted_request() -> None:
    session = Session([Response(status=302)])
    rows, meta = _producer(MemoryBackend(), session).get_day(DAY)
    assert rows == []
    assert meta["state"] == "CONTEXT_REJECTED"
    assert meta["api_calls"] == 1
    assert session.calls[0]["allow_redirects"] is False


def test_press_release_pipeline_uses_global_press_feed() -> None:
    session = Session([Response(_press_payload([_press()]))])
    rows, meta = _producer(MemoryBackend(), session).get_day(DAY, feed="press-release")
    assert len(rows) == 1 and meta["state"] == "REFRESHED"
    assert session.calls[0]["url"] == ZAPI_PRESS_RELEASE_URL
    assert session.calls[0]["params"] == {"page": 1, "length": PAGE_LENGTH}


@pytest.mark.parametrize("feed", ["announcements", "press-release"])
@pytest.mark.parametrize("first,second", [("PASTICUAN", "EMIR"), ("EMIR", "PASTICUAN")])
def test_second_scanner_reuses_global_day_without_key(feed: str, first: str, second: str) -> None:
    backend = MemoryBackend()
    payload = _announcement_payload([_announcement()]) if feed == "announcements" else _press_payload([_press()])
    first_rows, _ = _producer(backend, Session([Response(payload)]), client=first).get_day(DAY, feed=feed)
    second_session = Session([])
    second_rows, meta = _producer(backend, second_session, client=second, api_key="").get_day(DAY, feed=feed)
    assert first_rows == second_rows and not second_session.calls
    assert meta["cache_hit"] and meta["request_avoided"] and meta["api_calls"] == 0


def test_missing_key_on_cache_miss_makes_no_request() -> None:
    session = Session([])
    rows, meta = _producer(MemoryBackend(), session, api_key="").get_day(DAY)
    assert not rows and not session.calls and meta["state"] == "ENVIRONMENT_BLOCKED"


@pytest.mark.parametrize("status,reason", [(401, "HTTP_401"), (403, "HTTP_403"), (404, "HTTP_404"), (429, "HTTP_429")])
def test_http_failures_are_explicit(status: int, reason: str) -> None:
    rows, meta = _producer(MemoryBackend(), Session([Response(status=status)])).get_day(DAY)
    assert not rows and meta["state"] == reason and meta["api_calls"] == 1


@pytest.mark.parametrize(
    "outcome,reason",
    [(requests.Timeout("slow"), "TIMEOUT"), (requests.ConnectionError("offline"), "CONNECTION_ERROR")],
)
def test_network_failures_are_explicit(outcome: Exception, reason: str) -> None:
    rows, meta = _producer(MemoryBackend(), Session([outcome])).get_day(DAY)
    assert not rows and meta["state"] == reason and meta["api_calls"] == 1


@pytest.mark.parametrize(
    "response,reason",
    [
        (Response(content=b""), "EMPTY_RESPONSE"),
        (Response(malformed=True), "PARSE_FAILURE"),
        (Response({"unexpected": []}), "CONTEXT_REJECTED"),
        (Response({"data": [], "dataset": "announcements", "provider": "other"}), "CONTEXT_REJECTED"),
        (Response(_announcement_payload([])), "NO_REPORT"),
        (Response(_announcement_payload([_announcement(published="2026-08-01T10:00:00")])), "NO_REPORT"),
    ],
)
def test_empty_malformed_wrong_provider_and_wrong_day_fail_closed(response: Response, reason: str) -> None:
    rows, meta = _producer(MemoryBackend(), Session([response])).get_day(DAY)
    assert not rows and meta["state"] == reason


def test_validation_rejects_document_confirmation_without_document_hash() -> None:
    rows = normalize_feed_items([_announcement()], feed="announcements", publication_date=DAY)
    rows[0]["event_confirmation_state"] = "DOCUMENT_CONFIRMED"
    assert validate_announcement_rows(rows, feed="announcements", publication_date=DAY) == (
        False, "CONTEXT_REJECTED"
    )


def test_rows_have_no_scanner_conclusions_or_title_derived_semantics() -> None:
    row = normalize_feed_items([_announcement()], feed="announcements", publication_date=DAY)[0]
    forbidden = {"score", "rank", "gate", "signal", "recommendation", "contract_confirmed", "rights_issue_confirmed"}
    assert forbidden.isdisjoint(row)


def test_migration_preserves_metadata_confirmation_and_feed_readback_contract() -> None:
    migration = next((Path(__file__).resolve().parents[1] / "database").glob("migration_v*_shared_evidence_hub.sql"))
    sql = migration.read_text(encoding="utf-8").lower()
    assert "event_confirmation_state text not null" in sql
    assert "attachment_urls jsonb not null" in sql
    assert "evidence_announcements_feed_date_idx" in sql
    assert "material-event semantics require document confirmation" in sql
    assert "'evidence_announcements'" in sql
