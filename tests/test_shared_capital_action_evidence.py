from __future__ import annotations

from datetime import date
from pathlib import Path
import threading
from typing import Any, Mapping

import pytest
import requests

from shared_capital_action_evidence import (
    FEEDS,
    ISSUED_HISTORY_LENGTH,
    MAX_PAGES_PER_RUN,
    PAGE_LENGTH,
    REQUEST_TIMEOUT_SECONDS,
    SharedCapitalActionEvidence,
    normalize_capital_actions,
    validate_capital_action_rows,
)
from shared_evidence_hub import EvidenceKey, SharedEvidenceCoordinator


PERIOD = date(2026, 8, 1)
OBSERVED = date(2026, 9, 1)


def _issued(**changes: Any) -> dict[str, Any]:
    row = {
        "id": "issued-1", "code": "INET", "action": "waran",
        "shares": 3_200, "sharesAfter": 22_375_261_532,
        "listingDate": "2026-08-14", "publicationDate": "2026-08-15",
    }
    row.update(changes)
    return row


def _monthly(feed: str, items: list[Mapping[str, Any]], *, page: int = 1, total: int | None = None, has_more: bool | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "page": page, "year": 2026, "month": 8, "count": len(items),
        "items": list(items), "total": len(items) if total is None else total,
        "dataset": feed, "provider": "idx",
    }
    if has_more is not None:
        payload["hasMore"] = has_more
    return payload


def _issued_payload(items: list[Mapping[str, Any]], *, start: int = 0, total: int | None = None) -> dict[str, Any]:
    return {
        "items": list(items), "start": start,
        "total": len(items) if total is None else total,
        "length": ISSUED_HISTORY_LENGTH,
        "dataset": "issued-history", "provider": "idx",
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
        if table == "evidence_capital_actions":
            source = self.rows
        elif table == "evidence_provider_state":
            source = self.provider_states
        else:
            raise AssertionError(table)
        return [dict(row) for row in source if all(str(row.get(key)) == str(value) for key, value in filters.items())][:limit]

    def upsert_rows(self, table: str, rows: list[Mapping[str, Any]], *, conflict: tuple[str, ...]) -> list[dict[str, Any]]:
        assert table == "evidence_capital_actions"
        assert conflict == ("ticker", "event_type", "event_date", "source_id")
        keyed = {
            (row["ticker"], row["event_type"], row["event_date"], row["source_id"]): dict(row)
            for row in self.rows
        }
        for row in rows:
            key = (row["ticker"], row["event_type"], row["event_date"], row["source_id"])
            keyed[key] = dict(row)
        self.rows = list(keyed.values())
        return [dict(row) for row in rows]


def _producer(backend: MemoryBackend, session: Session, *, client: str = "PASTICUAN", api_key: str = "fixture-key") -> SharedCapitalActionEvidence:
    coordinator = SharedEvidenceCoordinator(backend, client_id=client, worker_id=f"{client}-worker")
    return SharedCapitalActionEvidence(
        client, backend=backend, coordinator=coordinator, session=session, api_key=api_key
    )


def test_issued_history_preserves_dates_and_derives_only_compatible_facts() -> None:
    row = normalize_capital_actions(
        [_issued()], feed="issued-history", source_period=OBSERVED, observed_on=OBSERVED
    )[0]
    assert row["event_type"] == "WARRANT_EXERCISE"
    assert row["event_date"] == "2026-08-14" and row["publication_date"] == "2026-08-15"
    assert row["event_shares"] == 3_200
    assert row["post_shares"] == 22_375_261_532
    assert row["pre_shares"] is None and row["delta_shares"] is None
    assert row["calculation_state"] == "EXPLICIT_EVENT_SHARES_POST_NO_DELTA"


def test_pre_and_post_are_explicit_and_delta_is_arithmetic() -> None:
    item = _issued(action="konversi", shares=None, sharesBefore=1000, sharesAfter=1250)
    row = normalize_capital_actions([item], feed="issued-history", source_period=OBSERVED, observed_on=OBSERVED)[0]
    assert row["event_type"] == "CONVERSION" and row["delta_shares"] == 250
    assert row["delta_percent"] == 25 and row["calculation_state"] == "EXPLICIT_PRE_POST"


def test_incompatible_explicit_share_fields_fail_closed() -> None:
    item = _issued(shares=None, sharesBefore=1000, sharesAfter=1200, deltaShares=300)
    with pytest.raises(RuntimeError, match="PARSE_FAILURE"):
        normalize_capital_actions([item], feed="issued-history", source_period=OBSERVED, observed_on=OBSERVED)


def test_rights_ratio_is_preserved_without_inventing_share_counts() -> None:
    item = {
        "id": "rights-1", "code": "BBRI", "eventDate": "2026-08-20",
        "publicationDate": "2026-08-05", "ratioBefore": 4, "ratioAfter": 1,
    }
    row = normalize_capital_actions([item], feed="rights-offerings", source_period=PERIOD, observed_on=OBSERVED)[0]
    assert row["event_type"] == "RIGHTS_OFFERING"
    assert row["ratio_before"] == 4 and row["ratio_after"] == 1
    assert row["pre_shares"] is None and row["post_shares"] is None
    assert row["delta_shares"] is None and row["delta_percent"] is None
    assert row["calculation_state"] == "NO_SHARE_FACTS"


def test_reverse_split_requires_explicit_action_not_title() -> None:
    explicit = {"id": "s1", "code": "ABCD", "effectiveDate": "2026-08-21", "action": "Reverse Stock Split"}
    title_only = {"id": "s2", "code": "EFGH", "effectiveDate": "2026-08-22", "title": "Reverse Stock Split"}
    rows = normalize_capital_actions([explicit, title_only], feed="stock-splits", source_period=PERIOD, observed_on=OBSERVED)
    assert [row["event_type"] for row in rows] == ["REVERSE_STOCK_SPLIT", "STOCK_SPLIT"]


def test_stock_split_ratio_does_not_create_delta() -> None:
    item = {"id": "split", "code": "BBCA", "exDate": "2026-08-10", "ratioBefore": 1, "ratioAfter": 5}
    row = normalize_capital_actions([item], feed="stock-splits", source_period=PERIOD, observed_on=OBSERVED)[0]
    assert row["delta_shares"] is None and row["delta_percent"] is None


def test_ambiguous_new_shares_and_free_text_description_are_not_inferred() -> None:
    rights = {
        "id": "rights", "code": "BBRI", "eventDate": "2026-08-11",
        "newShares": 500, "description": "Reverse split proposal",
    }
    right_row = normalize_capital_actions(
        [rights], feed="rights-offerings", source_period=PERIOD, observed_on=OBSERVED
    )[0]
    assert right_row["post_shares"] is None and right_row["delta_shares"] is None
    split = {"id": "split", "code": "BBCA", "eventDate": "2026-08-12", "description": "Reverse Stock Split"}
    split_row = normalize_capital_actions(
        [split], feed="stock-splits", source_period=PERIOD, observed_on=OBSERVED
    )[0]
    assert split_row["event_type"] == "STOCK_SPLIT" and split_row["raw_action"] is None


def test_additional_listing_uses_explicit_additional_shares() -> None:
    item = {
        "code": "TLKM",
        "startDate": "2026-08-03",
        "lastDate": "2026-08-09",
        "shares": "1.000.000",
        "actionType": "Warrant",
    }
    row = normalize_capital_actions(
        [item], feed="additional-listings", source_period=PERIOD, observed_on=OBSERVED
    )[0]
    assert row["delta_shares"] == 1_000_000
    assert row["event_type"] == "WARRANT_EXERCISE"
    assert row["raw_action"] == "Warrant"
    assert row["event_date"] == "2026-08-09"
    assert row["event_date_kind"] == "RANGE_END"
    assert row["event_start_date"] == "2026-08-03"
    assert row["event_end_date"] == "2026-08-09"
    assert row["pre_shares"] is None and row["post_shares"] is None


def test_publication_date_is_never_substituted_from_event_date() -> None:
    row = normalize_capital_actions(
        [_issued(publicationDate=None)], feed="issued-history", source_period=OBSERVED, observed_on=OBSERVED
    )[0]
    assert row["event_date"] == "2026-08-14" and row["publication_date"] is None


def test_deduplication_is_deterministic_and_provider_id_conflicts_fail() -> None:
    item = _issued()
    rows = normalize_capital_actions([item, dict(item)], feed="issued-history", source_period=OBSERVED, observed_on=OBSERVED)
    assert len(rows) == 1
    changed = dict(item, shares=3_201)
    with pytest.raises(RuntimeError, match="PARSE_FAILURE"):
        normalize_capital_actions([item, changed], feed="issued-history", source_period=OBSERVED, observed_on=OBSERVED)


def test_missing_provider_id_has_stable_hash_identity() -> None:
    item = _issued(id=None)
    first = normalize_capital_actions([item], feed="issued-history", source_period=OBSERVED, observed_on=OBSERVED)[0]
    second = normalize_capital_actions([dict(item)], feed="issued-history", source_period=OBSERVED, observed_on=OBSERVED)[0]
    assert first["source_id"] == second["source_id"] and first["source_id"].startswith("ISSUED-HISTORY:")


def test_nonofficial_explicit_source_url_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="CONTEXT_REJECTED"):
        normalize_capital_actions(
            [_issued(sourceUrl="https://evil.example/action")],
            feed="issued-history", source_period=OBSERVED, observed_on=OBSERVED,
        )


def test_invalid_ticker_or_missing_event_date_is_skipped() -> None:
    assert normalize_capital_actions(
        [_issued(code="BAD!"), _issued(id="two", listingDate=None)],
        feed="issued-history", source_period=OBSERVED, observed_on=OBSERVED,
    ) == []


def test_monthly_rows_outside_source_month_are_skipped() -> None:
    item = {"code": "BBCA", "startDate": "2026-07-01", "lastDate": "2026-07-31", "shares": 100}
    with pytest.raises(RuntimeError, match="WRONG_PERIOD"):
        normalize_capital_actions(
            [item], feed="additional-listings", source_period=PERIOD, observed_on=OBSERVED
        )


def test_validation_rejects_inconsistent_or_wrong_period_rows() -> None:
    row = normalize_capital_actions([_issued()], feed="issued-history", source_period=OBSERVED, observed_on=OBSERVED)[0]
    assert validate_capital_action_rows([row], feed="issued-history", source_period=OBSERVED, observed_on=OBSERVED) == (True, "VALID")
    bad = dict(row, delta_shares=999)
    assert validate_capital_action_rows([bad], feed="issued-history", source_period=OBSERVED, observed_on=OBSERVED)[1] == "PARSE_FAILURE"
    wrong = dict(row, observed_on="2026-08-31")
    assert validate_capital_action_rows([wrong], feed="issued-history", source_period=OBSERVED, observed_on=OBSERVED)[1] == "WRONG_PERIOD"
    negative = dict(row, pre_shares=-1, post_shares=3_199)
    assert validate_capital_action_rows([negative], feed="issued-history", source_period=OBSERVED, observed_on=OBSERVED)[1] == "CONTEXT_REJECTED"


def test_wrapped_zapi_content_envelope_is_supported() -> None:
    session = Session([Response({"content": _issued_payload([_issued()]), "message": "ok", "errors": None})])
    rows, meta = _producer(MemoryBackend(), session).get_issued_history(OBSERVED)
    assert len(rows) == 1 and meta["state"] == "REFRESHED"


def test_global_issued_history_uses_page_limit_pagination_without_ticker_calls() -> None:
    first = _issued_payload([_issued(id="one", code="BBCA")], total=501)
    second = _issued_payload([_issued(id="two", code="BBRI")], start=500, total=501)
    session = Session([Response(first), Response(second)])
    rows, meta = _producer(MemoryBackend(), session).get_issued_history(OBSERVED)
    assert {row["ticker"] for row in rows} == {"BBCA", "BBRI"}
    assert meta["api_calls"] == 2 and len(session.calls) == 2
    assert [call["params"] for call in session.calls] == [
        {"page": 1, "limit": ISSUED_HISTORY_LENGTH},
        {"page": 2, "limit": ISSUED_HISTORY_LENGTH},
    ]
    assert all("code" not in call["params"] for call in session.calls)


@pytest.mark.parametrize("feed", ["additional-listings", "rights-offerings", "stock-splits"])
def test_monthly_feeds_use_one_global_month_call(feed: str) -> None:
    item = {"id": "x", "code": "BBCA", "eventDate": "2026-08-12"}
    if feed == "additional-listings":
        item.pop("eventDate", None)
        item.update({"startDate": "2026-08-01", "lastDate": "2026-08-12", "shares": 100})
    session = Session([Response(_monthly(feed, [item]))])
    rows, meta = _producer(MemoryBackend(), session).get_month(2026, 8, feed=feed, observed_on=OBSERVED)
    assert len(rows) == 1 and meta["api_calls"] == 1
    assert session.calls[0]["url"] == FEEDS[feed]["url"]
    assert session.calls[0]["params"] == {"year": 2026, "month": 8, "page": 1, "length": PAGE_LENGTH}
    assert "search" not in session.calls[0]["params"]


def test_monthly_pagination_follows_explicit_has_more() -> None:
    one = {"id": "one", "code": "BBCA", "startDate": "2026-08-01", "lastDate": "2026-08-01", "shares": 100}
    two = {"id": "two", "code": "BBRI", "startDate": "2026-08-02", "lastDate": "2026-08-02", "shares": 200}
    session = Session([
        Response(_monthly("additional-listings", [one], has_more=True)),
        Response(_monthly("additional-listings", [two], page=2, has_more=False)),
    ])
    rows, meta = _producer(MemoryBackend(), session).get_month(
        2026, 8, feed="additional-listings", observed_on=OBSERVED
    )
    assert len(rows) == 2 and meta["pages"] == 2


@pytest.mark.parametrize("first,second", [("PASTICUAN", "EMIR"), ("EMIR", "PASTICUAN")])
def test_second_scanner_reuses_global_month_without_zapi_key(first: str, second: str) -> None:
    backend = MemoryBackend()
    item = {"id": "x", "code": "BBCA", "startDate": "2026-08-01", "lastDate": "2026-08-12", "shares": 100}
    first_rows, _ = _producer(backend, Session([Response(_monthly("additional-listings", [item]))]), client=first).get_month(
        2026, 8, feed="additional-listings", observed_on=OBSERVED
    )
    second_session = Session([])
    second_rows, meta = _producer(backend, second_session, client=second, api_key="").get_month(
        2026, 8, feed="additional-listings", observed_on=OBSERVED
    )
    assert first_rows == second_rows and not second_session.calls
    assert meta["cache_hit"] and meta["request_avoided"] and meta["api_calls"] == 0


def test_missing_key_on_cache_miss_makes_no_request() -> None:
    session = Session([])
    rows, meta = _producer(MemoryBackend(), session, api_key="").get_issued_history(OBSERVED)
    assert not rows and not session.calls and meta["state"] == "ENVIRONMENT_BLOCKED"


@pytest.mark.parametrize("status", [401, 403, 404, 429])
def test_http_failures_are_explicit(status: int) -> None:
    rows, meta = _producer(MemoryBackend(), Session([Response(status=status)])).get_issued_history(OBSERVED)
    assert not rows and meta["state"] == f"HTTP_{status}"


@pytest.mark.parametrize(
    "outcome,reason",
    [(requests.Timeout(), "TIMEOUT"), (requests.ConnectionError(), "CONNECTION_ERROR")],
)
def test_network_failures_are_explicit(outcome: Exception, reason: str) -> None:
    rows, meta = _producer(MemoryBackend(), Session([outcome])).get_issued_history(OBSERVED)
    assert not rows and meta["state"] == reason


@pytest.mark.parametrize(
    "response,reason",
    [
        (Response(content=b""), "EMPTY_RESPONSE"),
        (Response(malformed=True), "PARSE_FAILURE"),
        (Response({"items": [], "dataset": "wrong", "provider": "idx"}), "CONTEXT_REJECTED"),
        (Response({"items": {}, "dataset": "issued-history", "provider": "idx"}), "PARSE_FAILURE"),
    ],
)
def test_bad_responses_fail_closed(response: Response, reason: str) -> None:
    rows, meta = _producer(MemoryBackend(), Session([response])).get_issued_history(OBSERVED)
    assert not rows and meta["state"] == reason


def test_empty_valid_feed_is_explicit_valid_empty() -> None:
    rows, meta = _producer(MemoryBackend(), Session([Response(_issued_payload([]))])).get_issued_history(OBSERVED)
    assert not rows and meta["state"] == "REFRESHED_EMPTY"
    assert meta["bounded_complete"] is True
    assert meta["provider_rows"] == 0


def test_invalid_feed_and_month_make_no_request() -> None:
    session = Session([])
    producer = _producer(MemoryBackend(), session)
    assert producer.get_month(2026, 13)[1]["state"] == "WRONG_PERIOD"
    assert producer.get_month(2026, 8, feed="issued-history")[1]["state"] == "CONTEXT_REJECTED"
    assert not session.calls


def test_migration_contract_and_module_remain_scanner_neutral() -> None:
    root = Path(__file__).resolve().parents[1]
    migration_path = next((root / "database").glob("migration_v*_shared_evidence_hub.sql"))
    migration = migration_path.read_text()
    module = (root / "shared_capital_action_evidence.py").read_text()
    for column in (
        "source_feed text", "source_period date", "observed_on date", "ratio_before numeric",
        "ratio_after numeric", "raw_action text", "calculation_state text", "payload_hash text",
        "source_verified boolean",
    ):
        assert column in migration
    assert "grant select, insert, update on table public.%I to service_role" in migration
    assert "enable row level security" in migration
    lowered = module.lower()
    assert "score" not in lowered and "ranking" not in lowered and "production_gate" not in lowered


def test_request_is_bounded_and_redirects_are_disabled() -> None:
    session = Session([Response(_issued_payload([_issued()]))])
    rows, meta = _producer(MemoryBackend(), session).get_issued_history(OBSERVED)
    assert rows and meta["state"] == "REFRESHED"
    call = session.calls[0]
    assert call["timeout"] == REQUEST_TIMEOUT_SECONDS
    assert call["allow_redirects"] is False
    assert call["headers"]["User-Agent"] == "Shared-IDX-Evidence-Hub/capital-actions"


def test_redirect_response_is_rejected_and_attempt_is_counted() -> None:
    rows, meta = _producer(MemoryBackend(), Session([Response(status=302)])).get_issued_history(OBSERVED)
    assert not rows
    assert meta["state"] == "CONTEXT_REJECTED"
    assert meta["api_calls"] == 1


def test_page_budget_exhaustion_never_persists_partial_month() -> None:
    backend = MemoryBackend()
    item = {
        "id": "rights-1",
        "code": "BBRI",
        "eventDate": "2026-08-20",
        "publicationDate": "2026-08-05",
        "ratioBefore": 4,
        "ratioAfter": 1,
    }
    session = Session([Response(_monthly("rights-offerings", [item], has_more=True))])
    rows, meta = _producer(backend, session).get_month(
        2026, 8, feed="rights-offerings", observed_on=OBSERVED, max_pages=1
    )
    assert not rows and backend.rows == []
    assert meta["state"] == "CONTEXT_REJECTED"
    assert meta["page_budget"] == 1 and meta["api_calls"] == 1
    assert meta["bounded_complete"] is False


def test_empty_page_with_has_more_is_parse_failure_not_partial_success() -> None:
    backend = MemoryBackend()
    session = Session([Response(_monthly("rights-offerings", [], total=2, has_more=True))])
    rows, meta = _producer(backend, session).get_month(
        2026, 8, feed="rights-offerings", observed_on=OBSERVED, max_pages=1
    )
    assert not rows and backend.rows == []
    assert meta["state"] == "PARSE_FAILURE"
    assert meta["api_calls"] == 1


def test_conflicting_explicit_event_dates_fail_closed() -> None:
    item = _issued(effectiveDate="2026-08-13")
    with pytest.raises(RuntimeError, match="CONTEXT_REJECTED"):
        normalize_capital_actions(
            [item], feed="issued-history", source_period=OBSERVED, observed_on=OBSERVED
        )


def test_free_text_action_does_not_refine_event_identity_by_substring() -> None:
    item = {
        "id": "split-free-text",
        "code": "BBCA",
        "effectiveDate": "2026-08-10",
        "action": "Proposed Reverse Stock Split Plan",
    }
    row = normalize_capital_actions(
        [item], feed="stock-splits", source_period=PERIOD, observed_on=OBSERVED
    )[0]
    assert row["event_type"] == "STOCK_SPLIT"


def test_validation_rejects_nonofficial_source_url_and_future_publication_date() -> None:
    row = normalize_capital_actions(
        [_issued()], feed="issued-history", source_period=OBSERVED, observed_on=OBSERVED
    )[0]
    bad_url = dict(row, source_url="https://evil.example/action")
    assert validate_capital_action_rows(
        [bad_url], feed="issued-history", source_period=OBSERVED, observed_on=OBSERVED
    )[1] == "CONTEXT_REJECTED"

    future_publication = dict(row, publication_date="2026-09-02")
    assert validate_capital_action_rows(
        [future_publication], feed="issued-history", source_period=OBSERVED, observed_on=OBSERVED
    )[1] == "WRONG_PERIOD"


def test_default_page_budget_is_bounded() -> None:
    assert 1 <= MAX_PAGES_PER_RUN <= 10


def test_valid_empty_month_is_cached_as_fact_without_duplicate_request() -> None:
    backend = MemoryBackend()
    first_session = Session([Response(_monthly("rights-offerings", [], has_more=False))])
    first_rows, first_meta = _producer(backend, first_session, client="PASTICUAN").get_month(
        2026, 8, feed="rights-offerings", observed_on=OBSERVED
    )
    assert first_rows == []
    assert first_meta["state"] == "REFRESHED_EMPTY"
    assert first_meta["bounded_complete"] is True
    assert first_meta["provider_rows"] == 0
    assert first_meta["api_calls"] == 1
    assert backend.provider_states[-1]["response_state"] == "VALID_EMPTY"
    assert backend.provider_states[-1]["error_classification"] is None

    second_session = Session([])
    second_rows, second_meta = _producer(
        backend, second_session, client="EMIR", api_key=""
    ).get_month(2026, 8, feed="rights-offerings", observed_on=OBSERVED)
    assert second_rows == []
    assert not second_session.calls
    assert second_meta["state"] == "CACHE_HIT_EMPTY"
    assert second_meta["cache_hit"] and second_meta["request_avoided"]
    assert second_meta["api_calls"] == 0


def test_nonempty_provider_page_that_normalizes_to_zero_fails_closed() -> None:
    backend = MemoryBackend()
    out_of_period = {
        "id": "rights-wrong-period",
        "code": "BBRI",
        "eventDate": "2026-07-31",
        "publicationDate": "2026-07-15",
    }
    session = Session([
        Response(_monthly("rights-offerings", [out_of_period], has_more=False))
    ])
    rows, meta = _producer(backend, session).get_month(
        2026, 8, feed="rights-offerings", observed_on=OBSERVED
    )
    assert rows == [] and backend.rows == []
    assert meta["provider_rows"] == 1
    assert meta["state"] == "CONTEXT_REJECTED"


def test_additional_listing_requires_both_span_dates() -> None:
    missing_start = {
        "code": "BBCA", "lastDate": "2026-08-12", "shares": 100, "actionType": "ESOP"
    }
    missing_end = {
        "code": "BBCA", "startDate": "2026-08-01", "shares": 100, "actionType": "ESOP"
    }
    for item in (missing_start, missing_end):
        with pytest.raises(RuntimeError, match="CONTEXT_REJECTED"):
            normalize_capital_actions(
                [item], feed="additional-listings", source_period=PERIOD, observed_on=OBSERVED
            )


def test_additional_listing_rejects_reversed_or_cross_period_span() -> None:
    reversed_span = {
        "code": "BBCA", "startDate": "2026-08-12", "lastDate": "2026-08-01", "shares": 100
    }
    with pytest.raises(RuntimeError, match="CONTEXT_REJECTED"):
        normalize_capital_actions(
            [reversed_span], feed="additional-listings", source_period=PERIOD, observed_on=OBSERVED
        )

    cross_period = {
        "code": "BBCA", "startDate": "2026-07-31", "lastDate": "2026-08-01", "shares": 100
    }
    with pytest.raises(RuntimeError, match="WRONG_PERIOD"):
        normalize_capital_actions(
            [cross_period], feed="additional-listings", source_period=PERIOD, observed_on=OBSERVED
        )


def test_point_capital_actions_do_not_invent_date_span() -> None:
    row = normalize_capital_actions(
        [_issued()], feed="issued-history", source_period=OBSERVED, observed_on=OBSERVED
    )[0]
    assert row["event_date_kind"] == "POINT"
    assert row["event_start_date"] is None
    assert row["event_end_date"] is None


def test_validation_rejects_tampered_additional_listing_span() -> None:
    item = {
        "code": "BBCA",
        "startDate": "2026-08-01",
        "lastDate": "2026-08-12",
        "shares": 100,
        "actionType": "Warrant",
    }
    row = normalize_capital_actions(
        [item], feed="additional-listings", source_period=PERIOD, observed_on=OBSERVED
    )[0]
    assert validate_capital_action_rows(
        [row], feed="additional-listings", source_period=PERIOD, observed_on=OBSERVED
    ) == (True, "VALID")

    bad_end = dict(row, event_end_date="2026-08-11")
    assert validate_capital_action_rows(
        [bad_end], feed="additional-listings", source_period=PERIOD, observed_on=OBSERVED
    )[1] == "CONTEXT_REJECTED"

    bad_kind = dict(row, event_date_kind="POINT")
    assert validate_capital_action_rows(
        [bad_kind], feed="additional-listings", source_period=PERIOD, observed_on=OBSERVED
    )[1] == "CONTEXT_REJECTED"


def test_issued_history_page_limit_envelope_fallback_is_supported() -> None:
    first = {
        "items": [_issued(id="one", code="BBCA")],
        "page": 1,
        "limit": ISSUED_HISTORY_LENGTH,
        "total": ISSUED_HISTORY_LENGTH + 1,
        "dataset": "issued-history",
        "provider": "idx",
    }
    second = {
        "items": [_issued(id="two", code="BBRI")],
        "page": 2,
        "limit": ISSUED_HISTORY_LENGTH,
        "total": ISSUED_HISTORY_LENGTH + 1,
        "dataset": "issued-history",
        "provider": "idx",
    }
    session = Session([Response(first), Response(second)])
    rows, meta = _producer(MemoryBackend(), session).get_issued_history(OBSERVED)
    assert {row["ticker"] for row in rows} == {"BBCA", "BBRI"}
    assert meta["api_calls"] == 2


def test_failed_issued_history_reports_categorical_validation_detail() -> None:
    bad = _issued(id="bad", code="BBCA", shares=None, deltaShares=2_000, sharesAfter=1_000)
    session = Session([Response(_issued_payload([bad]))])
    rows, meta = _producer(MemoryBackend(), session).get_issued_history(OBSERVED)
    assert rows == []
    assert meta["state"] == "CONTEXT_REJECTED"
    assert meta["validation_valid_rows"] == 0
    assert meta["validation_failure_counts"] == {"CONTEXT_REJECTED": 1}
    assert meta["validation_failure_detail_counts"] == {"PRE_SHARES_NEGATIVE": 1}
    assert meta["validation_failure_action_counts"] == {"waran": 1}
    assert meta["validation_failure_calculation_state_counts"] == {"EXPLICIT_DELTA_POST_DERIVED_PRE": 1}
    assert meta["validation_failure_share_relation_counts"] == {"DELTA_GT_POST": 1}


def test_issued_history_generic_shares_can_exceed_post_without_inventing_negative_pre() -> None:
    item = _issued(
        id="legacy",
        code="BBCA",
        action="Delisting",
        shares=2_000,
        sharesAfter=1_000,
        listingDate="2009-01-01",
        publicationDate=None,
    )
    row = normalize_capital_actions(
        [item], feed="issued-history", source_period=OBSERVED, observed_on=OBSERVED
    )[0]
    assert row["event_shares"] == 2_000
    assert row["post_shares"] == 1_000
    assert row["pre_shares"] is None
    assert row["delta_shares"] is None
    assert row["calculation_state"] == "EXPLICIT_EVENT_SHARES_POST_NO_DELTA"
    assert validate_capital_action_rows(
        [row], feed="issued-history", source_period=OBSERVED, observed_on=OBSERVED
    ) == (True, "VALID")


def test_issued_history_signed_event_shares_are_factual_not_invalid_delta() -> None:
    item = _issued(
        id="partial-delisting",
        code="BBCA",
        action="Partial Delisting",
        shares=-5_516_000,
        sharesAfter=24_655_010_000,
        listingDate="2008-01-04",
        publicationDate=None,
    )
    row = normalize_capital_actions(
        [item], feed="issued-history", source_period=OBSERVED, observed_on=OBSERVED
    )[0]
    assert row["event_shares"] == -5_516_000
    assert row["post_shares"] == 24_655_010_000
    assert row["pre_shares"] is None and row["delta_shares"] is None
    assert validate_capital_action_rows(
        [row], feed="issued-history", source_period=OBSERVED, observed_on=OBSERVED
    ) == (True, "VALID")


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("Reverse Stock", "REVERSE_STOCK_SPLIT"),
        ("Delisting", "DELISTING"),
        ("Partial Delisting", "PARTIAL_DELISTING"),
        ("IPO", "IPO"),
        ("Private Placement", "PRIVATE_PLACEMENT"),
        ("Transaksi Material", "MATERIAL_TRANSACTION"),
    ],
)
def test_explicit_issued_history_action_labels_normalize_without_title_inference(
    action: str, expected: str
) -> None:
    row = normalize_capital_actions(
        [_issued(action=action)],
        feed="issued-history",
        source_period=OBSERVED,
        observed_on=OBSERVED,
    )[0]
    assert row["event_type"] == expected
    assert row["raw_action"] == action
