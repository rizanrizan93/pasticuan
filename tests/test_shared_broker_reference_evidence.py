from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import threading
from typing import Any, Mapping

import pytest
import requests

from shared_broker_reference_evidence import (
    BROKERS_URL,
    BROKER_SUMMARY_URL,
    EVIDENCE_SCOPE,
    MARKET_PROVIDER,
    MEMBER_PROVIDER,
    SUMMARY_LENGTH,
    SharedBrokerReferenceEvidence,
    normalize_exchange_members,
    normalize_market_summary,
    validate_market_rows,
    validate_member_rows,
)
from shared_evidence_hub import EvidenceKey, SharedEvidenceCoordinator


DAY = date(2026, 8, 21)
OBSERVED = date(2026, 9, 1)


def _member(code: str = "YU", **changes: Any) -> dict[str, Any]:
    row = {
        "code": code,
        "name": "CGS INTERNATIONAL SEKURITAS INDONESIA",
        "mkbd": 944_675_696_954.63,
        "license": "APERD, DMA, Margin, Online",
        "website": "www.cgsi.co.id",
        "city": "Jakarta Selatan",
        "logo": "https://www.idx.co.id/StaticData/Brokers/Logo/YU.jpg",
        "category": "patungan",
        "branchCount": 27,
        "memberStatus": "Aktif",
        "shareholders": [
            {"name": "Foreign Securities Pte Ltd", "type": "Badan Hukum", "country": "Singapura", "sharePct": 90.04, "ownership": "asing"},
            {"name": "PT Lokal", "type": "Badan Hukum", "country": "Indonesia", "sharePct": 9.96, "ownership": "lokal"},
        ],
        "paidUpCapital": 356_000_000_000,
        "foreignOwnershipPct": 90.04,
    }
    row.update(changes)
    return row


def _members_payload(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {"items": list(rows), "count": len(rows), "dataset": "brokers", "provider": "idx"}


def _summary(code: str = "YU", **changes: Any) -> dict[str, Any]:
    row = {
        "No": 1,
        "Date": "2026-08-21T00:00:00",
        "Value": 198_130_186_500,
        "IDFirm": code,
        "Volume": 296_854_100,
        "FirmName": "CGS International Sekuritas Indonesia",
        "Frequency": 22_106,
        "IDBrokerSummary": 957_289,
    }
    row.update(changes)
    return row


def _summary_payload(rows: list[Mapping[str, Any]], *, start: int = 0, total: int | None = None) -> dict[str, Any]:
    return {
        "data": list(rows), "start": start, "length": SUMMARY_LENGTH,
        "recordsTotal": len(rows) if total is None else total,
        "dataset": "broker-summary", "provider": "idx",
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
        self.tables: dict[str, list[dict[str, Any]]] = {
            "evidence_brokers": [], "evidence_broker_market_daily": [],
        }
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
        return [
            dict(row) for row in self.tables[table]
            if all(str(row.get(key)) == str(value) for key, value in filters.items())
        ][:limit]

    def upsert_rows(self, table: str, rows: list[Mapping[str, Any]], *, conflict: tuple[str, ...]) -> list[dict[str, Any]]:
        keyed = {tuple(row.get(key) for key in conflict): dict(row) for row in self.tables[table]}
        for row in rows:
            keyed[tuple(row.get(key) for key in conflict)] = dict(row)
        self.tables[table] = list(keyed.values())
        return [dict(row) for row in rows]


def _producer(
    backend: MemoryBackend,
    session: Session,
    *,
    client: str = "PASTICUAN",
    api_key: str = "fixture-key",
) -> SharedBrokerReferenceEvidence:
    coordinator = SharedEvidenceCoordinator(backend, client_id=client, worker_id=f"{client}-worker")
    return SharedBrokerReferenceEvidence(
        client, backend=backend, coordinator=coordinator, session=session, api_key=api_key
    )


def test_member_normalization_preserves_only_reported_reference_facts() -> None:
    row = normalize_exchange_members(_members_payload([_member()]), observed_on=OBSERVED)[0]
    assert row["broker_code"] == "YU" and row["member_status"] == "Aktif"
    assert row["ownership_category"] == "patungan" and row["foreign_ownership_percentage"] == 90.04
    assert row["paid_up_capital"] == 356_000_000_000 and row["mkbd"] == 944_675_696_954.63
    assert row["branch_count"] == 27 and row["evidence_scope"] == EVIDENCE_SCOPE
    assert row["profile"]["shareholders"][0]["reported_ownership"] in {"asing", "lokal"}


def test_member_profile_does_not_infer_unreported_classification() -> None:
    row = normalize_exchange_members(
        _members_payload([_member(category=None, foreignOwnershipPct=None, shareholders=[])]),
        observed_on=OBSERVED,
    )[0]
    assert row["ownership_category"] is None and row["foreign_ownership_percentage"] is None
    assert row["profile"]["shareholders"] == []


def test_single_profile_envelope_is_supported() -> None:
    payload = {**_member(), "dataset": "brokers", "provider": "idx"}
    rows = normalize_exchange_members(payload, observed_on=OBSERVED)
    assert [row["broker_code"] for row in rows] == ["YU"]


def test_member_change_state_is_deterministic() -> None:
    first = normalize_exchange_members(_members_payload([_member()]), observed_on=OBSERVED)[0]
    same = normalize_exchange_members(_members_payload([_member()]), observed_on=OBSERVED, previous=[first])[0]
    changed = normalize_exchange_members(
        _members_payload([_member(memberStatus="Suspended")]), observed_on=OBSERVED, previous=[first]
    )[0]
    assert first["change_state"] == "NEW" and same["change_state"] == "UNCHANGED"
    assert changed["change_state"] == "CHANGED" and first["payload_hash"] != changed["payload_hash"]


def test_identical_member_duplicates_deduplicate_but_conflicts_fail() -> None:
    rows = normalize_exchange_members(_members_payload([_member(), _member()]), observed_on=OBSERVED)
    assert len(rows) == 1
    with pytest.raises(RuntimeError, match="PARSE_FAILURE"):
        normalize_exchange_members(
            _members_payload([_member(), _member(memberStatus="Suspended")]), observed_on=OBSERVED
        )


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"code": "BAD"}, "PARSE_FAILURE"),
        ({"foreignOwnershipPct": 101}, "CONTEXT_REJECTED"),
        ({"mkbd": -1}, "CONTEXT_REJECTED"),
        ({"branchCount": 1.5}, "CONTEXT_REJECTED"),
        ({"paidUpCapital": "not-number"}, "PARSE_FAILURE"),
        ({"shareholders": {"name": "bad"}}, "PARSE_FAILURE"),
        ({"shareholders": [{"name": "A", "ownership": "unknown"}]}, "CONTEXT_REJECTED"),
    ],
)
def test_invalid_member_facts_fail_closed(changes: dict[str, Any], reason: str) -> None:
    with pytest.raises(RuntimeError, match=reason):
        normalize_exchange_members(_members_payload([_member(**changes)]), observed_on=OBSERVED)


def test_market_summary_is_explicitly_market_wide_and_ticker_free() -> None:
    row = normalize_market_summary([_summary()], activity_date=DAY)[0]
    assert row["broker_code"] == "YU" and row["activity_date"] == DAY.isoformat()
    assert row["traded_value"] == 198_130_186_500 and row["traded_volume"] == 296_854_100
    assert row["frequency"] == 22_106 and row["evidence_scope"] == EVIDENCE_SCOPE
    assert not {"ticker", "stock_code", "issuer"}.intersection(row)


def test_market_summary_wrong_date_and_negative_values_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="WRONG_PERIOD"):
        normalize_market_summary([_summary(Date="2026-08-20")], activity_date=DAY)
    with pytest.raises(RuntimeError, match="CONTEXT_REJECTED"):
        normalize_market_summary([_summary(Value=-1)], activity_date=DAY)


def test_market_summary_duplicate_identity_is_deterministic() -> None:
    assert len(normalize_market_summary([_summary(), _summary()], activity_date=DAY)) == 1
    with pytest.raises(RuntimeError, match="PARSE_FAILURE"):
        normalize_market_summary([_summary(), _summary(Value=1)], activity_date=DAY)


def test_validators_reject_scope_or_ticker_contamination() -> None:
    member = normalize_exchange_members(_members_payload([_member()]), observed_on=OBSERVED)[0]
    market = normalize_market_summary([_summary()], activity_date=DAY)[0]
    assert validate_member_rows([member]) == (True, "VALID")
    assert validate_market_rows([market], activity_date=DAY) == (True, "VALID")
    assert validate_member_rows([{**member, "evidence_scope": "TICKER"}])[1] == "CONTEXT_REJECTED"
    assert validate_market_rows([{**market, "ticker": "BBCA"}], activity_date=DAY)[1] == "CONTEXT_REJECTED"


def test_member_fetch_is_one_global_full_view_call() -> None:
    session = Session([Response(_members_payload([_member(), _member(code="YP", name="Mirae")]))])
    rows, meta = _producer(MemoryBackend(), session).get_members(OBSERVED)
    assert {row["broker_code"] for row in rows} == {"YU", "YP"}
    assert meta["api_calls"] == 1 and session.calls[0]["url"] == BROKERS_URL
    assert session.calls[0]["params"] == {"view": "full"} and "code" not in session.calls[0]["params"]


def test_market_fetch_uses_explicit_date_and_no_ticker_parameter() -> None:
    session = Session([Response(_summary_payload([_summary()]))])
    rows, meta = _producer(MemoryBackend(), session).get_market_summary(DAY)
    assert len(rows) == 1 and meta["api_calls"] == 1
    assert session.calls[0]["url"] == BROKER_SUMMARY_URL
    assert session.calls[0]["params"] == {"length": SUMMARY_LENGTH, "start": 0, "date": DAY.isoformat()}


def test_market_pagination_is_bounded_and_complete() -> None:
    session = Session([
        Response(_summary_payload([_summary()], total=SUMMARY_LENGTH + 1)),
        Response(_summary_payload([_summary(code="YP", IDBrokerSummary=957290)], start=SUMMARY_LENGTH, total=SUMMARY_LENGTH + 1)),
    ])
    rows, meta = _producer(MemoryBackend(), session).get_market_summary(DAY)
    assert {row["broker_code"] for row in rows} == {"YU", "YP"}
    assert meta["pages"] == 2 and session.calls[1]["params"]["start"] == SUMMARY_LENGTH


def test_truncated_market_pagination_is_not_persisted() -> None:
    backend = MemoryBackend()
    session = Session([Response(_summary_payload([_summary()], total=SUMMARY_LENGTH + 1))])
    rows, meta = _producer(backend, session).get_market_summary(DAY, max_pages=1)
    assert rows == [] and meta["state"] == "INSUFFICIENT_HISTORY"
    assert backend.tables["evidence_broker_market_daily"] == []


@pytest.mark.parametrize("first,second", [("EMIR", "PASTICUAN"), ("PASTICUAN", "EMIR")])
def test_one_member_fetch_serves_both_scanners(first: str, second: str) -> None:
    backend = MemoryBackend()
    first_session = Session([Response(_members_payload([_member()]))])
    first_rows, first_meta = _producer(backend, first_session, client=first).get_members(OBSERVED)
    second_session = Session([])
    second_rows, second_meta = _producer(backend, second_session, client=second, api_key="").get_members(OBSERVED)
    assert first_rows == second_rows and first_meta["state"] == "REFRESHED"
    assert second_meta["state"] == "CACHE_HIT" and second_meta["request_avoided"]
    assert len(first_session.calls) == 1 and second_session.calls == []


@pytest.mark.parametrize("first,second", [("EMIR", "PASTICUAN"), ("PASTICUAN", "EMIR")])
def test_one_market_fetch_serves_both_scanners(first: str, second: str) -> None:
    backend = MemoryBackend()
    first_session = Session([Response(_summary_payload([_summary()]))])
    first_rows, _ = _producer(backend, first_session, client=first).get_market_summary(DAY)
    second_session = Session([])
    second_rows, meta = _producer(backend, second_session, client=second, api_key="").get_market_summary(DAY)
    assert first_rows == second_rows and meta["state"] == "CACHE_HIT" and meta["request_avoided"]
    assert second_session.calls == []


def test_stale_member_directory_refreshes_and_marks_change() -> None:
    backend = MemoryBackend()
    stale = normalize_exchange_members(
        _members_payload([_member()]), observed_on=date(2026, 7, 1),
        fetched_at=datetime.now(timezone.utc) - timedelta(days=31),
    )[0]
    backend.tables["evidence_brokers"] = [stale]
    session = Session([Response(_members_payload([_member(memberStatus="Suspended")]))])
    rows, meta = _producer(backend, session).get_members(OBSERVED)
    assert meta["state"] == "REFRESHED" and meta["api_calls"] == 1
    assert rows[0]["change_state"] == "CHANGED" and rows[0]["member_status"] == "Suspended"


@pytest.mark.parametrize("status", [401, 403, 404, 429])
def test_http_failure_taxonomy(status: int) -> None:
    _, meta = _producer(MemoryBackend(), Session([Response(status=status)])).get_members(OBSERVED)
    assert meta["state"] == f"HTTP_{status}"


@pytest.mark.parametrize(
    "outcome,reason",
    [(requests.Timeout("slow"), "TIMEOUT"), (requests.ConnectionError("offline"), "CONNECTION_ERROR")],
)
def test_transport_failure_taxonomy(outcome: Exception, reason: str) -> None:
    _, meta = _producer(MemoryBackend(), Session([outcome])).get_members(OBSERVED)
    assert meta["state"] == reason


@pytest.mark.parametrize(
    "response,reason",
    [
        (Response(content=b""), "EMPTY_RESPONSE"),
        (Response(malformed=True), "PARSE_FAILURE"),
        (Response({"items": [], "dataset": "companies", "provider": "idx"}), "CONTEXT_REJECTED"),
        (Response(_members_payload([])), "NO_REPORT"),
    ],
)
def test_payload_failure_taxonomy(response: Response, reason: str) -> None:
    _, meta = _producer(MemoryBackend(), Session([response])).get_members(OBSERVED)
    assert meta["state"] == reason


def test_missing_runtime_configuration_stays_inert() -> None:
    producer = SharedBrokerReferenceEvidence("EMIR", backend=None, coordinator=None, api_key="")
    producer.backend = None
    producer.coordinator = None
    assert producer.get_members(OBSERVED)[1]["state"] == "ENVIRONMENT_BLOCKED"
    assert producer.get_market_summary(DAY)[1]["state"] == "ENVIRONMENT_BLOCKED"


def test_migration_has_separate_market_table_rls_grants_and_index() -> None:
    root = Path(__file__).resolve().parents[1]
    migration = next((root / "database").glob("migration_v*_shared_evidence_hub.sql")).read_text()
    verification = next((root / "database").glob("verify_v*_shared_evidence_hub.sql")).read_text()
    assert "create table if not exists public.evidence_broker_market_daily" in migration
    assert "primary key (provider, activity_date, broker_code)" in migration
    assert "evidence_scope = 'MARKET_WIDE'" in migration
    assert "'evidence_broker_market_daily'" in migration
    assert "evidence_broker_market_daily_lookup_idx" in migration
    assert "broker_market_daily_service_role" in verification
    assert "broker_market_daily_denied_anon" in verification


def test_producer_has_no_scanner_scoring_or_ticker_activity_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "shared_broker_reference_evidence.py").read_text().lower()
    assert "score" not in source and "ranking" not in source and "recommendation" not in source
    assert "evidence_participant_flow" not in source
    assert "ticker" not in MARKET_TABLE_COLUMNS


MARKET_TABLE_COLUMNS = {
    "provider", "activity_date", "broker_code", "broker_name", "traded_value", "traded_volume",
    "frequency", "source_event_id", "evidence_scope", "source_url", "payload_hash",
    "source_verified", "validation_state", "fetched_at",
}
