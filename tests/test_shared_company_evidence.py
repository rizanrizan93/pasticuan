from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
import threading
from typing import Any, Mapping

import pytest
import requests

from shared_company_evidence import (
    BULK_LENGTH,
    COMPANIES_URL,
    PROFILE_URL,
    REFERENCE_URL,
    SECURITIES_URL,
    SharedCompanyEvidence,
    normalize_company_directory,
    normalize_company_profile,
    normalize_reference_values,
    validate_company_rows,
)
from shared_evidence_hub import EvidenceKey, SharedEvidenceCoordinator


OBSERVED = date(2026, 9, 1)


def _company(code: str = "BBCA", **changes: Any) -> dict[str, Any]:
    row = {
        "KodeEmiten": code, "NamaEmiten": "PT Bank Central Asia Tbk", "Sektor": "Keuangan",
        "SubSektor": "Bank", "Industri": "Bank", "SubIndustri": "Bank",
        "PapanPencatatan": "Utama", "TanggalPencatatan": "2000-05-31T00:00:00",
        "KegiatanUsahaUtama": "Jasa Perbankan", "BAE": "PT Registra",
        "Website": "www.bca.co.id", "Alamat": "Jakarta", "EfekEmiten_Saham": True,
        "EfekEmiten_Obligasi": True, "EfekEmiten_ETF": False,
        "EfekEmiten_EBA": False, "EfekEmiten_SPEI": False,
    }
    row.update(changes)
    return row


def _security(code: str = "BBCA", **changes: Any) -> dict[str, Any]:
    row = {
        "Code": code, "Name": "Bank Central Asia Tbk.", "Shares": 122_042_299_500,
        "ListingDate": "2000-05-31T00:00:00", "ListingBoard": "Utama",
    }
    row.update(changes)
    return row


def _profile(code: str = "BBCA", **changes: Any) -> dict[str, Any]:
    row = {
        "code": code, "name": "PT Bank Central Asia Tbk.", "sector": "Keuangan",
        "industry": "Bank", "subSector": "Bank", "subIndustry": "Bank",
        "listingDate": "2000-05-31", "listingBoard": "Utama", "mainBusiness": "Jasa Perbankan",
        "website": "www.bca.co.id", "address": "Jakarta",
        "directors": [{"name": "Direktur A", "title": "PRESIDEN DIREKTUR"}],
        "commissioners": [{"name": "Komisaris A", "title": "KOMISARIS"}],
        "auditCommittee": [{"name": "Komite A", "title": "KETUA"}],
        "corporateSecretary": [{"name": "Sekretaris A", "email": "not-persisted@example.com"}],
        "shareholders": [{"name": "Pemegang A", "shares": 1000, "sharePct": 55.0, "category": "Lebih dari 5%"}],
        "subsidiaries": [{
            "name": "Anak A", "business": "Asuransi", "location": "Jakarta",
            "ownershipPct": 90, "operatingStatus": "Beroperasi", "commercialYear": "2014",
            "totalAssets": 100, "unit": "JUTAAN", "currency": "IDR",
        }],
        "dataset": "company-profile", "provider": "idx",
    }
    row.update(changes)
    return row


def _offset_payload(dataset: str, rows: list[Mapping[str, Any]], *, start: int = 0, total: int | None = None) -> dict[str, Any]:
    return {
        "data": list(rows), "start": start, "length": BULK_LENGTH,
        "recordsTotal": len(rows) if total is None else total,
        "recordsFiltered": len(rows) if total is None else total,
        "dataset": dataset, "provider": "idx",
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
            "evidence_companies": [], "evidence_reference_values": [],
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


def _producer(backend: MemoryBackend, session: Session, *, client: str = "PASTICUAN", api_key: str = "fixture-key") -> SharedCompanyEvidence:
    coordinator = SharedEvidenceCoordinator(backend, client_id=client, worker_id=f"{client}-worker")
    return SharedCompanyEvidence(client, backend=backend, coordinator=coordinator, session=session, api_key=api_key)


def test_directory_merges_company_and_security_facts() -> None:
    row = normalize_company_directory([_company()], [_security()], source_period=OBSERVED, observed_on=OBSERVED)[0]
    assert row["ticker"] == "BBCA" and row["sector"] == "Keuangan"
    assert row["sub_sector"] == "Bank" and row["listing_board"] == "Utama"
    assert row["listing_date"] == "2000-05-31" and row["listed_shares"] == 122_042_299_500
    assert row["main_business"] == "Jasa Perbankan" and row["change_state"] == "NEW"
    assert row["profile"]["security_flags"] == {"stock": True, "bond": True, "etf": False, "eba": False, "spei": False}


def test_directory_keeps_unmatched_rows_without_inventing_fields() -> None:
    rows = normalize_company_directory(
        [_company(code="BBCA")], [_security(code="BBRI", Name="Bank Rakyat Indonesia")],
        source_period=OBSERVED, observed_on=OBSERVED,
    )
    by_ticker = {row["ticker"]: row for row in rows}
    assert by_ticker["BBCA"]["listed_shares"] is None
    assert by_ticker["BBRI"]["sector"] is None and by_ticker["BBRI"]["company_name"] == "Bank Rakyat Indonesia"


def test_duplicate_conflicting_directory_rows_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="PARSE_FAILURE"):
        normalize_company_directory(
            [_company(), _company(Sektor="Teknologi")], [_security()],
            source_period=OBSERVED, observed_on=OBSERVED,
        )


def test_directory_delta_state_uses_only_payload_hash() -> None:
    first = normalize_company_directory([_company()], [_security()], source_period=OBSERVED, observed_on=OBSERVED)[0]
    same = normalize_company_directory(
        [_company()], [_security()], source_period=OBSERVED, observed_on=OBSERVED, previous=[first]
    )[0]
    changed = normalize_company_directory(
        [_company(Sektor="Teknologi")], [_security()], source_period=OBSERVED, observed_on=OBSERVED, previous=[first]
    )[0]
    assert same["change_state"] == "UNCHANGED" and changed["change_state"] == "CHANGED"
    assert same["payload_hash"] == first["payload_hash"] != changed["payload_hash"]


def test_profile_persists_factual_relationships_without_contact_details() -> None:
    row = normalize_company_profile(_profile(), ticker="BBCA", source_period=OBSERVED, observed_on=OBSERVED)[0]
    relationships = row["profile"]["relationships"]
    assert relationships["directors"] == [{"name": "Direktur A", "title": "PRESIDEN DIREKTUR"}]
    assert relationships["commissioners"][0]["name"] == "Komisaris A"
    assert relationships["shareholders"][0]["sharePct"] == 55.0
    assert relationships["subsidiaries"][0]["ownershipPct"] == 90
    assert relationships["corporate_secretary"] == [{"name": "Sekretaris A"}]
    assert "email" not in str(relationships)


def test_profile_issuer_mismatch_and_malformed_relationships_fail_closed() -> None:
    with pytest.raises(RuntimeError, match="ISSUER_MISMATCH"):
        normalize_company_profile(_profile(code="BBRI"), ticker="BBCA", source_period=OBSERVED, observed_on=OBSERVED)
    with pytest.raises(RuntimeError, match="PARSE_FAILURE"):
        normalize_company_profile(_profile(directors={"name": "bad"}), ticker="BBCA", source_period=OBSERVED, observed_on=OBSERVED)


def test_reference_values_are_deterministic_and_deduplicated() -> None:
    payload = {"set": "sectors", "count": 3, "items": ["Financials", "Technology", "Financials"], "dataset": "reference", "provider": "idx"}
    rows = normalize_reference_values(payload, set_name="sectors", source_period=OBSERVED, observed_on=OBSERVED)
    assert [row["label"] for row in rows] == ["Financials", "Technology"]
    assert rows[0]["value_key"] == normalize_reference_values(payload, set_name="sectors", source_period=OBSERVED, observed_on=OBSERVED)[0]["value_key"]


def test_reference_rejects_wrong_set_or_non_scalar_items() -> None:
    with pytest.raises(RuntimeError, match="CONTEXT_REJECTED"):
        normalize_reference_values(
            {"set": "boards", "items": ["Utama"], "dataset": "reference", "provider": "idx"},
            set_name="sectors", source_period=OBSERVED, observed_on=OBSERVED,
        )
    with pytest.raises(RuntimeError, match="PARSE_FAILURE"):
        normalize_reference_values(
            {"set": "sectors", "items": [{"name": "bad"}], "dataset": "reference", "provider": "idx"},
            set_name="sectors", source_period=OBSERVED, observed_on=OBSERVED,
        )


def test_directory_uses_two_bulk_calls_without_per_ticker_requests() -> None:
    session = Session([
        Response(_offset_payload("listed-companies", [_company(), _company(code="BBRI", NamaEmiten="Bank Rakyat")])),
        Response(_offset_payload("securities", [_security(), _security(code="BBRI", Name="Bank Rakyat")])),
    ])
    rows, meta = _producer(MemoryBackend(), session).get_directory(OBSERVED)
    assert {row["ticker"] for row in rows} == {"BBCA", "BBRI"}
    assert meta["api_calls"] == 2 and len(session.calls) == 2
    assert [call["url"] for call in session.calls] == [COMPANIES_URL, SECURITIES_URL]
    assert all(call["params"] == {"length": BULK_LENGTH, "start": 0} for call in session.calls)
    assert all("code" not in call["params"] for call in session.calls)


def test_directory_offset_pagination_is_bounded() -> None:
    session = Session([
        Response(_offset_payload("listed-companies", [_company()], total=1001)),
        Response(_offset_payload("listed-companies", [_company(code="BBRI")], start=1000, total=1001)),
        Response(_offset_payload("securities", [_security()])),
    ])
    rows, meta = _producer(MemoryBackend(), session).get_directory(OBSERVED)
    assert {row["ticker"] for row in rows} == {"BBCA", "BBRI"}
    assert meta["api_calls"] == 3 and session.calls[1]["params"]["start"] == BULK_LENGTH


def test_directory_never_persists_truncated_max_page_result() -> None:
    session = Session([Response(_offset_payload("listed-companies", [_company()], total=1001))])
    rows, meta = _producer(MemoryBackend(), session).get_directory(OBSERVED, max_pages=1)
    assert not rows and meta["state"] == "INSUFFICIENT_HISTORY" and meta["api_calls"] == 1


def test_profile_is_on_demand_single_ticker_and_not_all_company_daily() -> None:
    session = Session([Response(_profile())])
    rows, meta = _producer(MemoryBackend(), session).get_profile("bbca.jk", OBSERVED)
    assert len(rows) == 1 and meta["api_calls"] == 1
    assert session.calls[0]["url"] == PROFILE_URL and session.calls[0]["params"] == {"code": "BBCA"}
    assert meta["ttl_days"] == 30


@pytest.mark.parametrize("set_name", ["sectors", "boards", "market-time"])
def test_reference_sets_use_one_slow_ttl_call(set_name: str) -> None:
    session = Session([Response({"set": set_name, "count": 1, "items": ["Value"], "dataset": "reference", "provider": "idx"})])
    rows, meta = _producer(MemoryBackend(), session).get_reference(set_name, OBSERVED)
    assert len(rows) == 1 and meta["api_calls"] == 1 and meta["ttl_days"] == 90
    assert session.calls[0]["url"] == REFERENCE_URL and session.calls[0]["params"] == {"set": set_name}


@pytest.mark.parametrize("first,second", [("PASTICUAN", "EMIR"), ("EMIR", "PASTICUAN")])
@pytest.mark.parametrize("kind", ["directory", "profile", "reference"])
def test_second_scanner_reuses_slow_evidence_without_key(first: str, second: str, kind: str) -> None:
    backend = MemoryBackend()
    if kind == "directory":
        first_session = Session([
            Response(_offset_payload("listed-companies", [_company()])),
            Response(_offset_payload("securities", [_security()])),
        ])
        first_rows, _ = _producer(backend, first_session, client=first).get_directory(OBSERVED)
        second_session = Session([])
        second_rows, meta = _producer(backend, second_session, client=second, api_key="").get_directory(OBSERVED)
    elif kind == "profile":
        first_rows, _ = _producer(backend, Session([Response(_profile())]), client=first).get_profile("BBCA", OBSERVED)
        second_session = Session([])
        second_rows, meta = _producer(backend, second_session, client=second, api_key="").get_profile("BBCA", OBSERVED)
    else:
        payload = {"set": "sectors", "items": ["Financials"], "dataset": "reference", "provider": "idx"}
        first_rows, _ = _producer(backend, Session([Response(payload)]), client=first).get_reference("sectors", OBSERVED)
        second_session = Session([])
        second_rows, meta = _producer(backend, second_session, client=second, api_key="").get_reference("sectors", OBSERVED)
    assert first_rows == second_rows and not second_session.calls
    assert meta["cache_hit"] and meta["request_avoided"] and meta["api_calls"] == 0


def test_stale_directory_refresh_records_changed_delta() -> None:
    backend = MemoryBackend()
    old_stamp = datetime(2020, 1, 1, tzinfo=timezone.utc)
    backend.tables["evidence_companies"] = normalize_company_directory(
        [_company()], [_security()], source_period=OBSERVED, observed_on=OBSERVED, fetched_at=old_stamp
    )
    session = Session([
        Response(_offset_payload("listed-companies", [_company(Sektor="Teknologi")])),
        Response(_offset_payload("securities", [_security()])),
    ])
    rows, meta = _producer(backend, session).get_directory(OBSERVED)
    assert rows[0]["change_state"] == "CHANGED" and meta["state"] == "REFRESHED"
    assert meta["api_calls"] == 2


def test_missing_key_and_invalid_inputs_make_no_request() -> None:
    session = Session([])
    producer = _producer(MemoryBackend(), session, api_key="")
    assert producer.get_directory(OBSERVED)[1]["state"] == "ENVIRONMENT_BLOCKED"
    assert producer.get_profile("BAD!", OBSERVED)[1]["state"] == "ISSUER_IDENTITY_MISSING"
    assert producer.get_reference("bad", OBSERVED)[1]["state"] == "CONTEXT_REJECTED"
    assert not session.calls


@pytest.mark.parametrize("status", [401, 403, 404, 429])
def test_http_failures_are_explicit(status: int) -> None:
    rows, meta = _producer(MemoryBackend(), Session([Response(status=status)])).get_profile("BBCA", OBSERVED)
    assert not rows and meta["state"] == f"HTTP_{status}"


@pytest.mark.parametrize("outcome,reason", [(requests.Timeout(), "TIMEOUT"), (requests.ConnectionError(), "CONNECTION_ERROR")])
def test_network_failures_are_explicit(outcome: Exception, reason: str) -> None:
    rows, meta = _producer(MemoryBackend(), Session([outcome])).get_profile("BBCA", OBSERVED)
    assert not rows and meta["state"] == reason


@pytest.mark.parametrize(
    "response,reason",
    [
        (Response(content=b""), "EMPTY_RESPONSE"),
        (Response(malformed=True), "PARSE_FAILURE"),
        (Response({"code": "BBCA", "dataset": "wrong", "provider": "idx"}), "CONTEXT_REJECTED"),
    ],
)
def test_bad_profile_responses_fail_closed(response: Response, reason: str) -> None:
    rows, meta = _producer(MemoryBackend(), Session([response])).get_profile("BBCA", OBSERVED)
    assert not rows and meta["state"] == reason


def test_validation_rejects_negative_shares_and_no_scanner_conclusions() -> None:
    row = normalize_company_directory([_company()], [_security()], source_period=OBSERVED, observed_on=OBSERVED)[0]
    assert validate_company_rows([row], provider="IDX_COMPANY_DIRECTORY_VIA_ZAPI") == (True, "VALID")
    assert validate_company_rows([dict(row, listed_shares=-1)], provider="IDX_COMPANY_DIRECTORY_VIA_ZAPI")[1] == "CONTEXT_REJECTED"
    text = (Path(__file__).resolve().parents[1] / "shared_company_evidence.py").read_text().lower()
    for forbidden in ("ranking", "production_gate", "governance_score", "recommendation"):
        assert forbidden not in text


def test_migration_has_reference_table_rls_grants_and_lookup_indexes() -> None:
    root = Path(__file__).resolve().parents[1]
    migration = next((root / "database").glob("migration_v*_shared_evidence_hub.sql")).read_text()
    for fragment in (
        "create table if not exists public.evidence_reference_values",
        "evidence_companies_provider_freshness_idx",
        "evidence_reference_values_lookup_idx",
        "'evidence_companies', 'evidence_reference_values'",
        "grant select, insert, update on table public.%I to service_role",
        "enable row level security",
    ):
        assert fragment in migration


def test_company_provider_redirects_fail_closed() -> None:
    session = Session([Response(status=302)])
    rows, meta = _producer(MemoryBackend(), session).get_profile("BBCA", OBSERVED)
    assert rows == []
    assert meta["state"] == "CONTEXT_REJECTED"
    assert session.calls[0]["allow_redirects"] is False
