from __future__ import annotations

from datetime import date
from pathlib import Path
import threading
from typing import Any, Mapping

import pandas as pd
import pytest
import requests

from shared_evidence_hub import EvidenceKey, SharedEvidenceCoordinator
from shared_ownership_evidence import (
    CATEGORIES,
    INDEX_PAGE_SIZE,
    MAX_FILES_PER_PUBLICATION,
    MAX_INDEX_PAGES,
    OWNERSHIP_INDEX_URL,
    REQUEST_TIMEOUT_SECONDS,
    SharedOwnershipEvidence,
    derive_ownership_changes,
    parse_ownership_workbook,
    validate_ownership_rows,
)


PUBLICATION = date(2026, 7, 30)
OFFICIAL_URL = "https://www.idx.co.id/Media/example/ownership.xlsx"
FILE_BYTES = b"PK\x03\x04fixture-workbook"


def _frame(category: str, *, holder: str = "PT Contoh", shares: Any = "1.234.567", pct: Any = "5,25%") -> pd.DataFrame:
    identity_header = {
        "lima-persen": "Nama Pemegang Saham",
        "satu-persen": "Nama Pemegang Saham",
        "klasifikasi": "Klasifikasi Investor",
        "tipe": "Tipe Investor",
    }[category]
    identity = holder if category in {"lima-persen", "satu-persen"} else (
        "Institusi" if category == "klasifikasi" else "Reksa Dana"
    )
    return pd.DataFrame([
        ["DATA KEPEMILIKAN SAHAM", None, None, None, None, None],
        ["Kode Emiten", identity_header, "Jumlah Saham", "Persentase Kepemilikan", "Lokal/Asing", "Tanggal Posisi"],
        ["BBCA", identity, shares, pct, "Lokal", "2026-07-29"],
    ])


def _index(
    category: str = "lima-persen",
    published: str = "2026-07-30",
    url: str = OFFICIAL_URL,
    *,
    total: int = 1,
) -> dict[str, Any]:
    return {"data": [{
        "url": url,
        "category": category,
        "fileName": "ownership.xlsx",
        "publishedAt": published,
    }], "total": total}


class Response:
    def __init__(
        self,
        *,
        payload: Any = None,
        content: bytes = b"json",
        status: int = 200,
        url: str = "",
        content_type: str = "application/json",
        malformed: bool = False,
        headers: Mapping[str, str] | None = None,
    ):
        self.payload = payload
        self.content = content
        self.status_code = status
        self.url = url
        self.headers = {"Content-Type": content_type, **dict(headers or {})}
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
            "evidence_ownership_files": [],
            "evidence_ownership_snapshots": [],
            "evidence_ownership_changes": [],
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
        return [
            dict(row) for row in self.tables[table]
            if all(str(row.get(name)) == str(value) for name, value in filters.items())
        ][:limit]

    def upsert_rows(
        self, table: str, rows: list[Mapping[str, Any]], *, conflict: tuple[str, ...]
    ) -> list[dict[str, Any]]:
        with self.lock:
            keyed = {
                tuple(row.get(name) for name in conflict): dict(row)
                for row in self.tables[table]
            }
            for row in rows:
                keyed[tuple(row.get(name) for name in conflict)] = dict(row)
            self.tables[table] = list(keyed.values())
        return [dict(row) for row in rows]


def _producer(
    backend: MemoryBackend,
    session: Session,
    *,
    client: str = "PASTICUAN",
    api_key: str = "test-key",
    category: str = "lima-persen",
) -> SharedOwnershipEvidence:
    coordinator = SharedEvidenceCoordinator(backend, client_id=client, worker_id=f"{client}-worker")
    return SharedOwnershipEvidence(
        client,
        backend=backend,
        coordinator=coordinator,
        session=session,
        api_key=api_key,
        workbook_reader=lambda _: {"Sheet1": _frame(category)},
    )


@pytest.mark.parametrize("category", sorted(CATEGORIES))
def test_parses_all_documented_categories(category: str) -> None:
    rows = parse_ownership_workbook(
        {"Sheet1": _frame(category)},
        category=category,
        publication_date=PUBLICATION,
        source_url=OFFICIAL_URL,
        source_file_hash="a" * 64,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["ticker"] == "BBCA" and row["shares_held"] == 1234567
    assert row["ownership_percentage"] == 5.25
    assert row["report_date"] == "2026-07-29" and row["publication_date"] == "2026-07-30"
    assert row["source_verified"] and len(row["holder_identity_hash"]) == 64
    assert validate_ownership_rows(rows, category=category) == (True, "VALID")


def test_parser_rejects_nonofficial_source_duplicate_and_empty_workbook() -> None:
    with pytest.raises(RuntimeError, match="CONTEXT_REJECTED"):
        parse_ownership_workbook(
            {"Sheet1": _frame("lima-persen")}, category="lima-persen",
            publication_date=PUBLICATION, source_url="https://evil.example/file.xlsx", source_file_hash="a" * 64,
        )
    duplicate = pd.concat([_frame("lima-persen"), _frame("lima-persen").iloc[2:]], ignore_index=True)
    with pytest.raises(RuntimeError, match="PARSE_FAILURE"):
        parse_ownership_workbook(
            {"Sheet1": duplicate}, category="lima-persen", publication_date=PUBLICATION,
            source_url=OFFICIAL_URL, source_file_hash="a" * 64,
        )
    with pytest.raises(RuntimeError, match="PARSE_FAILURE"):
        parse_ownership_workbook(
            {"Sheet1": pd.DataFrame()}, category="lima-persen", publication_date=PUBLICATION,
            source_url=OFFICIAL_URL, source_file_hash="a" * 64,
        )


def test_pipeline_fetches_one_index_and_only_matching_official_file() -> None:
    backend = MemoryBackend()
    session = Session([
        Response(payload=_index()),
        Response(content=FILE_BYTES, url=OFFICIAL_URL, content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    ])
    rows, meta = _producer(backend, session).get_publication("lima-persen", PUBLICATION)
    assert len(rows) == 1 and meta["state"] == "REFRESHED"
    assert meta["api_calls"] == 1 and meta["file_calls"] == 1 and len(session.calls) == 2
    assert session.calls[0]["url"] == OWNERSHIP_INDEX_URL
    assert session.calls[0]["params"] == {
        "category": "lima-persen", "length": INDEX_PAGE_SIZE, "start": 0
    }
    assert session.calls[0]["allow_redirects"] is False
    assert session.calls[0]["timeout"] == REQUEST_TIMEOUT_SECONDS
    assert session.calls[0]["headers"]["User-Agent"] == "Shared-IDX-Evidence-Hub/ownership-index"
    assert session.calls[1]["headers"]["Referer"] == "https://www.idx.co.id/"
    assert "x-api-key" not in session.calls[1]["headers"]
    assert session.calls[1]["allow_redirects"] is False
    assert len(backend.tables["evidence_ownership_files"]) == 1
    assert len(backend.tables["evidence_ownership_snapshots"]) == 1
    assert "test-key" not in str(meta)


def test_index_pagination_is_bounded_and_stops_when_target_is_found() -> None:
    first_page = _index(published="2026-07-31", total=400)
    second_page = _index(published="2026-07-30", total=400)
    session = Session([
        Response(payload=first_page),
        Response(payload=second_page),
        Response(content=FILE_BYTES, url=OFFICIAL_URL, content_type="application/octet-stream"),
    ])
    rows, meta = _producer(MemoryBackend(), session).get_publication("lima-persen", PUBLICATION)
    assert len(rows) == 1 and meta["state"] == "REFRESHED"
    assert meta["api_calls"] == 2 and meta["file_calls"] == 1
    assert [call["params"]["start"] for call in session.calls[:2]] == [0, INDEX_PAGE_SIZE]


def test_index_page_cap_fails_closed_without_downloading_files() -> None:
    outcomes = [
        Response(payload=_index(published="2026-08-01", total=1000))
        for _ in range(MAX_INDEX_PAGES)
    ]
    session = Session(outcomes)
    rows, meta = _producer(MemoryBackend(), session).get_publication("lima-persen", PUBLICATION)
    assert not rows and meta["state"] == "NO_FILE"
    assert meta["api_calls"] == MAX_INDEX_PAGES
    assert meta["file_calls"] == 0
    assert len(session.calls) == MAX_INDEX_PAGES


def test_multiple_matching_files_fail_closed_before_download() -> None:
    payload = {
        "data": [
            {
                "url": OFFICIAL_URL,
                "category": "lima-persen",
                "fileName": "a.xlsx",
                "publishedAt": "2026-07-30",
            },
            {
                "url": "https://www.idx.co.id/Media/example/ownership-2.xlsx",
                "category": "lima-persen",
                "fileName": "b.xlsx",
                "publishedAt": "2026-07-30",
            },
        ],
        "total": 2,
    }
    session = Session([Response(payload=payload)])
    rows, meta = _producer(MemoryBackend(), session).get_publication("lima-persen", PUBLICATION)
    assert not rows and meta["state"] == "CONTEXT_REJECTED"
    assert meta["file_calls"] == 0
    assert MAX_FILES_PER_PUBLICATION == 1


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame([
            ["Kode Emiten", "Nama Pemegang Saham", "Jumlah Saham", "Persentase Kepemilikan", "Tanggal Posisi"],
            ["BBCA", "PT Contoh", 100, 5.0, None],
        ]),
        pd.DataFrame([
            ["Kode Emiten", "Nama Pemegang Saham", "Jumlah Saham", "Persentase Kepemilikan", "Tanggal Posisi"],
            ["BBCA", "PT Contoh", 100, 5.0, "2026-07-31"],
        ]),
    ],
)
def test_parser_rejects_missing_or_future_report_date(frame: pd.DataFrame) -> None:
    with pytest.raises(RuntimeError, match="WRONG_PERIOD"):
        parse_ownership_workbook(
            {"Sheet1": frame},
            category="lima-persen",
            publication_date=PUBLICATION,
            source_url=OFFICIAL_URL,
            source_file_hash="a" * 64,
        )


def test_parser_rejects_mixed_report_dates_in_one_publication() -> None:
    frame = _frame("lima-persen")
    frame.loc[len(frame)] = ["BBRI", "PT Lain", "2.000", "5,10%", "Lokal", "2026-07-28"]
    with pytest.raises(RuntimeError, match="WRONG_PERIOD"):
        parse_ownership_workbook(
            {"Sheet1": frame},
            category="lima-persen",
            publication_date=PUBLICATION,
            source_url=OFFICIAL_URL,
            source_file_hash="a" * 64,
        )


def test_redirect_is_not_followed_implicitly() -> None:
    session = Session([
        Response(payload=_index()),
        Response(
            status=302,
            url=OFFICIAL_URL,
            headers={"Location": "https://www.idx.co.id/Media/example/redirected.xlsx"},
        ),
    ])
    rows, meta = _producer(MemoryBackend(), session).get_publication("lima-persen", PUBLICATION)
    assert not rows and meta["state"] == "CONTEXT_REJECTED"
    assert meta["failure_stage"] == "OFFICIAL_FILE"
    assert len(session.calls) == 2
    assert session.calls[-1]["allow_redirects"] is False


@pytest.mark.parametrize("first,second", [("PASTICUAN", "EMIR"), ("EMIR", "PASTICUAN")])
def test_second_scanner_reuses_file_and_snapshots_without_key(first: str, second: str) -> None:
    backend = MemoryBackend()
    first_session = Session([
        Response(payload=_index()),
        Response(content=FILE_BYTES, url=OFFICIAL_URL, content_type="application/octet-stream"),
    ])
    first_rows, _ = _producer(backend, first_session, client=first).get_publication("lima-persen", PUBLICATION)
    second_session = Session([])
    second_rows, meta = _producer(backend, second_session, client=second, api_key="").get_publication(
        "lima-persen", PUBLICATION
    )
    assert first_rows == second_rows and not second_session.calls
    assert meta["cache_hit"] and meta["request_avoided"]


def test_missing_key_on_miss_makes_no_request() -> None:
    rows, meta = _producer(MemoryBackend(), Session([]), api_key="").get_publication(
        "lima-persen", PUBLICATION
    )
    assert not rows and meta["state"] == "ENVIRONMENT_BLOCKED"
    assert meta["api_calls"] == 0 and meta["file_calls"] == 0


@pytest.mark.parametrize(
    "status,reason",
    [
        (401, "HTTP_401"),
        (403, "HTTP_403"),
        (404, "HTTP_404"),
        (429, "HTTP_429"),
    ],
)
def test_index_http_failures_are_explicit(status: int, reason: str) -> None:
    rows, meta = _producer(MemoryBackend(), Session([Response(status=status)])).get_publication(
        "lima-persen", PUBLICATION
    )
    assert not rows and meta["state"] == reason
    assert meta["failure_stage"] == "ZAPI_INDEX"


@pytest.mark.parametrize(
    "outcome,reason",
    [
        (requests.Timeout("slow"), "TIMEOUT"),
        (requests.ConnectionError("offline"), "CONNECTION_ERROR"),
    ],
)
def test_index_network_failures_are_explicit(outcome: Exception, reason: str) -> None:
    rows, meta = _producer(MemoryBackend(), Session([outcome])).get_publication("lima-persen", PUBLICATION)
    assert not rows and meta["state"] == reason
    assert meta["failure_stage"] == "ZAPI_INDEX"


@pytest.mark.parametrize(
    "status,reason",
    [
        (401, "HTTP_401"),
        (403, "HTTP_403"),
        (404, "HTTP_404"),
        (429, "HTTP_429"),
    ],
)
def test_official_file_http_failures_are_stage_specific(status: int, reason: str) -> None:
    session = Session([
        Response(payload=_index()),
        Response(status=status, url=OFFICIAL_URL),
    ])
    rows, meta = _producer(MemoryBackend(), session).get_publication("lima-persen", PUBLICATION)
    assert not rows and meta["state"] == reason
    assert meta["failure_stage"] == "OFFICIAL_FILE"
    assert meta["api_calls"] == 1 and meta["file_calls"] == 1


@pytest.mark.parametrize(
    "response,reason",
    [
        (Response(content=b""), "EMPTY_RESPONSE"),
        (Response(malformed=True), "PARSE_FAILURE"),
        (Response(payload={"unexpected": []}), "PARSE_FAILURE"),
        (Response(payload=_index(published="2026-07-29")), "NO_FILE"),
        (Response(payload=_index(url="https://evil.example/file.xlsx")), "NO_FILE"),
    ],
)
def test_index_empty_malformed_wrong_date_and_nonofficial_entries_fail_closed(response: Response, reason: str) -> None:
    rows, meta = _producer(MemoryBackend(), Session([response])).get_publication("lima-persen", PUBLICATION)
    assert not rows and meta["state"] == reason


@pytest.mark.parametrize(
    "file_response",
    [
        Response(content=b"not-xlsx", url=OFFICIAL_URL, content_type="application/octet-stream"),
        Response(content=FILE_BYTES, url="https://evil.example/redirect.xlsx", content_type="application/octet-stream"),
        Response(content=FILE_BYTES, url=OFFICIAL_URL, content_type="text/html"),
    ],
)
def test_download_requires_official_final_url_xlsx_signature_and_type(file_response: Response) -> None:
    session = Session([Response(payload=_index()), file_response])
    rows, meta = _producer(MemoryBackend(), session).get_publication("lima-persen", PUBLICATION)
    assert not rows and meta["state"] == "INVALID_CONTENT_TYPE"


def _snapshot(
    holder: str, report: str, shares: int, pct: float, source_hash: str, *, category: str = "lima-persen"
) -> dict[str, Any]:
    return {
        "source_file_hash": source_hash, "category": category, "ticker": "BBCA",
        "holder_identity_hash": holder, "holder_name": holder, "report_date": report,
        "shares_held": shares, "ownership_percentage": pct, "source_verified": True,
        "validation_state": "VALID",
    }


def test_comparable_snapshots_derive_increase_new_and_exit_only() -> None:
    previous = [
        _snapshot("same", "2026-06-30", 100, 5.0, "old"),
        _snapshot("exit", "2026-06-30", 50, 2.0, "old"),
    ]
    current = [
        _snapshot("same", "2026-07-29", 120, 6.0, "new"),
        _snapshot("new", "2026-07-29", 30, 5.1, "new"),
    ]
    changes = derive_ownership_changes(previous, current)
    states = {row["holder_identity_hash"]: row["change_state"] for row in changes}
    assert states == {
        "exit": "EXITED_REPORTED_HOLDER",
        "new": "NEW_5PCT_HOLDER",
        "same": "INCREASED_REPORTED_HOLDING",
    }
    increased = next(row for row in changes if row["holder_identity_hash"] == "same")
    assert increased["delta_shares"] == 20 and increased["delta_percentage"] == 1.0


def test_noncomparable_or_first_snapshot_never_invents_changes() -> None:
    current = [_snapshot("same", "2026-07-29", 120, 6.0, "new")]
    assert derive_ownership_changes([], current) == []
    same_date = [_snapshot("same", "2026-07-29", 100, 5.0, "old")]
    assert derive_ownership_changes(same_date, current) == []
    mixed = current + [_snapshot("other", "2026-07-28", 20, 1.0, "new")]
    assert derive_ownership_changes(same_date, mixed) == []


def test_factual_rows_do_not_claim_beneficial_owner_broker_or_bandar() -> None:
    row = parse_ownership_workbook(
        {"Sheet1": _frame("lima-persen")}, category="lima-persen",
        publication_date=PUBLICATION, source_url=OFFICIAL_URL, source_file_hash="a" * 64,
    )[0]
    forbidden = {"beneficial_owner", "broker", "bandar", "score", "rank", "recommendation"}
    assert forbidden.isdisjoint(row)


def test_migration_persists_comparable_changes_under_same_security_contract() -> None:
    migration = next((Path(__file__).resolve().parents[1] / "database").glob("migration_v*_shared_evidence_hub.sql"))
    sql = migration.read_text(encoding="utf-8").lower()
    assert "create table if not exists public.evidence_ownership_changes" in sql
    assert "current_report_date > previous_report_date" in sql
    assert "'evidence_ownership_snapshots', 'evidence_ownership_changes'" in sql
    assert "reported-holder snapshots; not beneficial-owner, broker, or bandar identity" in sql
