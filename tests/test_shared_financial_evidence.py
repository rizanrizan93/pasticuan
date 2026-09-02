from __future__ import annotations

from datetime import date
from io import BytesIO
from pathlib import Path
import threading
from typing import Any, Mapping
import zipfile

import pytest
import requests

from shared_evidence_hub import EvidenceKey, SharedEvidenceCoordinator
from shared_financial_evidence import (
    REQUEST_TIMEOUT_SECONDS,
    SharedFinancialEvidence,
    ZAPI_FINANCIAL_REPORT_URL,
    parse_idx_xbrl_facts,
    period_contract,
    validate_financial_facts,
)


PERIOD_END = date(2026, 3, 31)
PUBLICATION = date(2026, 5, 1)
OFFICIAL_URL = "https://www.idx.co.id/official/TEST/instance.zip"


def _xbrl(
    *,
    issuer: str | None = "test_maker",
    scheme: str = "http://www.idx.co.id/xbrl",
    period_end: str = "2026-03-31",
    extra: str = "",
) -> bytes:
    identifier = f'<xbrli:identifier scheme="{scheme}">{issuer}</xbrli:identifier>' if issuer else ""
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<xbrli:xbrl xmlns:xbrli="http://www.xbrl.org/2003/instance" xmlns:id="http://www.idx.co.id/xbrl/taxonomy/2020">
  <xbrli:context id="CurrentYearDuration"><xbrli:entity>{identifier}</xbrli:entity><xbrli:period><xbrli:startDate>2026-01-01</xbrli:startDate><xbrli:endDate>{period_end}</xbrli:endDate></xbrli:period></xbrli:context>
  <xbrli:context id="CurrentYearInstant"><xbrli:entity>{identifier}</xbrli:entity><xbrli:period><xbrli:instant>{period_end}</xbrli:instant></xbrli:period></xbrli:context>
  <xbrli:unit id="IDR"><xbrli:measure>iso4217:IDR</xbrli:measure></xbrli:unit>
  <id:Revenues contextRef="CurrentYearDuration" unitRef="IDR">1000</id:Revenues>
  <id:ProfitLoss contextRef="CurrentYearDuration" unitRef="IDR">100</id:ProfitLoss>
  <id:NetCashFlowsReceivedFromUsedInOperatingActivities contextRef="CurrentYearDuration" unitRef="IDR">150</id:NetCashFlowsReceivedFromUsedInOperatingActivities>
  <id:PaymentsForAcquisitionOfPropertyAndEquipment contextRef="CurrentYearDuration" unitRef="IDR">30</id:PaymentsForAcquisitionOfPropertyAndEquipment>
  <id:PaymentsForAcquisitionOfIntangibleAssets contextRef="CurrentYearDuration" unitRef="IDR">20</id:PaymentsForAcquisitionOfIntangibleAssets>
  <id:FreeCashFlow contextRef="CurrentYearDuration" unitRef="IDR">100</id:FreeCashFlow>
  <id:Assets contextRef="CurrentYearInstant" unitRef="IDR">5000</id:Assets>
  <id:Liabilities contextRef="CurrentYearInstant" unitRef="IDR">2000</id:Liabilities>
  <id:Equity contextRef="CurrentYearInstant" unitRef="IDR">3000</id:Equity>
  <id:CashAndCashEquivalents contextRef="CurrentYearInstant" unitRef="IDR">600</id:CashAndCashEquivalents>
  <id:ShortTermDebt contextRef="CurrentYearInstant" unitRef="IDR">200</id:ShortTermDebt>
  <id:LongTermDebt contextRef="CurrentYearInstant" unitRef="IDR">800</id:LongTermDebt>
  <id:TotalDebt contextRef="CurrentYearInstant" unitRef="IDR">1000</id:TotalDebt>
  {extra}
</xbrli:xbrl>'''.encode()


def _parse(payload: bytes | None = None):
    return parse_idx_xbrl_facts(
        payload or _xbrl(), ticker="TEST", report_period="2026-Q1", period_type="Q1",
        period_end=PERIOD_END, publication_date=PUBLICATION, source_url=OFFICIAL_URL,
        filename="instance.xml",
    )


def _manifest(
    *,
    code: str = "TEST",
    year: str = "2026",
    period: str = "Q1",
    attachment_code: str = "TEST",
    filename: str = "instance.zip",
    path: str = "/official/TEST/instance.zip",
) -> dict[str, Any]:
    return {"data": [{
        "KodeEmiten": code,
        "Report_Year": year,
        "Report_Period": period,
        "File_Modified": "2026-05-01T10:00:00",
        "Attachments": [{
            "File_Name": filename, "File_Path": path, "Emiten_Code": attachment_code,
            "File_Modified": "2026-05-01T10:00:00", "Report_Type": "rdf",
        }],
    }]}


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
    ):
        self.payload = payload
        self.content = content
        self.status_code = status
        self.url = url
        self.headers = {"Content-Type": content_type}
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
            "evidence_financial_reports": [], "evidence_financial_facts": [],
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

    @staticmethod
    def _same(left: Any, right: Any) -> bool:
        if isinstance(left, bool):
            return str(left).lower() == str(right).lower()
        return str(left) == str(right)

    def read_rows(self, table: str, filters: Mapping[str, Any], *, limit: int, **_: Any) -> list[dict[str, Any]]:
        return [dict(row) for row in self.tables[table] if all(self._same(row.get(k), v) for k, v in filters.items())][:limit]

    def upsert_rows(self, table: str, rows: list[Mapping[str, Any]], *, conflict: tuple[str, ...]) -> list[dict[str, Any]]:
        keyed = {tuple(row.get(name) for name in conflict): dict(row) for row in self.tables[table]}
        for row in rows:
            keyed[tuple(row.get(name) for name in conflict)] = dict(row)
        self.tables[table] = list(keyed.values())
        return [dict(row) for row in rows]


def _producer(backend: MemoryBackend, session: Session, *, client: str = "PASTICUAN", api_key: str = "test-key") -> SharedFinancialEvidence:
    coordinator = SharedEvidenceCoordinator(backend, client_id=client, worker_id=f"{client}-worker")
    return SharedFinancialEvidence(client, backend=backend, coordinator=coordinator, session=session, api_key=api_key)


def _success_session(payload: bytes | None = None) -> Session:
    return Session([
        Response(payload=_manifest()),
        Response(content=payload or _xbrl(), url=OFFICIAL_URL, content_type="application/xml"),
    ])


def test_period_contract_is_deterministic() -> None:
    assert period_contract(2026, "tw1") == ("Q1", "2026-Q1", date(2026, 3, 31))
    assert period_contract(2025, "audit") == ("FY", "2025-FY", date(2025, 12, 31))
    with pytest.raises(ValueError):
        period_contract(2026, "q4")


def test_parser_preserves_raw_validated_canonical_facts_and_provenance() -> None:
    report, facts = _parse()
    values = {row["fact_name"]: row["fact_value"] for row in facts}
    assert values == {
        "assets": 5000, "capex": 50, "cash": 600, "equity": 3000,
        "free_cash_flow": 100, "liabilities": 2000, "long_term_debt": 800,
        "net_income": 100, "operating_cash_flow": 150, "revenue": 1000,
        "short_term_debt": 200, "total_debt": 1000,
    }
    assert report["issuer_identity"] == "test_maker" and report["issuer_match"] is True
    assert report["context_state"] == "CURRENT_PERIOD_ISSUER_MATCHED"
    assert report["source_document_hash"] == facts[0]["source_document_hash"]
    assert all(row["currency"] == "IDR" and row["unit_scale"] == 1 for row in facts)
    assert validate_financial_facts(facts, ticker="TEST", report_period="2026-Q1") == (True, "VALID")


def test_parser_sums_distinct_capex_components_without_deriving_fcf() -> None:
    payload = _xbrl().replace(b'<id:FreeCashFlow contextRef="CurrentYearDuration" unitRef="IDR">100</id:FreeCashFlow>', b"")
    _, facts = _parse(payload)
    values = {row["fact_name"]: row["fact_value"] for row in facts}
    assert values["capex"] == 50
    assert "free_cash_flow" not in values




def test_parser_rejects_unknown_or_mixed_currency_units() -> None:
    unknown = _xbrl().replace(b"iso4217:IDR", b"xbrli:shares")
    with pytest.raises(RuntimeError, match="CONTEXT_REJECTED"):
        _parse(unknown)

    mixed = _xbrl().replace(
        b'<xbrli:unit id="IDR"><xbrli:measure>iso4217:IDR</xbrli:measure></xbrli:unit>',
        b'<xbrli:unit id="IDR"><xbrli:measure>iso4217:IDR</xbrli:measure></xbrli:unit>'
        b'<xbrli:unit id="USD"><xbrli:measure>iso4217:USD</xbrli:measure></xbrli:unit>',
    ).replace(
        b'<id:Assets contextRef="CurrentYearInstant" unitRef="IDR">5000</id:Assets>',
        b'<id:Assets contextRef="CurrentYearInstant" unitRef="USD">5000</id:Assets>',
    )
    with pytest.raises(RuntimeError, match="CONTEXT_REJECTED"):
        _parse(mixed)


def test_parser_requires_ytd_duration_contexts() -> None:
    non_ytd = _xbrl().replace(b"<xbrli:startDate>2026-01-01</xbrli:startDate>", b"<xbrli:startDate>2026-02-01</xbrli:startDate>")
    with pytest.raises(RuntimeError, match="CONTEXT_REJECTED"):
        _parse(non_ytd)


def test_parser_rejects_incompatible_duration_contexts_for_cashflow_facts() -> None:
    payload = _xbrl(extra="""
      <xbrli:context id="AlternateYTD">
        <xbrli:entity><xbrli:identifier scheme="http://www.idx.co.id/xbrl">test_maker</xbrli:identifier></xbrli:entity>
        <xbrli:period><xbrli:startDate>2026-01-02</xbrli:startDate><xbrli:endDate>2026-03-31</xbrli:endDate></xbrli:period>
      </xbrli:context>
    """)
    payload = payload.replace(
        b'NetCashFlowsReceivedFromUsedInOperatingActivities contextRef="CurrentYearDuration"',
        b'NetCashFlowsReceivedFromUsedInOperatingActivities contextRef="AlternateYTD"',
    ).replace(
        b'PaymentsForAcquisitionOfPropertyAndEquipment contextRef="CurrentYearDuration"',
        b'PaymentsForAcquisitionOfPropertyAndEquipment contextRef="AlternateYTD"',
    ).replace(
        b'PaymentsForAcquisitionOfIntangibleAssets contextRef="CurrentYearDuration"',
        b'PaymentsForAcquisitionOfIntangibleAssets contextRef="AlternateYTD"',
    )
    with pytest.raises(RuntimeError, match="CONTEXT_REJECTED"):
        _parse(payload)


def test_validator_rejects_missing_or_mixed_currency() -> None:
    _, facts = _parse()
    missing = [dict(row) for row in facts]
    missing[0]["currency"] = None
    assert validate_financial_facts(missing, ticker="TEST", report_period="2026-Q1") == (
        False, "CONTEXT_REJECTED"
    )
    mixed = [dict(row) for row in facts]
    mixed[0]["currency"] = "USD"
    assert validate_financial_facts(mixed, ticker="TEST", report_period="2026-Q1") == (
        False, "CONTEXT_REJECTED"
    )


@pytest.mark.parametrize(
    "payload,reason",
    [
        (_xbrl(issuer="other_maker"), "ISSUER_MISMATCH"),
        (_xbrl(issuer=None), "ISSUER_IDENTITY_MISSING"),
        (_xbrl(scheme="https://example.com/xbrl"), "ISSUER_MISMATCH"),
        (_xbrl(period_end="2025-12-31"), "CONTEXT_REJECTED"),
        (b"<broken", "PARSE_FAILURE"),
    ],
)
def test_parser_fails_closed_for_wrong_issuer_period_and_malformed_xml(payload: bytes, reason: str) -> None:
    with pytest.raises(RuntimeError, match=reason):
        _parse(payload)


def test_zip_attachment_is_bounded_and_parsed() -> None:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("instance.xml", _xbrl())
        archive.writestr("readme.txt", "ignored")
    report, facts = _parse(buffer.getvalue())
    assert report["validation_state"] == "VALID" and len(facts) == 12


def test_pipeline_discovers_one_period_downloads_once_and_persists_reports_before_facts() -> None:
    backend = MemoryBackend()
    session = _success_session()
    rows, meta = _producer(backend, session).get_period("TEST.JK", 2026, "tw1")
    assert len(rows) == 12 and meta["state"] == "REFRESHED"
    assert meta["api_calls"] == 1 and meta["attachment_calls"] == 1 and len(session.calls) == 2
    assert session.calls[0]["url"] == ZAPI_FINANCIAL_REPORT_URL
    assert session.calls[0]["params"] == {"year": 2026, "period": "tw1", "code": "TEST", "length": 100, "start": 0}
    assert session.calls[0]["allow_redirects"] is False
    assert session.calls[1]["allow_redirects"] is False
    assert session.calls[0]["timeout"] == REQUEST_TIMEOUT_SECONDS
    assert session.calls[0]["headers"]["User-Agent"] == "Shared-IDX-Evidence-Hub/financial-manifest"
    assert session.calls[1]["headers"]["User-Agent"] == "Shared-IDX-Evidence-Hub/financial-attachment"
    assert len(backend.tables["evidence_financial_reports"]) == 1
    assert len(backend.tables["evidence_financial_facts"]) == 12
    assert "test-key" not in str(meta)


@pytest.mark.parametrize("first,second", [("PASTICUAN", "EMIR"), ("EMIR", "PASTICUAN")])
def test_second_scanner_reuses_verified_period_without_zapi_key(first: str, second: str) -> None:
    backend = MemoryBackend()
    first_rows, _ = _producer(backend, _success_session(), client=first).get_period("TEST", 2026, "tw1")
    second_session = Session([])
    second_rows, meta = _producer(backend, second_session, client=second, api_key="").get_period("TEST", 2026, "tw1")
    assert first_rows == second_rows and not second_session.calls
    assert meta["cache_hit"] and meta["request_avoided"] and meta["api_calls"] == 0


def test_missing_key_on_cache_miss_makes_no_request() -> None:
    session = Session([])
    rows, meta = _producer(MemoryBackend(), session, api_key="").get_period("TEST", 2026, "tw1")
    assert not rows and not session.calls and meta["state"] == "ENVIRONMENT_BLOCKED"


@pytest.mark.parametrize(
    "status,reason",
    [
        (401, "ZAPI_MANIFEST_HTTP_401"),
        (403, "ZAPI_MANIFEST_HTTP_403"),
        (404, "ZAPI_MANIFEST_HTTP_404"),
        (429, "ZAPI_MANIFEST_HTTP_429"),
    ],
)
def test_discovery_http_failures_are_explicit(status: int, reason: str) -> None:
    rows, meta = _producer(MemoryBackend(), Session([Response(status=status)])).get_period("TEST", 2026, "tw1")
    assert not rows and meta["state"] == reason


@pytest.mark.parametrize(
    "outcome,reason",
    [
        (requests.Timeout("slow"), "ZAPI_MANIFEST_TIMEOUT"),
        (requests.ConnectionError("offline"), "ZAPI_MANIFEST_CONNECTION_ERROR"),
    ],
)
def test_network_failures_are_explicit(outcome: Exception, reason: str) -> None:
    rows, meta = _producer(MemoryBackend(), Session([outcome])).get_period("TEST", 2026, "tw1")
    assert not rows and meta["state"] == reason




@pytest.mark.parametrize(
    "status,reason",
    [
        (401, "OFFICIAL_ATTACHMENT_HTTP_401"),
        (403, "OFFICIAL_ATTACHMENT_HTTP_403"),
        (404, "OFFICIAL_ATTACHMENT_HTTP_404"),
        (429, "OFFICIAL_ATTACHMENT_HTTP_429"),
    ],
)
def test_attachment_http_failures_are_stage_specific(status: int, reason: str) -> None:
    session = Session([
        Response(payload=_manifest()),
        Response(status=status, url=OFFICIAL_URL),
    ])
    rows, meta = _producer(MemoryBackend(), session).get_period("TEST", 2026, "tw1")
    assert not rows and meta["state"] == reason
    assert meta["api_calls"] == 1 and meta["attachment_calls"] == 1


def test_attachment_redirect_is_blocked() -> None:
    session = Session([
        Response(payload=_manifest()),
        Response(status=302, url=OFFICIAL_URL),
    ])
    rows, meta = _producer(MemoryBackend(), session).get_period("TEST", 2026, "tw1")
    assert not rows and meta["state"] == "OFFICIAL_ATTACHMENT_REDIRECT_BLOCKED"
    assert session.calls[-1]["allow_redirects"] is False


@pytest.mark.parametrize(
    "response,reason",
    [
        (Response(content=b""), "EMPTY_RESPONSE"),
        (Response(malformed=True), "PARSE_FAILURE"),
        (Response(payload={"unexpected": []}), "PARSE_FAILURE"),
        (Response(payload={"data": []}), "NO_REPORT"),
        (Response(payload=_manifest(code="OTHER")), "ISSUER_MISMATCH"),
        (Response(payload=_manifest(year="2025")), "WRONG_PERIOD"),
        (Response(payload=_manifest(filename="annual-report.pdf")), "NO_REPORT"),
    ],
)
def test_discovery_empty_malformed_wrong_issuer_period_and_non_xbrl_fail_closed(response: Response, reason: str) -> None:
    rows, meta = _producer(MemoryBackend(), Session([response])).get_period("TEST", 2026, "tw1")
    assert not rows and meta["state"] == reason


@pytest.mark.parametrize(
    "attachment",
    [
        Response(content=_xbrl(), url="https://evil.example/instance.xml", content_type="application/xml"),
        Response(content=_xbrl(), url=OFFICIAL_URL, content_type="application/pdf"),
    ],
)
def test_attachment_requires_official_final_url_and_xbrl_content_type(attachment: Response) -> None:
    rows, meta = _producer(MemoryBackend(), Session([Response(payload=_manifest()), attachment])).get_period("TEST", 2026, "tw1")
    assert not rows and meta["state"] == "INVALID_CONTENT_TYPE"


def test_manifest_attachment_issuer_mismatch_is_never_source_verified() -> None:
    session = Session([Response(payload=_manifest(attachment_code="OTHER"))])
    rows, meta = _producer(MemoryBackend(), session).get_period("TEST", 2026, "tw1")
    assert not rows and meta["state"] == "ISSUER_MISMATCH" and meta["attachment_calls"] == 0


def test_persisted_rows_contain_no_scanner_conclusions() -> None:
    report, facts = _parse()
    forbidden = {"score", "rank", "gate", "signal", "recommendation", "entry", "stop_loss", "take_profit"}
    assert forbidden.isdisjoint(report)
    assert all(forbidden.isdisjoint(row) for row in facts)


def test_migration_financial_tables_use_exact_service_role_and_rls_contract() -> None:
    migration = next((Path(__file__).resolve().parents[1] / "database").glob("migration_v*_shared_evidence_hub.sql"))
    sql = migration.read_text(encoding="utf-8").lower()
    assert "create table if not exists public.evidence_financial_reports" in sql
    assert "create table if not exists public.evidence_financial_facts" in sql
    assert "foreign key (ticker, report_period, source_document_hash)" in sql
    assert "evidence_financial_document_readback_idx" in sql
    assert "'evidence_financial_reports'" in sql and "'evidence_financial_facts'" in sql
