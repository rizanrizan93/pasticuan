from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from tools.phase5_6_bounded_live_validation import (
    ALLOWED_DELTA_PATHS,
    COHORT,
    FOREIGN_FLOW_ATTEMPT_CAP,
    FOREIGN_FLOW_FAMILY,
    FOREIGN_FLOW_MAX_LENGTH,
    FOREIGN_SPEC,
    OFFICIAL_DOWNLOAD_CAP,
    OFFICIAL_HOSTS,
    OFFICIAL_HTTP_CAP,
    OFFICIAL_INDEX_URL,
    PARTICIPANT_PROVENANCE,
    PARTICIPANT_SOURCE,
    PARTICIPANT_SPEC,
    PARTICIPANT_TARGET_DATE,
    STOCK_SUMMARY_ATTEMPT_CAP,
    STOCK_SUMMARY_FAMILY,
    STOCK_SPEC,
    BoundedOfficialTransport,
    BoundedZapiTransport,
    DenyOfficialTransport,
    DenyZapiTransport,
    GateFailure,
    OfficialRequestLedger,
    RequestLedger,
    cache_only_consume,
    classify_participant_rows,
    classify_rows,
    credentials_from_environment,
    discover_participant_url,
    download_participant_file,
    ensure_scanner_neutral,
    execute_gate,
    execute_participant_gate,
    facts_hash,
    fetch_foreign_flow,
    fetch_stock_summary,
    parse_participant_rows,
    verify_delta_allowlist,
)


DAY = date(2026, 8, 31)
PARTICIPANT_DAY = date(2026, 9, 1)
ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "full-forward-coverage.yml"


class FakeResponse:
    def __init__(
        self,
        payload=None,
        status_code=200,
        *,
        content=b"",
        text="",
        content_type="application/json",
    ):
        self._payload = payload if payload is not None else {}
        self.status_code = status_code
        self.content = content
        self.text = text or (content.decode("utf-8") if content else "")
        self.headers = {"content-type": content_type, "content-length": str(len(content))}
        self.closed = False

    def json(self):
        return self._payload

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if not self.responses:
            raise AssertionError("UNPLANNED_PROVIDER_CALL")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class MemoryBackend:
    def __init__(self):
        self.tables = {}
        self.failures = []
        self.provider_states = []

    def read_rows(self, table, filters, *, select="*", limit=10000):
        del select
        values = []
        for row in self.tables.get(table, []):
            if all(str(row.get(name)) == str(value) for name, value in filters.items()):
                values.append(dict(row))
        return values[:limit]

    def upsert_rows(self, table, rows, *, conflict):
        current = [dict(row) for row in self.tables.get(table, [])]
        for record in (dict(row) for row in rows):
            key = tuple(record.get(name) for name in conflict)
            current = [
                row for row in current
                if tuple(row.get(name) for name in conflict) != key
            ]
            current.append(record)
        self.tables[table] = current
        return [dict(row) for row in rows]

    def acquire_lease(self, key, holder, lease_seconds):
        del key, holder, lease_seconds
        return {"acquired": True, "lease_state": "ACQUIRED"}

    def complete_lease(self, key, holder, state="COMPLETED"):
        del key, holder, state
        return True

    def fail_lease(self, key, holder, reason):
        del key, holder
        self.failures.append(reason)
        return True

    def record_provider_state(self, row):
        self.provider_states.append(dict(row))


def stock_payload():
    return {
        "data": {
            "date": DAY.isoformat(),
            "recordsTotal": len(COHORT),
            "data": [
                {
                    "StockCode": ticker,
                    "Date": DAY.isoformat(),
                    "Open": 100 + index,
                    "High": 110 + index,
                    "Low": 90 + index,
                    "Close": 105 + index,
                    "Previous": 99 + index,
                    "Volume": 1000 + index,
                    "Value": 100000 + index,
                    "Frequency": 50 + index,
                    "ForeignBuy": 500 + index,
                    "ForeignSell": 400 + index,
                }
                for index, ticker in enumerate(COHORT)
            ],
        }
    }


def foreign_row(ticker, *, target_date=DAY):
    index = COHORT.index(ticker)
    return {
        "code": ticker,
        "date": target_date.isoformat(),
        "foreignBuyShares": 1000 + index,
        "foreignSellShares": 700 + index,
        "netForeignShares": 300,
        "volume": 10000 + index,
        "value": 1000000 + index,
    }


def foreign_payload(*, total=None, rows=None, target_date=DAY):
    values = rows if rows is not None else [foreign_row(ticker, target_date=target_date) for ticker in COHORT]
    return {
        "data": {
            "date": target_date.isoformat(),
            "recordsTotal": len(values) if total is None else total,
            "data": values,
        }
    }


def participant_csv(*, target_date=PARTICIPANT_DAY, tickers=COHORT):
    lines = ["asset|participant_buy|participant_sell|volume|value|tradingdate"]
    for index, ticker in enumerate(tickers):
        lines.append(
            f"{ticker}|B{index:02d}|S{index:02d}|{100 + index}|{10000 + index}|{target_date.isoformat()}"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def participant_url(target_date=PARTICIPANT_DAY):
    return (
        "https://www.idxdata3.co.id/IDX%20Reporting%20PSPP/Revitalisasi/PUBLIK/"
        f"Trade-Detail-Publik_{target_date:%Y%m%d}.csv"
    )


def cached_participant_rows(body=None):
    payload = body or participant_csv()
    return parse_participant_rows(
        payload,
        PARTICIPANT_DAY,
        participant_url(),
        hashlib.sha256(payload).hexdigest(),
    )


@pytest.fixture(autouse=True)
def no_unplanned_network(monkeypatch):
    def blocked(*args, **kwargs):
        del args, kwargs
        raise AssertionError("LIVE_NETWORK_FORBIDDEN_IN_TEST")

    monkeypatch.setattr(requests.Session, "request", blocked)


def configure_fake_credentials(monkeypatch):
    monkeypatch.setenv("SHARED_EVIDENCE_SUPABASE_URL", "https://mbtsvflwszcgdtijdgas.supabase.co")
    monkeypatch.setenv("SHARED_EVIDENCE_SUPABASE_SECRET_KEY", "test-secret-never-printed")
    monkeypatch.setenv("ZAPI_KEY", "test-zapi-never-printed")


def test_delta_allowlist_accepts_only_three_authorized_files():
    calls = []

    def runner(args, **kwargs):
        calls.append((args, kwargs))
        if args[1] == "merge-base":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(
            returncode=0,
            stdout="\n".join(sorted(ALLOWED_DELTA_PATHS)) + "\n",
            stderr="",
        )

    changed = verify_delta_allowlist("a" * 40, repo_root=ROOT, runner=runner)
    assert set(changed) == set(ALLOWED_DELTA_PATHS)
    assert calls[0][0][-2:] == ["a" * 40, "HEAD"]


def test_delta_allowlist_rejects_production_source_change():
    def runner(args, **kwargs):
        del kwargs
        if args[1] == "merge-base":
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="scanner.py\n", stderr="")

    with pytest.raises(GateFailure, match="DELTA_ALLOWLIST_VIOLATION"):
        verify_delta_allowlist("b" * 40, repo_root=ROOT, runner=runner)


def test_workflow_is_manual_read_only_and_non_pushing():
    text = WORKFLOW.read_text(encoding="utf-8")
    trigger_block = text.split("permissions:", 1)[0]
    assert "workflow_dispatch:" in trigger_block
    for forbidden in (
        "push:", "pull_request:", "schedule:", "workflow_run:",
        "repository_dispatch:", "workflow_call:", "deployment:",
    ):
        assert forbidden not in trigger_block
    assert "permissions:\n  contents: read" in text
    assert "persist-credentials: false" in text
    assert "concurrency:" in text and "timeout-minutes:" in text
    assert "git push" not in text and "git commit" not in text
    assert "actions/cache" not in text


def test_stock_summary_transport_cap_is_exactly_one():
    assert STOCK_SUMMARY_ATTEMPT_CAP == 1
    ledger = RequestLedger("PASTICUAN", 0, {STOCK_SUMMARY_FAMILY: 1})
    session = FakeSession([FakeResponse(stock_payload()), FakeResponse(stock_payload())])
    transport = BoundedZapiTransport("fake", ledger, session=session)
    transport.get_json(
        STOCK_SUMMARY_FAMILY,
        "https://example.invalid/stock-summary",
        params={"date": DAY.isoformat(), "length": 5000, "start": 0},
    )
    with pytest.raises(GateFailure, match="ZAPI_CIRCUIT_BREAKER"):
        transport.get_json(
            STOCK_SUMMARY_FAMILY,
            "https://example.invalid/stock-summary",
            params={"date": DAY.isoformat(), "length": 5000, "start": 0},
        )
    assert len(session.calls) == 1
    assert ledger.attempts == 1 and ledger.refused_attempts == 1


def test_redirect_is_not_followed_or_accepted_as_valid_evidence():
    ledger = RequestLedger("PASTICUAN", 0, {STOCK_SUMMARY_FAMILY: 1})
    session = FakeSession([
        FakeResponse(stock_payload(), status_code=302),
        FakeResponse(stock_payload()),
    ])
    transport = BoundedZapiTransport("fake", ledger, session=session)

    with pytest.raises(GateFailure, match="CONTEXT_REJECTED"):
        fetch_stock_summary(transport, DAY)

    assert len(session.calls) == 1
    assert session.calls[0][2]["allow_redirects"] is False
    assert ledger.attempts == 1
    assert ledger.entries[0]["result"] == "HTTP_302"


def test_foreign_flow_bulk_first_all_cohort_needs_no_delta_or_full_market_completion():
    assert FOREIGN_FLOW_ATTEMPT_CAP == 6
    assert FOREIGN_FLOW_MAX_LENGTH == 200
    ledger = RequestLedger("EMIR", 1, {FOREIGN_FLOW_FAMILY: FOREIGN_FLOW_ATTEMPT_CAP})
    session = FakeSession([FakeResponse(foreign_payload(total=1000))])
    rows = fetch_foreign_flow(BoundedZapiTransport("fake", ledger, session=session), DAY)
    assert {row["ticker"] for row in rows} == set(COHORT)
    assert ledger.attempts == 1
    assert session.calls[0][2]["params"] == {
        "date": DAY.isoformat(), "length": 200, "start": 0, "sort": "code",
    }


def test_foreign_flow_missing_one_uses_exactly_one_delta_and_never_duplicates_bulk_ticker():
    bulk_rows = [foreign_row(ticker) for ticker in COHORT if ticker != "TLKM"]
    ledger = RequestLedger("EMIR", 1, {FOREIGN_FLOW_FAMILY: FOREIGN_FLOW_ATTEMPT_CAP})
    session = FakeSession([
        FakeResponse(foreign_payload(total=900, rows=bulk_rows)),
        FakeResponse(foreign_payload(rows=[foreign_row("TLKM")])),
    ])
    rows = fetch_foreign_flow(BoundedZapiTransport("fake", ledger, session=session), DAY)
    assert {row["ticker"] for row in rows} == set(COHORT)
    assert ledger.attempts == 2
    assert "code" not in session.calls[0][2]["params"]
    assert [call[2]["params"].get("code") for call in session.calls[1:]] == ["TLKM"]


def test_foreign_flow_missing_multiple_requests_only_missing_tickers_for_one_session():
    present = {"ASII", "BBRI", "TLKM"}
    bulk_rows = [foreign_row(ticker) for ticker in COHORT if ticker in present]
    ledger = RequestLedger("EMIR", 1, {FOREIGN_FLOW_FAMILY: FOREIGN_FLOW_ATTEMPT_CAP})
    session = FakeSession([
        FakeResponse(foreign_payload(total=900, rows=bulk_rows)),
        FakeResponse(foreign_payload(rows=[foreign_row("BBCA")])),
        FakeResponse(foreign_payload(rows=[foreign_row("BMRI")])),
    ])
    rows = fetch_foreign_flow(BoundedZapiTransport("fake", ledger, session=session), DAY)
    assert {row["ticker"] for row in rows} == set(COHORT)
    assert [call[2]["params"].get("code") for call in session.calls] == [None, "BBCA", "BMRI"]
    assert {call[2]["params"]["date"] for call in session.calls} == {DAY.isoformat()}
    assert all(call[2]["params"]["start"] == 0 for call in session.calls)
    assert all(call[2]["params"]["length"] <= 200 for call in session.calls)
    assert all(call[2]["params"]["sort"] == "code" for call in session.calls)


def test_foreign_flow_incomplete_bounded_result_fails_closed_after_six_attempts():
    ledger = RequestLedger("EMIR", 2, {FOREIGN_FLOW_FAMILY: FOREIGN_FLOW_ATTEMPT_CAP})
    session = FakeSession([FakeResponse(foreign_payload(rows=[])) for _ in range(6)])
    with pytest.raises(GateFailure, match="INSUFFICIENT_HISTORY"):
        fetch_foreign_flow(BoundedZapiTransport("fake", ledger, session=session), DAY)
    assert len(session.calls) == 6
    assert ledger.attempts == 6
    assert ledger.declared_cumulative_after == 8
    assert [call[2]["params"].get("code") for call in session.calls] == [None, *COHORT]
    assert all(call[2]["params"]["length"] <= FOREIGN_FLOW_MAX_LENGTH for call in session.calls)


def test_foreign_flow_seventh_attempt_is_refused_before_transport():
    ledger = RequestLedger("EMIR", 0, {FOREIGN_FLOW_FAMILY: FOREIGN_FLOW_ATTEMPT_CAP})
    session = FakeSession([FakeResponse(foreign_payload()) for _ in range(7)])
    transport = BoundedZapiTransport("fake", ledger, session=session)
    params = {"date": DAY.isoformat(), "length": 200, "start": 0, "sort": "code"}
    for _ in range(6):
        transport.get_json(FOREIGN_FLOW_FAMILY, "https://example.invalid/foreign-flow", params=params)
    with pytest.raises(GateFailure, match="ZAPI_CIRCUIT_BREAKER"):
        transport.get_json(FOREIGN_FLOW_FAMILY, "https://example.invalid/foreign-flow", params=params)
    assert len(session.calls) == 6
    assert ledger.attempts == 6 and ledger.refused_attempts == 1


def test_foreign_flow_delta_wrong_ticker_fails_closed():
    bulk_rows = [foreign_row(ticker) for ticker in COHORT if ticker != "TLKM"]
    ledger = RequestLedger("EMIR", 0, {FOREIGN_FLOW_FAMILY: FOREIGN_FLOW_ATTEMPT_CAP})
    session = FakeSession([
        FakeResponse(foreign_payload(rows=bulk_rows)),
        FakeResponse(foreign_payload(rows=[foreign_row("BMRI")])),
    ])
    with pytest.raises(GateFailure, match="TICKER_MISMATCH"):
        fetch_foreign_flow(BoundedZapiTransport("fake", ledger, session=session), DAY)
    assert session.calls[1][2]["params"]["code"] == "TLKM"


def test_foreign_flow_missing_factual_value_is_not_coerced_to_zero():
    bulk_rows = [foreign_row(ticker) for ticker in COHORT if ticker != "TLKM"]
    malformed_delta = foreign_row("TLKM")
    malformed_delta.pop("netForeignShares")
    ledger = RequestLedger("EMIR", 0, {FOREIGN_FLOW_FAMILY: FOREIGN_FLOW_ATTEMPT_CAP})
    session = FakeSession([
        FakeResponse(foreign_payload(rows=bulk_rows)),
        FakeResponse(foreign_payload(rows=[malformed_delta])),
    ])
    with pytest.raises(GateFailure, match="PARSE_FAILURE"):
        fetch_foreign_flow(BoundedZapiTransport("fake", ledger, session=session), DAY)


def test_foreign_flow_redirect_is_disabled_and_fails_closed():
    ledger = RequestLedger("EMIR", 0, {FOREIGN_FLOW_FAMILY: FOREIGN_FLOW_ATTEMPT_CAP})
    session = FakeSession([FakeResponse(foreign_payload(), status_code=302)])
    with pytest.raises(GateFailure, match="CONTEXT_REJECTED"):
        fetch_foreign_flow(BoundedZapiTransport("fake", ledger, session=session), DAY)
    assert len(session.calls) == 1
    assert session.calls[0][2]["allow_redirects"] is False


def test_malformed_provider_payload_fails_closed():
    ledger = RequestLedger("EMIR", 0, {FOREIGN_FLOW_FAMILY: FOREIGN_FLOW_ATTEMPT_CAP})
    session = FakeSession([
        FakeResponse({"data": {"date": DAY.isoformat(), "data": {"malformed": True}}})
    ])

    with pytest.raises(GateFailure, match="PARSE_FAILURE"):
        fetch_foreign_flow(BoundedZapiTransport("fake", ledger, session=session), DAY)

    assert len(session.calls) == 1
    assert ledger.attempts == 1
    assert ledger.entries[0]["result"] == "HTTP_200"


def test_wrong_date_session_is_rejected():
    wrong_day = date(2026, 8, 28)
    payload = foreign_payload()
    payload["data"]["date"] = wrong_day.isoformat()
    for row in payload["data"]["data"]:
        row["date"] = wrong_day.isoformat()
    ledger = RequestLedger("EMIR", 0, {FOREIGN_FLOW_FAMILY: FOREIGN_FLOW_ATTEMPT_CAP})
    session = FakeSession([FakeResponse(payload)])

    with pytest.raises(GateFailure, match="WRONG_PERIOD"):
        fetch_foreign_flow(BoundedZapiTransport("fake", ledger, session=session), DAY)

    assert len(session.calls) == 1
    assert session.calls[0][2]["params"]["date"] == DAY.isoformat()
    assert ledger.attempts == 1


def test_cache_only_transport_hard_denies_before_network():
    ledger = RequestLedger("EMIR", 0, {})
    transport = DenyZapiTransport(ledger)
    with pytest.raises(GateFailure, match="CACHE_ONLY_PROVIDER_ACCESS_DENIED"):
        transport.get_json(
            STOCK_SUMMARY_FAMILY,
            "https://example.invalid",
            params={"date": DAY.isoformat()},
        )
    assert ledger.attempts == 0 and ledger.refused_attempts == 1


def test_missing_shared_credentials_fails_closed(monkeypatch):
    for name in (
        "SHARED_EVIDENCE_SUPABASE_URL",
        "SHARED_EVIDENCE_SUPABASE_SECRET_KEY",
        "SHARED_EVIDENCE_SUPABASE_SERVICE_ROLE_KEY",
        "ZAPI_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    config, status = credentials_from_environment("PASTICUAN")
    report, exit_code = execute_gate("PASTICUAN", 0)
    assert config is None and status["shared_db_key"] == "MISSING"
    assert report["state"] == "CREDENTIAL_MISSING" and exit_code == 2
    assert report["request_ledger"]["per_run_attempts"] == 0


def test_wrong_shared_db_project_fails_closed_without_exposing_url(monkeypatch):
    monkeypatch.setenv("SHARED_EVIDENCE_SUPABASE_URL", "https://wrongproject.supabase.co")
    monkeypatch.setenv("SHARED_EVIDENCE_SUPABASE_SECRET_KEY", "test-secret-never-printed")
    config, status = credentials_from_environment("PASTICUAN")
    assert config is None
    assert status["shared_db_project"] == "MISMATCH"
    assert "wrongproject" not in json.dumps(status)


def test_missing_error_and_stale_are_never_zero_or_neutral():
    state, reason, rows = classify_rows([], STOCK_SPEC, DAY)
    assert (state, reason, rows) == ("MISSING", "MISSING", [])
    invalid = [{
        "provider": "ZAPI", "trade_date": DAY.isoformat(), "ticker": ticker,
        "validation_state": "ERROR", "freshness_state": "CURRENT",
    } for ticker in COHORT]
    state, reason, _ = classify_rows(invalid, STOCK_SPEC, DAY)
    assert state == "ERROR" and reason == "PROVIDER_ERROR"
    assert facts_hash([], STOCK_SPEC) != facts_hash([{"ticker": "ZERO"}], STOCK_SPEC)


def test_scanner_conclusions_are_rejected_from_shared_evidence():
    row = {
        "provider": "ZAPI", "trade_date": DAY.isoformat(), "ticker": "BBCA",
        "validation_state": "VALID", "freshness_state": "CURRENT",
        "scanner_score": 99,
    }
    with pytest.raises(GateFailure, match="SCANNER_SEMANTIC_FIELD_REJECTED"):
        ensure_scanner_neutral([row], STOCK_SPEC)


def test_provider_error_remains_error(monkeypatch):
    configure_fake_credentials(monkeypatch)
    backend = MemoryBackend()
    report, exit_code = execute_gate(
        "PASTICUAN",
        0,
        backend=backend,
        transport_session=FakeSession([FakeResponse(status_code=500)]),
        target_date=DAY,
    )
    assert exit_code == 1
    assert report["mode"] == "ERROR"
    assert report["evidence"]["classification"] == "ERROR"
    assert report["request_ledger"]["per_run_attempts"] == 1


def test_exact_db_readback_mismatch_fails_closed(monkeypatch):
    class CorruptingReadbackBackend(MemoryBackend):
        def __init__(self):
            super().__init__()
            self.corrupt_readback = False

        def upsert_rows(self, table, rows, *, conflict):
            written = super().upsert_rows(table, rows, conflict=conflict)
            self.corrupt_readback = True
            return written

        def read_rows(self, table, filters, *, select="*", limit=10000):
            rows = super().read_rows(table, filters, select=select, limit=limit)
            if self.corrupt_readback and table == STOCK_SPEC.table and rows:
                rows[0]["close"] = (rows[0].get("close") or 0) + 1
            return rows

    configure_fake_credentials(monkeypatch)
    backend = CorruptingReadbackBackend()
    session = FakeSession([FakeResponse(stock_payload())])

    with pytest.raises(GateFailure, match="READBACK_FAILURE"):
        execute_gate(
            "PASTICUAN",
            0,
            backend=backend,
            transport_session=session,
            target_date=DAY,
        )

    assert len(session.calls) == 1


def test_three_run_state_machine_proves_two_family_reuse(monkeypatch):
    configure_fake_credentials(monkeypatch)
    backend = MemoryBackend()

    pasticuan_a, code_a = execute_gate(
        "PASTICUAN", 0, backend=backend,
        transport_session=FakeSession([FakeResponse(stock_payload())]),
        target_date=DAY,
    )
    assert code_a == 0 and pasticuan_a["state"] == "PASTICUAN_STATE_A"
    assert pasticuan_a["request_ledger"]["per_run_attempts"] == 1
    assert pasticuan_a["evidence"]["identical_readback"]

    emir_b, code_b = execute_gate(
        "EMIR", 1, backend=backend,
        transport_session=FakeSession([FakeResponse(foreign_payload())]),
        target_date=DAY,
    )
    assert code_b == 0 and emir_b["state"] == "EMIR_STATE_B"
    assert emir_b["stock_summary_cache"]["reason"] == "CACHE_ONLY_VALID"
    assert emir_b["request_ledger"]["per_run_attempts"] == 1
    assert emir_b["stock_summary_cache"]["facts_hash"] == pasticuan_a["evidence"]["facts_hash"]

    pasticuan_c, code_c = execute_gate(
        "PASTICUAN", 2, backend=backend, target_date=DAY
    )
    assert code_c == 0 and pasticuan_c["state"] == "PASTICUAN_STATE_C"
    assert pasticuan_c["request_ledger"]["per_run_attempts"] == 0
    assert pasticuan_c["evidence"]["facts_hash"] == emir_b["evidence"]["facts_hash"]
    assert pasticuan_c["evidence"]["identical_readback"]


def test_ledger_is_deterministic_and_machine_readable():
    first = RequestLedger("EMIR", 1, {FOREIGN_FLOW_FAMILY: FOREIGN_FLOW_ATTEMPT_CAP})
    second = RequestLedger("EMIR", 1, {FOREIGN_FLOW_FAMILY: FOREIGN_FLOW_ATTEMPT_CAP})
    params = {"date": DAY.isoformat(), "length": 200, "start": 0, "sort": "code"}
    for ledger in (first, second):
        index = ledger.before_attempt(FOREIGN_FLOW_FAMILY, "/foreign-flow", params)
        ledger.finish_attempt(index, "HTTP_200")
    assert first.as_dict() == second.as_dict()
    assert json.loads(json.dumps(first.as_dict(), sort_keys=True)) == first.as_dict()


def test_delta_contains_no_schema_acl_or_migration_file():
    assert not any(
        path.startswith("database/") or path.endswith(".sql")
        for path in ALLOWED_DELTA_PATHS
    )


def test_participant_workflow_selector_is_explicit_and_zapi_isolated():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "validation_family:" in text
    assert "- zapi" in text and "- participant" in text
    assert 'default: "2026-09-01"' in text
    participant_step = text.split(
        "- name: Execute bounded official participant state machine", 1
    )[1]
    assert "ZAPI_KEY" not in participant_step
    assert "--validation-family participant" in participant_step
    assert "--target-date" in participant_step


def test_participant_target_is_exactly_one_explicit_session():
    assert PARTICIPANT_TARGET_DATE == PARTICIPANT_DAY
    body = participant_csv()
    rows = parse_participant_rows(
        body, PARTICIPANT_DAY, participant_url(), hashlib.sha256(body).hexdigest()
    )
    assert {row["trade_date"] for row in rows} == {"2026-09-01"}


def test_official_host_allowlist_rejects_before_transport():
    assert OFFICIAL_HOSTS == {"idxdata3.co.id", "www.idxdata3.co.id"}
    ledger = OfficialRequestLedger("PASTICUAN")
    session = FakeSession([])
    transport = BoundedOfficialTransport(ledger, session=session)
    with pytest.raises(GateFailure, match="CONTEXT_REJECTED"):
        transport.request("https://example.invalid/file.csv", "DOWNLOAD", file_download=True)
    assert session.calls == [] and ledger.entries == [] and ledger.downloads == 0


def test_official_transport_disables_redirects_has_finite_timeout_and_no_retries():
    ledger = OfficialRequestLedger("PASTICUAN")
    session = FakeSession([
        FakeResponse(status_code=302, content_type="text/plain"),
    ])
    transport = BoundedOfficialTransport(ledger, session=session, timeout_seconds=12)
    with pytest.raises(GateFailure, match="HTTP_302"):
        transport.request(OFFICIAL_INDEX_URL, "DIRECTORY_INDEX")
    assert session.calls[0][2]["allow_redirects"] is False
    assert session.calls[0][2]["timeout"] == 12
    mounted = BoundedOfficialTransport(OfficialRequestLedger("P")).session.adapters
    assert mounted["https://"].max_retries.total == 0


def test_official_request_count_occurs_before_transport_and_sixth_is_refused():
    ledger = OfficialRequestLedger("PASTICUAN")
    session = FakeSession([
        FakeResponse(text="index", content_type="text/plain") for _ in range(OFFICIAL_HTTP_CAP + 1)
    ])
    transport = BoundedOfficialTransport(ledger, session=session)
    for _ in range(OFFICIAL_HTTP_CAP):
        response = transport.request(OFFICIAL_INDEX_URL, "DIRECTORY_INDEX")
        response.close()
    with pytest.raises(GateFailure, match="OFFICIAL_HTTP_CIRCUIT_BREAKER"):
        transport.request(OFFICIAL_INDEX_URL, "DIRECTORY_INDEX")
    assert len(session.calls) == OFFICIAL_HTTP_CAP
    assert len(ledger.entries) == OFFICIAL_HTTP_CAP and ledger.refused_attempts == 1


def test_official_file_download_cap_refuses_second_before_transport():
    assert OFFICIAL_DOWNLOAD_CAP == 1
    ledger = OfficialRequestLedger("PASTICUAN")
    session = FakeSession([
        FakeResponse(content=b"first", content_type="text/csv"),
        FakeResponse(content=b"second", content_type="text/csv"),
    ])
    transport = BoundedOfficialTransport(ledger, session=session)
    transport.request(participant_url(), "DOWNLOAD", file_download=True).close()
    with pytest.raises(GateFailure, match="OFFICIAL_DOWNLOAD_CIRCUIT_BREAKER"):
        transport.request(participant_url(), "DOWNLOAD", file_download=True)
    assert len(session.calls) == 1 and ledger.downloads == 1 and ledger.refused_downloads == 1


def test_directory_index_discovery_and_single_download_success():
    href = participant_url()
    ledger = OfficialRequestLedger("PASTICUAN")
    session = FakeSession([
        FakeResponse(text=f'<a href="{href}">file</a>', content_type="text/plain"),
        FakeResponse(content=participant_csv(), content_type="text/csv"),
    ])
    body, source_url, digest = download_participant_file(
        BoundedOfficialTransport(ledger, session=session), PARTICIPANT_DAY
    )
    assert body == participant_csv() and source_url == href
    assert digest == hashlib.sha256(body).hexdigest()
    assert len(session.calls) == 2 and ledger.downloads == 1


def test_documented_path_probe_success_within_five_request_envelope():
    ledger = OfficialRequestLedger("PASTICUAN")
    session = FakeSession([
        FakeResponse(text="no matching link", content_type="text/plain"),
        FakeResponse(status_code=404, content_type="text/csv"),
        FakeResponse(content_type="text/csv"),
    ])
    found = discover_participant_url(
        BoundedOfficialTransport(ledger, session=session), PARTICIPANT_DAY
    )
    assert found.startswith("https://idxdata3.co.id/")
    assert len(session.calls) == 3 and ledger.downloads == 0


def test_all_documented_paths_404_fail_closed_without_download():
    ledger = OfficialRequestLedger("PASTICUAN")
    session = FakeSession([
        FakeResponse(text="none", content_type="text/plain"),
        *[FakeResponse(status_code=404, content_type="text/csv") for _ in range(3)],
    ])
    with pytest.raises(GateFailure, match="HTTP_404"):
        discover_participant_url(
            BoundedOfficialTransport(ledger, session=session), PARTICIPANT_DAY
        )
    assert len(session.calls) == 4 and ledger.downloads == 0


@pytest.mark.parametrize("status", [403, 429])
def test_official_403_and_429_fail_closed(status):
    ledger = OfficialRequestLedger("PASTICUAN")
    session = FakeSession([FakeResponse(status_code=status, content_type="text/plain")])
    with pytest.raises(GateFailure, match=f"HTTP_{status}"):
        discover_participant_url(
            BoundedOfficialTransport(ledger, session=session), PARTICIPANT_DAY
        )
    assert len(session.calls) == 1 and ledger.downloads == 0


@pytest.mark.parametrize(
    "error,reason",
    [(requests.ConnectionError("offline"), "CONNECTION_ERROR"), (requests.Timeout("late"), "TIMEOUT")],
)
def test_official_connection_and_timeout_fail_closed(error, reason):
    ledger = OfficialRequestLedger("PASTICUAN")
    session = FakeSession([error])
    with pytest.raises(GateFailure, match=reason):
        discover_participant_url(
            BoundedOfficialTransport(ledger, session=session), PARTICIPANT_DAY
        )
    assert len(session.calls) == 1 and ledger.entries[0]["result"] == reason


def test_invalid_content_type_and_empty_file_body_fail_closed():
    invalid = BoundedOfficialTransport(
        OfficialRequestLedger("P"),
        session=FakeSession([FakeResponse(text="html", content_type="text/html")]),
    )
    with pytest.raises(GateFailure, match="PARSE_FAILURE"):
        discover_participant_url(invalid, PARTICIPANT_DAY)

    href = participant_url()
    empty_ledger = OfficialRequestLedger("P")
    empty = BoundedOfficialTransport(
        empty_ledger,
        session=FakeSession([
            FakeResponse(text=f'<a href="{href}">file</a>', content_type="text/plain"),
            FakeResponse(content=b"", content_type="text/csv"),
        ]),
    )
    with pytest.raises(GateFailure, match="EMPTY_RESPONSE"):
        download_participant_file(empty, PARTICIPANT_DAY)
    assert empty_ledger.downloads == 1


def test_malformed_csv_and_wrong_trade_date_fail_closed():
    with pytest.raises(GateFailure, match="PARSE_FAILURE"):
        parse_participant_rows(
            b"not,a,trade,detail\n", PARTICIPANT_DAY, participant_url(), "a" * 64
        )
    wrong = participant_csv(target_date=date(2026, 8, 31))
    with pytest.raises(GateFailure, match="WRONG_PERIOD"):
        parse_participant_rows(
            wrong, PARTICIPANT_DAY, participant_url(), hashlib.sha256(wrong).hexdigest()
        )


def test_participant_requires_all_five_tickers_without_inventing_zero_rows():
    four = participant_csv(tickers=COHORT[:-1])
    with pytest.raises(GateFailure, match="INSUFFICIENT_HISTORY"):
        parse_participant_rows(
            four, PARTICIPANT_DAY, participant_url(), hashlib.sha256(four).hexdigest()
        )


def test_participant_aggregation_and_file_hash_are_deterministic():
    body = participant_csv()
    digest = hashlib.sha256(body).hexdigest()
    first = parse_participant_rows(body, PARTICIPANT_DAY, participant_url(), digest)
    second = parse_participant_rows(body, PARTICIPANT_DAY, participant_url(), digest)
    assert facts_hash(first, PARTICIPANT_SPEC) == facts_hash(second, PARTICIPANT_SPEC)
    assert {row["source_file_hash"] for row in first} == {digest}
    assert {row["ticker"] for row in first} == set(COHORT)


def test_participant_semantics_reject_scanner_conclusions_and_preserve_disclaimer():
    rows = cached_participant_rows()
    assert all(row["provenance_state"] == PARTICIPANT_PROVENANCE for row in rows)
    assert "NOT_BENEFICIAL_OWNER" in PARTICIPANT_PROVENANCE
    contaminated = {**rows[0], "recommendation": "BUY"}
    with pytest.raises(GateFailure, match="SCANNER_SEMANTIC_FIELD_REJECTED"):
        ensure_scanner_neutral([contaminated], PARTICIPANT_SPEC)


def test_participant_cache_only_transport_hard_denies_before_network():
    ledger = OfficialRequestLedger("EMIR")
    with pytest.raises(GateFailure, match="CACHE_ONLY_OFFICIAL_ACCESS_DENIED"):
        DenyOfficialTransport(ledger).request(participant_url(), "DOWNLOAD", file_download=True)
    assert ledger.entries == [] and ledger.downloads == 0 and ledger.refused_attempts == 1


def test_pasticuan_producer_persists_exact_readback_without_zapi(monkeypatch):
    configure_fake_credentials(monkeypatch)
    monkeypatch.setattr(
        "tools.phase5_6_bounded_live_validation.BoundedZapiTransport",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("ZAPI_FORBIDDEN")),
    )
    backend = MemoryBackend()
    href = participant_url()
    session = FakeSession([
        FakeResponse(text=f'<a href="{href}">file</a>', content_type="text/plain"),
        FakeResponse(content=participant_csv(), content_type="text/csv"),
    ])
    report, code = execute_participant_gate(
        "PASTICUAN", backend=backend, official_session=session,
        target_date=PARTICIPANT_DAY, declared_cumulative_zapi=4,
    )
    assert code == 0 and report["mode"] == "OFFICIAL_LIVE_PRODUCER"
    assert report["evidence"]["ticker_breadth"] == 5
    assert report["evidence"]["identical_readback"]
    assert report["zapi_ledger"]["declared_cumulative_after"] == 4
    assert report["zapi_ledger"]["per_run_attempts"] == 0
    assert report["official_request_ledger"]["official_downloads"] == 1


def test_participant_readback_mismatch_fails_closed(monkeypatch):
    class CorruptingBackend(MemoryBackend):
        def __init__(self):
            super().__init__()
            self.corrupt = False

        def upsert_rows(self, table, rows, *, conflict):
            written = super().upsert_rows(table, rows, conflict=conflict)
            if table == PARTICIPANT_SPEC.table:
                self.corrupt = True
            return written

        def read_rows(self, table, filters, *, select="*", limit=10000):
            rows = super().read_rows(table, filters, select=select, limit=limit)
            if self.corrupt and table == PARTICIPANT_SPEC.table and rows:
                rows[0]["buy_value"] += 1
            return rows

    configure_fake_credentials(monkeypatch)
    href = participant_url()
    with pytest.raises(GateFailure, match="READBACK_FAILURE"):
        execute_participant_gate(
            "PASTICUAN",
            backend=CorruptingBackend(),
            official_session=FakeSession([
                FakeResponse(text=f'<a href="{href}">file</a>', content_type="text/plain"),
                FakeResponse(content=participant_csv(), content_type="text/csv"),
            ]),
            target_date=PARTICIPANT_DAY,
        )


def test_pasticuan_producer_then_emir_cache_only_identical_hash(monkeypatch):
    configure_fake_credentials(monkeypatch)
    backend = MemoryBackend()
    href = participant_url()
    producer, producer_code = execute_participant_gate(
        "PASTICUAN", backend=backend,
        official_session=FakeSession([
            FakeResponse(text=f'<a href="{href}">file</a>', content_type="text/plain"),
            FakeResponse(content=participant_csv(), content_type="text/csv"),
        ]),
        target_date=PARTICIPANT_DAY, declared_cumulative_zapi=4,
    )
    consumer_session = FakeSession([])
    consumer, consumer_code = execute_participant_gate(
        "EMIR", backend=backend, official_session=consumer_session,
        target_date=PARTICIPANT_DAY, declared_cumulative_zapi=4,
    )
    assert producer_code == consumer_code == 0
    assert consumer["mode"] == "CACHE_ONLY_CONSUMER"
    assert consumer_session.calls == []
    assert consumer["official_request_ledger"]["official_http_attempts"] == 0
    assert consumer["official_request_ledger"]["official_downloads"] == 0
    assert producer["evidence"]["facts_hash"] == consumer["evidence"]["facts_hash"]
    assert consumer["zapi_ledger"]["declared_cumulative_after"] == 4
