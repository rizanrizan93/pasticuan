from __future__ import annotations

from datetime import date
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
    STOCK_SUMMARY_ATTEMPT_CAP,
    STOCK_SUMMARY_FAMILY,
    STOCK_SPEC,
    BoundedZapiTransport,
    DenyZapiTransport,
    GateFailure,
    RequestLedger,
    cache_only_consume,
    classify_rows,
    credentials_from_environment,
    ensure_scanner_neutral,
    execute_gate,
    facts_hash,
    fetch_foreign_flow,
    fetch_stock_summary,
    verify_delta_allowlist,
)


DAY = date(2026, 8, 31)
ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "full-forward-coverage.yml"


class FakeResponse:
    def __init__(self, payload=None, status_code=200):
        self._payload = payload if payload is not None else {}
        self.status_code = status_code

    def json(self):
        return self._payload


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
