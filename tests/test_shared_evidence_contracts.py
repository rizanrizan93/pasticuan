from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
import requests

from shared_evidence_hub import (
    EvidenceKey,
    EvidenceState,
    MissingReason,
    SecretStatus,
    SharedEvidenceCoordinator,
    normalize_failure_reason,
    secret_status,
)
from shared_evidence_temporal import (
    EvidenceDateKind,
    evidence_available_as_of,
    filter_evidence_as_of,
)


AS_OF = "2026-08-31T17:00:00+07:00"


def test_all_five_read_classifications_are_explicit() -> None:
    assert {state.value for state in EvidenceState} == {"VALID", "STALE", "MISSING", "INSUFFICIENT", "ERROR"}


def test_taxonomy_exactly_contains_master_contract() -> None:
    expected = {
        "NO_REPORT", "NO_FILE", "NO_MATCH", "HTTP_401", "HTTP_403", "HTTP_404", "HTTP_429",
        "TIMEOUT", "CONNECTION_ERROR", "RATE_LIMIT", "QUOTA_EXHAUSTED", "PARSE_FAILURE",
        "ISSUER_MISMATCH", "ISSUER_IDENTITY_MISSING", "CONTEXT_REJECTED", "WRONG_PERIOD",
        "STALE", "INSUFFICIENT_HISTORY", "PROVIDER_NO_DATA", "INVALID_CONTENT_TYPE",
        "EMPTY_RESPONSE", "PERSIST_FAILURE", "READBACK_FAILURE", "REFRESH_LOCKED",
        "REFRESH_LEASE_EXPIRED", "ENVIRONMENT_BLOCKED",
    }
    assert {reason.value for reason in MissingReason} == expected


@pytest.mark.parametrize(
    "error,reason",
    [
        (requests.Timeout(), "TIMEOUT"),
        (requests.ConnectionError(), "CONNECTION_ERROR"),
        (RuntimeError("HTTP_401"), "HTTP_401"),
        (RuntimeError("HTTP_403"), "HTTP_403"),
        (RuntimeError("HTTP_404"), "HTTP_404"),
        (RuntimeError("HTTP_429"), "HTTP_429"),
        (RuntimeError("HTTP_408"), "TIMEOUT"),
        (RuntimeError("HTTP_422"), "CONTEXT_REJECTED"),
        (RuntimeError("HTTP_503"), "CONNECTION_ERROR"),
        (ValueError("bad payload"), "PARSE_FAILURE"),
        (RuntimeError("unknown provider failure"), "PROVIDER_NO_DATA"),
    ],
)
def test_failure_normalization_never_invents_exception_class_names(error: Exception, reason: str) -> None:
    assert normalize_failure_reason(error) == reason
    assert reason in {item.value for item in MissingReason}


@pytest.mark.parametrize(
    "value,rejected,expected",
    [
        (None, False, "MISSING"),
        ("", False, "EMPTY"),
        ("   ", False, "EMPTY"),
        ("configured-secret", False, "CONFIGURED"),
        ("configured-secret", True, "INVALID/REJECTED"),
    ],
)
def test_secret_status_returns_status_only(value, rejected: bool, expected: str) -> None:
    result = secret_status(value, rejected=rejected)
    assert result == expected and result in {item.value for item in SecretStatus}
    assert "configured-secret" not in result


def test_trade_date_requires_real_completed_idx_session() -> None:
    valid = evidence_available_as_of({"trade_date": "2026-08-31"}, as_of=AS_OF, date_kind="TRADE_DATE")
    holiday = evidence_available_as_of({"trade_date": "2026-08-25"}, as_of=AS_OF, date_kind="TRADE_DATE")
    before_close = evidence_available_as_of(
        {"trade_date": "2026-08-31"}, as_of="2026-08-31T15:00:00+07:00", date_kind="TRADE_DATE"
    )
    assert valid.available and valid.reason == "VALID"
    assert not holiday.available and holiday.reason == "WRONG_PERIOD"
    assert not before_close.available and before_close.reason == "CONTEXT_REJECTED"


def test_report_date_never_substitutes_for_publication_date() -> None:
    missing_publication = evidence_available_as_of(
        {"report_date": "2026-06-30"}, as_of=AS_OF, date_kind=EvidenceDateKind.REPORT_DATE
    )
    future_publication = evidence_available_as_of(
        {"report_date": "2026-06-30", "publication_date": "2026-09-02"},
        as_of=AS_OF, date_kind=EvidenceDateKind.REPORT_DATE,
    )
    visible = evidence_available_as_of(
        {"report_date": "2026-06-30", "publication_date": "2026-08-01"},
        as_of=AS_OF, date_kind=EvidenceDateKind.REPORT_DATE,
    )
    assert not missing_publication.available and missing_publication.reason == "CONTEXT_REJECTED"
    assert not future_publication.available and visible.available


def test_event_and_publication_dates_are_distinct() -> None:
    incoherent = evidence_available_as_of(
        {"event_date": "2026-09-02", "publication_date": "2026-08-01"},
        as_of=AS_OF, date_kind="EVENT_DATE",
    )
    published = evidence_available_as_of(
        {"event_date": "2026-08-01", "publication_date": "2026-08-02"},
        as_of=AS_OF, date_kind="EVENT_DATE",
    )
    assert not incoherent.available and incoherent.reason == "WRONG_PERIOD"
    assert published.available and published.evidence_date_kind == "EVENT_DATE"


def test_fetched_at_contract_is_timezone_aware() -> None:
    visible = evidence_available_as_of(
        {"fetched_at": "2026-08-31T09:00:00Z"}, as_of=AS_OF, date_kind="FETCHED_AT"
    )
    future = evidence_available_as_of(
        {"fetched_at": "2026-08-31T11:00:00Z"}, as_of=AS_OF, date_kind="FETCHED_AT"
    )
    assert visible.available and not future.available


def test_temporal_filter_counts_only_supplied_rows() -> None:
    rows = [
        {"publication_date": "2026-08-01", "id": 1},
        {"publication_date": "2026-09-01", "id": 2},
        {"id": 3},
    ]
    accepted, rejected = filter_evidence_as_of(rows, as_of=AS_OF, date_kind="PUBLICATION_DATE")
    assert accepted == [rows[0]]
    assert rejected == {"CONTEXT_REJECTED": 2}


class CompletionBackend:
    def __init__(self):
        self.rows = []
        self.states = []

    def acquire_lease(self, key, holder, lease_seconds): return {"acquired": True}
    def complete_lease(self, key, holder, state): return False
    def fail_lease(self, key, holder, reason): self.failed = reason; return True
    def record_provider_state(self, row): self.states.append(dict(row))


def test_failed_lease_completion_is_truthfully_classified() -> None:
    backend = CompletionBackend()
    coordinator = SharedEvidenceCoordinator(backend, client_id="EMIR", worker_id="worker")
    result = coordinator.get_or_refresh(
        EvidenceKey("ZAPI", "FIXTURE", "IDX_ALL", date(2026, 8, 31)),
        read_current=lambda: list(backend.rows),
        fetch=lambda: [{"validation_state": "VALID", "fetched_at": datetime.now(timezone.utc).isoformat()}],
        persist=lambda rows: backend.rows.extend(rows) or len(rows),
        validate=lambda rows: (True, "VALID"),
    )
    assert result.state is EvidenceState.ERROR and result.reason == "REFRESH_LEASE_EXPIRED"
    assert backend.failed == "REFRESH_LEASE_EXPIRED"
    assert backend.states[-1]["error_classification"] == "REFRESH_LEASE_EXPIRED"


def test_contract_modules_contain_no_secret_literals_or_scanner_decisions() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    text = "\n".join((root / name).read_text().lower() for name in (
        "shared_evidence_hub.py", "shared_evidence_temporal.py", "evidence_independence_audit.py",
    ))
    assert "zpi_x" not in text and "sb_secret_" not in text and "service_role_key=" not in text
    assert "entry_price" not in text and "take_profit" not in text and "stop_loss" not in text
