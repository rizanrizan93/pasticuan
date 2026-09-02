from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import threading
from typing import Any, Mapping

import pytest
import requests

from shared_evidence_hub import (
    EvidenceKey,
    EvidenceState,
    HubConfig,
    IDX_FLOW_SUPABASE_PROJECT_REF,
    MissingReason,
    SharedEvidenceCoordinator,
    SupabaseEvidenceBackend,
)


KEY = EvidenceKey("zapi", "stock_summary", "idx_all", date(2026, 8, 31))


class MemoryBackend:
    def __init__(self):
        self.rows: list[dict[str, Any]] = []
        self.leases: dict[tuple[str, str, str, date], dict[str, Any]] = {}
        self.provider_states: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    @staticmethod
    def _key(key: EvidenceKey) -> tuple[str, str, str, date]:
        normalized = key.normalized()
        return normalized.provider, normalized.family, normalized.scope, normalized.target_date

    def acquire_lease(self, key: EvidenceKey, holder: str, lease_seconds: int) -> Mapping[str, Any]:
        now = datetime.now(timezone.utc)
        identity = self._key(key)
        with self.lock:
            current = self.leases.get(identity)
            held = current and current["state"] == "HELD" and current["expires_at"] > now
            if held and current["holder"] != holder:
                return {"acquired": False, "lease_state": "HELD"}
            expired = bool(current and current["state"] == "HELD" and current["expires_at"] <= now)
            self.leases[identity] = {
                "state": "HELD",
                "holder": holder,
                "expires_at": now + timedelta(seconds=lease_seconds),
                "recovered": expired,
            }
            return {"acquired": True, "lease_state": "HELD", "expired_recovered": expired}

    def complete_lease(self, key: EvidenceKey, holder: str, state: str) -> bool:
        with self.lock:
            current = self.leases.get(self._key(key))
            if not current or current["holder"] != holder or current["state"] != "HELD":
                return False
            current["state"] = "COMPLETED"
            return True

    def fail_lease(self, key: EvidenceKey, holder: str, reason: str) -> bool:
        with self.lock:
            current = self.leases.get(self._key(key))
            if not current or current["holder"] != holder:
                return False
            current.update({"state": "FAILED", "reason": reason})
            return True

    def record_provider_state(self, row: Mapping[str, Any]) -> None:
        self.provider_states.append(dict(row))

    def read(self) -> list[Mapping[str, Any]]:
        with self.lock:
            return [dict(row) for row in self.rows]

    def persist(self, rows: list[Mapping[str, Any]]) -> int:
        with self.lock:
            by_ticker = {str(row["ticker"]): dict(row) for row in self.rows}
            for row in rows:
                by_ticker[str(row["ticker"])] = dict(row)
            self.rows = list(by_ticker.values())
        return len(rows)


def _valid_rows(ticker: str = "BBCA", *, fetched_at: datetime | None = None) -> list[dict[str, Any]]:
    stamp = fetched_at or datetime.now(timezone.utc)
    return [{
        "provider": "ZAPI",
        "trade_date": "2026-08-31",
        "ticker": ticker,
        "close": 8000,
        "validation_state": "VALID",
        "fetched_at": stamp.isoformat(),
    }]


def _validate(rows: list[Mapping[str, Any]]) -> tuple[bool, str]:
    return (bool(rows), "VALID" if rows else MissingReason.EMPTY_RESPONSE.value)


def _coordinator(backend: MemoryBackend, client: str) -> SharedEvidenceCoordinator:
    return SharedEvidenceCoordinator(backend, client_id=client, worker_id=f"{client}-worker")


def test_cache_hit_avoids_provider_call() -> None:
    backend = MemoryBackend()
    backend.rows = _valid_rows()
    calls = 0

    def fetch() -> list[Mapping[str, Any]]:
        nonlocal calls
        calls += 1
        return _valid_rows()

    result = _coordinator(backend, "EMIR").get_or_refresh(
        KEY, read_current=backend.read, fetch=fetch, persist=backend.persist, validate=_validate
    )
    assert result.state is EvidenceState.VALID
    assert result.cache_hit and result.request_avoided and not result.provider_called
    assert calls == 0


def test_stale_cache_triggers_one_refresh() -> None:
    backend = MemoryBackend()
    backend.rows = _valid_rows(fetched_at=datetime.now(timezone.utc) - timedelta(days=3))
    calls = 0

    def fetch() -> list[Mapping[str, Any]]:
        nonlocal calls
        calls += 1
        return _valid_rows()

    result = _coordinator(backend, "PASTICUAN").get_or_refresh(
        KEY,
        read_current=backend.read,
        fetch=fetch,
        persist=backend.persist,
        validate=_validate,
        max_age=timedelta(hours=24),
    )
    assert result.reason == "REFRESHED"
    assert result.provider_called and calls == 1


@pytest.mark.parametrize("first,second", [("EMIR", "PASTICUAN"), ("PASTICUAN", "EMIR")])
def test_second_scanner_reuses_first_scanner_facts(first: str, second: str) -> None:
    backend = MemoryBackend()
    calls: list[str] = []

    def fetch() -> list[Mapping[str, Any]]:
        calls.append(first)
        return _valid_rows()

    first_result = _coordinator(backend, first).get_or_refresh(
        KEY, read_current=backend.read, fetch=fetch, persist=backend.persist, validate=_validate
    )
    second_result = _coordinator(backend, second).get_or_refresh(
        KEY,
        read_current=backend.read,
        fetch=lambda: (_ for _ in ()).throw(AssertionError("duplicate provider call")),
        persist=backend.persist,
        validate=_validate,
    )
    assert first_result.provider_called
    assert second_result.cache_hit and second_result.request_avoided
    assert calls == [first]
    assert first_result.rows == second_result.rows


def test_concurrent_refresh_allows_only_one_provider_call() -> None:
    backend = MemoryBackend()
    entered, release = threading.Event(), threading.Event()
    calls = 0
    results: dict[str, Any] = {}

    def fetch() -> list[Mapping[str, Any]]:
        nonlocal calls
        calls += 1
        entered.set()
        assert release.wait(timeout=3)
        return _valid_rows()

    def first() -> None:
        results["first"] = _coordinator(backend, "EMIR").get_or_refresh(
            KEY, read_current=backend.read, fetch=fetch, persist=backend.persist, validate=_validate
        )

    thread = threading.Thread(target=first)
    thread.start()
    assert entered.wait(timeout=3)
    results["second"] = _coordinator(backend, "PASTICUAN").get_or_refresh(
        KEY,
        read_current=backend.read,
        fetch=lambda: (_ for _ in ()).throw(AssertionError("duplicate provider call")),
        persist=backend.persist,
        validate=_validate,
    )
    release.set()
    thread.join(timeout=3)
    assert calls == 1
    assert results["second"].reason == MissingReason.REFRESH_LOCKED.value
    assert results["second"].request_avoided
    assert results["first"].state is EvidenceState.VALID


def test_expired_refresh_lease_is_recovered() -> None:
    backend = MemoryBackend()
    backend.leases[backend._key(KEY)] = {
        "state": "HELD",
        "holder": "dead-worker",
        "expires_at": datetime.now(timezone.utc) - timedelta(seconds=1),
    }
    result = _coordinator(backend, "EMIR").get_or_refresh(
        KEY,
        read_current=backend.read,
        fetch=_valid_rows,
        persist=backend.persist,
        validate=_validate,
    )
    assert result.state is EvidenceState.VALID
    assert backend.leases[backend._key(KEY)]["state"] == "COMPLETED"


def test_empty_payload_preserves_truthful_error() -> None:
    backend = MemoryBackend()
    result = _coordinator(backend, "EMIR").get_or_refresh(
        KEY, read_current=backend.read, fetch=list, persist=backend.persist, validate=_validate
    )
    assert result.state is EvidenceState.ERROR
    assert result.reason == MissingReason.EMPTY_RESPONSE.value
    assert result.provider_called


def test_partial_persist_is_classified_persist_failure() -> None:
    backend = MemoryBackend()
    result = _coordinator(backend, "EMIR").get_or_refresh(
        KEY, read_current=backend.read, fetch=_valid_rows, persist=lambda rows: 0, validate=_validate
    )
    assert result.state is EvidenceState.ERROR
    assert result.reason == MissingReason.PERSIST_FAILURE.value


def test_missing_readback_is_classified_readback_failure() -> None:
    backend = MemoryBackend()
    result = _coordinator(backend, "EMIR").get_or_refresh(
        KEY, read_current=list, fetch=_valid_rows, persist=lambda rows: len(rows), validate=_validate
    )
    assert result.reason == MissingReason.READBACK_FAILURE.value


@pytest.mark.parametrize(
    "exc,reason",
    [
        (requests.Timeout(), MissingReason.TIMEOUT.value),
        (requests.ConnectionError(), MissingReason.CONNECTION_ERROR.value),
        (RuntimeError("HTTP_401"), MissingReason.HTTP_401.value),
        (RuntimeError("HTTP_403"), MissingReason.HTTP_403.value),
        (RuntimeError("HTTP_404"), MissingReason.HTTP_404.value),
        (RuntimeError("HTTP_429"), MissingReason.HTTP_429.value),
    ],
)
def test_provider_http_and_transport_reasons_are_preserved(exc: Exception, reason: str) -> None:
    assert SupabaseEvidenceBackend._error_reason(exc) == reason


def test_configuration_never_falls_back_to_idx_flow_or_exposes_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHARED_EVIDENCE_SUPABASE_URL", f"https://{IDX_FLOW_SUPABASE_PROJECT_REF}.supabase.co")
    monkeypatch.setenv("SHARED_EVIDENCE_SUPABASE_SECRET_KEY", "sb_secret_must_not_leak")
    config = HubConfig.from_environment(client_id="EMIR")
    assert not config.ready
    assert config.key == ""
    assert "sb_secret_must_not_leak" not in repr(config)
    assert "sb_secret_must_not_leak" not in str(config.status())


def test_migration_is_scanner_neutral_secured_and_additive() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    candidates = sorted((root / "database").glob("migration_v*_shared_evidence_hub.sql"))
    assert len(candidates) == 1
    sql = candidates[0].read_text(encoding="utf-8").lower()
    for table in (
        "evidence_provider_state", "evidence_refresh_leases", "evidence_market_daily",
        "evidence_foreign_flow", "evidence_participant_flow", "evidence_ownership_snapshots",
        "evidence_financial_facts", "evidence_announcements", "evidence_capital_actions",
        "evidence_risk_events", "evidence_trading_calendar",
    ):
        assert f"create table if not exists public.{table}" in sql
    assert "security invoker" in sql
    assert "set search_path = ''" in sql
    assert "enable row level security" in sql
    assert "to service_role" in sql
    assert "drop table" not in sql
    assert "emir_score" not in sql and "pasticuan_score" not in sql
    assert IDX_FLOW_SUPABASE_PROJECT_REF not in sql


def test_valid_empty_payload_can_be_cached_without_duplicate_provider_call() -> None:
    backend = MemoryBackend()
    empty_marker = {"ready": False}
    calls = 0

    def fetch() -> list[Mapping[str, Any]]:
        nonlocal calls
        calls += 1
        empty_marker["ready"] = True
        return []

    first = _coordinator(backend, "PASTICUAN").get_or_refresh(
        KEY,
        read_current=backend.read,
        fetch=fetch,
        persist=backend.persist,
        validate=_validate,
        allow_empty_valid=True,
        read_empty_current=lambda: empty_marker["ready"],
    )
    assert first.state is EvidenceState.VALID
    assert first.reason == "REFRESHED_EMPTY"
    assert first.provider_called and first.rows == ()
    assert backend.leases[backend._key(KEY)]["state"] == "COMPLETED"
    assert backend.provider_states[-1]["response_state"] == "VALID_EMPTY"
    assert backend.provider_states[-1]["error_classification"] is None

    second = _coordinator(backend, "EMIR").get_or_refresh(
        KEY,
        read_current=backend.read,
        fetch=lambda: (_ for _ in ()).throw(AssertionError("duplicate provider call")),
        persist=backend.persist,
        validate=_validate,
        allow_empty_valid=True,
        read_empty_current=lambda: empty_marker["ready"],
    )
    assert second.state is EvidenceState.VALID
    assert second.reason == "CACHE_HIT_EMPTY"
    assert second.cache_hit and second.request_avoided and not second.provider_called
    assert calls == 1


def test_empty_payload_still_errors_without_explicit_empty_contract() -> None:
    backend = MemoryBackend()
    result = _coordinator(backend, "EMIR").get_or_refresh(
        KEY,
        read_current=backend.read,
        fetch=list,
        persist=backend.persist,
        validate=_validate,
    )
    assert result.state is EvidenceState.ERROR
    assert result.reason == MissingReason.EMPTY_RESPONSE.value
