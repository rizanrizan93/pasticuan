from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import threading
from typing import Any, Mapping

import pytest

from shared_evidence_hub import EvidenceKey, SharedEvidenceCoordinator
from shared_evidence_validation import (
    PRODUCTION_UNIVERSE_SIZE,
    REQUIRED_PRODUCTION_GATES,
    ProductionCoverageRegistry,
    ProviderQuotaAccounting,
    evaluate_production_validation_gates,
    run_small_cohort_roundtrip,
)


DAY = date(2026, 8, 31)
TICKERS = ["BBCA", "BBRI", "BMRI", "TLKM", "ASII"]


class MemoryBackend:
    def __init__(self):
        self.tables = {"evidence_market_daily": []}
        self.leases: dict[tuple[str, str, str, date], dict[str, Any]] = {}
        self.states = []
        self.lock = threading.Lock()

    @staticmethod
    def identity(key: EvidenceKey):
        key = key.normalized()
        return key.provider, key.family, key.scope, key.target_date

    def read_rows(self, table: str, filters: Mapping[str, Any], *, limit: int):
        with self.lock:
            return [
                dict(row) for row in self.tables[table]
                if all(str(row.get(key)) == str(value) for key, value in filters.items())
            ][:limit]

    def upsert_rows(self, table: str, rows, *, conflict):
        with self.lock:
            current = {tuple(row.get(key) for key in conflict): dict(row) for row in self.tables[table]}
            for row in rows:
                current[tuple(row.get(key) for key in conflict)] = dict(row)
            self.tables[table] = list(current.values())
        return [dict(row) for row in rows]

    def acquire_lease(self, key, holder, lease_seconds):
        with self.lock:
            identity = self.identity(key)
            now = datetime.now(timezone.utc)
            current = self.leases.get(identity)
            if current and current["state"] == "HELD" and current["expires"] > now and current["holder"] != holder:
                return {"acquired": False}
            self.leases[identity] = {
                "state": "HELD", "holder": holder,
                "expires": now + timedelta(seconds=lease_seconds),
            }
            return {"acquired": True}

    def complete_lease(self, key, holder, state):
        with self.lock:
            current = self.leases[self.identity(key)]
            if current["holder"] != holder or current["state"] != "HELD":
                return False
            current["state"] = "COMPLETED"
            return True

    def fail_lease(self, key, holder, reason):
        with self.lock:
            self.leases[self.identity(key)] = {"state": "FAILED", "holder": holder, "reason": reason}
            return True

    def record_provider_state(self, row):
        self.states.append(dict(row))


def _rows(tickers=TICKERS):
    return [
        {
            "provider": "ZAPI", "trade_date": DAY.isoformat(), "ticker": ticker,
            "close": 100 + index, "validation_state": "VALID",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        for index, ticker in enumerate(tickers)
    ]


@pytest.mark.parametrize("first,second", [("EMIR", "PASTICUAN"), ("PASTICUAN", "EMIR")])
def test_bounded_cohort_roundtrip_reuses_one_fetch(first: str, second: str) -> None:
    result = run_small_cohort_roundtrip(
        TICKERS, target_date=DAY, backend=MemoryBackend(), fetch=_rows,
        first_client=first, second_client=second, provider="ZAPI", family="STOCK_SUMMARY",
        table="evidence_market_daily", date_field="trade_date",
        conflict=("provider", "trade_date", "ticker"),
    )
    assert result.provider_calls == 1 and result.second_request_avoided and result.same_facts
    assert result.first_reason == "REFRESHED" and result.second_reason == "CACHE_HIT"
    assert result.first_coverage_percentage == result.second_coverage_percentage == 100.0
    assert result.quota.as_dict() == {
        "calls_attempted": 1, "successful_calls": 1, "failed_calls": 0,
        "cache_hits": 1, "cache_misses": 1, "calls_avoided": 1,
        "latest_source_date": DAY.isoformat(), "quota_remaining": None,
    }


@pytest.mark.parametrize("count", [0, 4, 11])
def test_cohort_size_is_strictly_five_to_ten(count: int) -> None:
    with pytest.raises(ValueError, match="SMALL_COHORT_SIZE_MUST_BE_5_TO_10"):
        run_small_cohort_roundtrip(
            [f"T{index:03d}" for index in range(count)], target_date=DAY,
            backend=MemoryBackend(), fetch=list, first_client="EMIR", second_client="PASTICUAN",
            provider="ZAPI", family="FIXTURE", table="evidence_market_daily",
            date_field="trade_date", conflict=("provider", "trade_date", "ticker"),
        )


def test_cohort_rejects_missing_or_extra_rows_without_persistence() -> None:
    backend = MemoryBackend()
    with pytest.raises(RuntimeError, match="SMALL_COHORT_REUSE_CONTRACT_FAILED"):
        run_small_cohort_roundtrip(
            TICKERS, target_date=DAY, backend=backend, fetch=lambda: _rows(TICKERS[:-1]),
            first_client="EMIR", second_client="PASTICUAN", provider="ZAPI", family="STOCK_SUMMARY",
            table="evidence_market_daily", date_field="trade_date",
            conflict=("provider", "trade_date", "ticker"),
        )
    assert backend.tables["evidence_market_daily"] == []


def test_concurrent_cohort_refresh_has_one_lease_holder_and_one_provider_call() -> None:
    backend = MemoryBackend()
    key = EvidenceKey("ZAPI", "COHORT", "IDX_5", DAY)
    entered, release = threading.Event(), threading.Event()
    calls = 0
    results = {}

    def read(): return backend.read_rows("evidence_market_daily", {"provider": "ZAPI", "trade_date": DAY.isoformat(), "validation_state": "VALID"}, limit=100)
    def fetch():
        nonlocal calls
        calls += 1; entered.set(); assert release.wait(timeout=3); return _rows()
    def persist(rows): return len(backend.upsert_rows("evidence_market_daily", rows, conflict=("provider", "trade_date", "ticker")))
    def validate(rows): return (len(rows) == 5, "VALID")
    def first(): results["first"] = SharedEvidenceCoordinator(backend, client_id="EMIR", worker_id="emir").get_or_refresh(key, read_current=read, fetch=fetch, persist=persist, validate=validate, minimum_rows=5)

    thread = threading.Thread(target=first); thread.start(); assert entered.wait(timeout=3)
    results["second"] = SharedEvidenceCoordinator(backend, client_id="PASTICUAN", worker_id="pasticuan").get_or_refresh(
        key, read_current=read, fetch=lambda: (_ for _ in ()).throw(AssertionError("duplicate")),
        persist=persist, validate=validate, minimum_rows=5,
    )
    release.set(); thread.join(timeout=3)
    assert calls == 1 and results["second"].reason == "REFRESH_LOCKED"
    assert results["second"].request_avoided and results["first"].reason == "REFRESHED"


def test_quota_accounting_uses_exact_metrics_and_no_estimated_remaining() -> None:
    accounting = ProviderQuotaAccounting.from_coordinator_metrics({
        "calls_attempted": 3, "success": 2, "failure": 1,
        "cache_hits": 7, "cache_misses": 3, "calls_avoided": 7,
    }, latest_source_date="2026-08-31")
    snapshot = accounting.snapshot().as_dict()
    assert snapshot["calls_attempted"] == 3 and snapshot["successful_calls"] == 2
    assert snapshot["failed_calls"] == 1 and snapshot["quota_remaining"] is None


def test_explicit_provider_quota_remaining_is_preserved() -> None:
    accounting = ProviderQuotaAccounting.from_coordinator_metrics(
        {}, explicitly_reported_quota_remaining=123
    )
    assert accounting.snapshot().quota_remaining == 123


@pytest.mark.parametrize(
    "metrics,reason",
    [
        ({"calls_attempted": -1}, "INVALID_QUOTA_COUNTER"),
        ({"calls_attempted": 1, "success": 2}, "outcomes_exceed_attempts"),
        ({"cache_hits": 1.5}, "INVALID_QUOTA_COUNTER"),
        ({"calls_attempted": True}, "INVALID_QUOTA_COUNTER"),
    ],
)
def test_invalid_or_invented_quota_counts_fail_closed(metrics, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        ProviderQuotaAccounting.from_coordinator_metrics(metrics)


def test_quota_accounting_merges_bootstrap_runs_exactly() -> None:
    first = ProviderQuotaAccounting.from_coordinator_metrics(
        {"calls_attempted": 2, "success": 1, "failure": 1, "cache_misses": 2},
        latest_source_date="2026-08-30", explicitly_reported_quota_remaining=200,
    )
    second = ProviderQuotaAccounting.from_coordinator_metrics(
        {"calls_attempted": 1, "success": 1, "cache_hits": 5, "cache_misses": 1, "calls_avoided": 5},
        latest_source_date="2026-08-31", explicitly_reported_quota_remaining=199,
    )
    merged = first.merge(second).snapshot()
    assert merged.calls_attempted == 3 and merged.successful_calls == 2 and merged.failed_calls == 1
    assert merged.cache_hits == 5 and merged.calls_avoided == 5
    assert merged.latest_source_date == "2026-08-31" and merged.quota_remaining == 199


def test_production_gate_lists_every_unmet_requirement() -> None:
    decision = evaluate_production_validation_gates({})
    assert not decision.allowed and decision.state == "BLOCKED"
    assert decision.unmet == REQUIRED_PRODUCTION_GATES


def test_production_gate_requires_literal_true_not_truthy_values() -> None:
    gates = {name: True for name in REQUIRED_PRODUCTION_GATES}
    gates["luna_verified"] = "true"
    decision = evaluate_production_validation_gates(gates)
    assert not decision.allowed and decision.unmet == ("luna_verified",)


def test_production_registry_blocks_before_luna_and_release() -> None:
    gates = {name: True for name in REQUIRED_PRODUCTION_GATES}
    gates["luna_verified"] = False
    gates["release_approved"] = False
    registry = ProductionCoverageRegistry(gates)
    with pytest.raises(RuntimeError, match="luna_verified,release_approved"):
        registry.register(scanner="EMIR", universe_size=400, measurement_id="run-1")
    assert registry.measurements == {}


def test_one_exact_400_measurement_per_scanner_completes_authorized_plan() -> None:
    registry = ProductionCoverageRegistry({name: True for name in REQUIRED_PRODUCTION_GATES})
    registry.register(scanner="EMIR", universe_size=PRODUCTION_UNIVERSE_SIZE, measurement_id="emir-run")
    registry.register(scanner="PASTICUAN", universe_size=PRODUCTION_UNIVERSE_SIZE, measurement_id="pasticuan-run")
    assert registry.complete()
    with pytest.raises(RuntimeError, match="REPEATED_PRODUCTION_MEASUREMENT"):
        registry.register(scanner="EMIR", universe_size=400, measurement_id="emir-run-2")


def test_production_registry_rejects_non_400_measurement() -> None:
    registry = ProductionCoverageRegistry({name: True for name in REQUIRED_PRODUCTION_GATES})
    with pytest.raises(ValueError, match="PRODUCTION_UNIVERSE_MUST_BE_EXACTLY_400"):
        registry.register(scanner="EMIR", universe_size=399, measurement_id="bad")


def test_validation_module_has_no_network_or_scanner_decision_imports() -> None:
    source = __import__("pathlib").Path(__file__).resolve().parents[1].joinpath(
        "shared_evidence_validation.py"
    ).read_text().lower()
    assert "import requests" not in source and "zapi_key" not in source
    assert "entry_price" not in source and "take_profit" not in source and "stop_loss" not in source
