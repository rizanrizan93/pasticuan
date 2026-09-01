from __future__ import annotations

"""Release-gated small-cohort, quota, and production coverage validation."""

from dataclasses import dataclass, field
from datetime import date
import hashlib
import json
from typing import Any, Callable, Iterable, Mapping

from shared_evidence_hub import EvidenceKey, SharedEvidenceCoordinator


VALIDATION_VERSION = "1.0.0-phase5.6-task33-35"
PRODUCTION_UNIVERSE_SIZE = 400
REQUIRED_PRODUCTION_GATES = (
    "shared_schema_ready",
    "small_db_roundtrip_passed",
    "cross_scanner_reuse_passed",
    "producer_health_passed",
    "luna_verified",
    "release_approved",
)


def _ticker(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text[:-3] if text.endswith(".JK") else text


def _hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ProviderQuotaSnapshot:
    calls_attempted: int
    successful_calls: int
    failed_calls: int
    cache_hits: int
    cache_misses: int
    calls_avoided: int
    latest_source_date: str | None
    quota_remaining: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls_attempted": self.calls_attempted,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "calls_avoided": self.calls_avoided,
            "latest_source_date": self.latest_source_date,
            "quota_remaining": self.quota_remaining,
        }


@dataclass
class ProviderQuotaAccounting:
    calls_attempted: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    calls_avoided: int = 0
    latest_source_date: str | None = None
    quota_remaining: int | None = None

    @classmethod
    def from_coordinator_metrics(
        cls,
        metrics: Mapping[str, Any],
        *,
        latest_source_date: date | str | None = None,
        explicitly_reported_quota_remaining: int | None = None,
    ) -> "ProviderQuotaAccounting":
        values = {
            "calls_attempted": metrics.get("calls_attempted", 0),
            "successful_calls": metrics.get("success", 0),
            "failed_calls": metrics.get("failure", 0),
            "cache_hits": metrics.get("cache_hits", 0),
            "cache_misses": metrics.get("cache_misses", 0),
            "calls_avoided": metrics.get("calls_avoided", 0),
        }
        parsed: dict[str, int] = {}
        for name, value in values.items():
            if isinstance(value, bool):
                raise ValueError(f"INVALID_QUOTA_COUNTER:{name}")
            try:
                number = int(value)
            except (TypeError, ValueError):
                raise ValueError(f"INVALID_QUOTA_COUNTER:{name}") from None
            if number < 0 or number != value:
                raise ValueError(f"INVALID_QUOTA_COUNTER:{name}")
            parsed[name] = number
        if parsed["successful_calls"] + parsed["failed_calls"] > parsed["calls_attempted"]:
            raise ValueError("INVALID_QUOTA_COUNTER:outcomes_exceed_attempts")
        if explicitly_reported_quota_remaining is not None:
            if isinstance(explicitly_reported_quota_remaining, bool) or explicitly_reported_quota_remaining < 0:
                raise ValueError("INVALID_QUOTA_REMAINING")
        return cls(
            **parsed,
            latest_source_date=str(latest_source_date) if latest_source_date is not None else None,
            quota_remaining=explicitly_reported_quota_remaining,
        )

    def merge(self, other: "ProviderQuotaAccounting") -> "ProviderQuotaAccounting":
        dates = [value for value in (self.latest_source_date, other.latest_source_date) if value]
        quota = other.quota_remaining if other.quota_remaining is not None else self.quota_remaining
        return ProviderQuotaAccounting(
            calls_attempted=self.calls_attempted + other.calls_attempted,
            successful_calls=self.successful_calls + other.successful_calls,
            failed_calls=self.failed_calls + other.failed_calls,
            cache_hits=self.cache_hits + other.cache_hits,
            cache_misses=self.cache_misses + other.cache_misses,
            calls_avoided=self.calls_avoided + other.calls_avoided,
            latest_source_date=max(dates) if dates else None,
            quota_remaining=quota,
        )

    def snapshot(self) -> ProviderQuotaSnapshot:
        return ProviderQuotaSnapshot(
            self.calls_attempted, self.successful_calls, self.failed_calls,
            self.cache_hits, self.cache_misses, self.calls_avoided,
            self.latest_source_date, self.quota_remaining,
        )


@dataclass(frozen=True)
class CohortRoundTripResult:
    first_client: str
    second_client: str
    ticker_count: int
    provider_calls: int
    first_reason: str
    second_reason: str
    second_request_avoided: bool
    same_facts: bool
    first_coverage_percentage: float
    second_coverage_percentage: float
    facts_hash: str
    quota: ProviderQuotaSnapshot


def run_small_cohort_roundtrip(
    tickers: Iterable[Any],
    *,
    target_date: date,
    backend: Any,
    fetch: Callable[[], list[Mapping[str, Any]]],
    first_client: str,
    second_client: str,
    provider: str,
    family: str,
    table: str,
    date_field: str,
    conflict: tuple[str, ...],
    consumer: Callable[[list[dict[str, Any]], str], Mapping[str, Any]] | None = None,
) -> CohortRoundTripResult:
    names = list(dict.fromkeys(_ticker(value) for value in tickers if _ticker(value)))
    if not 5 <= len(names) <= 10:
        raise ValueError(f"SMALL_COHORT_SIZE_MUST_BE_5_TO_10:{len(names)}")
    first = str(first_client).strip().upper()
    second = str(second_client).strip().upper()
    if {first, second} != {"EMIR", "PASTICUAN"}:
        raise ValueError("SMALL_COHORT_REQUIRES_BOTH_SCANNERS")
    provider_name = str(provider).strip().upper()
    family_name = str(family).strip().upper()
    scope = f"IDX_COHORT_{len(names)}_{_hash(sorted(names))[:12].upper()}"
    provider_calls = 0

    def read_current() -> list[dict[str, Any]]:
        return backend.read_rows(
            table,
            {"provider": provider_name, date_field: target_date.isoformat(), "validation_state": "VALID"},
            limit=100,
        )

    def fetch_once() -> list[dict[str, Any]]:
        nonlocal provider_calls
        provider_calls += 1
        return [dict(row) for row in fetch()]

    def validate(rows: list[Mapping[str, Any]]) -> tuple[bool, str]:
        actual = [_ticker(row.get("ticker")) for row in rows]
        valid = (
            len(actual) == len(names)
            and len(set(actual)) == len(names)
            and set(actual) == set(names)
            and all(row.get("provider") == provider_name for row in rows)
            and all(str(row.get(date_field)) == target_date.isoformat() for row in rows)
            and all(row.get("validation_state") == "VALID" for row in rows)
        )
        return valid, "VALID" if valid else "CONTEXT_REJECTED"

    def persist(rows: list[Mapping[str, Any]]) -> int:
        return len(backend.upsert_rows(table, rows, conflict=conflict))

    key = EvidenceKey(provider_name, family_name, scope, target_date)
    first_coordinator = SharedEvidenceCoordinator(
        backend, client_id=first, worker_id=f"{first}-small-cohort"
    )
    second_coordinator = SharedEvidenceCoordinator(
        backend, client_id=second, worker_id=f"{second}-small-cohort"
    )
    first_result = first_coordinator.get_or_refresh(
        key, read_current=read_current, fetch=fetch_once, persist=persist,
        validate=validate, minimum_rows=len(names), lease_seconds=300,
    )
    second_result = second_coordinator.get_or_refresh(
        key,
        read_current=read_current,
        fetch=lambda: (_ for _ in ()).throw(AssertionError("DUPLICATE_PROVIDER_CALL")),
        persist=persist,
        validate=validate,
        minimum_rows=len(names),
        lease_seconds=300,
    )
    first_rows = [dict(row) for row in first_result.rows]
    second_rows = [dict(row) for row in second_result.rows]
    consume = consumer or (lambda rows, client: {
        "client": client,
        "observed": len({_ticker(row.get("ticker")) for row in rows}),
    })
    first_consumed = dict(consume(first_rows, first))
    second_consumed = dict(consume(second_rows, second))
    first_observed = len({_ticker(row.get("ticker")) for row in first_rows})
    second_observed = len({_ticker(row.get("ticker")) for row in second_rows})
    quota = ProviderQuotaAccounting.from_coordinator_metrics(
        first_coordinator.metrics(), latest_source_date=target_date
    ).merge(ProviderQuotaAccounting.from_coordinator_metrics(second_coordinator.metrics()))
    same_facts = _hash(first_rows) == _hash(second_rows)
    if provider_calls != 1 or not second_result.request_avoided or not same_facts:
        raise RuntimeError("SMALL_COHORT_REUSE_CONTRACT_FAILED")
    if first_consumed.get("client") != first or second_consumed.get("client") != second:
        raise RuntimeError("SMALL_COHORT_CONSUMER_IDENTITY_FAILED")
    return CohortRoundTripResult(
        first_client=first,
        second_client=second,
        ticker_count=len(names),
        provider_calls=provider_calls,
        first_reason=first_result.reason,
        second_reason=second_result.reason,
        second_request_avoided=second_result.request_avoided,
        same_facts=same_facts,
        first_coverage_percentage=round(100.0 * first_observed / len(names), 2),
        second_coverage_percentage=round(100.0 * second_observed / len(names), 2),
        facts_hash=_hash(first_rows),
        quota=quota.snapshot(),
    )


@dataclass(frozen=True)
class ProductionGateDecision:
    allowed: bool
    unmet: tuple[str, ...]
    state: str


def evaluate_production_validation_gates(gates: Mapping[str, Any]) -> ProductionGateDecision:
    unmet = tuple(name for name in REQUIRED_PRODUCTION_GATES if gates.get(name) is not True)
    return ProductionGateDecision(
        allowed=not unmet,
        unmet=unmet,
        state="AUTHORIZED" if not unmet else "BLOCKED",
    )


@dataclass
class ProductionCoverageRegistry:
    gates: Mapping[str, Any]
    measurements: dict[str, str] = field(default_factory=dict)

    def register(self, *, scanner: str, universe_size: int, measurement_id: str) -> None:
        decision = evaluate_production_validation_gates(self.gates)
        if not decision.allowed:
            raise RuntimeError(f"PRODUCTION_VALIDATION_BLOCKED:{','.join(decision.unmet)}")
        name = str(scanner).strip().upper()
        if name not in {"EMIR", "PASTICUAN"}:
            raise ValueError("UNKNOWN_SCANNER")
        if int(universe_size) != PRODUCTION_UNIVERSE_SIZE:
            raise ValueError(f"PRODUCTION_UNIVERSE_MUST_BE_EXACTLY_400:{universe_size}")
        identity = str(measurement_id).strip()
        if not identity:
            raise ValueError("MEASUREMENT_ID_MISSING")
        if name in self.measurements:
            raise RuntimeError(f"REPEATED_PRODUCTION_MEASUREMENT:{name}")
        self.measurements[name] = identity

    def complete(self) -> bool:
        return set(self.measurements) == {"EMIR", "PASTICUAN"}


__all__ = [
    "CohortRoundTripResult", "PRODUCTION_UNIVERSE_SIZE", "ProductionCoverageRegistry",
    "ProductionGateDecision", "ProviderQuotaAccounting", "ProviderQuotaSnapshot",
    "REQUIRED_PRODUCTION_GATES", "VALIDATION_VERSION", "evaluate_production_validation_gates",
    "run_small_cohort_roundtrip",
]
