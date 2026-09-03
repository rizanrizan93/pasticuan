from __future__ import annotations

"""Scanner-neutral persistence and refresh coordination for factual IDX evidence.

This module deliberately has no knowledge of scanner scores, rankings, gates, or
recommendations.  Both scanners use the same evidence identity and independently
derive their own conclusions after reading validated facts.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
import os
from threading import Lock
from typing import Any, Callable, Iterable, Mapping, Protocol
from urllib.parse import urlencode, urlparse
import uuid

import requests


SHARED_EVIDENCE_HUB_VERSION = "1.0.0-phase5.6"
IDX_FLOW_SUPABASE_PROJECT_REF = "djqvhbeonmicztxfisav"


class EvidenceState(str, Enum):
    VALID = "VALID"
    STALE = "STALE"
    MISSING = "MISSING"
    INSUFFICIENT = "INSUFFICIENT"
    ERROR = "ERROR"


class MissingReason(str, Enum):
    NO_REPORT = "NO_REPORT"
    NO_FILE = "NO_FILE"
    NO_MATCH = "NO_MATCH"
    HTTP_401 = "HTTP_401"
    HTTP_403 = "HTTP_403"
    HTTP_404 = "HTTP_404"
    HTTP_429 = "HTTP_429"
    TIMEOUT = "TIMEOUT"
    CONNECTION_ERROR = "CONNECTION_ERROR"
    RATE_LIMIT = "RATE_LIMIT"
    QUOTA_EXHAUSTED = "QUOTA_EXHAUSTED"
    PARSE_FAILURE = "PARSE_FAILURE"
    ISSUER_MISMATCH = "ISSUER_MISMATCH"
    ISSUER_IDENTITY_MISSING = "ISSUER_IDENTITY_MISSING"
    CONTEXT_REJECTED = "CONTEXT_REJECTED"
    WRONG_PERIOD = "WRONG_PERIOD"
    STALE = "STALE"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    PROVIDER_NO_DATA = "PROVIDER_NO_DATA"
    INVALID_CONTENT_TYPE = "INVALID_CONTENT_TYPE"
    EMPTY_RESPONSE = "EMPTY_RESPONSE"
    PERSIST_FAILURE = "PERSIST_FAILURE"
    READBACK_FAILURE = "READBACK_FAILURE"
    REFRESH_LOCKED = "REFRESH_LOCKED"
    REFRESH_LEASE_EXPIRED = "REFRESH_LEASE_EXPIRED"
    ENVIRONMENT_BLOCKED = "ENVIRONMENT_BLOCKED"


class SecretStatus(str, Enum):
    CONFIGURED = "CONFIGURED"
    MISSING = "MISSING"
    EMPTY = "EMPTY"
    INVALID_REJECTED = "INVALID/REJECTED"


_SECRET_MISSING = object()


def secret_status(value: Any = _SECRET_MISSING, *, rejected: bool = False) -> str:
    if rejected:
        return SecretStatus.INVALID_REJECTED.value
    if value is _SECRET_MISSING or value is None:
        return SecretStatus.MISSING.value
    if not str(value).strip():
        return SecretStatus.EMPTY.value
    return SecretStatus.CONFIGURED.value


def normalize_failure_reason(error: Exception | str) -> str:
    if isinstance(error, requests.Timeout):
        return MissingReason.TIMEOUT.value
    if isinstance(error, requests.ConnectionError):
        return MissingReason.CONNECTION_ERROR.value
    text = _clean(error).upper()
    allowed = {item.value for item in MissingReason}
    if text in allowed:
        return text
    if text.startswith("HTTP_"):
        try:
            status = int(text.split("_", 1)[1])
        except ValueError:
            status = 0
        if status in {401, 403, 404, 429}:
            return f"HTTP_{status}"
        if status == 408:
            return MissingReason.TIMEOUT.value
        if 400 <= status < 500:
            return MissingReason.CONTEXT_REJECTED.value
        if status >= 500:
            return MissingReason.CONNECTION_ERROR.value
    if isinstance(error, (ValueError, TypeError, KeyError)):
        return MissingReason.PARSE_FAILURE.value
    return MissingReason.PROVIDER_NO_DATA.value


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _project_ref(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host.split(".", 1)[0] if host.endswith(".supabase.co") else ""


def _secret(name: str) -> str:
    value = _clean(os.getenv(name, ""))
    if value:
        return value
    try:
        import streamlit as st

        return _clean(st.secrets.get(name, ""))
    except Exception:
        return ""


@dataclass(frozen=True)
class HubConfig:
    url: str = ""
    key: str = ""
    key_type: str = "NONE"
    client_id: str = "UNKNOWN"
    timeout_seconds: float = 20.0

    @classmethod
    def from_environment(cls, *, client_id: str) -> "HubConfig":
        url = _secret("SHARED_EVIDENCE_SUPABASE_URL").rstrip("/")
        key_candidates = (
            ("SECRET", _secret("SHARED_EVIDENCE_SUPABASE_SECRET_KEY")),
            ("SERVICE_ROLE", _secret("SHARED_EVIDENCE_SUPABASE_SERVICE_ROLE_KEY")),
        )
        key_type, key = next(((kind, value) for kind, value in key_candidates if value), ("NONE", ""))
        if _project_ref(url) == IDX_FLOW_SUPABASE_PROJECT_REF:
            return cls(client_id=_clean(client_id).upper() or "UNKNOWN")
        timeout = _clean(os.getenv("SHARED_EVIDENCE_TIMEOUT_SECONDS", "20"))
        try:
            timeout_seconds = max(2.0, min(60.0, float(timeout)))
        except ValueError:
            timeout_seconds = 20.0
        return cls(
            url=url,
            key=key,
            key_type=key_type,
            client_id=_clean(client_id).upper() or "UNKNOWN",
            timeout_seconds=timeout_seconds,
        )

    @property
    def ready(self) -> bool:
        return bool(self.url and self.key and self.key_type in {"SECRET", "SERVICE_ROLE"})

    def status(self) -> dict[str, Any]:
        return {
            "hub_version": SHARED_EVIDENCE_HUB_VERSION,
            "state": "CONFIGURED" if self.ready else "MISSING",
            "url_state": secret_status(self.url if self.url else None),
            "key_state": secret_status(self.key if self.key else None),
            "key_type": self.key_type,
            "client_id": self.client_id,
            "idx_flow_blocked": _project_ref(self.url) == IDX_FLOW_SUPABASE_PROJECT_REF,
        }

    def __repr__(self) -> str:
        return (
            "HubConfig(url='<configured>', key='<redacted>', "
            f"key_type={self.key_type!r}, client_id={self.client_id!r}, "
            f"ready={self.ready!r})"
        )


@dataclass(frozen=True)
class EvidenceKey:
    provider: str
    family: str
    scope: str
    target_date: date

    def normalized(self) -> "EvidenceKey":
        return EvidenceKey(
            provider=_clean(self.provider).upper(),
            family=_clean(self.family).upper(),
            scope=_clean(self.scope).upper(),
            target_date=self.target_date,
        )

    def as_dict(self) -> dict[str, str]:
        key = self.normalized()
        return {
            "provider": key.provider,
            "evidence_family": key.family,
            "scope": key.scope,
            "target_date": key.target_date.isoformat(),
        }


@dataclass(frozen=True)
class RefreshResult:
    state: EvidenceState
    reason: str
    rows: tuple[Mapping[str, Any], ...] = ()
    provider_called: bool = False
    cache_hit: bool = False
    request_avoided: bool = False
    lease_state: str = "NOT_REQUIRED"


class EvidenceBackend(Protocol):
    def acquire_lease(self, key: EvidenceKey, holder: str, lease_seconds: int) -> Mapping[str, Any]: ...
    def complete_lease(self, key: EvidenceKey, holder: str, state: str) -> bool: ...
    def fail_lease(self, key: EvidenceKey, holder: str, reason: str) -> bool: ...
    def record_provider_state(self, row: Mapping[str, Any]) -> None: ...


class SupabaseEvidenceBackend:
    """Minimal PostgREST client. It accepts backend keys only and never logs them."""

    def __init__(self, config: HubConfig, session: requests.Session | None = None):
        self.config = config
        self.session = session or requests.Session()

    def _headers(self, *, prefer: str = "") -> dict[str, str]:
        if not self.config.ready:
            raise RuntimeError(MissingReason.ENVIRONMENT_BLOCKED.value)
        headers = {
            "apikey": self.config.key,
            "Authorization": f"Bearer {self.config.key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": f"Shared-IDX-Evidence-Hub/{SHARED_EVIDENCE_HUB_VERSION}",
        }
        if prefer:
            headers["Prefer"] = prefer
        return headers

    @staticmethod
    def _error_reason(exc: Exception) -> str:
        return normalize_failure_reason(exc)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        payload: Any = None,
        prefer: str = "",
    ) -> Any:
        response = self.session.request(
            method,
            f"{self.config.url}/rest/v1/{path.lstrip('/')}",
            params=dict(params or {}),
            json=payload,
            headers=self._headers(prefer=prefer),
            timeout=self.config.timeout_seconds,
        )
        if response.status_code in {401, 403, 404, 429}:
            raise RuntimeError(f"HTTP_{response.status_code}")
        response.raise_for_status()
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError("MALFORMED_RESPONSE") from exc

    def read_rows(
        self,
        table: str,
        filters: Mapping[str, Any],
        *,
        select: str = "*",
        limit: int = 10000,
    ) -> list[dict[str, Any]]:
        requested = max(1, min(int(limit), 50000))
        # Supabase/PostgREST projects commonly enforce a server-side max-row
        # response cap (often 1,000) even when a larger `limit` is requested.
        # Page explicitly so cache consumers and readback validation do not
        # silently see only the first server-capped page.
        page_size = min(1000, requested)
        rows: list[dict[str, Any]] = []
        offset = 0
        while len(rows) < requested:
            params = {
                "select": select,
                "limit": min(page_size, requested - len(rows)),
                "offset": offset,
            }
            params.update({name: f"eq.{value}" for name, value in filters.items()})
            payload = self._request("GET", table, params=params)
            page = [dict(row) for row in payload] if isinstance(payload, list) else []
            rows.extend(page)
            if len(page) < int(params["limit"]):
                break
            offset += len(page)
        return rows[:requested]

    def upsert_rows(
        self,
        table: str,
        rows: Iterable[Mapping[str, Any]],
        *,
        conflict: Iterable[str],
    ) -> list[dict[str, Any]]:
        records = [dict(row) for row in rows]
        if not records:
            return []
        params = {"on_conflict": ",".join(conflict)}
        payload = self._request(
            "POST",
            table,
            params=params,
            payload=records,
            prefer="resolution=merge-duplicates,return=representation",
        )
        return [dict(row) for row in payload] if isinstance(payload, list) else []

    def _rpc(self, name: str, payload: Mapping[str, Any]) -> Any:
        return self._request("POST", f"rpc/{name}", payload=dict(payload))

    def acquire_lease(self, key: EvidenceKey, holder: str, lease_seconds: int) -> Mapping[str, Any]:
        payload = self._rpc(
            "evidence_acquire_refresh_lease",
            {
                "p_provider": key.normalized().provider,
                "p_family": key.normalized().family,
                "p_scope": key.normalized().scope,
                "p_target_date": key.target_date.isoformat(),
                "p_holder": holder,
                "p_lease_seconds": max(30, min(int(lease_seconds), 3600)),
            },
        )
        if isinstance(payload, list) and payload:
            return dict(payload[0])
        return {"acquired": False, "lease_state": "ERROR"}

    def complete_lease(self, key: EvidenceKey, holder: str, state: str = "COMPLETED") -> bool:
        normalized = key.normalized()
        payload = self._rpc("evidence_complete_refresh_lease", {
            "p_provider": normalized.provider,
            "p_family": normalized.family,
            "p_scope": normalized.scope,
            "p_target_date": normalized.target_date.isoformat(),
            "p_holder": holder,
            "p_result_state": _clean(state).upper() or "COMPLETED",
        })
        return bool(payload)

    def fail_lease(self, key: EvidenceKey, holder: str, reason: str) -> bool:
        normalized = key.normalized()
        payload = self._rpc("evidence_fail_refresh_lease", {
            "p_provider": normalized.provider,
            "p_family": normalized.family,
            "p_scope": normalized.scope,
            "p_target_date": normalized.target_date.isoformat(),
            "p_holder": holder,
            "p_reason": _clean(reason)[:160],
        })
        return bool(payload)

    def record_provider_state(self, row: Mapping[str, Any]) -> None:
        self.upsert_rows(
            "evidence_provider_state",
            [row],
            conflict=("provider", "endpoint_family", "scope", "target_date"),
        )


class SharedEvidenceCoordinator:
    """Cache-first single-flight refresh without polling or scanner conclusions."""

    def __init__(self, backend: EvidenceBackend, *, client_id: str, worker_id: str | None = None):
        self.backend = backend
        self.client_id = _clean(client_id).upper() or "UNKNOWN"
        self.worker_id = worker_id or f"{self.client_id}:{uuid.uuid4()}"
        self._metrics_lock = Lock()
        self._metrics = {
            "calls_attempted": 0,
            "success": 0,
            "failure": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "calls_avoided": 0,
            "refresh_lock_collisions": 0,
        }

    def metrics(self) -> dict[str, int]:
        with self._metrics_lock:
            return dict(self._metrics)

    def _inc(self, name: str) -> None:
        with self._metrics_lock:
            self._metrics[name] += 1

    @staticmethod
    def classify(
        rows: Iterable[Mapping[str, Any]],
        *,
        now: datetime | None = None,
        max_age: timedelta | None = None,
        minimum_rows: int = 1,
    ) -> EvidenceState:
        records = [dict(row) for row in rows]
        if not records:
            return EvidenceState.MISSING
        valid = [row for row in records if _clean(row.get("validation_state", "VALID")).upper() == "VALID"]
        if not valid:
            return EvidenceState.ERROR
        if len(valid) < max(1, int(minimum_rows)):
            return EvidenceState.INSUFFICIENT
        if max_age is not None:
            checked_at = now or datetime.now(timezone.utc)
            stamps: list[datetime] = []
            for row in valid:
                raw = row.get("fetched_at") or row.get("updated_at")
                if not raw:
                    continue
                try:
                    stamp = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                    stamps.append(stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc))
                except ValueError:
                    continue
            if not stamps or checked_at - max(stamps) > max_age:
                return EvidenceState.STALE
        return EvidenceState.VALID

    def get_or_refresh(
        self,
        key: EvidenceKey,
        *,
        read_current: Callable[[], list[Mapping[str, Any]]],
        fetch: Callable[[], list[Mapping[str, Any]]],
        persist: Callable[[list[Mapping[str, Any]]], int],
        validate: Callable[[list[Mapping[str, Any]]], tuple[bool, str]],
        max_age: timedelta | None = None,
        minimum_rows: int = 1,
        lease_seconds: int = 300,
        allow_empty_valid: bool = False,
        read_empty_current: Callable[[], bool] | None = None,
    ) -> RefreshResult:
        normalized = key.normalized()

        def empty_cache_hit() -> bool:
            return bool(
                allow_empty_valid
                and read_empty_current is not None
                and read_empty_current()
            )

        current = read_current()
        state = self.classify(current, max_age=max_age, minimum_rows=minimum_rows)
        if state is EvidenceState.VALID:
            self._inc("cache_hits")
            self._inc("calls_avoided")
            return RefreshResult(state, "CACHE_HIT", tuple(current), cache_hit=True, request_avoided=True)
        if not current and empty_cache_hit():
            self._inc("cache_hits")
            self._inc("calls_avoided")
            return RefreshResult(
                EvidenceState.VALID,
                "CACHE_HIT_EMPTY",
                (),
                cache_hit=True,
                request_avoided=True,
            )

        self._inc("cache_misses")
        lease = dict(self.backend.acquire_lease(normalized, self.worker_id, lease_seconds))
        if not bool(lease.get("acquired")):
            self._inc("refresh_lock_collisions")
            readback = read_current()
            readback_state = self.classify(readback, max_age=max_age, minimum_rows=minimum_rows)
            if readback_state is EvidenceState.VALID:
                self._inc("cache_hits")
                self._inc("calls_avoided")
                return RefreshResult(
                    readback_state, "CACHE_FILLED_BY_OTHER_CLIENT", tuple(readback),
                    cache_hit=True, request_avoided=True, lease_state="LOCKED_REUSED",
                )
            if not readback and empty_cache_hit():
                self._inc("cache_hits")
                self._inc("calls_avoided")
                return RefreshResult(
                    EvidenceState.VALID,
                    "CACHE_FILLED_EMPTY_BY_OTHER_CLIENT",
                    (),
                    cache_hit=True,
                    request_avoided=True,
                    lease_state="LOCKED_REUSED_EMPTY",
                )
            return RefreshResult(
                state, MissingReason.REFRESH_LOCKED.value, tuple(current),
                request_avoided=True, lease_state="LOCKED",
            )

        self._inc("calls_attempted")
        attempted_at = datetime.now(timezone.utc)
        try:
            fetched = [dict(row) for row in fetch()]
            if allow_empty_valid and not fetched:
                if not self.backend.complete_lease(normalized, self.worker_id, "COMPLETED_EMPTY"):
                    raise RuntimeError(MissingReason.REFRESH_LEASE_EXPIRED.value)
                self.backend.record_provider_state({
                    "provider": normalized.provider,
                    "endpoint_family": normalized.family,
                    "scope": normalized.scope,
                    "target_date": normalized.target_date.isoformat(),
                    "last_attempt_at": attempted_at.isoformat(),
                    "last_success_at": datetime.now(timezone.utc).isoformat(),
                    "latest_source_date": normalized.target_date.isoformat(),
                    "response_state": "VALID_EMPTY",
                    "error_classification": None,
                })
                self._inc("success")
                return RefreshResult(
                    EvidenceState.VALID,
                    "REFRESHED_EMPTY",
                    (),
                    provider_called=True,
                    lease_state="COMPLETED_EMPTY",
                )

            valid, reason = validate(fetched)
            if not valid:
                self._inc("failure")
                self.backend.fail_lease(normalized, self.worker_id, reason)
                self.backend.record_provider_state({
                    "provider": normalized.provider,
                    "endpoint_family": normalized.family,
                    "scope": normalized.scope,
                    "target_date": normalized.target_date.isoformat(),
                    "last_attempt_at": attempted_at.isoformat(),
                    "response_state": reason,
                    "error_classification": reason,
                })
                return RefreshResult(EvidenceState.ERROR, reason, provider_called=True, lease_state="FAILED")
            written = int(persist(fetched))
            if written < len(fetched):
                raise RuntimeError(MissingReason.PERSIST_FAILURE.value)
            readback = read_current()
            readback_state = self.classify(readback, max_age=max_age, minimum_rows=minimum_rows)
            if readback_state is not EvidenceState.VALID:
                raise RuntimeError(MissingReason.READBACK_FAILURE.value)
            if not self.backend.complete_lease(normalized, self.worker_id, "COMPLETED"):
                raise RuntimeError(MissingReason.REFRESH_LEASE_EXPIRED.value)
            self.backend.record_provider_state({
                "provider": normalized.provider,
                "endpoint_family": normalized.family,
                "scope": normalized.scope,
                "target_date": normalized.target_date.isoformat(),
                "last_attempt_at": attempted_at.isoformat(),
                "last_success_at": datetime.now(timezone.utc).isoformat(),
                "latest_source_date": normalized.target_date.isoformat(),
                "response_state": "VALID",
                "error_classification": None,
            })
            self._inc("success")
            return RefreshResult(
                EvidenceState.VALID, "REFRESHED", tuple(readback),
                provider_called=True, lease_state="COMPLETED",
            )
        except Exception as exc:
            reason = normalize_failure_reason(exc)
            self._inc("failure")
            self.backend.fail_lease(normalized, self.worker_id, reason)
            try:
                self.backend.record_provider_state({
                    "provider": normalized.provider,
                    "endpoint_family": normalized.family,
                    "scope": normalized.scope,
                    "target_date": normalized.target_date.isoformat(),
                    "last_attempt_at": attempted_at.isoformat(),
                    "response_state": "ERROR",
                    "error_classification": reason,
                })
            except Exception:
                pass
            return RefreshResult(EvidenceState.ERROR, reason, tuple(current), provider_called=True, lease_state="FAILED")


__all__ = [
    "EvidenceKey",
    "EvidenceState",
    "HubConfig",
    "IDX_FLOW_SUPABASE_PROJECT_REF",
    "MissingReason",
    "RefreshResult",
    "SecretStatus",
    "SHARED_EVIDENCE_HUB_VERSION",
    "SharedEvidenceCoordinator",
    "SupabaseEvidenceBackend",
    "normalize_failure_reason",
    "secret_status",
]
