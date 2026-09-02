from __future__ import annotations

"""Isolated Phase 5.6 bounded live-validation harness.

This module is intentionally scanner-neutral.  It may read and upsert factual
shared-evidence rows through the existing hub contract, but it has no scoring,
ranking, recommendation, migration, schema, or ACL behavior.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from io import BytesIO
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Protocol
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests

from idx_trading_calendar import latest_expected_completed_session
from shared_evidence_hub import (
    EvidenceKey,
    EvidenceState,
    HubConfig,
    IDX_FLOW_SUPABASE_PROJECT_REF,
    SharedEvidenceCoordinator,
    SupabaseEvidenceBackend,
)
from shared_stock_summary_evidence import normalize_stock_summary

try:
    from idx_public_participant_provider import aggregate_trade_detail
except ImportError:
    from public_idx_broker_flow import aggregate_trade_detail


VERSION = "1.0.0-phase5.6-bounded-live-gate"
GATE_BRANCH = "gate/phase5-6-bounded-live-validation"
GLOBAL_ZAPI_ATTEMPT_LIMIT = 12
SHARED_EVIDENCE_PROJECT_REF = "mbtsvflwszcgdtijdgas"
STOCK_SUMMARY_ATTEMPT_CAP = 1
FOREIGN_FLOW_ATTEMPT_CAP = 6
FOREIGN_FLOW_MAX_LENGTH = 200
COHORT = ("ASII", "BBCA", "BBRI", "BMRI", "TLKM")
PROVIDER = "ZAPI"
STOCK_SUMMARY_FAMILY = "STOCK_SUMMARY"
FOREIGN_FLOW_FAMILY = "FOREIGN_FLOW"
STOCK_SUMMARY_URL = "https://api.zpi.web.id/v1/finance:idx/stock-summary"
FOREIGN_FLOW_URL = "https://api.zpi.web.id/v1/finance:idx/foreign-flow"
SCOPE = "IDX_COHORT_5_PHASE5_6_LIVE_GATE"
PARTICIPANT_FAMILY = "PARTICIPANT"
PARTICIPANT_SOURCE = "IDX_PUBLIC_TRADE_DETAIL_PUBLIK"
PARTICIPANT_SCOPE = "IDX_COHORT_5_PHASE5_6_PARTICIPANT_GATE"
PARTICIPANT_TARGET_DATE = date(2026, 9, 1)
PARTICIPANT_TABLE = "evidence_participant_flow"
OFFICIAL_HTTP_CAP = 5
OFFICIAL_DOWNLOAD_CAP = 1
OFFICIAL_HOSTS = frozenset({"idxdata3.co.id", "www.idxdata3.co.id"})
OFFICIAL_INDEX_URL = (
    "https://www.idxdata3.co.id/INET_Specification/Market_Summary/Market_Indices/"
    "IX200720.TXT?directory=.%2FIDX+Reporting+PSPP%2FRevitalisasi%2FPUBLIK%2F"
)
PARTICIPANT_PROVENANCE = (
    "VERIFIED_IDX_PUBLIC_TRADE_DETAIL_PARTICIPANT_FLOW_NOT_BENEFICIAL_OWNER"
)

ALLOWED_DELTA_PATHS = frozenset(
    {
        ".github/workflows/full-forward-coverage.yml",
        "tools/phase5_6_bounded_live_validation.py",
        "tests/test_phase5_6_bounded_live_validation.py",
    }
)

STOCK_FIELDS = (
    "provider",
    "trade_date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "previous",
    "volume",
    "value",
    "frequency",
    "bid",
    "offer",
    "bid_volume",
    "offer_volume",
    "listed_shares",
    "tradeable_shares",
    "foreign_buy",
    "foreign_sell",
    "non_regular_volume",
    "non_regular_value",
    "non_regular_frequency",
    "source_url",
    "payload_hash",
    "freshness_state",
    "validation_state",
)
FOREIGN_FIELDS = (
    "provider",
    "trade_date",
    "ticker",
    "foreign_buy_shares",
    "foreign_sell_shares",
    "foreign_net_shares",
    "foreign_buy_value",
    "foreign_sell_value",
    "foreign_net_value",
    "volume",
    "value",
    "flow_unit",
    "source_family",
    "source_url",
    "payload_hash",
    "freshness_state",
    "validation_state",
)
STOCK_ALLOWED_WRITE_FIELDS = frozenset((*STOCK_FIELDS, "fetched_at"))
FOREIGN_ALLOWED_WRITE_FIELDS = frozenset((*FOREIGN_FIELDS, "fetched_at"))
PARTICIPANT_FIELDS = (
    "source",
    "trade_date",
    "ticker",
    "broker_code",
    "buy_value",
    "sell_value",
    "buy_volume",
    "sell_volume",
    "net_value",
    "net_volume",
    "buy_avg",
    "sell_avg",
    "source_url",
    "source_file_hash",
    "source_verified",
    "provenance_state",
    "validation_state",
)
PARTICIPANT_ALLOWED_WRITE_FIELDS = frozenset((*PARTICIPANT_FIELDS, "fetched_at"))


class GateFailure(RuntimeError):
    def __init__(self, code: str):
        self.code = str(code).strip().upper() or "ERROR"
        super().__init__(self.code)


class ProviderTransport(Protocol):
    def get_json(
        self,
        family: str,
        url: str,
        *,
        params: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


@dataclass
class RequestLedger:
    client_id: str
    declared_cumulative_before: int
    family_caps: Mapping[str, int]
    entries: list[dict[str, Any]] = field(default_factory=list)
    refused_attempts: int = 0

    def __post_init__(self) -> None:
        self.client_id = str(self.client_id).strip().upper()
        self.declared_cumulative_before = parse_declared_cumulative(
            self.declared_cumulative_before
        )
        self.family_caps = {
            str(name).strip().upper(): max(0, int(limit))
            for name, limit in self.family_caps.items()
        }

    def before_attempt(
        self,
        family: str,
        endpoint: str,
        params: Mapping[str, Any],
    ) -> int:
        family_name = str(family).strip().upper()
        used = sum(1 for item in self.entries if item["family"] == family_name)
        cap = self.family_caps.get(family_name, 0)
        if used >= cap or self.declared_cumulative_after >= GLOBAL_ZAPI_ATTEMPT_LIMIT:
            self.refused_attempts += 1
            raise GateFailure("ZAPI_CIRCUIT_BREAKER")
        entry = {
            "sequence": len(self.entries) + 1,
            "family": family_name,
            "endpoint": str(endpoint),
            "trade_date": str(params.get("date") or ""),
            "start": int(params.get("start") or 0),
            "length": int(params.get("length") or 0),
            "code": str(params.get("code") or "").strip().upper(),
            "result": "ATTEMPTED",
        }
        self.entries.append(entry)
        return len(self.entries) - 1

    def finish_attempt(self, index: int, result: str) -> None:
        self.entries[index]["result"] = str(result).strip().upper()

    @property
    def attempts(self) -> int:
        return len(self.entries)

    @property
    def declared_cumulative_after(self) -> int:
        return self.declared_cumulative_before + self.attempts

    @property
    def hard_cap(self) -> int:
        return sum(self.family_caps.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "per_run_attempts": self.attempts,
            "declared_cumulative_before": self.declared_cumulative_before,
            "declared_cumulative_after": self.declared_cumulative_after,
            "global_attempt_limit": GLOBAL_ZAPI_ATTEMPT_LIMIT,
            "per_run_hard_cap": self.hard_cap,
            "family_caps": dict(sorted(self.family_caps.items())),
            "refused_attempts": self.refused_attempts,
            "attempts": [dict(item) for item in self.entries],
        }


class BoundedZapiTransport:
    """One counted call per HTTP attempt, with retries and fallbacks disabled."""

    def __init__(
        self,
        api_key: str,
        ledger: RequestLedger,
        *,
        session: Any | None = None,
        timeout_seconds: float = 20.0,
    ):
        if not str(api_key).strip():
            raise GateFailure("CREDENTIAL_MISSING")
        self._api_key = str(api_key).strip()
        self.ledger = ledger
        self.timeout_seconds = max(2.0, min(float(timeout_seconds), 30.0))
        self.session = session or requests.Session()
        if session is None:
            adapter = requests.adapters.HTTPAdapter(max_retries=0)
            self.session.mount("https://", adapter)
            self.session.mount("http://", adapter)

    def get_json(
        self,
        family: str,
        url: str,
        *,
        params: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        endpoint = urlparse(url).path
        index = self.ledger.before_attempt(family, endpoint, params)
        try:
            response = self.session.request(
                "GET",
                url,
                params=dict(params),
                headers={
                    "Accept": "application/json",
                    "User-Agent": "Phase5.6-Bounded-Live-Gate/1.0",
                    "x-api-key": self._api_key,
                },
                timeout=self.timeout_seconds,
                allow_redirects=False,
            )
        except requests.Timeout as exc:
            self.ledger.finish_attempt(index, "TIMEOUT")
            raise GateFailure("TIMEOUT") from exc
        except requests.ConnectionError as exc:
            self.ledger.finish_attempt(index, "CONNECTION_ERROR")
            raise GateFailure("CONNECTION_ERROR") from exc
        except Exception as exc:
            self.ledger.finish_attempt(index, "CONNECTION_ERROR")
            raise GateFailure("CONNECTION_ERROR") from exc

        status = int(getattr(response, "status_code", 0) or 0)
        self.ledger.finish_attempt(index, f"HTTP_{status}")
        if status in {401, 403, 404, 429}:
            raise GateFailure(f"HTTP_{status}")
        if not 200 <= status < 300:
            raise GateFailure("CONNECTION_ERROR" if status >= 500 else "CONTEXT_REJECTED")
        try:
            payload = response.json()
        except (TypeError, ValueError) as exc:
            raise GateFailure("PARSE_FAILURE") from exc
        if not isinstance(payload, Mapping):
            raise GateFailure("PARSE_FAILURE")
        return payload


class DenyZapiTransport:
    """Cache-only transport: every provider call is rejected before the network."""

    def __init__(self, ledger: RequestLedger):
        self.ledger = ledger

    def get_json(
        self,
        family: str,
        url: str,
        *,
        params: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self.ledger.refused_attempts += 1
        raise GateFailure("CACHE_ONLY_PROVIDER_ACCESS_DENIED")


@dataclass
class OfficialRequestLedger:
    client_id: str
    entries: list[dict[str, Any]] = field(default_factory=list)
    downloads: int = 0
    refused_attempts: int = 0
    refused_downloads: int = 0

    def before_attempt(self, url: str, method: str, *, file_download: bool = False) -> int:
        if len(self.entries) >= OFFICIAL_HTTP_CAP:
            self.refused_attempts += 1
            raise GateFailure("OFFICIAL_HTTP_CIRCUIT_BREAKER")
        if file_download and self.downloads >= OFFICIAL_DOWNLOAD_CAP:
            self.refused_downloads += 1
            raise GateFailure("OFFICIAL_DOWNLOAD_CIRCUIT_BREAKER")
        if file_download:
            self.downloads += 1
        self.entries.append({
            "sequence": len(self.entries) + 1,
            "method": str(method).strip().upper(),
            "url": str(url),
            "file_download": bool(file_download),
            "result": "ATTEMPTED",
        })
        return len(self.entries) - 1

    def finish_attempt(self, index: int, result: str) -> None:
        self.entries[index]["result"] = str(result).strip().upper()

    def as_dict(self) -> dict[str, Any]:
        return {
            "client_id": str(self.client_id).strip().upper(),
            "official_http_attempts": len(self.entries),
            "official_http_cap": OFFICIAL_HTTP_CAP,
            "official_downloads": self.downloads,
            "official_download_cap": OFFICIAL_DOWNLOAD_CAP,
            "refused_attempts": self.refused_attempts,
            "refused_downloads": self.refused_downloads,
            "attempts": [dict(item) for item in self.entries],
        }


class BoundedOfficialTransport:
    """Official IDX-only HTTP with a separate pre-transport request ledger."""

    def __init__(
        self,
        ledger: OfficialRequestLedger,
        *,
        session: Any | None = None,
        timeout_seconds: float = 20.0,
    ):
        self.ledger = ledger
        self.timeout_seconds = max(2.0, min(float(timeout_seconds), 45.0))
        self.session = session or requests.Session()
        if session is None:
            adapter = requests.adapters.HTTPAdapter(max_retries=0)
            self.session.mount("https://", adapter)
            self.session.mount("http://", adapter)

    @staticmethod
    def _validate_url(url: str) -> None:
        parsed = urlparse(str(url))
        if parsed.scheme != "https" or (parsed.hostname or "").lower() not in OFFICIAL_HOSTS:
            raise GateFailure("CONTEXT_REJECTED")

    def request(self, url: str, method: str, *, file_download: bool = False) -> Any:
        self._validate_url(url)
        index = self.ledger.before_attempt(url, method, file_download=file_download)
        try:
            response = self.session.request(
                "GET",
                url,
                headers={
                    "Accept": "text/csv,text/plain,application/octet-stream,application/vnd.ms-excel",
                    "User-Agent": "Phase5.6-Bounded-Official-IDX-Gate/1.0",
                },
                timeout=self.timeout_seconds,
                allow_redirects=False,
                stream=method != "DIRECTORY_INDEX",
            )
        except requests.Timeout as exc:
            self.ledger.finish_attempt(index, "TIMEOUT")
            raise GateFailure("TIMEOUT") from exc
        except requests.ConnectionError as exc:
            self.ledger.finish_attempt(index, "CONNECTION_ERROR")
            raise GateFailure("CONNECTION_ERROR") from exc
        except Exception as exc:
            self.ledger.finish_attempt(index, "CONNECTION_ERROR")
            raise GateFailure("CONNECTION_ERROR") from exc

        status = int(getattr(response, "status_code", 0) or 0)
        self.ledger.finish_attempt(index, f"HTTP_{status}")
        if not 200 <= status < 300:
            close = getattr(response, "close", None)
            if callable(close):
                close()
            if status:
                raise GateFailure(f"HTTP_{status}")
            raise GateFailure("CONNECTION_ERROR")
        content_type = str(getattr(response, "headers", {}).get("content-type", "")).lower()
        allowed = ("text/csv", "text/plain", "octet-stream", "application/vnd.ms-excel")
        if not any(token in content_type for token in allowed):
            close = getattr(response, "close", None)
            if callable(close):
                close()
            raise GateFailure("PARSE_FAILURE")
        return response


class DenyOfficialTransport:
    """Cache-only participant capability that refuses before any network call."""

    def __init__(self, ledger: OfficialRequestLedger):
        self.ledger = ledger

    def request(self, url: str, method: str, *, file_download: bool = False) -> Any:
        del url, method, file_download
        self.ledger.refused_attempts += 1
        raise GateFailure("CACHE_ONLY_OFFICIAL_ACCESS_DENIED")


@dataclass(frozen=True)
class FamilySpec:
    family: str
    table: str
    fields: tuple[str, ...]
    allowed_write_fields: frozenset[str]


STOCK_SPEC = FamilySpec(
    STOCK_SUMMARY_FAMILY,
    "evidence_market_daily",
    STOCK_FIELDS,
    STOCK_ALLOWED_WRITE_FIELDS,
)
FOREIGN_SPEC = FamilySpec(
    FOREIGN_FLOW_FAMILY,
    "evidence_foreign_flow",
    FOREIGN_FIELDS,
    FOREIGN_ALLOWED_WRITE_FIELDS,
)
PARTICIPANT_SPEC = FamilySpec(
    PARTICIPANT_FAMILY,
    PARTICIPANT_TABLE,
    PARTICIPANT_FIELDS,
    PARTICIPANT_ALLOWED_WRITE_FIELDS,
)


def parse_declared_cumulative(value: Any) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise GateFailure("INVALID_DECLARED_CUMULATIVE_ATTEMPTS") from exc
    if not 0 <= parsed <= GLOBAL_ZAPI_ATTEMPT_LIMIT:
        raise GateFailure("INVALID_DECLARED_CUMULATIVE_ATTEMPTS")
    return parsed


def credentials_from_environment(client_id: str) -> tuple[HubConfig | None, dict[str, str]]:
    url = str(os.getenv("SHARED_EVIDENCE_SUPABASE_URL", "")).strip().rstrip("/")
    secret = str(os.getenv("SHARED_EVIDENCE_SUPABASE_SECRET_KEY", "")).strip()
    service_role = str(os.getenv("SHARED_EVIDENCE_SUPABASE_SERVICE_ROLE_KEY", "")).strip()
    key_type, key = ("SECRET", secret) if secret else (("SERVICE_ROLE", service_role) if service_role else ("NONE", ""))
    project_ref = (urlparse(url).hostname or "").split(".", 1)[0]
    blocked = project_ref == IDX_FLOW_SUPABASE_PROJECT_REF
    project_matches = project_ref == SHARED_EVIDENCE_PROJECT_REF
    status = {
        "shared_db_url": "CONFIGURED" if url and not blocked and project_matches else "MISSING",
        "shared_db_project": "CONFIGURED" if project_matches else "MISMATCH",
        "shared_db_key": "CONFIGURED" if key else "MISSING",
        "shared_db_key_type": key_type,
        "zapi_key": "CONFIGURED" if str(os.getenv("ZAPI_KEY", "")).strip() else "MISSING",
    }
    if not url or not key or blocked or not project_matches:
        return None, status
    return HubConfig(url=url, key=key, key_type=key_type, client_id=client_id), status


def _unwrap(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    current = payload
    for _ in range(3):
        nested = current.get("data")
        if isinstance(nested, Mapping) and any(
            name in nested for name in ("data", "recordsTotal", "recordsFiltered", "total", "date")
        ):
            current = nested
        else:
            break
    return current


def _date_value(value: Any) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _number(value: Any) -> int | float | None:
    if value is None or isinstance(value, bool) or str(value).strip() == "":
        return None
    try:
        number = Decimal(str(value).replace(",", ""))
    except (InvalidOperation, ValueError) as exc:
        raise GateFailure("PARSE_FAILURE") from exc
    if not number.is_finite():
        raise GateFailure("PARSE_FAILURE")
    return int(number) if number == number.to_integral_value() else float(number)


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        try:
            number = Decimal(str(value))
            if not number.is_finite():
                return str(value)
            normalized = number.normalize()
            return format(normalized, "f")
        except InvalidOperation:
            return str(value)
    return str(value)


def facts_hash(rows: Iterable[Mapping[str, Any]], spec: FamilySpec) -> str:
    canonical = [
        {field: _canonical_value(row.get(field)) for field in spec.fields}
        for row in rows
    ]
    canonical.sort(key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":")))
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ensure_scanner_neutral(rows: Iterable[Mapping[str, Any]], spec: FamilySpec) -> None:
    forbidden_fragments = (
        "score",
        "rank",
        "recommend",
        "signal",
        "entry",
        "stop_loss",
        "take_profit",
        "real_money",
    )
    for row in rows:
        extra = set(row) - set(spec.allowed_write_fields)
        if extra or any(fragment in name.lower() for name in row for fragment in forbidden_fragments):
            raise GateFailure("SCANNER_SEMANTIC_FIELD_REJECTED")


def select_cohort(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    wanted = set(COHORT)
    return [
        dict(row)
        for row in rows
        if str(row.get("ticker") or "").strip().upper().removesuffix(".JK") in wanted
    ]


def classify_rows(
    rows: Iterable[Mapping[str, Any]],
    spec: FamilySpec,
    target_date: date,
) -> tuple[str, str, list[dict[str, Any]]]:
    selected = select_cohort(rows)
    if not selected:
        return EvidenceState.MISSING.value, "MISSING", []
    tickers = [str(row.get("ticker") or "").strip().upper().removesuffix(".JK") for row in selected]
    if len(set(tickers)) != len(tickers):
        return EvidenceState.ERROR.value, "DUPLICATE_FACT", selected
    if any(str(row.get("provider") or "").upper() != PROVIDER for row in selected):
        return EvidenceState.ERROR.value, "PROVIDER_MISMATCH", selected
    if any(_date_value(row.get("trade_date")) != target_date for row in selected):
        return EvidenceState.ERROR.value, "WRONG_PERIOD", selected
    if any(str(row.get("validation_state") or "").upper() != "VALID" for row in selected):
        return EvidenceState.ERROR.value, "PROVIDER_ERROR", selected
    if any(str(row.get("freshness_state") or "CURRENT").upper() != "CURRENT" for row in selected):
        return EvidenceState.STALE.value, "STALE", selected
    if set(tickers) != set(COHORT):
        return EvidenceState.INSUFFICIENT.value, "INSUFFICIENT_HISTORY", selected
    if any(not str(row.get("payload_hash") or "").strip() for row in selected):
        return EvidenceState.ERROR.value, "PARSE_FAILURE", selected
    if spec is STOCK_SPEC and any(
        all(row.get(name) is None for name in ("open", "close", "volume", "value"))
        for row in selected
    ):
        return EvidenceState.ERROR.value, "PARSE_FAILURE", selected
    if spec is FOREIGN_SPEC and any(
        all(row.get(name) is None for name in ("foreign_buy_shares", "foreign_sell_shares", "foreign_net_shares"))
        for row in selected
    ):
        return EvidenceState.ERROR.value, "PARSE_FAILURE", selected
    return EvidenceState.VALID.value, "VALID", selected


def read_family_rows(backend: Any, spec: FamilySpec, target_date: date) -> list[dict[str, Any]]:
    rows = backend.read_rows(
        spec.table,
        {"provider": PROVIDER, "trade_date": target_date.isoformat()},
        limit=5000,
    )
    return select_cohort(rows)


def cache_only_consume(
    backend: Any,
    spec: FamilySpec,
    target_date: date,
    transport: DenyZapiTransport,
) -> dict[str, Any]:
    del transport  # The injected deny transport is the only legal provider capability here.
    state, reason, rows = classify_rows(read_family_rows(backend, spec, target_date), spec, target_date)
    if state != EvidenceState.VALID.value:
        return {
            "classification": state,
            "reason": reason,
            "facts_hash": "",
            "rows": len(rows),
            "identical_readback": False,
        }
    first_hash = facts_hash(rows, spec)
    readback = read_family_rows(backend, spec, target_date)
    second_hash = facts_hash(readback, spec)
    if first_hash != second_hash:
        raise GateFailure("READBACK_FAILURE")
    return {
        "classification": EvidenceState.VALID.value,
        "reason": "CACHE_ONLY_VALID",
        "facts_hash": first_hash,
        "rows": len(rows),
        "identical_readback": True,
    }


def _expected_participant_filename(target_date: date) -> str:
    return f"Trade-Detail-Publik_{target_date:%Y%m%d}.csv"


def _response_bytes(response: Any) -> bytes:
    raw = getattr(response, "content", None)
    if isinstance(raw, bytes):
        return raw
    iterator = getattr(response, "iter_content", None)
    if callable(iterator):
        return b"".join(chunk for chunk in iterator(chunk_size=1024 * 1024) if chunk)
    text = getattr(response, "text", "")
    return str(text).encode("utf-8")


def _recoverable_discovery_failure(code: str) -> bool:
    state = str(code).strip().upper()
    return state in {"CONNECTION_ERROR", "TIMEOUT", "HTTP_404"} or bool(
        re.fullmatch(r"HTTP_5\d\d", state)
    )


def discover_participant_url(
    transport: BoundedOfficialTransport,
    target_date: date,
) -> str:
    filename = _expected_participant_filename(target_date)
    last_recoverable = ""
    try:
        response = transport.request(OFFICIAL_INDEX_URL, "DIRECTORY_INDEX")
    except GateFailure as exc:
        if not _recoverable_discovery_failure(exc.code):
            raise
        last_recoverable = exc.code
    else:
        try:
            index_text = str(getattr(response, "text", ""))
            if not index_text:
                index_text = _response_bytes(response).decode("utf-8", errors="replace")
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
        matches = re.findall(
            r"href=[\"']([^\"']*" + re.escape(filename) + r")[\"']",
            index_text,
            flags=re.I,
        )
        for href in matches:
            candidate = urljoin(OFFICIAL_INDEX_URL, href)
            BoundedOfficialTransport._validate_url(candidate)
            if urlparse(candidate).path.rsplit("/", 1)[-1].lower() == filename.lower():
                return candidate

    candidates = (
        f"https://www.idxdata3.co.id/IDX%20Reporting%20PSPP/Revitalisasi/PUBLIK/{filename}",
        f"https://idxdata3.co.id/IDX%20Reporting%20PSPP/Revitalisasi/PUBLIK/{filename}",
        f"https://www.idxdata3.co.id/Market_Summary/Market_Summary/{filename}",
    )
    for candidate in candidates:
        try:
            response = transport.request(candidate, "DOCUMENTED_PATH_PROBE")
        except GateFailure as exc:
            if not _recoverable_discovery_failure(exc.code):
                raise
            last_recoverable = exc.code
            continue
        close = getattr(response, "close", None)
        if callable(close):
            close()
        return candidate
    raise GateFailure(last_recoverable or "HTTP_404")


def download_participant_file(
    transport: BoundedOfficialTransport,
    target_date: date,
) -> tuple[bytes, str, str]:
    source_url = discover_participant_url(transport, target_date)
    if urlparse(source_url).path.rsplit("/", 1)[-1] != _expected_participant_filename(target_date):
        raise GateFailure("CONTEXT_REJECTED")
    response = transport.request(source_url, "DOWNLOAD", file_download=True)
    try:
        body = _response_bytes(response)
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()
    if not body:
        raise GateFailure("EMPTY_RESPONSE")
    return body, source_url, hashlib.sha256(body).hexdigest()


def _read_participant_csv(body: bytes, target_date: date) -> pd.DataFrame:
    aliases = {
        "seccode": "asset", "code": "asset", "ticker": "asset",
        "brokersellid": "participant_sell", "brokerbuyid": "participant_buy",
        "sellbrokerid": "participant_sell", "buybrokerid": "participant_buy",
        "quantity": "volume", "tradedate": "tradingdate",
    }
    required = {"asset", "participant_buy", "participant_sell", "volume", "value", "tradingdate"}
    parsed: pd.DataFrame | None = None
    for separator in ("|", ","):
        try:
            candidate = pd.read_csv(BytesIO(body), sep=separator, dtype=str, low_memory=False)
        except Exception:
            continue
        candidate.columns = [str(column).strip().lower() for column in candidate.columns]
        candidate = candidate.rename(columns={
            old: new for old, new in aliases.items()
            if old in candidate.columns and new not in candidate.columns
        })
        if required.issubset(candidate.columns):
            parsed = candidate
            break
    if parsed is None or parsed.empty:
        raise GateFailure("PARSE_FAILURE")
    dates = pd.to_datetime(parsed["tradingdate"], errors="coerce").dt.date
    if dates.isna().any():
        raise GateFailure("PARSE_FAILURE")
    if set(dates) != {target_date}:
        raise GateFailure("WRONG_PERIOD")
    parsed["asset"] = parsed["asset"].astype(str).str.strip().str.upper().str.removesuffix(".JK")
    for name in ("participant_buy", "participant_sell"):
        parsed[name] = parsed[name].fillna("").astype(str).str.strip().str.upper()
    if ((parsed["participant_buy"] == "") & (parsed["participant_sell"] == "")).any():
        raise GateFailure("PARSE_FAILURE")
    for name in ("volume", "value"):
        numeric = pd.to_numeric(parsed[name], errors="coerce")
        if numeric.isna().any():
            raise GateFailure("PARSE_FAILURE")
    cohort_rows = parsed[parsed["asset"].isin(COHORT)]
    if set(cohort_rows["asset"]) != set(COHORT):
        raise GateFailure("INSUFFICIENT_HISTORY")
    return parsed


def parse_participant_rows(
    body: bytes,
    target_date: date,
    source_url: str,
    source_file_hash: str,
) -> list[dict[str, Any]]:
    _read_participant_csv(body, target_date)
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    path = Path(handle.name)
    try:
        handle.write(body)
        handle.close()
        try:
            frame = aggregate_trade_detail(path, target_date, universe=COHORT)
        except Exception as exc:
            raise GateFailure("PARSE_FAILURE") from exc
    finally:
        if not handle.closed:
            handle.close()
        path.unlink(missing_ok=True)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise GateFailure("PARSE_FAILURE")
    tickers = {
        str(value).strip().upper().removesuffix(".JK")
        for value in frame.get("ticker", pd.Series(dtype=str))
    }
    if tickers != set(COHORT):
        raise GateFailure("INSUFFICIENT_HISTORY")
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    for item in frame.to_dict(orient="records"):
        ticker = str(item.get("ticker") or "").strip().upper().removesuffix(".JK")
        broker = str(item.get("broker_code") or "").strip().upper()
        if ticker not in COHORT or not broker:
            continue
        values = {name: _number(item.get(name)) for name in (
            "buy_value", "sell_value", "buy_volume", "sell_volume",
            "net_value", "net_volume",
        )}
        for name in ("buy_avg", "sell_avg"):
            raw = item.get(name)
            values[name] = None if pd.isna(raw) else _number(raw)
        if any(values[name] is None for name in (
            "buy_value", "sell_value", "buy_volume", "sell_volume", "net_value", "net_volume"
        )):
            raise GateFailure("PARSE_FAILURE")
        rows.append({
            "source": PARTICIPANT_SOURCE,
            "trade_date": target_date.isoformat(),
            "ticker": ticker,
            "broker_code": broker,
            **values,
            "source_url": source_url,
            "source_file_hash": source_file_hash,
            "source_verified": True,
            "provenance_state": PARTICIPANT_PROVENANCE,
            "fetched_at": fetched_at,
            "validation_state": "VALID",
        })
    rows.sort(key=lambda row: (row["ticker"], row["broker_code"]))
    if {row["ticker"] for row in rows} != set(COHORT):
        raise GateFailure("INSUFFICIENT_HISTORY")
    ensure_scanner_neutral(rows, PARTICIPANT_SPEC)
    return rows


def classify_participant_rows(
    rows: Iterable[Mapping[str, Any]],
    target_date: date,
) -> tuple[str, str, list[dict[str, Any]]]:
    selected = [
        dict(row) for row in rows
        if str(row.get("ticker") or "").strip().upper().removesuffix(".JK") in COHORT
    ]
    if not selected:
        return EvidenceState.MISSING.value, "MISSING", []
    keys = [
        (str(row.get("ticker") or "").strip().upper(), str(row.get("broker_code") or "").strip().upper())
        for row in selected
    ]
    if len(keys) != len(set(keys)):
        return EvidenceState.ERROR.value, "DUPLICATE_FACT", selected
    if any(str(row.get("source") or "").upper() != PARTICIPANT_SOURCE for row in selected):
        return EvidenceState.ERROR.value, "PROVIDER_MISMATCH", selected
    if any(_date_value(row.get("trade_date")) != target_date for row in selected):
        return EvidenceState.ERROR.value, "WRONG_PERIOD", selected
    if any(str(row.get("validation_state") or "").upper() != "VALID" for row in selected):
        return EvidenceState.ERROR.value, "PROVIDER_ERROR", selected
    if any(not bool(row.get("source_verified")) for row in selected):
        return EvidenceState.ERROR.value, "CONTEXT_REJECTED", selected
    if any(str(row.get("provenance_state") or "") != PARTICIPANT_PROVENANCE for row in selected):
        return EvidenceState.ERROR.value, "CONTEXT_REJECTED", selected
    if any(not str(row.get("source_file_hash") or "").strip() for row in selected):
        return EvidenceState.ERROR.value, "PARSE_FAILURE", selected
    if len({str(row.get("source_file_hash")) for row in selected}) != 1:
        return EvidenceState.ERROR.value, "READBACK_FAILURE", selected
    if {ticker for ticker, _ in keys} != set(COHORT):
        return EvidenceState.INSUFFICIENT.value, "INSUFFICIENT_HISTORY", selected
    return EvidenceState.VALID.value, "VALID", selected


def read_participant_rows(backend: Any, target_date: date) -> list[dict[str, Any]]:
    return backend.read_rows(
        PARTICIPANT_TABLE,
        {"source": PARTICIPANT_SOURCE, "trade_date": target_date.isoformat()},
        limit=50000,
    )


def consume_participant_cache(
    backend: Any,
    target_date: date,
    transport: DenyOfficialTransport,
) -> dict[str, Any]:
    del transport  # The injected deny transport is the only legal official capability here.
    state, reason, rows = classify_participant_rows(read_participant_rows(backend, target_date), target_date)
    if state != EvidenceState.VALID.value:
        return {
            "classification": state, "reason": reason, "facts_hash": "",
            "rows": len(rows), "identical_readback": False,
        }
    first_hash = facts_hash(rows, PARTICIPANT_SPEC)
    second_hash = facts_hash(read_participant_rows(backend, target_date), PARTICIPANT_SPEC)
    if first_hash != second_hash:
        raise GateFailure("READBACK_FAILURE")
    return {
        "classification": EvidenceState.VALID.value,
        "reason": "CACHE_ONLY_VALID",
        "facts_hash": first_hash,
        "rows": len(rows),
        "identical_readback": True,
    }


def produce_participant(
    backend: Any,
    target_date: date,
    client_id: str,
    fetch: Any,
) -> dict[str, Any]:
    produced_hash = ""

    def read_current() -> list[dict[str, Any]]:
        return read_participant_rows(backend, target_date)

    def fetch_once() -> list[dict[str, Any]]:
        nonlocal produced_hash
        rows = [dict(row) for row in fetch()]
        ensure_scanner_neutral(rows, PARTICIPANT_SPEC)
        produced_hash = facts_hash(rows, PARTICIPANT_SPEC)
        return rows

    def persist(rows: list[Mapping[str, Any]]) -> int:
        written = backend.upsert_rows(
            PARTICIPANT_TABLE,
            rows,
            conflict=("source", "trade_date", "ticker", "broker_code"),
        )
        return len(written)

    def validate(rows: list[Mapping[str, Any]]) -> tuple[bool, str]:
        state, reason, _ = classify_participant_rows(rows, target_date)
        return state == EvidenceState.VALID.value, reason

    coordinator = SharedEvidenceCoordinator(
        backend,
        client_id=client_id,
        worker_id=f"{client_id}-phase5-6-participant-gate",
    )
    result = coordinator.get_or_refresh(
        EvidenceKey("IDX", "TRADE_DETAIL", PARTICIPANT_SCOPE, target_date),
        read_current=read_current,
        fetch=fetch_once,
        persist=persist,
        validate=validate,
        minimum_rows=len(COHORT),
        lease_seconds=600,
    )
    rows = [dict(row) for row in result.rows]
    state, reason, selected = classify_participant_rows(rows, target_date)
    if result.state is EvidenceState.ERROR:
        state, reason = EvidenceState.ERROR.value, result.reason
    readback_hash = facts_hash(selected, PARTICIPANT_SPEC) if state == EvidenceState.VALID.value else ""
    exact = bool(readback_hash and (not produced_hash or produced_hash == readback_hash))
    if result.reason == "REFRESHED" and not exact:
        raise GateFailure("READBACK_FAILURE")
    return {
        "classification": state,
        "reason": result.reason if state == EvidenceState.VALID.value else reason,
        "facts_hash": readback_hash,
        "rows": len(selected),
        "ticker_breadth": len({row.get("ticker") for row in selected}),
        "identical_readback": exact,
        "lease_state": result.lease_state,
        "official_called": result.provider_called,
    }


def fetch_stock_summary(
    transport: ProviderTransport,
    target_date: date,
) -> list[dict[str, Any]]:
    payload = transport.get_json(
        STOCK_SUMMARY_FAMILY,
        STOCK_SUMMARY_URL,
        params={"date": target_date.isoformat(), "length": 5000, "start": 0},
    )
    try:
        rows = normalize_stock_summary(payload, trade_date=target_date, universe=COHORT)
    except RuntimeError as exc:
        raise GateFailure(str(exc)) from exc
    state, reason, selected = classify_rows(rows, STOCK_SPEC, target_date)
    if state != EvidenceState.VALID.value:
        raise GateFailure(reason)
    ensure_scanner_neutral(selected, STOCK_SPEC)
    return selected


def _foreign_page_rows(
    payload: Mapping[str, Any],
    target_date: date,
    *,
    requested_ticker: str | None = None,
) -> list[dict[str, Any]]:
    root = _unwrap(payload)
    raw = root.get("data")
    if not isinstance(raw, list):
        raise GateFailure("PARSE_FAILURE")
    root_date_value = root.get("date") or root.get("Date")
    root_date = _date_value(root_date_value)
    if root_date_value and root_date is None:
        raise GateFailure("PARSE_FAILURE")
    if root_date is not None and root_date != target_date:
        raise GateFailure("WRONG_PERIOD")
    expected = str(requested_ticker or "").strip().upper().removesuffix(".JK")
    rows: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise GateFailure("PARSE_FAILURE")
        ticker = str(item.get("code") or item.get("StockCode") or "").strip().upper().removesuffix(".JK")
        if not ticker:
            raise GateFailure("PARSE_FAILURE")
        item_date_value = item.get("date") or item.get("Date")
        item_date = _date_value(item_date_value)
        if item_date_value and item_date is None:
            raise GateFailure("PARSE_FAILURE")
        item_date = item_date or root_date
        if item_date is None or item_date != target_date:
            raise GateFailure("WRONG_PERIOD")
        if expected and ticker != expected:
            raise GateFailure("TICKER_MISMATCH")
        if ticker not in COHORT:
            continue
        buy = _number(item.get("foreignBuyShares"))
        sell = _number(item.get("foreignSellShares"))
        net = _number(item.get("netForeignShares"))
        if buy is None or sell is None or net is None:
            raise GateFailure("PARSE_FAILURE")
        rows.append(
            {
                "provider": PROVIDER,
                "trade_date": target_date.isoformat(),
                "ticker": ticker,
                "foreign_buy_shares": buy,
                "foreign_sell_shares": sell,
                "foreign_net_shares": net,
                "foreign_buy_value": _number(item.get("foreignBuyValue")),
                "foreign_sell_value": _number(item.get("foreignSellValue")),
                "foreign_net_value": _number(item.get("netForeignValue")),
                "volume": _number(item.get("volume")),
                "value": _number(item.get("value")),
                "flow_unit": "SHARES",
                "source_family": "ZAPI_IDX_FOREIGN_FLOW",
                "source_url": FOREIGN_FLOW_URL,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "freshness_state": "CURRENT",
                "validation_state": "VALID",
            }
        )
    return rows


def fetch_foreign_flow(
    transport: ProviderTransport,
    target_date: date,
) -> list[dict[str, Any]]:
    common_params = {
        "date": target_date.isoformat(),
        "length": FOREIGN_FLOW_MAX_LENGTH,
        "start": 0,
        "sort": "code",
    }
    bulk_payload = transport.get_json(
        FOREIGN_FLOW_FAMILY,
        FOREIGN_FLOW_URL,
        params=common_params,
    )
    rows_by_ticker: dict[str, dict[str, Any]] = {}

    def retain_unique(rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            ticker = row["ticker"]
            if ticker in rows_by_ticker:
                raise GateFailure("PARSE_FAILURE")
            rows_by_ticker[ticker] = row

    retain_unique(_foreign_page_rows(bulk_payload, target_date))
    for ticker in COHORT:
        if ticker in rows_by_ticker:
            continue
        payload = transport.get_json(
            FOREIGN_FLOW_FAMILY,
            FOREIGN_FLOW_URL,
            params={**common_params, "length": 1, "code": ticker},
        )
        delta_rows = _foreign_page_rows(
            payload,
            target_date,
            requested_ticker=ticker,
        )
        if len(delta_rows) > 1:
            raise GateFailure("PARSE_FAILURE")
        retain_unique(delta_rows)
        if len(rows_by_ticker) == len(COHORT):
            break
    rows = [rows_by_ticker[ticker] for ticker in COHORT if ticker in rows_by_ticker]
    if len(rows) != len(COHORT):
        raise GateFailure("INSUFFICIENT_HISTORY")
    canonical = [
        {name: value for name, value in row.items() if name not in {"payload_hash", "fetched_at"}}
        for row in sorted(rows, key=lambda value: value["ticker"])
    ]
    payload_hash = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    for row in rows:
        row["payload_hash"] = payload_hash
    state, reason, selected = classify_rows(rows, FOREIGN_SPEC, target_date)
    if state != EvidenceState.VALID.value:
        raise GateFailure(reason)
    ensure_scanner_neutral(selected, FOREIGN_SPEC)
    return selected


def produce_family(
    backend: Any,
    spec: FamilySpec,
    target_date: date,
    client_id: str,
    fetch: Any,
) -> dict[str, Any]:
    produced_hash = ""

    def read_current() -> list[dict[str, Any]]:
        return read_family_rows(backend, spec, target_date)

    def fetch_once() -> list[dict[str, Any]]:
        nonlocal produced_hash
        rows = [dict(row) for row in fetch()]
        ensure_scanner_neutral(rows, spec)
        produced_hash = facts_hash(rows, spec)
        return rows

    def persist(rows: list[Mapping[str, Any]]) -> int:
        written = backend.upsert_rows(
            spec.table,
            rows,
            conflict=("provider", "trade_date", "ticker"),
        )
        return len(written)

    def validate(rows: list[Mapping[str, Any]]) -> tuple[bool, str]:
        state, reason, _ = classify_rows(rows, spec, target_date)
        return state == EvidenceState.VALID.value, reason

    coordinator = SharedEvidenceCoordinator(
        backend,
        client_id=client_id,
        worker_id=f"{client_id}-phase5-6-live-gate",
    )
    result = coordinator.get_or_refresh(
        EvidenceKey(PROVIDER, spec.family, SCOPE, target_date),
        read_current=read_current,
        fetch=fetch_once,
        persist=persist,
        validate=validate,
        minimum_rows=len(COHORT),
        lease_seconds=300,
    )
    rows = [dict(row) for row in result.rows]
    state, reason, selected = classify_rows(rows, spec, target_date)
    if result.state is EvidenceState.ERROR:
        state, reason = EvidenceState.ERROR.value, result.reason
    readback_hash = facts_hash(selected, spec) if state == EvidenceState.VALID.value else ""
    exact = bool(readback_hash and (not produced_hash or produced_hash == readback_hash))
    if result.reason == "REFRESHED" and not exact:
        raise GateFailure("READBACK_FAILURE")
    return {
        "classification": state,
        "reason": result.reason if state == EvidenceState.VALID.value else reason,
        "facts_hash": readback_hash,
        "rows": len(selected),
        "identical_readback": exact,
        "lease_state": result.lease_state,
        "provider_called": result.provider_called,
    }


def execute_participant_gate(
    client_id: str,
    *,
    backend: Any | None = None,
    official_session: Any | None = None,
    target_date: date = PARTICIPANT_TARGET_DATE,
    declared_cumulative_zapi: int = 4,
) -> tuple[dict[str, Any], int]:
    client = str(client_id).strip().upper()
    if client not in {"PASTICUAN", "EMIR"}:
        raise GateFailure("INVALID_CLIENT")
    if not isinstance(target_date, date):
        raise GateFailure("WRONG_PERIOD")
    declared = parse_declared_cumulative(declared_cumulative_zapi)
    config, credential_status = credentials_from_environment(client)
    ledger = OfficialRequestLedger(client)
    base_report: dict[str, Any] = {
        "version": VERSION,
        "validation_family": "participant",
        "client_id": client,
        "trade_date": target_date.isoformat(),
        "cohort": list(COHORT),
        "credential_status": credential_status,
        "schema_mutations": 0,
        "zapi_ledger": {
            "declared_cumulative_before": declared,
            "declared_cumulative_after": declared,
            "per_run_attempts": 0,
        },
    }
    if backend is None and config is None:
        return {
            **base_report,
            "state": "CREDENTIAL_MISSING",
            "mode": "FAIL_CLOSED",
            "official_request_ledger": ledger.as_dict(),
        }, 2
    active_backend = backend or SupabaseEvidenceBackend(config)
    state, reason, _ = classify_participant_rows(
        read_participant_rows(active_backend, target_date), target_date
    )
    if state in {EvidenceState.ERROR.value, EvidenceState.STALE.value}:
        return {
            **base_report,
            "state": state,
            "reason": reason,
            "mode": "FAIL_CLOSED",
            "official_request_ledger": ledger.as_dict(),
        }, 1

    if client == "EMIR" or state == EvidenceState.VALID.value:
        result = consume_participant_cache(
            active_backend, target_date, DenyOfficialTransport(ledger)
        )
        ok = result["classification"] == EvidenceState.VALID.value
        return {
            **base_report,
            "state": "EMIR_PARTICIPANT_CONSUMER" if client == "EMIR" else "PASTICUAN_PARTICIPANT_CACHE",
            "mode": "CACHE_ONLY_CONSUMER",
            "evidence": result,
            "official_request_ledger": ledger.as_dict(),
        }, 0 if ok else 1

    transport = BoundedOfficialTransport(ledger, session=official_session)

    def fetch() -> list[dict[str, Any]]:
        body, source_url, source_hash = download_participant_file(transport, target_date)
        return parse_participant_rows(body, target_date, source_url, source_hash)

    result = produce_participant(active_backend, target_date, client, fetch)
    ok = result["classification"] == EvidenceState.VALID.value
    return {
        **base_report,
        "state": "PASTICUAN_PARTICIPANT_PRODUCER",
        "mode": "OFFICIAL_LIVE_PRODUCER" if result["official_called"] else "CACHE_FILLED_BY_OTHER_CLIENT",
        "evidence": result,
        "official_request_ledger": ledger.as_dict(),
    }, 0 if ok else 1


def execute_gate(
    client_id: str,
    declared_cumulative: int,
    *,
    backend: Any | None = None,
    transport_session: Any | None = None,
    target_date: date | None = None,
) -> tuple[dict[str, Any], int]:
    client = str(client_id).strip().upper()
    if client not in {"PASTICUAN", "EMIR"}:
        raise GateFailure("INVALID_CLIENT")
    declared = parse_declared_cumulative(declared_cumulative)
    config, credential_status = credentials_from_environment(client)
    base_report: dict[str, Any] = {
        "version": VERSION,
        "client_id": client,
        "cohort": list(COHORT),
        "credential_status": credential_status,
        "schema_mutations": 0,
        "official_downloads": 0,
    }
    if backend is None and config is None:
        ledger = RequestLedger(client, declared, {})
        return {
            **base_report,
            "state": "CREDENTIAL_MISSING",
            "mode": "FAIL_CLOSED",
            "request_ledger": ledger.as_dict(),
        }, 2
    active_backend = backend or SupabaseEvidenceBackend(config)
    day = target_date or latest_expected_completed_session().date()
    base_report["trade_date"] = day.isoformat()

    stock_state, stock_reason, _ = classify_rows(
        read_family_rows(active_backend, STOCK_SPEC, day), STOCK_SPEC, day
    )
    foreign_state, foreign_reason, _ = classify_rows(
        read_family_rows(active_backend, FOREIGN_SPEC, day), FOREIGN_SPEC, day
    )

    if stock_state in {EvidenceState.ERROR.value, EvidenceState.STALE.value}:
        ledger = RequestLedger(client, declared, {})
        return {
            **base_report,
            "state": stock_state,
            "reason": stock_reason,
            "mode": "FAIL_CLOSED",
            "family": STOCK_SUMMARY_FAMILY,
            "request_ledger": ledger.as_dict(),
        }, 1

    if client == "PASTICUAN":
        if stock_state != EvidenceState.VALID.value:
            ledger = RequestLedger(client, declared, {STOCK_SUMMARY_FAMILY: STOCK_SUMMARY_ATTEMPT_CAP})
            if credential_status["zapi_key"] != "CONFIGURED":
                return {
                    **base_report,
                    "state": "CREDENTIAL_MISSING",
                    "mode": "FAIL_CLOSED",
                    "family": STOCK_SUMMARY_FAMILY,
                    "request_ledger": ledger.as_dict(),
                }, 2
            transport = BoundedZapiTransport(
                os.environ["ZAPI_KEY"], ledger, session=transport_session
            )
            result = produce_family(
                active_backend,
                STOCK_SPEC,
                day,
                client,
                lambda: fetch_stock_summary(transport, day),
            )
            ok = result["classification"] == EvidenceState.VALID.value
            return {
                **base_report,
                "state": "PASTICUAN_STATE_A",
                "mode": ("LIVE_PRODUCER" if result["provider_called"] else "CACHE_FILLED_BY_OTHER_CLIENT") if ok else "ERROR",
                "family": STOCK_SUMMARY_FAMILY,
                "evidence": result,
                "request_ledger": ledger.as_dict(),
            }, 0 if ok else 1
        if foreign_state in {EvidenceState.ERROR.value, EvidenceState.STALE.value}:
            ledger = RequestLedger(client, declared, {})
            return {
                **base_report,
                "state": foreign_state,
                "reason": foreign_reason,
                "mode": "FAIL_CLOSED",
                "family": FOREIGN_FLOW_FAMILY,
                "request_ledger": ledger.as_dict(),
            }, 1
        if foreign_state != EvidenceState.VALID.value:
            ledger = RequestLedger(client, declared, {})
            return {
                **base_report,
                "state": "PASTICUAN_STATE_B",
                "mode": "SAFE_STOP_WAITING_FOR_EMIR",
                "family": FOREIGN_FLOW_FAMILY,
                "classification": foreign_state,
                "reason": foreign_reason,
                "request_ledger": ledger.as_dict(),
            }, 0
        ledger = RequestLedger(client, declared, {})
        result = cache_only_consume(active_backend, FOREIGN_SPEC, day, DenyZapiTransport(ledger))
        return {
            **base_report,
            "state": "PASTICUAN_STATE_C",
            "mode": "CACHE_ONLY_CONSUMER",
            "family": FOREIGN_FLOW_FAMILY,
            "evidence": result,
            "request_ledger": ledger.as_dict(),
        }, 0 if result["classification"] == EvidenceState.VALID.value else 1

    if stock_state != EvidenceState.VALID.value:
        ledger = RequestLedger(client, declared, {})
        return {
            **base_report,
            "state": "EMIR_STATE_A",
            "mode": "SAFE_STOP_WAITING_FOR_PASTICUAN",
            "family": STOCK_SUMMARY_FAMILY,
            "classification": stock_state,
            "reason": stock_reason,
            "request_ledger": ledger.as_dict(),
        }, 0

    consume_ledger = RequestLedger(client, declared, {})
    stock_result = cache_only_consume(
        active_backend, STOCK_SPEC, day, DenyZapiTransport(consume_ledger)
    )
    if stock_result["classification"] != EvidenceState.VALID.value:
        return {
            **base_report,
            "state": "ERROR",
            "mode": "FAIL_CLOSED",
            "family": STOCK_SUMMARY_FAMILY,
            "evidence": stock_result,
            "request_ledger": consume_ledger.as_dict(),
        }, 1
    if foreign_state in {EvidenceState.ERROR.value, EvidenceState.STALE.value}:
        return {
            **base_report,
            "state": foreign_state,
            "reason": foreign_reason,
            "mode": "FAIL_CLOSED",
            "family": FOREIGN_FLOW_FAMILY,
            "stock_summary_cache": stock_result,
            "request_ledger": consume_ledger.as_dict(),
        }, 1
    if foreign_state == EvidenceState.VALID.value:
        foreign_result = cache_only_consume(
            active_backend, FOREIGN_SPEC, day, DenyZapiTransport(consume_ledger)
        )
        return {
            **base_report,
            "state": "EMIR_STATE_B",
            "mode": "CACHE_ONLY_ALREADY_COMPLETE",
            "family": FOREIGN_FLOW_FAMILY,
            "stock_summary_cache": stock_result,
            "evidence": foreign_result,
            "request_ledger": consume_ledger.as_dict(),
        }, 0

    ledger = RequestLedger(client, declared, {FOREIGN_FLOW_FAMILY: FOREIGN_FLOW_ATTEMPT_CAP})
    if credential_status["zapi_key"] != "CONFIGURED":
        return {
            **base_report,
            "state": "CREDENTIAL_MISSING",
            "mode": "FAIL_CLOSED",
            "family": FOREIGN_FLOW_FAMILY,
            "stock_summary_cache": stock_result,
            "request_ledger": ledger.as_dict(),
        }, 2
    transport = BoundedZapiTransport(os.environ["ZAPI_KEY"], ledger, session=transport_session)
    foreign_result = produce_family(
        active_backend,
        FOREIGN_SPEC,
        day,
        client,
        lambda: fetch_foreign_flow(transport, day),
    )
    ok = foreign_result["classification"] == EvidenceState.VALID.value
    return {
        **base_report,
        "state": "EMIR_STATE_B",
        "mode": (("CACHE_CONSUMER_THEN_LIVE_PRODUCER" if foreign_result["provider_called"] else "CACHE_ONLY_ALREADY_COMPLETE") if ok else "ERROR"),
        "family": FOREIGN_FLOW_FAMILY,
        "stock_summary_cache": stock_result,
        "evidence": foreign_result,
        "request_ledger": ledger.as_dict(),
    }, 0 if ok else 1


def verify_delta_allowlist(
    baseline: str,
    *,
    repo_root: Path,
    runner: Any = subprocess.run,
) -> tuple[str, ...]:
    baseline_sha = str(baseline).strip()
    if len(baseline_sha) != 40 or any(ch not in "0123456789abcdef" for ch in baseline_sha.lower()):
        raise GateFailure("INVALID_BASELINE_SHA")
    ancestor = runner(
        ["git", "merge-base", "--is-ancestor", baseline_sha, "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if ancestor.returncode != 0:
        raise GateFailure("BASELINE_NOT_ANCESTOR")
    diff = runner(
        ["git", "diff", "--name-only", f"{baseline_sha}...HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if diff.returncode != 0:
        raise GateFailure("DELTA_INSPECTION_FAILED")
    changed = tuple(sorted(line.strip() for line in diff.stdout.splitlines() if line.strip()))
    if not changed or not set(changed).issubset(ALLOWED_DELTA_PATHS):
        raise GateFailure("DELTA_ALLOWLIST_VIOLATION")
    return changed


def _emit(report: Mapping[str, Any]) -> None:
    print("PHASE56_LEDGER=" + json.dumps(report, sort_keys=True, separators=(",", ":")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    allowlist = subparsers.add_parser("allowlist")
    allowlist.add_argument("--baseline", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--client", required=True, choices=("PASTICUAN", "EMIR"))
    run.add_argument("--validation-family", required=True, choices=("zapi", "participant"))
    run.add_argument("--declared-cumulative-attempts", required=True)
    run.add_argument("--target-date", default=PARTICIPANT_TARGET_DATE.isoformat())
    args = parser.parse_args(argv)
    try:
        if args.command == "allowlist":
            changed = verify_delta_allowlist(
                args.baseline, repo_root=Path(__file__).resolve().parents[1]
            )
            _emit({"version": VERSION, "state": "DELTA_ALLOWLIST_VALID", "changed_files": list(changed)})
            return 0
        if args.validation_family == "participant":
            try:
                participant_day = date.fromisoformat(args.target_date)
            except ValueError as exc:
                raise GateFailure("WRONG_PERIOD") from exc
            report, exit_code = execute_participant_gate(
                args.client,
                target_date=participant_day,
                declared_cumulative_zapi=parse_declared_cumulative(
                    args.declared_cumulative_attempts
                ),
            )
        else:
            report, exit_code = execute_gate(
                args.client,
                parse_declared_cumulative(args.declared_cumulative_attempts),
            )
        _emit(report)
        return exit_code
    except GateFailure as exc:
        _emit(
            {
                "version": VERSION,
                "state": "ERROR",
                "reason": exc.code,
                "provider_attempts": 0,
                "official_downloads": 0,
                "schema_mutations": 0,
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
