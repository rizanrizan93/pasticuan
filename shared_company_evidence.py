from __future__ import annotations

"""Scanner-neutral slow-moving IDX company and reference evidence."""

from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
import os
import re
from typing import Any, Iterable, Mapping

import requests

from shared_evidence_hub import (
    EvidenceKey,
    HubConfig,
    MissingReason,
    SharedEvidenceCoordinator,
    SupabaseEvidenceBackend,
)


COMPANIES_URL = "https://api.zpi.web.id/v1/finance:idx/companies"
SECURITIES_URL = "https://api.zpi.web.id/v1/finance:idx/securities"
PROFILE_URL = "https://api.zpi.web.id/v1/finance:idx/company-profile"
REFERENCE_URL = "https://api.zpi.web.id/v1/finance:idx/reference"
COMPANY_TABLE = "evidence_companies"
REFERENCE_TABLE = "evidence_reference_values"
BULK_LENGTH = 1000
DIRECTORY_TTL = timedelta(days=30)
PROFILE_TTL = timedelta(days=30)
REFERENCE_TTL = timedelta(days=90)
REFERENCE_SETS = frozenset({"sectors", "boards", "market-time"})


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _secret(name: str) -> str:
    value = _clean(os.getenv(name, ""))
    if value:
        return value
    try:
        import streamlit as st

        return _clean(st.secrets.get(name, ""))
    except Exception:
        return ""


def _ticker(value: Any) -> str | None:
    text = _clean(value).upper().removesuffix(".JK")
    return text if re.fullmatch(r"[A-Z][A-Z0-9]{3,5}", text) else None


def _date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _clean(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _number(value: Any) -> int | float | None:
    if value is None or isinstance(value, bool) or not _clean(value):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return int(parsed) if parsed.is_integer() else parsed


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode("utf-8")
    ).hexdigest()


def _month(value: date) -> date:
    return value.replace(day=1)


def _unwrap(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise RuntimeError(MissingReason.PARSE_FAILURE.value)
    root: Mapping[str, Any] = payload
    for key in ("content", "data"):
        nested = root.get(key)
        if isinstance(nested, Mapping) and any(field in nested for field in ("dataset", "provider", "items", "data")):
            root = nested
    return root


def _offset_rows(payload: Any, *, dataset: str) -> tuple[list[Mapping[str, Any]], bool]:
    root = _unwrap(payload)
    if _clean(root.get("dataset")).lower() != dataset or _clean(root.get("provider")).lower() != "idx":
        raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
    values = root.get("data") if isinstance(root.get("data"), list) else root.get("items")
    if not isinstance(values, list) or any(not isinstance(item, Mapping) for item in values):
        raise RuntimeError(MissingReason.PARSE_FAILURE.value)
    start = int(root.get("start") or 0)
    length = int(root.get("length") or BULK_LENGTH)
    total = int(root.get("recordsFiltered") or root.get("recordsTotal") or root.get("total") or len(values))
    return list(values), start + max(1, length) < total


def _profile_root(payload: Any) -> Mapping[str, Any]:
    root = _unwrap(payload)
    if _clean(root.get("dataset")).lower() != "company-profile" or _clean(root.get("provider")).lower() != "idx":
        raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
    return root


def _change_state(payload_hash: str, previous: Mapping[str, Any] | None) -> str:
    if not previous:
        return "NEW"
    return "UNCHANGED" if _clean(previous.get("payload_hash")) == payload_hash else "CHANGED"


def normalize_company_directory(
    company_items: Iterable[Mapping[str, Any]],
    security_items: Iterable[Mapping[str, Any]],
    *,
    source_period: date,
    observed_on: date,
    previous: Iterable[Mapping[str, Any]] = (),
    fetched_at: datetime | None = None,
) -> list[dict[str, Any]]:
    companies: dict[str, Mapping[str, Any]] = {}
    securities: dict[str, Mapping[str, Any]] = {}
    for item in company_items:
        ticker = _ticker(item.get("KodeEmiten") or item.get("code"))
        if ticker:
            if ticker in companies and _canonical_hash(companies[ticker]) != _canonical_hash(item):
                raise RuntimeError(MissingReason.PARSE_FAILURE.value)
            companies[ticker] = item
    for item in security_items:
        ticker = _ticker(item.get("Code") or item.get("code"))
        if ticker:
            if ticker in securities and _canonical_hash(securities[ticker]) != _canonical_hash(item):
                raise RuntimeError(MissingReason.PARSE_FAILURE.value)
            securities[ticker] = item
    prior = {_ticker(row.get("ticker")): row for row in previous if _ticker(row.get("ticker"))}
    stamp = (fetched_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    rows: list[dict[str, Any]] = []
    for ticker in sorted(set(companies) | set(securities)):
        company = companies.get(ticker, {})
        security = securities.get(ticker, {})
        listing_date = _date(security.get("ListingDate") or company.get("TanggalPencatatan"))
        facts = {
            "company_name": _clean(company.get("NamaEmiten") or security.get("Name")) or None,
            "sector": _clean(company.get("Sektor")) or None,
            "sub_sector": _clean(company.get("SubSektor")) or None,
            "industry": _clean(company.get("Industri")) or None,
            "sub_industry": _clean(company.get("SubIndustri")) or None,
            "listing_board": _clean(security.get("ListingBoard") or company.get("PapanPencatatan")) or None,
            "listing_date": listing_date.isoformat() if listing_date else None,
            "listed_shares": _number(security.get("Shares")),
            "main_business": _clean(company.get("KegiatanUsahaUtama")) or None,
            "profile": {
                "registrar": _clean(company.get("BAE")) or None,
                "website": _clean(company.get("Website")) or None,
                "address": _clean(company.get("Alamat")) or None,
                "security_flags": {
                    "stock": bool(company.get("EfekEmiten_Saham")),
                    "bond": bool(company.get("EfekEmiten_Obligasi")),
                    "etf": bool(company.get("EfekEmiten_ETF")),
                    "eba": bool(company.get("EfekEmiten_EBA")),
                    "spei": bool(company.get("EfekEmiten_SPEI")),
                },
                "source_urls": [COMPANIES_URL, SECURITIES_URL],
            },
        }
        payload_hash = _canonical_hash({"ticker": ticker, **facts})
        rows.append({
            "provider": "IDX_COMPANY_DIRECTORY_VIA_ZAPI",
            "ticker": ticker,
            **facts,
            "profile_kind": "DIRECTORY",
            "source_period": _month(source_period).isoformat(),
            "observed_on": observed_on.isoformat(),
            "change_state": _change_state(payload_hash, prior.get(ticker)),
            "source_url": COMPANIES_URL,
            "payload_hash": payload_hash,
            "source_verified": True,
            "validation_state": "VALID",
            "fetched_at": stamp,
        })
    return rows


def _relationship_rows(root: Mapping[str, Any], key: str, fields: tuple[str, ...]) -> list[dict[str, Any]]:
    values = root.get(key) or []
    if not isinstance(values, list) or any(not isinstance(item, Mapping) for item in values):
        raise RuntimeError(MissingReason.PARSE_FAILURE.value)
    rows = [{field: item.get(field) for field in fields if item.get(field) not in (None, "")} for item in values]
    rows = [row for row in rows if row]
    return sorted(rows, key=lambda row: _canonical_hash(row))


def normalize_company_profile(
    payload: Any,
    *,
    ticker: str,
    source_period: date,
    observed_on: date,
    previous: Mapping[str, Any] | None = None,
    fetched_at: datetime | None = None,
) -> list[dict[str, Any]]:
    root = _profile_root(payload)
    expected = _ticker(ticker)
    actual = _ticker(root.get("code"))
    if expected is None or actual != expected:
        raise RuntimeError(MissingReason.ISSUER_MISMATCH.value)
    listing_date = _date(root.get("listingDate"))
    relationships = {
        "directors": _relationship_rows(root, "directors", ("name", "title")),
        "commissioners": _relationship_rows(root, "commissioners", ("name", "title")),
        "audit_committee": _relationship_rows(root, "auditCommittee", ("name", "title")),
        "corporate_secretary": _relationship_rows(root, "corporateSecretary", ("name",)),
        "shareholders": _relationship_rows(root, "shareholders", ("name", "shares", "sharePct", "category")),
        "subsidiaries": _relationship_rows(
            root, "subsidiaries",
            ("name", "business", "location", "ownershipPct", "operatingStatus", "commercialYear", "totalAssets", "unit", "currency"),
        ),
    }
    facts = {
        "company_name": _clean(root.get("name")) or None,
        "sector": _clean(root.get("sector")) or None,
        "sub_sector": _clean(root.get("subSector")) or None,
        "industry": _clean(root.get("industry")) or None,
        "sub_industry": _clean(root.get("subIndustry")) or None,
        "listing_board": _clean(root.get("listingBoard")) or None,
        "listing_date": listing_date.isoformat() if listing_date else None,
        "listed_shares": None,
        "main_business": _clean(root.get("mainBusiness")) or None,
        "profile": {
            "website": _clean(root.get("website")) or None,
            "address": _clean(root.get("address")) or None,
            "relationships": relationships,
        },
    }
    payload_hash = _canonical_hash({"ticker": actual, **facts})
    stamp = (fetched_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    return [{
        "provider": "IDX_COMPANY_PROFILE_VIA_ZAPI",
        "ticker": actual,
        **facts,
        "profile_kind": "DETAILED_PROFILE",
        "source_period": _month(source_period).isoformat(),
        "observed_on": observed_on.isoformat(),
        "change_state": _change_state(payload_hash, previous),
        "source_url": PROFILE_URL,
        "payload_hash": payload_hash,
        "source_verified": True,
        "validation_state": "VALID",
        "fetched_at": stamp,
    }]


def normalize_reference_values(
    payload: Any,
    *,
    set_name: str,
    source_period: date,
    observed_on: date,
    fetched_at: datetime | None = None,
) -> list[dict[str, Any]]:
    if set_name not in REFERENCE_SETS:
        raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
    root = _unwrap(payload)
    if (
        _clean(root.get("dataset")).lower() != "reference"
        or _clean(root.get("provider")).lower() != "idx"
        or _clean(root.get("set")) != set_name
    ):
        raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
    values = root.get("items")
    if not isinstance(values, list) or any(not isinstance(value, (str, int, float)) or isinstance(value, bool) for value in values):
        raise RuntimeError(MissingReason.PARSE_FAILURE.value)
    stamp = (fetched_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    rows: dict[str, dict[str, Any]] = {}
    for value in values:
        label = _clean(value)
        if not label:
            continue
        value_key = hashlib.sha256(label.casefold().encode("utf-8")).hexdigest()
        row = {
            "provider": "IDX_REFERENCE_VIA_ZAPI",
            "set_name": set_name,
            "value_key": value_key,
            "label": label,
            "source_period": _month(source_period).isoformat(),
            "observed_on": observed_on.isoformat(),
            "source_url": REFERENCE_URL,
            "payload_hash": _canonical_hash({"set": set_name, "label": label}),
            "source_verified": True,
            "validation_state": "VALID",
            "fetched_at": stamp,
        }
        current = rows.get(value_key)
        if current is not None and current["label"] != label:
            raise RuntimeError(MissingReason.PARSE_FAILURE.value)
        rows[value_key] = row
    return sorted(rows.values(), key=lambda row: row["label"].casefold())


def validate_company_rows(rows: Iterable[Mapping[str, Any]], *, provider: str) -> tuple[bool, str]:
    records = [dict(row) for row in rows]
    if not records:
        return False, MissingReason.NO_REPORT.value
    tickers: set[str] = set()
    for row in records:
        ticker = _ticker(row.get("ticker"))
        if ticker is None or ticker in tickers or row.get("provider") != provider:
            return False, MissingReason.PARSE_FAILURE.value
        tickers.add(ticker)
        if not row.get("source_verified") or row.get("validation_state") != "VALID":
            return False, MissingReason.CONTEXT_REJECTED.value
        if row.get("change_state") not in {"NEW", "CHANGED", "UNCHANGED"}:
            return False, MissingReason.CONTEXT_REJECTED.value
        if not isinstance(row.get("profile"), Mapping) or not _clean(row.get("payload_hash")):
            return False, MissingReason.PARSE_FAILURE.value
        shares = row.get("listed_shares")
        if shares is not None and (not isinstance(shares, (int, float)) or isinstance(shares, bool) or shares < 0):
            return False, MissingReason.CONTEXT_REJECTED.value
    return True, "VALID"


def validate_reference_rows(rows: Iterable[Mapping[str, Any]], *, set_name: str) -> tuple[bool, str]:
    records = [dict(row) for row in rows]
    if not records:
        return False, MissingReason.NO_REPORT.value
    identities: set[str] = set()
    for row in records:
        identity = _clean(row.get("value_key"))
        if not identity or identity in identities or row.get("set_name") != set_name:
            return False, MissingReason.PARSE_FAILURE.value
        identities.add(identity)
        if row.get("provider") != "IDX_REFERENCE_VIA_ZAPI" or not row.get("source_verified"):
            return False, MissingReason.CONTEXT_REJECTED.value
    return True, "VALID"


class SharedCompanyEvidence:
    def __init__(
        self,
        client_id: str,
        *,
        backend: Any | None = None,
        coordinator: SharedEvidenceCoordinator | None = None,
        session: Any | None = None,
        api_key: str | None = None,
    ):
        self.client_id = _clean(client_id).upper() or "UNKNOWN"
        self.config = HubConfig.from_environment(client_id=self.client_id)
        self.backend = backend or (SupabaseEvidenceBackend(self.config) if self.config.ready else None)
        self.coordinator = coordinator or (
            SharedEvidenceCoordinator(self.backend, client_id=self.client_id) if self.backend is not None else None
        )
        self.session = session or requests.Session()
        self.api_key = _secret("ZAPI_KEY") if api_key is None else _clean(api_key)

    @property
    def ready(self) -> bool:
        return self.backend is not None and self.coordinator is not None

    def _request(self, url: str, params: Mapping[str, Any]) -> Any:
        if not self.api_key:
            raise RuntimeError(MissingReason.ENVIRONMENT_BLOCKED.value)
        try:
            response = self.session.request(
                "GET", url, params=dict(params),
                headers={"Accept": "application/json", "x-api-key": self.api_key}, timeout=30,
            )
        except requests.Timeout as exc:
            raise RuntimeError(MissingReason.TIMEOUT.value) from exc
        except requests.ConnectionError as exc:
            raise RuntimeError(MissingReason.CONNECTION_ERROR.value) from exc
        status = int(getattr(response, "status_code", 0) or 0)
        if not 200 <= status < 300:
            raise RuntimeError(f"HTTP_{status}")
        if not getattr(response, "content", b""):
            raise RuntimeError(MissingReason.EMPTY_RESPONSE.value)
        try:
            return response.json()
        except (TypeError, ValueError) as exc:
            raise RuntimeError(MissingReason.PARSE_FAILURE.value) from exc

    def get_directory(self, observed_on: date, *, max_pages: int = 3) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not self.ready:
            return [], {"state": MissingReason.ENVIRONMENT_BLOCKED.value, "api_calls": 0}
        source_period = _month(observed_on)
        provider = "IDX_COMPANY_DIRECTORY_VIA_ZAPI"
        meta: dict[str, Any] = {"api_calls": 0, "pages": 0, "ttl_days": DIRECTORY_TTL.days}

        def read_current() -> list[dict[str, Any]]:
            return self.backend.read_rows(COMPANY_TABLE, {"provider": provider, "validation_state": "VALID"}, limit=50000)

        def fetch() -> list[dict[str, Any]]:
            previous = read_current()
            all_rows: dict[str, list[Mapping[str, Any]]] = {"listed-companies": [], "securities": []}
            for dataset, url in (("listed-companies", COMPANIES_URL), ("securities", SECURITIES_URL)):
                completed = False
                for page in range(max(1, int(max_pages))):
                    payload = self._request(url, {"length": BULK_LENGTH, "start": page * BULK_LENGTH})
                    meta["api_calls"] += 1
                    meta["pages"] += 1
                    rows, has_more = _offset_rows(payload, dataset=dataset)
                    all_rows[dataset].extend(rows)
                    if not has_more or not rows:
                        completed = True
                        break
                if not completed:
                    raise RuntimeError(MissingReason.INSUFFICIENT_HISTORY.value)
            rows = normalize_company_directory(
                all_rows["listed-companies"], all_rows["securities"],
                source_period=source_period, observed_on=observed_on, previous=previous,
            )
            if not rows:
                raise RuntimeError(MissingReason.NO_REPORT.value)
            return rows

        result = self.coordinator.get_or_refresh(
            EvidenceKey("ZAPI", "COMPANIES", "IDX_ALL", source_period),
            read_current=read_current,
            fetch=fetch,
            persist=lambda rows: len(self.backend.upsert_rows(COMPANY_TABLE, rows, conflict=("provider", "ticker"))),
            validate=lambda rows: validate_company_rows(rows, provider=provider),
            max_age=DIRECTORY_TTL,
            minimum_rows=1,
            lease_seconds=300,
        )
        rows = [dict(row) for row in result.rows]
        return rows, {
            "state": result.reason, "rows": len(rows), "source_period": source_period.isoformat(),
            "cache_hit": result.cache_hit, "request_avoided": result.request_avoided,
            "lease_state": result.lease_state, **meta,
        }

    def get_profile(self, ticker: str, observed_on: date) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        normalized_ticker = _ticker(ticker)
        if normalized_ticker is None:
            return [], {"state": MissingReason.ISSUER_IDENTITY_MISSING.value, "api_calls": 0}
        if not self.ready:
            return [], {"state": MissingReason.ENVIRONMENT_BLOCKED.value, "api_calls": 0}
        source_period = _month(observed_on)
        provider = "IDX_COMPANY_PROFILE_VIA_ZAPI"
        meta: dict[str, Any] = {"api_calls": 0, "ttl_days": PROFILE_TTL.days}

        def read_current() -> list[dict[str, Any]]:
            return self.backend.read_rows(
                COMPANY_TABLE,
                {"provider": provider, "ticker": normalized_ticker, "validation_state": "VALID"},
                limit=1,
            )

        def fetch() -> list[dict[str, Any]]:
            payload = self._request(PROFILE_URL, {"code": normalized_ticker})
            meta["api_calls"] += 1
            current = read_current()
            return normalize_company_profile(
                payload, ticker=normalized_ticker, source_period=source_period,
                observed_on=observed_on, previous=current[0] if current else None,
            )

        result = self.coordinator.get_or_refresh(
            EvidenceKey("ZAPI", "COMPANY_PROFILE", normalized_ticker, source_period),
            read_current=read_current,
            fetch=fetch,
            persist=lambda rows: len(self.backend.upsert_rows(COMPANY_TABLE, rows, conflict=("provider", "ticker"))),
            validate=lambda rows: validate_company_rows(rows, provider=provider),
            max_age=PROFILE_TTL,
            minimum_rows=1,
            lease_seconds=300,
        )
        rows = [dict(row) for row in result.rows]
        return rows, {
            "state": result.reason, "ticker": normalized_ticker, "rows": len(rows),
            "cache_hit": result.cache_hit, "request_avoided": result.request_avoided,
            "lease_state": result.lease_state, **meta,
        }

    def get_reference(self, set_name: str, observed_on: date) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if set_name not in REFERENCE_SETS:
            return [], {"state": MissingReason.CONTEXT_REJECTED.value, "api_calls": 0}
        if not self.ready:
            return [], {"state": MissingReason.ENVIRONMENT_BLOCKED.value, "api_calls": 0}
        source_period = _month(observed_on)
        meta: dict[str, Any] = {"api_calls": 0, "ttl_days": REFERENCE_TTL.days}

        def read_current() -> list[dict[str, Any]]:
            return self.backend.read_rows(
                REFERENCE_TABLE,
                {"provider": "IDX_REFERENCE_VIA_ZAPI", "set_name": set_name, "validation_state": "VALID"},
                limit=5000,
            )

        def fetch() -> list[dict[str, Any]]:
            payload = self._request(REFERENCE_URL, {"set": set_name})
            meta["api_calls"] += 1
            rows = normalize_reference_values(
                payload, set_name=set_name, source_period=source_period, observed_on=observed_on
            )
            if not rows:
                raise RuntimeError(MissingReason.NO_REPORT.value)
            return rows

        result = self.coordinator.get_or_refresh(
            EvidenceKey("ZAPI", "REFERENCE", f"IDX_{set_name}", source_period),
            read_current=read_current,
            fetch=fetch,
            persist=lambda rows: len(self.backend.upsert_rows(
                REFERENCE_TABLE, rows, conflict=("provider", "set_name", "value_key")
            )),
            validate=lambda rows: validate_reference_rows(rows, set_name=set_name),
            max_age=REFERENCE_TTL,
            minimum_rows=1,
            lease_seconds=300,
        )
        rows = [dict(row) for row in result.rows]
        return rows, {
            "state": result.reason, "set_name": set_name, "rows": len(rows),
            "cache_hit": result.cache_hit, "request_avoided": result.request_avoided,
            "lease_state": result.lease_state, **meta,
        }


__all__ = [
    "BULK_LENGTH", "COMPANIES_URL", "DIRECTORY_TTL", "PROFILE_TTL", "PROFILE_URL",
    "REFERENCE_SETS", "REFERENCE_TTL", "REFERENCE_URL", "SECURITIES_URL",
    "SharedCompanyEvidence", "normalize_company_directory", "normalize_company_profile",
    "normalize_reference_values", "validate_company_rows", "validate_reference_rows",
]
