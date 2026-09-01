from __future__ import annotations

"""Scanner-neutral IDX exchange-member and market-wide broker evidence."""

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


BROKERS_URL = "https://api.zpi.web.id/v1/finance:idx/brokers"
BROKER_SUMMARY_URL = "https://api.zpi.web.id/v1/finance:idx/broker-summary"
MEMBER_TABLE = "evidence_brokers"
MARKET_TABLE = "evidence_broker_market_daily"
MEMBER_PROVIDER = "IDX_EXCHANGE_MEMBERS_VIA_ZAPI"
MARKET_PROVIDER = "IDX_BROKER_SUMMARY_VIA_ZAPI"
EVIDENCE_SCOPE = "MARKET_WIDE"
SUMMARY_LENGTH = 5000
MEMBER_TTL = timedelta(days=30)


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


def _broker_code(value: Any) -> str | None:
    text = _clean(value).upper()
    return text if re.fullmatch(r"[A-Z0-9]{2}", text) else None


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
    if value is None or isinstance(value, bool) or (isinstance(value, str) and not value.strip()):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return int(parsed) if parsed.is_integer() else parsed


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _month(value: date) -> date:
    return value.replace(day=1)


def _unwrap(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise RuntimeError(MissingReason.PARSE_FAILURE.value)
    root: Mapping[str, Any] = payload
    for key in ("content", "data"):
        nested = root.get(key)
        if isinstance(nested, Mapping) and any(
            field in nested for field in ("dataset", "provider", "items", "data", "code")
        ):
            root = nested
    return root


def _validated_non_negative(value: Any, *, integral: bool = False) -> int | float | None:
    parsed = _number(value)
    if value not in (None, "") and parsed is None:
        raise RuntimeError(MissingReason.PARSE_FAILURE.value)
    if parsed is not None and (parsed < 0 or (integral and not float(parsed).is_integer())):
        raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
    return int(parsed) if integral and parsed is not None else parsed


def _shareholders(value: Any) -> list[dict[str, Any]]:
    if value in (None, ""):
        return []
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise RuntimeError(MissingReason.PARSE_FAILURE.value)
    rows: list[dict[str, Any]] = []
    for item in value:
        share_pct = _number(item.get("sharePct"))
        if share_pct is not None and not 0 <= share_pct <= 100:
            raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
        ownership = _clean(item.get("ownership")).lower() or None
        if ownership not in {None, "asing", "lokal"}:
            raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
        row = {
            "name": _clean(item.get("name")) or None,
            "type": _clean(item.get("type")) or None,
            "country": _clean(item.get("country")) or None,
            "share_percentage": share_pct,
            "reported_ownership": ownership,
        }
        if row["name"]:
            rows.append(row)
    return sorted(rows, key=_canonical_hash)


def _member_items(payload: Any) -> list[Mapping[str, Any]]:
    root = _unwrap(payload)
    if _clean(root.get("dataset")).lower() != "brokers" or _clean(root.get("provider")).lower() != "idx":
        raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
    if _broker_code(root.get("code")):
        return [root]
    values = root.get("items") if isinstance(root.get("items"), list) else root.get("data")
    if not isinstance(values, list) or any(not isinstance(item, Mapping) for item in values):
        raise RuntimeError(MissingReason.PARSE_FAILURE.value)
    return list(values)


def normalize_exchange_members(
    payload: Any,
    *,
    observed_on: date,
    previous: Iterable[Mapping[str, Any]] = (),
    fetched_at: datetime | None = None,
) -> list[dict[str, Any]]:
    prior = {
        code: row
        for row in previous
        if (code := _broker_code(row.get("broker_code"))) is not None
    }
    stamp = (fetched_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    rows: dict[str, dict[str, Any]] = {}
    for item in _member_items(payload):
        code = _broker_code(item.get("code"))
        if code is None:
            raise RuntimeError(MissingReason.PARSE_FAILURE.value)
        foreign_pct = _number(item.get("foreignOwnershipPct"))
        if foreign_pct is not None and not 0 <= foreign_pct <= 100:
            raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
        facts = {
            "broker_name": _clean(item.get("name")) or None,
            "member_status": _clean(item.get("memberStatus")) or None,
            "ownership_category": _clean(item.get("category")) or None,
            "foreign_ownership_percentage": foreign_pct,
            "paid_up_capital": _validated_non_negative(item.get("paidUpCapital")),
            "mkbd": _validated_non_negative(item.get("mkbd")),
            "branch_count": _validated_non_negative(item.get("branchCount"), integral=True),
            "profile": {
                "license": _clean(item.get("license")) or None,
                "website": _clean(item.get("website")) or None,
                "city": _clean(item.get("city")) or None,
                "logo": _clean(item.get("logo")) or None,
                "shareholders": _shareholders(item.get("shareholders")),
            },
        }
        payload_hash = _canonical_hash({"broker_code": code, **facts})
        old_hash = _clean(prior.get(code, {}).get("payload_hash"))
        change_state = "NEW" if not old_hash else ("UNCHANGED" if old_hash == payload_hash else "CHANGED")
        row = {
            "provider": MEMBER_PROVIDER,
            "broker_code": code,
            **facts,
            "profile_kind": "EXCHANGE_MEMBER",
            "evidence_scope": EVIDENCE_SCOPE,
            "source_period": _month(observed_on).isoformat(),
            "observed_on": observed_on.isoformat(),
            "change_state": change_state,
            "source_url": BROKERS_URL,
            "payload_hash": payload_hash,
            "source_verified": True,
            "validation_state": "VALID",
            "fetched_at": stamp,
        }
        current = rows.get(code)
        if current is not None and current["payload_hash"] != payload_hash:
            raise RuntimeError(MissingReason.PARSE_FAILURE.value)
        rows[code] = row
    return [rows[code] for code in sorted(rows)]


def _summary_page(payload: Any) -> tuple[list[Mapping[str, Any]], bool]:
    root = _unwrap(payload)
    if _clean(root.get("dataset")).lower() != "broker-summary" or _clean(root.get("provider")).lower() != "idx":
        raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
    values = root.get("data")
    if not isinstance(values, list) or any(not isinstance(item, Mapping) for item in values):
        raise RuntimeError(MissingReason.PARSE_FAILURE.value)
    start = _validated_non_negative(root.get("start") or 0, integral=True) or 0
    length = _validated_non_negative(root.get("length") or SUMMARY_LENGTH, integral=True) or SUMMARY_LENGTH
    total = _validated_non_negative(
        root.get("recordsFiltered") or root.get("recordsTotal") or len(values), integral=True
    )
    if total is None:
        raise RuntimeError(MissingReason.PARSE_FAILURE.value)
    return list(values), start + max(1, length) < total


def normalize_market_summary(
    items: Iterable[Mapping[str, Any]],
    *,
    activity_date: date,
    fetched_at: datetime | None = None,
) -> list[dict[str, Any]]:
    stamp = (fetched_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    rows: dict[str, dict[str, Any]] = {}
    for item in items:
        code = _broker_code(item.get("IDFirm"))
        row_date = _date(item.get("Date"))
        if code is None or row_date is None:
            raise RuntimeError(MissingReason.PARSE_FAILURE.value)
        if row_date != activity_date:
            raise RuntimeError(MissingReason.WRONG_PERIOD.value)
        facts = {
            "broker_name": _clean(item.get("FirmName")) or None,
            "traded_value": _validated_non_negative(item.get("Value")),
            "traded_volume": _validated_non_negative(item.get("Volume")),
            "frequency": _validated_non_negative(item.get("Frequency"), integral=True),
            "source_event_id": _clean(item.get("IDBrokerSummary")) or None,
        }
        payload_hash = _canonical_hash({"activity_date": activity_date.isoformat(), "broker_code": code, **facts})
        row = {
            "provider": MARKET_PROVIDER,
            "activity_date": activity_date.isoformat(),
            "broker_code": code,
            **facts,
            "evidence_scope": EVIDENCE_SCOPE,
            "source_url": BROKER_SUMMARY_URL,
            "payload_hash": payload_hash,
            "source_verified": True,
            "validation_state": "VALID",
            "fetched_at": stamp,
        }
        current = rows.get(code)
        if current is not None and current["payload_hash"] != payload_hash:
            raise RuntimeError(MissingReason.PARSE_FAILURE.value)
        rows[code] = row
    return [rows[code] for code in sorted(rows)]


def validate_member_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[bool, str]:
    records = [dict(row) for row in rows]
    if not records:
        return False, MissingReason.NO_REPORT.value
    codes: set[str] = set()
    for row in records:
        code = _broker_code(row.get("broker_code"))
        if code is None or code in codes:
            return False, MissingReason.PARSE_FAILURE.value
        codes.add(code)
        if row.get("provider") != MEMBER_PROVIDER or row.get("evidence_scope") != EVIDENCE_SCOPE:
            return False, MissingReason.CONTEXT_REJECTED.value
        if row.get("profile_kind") != "EXCHANGE_MEMBER" or row.get("change_state") not in {"NEW", "CHANGED", "UNCHANGED"}:
            return False, MissingReason.CONTEXT_REJECTED.value
        if not row.get("source_verified") or row.get("validation_state") != "VALID":
            return False, MissingReason.CONTEXT_REJECTED.value
        pct = row.get("foreign_ownership_percentage")
        if pct is not None and (not isinstance(pct, (int, float)) or isinstance(pct, bool) or not 0 <= pct <= 100):
            return False, MissingReason.CONTEXT_REJECTED.value
        if not isinstance(row.get("profile"), Mapping) or not _clean(row.get("payload_hash")):
            return False, MissingReason.PARSE_FAILURE.value
    return True, "VALID"


def validate_market_rows(rows: Iterable[Mapping[str, Any]], *, activity_date: date) -> tuple[bool, str]:
    records = [dict(row) for row in rows]
    if not records:
        return False, MissingReason.PROVIDER_NO_DATA.value
    codes: set[str] = set()
    for row in records:
        code = _broker_code(row.get("broker_code"))
        if code is None or code in codes:
            return False, MissingReason.PARSE_FAILURE.value
        codes.add(code)
        if row.get("provider") != MARKET_PROVIDER or row.get("evidence_scope") != EVIDENCE_SCOPE:
            return False, MissingReason.CONTEXT_REJECTED.value
        if _date(row.get("activity_date")) != activity_date or not row.get("source_verified"):
            return False, MissingReason.WRONG_PERIOD.value
        if any(field in row for field in ("ticker", "stock_code", "issuer")):
            return False, MissingReason.CONTEXT_REJECTED.value
        for field in ("traded_value", "traded_volume", "frequency"):
            value = row.get(field)
            if value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0):
                return False, MissingReason.CONTEXT_REJECTED.value
    return True, "VALID"


class SharedBrokerReferenceEvidence:
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
                headers={"Accept": "application/json", "x-api-key": self.api_key}, timeout=45,
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

    def get_members(self, observed_on: date) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not self.ready:
            return [], {"state": MissingReason.ENVIRONMENT_BLOCKED.value, "api_calls": 0}
        source_period = _month(observed_on)
        meta: dict[str, Any] = {"api_calls": 0, "ttl_days": MEMBER_TTL.days}

        def read_current() -> list[dict[str, Any]]:
            return self.backend.read_rows(
                MEMBER_TABLE, {"provider": MEMBER_PROVIDER, "validation_state": "VALID"}, limit=5000
            )

        def fetch() -> list[dict[str, Any]]:
            previous = read_current()
            payload = self._request(BROKERS_URL, {"view": "full"})
            meta["api_calls"] += 1
            rows = normalize_exchange_members(payload, observed_on=observed_on, previous=previous)
            if not rows:
                raise RuntimeError(MissingReason.NO_REPORT.value)
            return rows

        result = self.coordinator.get_or_refresh(
            EvidenceKey("ZAPI", "EXCHANGE_MEMBERS", "IDX_ALL", source_period),
            read_current=read_current,
            fetch=fetch,
            persist=lambda rows: len(self.backend.upsert_rows(
                MEMBER_TABLE, rows, conflict=("provider", "broker_code")
            )),
            validate=validate_member_rows,
            max_age=MEMBER_TTL,
            minimum_rows=1,
            lease_seconds=600,
        )
        rows = [dict(row) for row in result.rows]
        return rows, {
            "state": result.reason, "rows": len(rows), "source_period": source_period.isoformat(),
            "cache_hit": result.cache_hit, "request_avoided": result.request_avoided,
            "lease_state": result.lease_state, **meta,
        }

    def get_market_summary(
        self, activity_date: date, *, max_pages: int = 2
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not self.ready:
            return [], {"state": MissingReason.ENVIRONMENT_BLOCKED.value, "api_calls": 0}
        meta: dict[str, Any] = {"api_calls": 0, "pages": 0}

        def read_current() -> list[dict[str, Any]]:
            return self.backend.read_rows(
                MARKET_TABLE,
                {"provider": MARKET_PROVIDER, "activity_date": activity_date.isoformat(), "validation_state": "VALID"},
                limit=5000,
            )

        def fetch() -> list[dict[str, Any]]:
            items: list[Mapping[str, Any]] = []
            completed = False
            for page in range(max(1, int(max_pages))):
                payload = self._request(
                    BROKER_SUMMARY_URL,
                    {"length": SUMMARY_LENGTH, "start": page * SUMMARY_LENGTH, "date": activity_date.isoformat()},
                )
                meta["api_calls"] += 1
                meta["pages"] += 1
                page_rows, has_more = _summary_page(payload)
                items.extend(page_rows)
                if not has_more or not page_rows:
                    completed = True
                    break
            if not completed:
                raise RuntimeError(MissingReason.INSUFFICIENT_HISTORY.value)
            return normalize_market_summary(items, activity_date=activity_date)

        result = self.coordinator.get_or_refresh(
            EvidenceKey("ZAPI", "BROKER_SUMMARY", "IDX_ALL", activity_date),
            read_current=read_current,
            fetch=fetch,
            persist=lambda rows: len(self.backend.upsert_rows(
                MARKET_TABLE, rows, conflict=("provider", "activity_date", "broker_code")
            )),
            validate=lambda rows: validate_market_rows(rows, activity_date=activity_date),
            minimum_rows=1,
            lease_seconds=300,
        )
        rows = [dict(row) for row in result.rows]
        return rows, {
            "state": result.reason, "rows": len(rows), "activity_date": activity_date.isoformat(),
            "cache_hit": result.cache_hit, "request_avoided": result.request_avoided,
            "lease_state": result.lease_state, **meta,
        }


__all__ = [
    "BROKERS_URL", "BROKER_SUMMARY_URL", "EVIDENCE_SCOPE", "MARKET_PROVIDER",
    "MEMBER_PROVIDER", "MEMBER_TTL", "SUMMARY_LENGTH", "SharedBrokerReferenceEvidence",
    "normalize_exchange_members", "normalize_market_summary", "validate_market_rows",
    "validate_member_rows",
]
