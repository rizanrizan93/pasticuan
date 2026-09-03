from __future__ import annotations

"""Structured issuer-shareholder evidence for Phase 5.6.

Priority:
1. IDX company-profile via ZAPI (issuer/exchange profile facts).
2. Pluang company-profile via ZAPI only when IDX profile exposes no quantitative
   shareholder rows.

This family is deliberately separate from KSEI >1%/>5% workbook evidence.
"""

from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from typing import Any, Iterable, Mapping

import requests

from shared_evidence_hub import HubConfig, MissingReason, SupabaseEvidenceBackend


TABLE = "evidence_shareholder_profiles"
PLUANG_PROFILE_URL = "https://api.zpi.web.id/v1/finance:pluang/company-profile"
REQUEST_TIMEOUT_SECONDS = 20


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _ticker(value: Any) -> str:
    text = _clean(value).upper()
    return text[:-3] if text.endswith(".JK") else text


def _secret(name: str) -> str:
    value = _clean(os.getenv(name, ""))
    if value:
        return value
    try:
        import streamlit as st

        return _clean(st.secrets.get(name, ""))
    except Exception:
        return ""


def _number(value: Any) -> float | None:
    try:
        number = float(str(value).replace("%", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def normalize_idx_company_profile_row(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    ticker = _ticker(row.get("ticker"))
    profile = row.get("profile") if isinstance(row.get("profile"), Mapping) else {}
    relationships = profile.get("relationships") if isinstance(profile.get("relationships"), Mapping) else {}
    shareholders = relationships.get("shareholders") if isinstance(relationships.get("shareholders"), list) else []
    source_period = _clean(row.get("source_period"))
    observed_on = _clean(row.get("observed_on"))
    source_hash = _clean(row.get("payload_hash"))
    source_url = _clean(row.get("source_url"))
    fetched_at = _clean(row.get("fetched_at")) or datetime.now(timezone.utc).isoformat()
    if not ticker or not source_period or not observed_on or not source_hash or not source_url:
        return []
    output: list[dict[str, Any]] = []
    for item in shareholders:
        if not isinstance(item, Mapping):
            continue
        name = _clean(item.get("name"))
        shares = _number(item.get("shares"))
        pct = _number(item.get("sharePct"))
        category = _clean(item.get("category")) or None
        if not name or (shares is None and pct is None):
            continue
        identity = hashlib.sha256(f"{name.casefold()}|{(category or '').casefold()}".encode("utf-8")).hexdigest()
        output.append({
            "provider": "IDX_COMPANY_PROFILE_VIA_ZAPI",
            "ticker": ticker,
            "source_period": source_period,
            "observed_on": observed_on,
            "holder_identity_hash": identity,
            "holder_name": name,
            "shares_held": shares,
            "ownership_percentage": pct,
            "holder_category": category,
            "source_profile_hash": source_hash,
            "source_url": source_url,
            "source_verified": bool(row.get("source_verified")),
            "validation_state": "VALID",
            "fetched_at": fetched_at,
        })
    return output


def normalize_pluang_profile(payload: Mapping[str, Any], *, observed_on: date) -> list[dict[str, Any]]:
    ticker = _ticker(payload.get("code"))
    if not ticker or _clean(payload.get("source")).lower() not in {"", "pluang"}:
        raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
    shareholders = payload.get("shareholders") if isinstance(payload.get("shareholders"), list) else []
    source_hash = _hash(payload)
    source_period = observed_on.replace(day=1).isoformat()
    fetched_at = datetime.now(timezone.utc).isoformat()
    output: list[dict[str, Any]] = []
    for item in shareholders:
        if not isinstance(item, Mapping):
            continue
        name = _clean(item.get("name"))
        pct = _number(item.get("share"))
        if not name or pct is None:
            continue
        identity = hashlib.sha256(name.casefold().encode("utf-8")).hexdigest()
        output.append({
            "provider": "PLUANG_COMPANY_PROFILE_VIA_ZAPI",
            "ticker": ticker,
            "source_period": source_period,
            "observed_on": observed_on.isoformat(),
            "holder_identity_hash": identity,
            "holder_name": name,
            "shares_held": None,
            "ownership_percentage": pct,
            "holder_category": None,
            "source_profile_hash": source_hash,
            "source_url": PLUANG_PROFILE_URL,
            "source_verified": True,
            "validation_state": "VALID",
            "fetched_at": fetched_at,
        })
    return output


def validate_shareholder_rows(rows: Iterable[Mapping[str, Any]]) -> tuple[bool, str]:
    records = [dict(row) for row in rows]
    if not records:
        return False, MissingReason.EMPTY_RESPONSE.value
    for row in records:
        if not _ticker(row.get("ticker")) or not _clean(row.get("holder_name")):
            return False, MissingReason.PARSE_FAILURE.value
        if not _clean(row.get("holder_identity_hash")) or not _clean(row.get("source_profile_hash")):
            return False, MissingReason.PARSE_FAILURE.value
        shares = _number(row.get("shares_held"))
        pct = _number(row.get("ownership_percentage"))
        if shares is None and pct is None:
            return False, MissingReason.PARSE_FAILURE.value
        if shares is not None and shares < 0:
            return False, MissingReason.CONTEXT_REJECTED.value
        if pct is not None and not 0 <= pct <= 100:
            return False, MissingReason.CONTEXT_REJECTED.value
        if not bool(row.get("source_verified")):
            return False, MissingReason.CONTEXT_REJECTED.value
    return True, "VALID"


class SharedStructuredOwnershipEvidence:
    def __init__(
        self,
        client_id: str,
        *,
        config: HubConfig | None = None,
        backend: SupabaseEvidenceBackend | None = None,
        api_key: str | None = None,
        session: requests.Session | None = None,
    ):
        self.config = config or HubConfig.from_environment(client_id=client_id)
        self.backend = backend or SupabaseEvidenceBackend(self.config)
        self.api_key = _secret("ZAPI_KEY") if api_key is None else _clean(api_key)
        self.session = session or requests.Session()

    @property
    def ready(self) -> bool:
        return bool(self.config.ready)

    def persist_idx_profile(self, company_row: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rows = normalize_idx_company_profile_row(company_row)
        if not rows:
            return [], {"state": "NO_QUANTITATIVE_SHAREHOLDERS", "rows": 0}
        valid, reason = validate_shareholder_rows(rows)
        if not valid:
            return [], {"state": reason, "rows": 0}
        written = self.backend.upsert_rows(
            TABLE, rows, conflict=("provider", "ticker", "source_period", "holder_identity_hash")
        )
        return [dict(row) for row in written], {
            "state": "PERSISTED",
            "rows": len(written),
            "ticker": rows[0]["ticker"],
            "provider": rows[0]["provider"],
        }

    def refresh_pluang(self, ticker: str, *, observed_on: date) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        code = _ticker(ticker)
        if not code or not self.ready or not self.api_key:
            return [], {"state": MissingReason.ENVIRONMENT_BLOCKED.value, "api_calls": 0}
        try:
            response = self.session.get(
                PLUANG_PROFILE_URL,
                params={"code": code},
                headers={"x-api-key": self.api_key, "Accept": "application/json"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
        except requests.Timeout as exc:
            raise RuntimeError(MissingReason.TIMEOUT.value) from exc
        except requests.ConnectionError as exc:
            raise RuntimeError(MissingReason.CONNECTION_ERROR.value) from exc
        if response.status_code in {401, 403, 404, 429}:
            raise RuntimeError(f"HTTP_{response.status_code}")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, Mapping):
            raise RuntimeError(MissingReason.PARSE_FAILURE.value)
        rows = normalize_pluang_profile(payload, observed_on=observed_on)
        valid, reason = validate_shareholder_rows(rows)
        if not valid:
            return [], {"state": reason, "api_calls": 1, "rows": 0}
        written = self.backend.upsert_rows(
            TABLE, rows, conflict=("provider", "ticker", "source_period", "holder_identity_hash")
        )
        return [dict(row) for row in written], {
            "state": "REFRESHED",
            "api_calls": 1,
            "rows": len(written),
            "ticker": code,
            "provider": "PLUANG_COMPANY_PROFILE_VIA_ZAPI",
        }


__all__ = [
    "PLUANG_PROFILE_URL",
    "SharedStructuredOwnershipEvidence",
    "TABLE",
    "normalize_idx_company_profile_row",
    "normalize_pluang_profile",
    "validate_shareholder_rows",
]
