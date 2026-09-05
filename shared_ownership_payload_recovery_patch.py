from __future__ import annotations

"""Final persistence-boundary recovery for shared ownership evidence.

Production proof showed that canonical public-ownership and KSEI facts existed in
Shared Hub but could still be absent from ``multibagger_snapshots`` after the
runtime focus-wrapper chain.  This patch recovers only the factual ownership
families at ``ScannerDatabase.build_payloads``.  It never derives free float,
beneficial ownership, score, ranking, gate, recommendation, or authorization.
"""

from functools import wraps
from typing import Any

from shared_ownership_runtime_transport_patch import (
    _shared_ksei_ownership,
    _shared_public_ownership,
)

PATCH_VERSION = "1.0.0-postproof-shared-ownership-payload-recovery"

PUBLIC_PREFIX = "ownership_public_"
KSEI_PREFIX = "ownership_ksei_"


def _ticker(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    return text if text.endswith(".JK") else f"{text}.JK"


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _public_valid(row: dict[str, Any]) -> bool:
    return bool(
        _number(row.get("ownership_public_context_coverage_pct"), 0.0) > 0
        and str(row.get("ownership_public_source_authority") or "").strip()
    )


def _ksei_valid(row: dict[str, Any]) -> bool:
    return bool(
        _number(row.get("ownership_ksei_context_coverage_pct"), 0.0) > 0
        and bool(row.get("ownership_ksei_official_verified"))
        and str(row.get("ownership_ksei_source_authority") or "").strip().upper() == "OFFICIAL_KSEI"
    )


def _copy_family(row: dict[str, Any], context: dict[str, Any], prefix: str) -> None:
    for key, value in context.items():
        if str(key).startswith(prefix):
            row[key] = value


def _recover_payload_families(payloads: dict[str, Any]) -> dict[str, Any]:
    rows = payloads.get("multibagger_snapshots")
    if not isinstance(rows, list) or not rows:
        return payloads
    tickers = list(dict.fromkeys(_ticker(row.get("ticker")) for row in rows if isinstance(row, dict) and _ticker(row.get("ticker"))))
    if not tickers:
        return payloads

    public = _shared_public_ownership(tickers)
    ksei = _shared_ksei_ownership(tickers)
    for row in rows:
        if not isinstance(row, dict):
            continue
        ticker = _ticker(row.get("ticker"))
        if not ticker:
            continue
        public_context = dict(public.get(ticker) or {})
        ksei_context = dict(ksei.get(ticker) or {})
        if public_context and not _public_valid(row):
            _copy_family(row, public_context, PUBLIC_PREFIX)
        if ksei_context and not _ksei_valid(row):
            _copy_family(row, ksei_context, KSEI_PREFIX)
    return payloads


def install() -> dict[str, str]:
    import scanner_database as database
    import pasticuan_ownership_calibration_telemetry_patch as public_telemetry
    import pasticuan_ksei_calibration_telemetry_patch as ksei_telemetry

    spec = database.TABLE_FIELD_TYPES.get("multibagger_snapshots")
    if not isinstance(spec, dict):
        raise RuntimeError("multibagger_snapshots field contract unavailable")
    spec.setdefault("text", set()).update(public_telemetry.TEXT_FIELDS | ksei_telemetry.TEXT_FIELDS)
    spec.setdefault("numeric", set()).update(public_telemetry.NUMERIC_FIELDS | ksei_telemetry.NUMERIC_FIELDS)
    spec.setdefault("integer", set()).update(public_telemetry.INTEGER_FIELDS)
    spec.setdefault("boolean", set()).update(public_telemetry.BOOLEAN_FIELDS | ksei_telemetry.BOOLEAN_FIELDS)

    scanner_cls = database.ScannerDatabase
    original = scanner_cls.build_payloads
    if getattr(original, "__shared_ownership_payload_recovery_v1__", False):
        return {"patch_version": PATCH_VERSION, "state": "ALREADY_INSTALLED"}

    @wraps(original)
    def build_payloads_with_recovery(self: Any, result: Any):
        payloads = original(self, result)
        if isinstance(payloads, dict):
            return _recover_payload_families(payloads)
        return payloads

    build_payloads_with_recovery.__shared_ownership_payload_recovery_v1__ = True
    scanner_cls.build_payloads = build_payloads_with_recovery
    return {
        "patch_version": PATCH_VERSION,
        "state": "INSTALLED",
        "source": "CANONICAL_SHARED_HUB_PUBLIC_AND_KSEI",
        "scope": "PERSISTENCE_BOUNDARY_FACTUAL_RECOVERY_ONLY",
        "regulatory_free_float": "NOT_INFERRED",
        "beneficial_ownership": "NOT_INFERRED",
        "authorization": "UNCHANGED",
    }


__all__ = ["PATCH_VERSION", "install", "_recover_payload_families"]
