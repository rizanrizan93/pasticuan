from __future__ import annotations

"""Persist Phase 5.6 public ownership context as decision-time telemetry.

This patch is deliberately observability-only.  The Yahoo-derived public
ownership concentration context is copied from the final PASTICUAN focus frame
into ``multibagger_snapshots`` so the immutable calibration trigger can capture
what the decision runtime actually saw.

It MUST NOT populate KSEI, regulatory free-float, beneficial ownership,
scoring, ranking, or execution-authorization fields.
"""

from typing import Any, Mapping

import pandas as pd

PATCH_VERSION = "1.0.0-v34-ownership-calibration-telemetry"

TEXT_FIELDS = {
    "ownership_public_source_period",
    "ownership_public_observed_on",
    "ownership_public_source_authority",
    "ownership_public_context_provenance_state",
    "ownership_public_context_state",
}
NUMERIC_FIELDS = {
    "ownership_public_insiders_held_pct",
    "ownership_public_institutions_held_pct",
    "ownership_public_institutions_float_held_pct",
    "ownership_public_context_coverage_pct",
}
INTEGER_FIELDS = {"ownership_public_institutions_count"}
BOOLEAN_FIELDS = {"ownership_public_official_verified"}
ALL_FIELDS = TEXT_FIELDS | NUMERIC_FIELDS | INTEGER_FIELDS | BOOLEAN_FIELDS


def _ticker(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    return text if text.endswith(".JK") else f"{text}.JK"


def _present(value: Any) -> bool:
    if value is None:
        return False
    try:
        return not bool(pd.isna(value))
    except (TypeError, ValueError):
        return True


def _extract_context(frame: Any) -> dict[str, dict[str, Any]]:
    """Extract only explicit public-context fields from the final runtime frame."""
    if not isinstance(frame, pd.DataFrame) or frame.empty or "ticker" not in frame.columns:
        return {}
    output: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        ticker = _ticker(row.get("ticker"))
        if not ticker:
            continue
        values = {field: row.get(field) for field in ALL_FIELDS if _present(row.get(field))}
        if values:
            output[ticker] = values
    return output


def install() -> dict[str, str]:
    import scanner_database as database

    marker = "_pasticuan_ownership_calibration_telemetry_patch"
    scanner_cls = database.ScannerDatabase
    if getattr(scanner_cls, marker, "") == PATCH_VERSION:
        return {"patch_version": PATCH_VERSION, "state": "ALREADY_INSTALLED"}

    spec = database.TABLE_FIELD_TYPES.get("multibagger_snapshots")
    if not isinstance(spec, dict):
        raise RuntimeError("multibagger_snapshots field contract unavailable")
    spec.setdefault("text", set()).update(TEXT_FIELDS)
    spec.setdefault("numeric", set()).update(NUMERIC_FIELDS)
    spec.setdefault("integer", set()).update(INTEGER_FIELDS)
    spec.setdefault("boolean", set()).update(BOOLEAN_FIELDS)

    original_build_payloads = scanner_cls.build_payloads

    def build_payloads_with_ownership(self: Any, result: Mapping[str, Any]):
        payloads = original_build_payloads(self, result)
        if not isinstance(payloads, dict):
            return payloads
        focus = result.get("focus_screens", {}) if isinstance(result, Mapping) else {}
        frame = focus.get("multibagger") if isinstance(focus, Mapping) else None
        context = _extract_context(frame)
        if not context:
            return payloads
        rows = payloads.get("multibagger_snapshots")
        if not isinstance(rows, list):
            return payloads
        for row in rows:
            if not isinstance(row, dict):
                continue
            values = context.get(_ticker(row.get("ticker")))
            if values:
                row.update(values)
        return payloads

    scanner_cls.build_payloads = build_payloads_with_ownership
    setattr(scanner_cls, marker, PATCH_VERSION)
    return {
        "patch_version": PATCH_VERSION,
        "state": "INSTALLED",
        "semantics": "PUBLIC_CONTEXT_TELEMETRY_ONLY_NOT_KSEI_FREE_FLOAT",
    }


__all__ = [
    "PATCH_VERSION",
    "ALL_FIELDS",
    "TEXT_FIELDS",
    "NUMERIC_FIELDS",
    "INTEGER_FIELDS",
    "BOOLEAN_FIELDS",
    "_extract_context",
    "install",
]
