from __future__ import annotations

"""Persist runtime-used official KSEI composition as immutable calibration telemetry.

Observability only.  This patch never maps KSEI scripless composition to
regulatory free float, ownership score, ranking, or execution authorization.
"""

from typing import Any, Mapping

import pandas as pd

PATCH_VERSION = "1.0.0-v35-ksei-ownership-calibration-telemetry"
TEXT_FIELDS = {
    "ownership_ksei_observed_on",
    "ownership_ksei_source_url",
    "ownership_ksei_source_authority",
    "ownership_ksei_provenance_state",
    "ownership_ksei_context_state",
}
NUMERIC_FIELDS = {
    "ownership_ksei_total_shares",
    "ownership_ksei_scripless_pct",
    "ownership_ksei_local_pct",
    "ownership_ksei_foreign_pct",
    "ownership_ksei_context_coverage_pct",
}
BOOLEAN_FIELDS = {"ownership_ksei_official_verified"}
ALL_FIELDS = TEXT_FIELDS | NUMERIC_FIELDS | BOOLEAN_FIELDS


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

    scanner_cls = database.ScannerDatabase
    marker = "_pasticuan_ksei_calibration_telemetry_patch"
    if getattr(scanner_cls, marker, "") == PATCH_VERSION:
        return {"patch_version": PATCH_VERSION, "state": "ALREADY_INSTALLED"}
    spec = database.TABLE_FIELD_TYPES.get("multibagger_snapshots")
    if not isinstance(spec, dict):
        raise RuntimeError("multibagger_snapshots field contract unavailable")
    spec.setdefault("text", set()).update(TEXT_FIELDS)
    spec.setdefault("numeric", set()).update(NUMERIC_FIELDS)
    spec.setdefault("boolean", set()).update(BOOLEAN_FIELDS)

    original_build_payloads = scanner_cls.build_payloads

    def build_payloads_with_ksei(self: Any, result: Mapping[str, Any]):
        payloads = original_build_payloads(self, result)
        if not isinstance(payloads, dict):
            return payloads
        focus = result.get("focus_screens", {}) if isinstance(result, Mapping) else {}
        frame = focus.get("multibagger") if isinstance(focus, Mapping) else None
        context = _extract_context(frame)
        rows = payloads.get("multibagger_snapshots")
        if not context or not isinstance(rows, list):
            return payloads
        for row in rows:
            if not isinstance(row, dict):
                continue
            values = context.get(_ticker(row.get("ticker")))
            if values:
                row.update(values)
        return payloads

    scanner_cls.build_payloads = build_payloads_with_ksei
    setattr(scanner_cls, marker, PATCH_VERSION)
    return {
        "patch_version": PATCH_VERSION,
        "state": "INSTALLED",
        "semantics": "OFFICIAL_KSEI_COMPOSITION_TELEMETRY_NOT_REGULATORY_FREE_FLOAT",
    }


__all__ = ["PATCH_VERSION", "ALL_FIELDS", "TEXT_FIELDS", "NUMERIC_FIELDS", "BOOLEAN_FIELDS", "_extract_context", "install"]
