from __future__ import annotations

"""Persistence contract hardening for v9 Next-Leader score components.

The database schema already exposes these columns. The production serializer in
scanner_database v16 did not whitelist/project them, so a valid in-memory v9
ranking lost its component lineage on persistence. This module patches only the
persistence contract; it does not alter scoring weights, gates, or ranking.
"""

from typing import Any, Mapping

import pandas as pd

from scanner_database import (
    ScannerDatabaseBridge,
    TABLE_FIELD_TYPES,
    _coerce_integer,
    _coerce_numeric,
    _coerce_text,
)

PERSISTENCE_CONTRACT_PATCH_VERSION = "16.0.1-v9-score-contract"

V9_NUMERIC_FIELDS = {
    "ranking_score",
    "research_score",
    "score_coverage_pct",
    "business_quality_score",
    "business_quality_coverage_pct",
    "future_fundamental_score",
    "future_fundamental_coverage_pct",
    "valuation_mos_score",
    "valuation_mos_coverage_pct",
    "management_capital_score",
    "management_capital_coverage_pct",
    "issuer_macro_alignment_score",
    "issuer_macro_alignment_coverage_pct",
    "narrative_flow_score",
    "narrative_flow_coverage_pct",
    "technical_readiness_score",
    "technical_readiness_coverage_pct",
}
V9_TEXT_FIELDS = {"ranking_score_state"}
V9_INTEGER_FIELDS = {"real_money_risk_lots_cap"}


def _extend_field_contract() -> None:
    spec = TABLE_FIELD_TYPES.setdefault(
        "multibagger_snapshots",
        {"text": set(), "numeric": set(), "integer": set(), "boolean": set(), "json": set(), "required": set()},
    )
    spec.setdefault("text", set()).update(V9_TEXT_FIELDS)
    spec.setdefault("numeric", set()).update(V9_NUMERIC_FIELDS)
    spec.setdefault("integer", set()).update(V9_INTEGER_FIELDS)


def enrich_multibagger_payload_records(
    records: list[dict[str, Any]],
    source: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    """Copy v9 lineage fields from the ranking frame into persisted records."""
    if not records or not isinstance(source, pd.DataFrame) or source.empty or "ticker" not in source.columns:
        return records
    local = source.drop_duplicates("ticker", keep="last")
    lookup = {
        str(row.get("ticker") or "").strip().upper(): row
        for row in local.to_dict("records")
        if str(row.get("ticker") or "").strip()
    }
    for record in records:
        ticker = str(record.get("ticker") or "").strip().upper()
        row = lookup.get(ticker)
        if not isinstance(row, Mapping):
            continue
        for field in V9_NUMERIC_FIELDS:
            if field in row:
                record[field] = _coerce_numeric(row.get(field))
        for field in V9_TEXT_FIELDS:
            if field in row:
                record[field] = _coerce_text(row.get(field))
        for field in V9_INTEGER_FIELDS:
            if field in row:
                record[field] = _coerce_integer(row.get(field))
    return records


def apply_persistence_contract_patch() -> None:
    _extend_field_contract()
    if getattr(ScannerDatabaseBridge, "_v9_score_contract_patch", False):
        return
    original = ScannerDatabaseBridge.build_payloads

    def build_payloads_with_v9_contract(self: ScannerDatabaseBridge, result: Mapping[str, Any]):
        payloads = original(self, result)
        focus = result.get("focus_screens", {}) if isinstance(result, Mapping) else {}
        source = focus.get("multibagger", pd.DataFrame()) if isinstance(focus, Mapping) else pd.DataFrame()
        records = payloads.get("multibagger_snapshots", []) if isinstance(payloads, dict) else []
        if isinstance(records, list):
            payloads["multibagger_snapshots"] = enrich_multibagger_payload_records(records, source)
        return payloads

    ScannerDatabaseBridge.build_payloads = build_payloads_with_v9_contract
    ScannerDatabaseBridge._v9_score_contract_patch = True
    ScannerDatabaseBridge._v9_score_contract_patch_version = PERSISTENCE_CONTRACT_PATCH_VERSION


apply_persistence_contract_patch()


__all__ = [
    "PERSISTENCE_CONTRACT_PATCH_VERSION",
    "V9_NUMERIC_FIELDS",
    "V9_TEXT_FIELDS",
    "V9_INTEGER_FIELDS",
    "enrich_multibagger_payload_records",
    "apply_persistence_contract_patch",
]
