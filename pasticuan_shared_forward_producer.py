from __future__ import annotations

"""Publish strict PASTICUAN project facts into the canonical Shared Hub.

`project_events` remains PASTICUAN's producer/audit table. This module copies
only strict factual fields into `evidence_forward_events`, reconciles existing
producer provenance first, and never publishes PASTICUAN scores or decisions.
"""

from typing import Any, Mapping

from shared_evidence_hub import HubConfig, SupabaseEvidenceBackend
from shared_forward_evidence import (
    canonicalize_pasticuan_row,
    merge_equivalent_rows,
    read_canonical_forward_rows,
    upsert_canonical_forward_rows,
)

PRODUCER_VERSION = "1.0.0-pasticuan-canonical-forward-producer"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "verified", "pass", "valid"}


def _strict_source_row(row: Mapping[str, Any]) -> bool:
    try:
        quorum = int(float(row.get("source_quorum_count") or 0))
    except (TypeError, ValueError):
        quorum = 0
    urls = str(row.get("project_source_urls") or row.get("source_url") or "").strip()
    return bool(
        _truthy(row.get("project_source_quorum_verified"))
        and quorum >= 2
        and _truthy(row.get("entity_match_verified"))
        and "https://" in urls.lower()
        and (row.get("evidence_date") or row.get("event_date") or row.get("last_verified_at"))
    )


def _factual_only(row: Mapping[str, Any]) -> dict[str, Any]:
    item = canonicalize_pasticuan_row(row)
    # Producer-specific interpretation/audit labels never enter the shared
    # factual payload. Quantified contract/capex/revenue facts may remain.
    payload = dict(item.get("payload") or {})
    for key in ("project_stage", "project_execution_flags", "review_origin"):
        payload.pop(key, None)
    item["payload"] = payload
    return item


def publish_project_events() -> dict[str, Any]:
    config = HubConfig.from_environment(client_id="PASTICUAN")
    if not config.ready:
        return {"state": "SHARED_HUB_UNAVAILABLE", **config.status()}

    try:
        rows = SupabaseEvidenceBackend(config).read_rows("project_events", {}, limit=50000)
    except Exception as exc:
        return {"state": "PROJECT_EVENTS_READ_FAIL_SOFT", "error": f"{type(exc).__name__}: {str(exc)[:240]}"}

    local: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping) or not _strict_source_row(row):
            continue
        try:
            item = _factual_only(row)
        except Exception:
            continue
        if item.get("ticker") and item.get("canonical_event_id") and item.get("primary_source_url"):
            local.append(item)

    if not local:
        return {"state": "NO_STRICT_PROJECT_EVENTS", "rows": 0, "producer_version": PRODUCER_VERSION}

    tickers = sorted({str(row.get("ticker") or "") for row in local if str(row.get("ticker") or "")})
    existing, audit = read_canonical_forward_rows(tickers, client_id="PASTICUAN")
    if str(audit.get("state") or "") != "SHARED_CANONICAL_FORWARD":
        # Blind upsert could erase EMIR provenance JSON; fail closed instead.
        return {
            "state": "RECONCILE_READ_UNAVAILABLE",
            "rows": 0,
            "producer_version": PRODUCER_VERSION,
            "error": str(audit.get("error") or ""),
        }

    existing_by_id = {
        str(row.get("canonical_event_id") or ""): dict(row)
        for row in existing
        if str(row.get("canonical_event_id") or "")
    }
    reconciled: list[dict[str, Any]] = []
    for item in local:
        previous = existing_by_id.get(str(item.get("canonical_event_id") or ""))
        reconciled.append(merge_equivalent_rows(previous, item) if previous else item)

    _, write_audit = upsert_canonical_forward_rows(reconciled, client_id="PASTICUAN")
    return {
        "state": str(write_audit.get("state") or "UNKNOWN"),
        "rows": len(reconciled),
        "producer_version": PRODUCER_VERSION,
        "source_rows": len(rows),
        "strict_source_rows": len(local),
    }


__all__ = ["PRODUCER_VERSION", "publish_project_events", "_factual_only", "_strict_source_row"]
