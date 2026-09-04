from __future__ import annotations

"""Refresh the read-only Phase 5.6 public factual fundamental projection.

The source of truth remains ``evidence_fundamental_metrics``.  This module only
materializes one scanner-neutral row per ticker so EMIR/PASTICUAN consumers can
read the same facts efficiently without cross-database service-role access.
"""

from datetime import datetime, timezone
from typing import Any

from shared_fundamental_runtime import canonicalize_metric_rows

SOURCE_TABLE = "evidence_fundamental_metrics"
PROJECTION_TABLE = "phase56_public_fundamental_snapshots"
PROJECTION_STATE = "PHASE5_6_PUBLIC_FACTUAL_PROJECTION"
SELECT_FIELDS = (
    "provider,ticker,period_end,statement_date,metric_name,metric_value,metric_unit,"
    "source_families,official_verified,source_record_hash,lineage_state,observed_at,"
    "validation_state,fetched_at"
)


def _projection_row(item: dict[str, Any], refreshed_at: str) -> dict[str, Any]:
    return {
        "ticker": str(item.get("ticker") or "").strip().upper(),
        "proxy_period_end": item.get("proxy_period_end"),
        "proxy_observed_at": item.get("proxy_observed_at"),
        "official_period_end": item.get("official_period_end"),
        "official_observed_at": item.get("official_observed_at"),
        "proxy_metrics": dict(item.get("proxy_metrics") or {}),
        "official_metrics": dict(item.get("official_metrics") or {}),
        "source_families": list(item.get("source_families") or []),
        "official_coverage_pct": float(item.get("official_coverage_pct") or 0.0),
        "source_state": PROJECTION_STATE,
        "refreshed_at": refreshed_at,
    }


def refresh_public_fundamental_projection(backend: Any, *, batch_size: int = 250) -> dict[str, Any]:
    rows = backend.read_rows(SOURCE_TABLE, {}, select=SELECT_FIELDS, limit=50000)
    bundle = canonicalize_metric_rows(rows)
    refreshed_at = datetime.now(timezone.utc).isoformat()
    records = [
        _projection_row(bundle[ticker], refreshed_at)
        for ticker in sorted(bundle)
        if str(bundle[ticker].get("ticker") or "").strip()
    ]

    persisted = 0
    size = max(1, min(int(batch_size), 500))
    for start in range(0, len(records), size):
        batch = records[start : start + size]
        written = backend.upsert_rows(PROJECTION_TABLE, batch, conflict=("ticker",))
        persisted += len(written)

    return {
        "state": "REFRESHED" if records and persisted == len(records) else "EMPTY" if not records else "PARTIAL",
        "source_rows": len(rows),
        "projection_rows": len(records),
        "persisted_rows": persisted,
        "official_tickers": sum(bool(row.get("official_metrics")) for row in records),
        "period_anchored_tickers": sum(bool(row.get("proxy_period_end")) for row in records),
        "policy": "FACTS_ONLY_NO_SCORING_OR_GATE_CHANGE",
    }


__all__ = [
    "PROJECTION_STATE",
    "PROJECTION_TABLE",
    "SOURCE_TABLE",
    "refresh_public_fundamental_projection",
]
