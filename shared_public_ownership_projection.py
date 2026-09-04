from __future__ import annotations

"""Materialize scanner-neutral Phase 5.6 public ownership context.

The private source remains ``evidence_ownership_concentration_metrics``. This
module emits one facts-only row per ticker. It never derives free float, KSEI
status, beneficial ownership, broker/bandar identity, score, rank, or gate.
"""

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

SOURCE_TABLE = "evidence_ownership_concentration_metrics"
PROJECTION_TABLE = "phase56_public_ownership_snapshots"
SOURCE_PROVIDER = "YAHOO_DIRECT_OWNERSHIP_CONCENTRATION"
PROJECTION_STATE = "PHASE5_6_PUBLIC_OWNERSHIP_CONTEXT"
PROVENANCE_STATE = "PUBLIC_PROVIDER_YAHOO_CONCENTRATION_NOT_IDX_KSEI"
EXPECTED_METRICS = (
    "insiders_held_pct",
    "institutions_held_pct",
    "institutions_float_held_pct",
    "institutions_count",
)
SELECT_FIELDS = "provider,ticker,source_period,observed_on,metric_name,metric_value,metric_unit,source_authority,official_verified,lineage_state,validation_state,fetched_at"


def _ticker(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text[:-3] if text.endswith(".JK") else text


def _ordinal(value: Any) -> str:
    return str(value or "").strip()


def canonicalize_ownership_concentration(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        ticker = _ticker(row.get("ticker"))
        metric = str(row.get("metric_name") or "").strip()
        if (
            row.get("provider") != SOURCE_PROVIDER
            or not ticker
            or metric not in EXPECTED_METRICS
            or str(row.get("validation_state") or "").upper() != "VALID"
            or str(row.get("source_authority") or "") != "PUBLIC_PROVIDER"
            or bool(row.get("official_verified"))
        ):
            continue
        try:
            value = float(row.get("metric_value"))
        except (TypeError, ValueError, OverflowError):
            continue
        key = (ticker, metric)
        new_order = (_ordinal(row.get("source_period")), _ordinal(row.get("observed_on")), _ordinal(row.get("fetched_at")))
        old = latest.get(key)
        old_order = (
            _ordinal(old.get("source_period")), _ordinal(old.get("observed_on")), _ordinal(old.get("fetched_at"))
        ) if old else ("", "", "")
        if new_order >= old_order:
            row["metric_value"] = value
            latest[key] = row

    output: dict[str, dict[str, Any]] = {}
    tickers = sorted({ticker for ticker, _ in latest})
    for ticker in tickers:
        metric_rows = {metric: latest[(ticker, metric)] for metric in EXPECTED_METRICS if (ticker, metric) in latest}
        if not metric_rows:
            continue
        source_period = max((_ordinal(row.get("source_period")) for row in metric_rows.values()), default="") or None
        observed_on = max((_ordinal(row.get("observed_on")) for row in metric_rows.values()), default="") or None
        output[ticker] = {
            "ticker": ticker,
            "source_period": source_period,
            "observed_on": observed_on,
            **{metric: metric_rows[metric]["metric_value"] if metric in metric_rows else None for metric in EXPECTED_METRICS},
            "coverage_pct": 25.0 * len(metric_rows),
            "source_authority": "PUBLIC_PROVIDER",
            "official_verified": False,
            "provenance_state": PROVENANCE_STATE,
            "source_state": PROJECTION_STATE,
        }
    return output


def refresh_public_ownership_projection(backend: Any, *, batch_size: int = 250) -> dict[str, Any]:
    rows = backend.read_rows(
        SOURCE_TABLE,
        {"provider": f"eq.{SOURCE_PROVIDER}"},
        select=SELECT_FIELDS,
        limit=10000,
    )
    bundle = canonicalize_ownership_concentration(rows)
    refreshed_at = datetime.now(timezone.utc).isoformat()
    records = [{**bundle[ticker], "refreshed_at": refreshed_at} for ticker in sorted(bundle)]
    persisted = 0
    size = max(1, min(int(batch_size), 500))
    for start in range(0, len(records), size):
        written = backend.upsert_rows(PROJECTION_TABLE, records[start:start + size], conflict=("ticker",))
        persisted += len(written)
    return {
        "state": "REFRESHED" if records and persisted == len(records) else "EMPTY" if not records else "PARTIAL",
        "source_rows": len(rows),
        "projection_rows": len(records),
        "persisted_rows": persisted,
        "four_metric_tickers": sum(float(row.get("coverage_pct") or 0) == 100.0 for row in records),
        "policy": "FACTS_ONLY_NO_FREE_FLOAT_KSEI_SCORE_OR_GATE_INFERENCE",
    }


__all__ = [
    "EXPECTED_METRICS",
    "PROJECTION_STATE",
    "PROJECTION_TABLE",
    "PROVENANCE_STATE",
    "SOURCE_PROVIDER",
    "SOURCE_TABLE",
    "canonicalize_ownership_concentration",
    "refresh_public_ownership_projection",
]
