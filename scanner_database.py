from __future__ import annotations

"""Fail-soft database bridge for the IDX scanner.

The scanner must remain fully usable before an external database exists.  This
module therefore defaults to DISABLED and performs no network I/O unless the
operator explicitly enables it through Streamlit secrets/environment values.

Supported modes:
- DISABLED: production default; returns an audit row only.
- OUTBOX_ONLY: writes JSONL payloads to a local outbox for later import.
- SUPABASE_REST: upserts bounded snapshot batches through PostgREST.

No database failure is allowed to stop price scanning, Multibagger scoring,
Core Swing ranking, EOFF, or portfolio review.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
import hashlib
import json
import math
import os

import numpy as np
import pandas as pd
import requests

DATABASE_BRIDGE_VERSION = "1.0-v6.8.1"
DATABASE_SCHEMA_VERSION = "scanner_schema_v1"


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        return stamp.isoformat()
    if isinstance(value, (dict, Mapping)):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value if isinstance(value, (str, bytes)) else str(value)


def _frame_records(frame: pd.DataFrame | None, columns: Iterable[str]) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    selected = [column for column in columns if column in frame.columns]
    if not selected:
        return []
    local = frame[selected].copy()
    return [{key: _json_safe(value) for key, value in row.items()} for row in local.to_dict("records")]


def _snapshot_id(table: str, record: Mapping[str, Any], as_of: str) -> str:
    identity = "|".join(
        _clean_text(record.get(key))
        for key in ("ticker", "period_end", "event_date", "best_buy_date", "model_version")
    )
    raw = f"{table}|{identity}|{as_of}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class DatabaseSettings:
    enabled: bool = False
    mode: str = "DISABLED"
    supabase_url: str = ""
    supabase_key: str = ""
    schema: str = "public"
    timeout_seconds: float = 8.0
    outbox_path: str = ".scanner_cache/database_outbox.jsonl"
    max_rows_per_table: int = 500

    @classmethod
    def from_env(cls) -> "DatabaseSettings":
        enabled = _truthy(os.getenv("SCANNER_DATABASE_ENABLED"))
        requested_mode = os.getenv("SCANNER_DATABASE_MODE", "").strip().upper()
        url = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
        key = os.getenv("SUPABASE_ANON_KEY", "").strip()
        if not enabled:
            mode = "DISABLED"
        elif requested_mode == "OUTBOX_ONLY":
            mode = "OUTBOX_ONLY"
        elif url and key:
            mode = "SUPABASE_REST"
        else:
            mode = "CONFIG_INCOMPLETE"
        return cls(
            enabled=enabled,
            mode=mode,
            supabase_url=url,
            supabase_key=key,
            schema=os.getenv("SCANNER_DATABASE_SCHEMA", "public").strip() or "public",
            timeout_seconds=max(2.0, float(os.getenv("SCANNER_DATABASE_TIMEOUT", "8"))),
            outbox_path=os.getenv("SCANNER_DATABASE_OUTBOX", ".scanner_cache/database_outbox.jsonl"),
            max_rows_per_table=max(20, int(os.getenv("SCANNER_DATABASE_MAX_ROWS", "500"))),
        )


class ScannerDatabaseBridge:
    """Persist bounded scanner snapshots without becoming a runtime dependency."""

    def __init__(self, settings: DatabaseSettings | None = None) -> None:
        self.settings = settings or DatabaseSettings.from_env()

    def status_row(self, state: str | None = None, detail: str = "") -> dict[str, Any]:
        return {
            "bridge_version": DATABASE_BRIDGE_VERSION,
            "schema_version": DATABASE_SCHEMA_VERSION,
            "database_mode": self.settings.mode,
            "state": state or ("READY" if self.settings.mode in {"OUTBOX_ONLY", "SUPABASE_REST"} else self.settings.mode),
            "table": "",
            "rows_attempted": 0,
            "rows_written": 0,
            "detail": detail,
        }

    def build_payloads(self, result: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
        as_of = datetime.now(timezone.utc).isoformat()
        focus = result.get("focus_screens", {}) if isinstance(result.get("focus_screens", {}), Mapping) else {}
        multibagger = focus.get("multibagger", pd.DataFrame())
        fundamentals = result.get("fundamentals", pd.DataFrame())
        projects = result.get("project_management_review", pd.DataFrame())

        payloads: dict[str, list[dict[str, Any]]] = {
            "fundamental_snapshots": _frame_records(
                fundamentals,
                (
                    "ticker", "period_end", "statement_date", "fundamental_score", "fundamental_score_10",
                    "fundamental_coverage", "fundamental_data_grade", "fundamental_reliability",
                    "revenue_growth", "earnings_growth", "roe", "roa", "roic_proxy", "net_margin",
                    "operating_margin", "operating_cash_flow", "free_cash_flow", "cash_conversion_ttm",
                    "debt_equity", "net_debt_ebitda", "interest_coverage", "market_cap",
                    "fundamental_source_families", "fundamental_official_verified", "statement_age_days",
                ),
            ),
            "multibagger_snapshots": _frame_records(
                multibagger,
                (
                    "ticker", "multibagger_status", "multibagger_quality_score", "execution_readiness_score",
                    "research_recommendation_status", "multibagger_candidate_type", "economic_earnings_score",
                    "economic_earnings_confidence", "minority_leakage_pct", "ocf_ebitda_conversion",
                    "silent_accumulation_score", "silent_accumulation_state", "absorption_confirmed_days20",
                    "failed_absorption_days20", "effort_result_absorption20", "effort_result_distribution20",
                    "persistent_bid_score", "project_pipeline_score", "project_stage", "project_stage_probability_pct",
                    "project_success_probability_pct", "future_fundamental_impact_score", "best_buy_date",
                    "best_buy_window_start", "best_buy_window_end", "eoff_state", "eoff_reconstruction_score",
                    "eoff_public_validation_state", "eoff_public_lift", "last_price",
                ),
            ),
            "project_events": _frame_records(
                projects,
                (
                    "ticker", "project_name", "project_names", "project_stage", "project_completion_pct",
                    "project_funding_secured_pct", "project_ownership_pct", "project_capex_idr",
                    "project_expected_revenue_idr", "project_expected_ebitda_idr", "project_data_coverage",
                    "project_source_families", "project_source_urls", "project_source_quorum_verified",
                    "project_execution_flags", "last_verified_at", "review_origin",
                ),
            ),
        }
        model_version = _clean_text(result.get("scanner_version")) or "6.8.1"
        for table, records in payloads.items():
            bounded = records[: self.settings.max_rows_per_table]
            for record in bounded:
                record["as_of"] = as_of
                record["model_version"] = model_version
                record["schema_version"] = DATABASE_SCHEMA_VERSION
                record["snapshot_id"] = _snapshot_id(table, record, as_of)
            payloads[table] = bounded
        return payloads

    def persist_scan_result(self, result: Mapping[str, Any]) -> pd.DataFrame:
        if self.settings.mode == "DISABLED":
            return pd.DataFrame([self.status_row("DISABLED_NO_DATABASE", "Scanner tetap berjalan dengan cache lokal; database eksternal belum diaktifkan.")])
        if self.settings.mode == "CONFIG_INCOMPLETE":
            return pd.DataFrame([self.status_row("CONFIG_INCOMPLETE", "Aktifkan OUTBOX_ONLY atau isi SUPABASE_URL dan SUPABASE_ANON_KEY.")])
        payloads = self.build_payloads(result)
        rows: list[dict[str, Any]] = []
        for table, records in payloads.items():
            if not records:
                row = self.status_row("NO_ROWS")
                row.update({"table": table})
                rows.append(row)
                continue
            try:
                written = self._write_outbox(table, records) if self.settings.mode == "OUTBOX_ONLY" else self._upsert_supabase(table, records)
                row = self.status_row("OK")
                row.update({"table": table, "rows_attempted": len(records), "rows_written": written})
            except Exception as exc:  # fail-soft by design
                row = self.status_row("DATABASE_FAIL_SOFT", f"{type(exc).__name__}: {str(exc)[:240]}")
                row.update({"table": table, "rows_attempted": len(records), "rows_written": 0})
            rows.append(row)
        return pd.DataFrame(rows)

    def _write_outbox(self, table: str, records: list[dict[str, Any]]) -> int:
        path = Path(self.settings.outbox_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps({"table": table, "record": record}, ensure_ascii=False) + "\n")
        return len(records)

    def _upsert_supabase(self, table: str, records: list[dict[str, Any]]) -> int:
        endpoint = f"{self.settings.supabase_url}/rest/v1/{table}?on_conflict=snapshot_id"
        headers = {
            "apikey": self.settings.supabase_key,
            "Authorization": f"Bearer {self.settings.supabase_key}",
            "Content-Type": "application/json",
            "Accept-Profile": self.settings.schema,
            "Content-Profile": self.settings.schema,
            "Prefer": "resolution=merge-duplicates,return=minimal",
            "User-Agent": f"idx-scanner/{DATABASE_BRIDGE_VERSION}",
        }
        written = 0
        for start in range(0, len(records), 100):
            batch = records[start : start + 100]
            response = requests.post(endpoint, headers=headers, json=batch, timeout=self.settings.timeout_seconds)
            response.raise_for_status()
            written += len(batch)
        return written


__all__ = [
    "DATABASE_BRIDGE_VERSION",
    "DATABASE_SCHEMA_VERSION",
    "DatabaseSettings",
    "ScannerDatabaseBridge",
]
