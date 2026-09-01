from __future__ import annotations

"""Static migration and deterministic-identity contract for the shared hub."""

from dataclasses import dataclass
import re
from typing import Mapping


EVIDENCE_IDENTITIES: Mapping[str, tuple[str, ...]] = {
    "evidence_provider_state": ("provider", "endpoint_family", "scope", "target_date"),
    "evidence_refresh_leases": ("provider", "evidence_family", "scope", "target_date"),
    "evidence_ingestion_runs": ("id",),
    "evidence_failures": ("id",),
    "evidence_raw_payloads": ("payload_hash",),
    "evidence_market_daily": ("provider", "trade_date", "ticker"),
    "evidence_foreign_flow": ("provider", "trade_date", "ticker"),
    "evidence_participant_flow": ("source", "trade_date", "ticker", "broker_code"),
    "evidence_ownership_files": ("source_file_hash", "category"),
    "evidence_ownership_snapshots": ("source_file_hash", "ticker", "holder_identity_hash"),
    "evidence_ownership_changes": ("source_file_hash", "ticker", "holder_identity_hash", "change_state"),
    "evidence_financial_reports": ("ticker", "report_period", "source_document_hash"),
    "evidence_financial_facts": ("ticker", "report_period", "fact_name", "source_document_hash"),
    "evidence_announcements": ("source_event_id",),
    "evidence_capital_actions": ("ticker", "event_type", "event_date", "source_id"),
    "evidence_companies": ("provider", "ticker"),
    "evidence_reference_values": ("provider", "set_name", "value_key"),
    "evidence_brokers": ("provider", "broker_code"),
    "evidence_broker_market_daily": ("provider", "activity_date", "broker_code"),
    "evidence_risk_events": ("provider", "event_type", "event_date", "ticker", "source_id"),
    "evidence_trading_calendar": ("trade_date",),
}

EXPECTED_PRODUCER_CONFLICTS: Mapping[str, frozenset[tuple[str, ...]]] = {
    "shared_stock_summary_evidence.py": frozenset({("provider", "trade_date", "ticker")}),
    "shared_participant_evidence.py": frozenset({("source", "trade_date", "ticker", "broker_code")}),
    "shared_ownership_evidence.py": frozenset({
        ("source_file_hash", "category"),
        ("source_file_hash", "ticker", "holder_identity_hash"),
        ("source_file_hash", "ticker", "holder_identity_hash", "change_state"),
    }),
    "shared_financial_evidence.py": frozenset({
        ("ticker", "report_period", "source_document_hash"),
        ("ticker", "report_period", "fact_name", "source_document_hash"),
    }),
    "shared_announcement_evidence.py": frozenset({("source_event_id",)}),
    "shared_capital_action_evidence.py": frozenset({("ticker", "event_type", "event_date", "source_id")}),
    "shared_company_evidence.py": frozenset({
        ("provider", "ticker"), ("provider", "set_name", "value_key"),
    }),
    "shared_risk_event_evidence.py": frozenset({
        ("provider", "event_type", "event_date", "ticker", "source_id"),
    }),
    "shared_broker_reference_evidence.py": frozenset({
        ("provider", "broker_code"), ("provider", "activity_date", "broker_code"),
    }),
}


@dataclass(frozen=True)
class MigrationAudit:
    passed: bool
    tables: tuple[str, ...]
    identities: Mapping[str, tuple[str, ...]]
    issues: tuple[str, ...]


def _columns(value: str) -> tuple[str, ...]:
    return tuple(part.strip().lower() for part in value.split(",") if part.strip())


def _table_blocks(sql: str) -> dict[str, str]:
    pattern = re.compile(
        r"create\s+table\s+if\s+not\s+exists\s+public\.(\w+)\s*\((.*?)\n\);",
        flags=re.IGNORECASE | re.DOTALL,
    )
    return {name.lower(): block for name, block in pattern.findall(sql)}


def audit_shared_migration(sql: str) -> MigrationAudit:
    lowered = sql.lower()
    blocks = _table_blocks(sql)
    issues: list[str] = []
    identities: dict[str, tuple[str, ...]] = {}
    for table, expected in EVIDENCE_IDENTITIES.items():
        block = blocks.get(table)
        if block is None:
            issues.append(f"MISSING_TABLE:{table}")
            continue
        match = re.search(r"primary\s+key\s*\(([^)]+)\)", block, flags=re.IGNORECASE)
        if match:
            actual = _columns(match.group(1))
        else:
            inline = re.search(r"^\s*(\w+)\s+[^,\n]*\bprimary\s+key\b", block, flags=re.IGNORECASE | re.MULTILINE)
            actual = (inline.group(1).lower(),) if inline else ()
        identities[table] = actual
        if actual != expected:
            issues.append(f"IDENTITY_MISMATCH:{table}:{','.join(actual)}")

    for forbidden in (
        "drop table", "truncate table", "drop column", "alter column", "rename column",
        "security definer", "emir_score", "pasticuan_score", "entry_price", "take_profit", "stop_loss",
    ):
        if forbidden in lowered:
            issues.append(f"FORBIDDEN_SQL:{forbidden}")
    if re.search(r"add\s+constraint\s+if\s+not\s+exists", lowered):
        issues.append("INVALID_ADD_CONSTRAINT_IF_NOT_EXISTS")
    for statement in re.findall(r"create\s+(?:unique\s+)?index[^;]+;", lowered, flags=re.DOTALL):
        if "if not exists" not in statement:
            issues.append("NON_IDEMPOTENT_INDEX")
    for function in (
        "evidence_acquire_refresh_lease", "evidence_complete_refresh_lease", "evidence_fail_refresh_lease",
    ):
        marker = f"create or replace function public.{function}"
        start = lowered.find(marker)
        end = lowered.find("$$;", start)
        body = lowered[start:end] if start >= 0 and end >= 0 else ""
        if "security invoker" not in body or "set search_path = ''" not in body:
            issues.append(f"UNSAFE_FUNCTION:{function}")
    for table in EVIDENCE_IDENTITIES:
        if f"'{table}'" not in lowered:
            issues.append(f"RLS_LOOP_MISSING:{table}")
    for required in (
        "enable row level security",
        "revoke all on table public.%i from public, anon, authenticated",
        "grant select, insert, update on table public.%i to service_role",
        "revoke all on function public.evidence_acquire_refresh_lease",
        "grant execute on function public.evidence_acquire_refresh_lease",
        "evidence_brokers_reference_check",
    ):
        if required not in lowered:
            issues.append(f"SECURITY_OR_CONSTRAINT_MISSING:{required}")
    if " grant delete " in f" {lowered} ":
        issues.append("EXCESSIVE_DELETE_GRANT")
    return MigrationAudit(
        passed=not issues,
        tables=tuple(sorted(blocks)),
        identities=identities,
        issues=tuple(issues),
    )


__all__ = [
    "EVIDENCE_IDENTITIES", "EXPECTED_PRODUCER_CONFLICTS", "MigrationAudit", "audit_shared_migration",
]
