from __future__ import annotations

import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TABLES = (
    "evidence_provider_state",
    "evidence_refresh_leases",
    "evidence_ingestion_runs",
    "evidence_failures",
    "evidence_raw_payloads",
    "evidence_market_daily",
    "evidence_foreign_flow",
    "evidence_participant_flow",
    "evidence_ownership_files",
    "evidence_ownership_snapshots",
    "evidence_ownership_changes",
    "evidence_financial_reports",
    "evidence_financial_facts",
    "evidence_announcements",
    "evidence_capital_actions",
    "evidence_companies",
    "evidence_reference_values",
    "evidence_brokers",
    "evidence_broker_market_daily",
    "evidence_risk_events",
    "evidence_trading_calendar",
)
ORIGINAL_MIGRATION_SHA256 = "63a5866d732a3801aa2baa77e8aef41c620ee16b7942dd47a4b5d7654004fbc1"


def _single(pattern: str) -> Path:
    matches = sorted((ROOT / "database").glob(pattern))
    assert len(matches) == 1, matches
    return matches[0]


def _migration() -> str:
    return _single("migration_v*_shared_evidence_acl_hardening.sql").read_text(encoding="utf-8")


def _verification() -> str:
    return _single("verify_v*_shared_evidence_acl_hardening.sql").read_text(encoding="utf-8")


def _first_inventory(sql: str) -> tuple[str, ...]:
    match = re.search(r"foreach table_name in array array\[(.*?)\]\s*loop", sql, re.I | re.S)
    assert match
    return tuple(re.findall(r"'(evidence_[a-z0-9_]+)'", match.group(1)))


def test_acl_hardening_uses_exact_table_inventory() -> None:
    assert _first_inventory(_migration()) == EXPECTED_TABLES
    assert _first_inventory(_verification()) == EXPECTED_TABLES


def test_acl_hardening_revokes_before_exact_regrant() -> None:
    sql = _migration().lower()
    revoke = "revoke all privileges on table public.%i from service_role"
    grant = "grant select, insert, update on table public.%i to service_role"
    assert sql.count(revoke) == 1
    assert sql.count(grant) == 1
    assert sql.index(revoke) < sql.index(grant)


def test_acl_hardening_is_idempotent_and_acl_only() -> None:
    sql = _migration().lower()
    assert "alter default privileges" not in sql
    assert "drop " not in sql
    assert "truncate table" not in sql
    assert "delete from" not in sql
    assert "insert into" not in sql
    assert not re.search(r"update\s+public\.", sql)
    assert "alter table" not in sql
    assert "create table" not in sql
    assert " function " not in sql
    assert "sequence" not in sql
    assert "owner to" not in sql
    assert "idx_flow" not in sql
    assert not re.search(r"(emir|pasticuan)_(score|rank|gate)", sql)


def test_verification_requires_exact_effective_and_raw_acl() -> None:
    sql = _verification().lower()
    for privilege in ("select", "insert", "update", "delete", "truncate", "references", "trigger", "maintain"):
        assert f"has_table_privilege('service_role', relation_oid, '{privilege.upper()}')".lower() in sql
    assert "array['insert', 'select', 'update']::text[]" in sql
    assert "acl.grantee = 0" in sql
    assert "has_table_privilege('anon', relation_oid, privilege_name)" in sql
    assert "has_table_privilege('authenticated', relation_oid, privilege_name)" in sql
    assert "relrowsecurity" in sql


def test_verification_preserves_function_security_contract() -> None:
    sql = _verification().lower()
    assert "prosecdef" in sql
    assert "search_path=" in sql
    assert "has_function_privilege('service_role', oid, 'execute')" in sql
    assert "has_function_privilege('anon', oid, 'execute')" in sql
    assert "has_function_privilege('authenticated', oid, 'execute')" in sql


def test_original_shared_hub_migration_is_immutable() -> None:
    original = _single("migration_v*_shared_evidence_hub.sql").read_bytes()
    assert hashlib.sha256(original).hexdigest() == ORIGINAL_MIGRATION_SHA256
