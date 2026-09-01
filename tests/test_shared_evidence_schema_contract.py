from __future__ import annotations

import ast
from pathlib import Path

from shared_evidence_schema_contract import (
    EVIDENCE_IDENTITIES,
    EXPECTED_PRODUCER_CONFLICTS,
    audit_shared_migration,
)


ROOT = Path(__file__).resolve().parents[1]


def _migration() -> str:
    return next((ROOT / "database").glob("migration_v*_shared_evidence_hub.sql")).read_text(encoding="utf-8")


def _conflict_tuples(path: Path) -> set[tuple[str, ...]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[tuple[str, ...]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "upsert_rows":
            continue
        keyword = next((item for item in node.keywords if item.arg == "conflict"), None)
        if keyword and isinstance(keyword.value, (ast.Tuple, ast.List)):
            values = tuple(
                item.value for item in keyword.value.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
            if values:
                found.add(values)
    return found


def test_migration_safety_audit_passes() -> None:
    report = audit_shared_migration(_migration())
    assert report.passed, report.issues
    assert set(report.tables) == set(EVIDENCE_IDENTITIES)


def test_every_table_has_expected_deterministic_primary_identity() -> None:
    report = audit_shared_migration(_migration())
    assert report.identities == EVIDENCE_IDENTITIES


def test_all_producer_upsert_conflicts_match_declared_identity() -> None:
    for filename, expected in EXPECTED_PRODUCER_CONFLICTS.items():
        actual = _conflict_tuples(ROOT / filename)
        assert expected <= actual, f"{filename}: expected {expected}, got {actual}"


def test_migration_is_additive_idempotent_and_scanner_neutral() -> None:
    sql = _migration().lower()
    assert sql.count("create table if not exists public.") == len(EVIDENCE_IDENTITIES)
    assert "create index if not exists" in sql
    assert "drop table" not in sql and "truncate table" not in sql and "drop column" not in sql
    assert "add constraint if not exists" not in sql
    assert "emir_score" not in sql and "pasticuan_score" not in sql


def test_broker_reference_constraint_is_idempotent_and_fail_closed() -> None:
    sql = _migration().lower()
    assert "where conname = 'evidence_brokers_reference_check'" in sql
    assert "foreign_ownership_percentage between 0 and 100" in sql
    assert "paid_up_capital is null or paid_up_capital >= 0" in sql
    assert "evidence_scope = 'market_wide'" in sql


def test_rls_grants_and_functions_follow_least_privilege() -> None:
    sql = _migration().lower()
    assert "enable row level security" in sql
    assert "from public, anon, authenticated" in sql
    assert "grant select, insert, update on table public.%i to service_role" in sql
    assert "grant delete" not in sql
    assert sql.count("security invoker") == 3
    assert sql.count("set search_path = ''") == 3
    assert "security definer" not in sql
