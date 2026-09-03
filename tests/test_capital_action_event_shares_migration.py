from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v26_event_shares_migration_is_additive_and_explicit() -> None:
    sql = (ROOT / "database" / "migration_v26_capital_action_event_shares.sql").read_text(
        encoding="utf-8"
    ).lower()
    assert "alter table public.evidence_capital_actions" in sql
    assert "add column if not exists event_shares numeric" in sql
    assert "not assumed to equal delta_shares" in sql
    for forbidden in ("drop table", "drop column", "truncate", "delete from", "alter column"):
        assert forbidden not in sql


def test_v26_event_shares_has_read_only_verifier() -> None:
    sql = (ROOT / "database" / "verify_v26_capital_action_event_shares.sql").read_text(
        encoding="utf-8"
    ).lower()
    assert "event_shares" in sql
    assert "data_type='numeric'" in sql
    assert "raise exception" in sql
    assert "insert into" not in sql
    assert "update " not in sql
    assert "delete from" not in sql
