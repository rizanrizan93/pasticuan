from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v27_reported_post_shares_migration_is_additive() -> None:
    sql = (ROOT / "database" / "migration_v27_capital_action_reported_post_shares.sql").read_text(
        encoding="utf-8"
    ).lower()
    assert "alter table public.evidence_capital_actions" in sql
    assert "add column if not exists reported_post_shares numeric" in sql
    assert "post_shares remains the normalized usable non-negative total" in sql
    for forbidden in ("drop table", "drop column", "truncate", "delete from", "alter column"):
        assert forbidden not in sql


def test_v27_reported_post_shares_has_read_only_verifier() -> None:
    sql = (ROOT / "database" / "verify_v27_capital_action_reported_post_shares.sql").read_text(
        encoding="utf-8"
    ).lower()
    assert "reported_post_shares" in sql
    assert "data_type='numeric'" in sql
    assert "raise exception" in sql
    assert "insert into" not in sql
    assert "update " not in sql
    assert "delete from" not in sql
