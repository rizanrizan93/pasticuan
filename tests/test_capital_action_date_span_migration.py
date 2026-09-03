from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v25_capital_action_date_span_migration_is_additive_and_explicit() -> None:
    sql = (ROOT / "database" / "migration_v25_capital_action_event_span.sql").read_text(
        encoding="utf-8"
    ).lower()
    assert "alter table public.evidence_capital_actions" in sql
    for column in ("event_date_kind", "event_start_date", "event_end_date"):
        assert f"add column if not exists {column}" in sql
    assert "event_date_kind = 'point'" in sql
    assert "event_date_kind = 'range_end'" in sql
    assert "event_start_date <= event_end_date" in sql
    assert "event_date = event_end_date" in sql
    for forbidden in ("drop table", "drop column", "truncate", "delete from", "alter column"):
        assert forbidden not in sql


def test_v25_has_read_only_verifier() -> None:
    sql = (ROOT / "database" / "verify_v25_capital_action_event_span.sql").read_text(
        encoding="utf-8"
    ).lower()
    assert "event_date_kind" in sql
    assert "event_start_date" in sql
    assert "event_end_date" in sql
    assert "raise exception" in sql
    assert "insert into" not in sql
    assert "update " not in sql
    assert "delete from" not in sql
