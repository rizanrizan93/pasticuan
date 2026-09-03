from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v28_calculation_state_constraint_expands_without_data_mutation() -> None:
    sql = (ROOT / "database" / "migration_v28_capital_action_calculation_states.sql").read_text(
        encoding="utf-8"
    ).lower()
    assert "drop constraint if exists evidence_capital_actions_calculation_check" in sql
    assert "add constraint evidence_capital_actions_calculation_check" in sql
    assert "validate constraint evidence_capital_actions_calculation_check" in sql
    for state in (
        "explicit_event_shares_post_no_delta",
        "reported_post_negative_event_shares_only",
        "reported_post_negative_no_usable_total",
    ):
        assert state in sql
    for forbidden in ("drop table", "drop column", "truncate", "delete from", "update "):
        assert forbidden not in sql


def test_v28_has_read_only_verifier() -> None:
    sql = (ROOT / "database" / "verify_v28_capital_action_calculation_states.sql").read_text(
        encoding="utf-8"
    ).lower()
    assert "pg_get_constraintdef" in sql
    assert "raise exception" in sql
    assert "insert into" not in sql
    assert "update " not in sql
    assert "delete from" not in sql
