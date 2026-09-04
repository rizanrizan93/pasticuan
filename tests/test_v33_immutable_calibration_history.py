from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (ROOT / "database" / "migration_v33_immutable_calibration_history.sql").read_text(encoding="utf-8")
SCANNER_DB = (ROOT / "scanner_database.py").read_text(encoding="utf-8")


def test_runtime_snapshot_identity_remains_bounded_latest_state() -> None:
    """v33 must not inflate the existing same-day runtime snapshot stores."""
    assert 'as_of_day = _clean_text(as_of)[:10]' in SCANNER_DB
    assert '"multibagger_snapshots": ("ticker", "model_version")' in SCANNER_DB
    assert '"technical_snapshots": ("ticker", "model_version")' in SCANNER_DB


def test_calibration_history_is_per_run_per_ticker() -> None:
    assert "pasticuan_calibration_snapshots_scan_ticker_key unique (scan_id, ticker)" in MIGRATION
    assert "v_calibration_id := v_scan_id || '|' || v_ticker" in MIGRATION
    assert "scan_id text not null" in MIGRATION
    assert "ticker text not null" in MIGRATION


def test_all_decision_source_lanes_feed_history() -> None:
    for table in ("multibagger_snapshots", "technical_snapshots", "fundamental_snapshots"):
        assert f"after insert or update on public.{table}" in MIGRATION
        assert f"'{table}'" in MIGRATION
    assert "has_multibagger boolean not null default false" in MIGRATION
    assert "has_technical boolean not null default false" in MIGRATION
    assert "has_fundamental boolean not null default false" in MIGRATION


def test_decision_time_and_forward_outcomes_are_separate() -> None:
    assert "create table if not exists public.pasticuan_calibration_outcomes" in MIGRATION
    assert "horizon_sessions in (1, 3, 5, 10, 20)" in MIGRATION
    assert "return_pct numeric" in MIGRATION
    assert "mfe_pct numeric" in MIGRATION
    assert "mae_pct numeric" in MIGRATION
    assert "tp1_hit boolean" in MIGRATION
    assert "stop_hit boolean" in MIGRATION


def test_calibration_history_is_not_public_or_shared_evidence() -> None:
    assert "revoke all on public.pasticuan_calibration_snapshots from public, anon, authenticated" in MIGRATION
    assert "grant select on public.pasticuan_calibration_snapshots to service_role" in MIGRATION
    assert "grant select, insert, update on public.pasticuan_calibration_outcomes to service_role" in MIGRATION
    assert "PASTICUAN-only immutable per-run decision history" in MIGRATION
    assert "NOT part of the Shared Evidence Hub" in MIGRATION


def test_missing_overwritten_same_day_history_is_not_fabricated() -> None:
    assert "cannot be reconstructed safely and are deliberately not invented" in MIGRATION
