from __future__ import annotations

import pandas as pd

import scanner_database
import technical_persistence_semantics_patch as patch


def test_current_strategy_and_setup_status_fill_legacy_audit_columns(monkeypatch) -> None:
    original = scanner_database._frame_records
    frame = pd.DataFrame([
        {
            "ticker": "MARK.JK",
            "strategy": "CORE_SWING",
            "setup_status": "ENTRY_PLAN_READY",
            "entry": 930.0,
            "stop_loss": 890.0,
        }
    ])
    try:
        state = patch.install()
        rows = scanner_database._frame_records(
            frame,
            ("ticker", "active_setup", "technical_entry_state", "entry", "stop_loss"),
        )
    finally:
        monkeypatch.setattr(scanner_database, "_frame_records", original)

    assert state["state"] in {"INSTALLED", "ALREADY_INSTALLED"}
    assert rows == [{
        "ticker": "MARK.JK",
        "active_setup": "CORE_SWING",
        "technical_entry_state": "ENTRY_PLAN_READY",
        "entry": 930.0,
        "stop_loss": 890.0,
    }]


def test_existing_explicit_audit_columns_win() -> None:
    frame = pd.DataFrame([
        {
            "ticker": "TEST.JK",
            "active_setup": "EXPLICIT_SETUP",
            "technical_entry_state": "EXPLICIT_STATE",
            "strategy": "CORE_SWING",
            "setup_status": "WATCHLIST",
        }
    ])
    out = patch._with_current_aliases(
        frame,
        ("ticker", "active_setup", "technical_entry_state"),
    )
    assert out is not None
    assert out.loc[0, "active_setup"] == "EXPLICIT_SETUP"
    assert out.loc[0, "technical_entry_state"] == "EXPLICIT_STATE"


def test_patch_is_persistence_only() -> None:
    source = open("technical_persistence_semantics_patch.py", encoding="utf-8").read()
    assert "scanner_database._frame_records" in source
    assert "setup_score" not in source
    assert "ranking_score" not in source
    assert "real_money_authorization" not in source
