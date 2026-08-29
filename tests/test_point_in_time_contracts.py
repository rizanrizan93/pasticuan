from __future__ import annotations

import pandas as pd

from scanner import SetupPlan, build_fundamental_history_features, normalize_fundamental_history


def _statement_history() -> pd.DataFrame:
    return pd.DataFrame([{
        "ticker": "TEST.JK",
        "period_end": "2026-06-30",
        "period_type": "Q2",
        "statement_basis": "STANDALONE_QUARTER",
        "source_family": "IDX_OFFICIAL_XBRL",
        "source_name": "IDX filing",
        "source_url": "https://www.idx.co.id/example.xml",
        "currency": "IDR",
        "revenue": 1_000_000_000.0,
        "net_income": 100_000_000.0,
        "operating_cash_flow": 120_000_000.0,
        "total_assets": 2_000_000_000.0,
        "total_liabilities": 700_000_000.0,
        "equity": 1_300_000_000.0,
        "available_at": "2026-08-01T09:00:00+07:00",
        "source_verified": True,
    }])


def test_fundamental_period_end_does_not_imply_information_availability():
    history = _statement_history()

    before = build_fundamental_history_features(
        history,
        now="2026-07-15T16:30:00+07:00",
    )
    after = build_fundamental_history_features(
        history,
        now="2026-08-02T16:30:00+07:00",
    )

    assert len(before) == 1
    assert before.iloc[0]["fundamental_point_in_time_state"] == "NO_EVIDENCE_AVAILABLE_AS_OF"
    assert before.iloc[0]["fundamental_data_grade"] == "D"

    assert len(after) == 1
    assert after.iloc[0]["fundamental_point_in_time_state"] == "AVAILABLE_AS_OF"
    assert pd.Timestamp(after.iloc[0]["fundamental_latest_available_at"]) == pd.Timestamp("2026-08-01T09:00:00")


def test_legacy_statement_without_availability_is_not_backdated_to_period_end():
    normalized = normalize_fundamental_history(
        _statement_history().drop(columns=["available_at"])
    )
    row = normalized.iloc[0]

    assert pd.notna(row["available_at"])
    assert pd.Timestamp(row["available_at"]) > pd.Timestamp(row["period_end"])
    assert "AVAILABILITY_TIMESTAMP_ASSIGNED_AT_INGESTION" in str(row["validation_flags"])


def test_setup_plan_exposes_separate_origin_confirmation_and_execution_timestamps():
    plan = SetupPlan(
        ticker="TEST.JK",
        setup="BREAKOUT_RETEST",
        detected=True,
        setup_score=80.0,
        signal_date=pd.Timestamp("2026-08-03"),
        confirmation_date=pd.Timestamp("2026-08-05"),
        executable_date=pd.Timestamp("2026-08-06"),
    )
    payload = plan.to_dict()

    assert payload["signal_date"] < payload["confirmation_date"] < payload["executable_date"]
    assert payload["timestamp_semantics"] == "SIGNAL_DATE_IS_SETUP_ORIGIN_NOT_ENTRY"
