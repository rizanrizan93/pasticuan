import numpy as np
import pandas as pd

from decision_overlay import apply_execution_plan_integrity


def test_consumed_target_is_expired_without_changing_research_score():
    frame = pd.DataFrame([{
        "ticker": "MARK.JK", "status": "BUY_ZONE", "ranking_score": 65.8,
        "last_price": 1115.0, "entry": 1075.0, "entry_low": 1030.0, "entry_high": 1070.0,
        "trigger": 1075.0, "stop_loss": 940.0, "tp1": 1105.0, "tp2": 1125.0,
        "rr1": 0.22, "rr2": 0.37, "entry_zone_is_executable": True,
        "recommended_lots": 10, "recommended_allocation_idr": 1000000.0,
    }])
    out = apply_execution_plan_integrity(frame, model="NEXT_LEADER")
    row = out.iloc[0]
    assert row["ranking_score"] == 65.8
    assert row["execution_plan_integrity_state"] == "STALE_TARGET_ALREADY_REACHED"
    assert not bool(row["execution_plan_is_current"])
    assert row["status"] == "WAIT"
    assert np.isnan(row["entry_low"]) and np.isnan(row["tp1"]) and np.isnan(row["stop_loss"])
    assert int(row["recommended_lots"]) == 0


def test_stop_inside_executable_zone_is_invalidated():
    frame = pd.DataFrame([{
        "ticker": "DOOH.JK", "status": "ENTRY_PLAN_READY", "ranking_score": 57.1,
        "last_price": 250.0, "entry": 236.0, "entry_low": 216.0, "entry_high": 234.0,
        "trigger": 236.0, "stop_loss": 224.0, "tp1": 258.0, "tp2": 280.0,
        "rr1": 1.83, "rr2": 3.67, "entry_zone_is_executable": True,
        "order_builder_eligible": True, "order_ready": True,
    }])
    out = apply_execution_plan_integrity(frame, model="SWING_READY")
    row = out.iloc[0]
    assert row["execution_plan_integrity_state"] == "INVALID_ENTRY_STOP_GEOMETRY"
    assert row["status"] == "RESEARCH_ONLY"
    assert not bool(row["entry_zone_is_executable"])
    assert not bool(row["order_builder_eligible"])
    assert np.isnan(row["entry_low"]) and np.isnan(row["stop_loss"])


def test_current_valid_plan_is_preserved():
    frame = pd.DataFrame([{
        "ticker": "DGWG.JK", "status": "ENTRY_PLAN_READY", "ranking_score": 68.9,
        "last_price": 292.0, "entry": 294.0, "entry_low": 286.0, "entry_high": 296.0,
        "trigger": 294.0, "stop_loss": 270.0, "tp1": 306.0, "tp2": 326.0,
        "rr1": 0.5, "rr2": 1.33, "entry_zone_is_executable": True,
    }])
    out = apply_execution_plan_integrity(frame, model="SWING_READY")
    row = out.iloc[0]
    assert row["execution_plan_integrity_state"] == "CURRENT_PLAN"
    assert bool(row["execution_plan_is_current"])
    assert row["entry_low"] == 286.0
    assert row["tp1"] == 306.0
