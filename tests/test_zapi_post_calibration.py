import pandas as pd

from zapi_post_calibration import apply_super_foreign_shock_guard, enrich_super_shadow


def _row(**overrides):
    base = {
        "ticker": "TEST.JK",
        "zapi_foreign_flow_coverage_pct": 100.0,
        "zapi_foreign_net_participation_1d": 0.0,
        "zapi_foreign_net_participation_5d": 0.02,
        "zapi_foreign_net_participation_20d": 0.03,
        "zapi_foreign_state": "NET_ACCUMULATION",
        "real_money_authorization_state": "REAL_MONEY_DIRECT_VERIFIED_READY",
        "real_money_authorization_pass": True,
        "real_money_authorization_blockers": "",
        "real_money_manual_checks": "",
        "order_builder_eligible": True,
        "order_ready": True,
        "actionable_rank_eligible": True,
        "execution_gate_state": "PASS",
    }
    base.update(overrides)
    return base


def test_extreme_one_day_foreign_sell_requires_reclaim_without_hard_blocking():
    frame = pd.DataFrame([
        _row(
            ticker="OMED.JK",
            zapi_foreign_net_participation_1d=-0.254126,
            zapi_foreign_net_participation_5d=0.004359,
            zapi_foreign_net_participation_20d=0.001841,
            zapi_foreign_state="MIXED_NEUTRAL",
        )
    ])
    out = apply_super_foreign_shock_guard(frame).iloc[0]
    assert out["zapi_foreign_shock_state"] == "EXTREME_ONE_DAY_FOREIGN_SELL_SHOCK_RECLAIM_REQUIRED"
    assert out["real_money_authorization_state"] == "REAL_MONEY_MANUAL_CONFIRMATION_REQUIRED"
    assert bool(out["real_money_authorization_pass"]) is False
    assert bool(out["order_builder_eligible"]) is False
    assert bool(out["actionable_rank_eligible"]) is False
    assert "ZAPI_FOREIGN_SHOCK_REQUIRE_ABSORPTION_RECLAIM" in out["real_money_manual_checks"]
    assert not str(out.get("real_money_authorization_blockers", "")).strip()


def test_persistent_distribution_waits_for_stabilization():
    frame = pd.DataFrame([
        _row(
            zapi_foreign_net_participation_1d=-0.12,
            zapi_foreign_net_participation_5d=-0.03,
            zapi_foreign_net_participation_20d=-0.02,
            zapi_foreign_state="NET_DISTRIBUTION",
        )
    ])
    out = apply_super_foreign_shock_guard(frame).iloc[0]
    assert out["zapi_foreign_shock_state"] == "PERSISTENT_FOREIGN_DISTRIBUTION_WAIT"
    assert out["zapi_execution_flow_guard_state"] == "WAIT_FLOW_STABILIZATION_AND_RECLAIM"
    assert bool(out["order_ready"]) is False
    assert bool(out["actionable_rank_eligible"]) is False
    assert "ZAPI_FOREIGN_DISTRIBUTION_WAIT_STABILIZATION" in out["real_money_manual_checks"]


def test_strong_positive_foreign_flow_never_promotes_existing_nonready_row():
    frame = pd.DataFrame([
        _row(
            zapi_foreign_net_participation_1d=0.12,
            real_money_authorization_state="REAL_MONEY_MANUAL_CONFIRMATION_REQUIRED",
            real_money_authorization_pass=False,
            order_builder_eligible=False,
            order_ready=False,
            actionable_rank_eligible=False,
            execution_gate_state="BLOCKED",
        )
    ])
    out = apply_super_foreign_shock_guard(frame).iloc[0]
    assert out["zapi_foreign_shock_state"] == "STRONG_ONE_DAY_FOREIGN_ACCUMULATION_CONFIRMATION_ONLY"
    assert bool(out["real_money_authorization_pass"]) is False
    assert bool(out["order_builder_eligible"]) is False
    assert bool(out["actionable_rank_eligible"]) is False


def test_low_coverage_fail_soft_keeps_authorization():
    frame = pd.DataFrame([
        _row(
            zapi_foreign_flow_coverage_pct=40.0,
            zapi_foreign_net_participation_1d=-0.30,
        )
    ])
    out = apply_super_foreign_shock_guard(frame).iloc[0]
    assert out["zapi_foreign_shock_state"] == "ZAPI_INSUFFICIENT_OR_STALE_FOR_EXECUTION_GUARD"
    assert bool(out["real_money_authorization_pass"]) is True
    assert bool(out["order_builder_eligible"]) is True
    assert bool(out["actionable_rank_eligible"]) is True


def test_shadow_audit_persists_pre_post_delta_without_changing_score():
    frame = pd.DataFrame([
        {
            "ticker": "TEST.JK",
            "zapi_super_original_silent_score": 60.0,
            "zapi_super_post_silent_score": 69.0,
            "zapi_foreign_flow_coverage_pct": 100.0,
            "ranking_score": 72.5,
        }
    ])
    out = enrich_super_shadow(frame).iloc[0]
    assert float(out["zapi_shadow_pre_silent_score"]) == 60.0
    assert float(out["zapi_shadow_post_silent_score"]) == 69.0
    assert float(out["zapi_shadow_silent_score_delta"]) == 9.0
    assert out["zapi_shadow_calibration_state"] == "PENDING_FORWARD_OUTCOME"
    assert out["zapi_shadow_forward_horizons"] == "5D|20D|60D"
    assert float(out["ranking_score"]) == 72.5
