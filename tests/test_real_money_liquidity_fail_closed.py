from __future__ import annotations

import pandas as pd

from real_money_guard import apply_real_money_authorization


def _strong_leader(**overrides):
    row = {
        "ticker": "TEST.JK",
        "status": "BUY_ZONE",
        "production_gate_pass": True,
        "methodology_gate_pass": True,
        "distribution_risk_score": 10.0,
        "market_regime": "RISK_ON",
        "market_context_coverage_pct": 100.0,
        "fundamental_cashflow_state": "OCF_FCF_POSITIVE",
        "fundamental_leverage_risk_state": "BALANCE_SHEET_CAPACITY_OK",
        "fundamental_data_quality_score": 85.0,
        "fundamental_refresh_state": "CURRENT",
        "fundamental_trend_state": "FUNDAMENTAL_STABLE",
        "fundamental_growth_conflict_state": "NONE",
        "fundamental_extreme_earnings_base_review": False,
        "fundamental_official_verified": True,
        "fundamental_official_source_coverage_pct": 90.0,
        "independent_price_verified": True,
        "v9_next_leader_score": 82.0,
        "score_coverage_pct": 90.0,
        "business_quality_score": 80.0,
        "future_fundamental_score": 75.0,
        "technical_readiness_score": 78.0,
        "rr1": 2.0,
        "entry": 1000.0,
        "stop_loss": 950.0,
    }
    row.update(overrides)
    return row


def test_unknown_adtv_requires_manual_liquidity_confirmation():
    out = apply_real_money_authorization(
        pd.DataFrame([_strong_leader()]),
        model="NEXT_LEADER",
        account_size_idr=5_000_000,
    ).iloc[0]

    assert out["real_money_authorization_state"] == "REAL_MONEY_MANUAL_CONFIRMATION_REQUIRED"
    assert bool(out["real_money_authorization_pass"]) is False
    assert "LIQUIDITY_ADTV_CONFIRMATION" in out["real_money_manual_checks"]


def test_sub_250m_adtv_is_hard_blocked():
    out = apply_real_money_authorization(
        pd.DataFrame([_strong_leader(adtv20_idr=249_999_999.0)]),
        model="NEXT_LEADER",
        account_size_idr=5_000_000,
    ).iloc[0]

    assert out["real_money_authorization_state"] == "REAL_MONEY_BLOCKED"
    assert bool(out["real_money_authorization_pass"]) is False
    assert "LIQUIDITY<250M_ADTV" in out["real_money_authorization_blockers"]
