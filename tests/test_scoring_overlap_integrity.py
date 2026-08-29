import numpy as np
import pandas as pd

from narrative_engine import _verified_forward_profile
from simple_focus import (
    _business_component,
    _future_component,
    _management_component,
    _technical_component,
    _weighted_final,
)


def _realised_row(revenue_growth: float, earnings_growth: float) -> dict[str, float]:
    return {
        "fund_revenue_growth": revenue_growth,
        "fund_earnings_growth": earnings_growth,
        # Historical-outcome derivatives cannot authorize the direct Future
        # pillar, regardless of how complete they look.
        "fund_forward_financial_capacity_score": 72.0,
        "fund_forward_financial_capacity_coverage_pct": 100.0,
        "fund_reinvestment_runway_pillar": 68.0,
        "fund_reinvestment_runway_coverage_pct": 100.0,
        "fund_forward_growth_persistence_score": 74.0,
        "fund_forward_growth_persistence_coverage_pct": 100.0,
    }


def test_realised_growth_changes_business_but_cannot_score_future_pillar():
    weak = _realised_row(-0.20, -0.30)
    strong = _realised_row(0.35, 0.50)

    assert _business_component(strong)[0] > _business_component(weak)[0]
    assert np.isnan(_future_component(strong)[0])
    assert _future_component(strong)[1] == 0.0
    assert _future_component(strong)[:2] == _future_component(weak)[:2]


def test_verified_direct_forward_evidence_scores_with_measured_coverage():
    profile = _verified_forward_profile([{
        "project_pipeline_score_observed": 70.0,
        "future_fundamental_impact_score_observed": 80.0,
        "project_data_coverage": 92.0,
        "project_source_quorum_verified": True,
        "project_source_urls": ["https://www.idx.co.id/id/perusahaan-tercatat/keterbukaan-informasi"],
        "project_source_families": "IDX_DISCLOSURE",
        "last_verified_at": "2026-08-01T00:00:00Z",
    }], pd.Timestamp("2026-08-15T00:00:00Z"))
    row = {f"nar_{key}": value for key, value in profile.items()}

    score, coverage, evidence = _future_component(row)
    assert score == 75.5
    assert coverage == 92.0
    assert "PROJECT_PIPELINE_AND_IMPACT" in evidence


def test_forward_score_without_source_lineage_remains_missing():
    profile = _verified_forward_profile([{
        "project_pipeline_score": 99.0,
        "future_fundamental_impact_score": 99.0,
        "project_data_coverage": 100.0,
        "project_source_quorum_verified": True,
        "last_verified_at": "2026-08-01T00:00:00Z",
    }], pd.Timestamp("2026-08-15T00:00:00Z"))
    row = {f"nar_{key}": value for key, value in profile.items()}

    assert profile["forward_evidence_state"] == "NOT_SCORED_FORWARD_LINEAGE_INCOMPLETE"
    assert np.isnan(_future_component(row)[0])
    assert _future_component(row)[1] == 0.0


def test_realised_growth_does_not_move_management_capital_pillar():
    direct = {
        "nar_issuer_action_alignment_effective_score": 68.0,
        "nar_issuer_action_alignment_coverage_pct": 80.0,
        "fund_insider_ownership_pct": 12.0,
        "fund_history_share_dilution_yoy": 0.01,
    }
    weak = {**direct, **_realised_row(-0.20, -0.30)}
    strong = {**direct, **_realised_row(0.35, 0.50)}

    assert _management_component(strong)[:2] == _management_component(weak)[:2]


def test_no_dilution_alone_cannot_claim_perfect_management_quality():
    score, coverage, state = _management_component({
        "fund_history_share_dilution_yoy": 0.0,
    })

    assert np.isnan(score)
    assert coverage == 0.0
    assert state == "MANAGEMENT_EVIDENCE_INSUFFICIENT"


def test_weighted_final_shrinks_each_component_by_its_own_coverage():
    score, coverage = _weighted_final(
        {"thin": (100.0, 10.0), "complete": (50.0, 100.0)},
        {"thin": 0.5, "complete": 0.5},
        min_coverage=50.0,
    )

    assert score == 52.5
    assert coverage == 55.0



def test_setup_state_caps_technical_score_instead_of_becoming_a_second_vote():
    base = {
        "sig_quality_score": 90.0,
        "sig_momentum_score": 9.0,
        "sig_rr1": 2.0,
    }
    ready = _technical_component({**base, "sig_setup_status": "EXECUTION_READY"})
    watch = _technical_component({**base, "sig_setup_status": "WATCHLIST"})

    assert ready[0] > watch[0]
    assert watch[0] <= 68.0
    assert "STATE_CAP:WATCHLIST" in watch[2]
    assert "SETUP_STATE" not in watch[2]
