import pandas as pd

from simple_focus import (
    NEXT_LEADER_WEIGHTS,
    _apply_decision_priority_guardrail,
    _coverage_gap_profile,
)


def test_coverage_gap_profile_explains_fixed_denominator_shortfall():
    components = {
        "business_quality": (70.0, 75.74),
        "future_fundamental": (float("nan"), 0.0),
        "valuation_mos": (70.0, 74.65),
        "management_capital": (60.0, 1.69),
        "macro_sector": (65.0, 80.0),
        "narrative_flow": (60.0, 67.52),
        "technical_readiness": (60.0, 67.08),
    }

    profile = _coverage_gap_profile(components, NEXT_LEADER_WEIGHTS.__dict__)

    assert profile["coverage_gap_pct"] == 48.2
    assert profile["coverage_primary_gap"] == "FUTURE_FUNDAMENTAL:20.0"
    assert profile["coverage_recovery_priority"].startswith(
        "FUTURE_FUNDAMENTAL:20.0 | MANAGEMENT_CAPITAL:9.8"
    )


def test_decision_guardrail_preserves_research_score_and_downranks_confirmed_conflicts():
    frame = pd.DataFrame([
        {
            "ticker": "CLEAN.JK", "ranking_score": 65.0, "research_score": 65.0,
            "ranking_score_state": "RESEARCH_SCORE", "methodology_gate_pass": True,
            "distribution_risk_score": 20.0, "anti_chase_gate": False,
            "technical_readiness_score": 70.0, "technical_readiness_coverage_pct": 80.0,
            "silent_accumulation_score": 70.0, "silent_accumulation_confidence": 80.0,
        },
        {
            "ticker": "ANTI.JK", "ranking_score": 65.0, "research_score": 65.0,
            "ranking_score_state": "RESEARCH_SCORE", "methodology_gate_pass": True,
            "distribution_risk_score": 20.0, "anti_chase_gate": True,
            "technical_readiness_score": 70.0, "technical_readiness_coverage_pct": 80.0,
            "silent_accumulation_score": 70.0, "silent_accumulation_confidence": 80.0,
        },
        {
            "ticker": "CONFLICT.JK", "ranking_score": 65.0, "research_score": 65.0,
            "ranking_score_state": "RESEARCH_SCORE", "methodology_gate_pass": True,
            "distribution_risk_score": 20.0, "anti_chase_gate": False,
            "technical_readiness_score": 10.0, "technical_readiness_coverage_pct": 65.0,
            "silent_accumulation_score": 10.0, "silent_accumulation_confidence": 65.0,
        },
        {
            "ticker": "UNKNOWN.JK", "ranking_score": 65.0, "research_score": 65.0,
            "ranking_score_state": "RESEARCH_SCORE", "methodology_gate_pass": True,
            "distribution_risk_score": 20.0, "anti_chase_gate": False,
            "technical_readiness_score": 10.0, "technical_readiness_coverage_pct": 0.0,
            "silent_accumulation_score": 10.0, "silent_accumulation_confidence": 0.0,
        },
    ])

    out = _apply_decision_priority_guardrail(frame).set_index("ticker")

    assert out.loc["CLEAN.JK", "ranking_score"] == 65.0
    assert out.loc["ANTI.JK", "ranking_score"] == 62.0
    assert out.loc["CONFLICT.JK", "ranking_score"] == 60.0
    assert out.loc["UNKNOWN.JK", "ranking_score"] == 65.0
    assert out["research_score"].eq(65.0).all()
    assert out.loc["CONFLICT.JK", "raw_ranking_score"] == 65.0
    assert out.loc["CONFLICT.JK", "ranking_guardrail_penalty_points"] == 5.0
    assert "WEAK_TECHNICAL_CONFIRMATION" in out.loc["CONFLICT.JK", "ranking_guardrail_reasons"]
    assert "WEAK_SMART_MONEY_CONFIRMATION" in out.loc["CONFLICT.JK", "ranking_guardrail_reasons"]
    assert "DECISION_PRIORITY_PENALTY:5.00" in out.loc["CONFLICT.JK", "primary_risk"]
    assert out.loc["UNKNOWN.JK", "ranking_guardrail_state"] == "CLEAN"


def test_distribution_block_is_research_visible_but_decision_downranked():
    frame = pd.DataFrame([{
        "ticker": "DIST.JK", "ranking_score": 66.0, "research_score": 66.0,
        "ranking_score_state": "RESEARCH_SCORE", "methodology_gate_pass": False,
        "distribution_risk_score": 72.0, "anti_chase_gate": False,
        "technical_readiness_score": 70.0, "technical_readiness_coverage_pct": 80.0,
        "silent_accumulation_score": 70.0, "silent_accumulation_confidence": 80.0,
    }])

    row = _apply_decision_priority_guardrail(frame).iloc[0]

    assert row["research_score"] == 66.0
    assert row["raw_ranking_score"] == 66.0
    assert row["ranking_score"] == 58.0
    assert row["distribution_penalty_points"] == 8.0
    assert row["distribution_evidence_state"] == "DISTRIBUTION_BLOCK"
    assert row["ranking_score_state"] == "RESEARCH_SCORE_GUARDED"
