from __future__ import annotations

import pandas as pd

from evidence_governance import (
    ProviderNegativeCache,
    apply_three_rank_contract,
    calibrate_guardrails_walk_forward,
    select_enrichment_shortlist,
    validate_official_evidence,
)


def test_verified_evidence_requires_https_entity_date_and_quorum():
    ok = validate_official_evidence(
        source_url="https://issuer.example/presentation.pdf",
        source_urls=["https://issuer.example/presentation.pdf", "https://issuer.example/disclosure"],
        evidence_date="2026-06-19",
        entity_match_verified=True,
        source_verified=True,
        quorum_required=True,
    )
    assert ok["evidence_production_valid"] is True
    assert ok["source_quorum_count"] == 2

    bad = validate_official_evidence(
        source_url="http://issuer.example/presentation.pdf",
        source_urls=["http://issuer.example/presentation.pdf"],
        evidence_date=None,
        entity_match_verified=False,
        source_verified=True,
        quorum_required=True,
    )
    assert bad["evidence_production_valid"] is False
    assert "HTTPS_SOURCE_MISSING" in bad["evidence_validation_reasons"]
    assert "SOURCE_QUORUM_NOT_MET" in bad["evidence_validation_reasons"]


def test_three_rank_contract_keeps_raw_guarded_and_real_money_separate():
    frame = pd.DataFrame({
        "ticker": ["AAA.JK", "BBB.JK", "CCC.JK"],
        "raw_ranking_score": [90.0, 88.0, 85.0],
        "ranking_score": [82.0, 86.0, 80.0],
        "rank_eligible": [True, True, True],
        "real_money_authorization_pass": [False, True, False],
    })
    out = apply_three_rank_contract(
        frame,
        raw_score_col="raw_ranking_score",
        guarded_score_col="ranking_score",
        research_eligible_col="rank_eligible",
        guarded_eligible_col="rank_eligible",
        production_eligible_cols=("real_money_authorization_pass",),
    ).set_index("ticker")
    assert int(out.loc["AAA.JK", "raw_research_rank"]) == 1
    assert int(out.loc["BBB.JK", "guarded_decision_priority_rank"]) == 1
    assert int(out.loc["BBB.JK", "production_real_money_rank"]) == 1
    assert pd.isna(out.loc["AAA.JK", "production_real_money_rank"])


def test_oos_calibration_refuses_unmatured_outcomes():
    outcomes = pd.DataFrame({
        "signal_date": ["2026-08-12", "2026-08-13"],
        "raw_ranking_score": [80.0, 82.0],
        "forward_return_20d": [None, None],
        "outcome_verified": [False, False],
    })
    result = calibrate_guardrails_walk_forward(outcomes)
    assert result["active"] is False
    assert "INSUFFICIENT" in result["calibration_state"]


def test_enrichment_shortlist_prefers_research_relevance_not_missingness_alone():
    frame = pd.DataFrame({
        "ticker": ["TOP.JK", "MID.JK", "LOW.JK"],
        "ranking_score": [90.0, 70.0, 20.0],
        "forward_source_quorum_verified": [False, True, False],
    })
    selected = select_enrichment_shortlist(frame, limit=2)
    assert selected[0] == "TOP.JK"
    assert "LOW.JK" not in selected


def test_provider_negative_cache_is_provider_specific_and_clears_on_success():
    cache = ProviderNegativeCache()
    cache.record_failure("IDX", "FORWARD", "AAA.JK", "TIMEOUT")
    assert cache.should_skip("IDX", "FORWARD", "AAA.JK") is True
    assert cache.should_skip("KSEI", "FORWARD", "AAA.JK") is False
    cache.record_success("IDX", "FORWARD", "AAA.JK")
    assert cache.should_skip("IDX", "FORWARD", "AAA.JK") is False
