import numpy as np
import pandas as pd

from phase56_coverage_runtime_patch import _future_component, _merge_context, _technical_component


def test_technical_research_coverage_uses_observed_indicators_without_inventing_rr():
    row = {
        "sig_status": "WATCHLIST",
        "sig_last_price": 110.0,
        "sig_ema20": 105.0,
        "sig_ema50": 100.0,
        "sig_ema200": 90.0,
        "sig_adx14": 28.0,
        "sig_cmf20": 0.08,
        "sig_rsi14": 61.0,
        "sig_roc60": 12.0,
        "sig_relative_strength60": 72.0,
        "sig_rr1": np.nan,
    }
    score, coverage, basis = _technical_component(row)
    assert np.isfinite(score)
    assert coverage == 75.0
    assert score <= 68.0  # WATCHLIST state cap remains binding.
    assert "RR_MISSING_NO_EXECUTION_INFERENCE" in basis


def test_future_research_fallback_cannot_cross_direct_authorization_floor():
    def direct_only(_row):
        return np.nan, 0.0, "DIRECT_FORWARD_PENDING"

    row = {
        "nar_forward_project_pipeline_score": 80.0,
        "nar_forward_future_fundamental_impact_score": 90.0,
        "nar_forward_project_data_coverage_pct": 100.0,
    }
    score, coverage, basis = _future_component(direct_only, row)
    assert np.isfinite(score)
    assert score <= 68.0
    assert 0.0 < coverage < 40.0
    assert basis.startswith("RESEARCH_NON_QUORUM_FORWARD")


def test_direct_future_evidence_remains_authoritative():
    def direct_only(_row):
        return 84.0, 72.0, "DIRECT_FORWARD_VERIFIED"

    score, coverage, basis = _future_component(direct_only, {})
    assert score == 84.0
    assert coverage == 72.0
    assert basis == "DIRECT_FORWARD_VERIFIED"


def test_ownership_context_normalizes_bare_and_jk_tickers_without_score_relabel():
    frame = pd.DataFrame([{"ticker": "ADMR.JK", "ranking_score": 77.0}])
    context = {
        "ADMR.JK": {
            "ownership_public_context_coverage_pct": 100.0,
            "ownership_public_official_verified": False,
            "ownership_public_context_state": "CONTEXT_ONLY_NOT_REGULATORY_FREE_FLOAT",
        }
    }
    merged = _merge_context(frame, context)
    assert merged.loc[0, "ticker"] == "ADMR.JK"
    assert merged.loc[0, "ownership_public_context_coverage_pct"] == 100.0
    assert not bool(merged.loc[0, "ownership_public_official_verified"])
    assert merged.loc[0, "ranking_score"] == 77.0
