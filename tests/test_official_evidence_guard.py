import pandas as pd

from official_evidence_guard import canonicalize_official_fundamental_evidence


def test_embedded_idx_xbrl_cannot_be_downgraded_by_yahoo_flags():
    frame = pd.DataFrame([{
        "ticker": "MARK.JK",
        "fundamental_source_families": "YAHOO",
        "fundamental_official_verified": False,
        "fundamental_official_reference": False,
        "fundamental_official_source_coverage_pct": 0.0,
        "fundamental_cashflow_statement_coverage_pct": 0.0,
        "fundamental_current_period_official_verified": True,
        "idx_official_provenance_state": "IDX_OFFICIAL_XBRL_INSTANCE",
        "idx_official_source_url": "https://www.idx.id/example/instance.zip",
        "idx_official_period_end": "2026-06-30",
        "idx_official_revenue": 572_197_437_488,
        "idx_official_net_income": 183_915_039_637,
        "idx_official_assets": 1_024_525_000_893,
        "idx_official_equity": 891_753_183_643,
        "idx_official_ocf": 231_667_847_175,
        "idx_official_fcf_proxy": 215_742_296_161,
        "fundamental_source_count": 1,
    }])
    row = canonicalize_official_fundamental_evidence(frame).iloc[0]
    assert bool(row["fundamental_official_verified"]) is True
    assert bool(row["fundamental_official_reference"]) is True
    assert row["fundamental_official_source_coverage_pct"] == 100.0
    assert row["fundamental_cashflow_statement_coverage_pct"] == 100.0
    assert row["fundamental_source_families"] == "IDX_OFFICIAL_XBRL • YAHOO"
    assert row["fundamental_reconciliation_state"] == "OFFICIAL_PLUS_PUBLIC_CROSSCHECK"
    assert float(row["fundamental_source_count"]) >= 2.0


def test_partial_official_statement_gets_partial_not_fabricated_coverage():
    frame = pd.DataFrame([{
        "ticker": "TEST.JK",
        "fundamental_current_period_official_verified": True,
        "idx_official_provenance_state": "IDX_OFFICIAL_XBRL_INSTANCE",
        "idx_official_source_url": "https://www.idx.id/example/instance.zip",
        "idx_official_revenue": 100,
        "idx_official_net_income": 10,
    }])
    row = canonicalize_official_fundamental_evidence(frame).iloc[0]
    assert row["fundamental_official_source_coverage_pct"] == 33.3
    assert row["fundamental_cashflow_statement_coverage_pct"] == 0.0
