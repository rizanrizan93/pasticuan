from __future__ import annotations

import numpy as np
import pandas as pd

from evidence_enrichment import enrich_fundamental_evidence
from narrative_engine import _event_record
from scanner import enrich_fundamentals_with_valuation
from scanner_database import _semantic_hash
from simple_focus import _future_component, _management_component


def check_financial_outcome_proxies_keep_truthful_lineage() -> None:
    source = pd.DataFrame([{
        "ticker": "TEST.JK",
        "history_revenue_growth": 0.18,
        "history_earnings_growth": 0.24,
        "history_revenue_growth_acceleration": 0.05,
        "history_earnings_growth_acceleration": 0.08,
        "history_roe": 0.19,
        "history_roic_proxy": 0.16,
        "history_operating_margin": 0.17,
        "history_cash_conversion": 1.05,
        "history_fcf_ttm": 120_000_000_000,
        "history_share_dilution_yoy": 0.01,
        "history_debt_equity": 0.42,
        "history_positive_earnings_ratio": 1.0,
        "history_positive_ocf_ratio": 1.0,
        "history_margin_stability": 0.82,
        "history_accruals_to_assets": 0.01,
        "history_leverage_change_yoy": -0.02,
    }])
    row = enrich_fundamental_evidence(source).iloc[0].to_dict()
    assert np.isfinite(row["forward_financial_capacity_score"])
    assert np.isfinite(row["management_execution_proxy_score"])
    assert row["quality_pillar_coverage_pct"] >= 80.0
    assert row["derived_evidence_provenance_state"] == (
        "PERIOD_ALIGNED_FINANCIAL_OUTCOME_PROXY_NOT_DIRECT_GUIDANCE_OR_BIOGRAPHY"
    )

    prefixed = {f"fund_{key}": value for key, value in row.items()}
    future_score, future_coverage, _ = _future_component(prefixed)
    management_score, management_coverage, _ = _management_component(prefixed)
    assert np.isfinite(future_score) and future_coverage >= 50.0
    assert np.isfinite(management_score) and management_coverage >= 50.0
    assert "fund_project_pipeline_score" not in prefixed


def check_ksei_point_in_time_valuation() -> None:
    now = pd.Timestamp("2026-08-10", tz="Asia/Jakarta")
    fundamentals = pd.DataFrame([{
        "ticker": "TEST.JK",
        "fundamental_primary_currency": "IDR",
        "latest_statement_date": "2026-03-31",
        "history_latest_fy_period": "2025-12-31",
        "history_net_income_ttm": 100_000_000_000,
        "history_fcf_ttm": 55_000_000_000,
        "history_ebitda_ttm": 150_000_000_000,
        "history_equity_latest": 500_000_000_000,
        "history_total_debt_latest": 120_000_000_000,
        "history_cash_latest": 70_000_000_000,
        "history_earnings_growth": 0.20,
        "history_positive_earnings_ratio": 1.0,
        "ksei_total_shares": 1_000_000_000,
        "ksei_shares_verified": True,
        "ksei_shares_observed_at": "2026-08-09T00:00:00+00:00",
        "fundamental_official_verified": True,
    }])
    prices = {"TEST.JK": pd.DataFrame(
        {"Close": [1_000.0]}, index=pd.to_datetime(["2026-08-07"])
    )}
    row = enrich_fundamentals_with_valuation(fundamentals, prices, now=now).iloc[0]
    assert row["valuation_market_cap_mode"] == "PRICE_TIMES_CURRENT_KSEI_TOTAL_SHARES"
    assert row["valuation_shares_basis"] == "KSEI_CURRENT_TOTAL_SHARES"
    assert row["market_cap"] == 1_000_000_000_000
    assert round(float(row["trailing_pe"]), 2) == 10.0
    assert bool(row["valuation_score_eligible"])


def check_current_idx_domain_is_official_without_manual_claim() -> None:
    event = _event_record(
        ticker="TEST.JK",
        headline="TEST menyampaikan laporan keuangan",
        source_url="https://www.idx.id/id/perusahaan-tercatat/laporan-keuangan",
        source_family="IDX_DISCLOSURE",
        official_verified=False,
        event_date="2026-08-08",
        detected_at="2026-08-09T00:00:00Z",
    )
    assert event is not None
    assert event["source_hostname"] == "www.idx.id"
    assert event["official_verified"] is True
    assert event["source_quality_score"] == 95.0


def check_semantic_hash_survives_jsonb_numeric_round_trip() -> None:
    left = {"ticker": "TEST.JK", "nested": {"count": 1, "ratio": 1.25}}
    right = {"ticker": "TEST.JK", "nested": {"count": 1.0, "ratio": 1.2500}}
    assert _semantic_hash(left) == _semantic_hash(right)


if __name__ == "__main__":
    checks = [
        check_financial_outcome_proxies_keep_truthful_lineage,
        check_ksei_point_in_time_valuation,
        check_current_idx_domain_is_official_without_manual_claim,
        check_semantic_hash_survives_jsonb_numeric_round_trip,
    ]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print("VALIDATION_V9_8_3_EVIDENCE_INTEGRITY=PASS")
