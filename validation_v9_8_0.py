from __future__ import annotations

import numpy as np
import pandas as pd

import macro_engine
import scanner
from real_money_guard import apply_real_money_authorization, fundamental_conviction_profile
from v9_dashboard import render_dashboard_html


def check_official_priority() -> None:
    frame = pd.DataFrame([
        {"ticker":"TEST.JK","period_end":"2026-06-30","period_type":"Q2","source_family":"YAHOO","source_verified":False,"revenue":200.0,"net_income":20.0,"operating_cash_flow":18.0,"total_assets":400.0,"total_liabilities":180.0,"equity":220.0,"total_debt":80.0},
        {"ticker":"TEST.JK","period_end":"2026-06-30","period_type":"Q2","source_family":"IDX_OFFICIAL_XBRL","source_verified":True,"revenue":100.0,"net_income":12.0,"operating_cash_flow":15.0,"total_assets":300.0,"total_liabilities":130.0,"equity":170.0,"total_debt":60.0},
    ])
    norm = scanner.normalize_fundamental_history(frame)
    picked = scanner._median_statement_periods(norm, annual=False)
    assert float(picked.iloc[-1]["revenue"]) == 100.0
    features = scanner._history_features_for_ticker("TEST.JK", norm, now="2026-08-08")
    assert features["fundamental_official_verified"] is True
    assert features["fundamental_official_source_coverage_pct"] >= 83.0
    assert features["fundamental_reconciliation_state"] == "OFFICIAL_PRIORITY_PROXY_MISMATCH"


def check_fundamental_caps() -> None:
    good = {
        "fund_fundamental_data_grade":"A", "fund_fundamental_history_coverage":95,
        "fund_fundamental_cashflow_statement_coverage_pct":100,
        "fund_fundamental_official_source_coverage_pct":100,
        "fund_fundamental_official_verified":True, "fund_fundamental_consensus_score":94,
        "fund_history_ocf_ttm":100, "fund_history_fcf_ttm":75, "fund_history_debt_equity":0.45,
        "fund_sector":"INDUSTRIALS",
    }
    weak = dict(good)
    weak.update({"fund_fundamental_data_grade":"C", "fund_history_ocf_ttm":np.nan, "fund_history_fcf_ttm":np.nan, "fund_history_debt_equity":2.2, "fund_fundamental_official_verified":False})
    a = fundamental_conviction_profile(good)
    b = fundamental_conviction_profile(weak)
    assert a["fundamental_conviction_cap"] >= 95
    assert b["fundamental_conviction_cap"] <= 68
    assert "OCF_MISSING" in b["fundamental_score_cap_reason"]


def check_blended_market_context() -> None:
    idx = pd.date_range("2025-01-01", periods=240, freq="B")
    benchmark = pd.DataFrame({"Close": np.linspace(100, 125, len(idx))}, index=idx)
    prepared = {}
    for i in range(100):
        slope = 1.2 if i < 70 else 0.8
        prepared[f"T{i}.JK"] = pd.DataFrame({"Close": np.linspace(100, 100*slope, len(idx))}, index=idx)
    out = macro_engine.build_macro_regime(benchmark=benchmark, prepared=prepared, fundamentals=pd.DataFrame({"ticker":["T0.JK"],"sector":["INDUSTRIALS"]}), macro_series={})
    row = out.snapshot.iloc[0]
    assert row["market_context_provenance_state"] == "BLENDED_IHSG_AND_UNIVERSE_BREADTH"
    assert np.isfinite(float(row["market_context_score"]))
    assert out.issuer_map.iloc[0]["market_regime"] == row["market_regime"]


def check_authorization_separate_from_ranking() -> None:
    base = pd.DataFrame([{
        "ticker":"READY.JK", "status":"BUY_ZONE", "v9_next_leader_score":84,
        "score_coverage_pct":90, "business_quality_score":78, "future_fundamental_score":72,
        "technical_readiness_score":75, "adtv20_idr":2_000_000_000, "rr1":2.1,
        "production_gate_pass":True, "methodology_gate_pass":True,
        "distribution_risk_score":18, "market_regime":"SELECTIVE_RISK_ON",
        "market_context_coverage_pct":90, "fundamental_data_quality_score":88,
        "fundamental_cashflow_state":"OCF_FCF_POSITIVE", "fundamental_leverage_risk_state":"BALANCE_SHEET_CAPACITY_OK",
        "fundamental_official_verified":True, "fundamental_official_source_coverage_pct":85,
        "independent_price_verified":False, "recommended_allocation_idr":1_000_000, "recommended_lots":10, "entry":1000, "stop_loss":950,
    }])
    manual = apply_real_money_authorization(base, model="NEXT_LEADER", account_size_idr=5_000_000)
    assert manual.iloc[0]["real_money_authorization_state"] == "REAL_MONEY_MANUAL_CONFIRMATION_REQUIRED"
    assert manual.iloc[0]["recommended_allocation_idr"] == 0
    direct_source = base.copy(); direct_source.loc[0,"independent_price_verified"] = True
    direct = apply_real_money_authorization(direct_source, model="NEXT_LEADER", account_size_idr=5_000_000)
    assert direct.iloc[0]["real_money_authorization_state"] == "REAL_MONEY_DIRECT_VERIFIED_READY"
    assert direct.iloc[0]["recommended_allocation_idr"] == 700_000
    assert direct.iloc[0]["recommended_lots"] == 7
    assert direct.iloc[0]["real_money_risk_budget_idr"] == 37_500
    # ranking/status remains a thesis state, independent of order authorization.
    assert manual.iloc[0]["status"] == "BUY_ZONE"



def check_financial_sector_guard() -> None:
    bank = {
        "fund_fundamental_data_grade":"A", "fund_fundamental_history_coverage":92,
        "fund_fundamental_cashflow_statement_coverage_pct":0,
        "fund_fundamental_official_source_coverage_pct":83,
        "fund_fundamental_official_verified":True, "fund_fundamental_consensus_score":90,
        "fund_history_ocf_ttm":np.nan, "fund_history_fcf_ttm":np.nan,
        "fund_history_debt_equity":6.0, "fund_sector":"FINANCIALS",
    }
    out = fundamental_conviction_profile(bank)
    assert out["fundamental_conviction_cap"] >= 90
    assert out["fundamental_cashflow_state"] == "SECTOR_SPECIFIC_FINANCIAL_CASHFLOW_NOT_PRIMARY"
    assert out["fundamental_leverage_risk_state"] == "SECTOR_SPECIFIC_FINANCIAL_LEVERAGE"


def check_v14_contract() -> None:
    from pathlib import Path
    migration = Path("database/migration_v14_guarded_real_money.sql").read_text()
    db = Path("scanner_database.py").read_text()
    assert "real_money_authorization_state" in migration
    assert "fundamental_reconciliation_state" in migration
    assert "MIGRATION_REQUIRED_V14" in db
    assert "HEALTHY_V14_GUARDED_REAL_MONEY" in db

def check_dashboard_auth() -> None:
    row = pd.DataFrame([{
        "dashboard_rank":1,"ticker":"TEST.JK","sector":"ENERGY","status":"BUY_ZONE","final_score":82,
        "score_coverage_pct":90,"business_quality_score":80,"future_fundamental_score":78,"valuation_mos_score":70,
        "management_capital_score":72,"issuer_macro_alignment_score":68,"narrative_flow_score":76,"silent_accumulation_score":74,
        "technical_readiness_score":70,"accumulation_dominance_pct":75,"inventory_multi_horizon_score":73,"distribution_risk_score":20,
        "reaccumulation_quality_score":72,"inventory_lifecycle":"INVENTORY_COLLECTION","real_money_authorization_state":"REAL_MONEY_BLOCKED",
        "real_money_authorization_blockers":"OCF_MISSING","real_money_risk_budget_cap_pct":0.75,"market_regime":"SELECTIVE_RISK_ON",
    }])
    html = render_dashboard_html(row, model="NEXT_LEADER", scan_id="v980")
    assert "REAL MONEY AUTHORIZATION" in html
    assert "REAL MONEY BLOCKED" in html


def main() -> None:
    checks = [
        check_official_priority, check_fundamental_caps, check_blended_market_context,
        check_authorization_separate_from_ranking, check_financial_sector_guard,
        check_v14_contract, check_dashboard_auth,
    ]
    for fn in checks:
        fn(); print("PASS", fn.__name__)
    print("VALIDATION_V9_8_0=PASS")


if __name__ == "__main__":
    main()
