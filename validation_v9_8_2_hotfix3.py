from __future__ import annotations

import pathlib
import py_compile

import numpy as np
import pandas as pd

from fundamental_calibration import (
    reporting_refresh_profile,
    latest_growth_profile,
    classify_thesis_archetype,
    maintenance_refresh_priority,
)
from real_money_guard import fundamental_conviction_profile, apply_real_money_authorization
from scanner import enrich_fundamentals_with_history
from simple_focus import NEXT_LEADER_WEIGHTS, SWING_WEIGHTS, build_next_leaders
from v9_dashboard import render_dashboard_html, select_top_candidates

ROOT = pathlib.Path(__file__).resolve().parent


def check_compile() -> None:
    for path in ROOT.glob('*.py'):
        py_compile.compile(str(path), doraise=True)


def check_weights_frozen() -> None:
    assert NEXT_LEADER_WEIGHTS.__dict__ == {
        'business_quality':0.25,'future_fundamental':0.20,'valuation_mos':0.15,
        'management_capital':0.10,'macro_sector':0.10,'narrative_flow':0.15,
        'technical_readiness':0.05,
    }
    assert SWING_WEIGHTS.__dict__ == {
        'technical_execution':0.40,'macro_sector':0.15,'narrative_flow':0.25,
        'business_quality':0.10,'risk_data':0.10,
    }


def check_calendar_refresh_window() -> None:
    q1 = reporting_refresh_profile(
        {'fundamental_history_latest_period':'2026-03-31'}, now='2026-08-09T16:00:00+07:00'
    )
    assert q1['fundamental_refresh_state'] == 'REFRESH_WINDOW'
    assert q1['fundamental_refresh_due'] is True
    q2 = reporting_refresh_profile(
        {'fundamental_history_latest_period':'2026-06-30'}, now='2026-08-09T16:00:00+07:00'
    )
    assert q2['fundamental_refresh_state'] == 'CURRENT'
    assert q2['fundamental_refresh_due'] is False


def check_growth_latest_period_overrides_proxy() -> None:
    row = {
        'fund_history_revenue_growth':-0.26,
        'fund_history_earnings_growth':-0.36,
        'fund_revenue_growth_snapshot':0.36,
        'fund_earnings_growth_snapshot':0.65,
        'fund_history_prior_revenue_growth':0.10,
        'fund_history_prior_earnings_growth':0.12,
    }
    profile = latest_growth_profile(row)
    assert profile['fundamental_latest_revenue_growth'] == -0.26
    assert profile['fundamental_latest_earnings_growth'] == -0.36
    assert profile['fundamental_trend_state'] == 'FUNDAMENTAL_DETERIORATION'
    assert 'REVENUE_GROWTH_SIGN_CONFLICT' in profile['fundamental_growth_conflict_state']
    assert 'EARNINGS_GROWTH_SIGN_CONFLICT' in profile['fundamental_growth_conflict_state']



def check_history_override_preserves_proxy_audit() -> None:
    base = pd.DataFrame([{
        'ticker':'SUNI.JK','revenue_growth':0.35,'earnings_growth':0.65,
        'fundamental_score':70,'fundamental_coverage':80,'latest_statement_date':'2026-03-31','sector':'ENERGY',
    }])
    rows = []
    for d, revenue, earnings in [
        ('2025-06-30',100,20),('2025-09-30',110,22),('2025-12-31',120,24),
        ('2026-03-31',130,26),('2026-06-30',74,12.8),
    ]:
        rows.append({
            'ticker':'SUNI.JK','period_end':d,'period_type':'Q','source_family':'YAHOO','source_verified':False,'currency':'IDR',
            'revenue':revenue,'net_income':earnings,'total_assets':500,'equity':300,'total_debt':100,'cash':50,
            'operating_income':earnings*1.3,'gross_profit':revenue*.3,'operating_cash_flow':earnings*1.1,'capex':-5,'shares_outstanding':100,
        })
    out = enrich_fundamentals_with_history(base, pd.DataFrame(rows), now='2026-08-09')
    row = out.iloc[0]
    assert abs(float(row['revenue_growth_snapshot']) - 0.35) < 1e-9
    assert abs(float(row['earnings_growth_snapshot']) - 0.65) < 1e-9
    assert abs(float(row['revenue_growth']) - (-0.26)) < 1e-9
    assert abs(float(row['earnings_growth']) - (-0.36)) < 1e-9
    assert str(pd.Timestamp(row['latest_statement_date']).date()) == '2026-06-30'

def check_benchmark_thesis_archetypes() -> None:
    mark = {
        'fund_sector':'INDUSTRIALS','fund_history_revenue_growth':0.50,'fund_history_earnings_growth':0.64,
        'fund_history_prior_revenue_growth':0.18,'fund_history_prior_earnings_growth':0.22,
        'fund_roe':0.36,'fund_debt_equity':0.004,'fund_history_cash_conversion':1.0,
    }
    assert classify_thesis_archetype(mark, business_score=80, future_score=55) == 'GROWTH_COMPOUNDER'

    elsa = {
        'fund_sector':'ENERGY','fund_history_revenue_growth':0.09,'fund_history_earnings_growth':0.29,
        'fund_roe':0.15,'fund_debt_equity':0.3,'fund_history_cash_conversion':0.8,
    }
    assert classify_thesis_archetype(elsa, business_score=70, future_score=60) == 'CYCLICAL_RECOVERY'

    bksl = {
        'fund_sector':'PROPERTY & REAL ESTATE','fund_history_revenue_growth':0.54,'fund_history_earnings_growth':5.0,
        'fund_history_prior_revenue_growth':-0.10,'fund_history_prior_earnings_growth':-0.30,
        'fund_roe':0.05,'fund_debt_equity':0.8,
    }
    # Extreme earnings base gets an explicit review label instead of being
    # mistaken for a high-quality compounder.
    assert classify_thesis_archetype(bksl, business_score=68, future_score=58) == 'BASE_EFFECT_REVIEW'

    suni = {
        'fund_sector':'ENERGY','fund_history_revenue_growth':-0.26,'fund_history_earnings_growth':-0.36,
        'fund_roe':0.10,'fund_debt_equity':0.4,
    }
    assert classify_thesis_archetype(suni, business_score=55, future_score=60) == 'FUNDAMENTAL_DETERIORATION'


def check_refresh_priority_bounded_lane() -> None:
    due_row = {'sector':'ENERGY','latest_statement_date':'2026-03-31','fundamental_score_eligible':True}
    fresh_row = {'sector':'ENERGY','latest_statement_date':'2026-06-30','fundamental_score_eligible':True}
    due = maintenance_refresh_priority(due_row, history_count=8, now='2026-08-09T16:00:00+07:00')
    fresh = maintenance_refresh_priority(fresh_row, history_count=8, now='2026-08-09T16:00:00+07:00')
    assert due[0] == 0
    assert fresh[0] == 9


def check_conviction_cap_and_authorization() -> None:
    row = {
        'fundamental_data_grade':'A','fundamental_history_coverage':90,
        'fundamental_cashflow_statement_coverage_pct':100,'fundamental_official_source_coverage_pct':80,
        'fundamental_official_verified':True,'fundamental_consensus_score':90,
        'history_ocf_ttm':100,'history_fcf_ttm':80,'history_debt_equity':0.2,
        'latest_statement_date':'2026-03-31',
        'history_revenue_growth':0.30,'history_earnings_growth':0.40,
    }
    cap = fundamental_conviction_profile(row)
    assert cap['fundamental_refresh_state'] == 'REFRESH_WINDOW'
    assert cap['fundamental_conviction_cap'] <= 84

    auth_frame = pd.DataFrame([{
        **cap,
        'status':'BUY_ZONE','production_gate_pass':True,'methodology_gate_pass':True,
        'distribution_risk_score':10,'independent_price_verified':True,
        'market_regime':'SELECTIVE_RISK_ON','market_context_coverage_pct':100,
        'v9_next_leader_score':80,'score_coverage_pct':90,'business_quality_score':75,
        'future_fundamental_score':70,'technical_readiness_score':75,'adtv20_idr':1e9,
        'rr1':2.0,'entry':100,'stop_loss':90,
    }])
    out = apply_real_money_authorization(auth_frame, model='NEXT_LEADER', account_size_idr=5_000_000, requested_risk_budget_pct=0.5)
    assert out.iloc[0]['real_money_authorization_state'] == 'REAL_MONEY_MANUAL_CONFIRMATION_REQUIRED'
    assert 'LATEST_REPORT_REFRESH_REQUIRED' in out.iloc[0]['real_money_manual_checks']


def check_dashboard_marks_research_swing() -> None:
    frame = pd.DataFrame([{
        'ticker':'TEST.JK','rank_eligible':True,'status':'RESEARCH_ONLY','ranking_score':70,
        'score_coverage_pct':70,'real_money_authorization_state':'REAL_MONEY_BLOCKED',
        'thesis_archetype':'TURNAROUND_RECOVERY','fundamental_refresh_state':'REFRESH_WINDOW',
        'fundamental_trend_state':'TURNAROUND_RECOVERY','inventory_lifecycle':'INVENTORY_COLLECTION',
    }])
    top = select_top_candidates(frame, model='SWING_READY', limit=3)
    html = render_dashboard_html(top, model='SWING_READY', scan_id='calibration')
    assert 'SWING WATCH / RESEARCH' in html
    assert 'REPORT REFRESH DUE' in html


def main() -> None:
    checks = [
        check_compile, check_weights_frozen, check_calendar_refresh_window,
        check_growth_latest_period_overrides_proxy, check_history_override_preserves_proxy_audit, check_benchmark_thesis_archetypes,
        check_refresh_priority_bounded_lane, check_conviction_cap_and_authorization,
        check_dashboard_marks_research_swing,
    ]
    for fn in checks:
        fn(); print('PASS', fn.__name__)
    print('VALIDATION_V9_8_2_HOTFIX3=PASS')


if __name__ == '__main__':
    main()
