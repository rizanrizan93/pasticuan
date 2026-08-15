import pandas as pd

from release_contract import SCANNER_RELEASE_VERSION
from v9_dashboard import SCANNER_VERSION as DASHBOARD_SCANNER_VERSION, select_top_candidates
from v9_dashboard_legacy import SCANNER_VERSION as LEGACY_DASHBOARD_SCANNER_VERSION


def test_research_lane_uses_guarded_score_before_execution_status():
    leaders = pd.DataFrame([
        {
            "ticker": "TOTL.JK", "rank_eligible": True, "status": "WAIT",
            "ranking_score": 59.8, "score_coverage_pct": 54.1,
            "methodology_priority": 6, "silent_accumulation_score": 81.0,
        },
        {
            "ticker": "PBID.JK", "rank_eligible": True, "status": "RESEARCH_ONLY",
            "ranking_score": 65.1, "score_coverage_pct": 57.7,
            "methodology_priority": 6, "silent_accumulation_score": 50.7,
        },
        {
            "ticker": "TSPC.JK", "rank_eligible": True, "status": "RESEARCH_ONLY",
            "ranking_score": 64.7, "score_coverage_pct": 57.8,
            "methodology_priority": 6, "silent_accumulation_score": 71.1,
        },
        {
            "ticker": "KINO.JK", "rank_eligible": True, "status": "WAIT",
            "ranking_score": 61.9, "score_coverage_pct": 57.6,
            "methodology_priority": 6, "silent_accumulation_score": 60.6,
        },
    ])

    top = select_top_candidates(leaders, model="NEXT_LEADER", limit=3, lane="RESEARCH")

    assert top["ticker"].tolist() == ["PBID.JK", "TSPC.JK", "KINO.JK"]


def test_non_research_lane_retains_execution_status_priority():
    leaders = pd.DataFrame([
        {
            "ticker": "WATCH.JK", "portfolio_rank_eligible": True, "status": "WATCH",
            "ranking_score": 90.0,
        },
        {
            "ticker": "BUY.JK", "portfolio_rank_eligible": True, "status": "BUY_ZONE",
            "ranking_score": 75.0,
        },
    ])

    top = select_top_candidates(leaders, model="NEXT_LEADER", limit=2, lane="PORTFOLIO")

    assert top["ticker"].tolist() == ["BUY.JK", "WATCH.JK"]


def test_dashboard_modules_share_release_lineage():
    assert DASHBOARD_SCANNER_VERSION == SCANNER_RELEASE_VERSION
    assert LEGACY_DASHBOARD_SCANNER_VERSION == SCANNER_RELEASE_VERSION
