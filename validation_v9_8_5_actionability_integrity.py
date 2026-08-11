from __future__ import annotations

import numpy as np
import pandas as pd

from simple_focus import _technical_component, _research_accumulation_plan, _entry_zone_role
from v9_dashboard import select_top_candidates


def main() -> None:
    score, coverage, basis = _technical_component({
        "sig_setup_status": "WATCHLIST_ENTRY", "sig_quality_score": 90,
        "sig_momentum_score": 8, "sig_rr1": 2.0,
    })
    assert np.isfinite(score) and coverage == 100.0 and "SETUP_STATE" in basis

    research = _research_accumulation_plan({
        "sig_last_price": 700, "sig_atr14": 24, "sig_ema20": 682,
        "sig_ema50": 650, "sig_vwap20": 676, "sig_last_pivot_low": 640,
    })
    assert research["research_zone_state"] == "RESEARCH_ONLY_AVAILABLE"
    assert research["research_accumulation_zone_low"] < research["research_accumulation_zone_high"]

    role = _entry_zone_role({"sig_entry_low": 510, "sig_entry_high": 535}, 540, 520, 540)
    assert role == "PULLBACK_OBSERVATION_ZONE"

    swing = pd.DataFrame([
        {"ticker":"A.JK","rank_eligible":True,"production_rank_eligible":False,"actionable_rank_eligible":False,"status":"RESEARCH_ONLY","ranking_score":90,"v9_swing_score":np.nan},
        {"ticker":"B.JK","rank_eligible":True,"production_rank_eligible":True,"actionable_rank_eligible":True,"status":"ENTRY_PLAN_READY","ranking_score":70,"v9_swing_score":70},
    ])
    actionable = select_top_candidates(swing, model="SWING_READY", limit=3, lane="ACTIONABLE")
    assert actionable["ticker"].tolist() == ["B.JK"]

    leader = pd.DataFrame([
        {"ticker":"ILLIQ.JK","rank_eligible":True,"portfolio_rank_eligible":False,"status":"WATCH","ranking_score":90,"v9_next_leader_score":90},
        {"ticker":"LIQ.JK","rank_eligible":True,"portfolio_rank_eligible":True,"status":"WATCH","ranking_score":75,"v9_next_leader_score":75},
    ])
    portfolio = select_top_candidates(leader, model="NEXT_LEADER", limit=3, lane="PORTFOLIO")
    assert portfolio["ticker"].tolist() == ["LIQ.JK"]
    print("PASS v9.8.5 actionability integrity")


if __name__ == "__main__":
    main()
