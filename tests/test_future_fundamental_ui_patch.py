from __future__ import annotations

import pandas as pd

import future_fundamental_ui_patch as ui


def test_missing_future_score_shows_evidence_state_not_zero_stars():
    import v9_dashboard

    ui.install()
    top = pd.DataFrame([
        {
            "dashboard_rank": 1,
            "ticker": "AAA.JK",
            "last_price": 100.0,
            "future_fundamental_score": float("nan"),
            "future_fundamental_coverage_pct": 0.0,
            "forward_collection_state": "FORWARD_CHECK_COMPLETED_NO_MATERIAL_EVENT",
        },
        {
            "dashboard_rank": 2,
            "ticker": "BBB.JK",
            "last_price": 200.0,
            "future_fundamental_score": float("nan"),
            "future_fundamental_coverage_pct": 0.0,
            "forward_collection_state": "MATERIAL_FORWARD_RESEARCH_EVIDENCE_FOUND",
        },
        {
            "dashboard_rank": 3,
            "ticker": "CCC.JK",
            "last_price": 300.0,
            "future_fundamental_score": 80.0,
            "future_fundamental_coverage_pct": 90.0,
        },
    ])
    html = v9_dashboard.render_dashboard_html(top, model="NEXT_LEADER")
    assert ">CHECKED</b>" in html
    assert ">RESEARCH</b>" in html
    # A genuinely scored row keeps normal report-card stars rather than being relabelled pending.
    assert html.count(">PENDING</b>") == 0


def test_lineage_incomplete_is_pending_not_fake_score():
    import v9_dashboard

    ui.install()
    top = pd.DataFrame([{
        "dashboard_rank": 1,
        "ticker": "AAA.JK",
        "future_fundamental_score": float("nan"),
        "future_fundamental_coverage_pct": 0.0,
        "forward_evidence_state": "NOT_SCORED_FORWARD_LINEAGE_INCOMPLETE",
    }])
    html = v9_dashboard.render_dashboard_html(top, model="NEXT_LEADER")
    assert ">PENDING</b>" in html
    assert "lineage/quorum" in html
