import base64
import re
from io import StringIO

import pandas as pd

from zapi_runtime_patch import _actionable_selector_input, _top3_csv_download_block


def test_top3_html_embeds_downloadable_full_csv_with_zapi_audit_columns():
    top = pd.DataFrame([
        {
            "ticker": "OMED.JK",
            "ranking_score": 66.0,
            "zapi_foreign_flow_score": 72.1,
            "zapi_foreign_flow_coverage_pct": 100.0,
        },
        {
            "ticker": "MARK.JK",
            "ranking_score": 65.0,
            "zapi_foreign_flow_score": 61.0,
            "zapi_foreign_flow_coverage_pct": 95.0,
        },
    ])

    html = _top3_csv_download_block(top, model="NEXT_LEADER", scan_id="scan-123")

    assert "Download Top 3 CSV" in html
    assert 'download="idx_super_top3_next_leader_scan-123.csv"' in html
    assert "ZAPI audit fields included" in html

    match = re.search(r"base64,([A-Za-z0-9+/=]+)", html)
    assert match is not None
    decoded = base64.b64decode(match.group(1)).decode("utf-8-sig")
    restored = pd.read_csv(StringIO(decoded))

    assert restored["ticker"].tolist() == ["OMED.JK", "MARK.JK"]
    assert "zapi_foreign_flow_score" in restored.columns
    assert "zapi_foreign_flow_coverage_pct" in restored.columns


def test_actionable_swing_top3_requires_direct_authorization_and_execution_gate_pass():
    frame = pd.DataFrame([
        {
            "ticker": "BLOCKED.JK",
            "actionable_rank_eligible": True,
            "order_builder_eligible": False,
            "real_money_authorization_pass": False,
            "real_money_authorization_state": "REAL_MONEY_BLOCKED",
            "execution_gate_state": "BLOCKED",
        },
        {
            "ticker": "MANUAL.JK",
            "actionable_rank_eligible": True,
            "order_builder_eligible": True,
            "real_money_authorization_pass": False,
            "real_money_authorization_state": "REAL_MONEY_MANUAL_CONFIRMATION_REQUIRED",
            "execution_gate_state": "BLOCKED",
        },
        {
            "ticker": "SAFE.JK",
            "actionable_rank_eligible": True,
            "order_builder_eligible": True,
            "real_money_authorization_pass": True,
            "real_money_authorization_state": "REAL_MONEY_DIRECT_VERIFIED_READY",
            "execution_gate_state": "PASS",
        },
    ])

    filtered = _actionable_selector_input(frame, model="SWING_READY", lane="ACTIONABLE")

    assert filtered["ticker"].tolist() == ["SAFE.JK"]


def test_research_lane_is_not_filtered_by_order_builder_contract():
    frame = pd.DataFrame([
        {
            "ticker": "MMIX.JK",
            "order_builder_eligible": False,
            "real_money_authorization_state": "REAL_MONEY_BLOCKED",
        }
    ])

    filtered = _actionable_selector_input(frame, model="SWING_READY", lane="RESEARCH")

    assert filtered["ticker"].tolist() == ["MMIX.JK"]
