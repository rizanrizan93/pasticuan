import base64
import re
from io import StringIO

import pandas as pd

from zapi_runtime_patch import _top3_csv_download_block


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
