from __future__ import annotations

"""Presentation-only Future Fundamental evidence-state labels.

A missing strict score is not the same as an acquisition failure. This patch
keeps score math untouched and replaces the report-card stars only when scoring
coverage is absent, so users can distinguish checked/no-event, research-only,
retryable acquisition failure, and official-evidence pending states.
"""

from functools import wraps
from html import escape
from typing import Any, Mapping
import re

import numpy as np
import pandas as pd

PATCH_VERSION = "1.0.0"


def _num(value: Any, default: float = np.nan) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


def _status(row: Mapping[str, Any]) -> tuple[str, str]:
    collection = str(row.get("forward_collection_state") or "").upper()
    future = str(row.get("future_fundamental_state") or row.get("forward_evidence_state") or "").upper()
    state = f"{collection}|{future}"
    if "MATERIAL_FORWARD_RESEARCH_EVIDENCE_FOUND" in state:
        return "RESEARCH", "Forward research evidence found; official production quorum is still pending."
    if "NO_MATERIAL_FORWARD_EVENT" in state or "CHECK_COMPLETED_NO_MATERIAL" in state:
        return "CHECKED", "Forward sources were checked; no material forward event was found in the active lookback."
    if "FAILED_RETRYABLE" in state or "COLLECTION_FAILED" in state:
        return "RETRY", "Forward source acquisition failed transiently and should be retried."
    if "LINEAGE_INCOMPLETE" in state:
        return "PENDING", "Forward evidence exists but official lineage/quorum is incomplete."
    return "PENDING", "Strict Future Fundamental evidence/coverage is not yet sufficient for scoring."


def _should_show_state(row: Mapping[str, Any]) -> bool:
    score = _num(row.get("future_fundamental_score"))
    coverage = _num(row.get("future_fundamental_coverage_pct"), 0.0)
    return not np.isfinite(score) or coverage <= 0.0


def _replace_report_row(html: str, *, cursor: int, label: str, detail: str) -> tuple[str, int]:
    pattern = re.compile(r'<div class="v9-report-row"><span>Future Fundamental</span><b>.*?</b></div>', re.DOTALL)
    match = pattern.search(html, max(0, cursor))
    if match is None:
        return html, cursor
    replacement = (
        '<div class="v9-report-row"><span>Future Fundamental</span>'
        f'<b title="{escape(detail, quote=True)}">{escape(label)}</b></div>'
    )
    html = html[:match.start()] + replacement + html[match.end():]
    return html, match.start() + len(replacement)


def install() -> None:
    try:
        import v9_dashboard as dashboard
    except Exception:
        return
    original = getattr(dashboard, "render_dashboard_html", None)
    if not callable(original) or getattr(original, "__future_ff_status_ui_v1__", False):
        return

    @wraps(original)
    def wrapped(top: pd.DataFrame, *args: Any, **kwargs: Any) -> str:
        html = original(top, *args, **kwargs)
        if not isinstance(top, pd.DataFrame) or top.empty:
            return html
        cursor = 0
        for _, row in top.iterrows():
            if _should_show_state(row):
                label, detail = _status(row)
                html, cursor = _replace_report_row(html, cursor=cursor, label=label, detail=detail)
            else:
                # Advance past the current card's Future Fundamental row without changing it.
                pattern = re.compile(r'<div class="v9-report-row"><span>Future Fundamental</span><b>.*?</b></div>', re.DOTALL)
                match = pattern.search(html, max(0, cursor))
                if match is not None:
                    cursor = match.end()
        return html

    wrapped.__future_ff_status_ui_v1__ = True
    setattr(dashboard, "render_dashboard_html", wrapped)


__all__ = ["PATCH_VERSION", "install"]
