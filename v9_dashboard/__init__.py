from __future__ import annotations

"""Compatibility facade for the production dashboard module.

The Streamlit app passes a presentation-only ``completeness_note`` keyword.
The underlying dashboard renderer predates that keyword. Keep the analytical
implementation untouched and adapt only the UI call contract so a completed
scan cannot crash while rendering Top 3 cards.
"""

from html import escape
import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd


_IMPL_PATH = Path(__file__).resolve().parents[1] / "v9_dashboard.py"
_SPEC = importlib.util.spec_from_file_location("_v9_dashboard_impl", _IMPL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load dashboard implementation from {_IMPL_PATH}")
_IMPL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_IMPL)

V9_DASHBOARD_VERSION = "1.3.1-completeness-note-contract"
select_top_candidates = _IMPL.select_top_candidates
recommendation_meta = _IMPL.recommendation_meta
authorization_meta = _IMPL.authorization_meta


def render_dashboard_html(
    top: pd.DataFrame,
    *,
    model: str,
    scan_id: str = "",
    as_of: Any = "",
    market_regime: str = "",
    completeness_note: str = "",
) -> str:
    """Render the existing dashboard while accepting the current app contract."""
    rendered = _IMPL.render_dashboard_html(
        top,
        model=model,
        scan_id=scan_id,
        as_of=as_of,
        market_regime=market_regime,
    )
    note = str(completeness_note or "").strip()
    if not note:
        return rendered
    safe_note = escape(note, quote=True)
    marker = '<div class="v9-footer">'
    note_html = f'<div class="v9-method"><small>{safe_note}</small></div>'
    if marker in rendered:
        return rendered.replace(marker, note_html + marker, 1)
    return rendered + note_html


__all__ = [
    "V9_DASHBOARD_VERSION",
    "select_top_candidates",
    "recommendation_meta",
    "authorization_meta",
    "render_dashboard_html",
]
