from __future__ import annotations

"""Runtime hook: bounded ZAPI confirmation for Super Scanner final decision flow.

This patch also injects a standalone CSV download link into every Top 3 HTML
dashboard. The CSV contains the complete selected rows, including ZAPI audit
columns when they are present, so the HTML report is self-contained.
"""

from functools import wraps
from html import escape
from typing import Any
import base64
import re

import pandas as pd

from zapi_flow_enrichment import (
    ZAPI_FLOW_ENRICHMENT_VERSION,
    enrich_super_universe,
)

PATCH_VERSION = "1.1.0-super-zapi-flow-top3-csv"


def _canonical(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text[:-3] if text.endswith(".JK") else text


def _merge_audit_fields(out: pd.DataFrame, enriched: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(out, pd.DataFrame) or out.empty or not isinstance(enriched, pd.DataFrame) or enriched.empty:
        return out
    if "ticker" not in out.columns or "ticker" not in enriched.columns:
        return out
    wanted = [
        "ticker",
        "zapi_foreign_flow_score",
        "zapi_foreign_flow_coverage_pct",
        "zapi_foreign_net_participation_1d",
        "zapi_foreign_net_participation_5d",
        "zapi_foreign_net_participation_20d",
        "zapi_foreign_positive_days_ratio_20d",
        "zapi_foreign_state",
        "zapi_accumulation_confirmation_score",
        "zapi_smart_money_confirmation_score",
        "zapi_smc_flow_confirmation_score",
        "zapi_flow_evidence_type",
        "zapi_confirmation_weight_pct",
        "zapi_super_original_silent_score",
        "zapi_super_original_silent_coverage_pct",
        "zapi_super_flow_basis",
        "zapi_flow_meta_state",
    ]
    cols = [column for column in wanted if column in enriched.columns]
    audit = enriched[cols].copy()
    audit["_zapi_key"] = audit["ticker"].map(_canonical)
    audit = audit.drop(columns=["ticker"]).drop_duplicates("_zapi_key", keep="last")
    result = out.copy()
    result["_zapi_key"] = result["ticker"].map(_canonical)
    duplicate = [column for column in audit.columns if column != "_zapi_key" and column in result.columns]
    if duplicate:
        result = result.drop(columns=duplicate)
    return result.merge(audit, on="_zapi_key", how="left").drop(columns=["_zapi_key"])


def _wrap_focus_builder(owner: Any, name: str) -> None:
    original = getattr(owner, name, None)
    if not callable(original) or getattr(original, "__zapi_flow_confirmation_v1__", False):
        return

    @wraps(original)
    def wrapped(universe: pd.DataFrame, *args: Any, **kwargs: Any):
        enriched = universe
        try:
            if isinstance(universe, pd.DataFrame) and not universe.empty:
                enriched = enrich_super_universe(universe)
        except Exception:
            enriched = universe
        out = original(enriched, *args, **kwargs)
        if isinstance(out, pd.DataFrame):
            try:
                out = _merge_audit_fields(out, enriched)
            except Exception:
                pass
        return out

    wrapped.__zapi_flow_confirmation_v1__ = True
    setattr(owner, name, wrapped)


def _safe_filename_token(value: Any, default: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "").strip()).strip("_")
    return token[:80] or default


def _top3_csv_download_block(top: pd.DataFrame, *, model: str = "", scan_id: str = "") -> str:
    """Return an in-HTML data-URI download control for the selected Top 3 rows."""
    if not isinstance(top, pd.DataFrame) or top.empty:
        return ""
    csv_bytes = top.to_csv(index=False).encode("utf-8-sig")
    payload = base64.b64encode(csv_bytes).decode("ascii")
    model_token = _safe_filename_token(model, "TOP3").lower()
    scan_token = _safe_filename_token(scan_id, "latest")
    filename = f"idx_super_top3_{model_token}_{scan_token}.csv"
    zapi_present = any(column.startswith("zapi_") for column in top.columns)
    audit_note = (
        "Full Top 3 row export • ZAPI audit fields included."
        if zapi_present
        else "Full Top 3 row export • ZAPI fields unavailable in this rendered frame."
    )
    return (
        '<div class="v9-csv-download">'
        f'<a href="data:text/csv;charset=utf-8;base64,{payload}" '
        f'download="{escape(filename)}">⬇ Download Top 3 CSV</a>'
        f'<small>{escape(audit_note)}</small>'
        "</div>"
    )


def _actionable_selector_input(frame: pd.DataFrame, *, model: str, lane: str) -> pd.DataFrame:
    """Prevent a blocked/order-builder-ineligible swing from entering the ACTIONABLE Top 3."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    model_name = str(model or "").upper()
    lane_name = str(lane or "RESEARCH").upper()
    if model_name == "NEXT_LEADER" or lane_name not in {"ACTIONABLE", "EXECUTION"}:
        return frame
    local = frame.copy()
    if "order_builder_eligible" in local.columns:
        local = local.loc[local["order_builder_eligible"].fillna(False).astype(bool)].copy()
    if "real_money_authorization_state" in local.columns:
        state = local["real_money_authorization_state"].fillna("").astype(str).str.upper()
        local = local.loc[state.ne("REAL_MONEY_BLOCKED")].copy()
    return local


def _wrap_top_selector(owner: Any) -> None:
    original = getattr(owner, "select_top_candidates", None)
    if not callable(original) or getattr(original, "__actionable_contract_v1__", False):
        return

    @wraps(original)
    def wrapped(frame: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        model = str(kwargs.get("model") or "")
        lane = str(kwargs.get("lane") or "RESEARCH")
        source = _actionable_selector_input(frame, model=model, lane=lane)
        return original(source, *args, **kwargs)

    wrapped.__actionable_contract_v1__ = True
    setattr(owner, "select_top_candidates", wrapped)


def _wrap_dashboard_renderer(owner: Any) -> None:
    original = getattr(owner, "render_dashboard_html", None)
    if not callable(original) or getattr(original, "__top3_csv_download_v1__", False):
        return

    @wraps(original)
    def wrapped(top: pd.DataFrame, *args: Any, **kwargs: Any) -> str:
        html = original(top, *args, **kwargs)
        if not isinstance(html, str) or not html:
            return html
        block = _top3_csv_download_block(
            top,
            model=str(kwargs.get("model") or ""),
            scan_id=str(kwargs.get("scan_id") or ""),
        )
        if not block:
            return html
        css = """
        .v9-csv-download{margin:10px 0 12px;padding:10px 12px;border:1px solid #315a70;border-radius:9px;background:#071a25;text-align:left}
        .v9-csv-download a{display:inline-block;padding:7px 11px;border-radius:7px;background:#0b7a75;color:#ecfeff!important;text-decoration:none;font-size:11px;font-weight:800;letter-spacing:.2px}
        .v9-csv-download a:hover{filter:brightness(1.12)}
        .v9-csv-download small{display:block;margin-top:6px;color:#8fb3c4;font-size:8px;line-height:1.35}
        """
        if "</style>" in html:
            html = html.replace("</style>", css + "</style>", 1)
        if '<div class="v9-footer">' in html:
            return html.replace('<div class="v9-footer">', block + '<div class="v9-footer">', 1)
        if "</body>" in html:
            return html.replace("</body>", block + "</body>", 1)
        return html + block

    wrapped.__top3_csv_download_v1__ = True
    setattr(owner, "render_dashboard_html", wrapped)


def install() -> dict[str, str]:
    import simple_focus
    import v9_dashboard

    _wrap_focus_builder(simple_focus, "build_next_leaders")
    _wrap_focus_builder(simple_focus, "build_swing_ready")
    _wrap_top_selector(v9_dashboard)
    _wrap_dashboard_renderer(v9_dashboard)
    return {
        "patch_version": PATCH_VERSION,
        "zapi_version": ZAPI_FLOW_ENRICHMENT_VERSION,
        "policy": "BOUNDED_CONFIRMATION_INSIDE_EXISTING_NARRATIVE_FLOW_PILLAR",
        "smc_policy": "PRICE_STRUCTURE_PRIMARY_ZAPI_FLOW_CONFIRMATION_ONLY",
        "identity_policy": "FOREIGN_FLOW_IS_NOT_BROKER_OR_BENEFICIAL_OWNER_IDENTITY",
        "top3_csv_policy": "SELF_CONTAINED_HTML_DATA_URI_FULL_ROW_EXPORT_WITH_ZAPI_AUDIT_WHEN_AVAILABLE",
        "actionable_top3_policy": "ORDER_BUILDER_ELIGIBLE_AND_NOT_REAL_MONEY_BLOCKED",
    }


__all__ = [
    "PATCH_VERSION",
    "install",
    "_top3_csv_download_block",
    "_actionable_selector_input",
]
