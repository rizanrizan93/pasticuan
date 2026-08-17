from __future__ import annotations

"""Runtime hook: bounded ZAPI confirmation for Super Scanner final decision flow.

This patch adds a native Streamlit CSV download button for every Top 3 dashboard,
keeps a self-contained fallback download link inside the rendered HTML, exposes
full ZAPI audit lineage, and applies a post-authorization foreign-flow shock
guard to Swing execution only. ZAPI can delay/de-authorize an entry but can
never promote READY or rewrite the underlying research thesis.
"""

from functools import wraps
from html import escape
from typing import Any
import base64
import hashlib
import re

import numpy as np
import pandas as pd

from zapi_flow_enrichment import (
    ZAPI_FLOW_ENRICHMENT_VERSION,
    enrich_super_universe,
)
from zapi_post_calibration import (
    POST_CALIBRATION_VERSION,
    apply_super_foreign_shock_guard,
    enrich_super_shadow,
)

PATCH_VERSION = "1.3.0-super-zapi-post-calibration"


def _canonical(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text[:-3] if text.endswith(".JK") else text


def _merge_audit_fields(out: pd.DataFrame, enriched: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(out, pd.DataFrame) or out.empty or not isinstance(enriched, pd.DataFrame) or enriched.empty:
        return out
    if "ticker" not in out.columns or "ticker" not in enriched.columns:
        return out

    audit_source = enriched.copy()
    if "flow_silent_accumulation_score" in audit_source.columns:
        audit_source["zapi_super_post_silent_score"] = pd.to_numeric(
            audit_source["flow_silent_accumulation_score"], errors="coerce"
        )
        original = pd.to_numeric(
            audit_source.get("zapi_super_original_silent_score", pd.Series(np.nan, index=audit_source.index)),
            errors="coerce",
        )
        audit_source["zapi_super_silent_score_delta"] = (
            audit_source["zapi_super_post_silent_score"] - original
        )

    wanted = [
        "ticker",
        "zapi_foreign_latest_trade_date",
        "zapi_foreign_observed_days",
        "zapi_foreign_net_shares_1d",
        "zapi_foreign_net_shares_5d",
        "zapi_foreign_net_shares_20d",
        "zapi_foreign_flow_score",
        "zapi_foreign_flow_coverage_pct",
        "zapi_foreign_net_participation_1d",
        "zapi_foreign_net_participation_5d",
        "zapi_foreign_net_participation_20d",
        "zapi_foreign_positive_days_ratio_5d",
        "zapi_foreign_positive_days_ratio_20d",
        "zapi_foreign_buy_ratio_5d",
        "zapi_foreign_buy_ratio_20d",
        "zapi_foreign_state",
        "zapi_accumulation_confirmation_score",
        "zapi_smart_money_confirmation_score",
        "zapi_smc_flow_confirmation_score",
        "zapi_flow_source",
        "zapi_flow_unit",
        "zapi_flow_evidence_type",
        "zapi_flow_enrichment_version",
        "zapi_confirmation_weight_pct",
        "zapi_super_original_silent_score",
        "zapi_super_original_silent_coverage_pct",
        "zapi_super_post_silent_score",
        "zapi_super_silent_score_delta",
        "zapi_super_flow_basis",
        "zapi_flow_meta_state",
        "zapi_shared_cache_state",
        "zapi_direct_state",
    ]
    cols = [column for column in wanted if column in audit_source.columns]
    audit = audit_source[cols].copy()
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
    if not callable(original) or getattr(original, "__zapi_flow_confirmation_v2__", False):
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
                out = enrich_super_shadow(out)
                if name == "build_swing_ready":
                    out = apply_super_foreign_shock_guard(out)
            except Exception:
                pass
        return out

    wrapped.__zapi_flow_confirmation_v2__ = True
    setattr(owner, name, wrapped)


def _safe_filename_token(value: Any, default: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "").strip()).strip("_")
    return token[:80] or default


def _top3_csv_payload(top: pd.DataFrame, *, model: str = "", scan_id: str = "") -> tuple[bytes, str]:
    if not isinstance(top, pd.DataFrame) or top.empty:
        return b"", ""
    csv_bytes = top.to_csv(index=False).encode("utf-8-sig")
    model_token = _safe_filename_token(model, "TOP3").lower()
    scan_token = _safe_filename_token(scan_id, "latest")
    return csv_bytes, f"idx_super_top3_{model_token}_{scan_token}.csv"


def _top3_csv_download_block(top: pd.DataFrame, *, model: str = "", scan_id: str = "") -> str:
    """Return an in-HTML data-URI fallback download control for selected Top 3 rows."""
    csv_bytes, filename = _top3_csv_payload(top, model=model, scan_id=scan_id)
    if not csv_bytes:
        return ""
    payload = base64.b64encode(csv_bytes).decode("ascii")
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


def _emit_native_top3_download(
    top: pd.DataFrame,
    *,
    model: str = "",
    scan_id: str = "",
    context_hint: str = "",
) -> bool:
    """Render a native Streamlit download button when running inside Streamlit."""
    csv_bytes, filename = _top3_csv_payload(top, model=model, scan_id=scan_id)
    if not csv_bytes:
        return False
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is None:
            return False
        digest = hashlib.sha256(
            csv_bytes + str(context_hint or "").encode("utf-8")
        ).hexdigest()[:12]
        model_token = _safe_filename_token(model, "TOP3").lower()
        st.download_button(
            "⬇ Download Top 3 CSV",
            data=csv_bytes,
            file_name=filename,
            mime="text/csv",
            key=f"super_top3_csv_{model_token}_{digest}",
            width="stretch",
        )
        return True
    except Exception:
        return False


def _actionable_selector_input(frame: pd.DataFrame, *, model: str, lane: str) -> pd.DataFrame:
    """Prevent blocked/order-builder-ineligible swings from entering ACTIONABLE Top 3."""
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
    if "zapi_execution_flow_guard_state" in local.columns:
        guard = local["zapi_execution_flow_guard_state"].fillna("").astype(str).str.upper()
        local = local.loc[~guard.isin({"WAIT_FLOW_STABILIZATION_AND_RECLAIM", "REQUIRE_ABSORPTION_OR_RECLAIM_BEFORE_ENTRY"})].copy()
    return local


def _wrap_top_selector(owner: Any) -> None:
    original = getattr(owner, "select_top_candidates", None)
    if not callable(original) or getattr(original, "__actionable_contract_v2__", False):
        return

    @wraps(original)
    def wrapped(frame: pd.DataFrame, *args: Any, **kwargs: Any) -> pd.DataFrame:
        model = str(kwargs.get("model") or "")
        lane = str(kwargs.get("lane") or "RESEARCH")
        source = _actionable_selector_input(frame, model=model, lane=lane)
        return original(source, *args, **kwargs)

    wrapped.__actionable_contract_v2__ = True
    setattr(owner, "select_top_candidates", wrapped)


def _wrap_dashboard_renderer(owner: Any) -> None:
    original = getattr(owner, "render_dashboard_html", None)
    if not callable(original) or getattr(original, "__top3_csv_download_v2__", False):
        return

    @wraps(original)
    def wrapped(top: pd.DataFrame, *args: Any, **kwargs: Any) -> str:
        model = str(kwargs.get("model") or "")
        scan_id = str(kwargs.get("scan_id") or "")
        completeness_note = str(kwargs.get("completeness_note") or "")
        _emit_native_top3_download(
            top,
            model=model,
            scan_id=scan_id,
            context_hint=completeness_note,
        )
        html = original(top, *args, **kwargs)
        if not isinstance(html, str) or not html:
            return html
        block = _top3_csv_download_block(top, model=model, scan_id=scan_id)
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

    wrapped.__top3_csv_download_v2__ = True
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
        "post_calibration_version": POST_CALIBRATION_VERSION,
        "policy": "BOUNDED_CONFIRMATION_INSIDE_EXISTING_NARRATIVE_FLOW_PILLAR",
        "smc_policy": "PRICE_STRUCTURE_PRIMARY_ZAPI_FLOW_CONFIRMATION_ONLY",
        "identity_policy": "FOREIGN_FLOW_IS_NOT_BROKER_OR_BENEFICIAL_OWNER_IDENTITY",
        "top3_csv_policy": "NATIVE_STREAMLIT_DOWNLOAD_PLUS_SELF_CONTAINED_HTML_FALLBACK_WITH_ZAPI_AUDIT",
        "actionable_top3_policy": "ORDER_BUILDER_ELIGIBLE_NOT_REAL_MONEY_BLOCKED_AND_NO_ZAPI_SELL_SHOCK_WAIT",
        "execution_guard_policy": "ZAPI_SELL_SHOCK_CAN_ONLY_DELAY_OR_DEAUTHORIZE_NEVER_PROMOTE_READY",
        "shadow_policy": "CAPTURE_PRE_POST_ZAPI_SILENT_SCORE_FOR_5D_20D_60D_FORWARD_OOS",
    }


__all__ = [
    "PATCH_VERSION",
    "install",
    "_top3_csv_payload",
    "_top3_csv_download_block",
    "_emit_native_top3_download",
    "_actionable_selector_input",
]
