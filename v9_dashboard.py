from __future__ import annotations

from html import escape
from typing import Any, Mapping

import numpy as np
import pandas as pd
import v9_dashboard_legacy as _legacy

from release_contract import SCANNER_RELEASE_VERSION

from v9_dashboard_legacy import *  # noqa: F401,F403

V9_DASHBOARD_VERSION = "1.5.0-research-rank-integrity"
SMART_MONEY_COST_BASIS_VERSION = "1.0.0"
SCANNER_VERSION = SCANNER_RELEASE_VERSION


def _num(value: Any, default: float = np.nan) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


def _cost_state(distance_pct: float) -> str:
    if not np.isfinite(distance_pct):
        return "COST_UNAVAILABLE"
    if distance_pct < -3.0:
        return "UNDER_PROXY_COST"
    if distance_pct <= 5.0:
        return "AT_PROXY_COST"
    if distance_pct <= 15.0:
        return "EARLY_MARKUP"
    if distance_pct <= 35.0:
        return "MARKUP"
    return "EXTENDED_MARKUP"


def _cost_basis_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    last = _num(row.get("last_price"), np.nan)

    direct_mid = _num(
        row.get("dominant_broker_avg_cost",
                row.get("broker_avg_buy_price",
                        row.get("verified_broker_average_cost"))),
        np.nan,
    )
    direct_low = _num(row.get("dominant_broker_cost_low", row.get("broker_cost_low")), np.nan)
    direct_high = _num(row.get("dominant_broker_cost_high", row.get("broker_cost_high")), np.nan)
    evidence_text = str(row.get("broker_inventory_evidence_type", "") or "").upper()

    if "DIRECT" in evidence_text and np.isfinite(direct_mid):
        low = direct_low if np.isfinite(direct_low) else direct_mid * 0.985
        high = direct_high if np.isfinite(direct_high) else direct_mid * 1.015
        confidence = min(100.0, max(78.0, _num(row.get("broker_inventory_coverage_pct"), 78.0)))
        evidence_type = "DIRECT_BROKER_EVIDENCE"
        note = "Direct broker-level average cost evidence; broker code is not beneficial-owner identity."
        mid = direct_mid
    else:
        zone_pairs = (
            ("research_accumulation_zone_low", "research_accumulation_zone_high", "RESEARCH_ACCUMULATION_ZONE"),
            ("order_block_low", "order_block_high", "SMC_ORDER_BLOCK_PROXY"),
            ("entry_low", "entry_high", "EXECUTION_ZONE_PROXY"),
        )
        low = high = mid = np.nan
        basis = ""
        for low_key, high_key, label in zone_pairs:
            lv, hv = _num(row.get(low_key), np.nan), _num(row.get(high_key), np.nan)
            if np.isfinite(lv) and np.isfinite(hv) and lv > 0 and hv > 0:
                low, high = min(lv, hv), max(lv, hv)
                mid = (low + high) / 2.0
                basis = label
                break
        if not np.isfinite(mid):
            reference = _num(row.get("research_accumulation_reference",
                                     row.get("research_preferred_reentry",
                                             row.get("entry"))), np.nan)
            if np.isfinite(reference) and reference > 0:
                mid = reference
                low, high = reference * 0.975, reference * 1.025
                basis = "ACCUMULATION_REFERENCE_PROXY"
        coverage = _num(row.get("inventory_multi_horizon_coverage_pct"),
                        _num(row.get("score_coverage_pct"), 0.0))
        silent = _num(row.get("silent_accumulation_score"), 50.0)
        confidence = min(72.0, max(0.0, 0.65 * coverage + 0.35 * silent)) if np.isfinite(mid) else 0.0
        evidence_type = "OHLCV_INVENTORY_COST_PROXY" if np.isfinite(mid) else "COST_UNAVAILABLE"
        note = (
            f"{basis}; estimated inventory cost proxy, not an identified broker/beneficial-owner cost."
            if np.isfinite(mid) else
            "No defensible accumulation/inventory price anchor available."
        )

    distance = 100.0 * (last / mid - 1.0) if np.isfinite(last) and np.isfinite(mid) and mid > 0 else np.nan
    return {
        "estimated_smart_money_cost": round(mid, 4) if np.isfinite(mid) else np.nan,
        "estimated_smart_money_cost_low": round(low, 4) if np.isfinite(low) else np.nan,
        "estimated_smart_money_cost_high": round(high, 4) if np.isfinite(high) else np.nan,
        "smart_money_cost_distance_pct": round(distance, 2) if np.isfinite(distance) else np.nan,
        "smart_money_cost_state": _cost_state(distance),
        "smart_money_cost_confidence_pct": round(confidence, 1),
        "smart_money_cost_evidence_type": evidence_type,
        "smart_money_cost_note": note,
        "smart_money_cost_basis_version": SMART_MONEY_COST_BASIS_VERSION,
    }


def _enrich_cost_basis(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    out = frame.copy()
    extra = out.apply(lambda row: pd.Series(_cost_basis_from_row(row)), axis=1)
    for column in extra.columns:
        out[column] = extra[column]
    return out


def select_top_candidates(frame: pd.DataFrame, *, model: str, limit: int = 3, lane: str = "RESEARCH") -> pd.DataFrame:
    selected = _legacy.select_top_candidates(frame, model=model, limit=limit, lane=lane)
    return _enrich_cost_basis(selected)


def _fmt_rupiah(value: Any) -> str:
    parsed = _num(value)
    if not np.isfinite(parsed) or parsed <= 0:
        return "—"
    return "Rp" + f"{parsed:,.0f}".replace(",", ".")


def _fmt_pct(value: Any) -> str:
    parsed = _num(value)
    return "—" if not np.isfinite(parsed) else f"{parsed:+.1f}%"


def _cost_block(row: Mapping[str, Any]) -> str:
    low = _fmt_rupiah(row.get("estimated_smart_money_cost_low"))
    high = _fmt_rupiah(row.get("estimated_smart_money_cost_high"))
    mid = _fmt_rupiah(row.get("estimated_smart_money_cost"))
    state = escape(str(row.get("smart_money_cost_state") or "COST_UNAVAILABLE"))
    evidence = escape(str(row.get("smart_money_cost_evidence_type") or "COST_UNAVAILABLE"))
    conf = _num(row.get("smart_money_cost_confidence_pct"), 0.0)
    distance = _fmt_pct(row.get("smart_money_cost_distance_pct"))
    return (
        '<div class="v9-cost-basis">'
        '<span>EST. SMART MONEY COST</span>'
        f'<strong>{low} – {high}</strong>'
        f'<small>Mid {mid} • Price {distance} • {state} • conf {conf:.0f}%</small>'
        f'<em>{evidence} — bukan beneficial-owner cost basis</em>'
        '</div>'
    )


def render_dashboard_html(
    top: pd.DataFrame,
    *,
    model: str,
    scan_id: str = "",
    as_of: Any = "",
    market_regime: str = "",
    completeness_note: str = "",
    **_: Any,
) -> str:
    enriched = _enrich_cost_basis(top)
    html = _legacy.render_dashboard_html(
        enriched,
        model=model,
        scan_id=scan_id,
        as_of=as_of,
        market_regime=market_regime,
    )
    css = """
    .v9-cost-basis{margin-top:8px;padding:8px;border:1px solid #2b6f84;border-radius:8px;background:#092433;text-align:left}
    .v9-cost-basis span{display:block;color:#8eb8c8;font-size:8px;font-weight:800;letter-spacing:.5px}
    .v9-cost-basis strong{display:block;color:#77f0ba;font-size:13px;margin:3px 0}
    .v9-cost-basis small,.v9-cost-basis em{display:block;color:#a6c4d1;font-size:8px;font-style:normal;line-height:1.35}
    """
    html = html.replace("</style>", css + "</style>", 1)
    marker = "</div><p>Multi-horizon OHLCV proxy 20/60/120/252/504/756D — bukan identitas broker.</p>"
    for _, row in enriched.iterrows():
        replacement = "</div>" + _cost_block(row) + "<p>Multi-horizon OHLCV proxy 20/60/120/252/504/756D — bukan identitas broker.</p>"
        html = html.replace(marker, replacement, 1)
    if completeness_note:
        note = f'<div class="v9-method"><small>{escape(str(completeness_note))}</small></div>'
        html = html.replace('<div class="v9-footer">', note + '<div class="v9-footer">', 1)
    return html


__all__ = list(getattr(_legacy, "__all__", []))
for _name in ("V9_DASHBOARD_VERSION", "SMART_MONEY_COST_BASIS_VERSION", "SCANNER_VERSION", "select_top_candidates", "render_dashboard_html"):
    if _name not in __all__:
        __all__.append(_name)
