from __future__ import annotations

from html import escape
from typing import Any, Mapping

import numpy as np
import pandas as pd
import v9_dashboard_legacy as _legacy

from release_contract import SCANNER_RELEASE_VERSION

from v9_dashboard_legacy import *  # noqa: F401,F403

V9_DASHBOARD_VERSION = "1.6.0-institutional-ui"
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


_INSTITUTIONAL_CSS = """
/* Presentation-only institutional skin. Calculation and rank semantics stay in legacy. */
.v9-wrap{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif!important;color:#e7eef6!important;background:#071019!important;border:1px solid #203244!important;border-radius:14px!important;padding:16px!important;box-shadow:none!important}
.v9-title{text-align:left!important;margin:2px 0 0!important;font-size:clamp(23px,3vw,36px)!important;font-weight:720!important;letter-spacing:-.035em!important;color:#f3f7fb!important}.v9-title b{color:#58c8b8!important;font-weight:720!important}
.v9-method{max-width:none!important;text-align:left!important;margin:10px 0 16px!important;padding:10px 12px!important;background:#0b1621!important;border:1px solid #1d3042!important;border-radius:9px!important;color:#8fa3b6!important;font-size:11px!important;line-height:1.5!important}.v9-method strong{color:#c7d6e4!important}.v9-method small{color:#7890a4!important}
.v9-card{position:relative!important;margin:12px 0!important;padding:13px!important;background:#0a1520!important;border:1px solid #203244!important;border-radius:12px!important;box-shadow:0 8px 24px rgba(0,0,0,.16)!important;overflow:hidden!important}.v9-card:before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:#51687b}.v9-card.rank1:before{background:#18b89f}.v9-card.rank2:before{background:#b99a4b}.v9-card.rank3:before{background:#5c87ad}.v9-card.rank1,.v9-card.rank2,.v9-card.rank3{border-color:#203244!important}
.v9-card-head{display:grid!important;grid-template-columns:54px minmax(190px,2fr) minmax(115px,.72fr) minmax(155px,1fr) minmax(140px,.9fr)!important;gap:9px!important;align-items:stretch!important}
.v9-rank{border-radius:9px!important;background:#11231f!important;border:1px solid #245347!important;color:#d9f5ef!important}.rank2 .v9-rank{background:#211e13!important;border-color:#584c25!important;color:#f1e3b0!important}.rank3 .v9-rank{background:#101d28!important;border-color:#2c4d68!important;color:#d5e7f5!important}.v9-rank small{font-size:9px!important;letter-spacing:.08em!important;color:inherit!important;opacity:.7}.v9-rank strong{font-size:30px!important;font-variant-numeric:tabular-nums!important}
.v9-identity,.v9-score,.v9-rec,.v9-price{padding:10px 11px!important;border-radius:9px!important;background:#0d1925!important;border:1px solid #203244!important}.v9-identity h2{font-size:29px!important;line-height:1!important;letter-spacing:-.025em!important;color:#f5f8fb!important}.v9-identity p{margin:5px 0 2px!important;color:#9eb0c0!important;font-size:10px!important}.v9-identity em{color:#58c8b8!important;font-size:10px!important;letter-spacing:.04em!important}.v9-chip{margin:6px 4px 0 0!important;padding:2px 6px!important;background:#101f2d!important;border:1px solid #294055!important;border-radius:5px!important;color:#9fb3c4!important;font-size:8px!important;font-weight:650!important}
.v9-score span,.v9-rec span,.v9-price span{font-size:8px!important;letter-spacing:.08em!important;color:#778da1!important}.v9-score strong{font-size:36px!important;color:#69d7c6!important;font-variant-numeric:tabular-nums!important}.v9-score small,.v9-rec small,.v9-price small{color:#8298aa!important;font-size:8px!important}.v9-rec strong{font-size:17px!important;margin:5px 0!important}.v9-price strong{font-size:20px!important;margin:5px 0!important;font-variant-numeric:tabular-nums!important}.v9-rec.green strong{color:#63d6a8!important}.v9-rec.gold strong{color:#d7bc68!important}.v9-rec.orange strong{color:#d99b63!important}.v9-rec.blue strong{color:#75a9d4!important}.v9-rec.red strong{color:#d87570!important}
.v9-auth{margin-top:9px!important;padding:8px 10px!important;border-radius:8px!important;background:#0c1823!important;border:1px solid #203244!important;font-size:8px!important}.v9-auth strong{font-size:9px!important;letter-spacing:.03em!important}.v9-auth span,.v9-auth small{color:#8195a7!important}.v9-auth.green strong{color:#63d6a8!important}.v9-auth.gold strong{color:#d7bc68!important}.v9-auth.red strong{color:#d87570!important}
.v9-gates{gap:6px!important;margin-top:7px!important}.v9-gates span{padding:6px!important;border-radius:7px!important;background:#0c1823!important;border:1px solid #203244!important;color:#72899c!important;font-size:8px!important;letter-spacing:.04em!important}.v9-gates b{color:#d7e1e9!important}
.v9-grid-main{grid-template-columns:1fr 1.2fr .95fr!important;gap:9px!important;margin-top:9px!important}.v9-grid-bottom{grid-template-columns:.9fr 1.15fr 1fr!important;gap:9px!important;margin-top:9px!important}.v9-panel{background:#0c1823!important;border:1px solid #203244!important;border-radius:9px!important;padding:10px!important}.v9-panel h3{text-align:left!important;color:#8fa6b8!important;font-size:9px!important;letter-spacing:.09em!important;margin:0 0 9px!important}
.v9-plan-row{padding:5px 0!important;border-bottom:1px solid rgba(83,111,134,.18)!important;font-size:9px!important;color:#91a5b5!important}.v9-plan-row b{color:#d8e2e9!important;font-variant-numeric:tabular-nums!important}.v9-plan-row.stop b{color:#dc7b75!important}.v9-plan-row.target b{color:#69d3a7!important}
.v9-factor{grid-template-columns:105px 1fr 26px!important;gap:7px!important;font-size:9px!important;margin:6px 0!important;color:#90a5b6!important}.v9-factor b{font-variant-numeric:tabular-nums!important;color:#cdd8e0!important}.v9-bar{height:5px!important;background:#152637!important}.v9-bar i{background:#28a995!important}
.v9-gauge{width:98px!important;height:98px!important;background:conic-gradient(#2bb49e var(--pct),#5b2930 0)!important}.v9-gauge:before{width:72px!important;height:72px!important;background:#0c1823!important}.v9-gauge strong{font-size:20px!important}.v9-gauge small{font-size:7px!important;color:#8399aa!important}.v9-flow-stats{font-size:8px!important}.v9-flow-stats span{padding:5px!important;background:#101e2b!important;border:1px solid #1c3042!important}.v9-flow p{font-size:7px!important;color:#6f879a!important}
.v9-report-row{font-size:9px!important;border-bottom:1px solid rgba(83,111,134,.16)!important}.v9-report-row b{color:#cfb660!important}.v9-reasons{font-size:9px!important}.v9-reasons li{padding:4px 0!important}.v9-reasons li.positive span{color:#63d6a8!important}.v9-reasons li.developing span{color:#d7bc68!important}.v9-reasons li.warning span{color:#d99b63!important}.v9-highlight p{font-size:9px!important;line-height:1.55!important;color:#c7d3dc!important}.v9-highlight small{color:#71889a!important;font-size:8px!important}.v9-confidence{padding:6px!important;border-radius:6px!important;background:#101e2b!important;font-size:8px!important;letter-spacing:.05em!important}
.v9-cost-basis{margin-top:8px!important;padding:8px 9px!important;border:1px solid #234154!important;border-radius:7px!important;background:#0f1d29!important;text-align:left!important}.v9-cost-basis span{display:block;color:#7894a6!important;font-size:7px!important;font-weight:750;letter-spacing:.08em}.v9-cost-basis strong{display:block;color:#69d7c6!important;font-size:12px!important;margin:3px 0}.v9-cost-basis small,.v9-cost-basis em{display:block;color:#8095a6!important;font-size:7px!important;font-style:normal!important;line-height:1.4}
.v9-footer{margin-top:12px!important;padding:9px 0 0!important;border-top:1px solid #203244!important;color:#637b8e!important;font-size:8px!important}
@media(max-width:980px){.v9-card-head{grid-template-columns:50px 1.5fr 100px!important}.v9-rec,.v9-price{grid-column:span 1}.v9-grid-main,.v9-grid-bottom{grid-template-columns:1fr!important}.v9-auth{grid-template-columns:1fr!important}.v9-identity h2{font-size:25px!important}}
@media(max-width:640px){.v9-wrap{padding:8px!important;border-radius:10px!important}.v9-title{font-size:22px!important}.v9-method{font-size:9px!important;margin-bottom:10px!important}.v9-card{padding:9px!important}.v9-card-head{grid-template-columns:42px 1fr!important}.v9-score,.v9-rec,.v9-price{grid-column:span 1!important}.v9-score strong{font-size:31px!important}.v9-rank strong{font-size:26px!important}.v9-gates{grid-template-columns:1fr!important}.v9-factor{grid-template-columns:94px 1fr 24px!important}}
"""


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
    html = html.replace("</style>", _INSTITUTIONAL_CSS + "</style>", 1)
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
