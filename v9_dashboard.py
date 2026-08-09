from __future__ import annotations

"""Presentation-only dashboard for v9 scanner outputs.

No scanner, database, Streamlit or provider imports live here.  This keeps UI
changes isolated from acquisition/scoring logic and makes the dashboard easy to
unit test with plain DataFrames.
"""

from html import escape
from typing import Any, Mapping

import numpy as np
import pandas as pd


V9_DASHBOARD_VERSION = "1.2.0-calibrated-freshness-thesis"

LEADER_FACTORS = {
    "Business": "business_quality_score",
    "Future Fundamental": "future_fundamental_score",
    "Valuation / MOS": "valuation_mos_score",
    "Management": "management_capital_score",
    "Macro / Sector": "issuer_macro_alignment_score",
    "Narrative / Flow": "narrative_flow_score",
    "Silent Accum": "silent_accumulation_score",
    "Technical": "technical_readiness_score",
}
SWING_FACTORS = {
    "Technical": "technical_execution_score",
    "Macro / Sector": "issuer_macro_alignment_score",
    "Narrative / Flow": "narrative_flow_score",
    "Silent Accum": "silent_accumulation_score",
    "Business": "business_quality_score",
    "Risk / Data": "risk_data_score",
    "Next Leader": "next_leader_score",
}

LEADER_STATUS_PRIORITY = {"BUY_ZONE": 0, "WATCH": 1, "WAIT": 2, "RESEARCH_ONLY": 4}
SWING_STATUS_PRIORITY = {"EXECUTION_READY": 0, "ENTRY_PLAN_READY": 1, "WATCHLIST": 2, "WAIT": 3, "RESEARCH_ONLY": 5}


def _num(value: Any, default: float = np.nan) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


def _score(value: Any, default: float = 0.0) -> float:
    parsed = _num(value, default)
    return float(np.clip(parsed, 0.0, 100.0))


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().upper() in {"1", "TRUE", "YES", "Y", "PASS"}


def _esc(value: Any) -> str:
    return escape(str(value if value is not None else ""), quote=True)


def _fmt_rupiah(value: Any) -> str:
    parsed = _num(value)
    if not np.isfinite(parsed) or parsed <= 0:
        return "—"
    return "Rp" + f"{parsed:,.0f}".replace(",", ".")


def _fmt_score(value: Any) -> str:
    parsed = _num(value)
    return "—" if not np.isfinite(parsed) else f"{parsed:.0f}"


def _fmt_pct(value: Any, signed: bool = False) -> str:
    parsed = _num(value)
    if not np.isfinite(parsed):
        return "—"
    return f"{parsed:+.1f}%" if signed else f"{parsed:.1f}%"


def _stars(value: Any) -> str:
    filled = int(np.clip(np.round(_score(value) / 20.0), 0, 5))
    return "★" * filled + "☆" * (5 - filled)


def _final_score(row: Mapping[str, Any], model: str) -> float:
    field = "v9_next_leader_score" if model == "NEXT_LEADER" else "v9_swing_score"
    for name in ("final_score", field, "ranking_score", "research_score"):
        value = _num(row.get(name), np.nan)
        if np.isfinite(value):
            return _score(value)
    return 0.0


def select_top_candidates(frame: pd.DataFrame, *, model: str, limit: int = 3) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    model_name = str(model).upper()
    score_field = "v9_next_leader_score" if model_name == "NEXT_LEADER" else "v9_swing_score"
    priority_map = LEADER_STATUS_PRIORITY if model_name == "NEXT_LEADER" else SWING_STATUS_PRIORITY
    local = frame.copy()
    if "rank_eligible" in local.columns:
        local = local.loc[local["rank_eligible"].fillna(False).astype(bool)].copy()
    if local.empty:
        return local
    local["_status_priority"] = local.get("status", pd.Series(index=local.index, dtype=str)).astype(str).str.upper().map(priority_map).fillna(50)
    if "ranking_score" in local.columns:
        score_source = pd.to_numeric(local["ranking_score"], errors="coerce")
    elif score_field in local.columns:
        score_source = pd.to_numeric(local[score_field], errors="coerce")
    elif "final_score" in local.columns:
        score_source = pd.to_numeric(local["final_score"], errors="coerce")
    else:
        score_source = pd.Series(np.nan, index=local.index)
    coverage_source = local["score_coverage_pct"] if "score_coverage_pct" in local.columns else pd.Series(np.nan, index=local.index)
    accum_source = local["accumulation_dominance_pct"] if "accumulation_dominance_pct" in local.columns else local["silent_accumulation_score"] if "silent_accumulation_score" in local.columns else pd.Series(np.nan, index=local.index)
    method_source = local["methodology_priority"] if "methodology_priority" in local.columns else pd.Series(7, index=local.index)
    local["_score"] = pd.to_numeric(score_source, errors="coerce")
    local["_coverage"] = pd.to_numeric(coverage_source, errors="coerce")
    local["_accum"] = pd.to_numeric(accum_source, errors="coerce")
    local["_method_priority"] = pd.to_numeric(method_source, errors="coerce").fillna(7)
    local = local.sort_values(
        ["_status_priority", "_score", "_method_priority", "_coverage", "_accum", "ticker"],
        ascending=[True, False, True, False, False, True], na_position="last", kind="stable",
    ).head(max(0, int(limit))).drop(columns=["_status_priority", "_score", "_coverage", "_accum", "_method_priority"])
    local = local.reset_index(drop=True)
    local.insert(0, "dashboard_rank", np.arange(1, len(local) + 1))
    return local


def recommendation_meta(row: Mapping[str, Any], model: str) -> tuple[str, str, str]:
    overlay = str(row.get("decision_overlay_state", "")).upper()
    status = str(row.get("status", "")).upper()
    if overlay == "V9_WAIT_REACCUMULATION":
        return "WAIT REACCUM", "orange", "Markup sudah advanced; jangan chase"
    if overlay == "V9_DISTRIBUTION_BLOCK":
        return "AVOID / RESEARCH", "red", "Distribusi mengalahkan akumulasi"
    if model == "NEXT_LEADER":
        mapping = {
            "BUY_ZONE": ("BUY ZONE", "green", "Thesis dan execution gate terpenuhi"),
            "WATCH": ("WATCH BUY", "gold", "Thesis kuat; tunggu trigger/price"),
            "WAIT": ("WAIT", "orange", "Belum pada geometry entry terbaik"),
        }
    else:
        mapping = {
            "EXECUTION_READY": ("EXECUTION READY", "green", "Entry plan dan risk gate siap"),
            "ENTRY_PLAN_READY": ("ENTRY PLAN", "gold", "Plan valid; tunggu trigger"),
            "WATCHLIST": ("WATCH", "orange", "Setup belum cukup matang"),
            "WAIT": ("WAIT", "blue", "Belum ada edge eksekusi"),
        }
    return mapping.get(status, ("RESEARCH ONLY", "blue", "Belum memenuhi production gate"))




def authorization_meta(row: Mapping[str, Any]) -> tuple[str, str, str]:
    state = str(row.get("real_money_authorization_state", "REAL_MONEY_BLOCKED")).upper()
    if state == "REAL_MONEY_DIRECT_VERIFIED_READY":
        return "DIRECT VERIFIED READY", "green", "Seluruh real-money gate utama terverifikasi"
    if state == "REAL_MONEY_MANUAL_CONFIRMATION_REQUIRED":
        checks = str(row.get("real_money_manual_checks", "")).replace("_", " ")
        return "MANUAL CONFIRM", "gold", checks or "Perlu verifikasi manual sebelum order"
    blockers = str(row.get("real_money_authorization_blockers", "")).replace("_", " ")
    return "REAL MONEY BLOCKED", "red", blockers or "Real-money gate belum terpenuhi"

def _badges(row: Mapping[str, Any]) -> list[str]:
    items: list[str] = []
    lifecycle = str(row.get("inventory_lifecycle", "")).replace("_", " ")
    if lifecycle and lifecycle != "UNKNOWN":
        items.append(lifecycle)
    thesis = str(row.get("thesis_archetype", "")).replace("_", " ")
    if thesis:
        items.append(thesis)
    candidate = str(row.get("candidate_type", "")).replace("_", " ")
    if candidate and candidate != thesis:
        items.append(candidate)
    refresh = str(row.get("fundamental_refresh_state", "")).upper()
    if refresh == "REFRESH_WINDOW":
        items.append("REPORT REFRESH DUE")
    coverage = _num(row.get("score_coverage_pct"))
    if np.isfinite(coverage) and coverage >= 80:
        items.append("HIGH COVERAGE")
    if _score(row.get("silent_accumulation_score")) >= 65:
        items.append("SILENT ACCUM")
    if _truthy(row.get("fundamental_official_verified", False)):
        items.append("IDX OFFICIAL")
    return items[:5]


def _factor_rows(row: Mapping[str, Any], factors: Mapping[str, str]) -> str:
    rows: list[str] = []
    for label, field in factors.items():
        score = _score(row.get(field))
        rows.append(
            f'<div class="v9-factor"><span>{_esc(label)}</span><div class="v9-bar"><i style="width:{score:.1f}%"></i></div><b>{score:.0f}</b></div>'
        )
    return "".join(rows)


def _report_rows(row: Mapping[str, Any], factors: Mapping[str, str]) -> str:
    return "".join(
        f'<div class="v9-report-row"><span>{_esc(label)}</span><b>{_esc(_stars(row.get(field)))}</b></div>'
        for label, field in list(factors.items())[:6]
    )


def _plan_rows(row: Mapping[str, Any], model: str) -> str:
    trigger = row.get("trigger", row.get("trigger_price"))
    plan = [
        ("Entry zone", f"{_fmt_rupiah(row.get('entry_low'))} – {_fmt_rupiah(row.get('entry_high'))}", "entry"),
        ("Trigger", _fmt_rupiah(trigger), "trigger"),
        ("Stop", "< " + _fmt_rupiah(row.get("stop_loss")), "stop"),
        ("TP1", _fmt_rupiah(row.get("tp1")), "target"),
        ("TP2", _fmt_rupiah(row.get("tp2")), "target"),
        ("RR1", f"1 : {_num(row.get('rr1')):.2f}" if np.isfinite(_num(row.get("rr1"))) else "—", "rr"),
        ("RR2", f"1 : {_num(row.get('rr2')):.2f}" if np.isfinite(_num(row.get("rr2"))) else "—", "rr"),
    ]
    if model == "NEXT_LEADER":
        lots = _num(row.get("recommended_lots"))
        plan.append(("Suggested lots", f"{int(lots)}" if np.isfinite(lots) else "—", "rr"))
    else:
        lots = _num(row.get("stockbit_order_lots"))
        plan.append(("Stockbit lots", f"{int(lots)}" if np.isfinite(lots) else "—", "rr"))
    return "".join(
        f'<div class="v9-plan-row {kind}"><span>{_esc(label)}</span><b>{_esc(value)}</b></div>' for label, value, kind in plan
    )


def _reason_lines(row: Mapping[str, Any], model: str) -> list[str]:
    lines: list[str] = []
    if model == "NEXT_LEADER":
        lines.append(f"Business {_fmt_score(row.get('business_quality_score'))} / Future {_fmt_score(row.get('future_fundamental_score'))}")
        lines.append(f"Valuation MOS {_fmt_score(row.get('valuation_mos_score'))} / Macro {_fmt_score(row.get('issuer_macro_alignment_score'))}")
    else:
        lines.append(f"Technical {_fmt_score(row.get('technical_execution_score'))} / Macro {_fmt_score(row.get('issuer_macro_alignment_score'))}")
        lines.append(f"Next Leader {_fmt_score(row.get('next_leader_score'))} / Risk data {_fmt_score(row.get('risk_data_score'))}")
    lines.append(f"Narrative-flow {_fmt_score(row.get('narrative_flow_score'))} / Silent accum {_fmt_score(row.get('silent_accumulation_score'))}")
    lines.append(f"Inventory {_fmt_score(row.get('inventory_multi_horizon_score'))} • lifecycle {str(row.get('inventory_lifecycle', 'UNKNOWN')).replace('_', ' ')}")
    thesis = str(row.get("thesis_archetype", "")).replace("_", " ")
    trend = str(row.get("fundamental_trend_state", "")).replace("_", " ")
    refresh = str(row.get("fundamental_refresh_state", "")).replace("_", " ")
    if thesis:
        lines.append(f"Thesis {thesis} • fundamental trend {trend or 'UNKNOWN'}")
    if refresh and refresh != "CURRENT":
        lines.append(f"Statement freshness {refresh} • bounded refresh diprioritaskan")
    conflict = str(row.get("fundamental_growth_conflict_state", ""))
    if conflict and "SIGN_CONFLICT" in conflict.upper():
        lines.append("Latest-period growth berbeda arah dengan proxy lama; latest history diprioritaskan")
    cap = _num(row.get("fundamental_conviction_cap"))
    if np.isfinite(cap) and cap < 100:
        lines.append(f"Fundamental conviction cap {cap:.0f} • {str(row.get('fundamental_score_cap_reason', '')).replace('_', ' ')}")
    market_state = str(row.get("market_regime", "DATA_PENDING")).replace("_", " ")
    lines.append(f"Market {market_state} • context {_fmt_score(row.get('market_context_score'))}")
    if _truthy(row.get("anti_chase_gate", False)):
        lines.append("Anti-chase aktif: tunggu pullback/base atau breakout-retest")
    return lines


def _card_html(row: Mapping[str, Any], rank: int, model: str) -> str:
    rec, tone, note = recommendation_meta(row, model)
    auth_label, auth_tone, auth_note = authorization_meta(row)
    factors = LEADER_FACTORS if model == "NEXT_LEADER" else SWING_FACTORS
    score = _final_score(row, model)
    accumulation = _score(row.get("accumulation_dominance_pct", row.get("silent_accumulation_score", 0)))
    distribution = _score(row.get("distribution_risk_score", 0))
    ticker = str(row.get("ticker", "")).replace(".JK", "")
    sector = str(row.get("sector") or "SECTOR UNKNOWN")
    badges = "".join(f'<span class="v9-chip">{_esc(item)}</span>' for item in _badges(row))
    reasons = "".join(f'<li>{_esc(item)}</li>' for item in _reason_lines(row, model))
    summary = row.get("selected_reason") or note
    primary_risk = row.get("primary_risk") or row.get("production_gate_reason") or "Evidence dapat berubah"
    confidence = "HIGH" if score >= 80 else "GOOD" if score >= 70 else "MODERATE" if score >= 58 else "LOW"
    if model == "NEXT_LEADER":
        model_label = "THE NEXT LEADER"
    else:
        auth_state = str(row.get("real_money_authorization_state", "")).upper()
        status_state = str(row.get("status", "")).upper()
        model_label = "SWING READY" if auth_state == "REAL_MONEY_DIRECT_VERIFIED_READY" and status_state == "EXECUTION_READY" else "SWING WATCH / RESEARCH"
    top_color = {1: "rank1", 2: "rank2", 3: "rank3"}.get(rank, "rank3")
    return f"""
    <section class="v9-card {top_color}">
      <div class="v9-card-head">
        <div class="v9-rank"><small>TOP</small><strong>{rank}</strong></div>
        <div class="v9-identity"><h2>{_esc(ticker)}</h2><p>{_esc(model_label)}</p><em>{_esc(sector)}</em><div>{badges}</div></div>
        <div class="v9-score"><span>{'FINAL SCORE' if str(row.get('ranking_score_state','')).upper() == 'PRODUCTION_SCORE' else 'RANKING SCORE'}</span><strong>{score:.0f}</strong><small>/100 • cov {_fmt_pct(row.get('score_coverage_pct'))}</small></div>
        <div class="v9-rec {tone}"><span>REKOMENDASI</span><strong>{_esc(rec)}</strong><small>{_esc(note)}</small></div>
        <div class="v9-price"><span>HARGA SAAT INI</span><strong>{_fmt_rupiah(row.get('last_price'))}</strong><small>Extension {_fmt_pct(row.get('markup_extension_pct'))}</small></div>
      </div>
      <div class="v9-auth {auth_tone}"><strong>REAL MONEY AUTHORIZATION: {_esc(auth_label)}</strong><span>{_esc(auth_note)}</span><small>Risk budget cap {_fmt_pct(row.get('real_money_risk_budget_cap_pct'))} per idea • ranking tetap terpisah dari izin order</small></div>
      <div class="v9-grid-main">
        <div class="v9-panel"><h3>PLAN TRADING</h3>{_plan_rows(row, model)}</div>
        <div class="v9-panel"><h3>FAKTOR SCANNER</h3>{_factor_rows(row, factors)}</div>
        <div class="v9-panel v9-flow"><h3>FLOW / INVENTORY</h3>
          <div class="v9-gauge" style="--pct:{accumulation:.0f}%"><div><strong>{accumulation:.0f}%</strong><small>AKUMULASI</small></div></div>
          <div class="v9-flow-stats">
            <span>Silent <b>{_fmt_score(row.get('silent_accumulation_score'))}</b></span>
            <span>Inventory <b>{_fmt_score(row.get('inventory_multi_horizon_score'))}</b></span>
            <span>Distribution <b>{distribution:.0f}</b></span>
            <span>Reaccum <b>{_fmt_score(row.get('reaccumulation_quality_score'))}</b></span>
          </div><p>Multi-horizon OHLCV proxy 20/60/120/252/504/756D — bukan identitas broker.</p>
        </div>
      </div>
      <div class="v9-grid-bottom">
        <div class="v9-panel"><h3>REPORT CARD</h3>{_report_rows(row, factors)}</div>
        <div class="v9-panel"><h3>RINGKASAN ALASAN</h3><ul class="v9-reasons">{reasons}</ul></div>
        <div class="v9-panel v9-highlight"><h3>HIGHLIGHT / RISIKO</h3><p>{_esc(summary)}</p><div class="v9-confidence {tone}">CONFIDENCE {confidence}</div><small>{_esc(primary_risk)}</small></div>
      </div>
    </section>
    """


def render_dashboard_html(
    top: pd.DataFrame,
    *,
    model: str,
    scan_id: str = "",
    as_of: Any = "",
    market_regime: str = "",
) -> str:
    model_name = str(model).upper()
    cards = "".join(_card_html(row, int(row.get("dashboard_rank", idx + 1)), model_name) for idx, row in top.iterrows())
    title = "THE NEXT LEADER" if model_name == "NEXT_LEADER" else "SWING READY"
    method = (
        "Business • Future Fundamental • Valuation • Management • Macro • Narrative/Flow • Silent Accum • Technical"
        if model_name == "NEXT_LEADER" else
        "Technical • Macro • Narrative/Flow • Silent Accum • Business • Risk/Data • Next Leader confirmation"
    )
    return f"""
    <style>
    .v9-wrap{{font-family:Inter,Arial,sans-serif;color:#eaf7ff;background:linear-gradient(180deg,#06111e,#071827);padding:18px;border:1px solid #164968;border-radius:18px}}
    .v9-title{{text-align:center;margin:0;font-size:clamp(27px,5vw,50px);letter-spacing:.4px}} .v9-title b{{color:#48e89b}}
    .v9-method{{margin:12px auto 18px;max-width:1120px;text-align:center;padding:12px;border:1px solid #1b5878;border-radius:12px;background:#071421;color:#a9c6d8;font-size:12px}} .v9-method strong{{color:#56efad}}
    .v9-card{{margin:18px 0;padding:14px;background:linear-gradient(135deg,#071a2b,#061320);border:1px solid #1d5e7e;border-radius:17px;box-shadow:0 12px 32px rgba(0,0,0,.28)}}
    .v9-card.rank1{{border-color:#20c979}} .v9-card.rank2{{border-color:#b9a839}} .v9-card.rank3{{border-color:#3486d7}}
    .v9-card-head{{display:grid;grid-template-columns:74px minmax(180px,2fr) minmax(135px,.9fr) minmax(170px,1.1fr) minmax(150px,1fr);gap:12px;align-items:stretch}}
    .v9-rank{{display:flex;flex-direction:column;align-items:center;justify-content:center;border-radius:12px;background:linear-gradient(180deg,#17b769,#08613e);font-weight:800}} .rank2 .v9-rank{{background:linear-gradient(180deg,#b59c31,#665514)}} .rank3 .v9-rank{{background:linear-gradient(180deg,#2b88d6,#164577)}} .v9-rank strong{{font-size:42px;line-height:1}}
    .v9-identity,.v9-score,.v9-rec,.v9-price{{padding:12px;border-radius:12px;background:#081c2c;border:1px solid #1a4a64}} .v9-identity h2{{font-size:34px;margin:0}} .v9-identity p{{margin:2px 0;color:#c4d9e5;font-size:11px}} .v9-identity em{{font-style:normal;color:#59e5a5;font-weight:700;font-size:12px}}
    .v9-chip{{display:inline-block;margin:7px 5px 0 0;padding:3px 7px;border:1px solid #2c6b82;border-radius:20px;color:#b6d7e7;font-size:9px}}
    .v9-score,.v9-rec,.v9-price{{text-align:center;display:flex;flex-direction:column;justify-content:center}} .v9-score span,.v9-rec span,.v9-price span{{font-size:10px;color:#a9c8d8;font-weight:700}} .v9-score strong{{font-size:45px;color:#74f7bd;line-height:1}} .v9-score small,.v9-rec small,.v9-price small{{color:#a9c8d8;font-size:9px}}
    .v9-rec strong{{font-size:22px;line-height:1.15;margin:6px 0}} .v9-rec.green strong{{color:#53ed9c}} .v9-rec.gold strong{{color:#ffd85a}} .v9-rec.orange strong{{color:#ff9a4c}} .v9-rec.blue strong{{color:#7fc7ff}} .v9-rec.red strong{{color:#ff725f}} .v9-price strong{{font-size:25px;margin:6px 0}}
    .v9-auth{{display:grid;grid-template-columns:minmax(180px,.8fr) 2fr minmax(220px,1fr);gap:10px;align-items:center;margin-top:10px;padding:9px 12px;border-radius:10px;background:#0a2030;border:1px solid #2c6078;font-size:9px}} .v9-auth strong{{font-size:11px}} .v9-auth.green strong{{color:#54efa0}} .v9-auth.gold strong{{color:#ffe16d}} .v9-auth.red strong{{color:#ff8d82}} .v9-auth span,.v9-auth small{{color:#9eb9c8}}
    .v9-grid-main{{display:grid;grid-template-columns:1fr 1.28fr 1fr;gap:12px;margin-top:12px}} .v9-grid-bottom{{display:grid;grid-template-columns:.9fr 1.15fr 1fr;gap:12px;margin-top:12px}}
    .v9-panel{{background:#071a2a;border:1px solid #174965;border-radius:12px;padding:12px;min-width:0}} .v9-panel h3{{font-size:12px;color:#a9d8ef;text-align:center;margin:0 0 10px}}
    .v9-plan-row{{display:flex;justify-content:space-between;gap:8px;padding:5px 0;border-bottom:1px solid rgba(93,151,177,.15);font-size:11px}} .v9-plan-row.stop b{{color:#ff6c5d}} .v9-plan-row.target b{{color:#60ec9f}}
    .v9-factor{{display:grid;grid-template-columns:118px 1fr 28px;align-items:center;gap:8px;font-size:10px;margin:7px 0}} .v9-bar{{height:8px;background:#102d3b;border-radius:10px;overflow:hidden}} .v9-bar i{{display:block;height:100%;background:linear-gradient(90deg,#2fc47c,#79f5b8)}}
    .v9-flow{{text-align:center}} .v9-gauge{{--pct:50;width:118px;height:118px;margin:4px auto 10px;border-radius:50%;background:conic-gradient(#37d486 var(--pct),#ef604d 0);display:grid;place-items:center;position:relative}} .v9-gauge:before{{content:"";width:82px;height:82px;border-radius:50%;background:#071a2a;position:absolute}} .v9-gauge div{{position:relative;z-index:1;display:flex;flex-direction:column}} .v9-gauge strong{{font-size:25px}} .v9-gauge small{{font-size:9px}}
    .v9-flow-stats{{display:grid;grid-template-columns:1fr 1fr;gap:5px;text-align:left;font-size:9px}} .v9-flow-stats span{{padding:5px;background:#0a2233;border-radius:7px}} .v9-flow p{{font-size:8px;color:#88aab9;margin:9px 0 0}}
    .v9-report-row{{display:flex;justify-content:space-between;padding:4px 0;font-size:10px;border-bottom:1px solid rgba(93,151,177,.14)}} .v9-report-row b{{color:#ffd052;letter-spacing:1px}}
    .v9-reasons{{list-style:none;margin:0;padding:0;font-size:10px}} .v9-reasons li{{padding:4px 0}} .v9-reasons li:before{{content:"✓";color:#4aed94;font-weight:800;margin-right:7px}}
    .v9-highlight p{{font-size:11px;line-height:1.5;color:#d5e6ef}} .v9-highlight small{{display:block;margin-top:8px;color:#839fac;font-size:9px;word-break:break-word}} .v9-confidence{{text-align:center;padding:7px;border-radius:7px;font-size:10px;font-weight:800;background:#12334a}} .v9-confidence.green{{color:#54efa0;border:1px solid #25ba73}} .v9-confidence.gold{{color:#ffe16d;border:1px solid #a88928}} .v9-confidence.orange{{color:#ffac6b;border:1px solid #b56327}} .v9-confidence.blue{{color:#8dceff;border:1px solid #3478aa}} .v9-confidence.red{{color:#ff8d82;border:1px solid #a74036}}
    .v9-footer{{display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap;margin-top:16px;padding:10px;border-top:1px solid #174965;color:#8facbb;font-size:9px}}
    @media(max-width:900px){{.v9-auth{{grid-template-columns:1fr}} .v9-card-head{{grid-template-columns:56px 1fr 105px}} .v9-grid-main,.v9-grid-bottom{{grid-template-columns:1fr}} .v9-identity h2{{font-size:28px}}}}
    @media(max-width:580px){{.v9-wrap{{padding:9px}} .v9-card{{padding:9px}} .v9-card-head{{grid-template-columns:48px 1fr}} .v9-score,.v9-rec,.v9-price{{grid-column:span 1}} .v9-score strong{{font-size:36px}} .v9-rank strong{{font-size:32px}} .v9-factor{{grid-template-columns:100px 1fr 25px}}}}
    </style>
    <div class="v9-wrap">
      <h1 class="v9-title">TOP 3 <b>{_esc(title)}</b></h1>
      <div class="v9-method"><strong>v9 Macro-First + Inventory + Guarded Real Money</strong><br>{_esc(method)}<br><small>Ranking thesis tetap terpisah dari real-money authorization; official filing, cash-flow, leverage, regime, RR dan independent price dapat memblokir order tanpa menghapus kandidat dari ranking.</small></div>
      {cards if cards else '<div class="v9-panel">Tidak ada kandidat rank-eligible. Scanner tidak memaksa saham blocked/research-only masuk Top 3.</div>'}
      <div class="v9-footer"><span>Scan ID: {_esc(scan_id)}</span><span>As-of: {_esc(as_of)}</span><span>Market: {_esc(market_regime)}</span><span>Dashboard {V9_DASHBOARD_VERSION}</span></div>
    </div>
    """


__all__ = [
    "V9_DASHBOARD_VERSION",
    "select_top_candidates",
    "recommendation_meta",
    "authorization_meta",
    "render_dashboard_html",
]
