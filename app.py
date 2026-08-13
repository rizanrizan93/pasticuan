from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import streamlit as st

APP_VERSION = "9.8.7-production-hardening"
APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

st.set_page_config(page_title="IDX Scanner v9.8.7", page_icon="📊", layout="wide")

REQUIRED_FILES = (
    "scanner.py", "scanner_database.py", "narrative_engine.py", "macro_engine.py",
    "simple_focus.py", "free_data_providers.py", "ihsg_direction.py",
    "decision_overlay.py", "real_money_guard.py", "fundamental_calibration.py", "v9_dashboard.py",
    "resumable_app_engine.py", "fast_scan_engine.py", "evidence_enrichment.py",
)
missing = [name for name in REQUIRED_FILES if not (APP_ROOT / name).is_file()]
if missing:
    st.error("Deployment v9.8.5 tidak lengkap.")
    st.code("\n".join(missing), language="text")
    st.stop()

from scanner import parse_portfolio_csv, parse_universe_csv  # noqa: E402
from macro_engine import MACRO_ENGINE_VERSION  # noqa: E402
from simple_focus import SIMPLE_FOCUS_VERSION  # noqa: E402
from decision_overlay import DECISION_OVERLAY_VERSION  # noqa: E402
from v9_dashboard import V9_DASHBOARD_VERSION, render_dashboard_html, select_top_candidates  # noqa: E402
from fast_scan_engine import FAST_SCAN_VERSION, run_fast_single_scan  # noqa: E402
from fundamental_calibration import CALIBRATION_VERSION  # noqa: E402

st.markdown(
    """
    <style>
      .block-container {padding-top: 1rem; padding-bottom: 2.5rem;}
      [data-testid="stMetricValue"] {font-size: 1.35rem;}
      .v9-note {border:1px solid #334155; border-radius:10px; padding:12px 14px; background:#0f172a;}
      .small-muted {font-size:.84rem; color:#94a3b8;}
    </style>
    """, unsafe_allow_html=True,
)


def _finite(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _safe_display(frame: pd.DataFrame | None, columns: Iterable[str] | None = None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if columns is not None:
        selected = [column for column in columns if column in out.columns]
        out = out.loc[:, selected]
    for column in out.select_dtypes(include=["object"]).columns:
        out[column] = out[column].map(lambda value: "" if value is None else str(value))
    return out.reset_index(drop=True)


def _runtime_tokens(itick_token: str = "", twelve_token: str = "") -> dict[str, str]:
    def secret(name: str) -> str:
        try:
            return str(st.secrets.get(name, "") or "").strip()
        except Exception:
            return ""
    return {
        "itick_api_token": str(itick_token or os.getenv("ITICK_API_TOKEN", "") or secret("ITICK_API_TOKEN")).strip(),
        "twelve_data_api_key": str(twelve_token or os.getenv("TWELVE_DATA_API_KEY", "") or secret("TWELVE_DATA_API_KEY")).strip(),
    }


st.title("IDX Super Scanner v9.8.7 — Production Hardening")
st.caption(
    f"{APP_VERSION} • fast {FAST_SCAN_VERSION} • macro {MACRO_ENGINE_VERSION} • "
    f"decision {SIMPLE_FOCUS_VERSION} • calibration {CALIBRATION_VERSION} • inventory {DECISION_OVERLAY_VERSION} • dashboard {V9_DASHBOARD_VERSION}"
)
st.markdown(
    """
    <div class="v9-note">
      <b>Upload → SCAN → Ranking</b><br>
      Semua ticker tetap <b>eligible</b>. Feature cache current dipakai langsung; hanya ticker missing/stale yang membuka OHLCV panjang.
      Supabase berfungsi sebagai akselerator/persistence <b>fail-soft</b>, bukan dependency yang boleh menahan ranking.
      Deep fundamental/news/official IDX hanya untuk kandidat prioritas + maintenance rotation. Latest-report refresh window diprioritaskan tanpa mengubah bobot engine.
    </div>
    """, unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Input")
    ticker_file = st.file_uploader("Universe ticker CSV", type=["csv"])
    portfolio_file = st.file_uploader("Portfolio CSV (opsional)", type=["csv"])
    account_size = float(st.number_input("Nilai akun (Rp)", min_value=0, value=5_000_000, step=500_000))
    cash_on_hand = float(st.number_input("Cash tersedia (Rp)", min_value=0, value=1_000_000, step=100_000))
    with st.expander("Pengaturan scan", expanded=False):
        period = st.selectbox("Riwayat OHLCV", ["3y", "5y", "10y"], index=1)
        risk_per_trade_pct = st.slider("Risiko per transaksi", 0.25, 2.00, 0.75, 0.25) / 100.0
        itick_token = st.text_input("iTick API token", value=os.getenv("ITICK_API_TOKEN", ""), type="password")
        twelve_token = st.text_input("Twelve Data API key", value=os.getenv("TWELVE_DATA_API_KEY", ""), type="password")
        st.caption("Engine policy otomatis: 400 technical discovery → shortlist evidence → final ranking. Tidak ada mode isi database.")
    run_scan = st.button("SCAN", type="primary", width="stretch")

if run_scan:
    if ticker_file is None:
        st.error("Upload CSV ticker terlebih dahulu.")
        st.stop()
    try:
        universe_metadata = parse_universe_csv(ticker_file, max_tickers=400, strict_limit=True)
        tickers = universe_metadata["ticker"].tolist()
    except Exception as exc:
        st.error(f"CSV ticker tidak valid: {exc}")
        st.stop()
    portfolio = pd.DataFrame()
    if portfolio_file is not None:
        try:
            portfolio = parse_portfolio_csv(portfolio_file)
        except Exception as exc:
            st.error(f"Portfolio CSV tidak valid: {exc}")
            st.stop()
    portfolio_tickers = portfolio["ticker"].dropna().astype(str).drop_duplicates().tolist() if not portfolio.empty and "ticker" in portfolio else []
    universe = list(dict.fromkeys(portfolio_tickers + tickers))[:400]
    config = {
        "period": period,
        "account_size_idr": account_size,
        "cash_on_hand_idr": cash_on_hand,
        "risk_per_trade_pct": risk_per_trade_pct,
        "portfolio_records": portfolio.to_dict("records") if not portfolio.empty else [],
        # Preserve uploaded IDX-IC classification through the ticker-only
        # technical path. This is source metadata, not a score override.
        "universe_records": universe_metadata.to_dict("records"),
        # Hard runtime budgets: analysis depth is concentrated on names that can
        # actually enter Top 3/Top ranking. Database coverage expands gradually.
        "evidence_refresh_cap": 20,
        "decision_evidence_cap": 12,
        "evidence_fundamental_cap": 20,
        "evidence_official_cap": 12,
        "evidence_snapshot_cap": 16,
        "evidence_market_cap": 6,
        "evidence_news_cap": 10,
        "execution_verification_cap": 8,
        "daily_market_refresh_limit": 6,
        "macro_external_enabled": True,
        "macro_timeout_seconds": 3,
        "lean_persistence": True,
        "lean_skip_narrative_history": True,
    }
    runtime = _runtime_tokens(itick_token, twelve_token)
    progress_bar = st.progress(0, text="Menyiapkan scanner")
    status_box = st.empty()

    def progress(label: str, fraction: float) -> None:
        pct = max(0, min(100, int(round(100 * fraction))))
        progress_bar.progress(pct, text=label)
        status_box.caption(label)

    try:
        with st.spinner("Scanner berjalan. Jalur ini tidak menunggu scan_jobs/finalizer Supabase."):
            result = run_fast_single_scan(universe, config=config, runtime=runtime, progress=progress)
        st.session_state["v9_scan_result"] = result
        progress_bar.progress(100, text="Selesai")
        db_state = str(result.get("database_transport_state", ""))
        if db_state == "READ_WRITE_DEGRADED_FAIL_SOFT":
            st.warning(
                "Supabase read dan write sama-sama terganggu pada scan ini. Ranking tetap selesai dari memory/provider; persistence scan ini mungkin parsial."
            )
        elif db_state == "READ_DEGRADED_WRITE_AVAILABLE":
            st.info(
                "Sebagian read Supabase timeout, tetapi jalur write tetap aktif. Scanner tetap mencoba mempersist hasil final; cek Database Sync Report untuk status exact."
            )
        elif db_state == "WRITE_DEGRADED_READ_AVAILABLE":
            st.warning(
                "Cache Supabase masih dapat dibaca, tetapi sebagian write terganggu. Ranking valid dari data yang tersedia; persistence dapat parsial."
            )
        else:
            st.success(f"Scan selesai dalam {_finite(result.get('scan_elapsed_seconds')):.1f} detik.")
    except Exception as exc:
        st.error(f"Scan gagal: {type(exc).__name__}: {str(exc)[:800]}")
        st.stop()

result = st.session_state.get("v9_scan_result")
if not result:
    st.info("Upload universe ticker lalu jalankan scanner. Maksimum 400 ticker.")
    st.stop()

focus = result.get("focus_screens", {})
leaders = focus.get("next_leaders", pd.DataFrame()).copy()
swings = focus.get("swing_ready", pd.DataFrame()).copy()
macro_snapshot = result.get("macro_snapshot", pd.DataFrame())
elapsed = _finite(result.get("scan_elapsed_seconds"), 0.0)
ranking_state = str(result.get("ranking_state", "FINAL")).upper()
coverage_summary = result.get("scan_coverage_summary", pd.DataFrame())
if ranking_state == "PROVISIONAL":
    basis = ""
    if isinstance(coverage_summary, pd.DataFrame) and not coverage_summary.empty:
        row = coverage_summary.iloc[0]
        requested = int(_finite(row.get("requested_tickers"), 0))
        ready = int(_finite(row.get("technical_ready_tickers", row.get("completed_tickers")), 0))
        unavailable = int(_finite(row.get("technical_unavailable_tickers"), 0))
        basis = f" Basis saat ini {ready}/{requested} ticker technical-ready; {unavailable} belum memiliki OHLCV valid."
    st.warning(
        "Ranking yang tampil masih PROVISIONAL karena job sedang FINALIZING. "
        "Entry/SL/TP dan execution readiness dapat berubah setelah verifikasi harga selesai." + basis
    )

coverage_row = coverage_summary.iloc[0] if isinstance(coverage_summary, pd.DataFrame) and not coverage_summary.empty else pd.Series(dtype=object)
requested_metric = int(_finite(coverage_row.get("requested_tickers"), len(result.get("prepared", {}))))
ohlcv_ready_metric = int(_finite(coverage_row.get("ohlcv_ready_tickers"), len(result.get("prepared", {}))))
leader_valid_metric = int(leaders.get("rank_eligible", pd.Series(True, index=leaders.index)).fillna(False).astype(bool).sum()) if isinstance(leaders, pd.DataFrame) and not leaders.empty else 0
swing_valid_metric = int(swings.get("rank_eligible", pd.Series(True, index=swings.index)).fillna(False).astype(bool).sum()) if isinstance(swings, pd.DataFrame) and not swings.empty else 0
metrics = st.columns(6)
metrics[0].metric("Universe", requested_metric)
metrics[1].metric("OHLCV ready", f"{ohlcv_ready_metric}/{requested_metric}" if requested_metric else ohlcv_ready_metric)
metrics[2].metric("Leader ranked", leader_valid_metric)
metrics[3].metric("Swing ranked", swing_valid_metric)
metrics[4].metric("Macro regime", str(macro_snapshot.iloc[0].get("macro_regime", "DATA_PENDING")) if not macro_snapshot.empty else "DATA_PENDING")
metrics[5].metric("Waktu", f"{elapsed:.1f} dtk")
feature_hits = int(_finite(result.get("feature_cache_hits"), 0))
feature_refreshes = int(_finite(result.get("feature_cache_refreshes"), 0))
st.caption(f"ALL_ELIGIBLE_LITE • feature-cache hit {feature_hits}/{requested_metric} • technical refresh {feature_refreshes}/{requested_metric}")
leader_prod = int(leaders.get("production_rank_eligible", pd.Series(False, index=leaders.index)).fillna(False).astype(bool).sum()) if isinstance(leaders, pd.DataFrame) and not leaders.empty else 0
swing_prod = int(swings.get("production_rank_eligible", pd.Series(False, index=swings.index)).fillna(False).astype(bool).sum()) if isinstance(swings, pd.DataFrame) and not swings.empty else 0
st.caption(f"Production-qualified: Next Leader {leader_prod} • Swing {swing_prod}. Research ranking tetap terpisah dari izin eksekusi.")
ranking_quality_state = str(result.get("ranking_quality_state", coverage_row.get("ranking_state", "UNKNOWN"))).upper()
if ranking_quality_state not in {"VALID", "UNKNOWN"}:
    st.error(
        f"Ranking quality: {ranking_quality_state}. Hasil hanya mencakup {ohlcv_ready_metric}/{requested_metric} ticker OHLCV-ready dan belum layak menjadi dasar order penuh."
    )
if isinstance(coverage_summary, pd.DataFrame) and not coverage_summary.empty:
    requested_count = int(_finite(coverage_row.get("requested_tickers"), 0))
    fundamental_count = int(_finite(coverage_row.get("fundamental_ready"), 0))
    if requested_count > 0 and fundamental_count < requested_count:
        st.warning(
            f"Evidence fundamental production-grade tersedia untuk {fundamental_count}/{requested_count} ticker. "
            "Ticker lain tetap boleh masuk ranking research ALL_ELIGIBLE jika coverage minimum tercapai, tetapi Real Money Gate tetap memblokir eksekusi sampai evidence lengkap."
        )

if not leaders.empty:
    leaders["Ranking Score"] = pd.to_numeric(leaders.get("ranking_score"), errors="coerce")
    leaders["Production Score"] = pd.to_numeric(leaders.get("final_score", leaders.get("v9_next_leader_score")), errors="coerce")
if not swings.empty:
    swings["Ranking Score"] = pd.to_numeric(swings.get("ranking_score"), errors="coerce")
    swings["Production Score"] = pd.to_numeric(swings.get("final_score", swings.get("v9_swing_score")), errors="coerce")

dashboard_tab, market_tab, leader_tab, swing_tab, portfolio_tab = st.tabs([
    "Top 3 Dashboard", "Market Map", "The Next Leader", "Swing Ready", "Portfolio & Audit",
])

with dashboard_tab:
    market_regime = str(macro_snapshot.iloc[0].get("macro_regime", "DATA_PENDING")) if not macro_snapshot.empty else "DATA_PENDING"
    scan_id = str(result.get("scan_id", ""))
    as_of = result.get("scan_finished_at", result.get("scan_started_at", ""))
    top_leaders = select_top_candidates(leaders, model="NEXT_LEADER", limit=3, lane="RESEARCH")
    top_investable = select_top_candidates(leaders, model="NEXT_LEADER", limit=3, lane="PORTFOLIO")
    top_swings = select_top_candidates(swings, model="SWING_READY", limit=3, lane="ACTIONABLE")
    top_swing_watch = select_top_candidates(swings, model="SWING_READY", limit=3, lane="RESEARCH")
    top_leader_tab, top_investable_tab, top_swing_tab, top_swing_watch_tab = st.tabs([
        "Top 3 Next Leader", "Top 3 Investable Leader", "Top 3 Swing Ready", "Top 3 Swing Watch"
    ])
    with top_leader_tab:
        st.markdown(render_dashboard_html(top_leaders, model="NEXT_LEADER", scan_id=scan_id, as_of=as_of, market_regime=market_regime), unsafe_allow_html=True)
        if not top_leaders.empty:
            st.download_button("Download Top 3 Next Leader CSV", top_leaders.to_csv(index=False).encode("utf-8-sig"), "v9_top3_next_leader.csv", "text/csv")
    with top_investable_tab:
        if top_investable.empty:
            st.warning("Tidak ada Next Leader yang lolos portfolio/investability gate saat ini. Research ranking tetap tersedia.")
        else:
            st.markdown(render_dashboard_html(top_investable, model="NEXT_LEADER", scan_id=scan_id, as_of=as_of, market_regime=market_regime), unsafe_allow_html=True)
            st.download_button("Download Top 3 Investable Leader CSV", top_investable.to_csv(index=False).encode("utf-8-sig"), "v9_top3_investable_leader.csv", "text/csv")
    with top_swing_tab:
        if top_swings.empty:
            st.warning("Tidak ada production-qualified actionable Swing setup. Scanner tidak memaksa RESEARCH_ONLY menjadi Top 3 Swing Ready.")
        else:
            st.markdown(render_dashboard_html(top_swings, model="SWING_READY", scan_id=scan_id, as_of=as_of, market_regime=market_regime), unsafe_allow_html=True)
            st.download_button("Download Top 3 Swing Ready CSV", top_swings.to_csv(index=False).encode("utf-8-sig"), "v9_top3_swing_ready.csv", "text/csv")
    with top_swing_watch_tab:
        st.markdown(render_dashboard_html(top_swing_watch, model="SWING_READY", scan_id=scan_id, as_of=as_of, market_regime=market_regime), unsafe_allow_html=True)
        if not top_swing_watch.empty:
            st.download_button("Download Top 3 Swing Watch CSV", top_swing_watch.to_csv(index=False).encode("utf-8-sig"), "v9_top3_swing_watch.csv", "text/csv")

with market_tab:
    if not macro_snapshot.empty:
        row = macro_snapshot.iloc[0]
        cols = st.columns(4)
        cols[0].metric("Market context", f"{_finite(row.get('market_context_score', row.get('macro_regime_score'))):.1f}")
        cols[1].metric("Coverage", f"{_finite(row.get('macro_data_coverage_pct')):.0f}%")
        cols[2].metric("Breadth > EMA50", f"{_finite(row.get('breadth_above_ema50_pct')):.1f}%")
        cols[3].metric("IHSG 20D", f"{100.0 * _finite(row.get('ihsg_return_20d')):.2f}%")
        factor_cols = [column for column in macro_snapshot.columns if column.startswith("factor_")]
        factor_view = pd.DataFrame({
            "factor": [column.replace("factor_", "").replace("_score", "") for column in factor_cols],
            "score": [row.get(column) for column in factor_cols],
        })
        st.subheader("Macro factors")
        st.dataframe(_safe_display(factor_view), width="stretch", hide_index=True)
    st.subheader("Sector opportunity map")
    st.dataframe(_safe_display(result.get("macro_sector_map", pd.DataFrame())), width="stretch", hide_index=True)
    ihsg = result.get("ihsg_direction") or {}
    if isinstance(ihsg, Mapping):
        st.caption(
            f"IHSG model: {ihsg.get('consensus_direction', 'NO_EDGE')} • "
            f"confidence {_finite(ihsg.get('consensus_confidence')):.1f}% • "
            f"risk multiplier {_finite(ihsg.get('risk_budget_multiplier'), 0.5):.2f}x"
        )
    with st.expander("Macro source audit"):
        st.dataframe(_safe_display(result.get("macro_source_report", pd.DataFrame())), width="stretch", hide_index=True)

with leader_tab:
    columns = [
        "rank", "production_rank", "portfolio_rank", "ticker", "sector", "candidate_type", "thesis_archetype", "status",
        "Ranking Score", "Production Score", "ranking_score_state", "production_rank_eligible", "score_coverage_pct",
        "fundamental_freshness_state", "fundamental_refresh_state", "fundamental_trend_state", "fundamental_growth_basis_state", "fundamental_growth_conflict_state", "fundamental_data_quality_score", "fundamental_conviction_cap",
        "fundamental_official_verified", "fundamental_official_source_coverage_pct", "fundamental_cashflow_state",
        "market_regime", "market_context_score", "real_money_authorization_state", "real_money_authorization_blockers", "real_money_manual_checks",
        "sector_source", "sector_confidence_pct",
        "business_quality_score", "future_fundamental_score", "valuation_mos_score",
        "management_capital_score", "issuer_macro_alignment_score",
        "narrative_flow_score", "technical_readiness_score",
        "silent_accumulation_score", "inventory_multi_horizon_score", "inventory_lifecycle",
        "distribution_risk_score", "anti_chase_gate", "decision_overlay_state", "retail_adoption_stage",
        "research_gate_state", "portfolio_gate_state", "execution_gate_state", "thesis_confidence_pct", "research_accumulation_zone_low", "research_accumulation_zone_high", "research_preferred_reentry", "research_invalidation_reference", "research_zone_state", "entry_low", "entry_high", "trigger", "stop_loss", "tp1", "tp2", "rr1",
        "recommended_allocation_idr", "recommended_lots", "selected_reason", "primary_risk",
    ]
    view = _safe_display(leaders.head(50), columns)
    if view.empty:
        st.warning("Belum ada kandidat The Next Leader.")
    else:
        st.dataframe(view, width="stretch", hide_index=True)
        st.download_button("Download The Next Leader CSV", view.to_csv(index=False).encode("utf-8-sig"), "v9_the_next_leader.csv", "text/csv")

with swing_tab:
    columns = [
        "rank", "production_rank", "actionable_rank", "ticker", "sector", "status", "Ranking Score", "Production Score",
        "ranking_score_state", "production_rank_eligible", "score_coverage_pct",
        "real_money_authorization_state", "real_money_authorization_blockers", "real_money_manual_checks",
        "fundamental_data_quality_score", "fundamental_official_verified", "market_regime", "market_context_score",
        "production_gate_reason", "technical_execution_score", "issuer_macro_alignment_score", "narrative_flow_score",
        "silent_accumulation_score", "inventory_multi_horizon_score", "inventory_lifecycle",
        "distribution_risk_score", "anti_chase_gate", "decision_overlay_state",
        "business_quality_score", "risk_data_score", "next_leader_score", "strategy",
        "entry_zone_role", "entry_zone_is_executable", "execution_entry", "entry_low", "entry_high", "trigger_price", "stop_loss", "tp1", "tp2", "rr1", "rr2",
        "stockbit_order_lots", "next_action", "selected_reason", "primary_risk",
    ]
    view = _safe_display(swings.head(50), columns)
    if view.empty:
        st.warning("Belum ada setup Swing Ready.")
    else:
        st.dataframe(view, width="stretch", hide_index=True)
        st.download_button("Download Swing Ready CSV", view.to_csv(index=False).encode("utf-8-sig"), "v9_swing_ready.csv", "text/csv")

with portfolio_tab:
    portfolio_analysis = result.get("portfolio_analysis", pd.DataFrame())
    portfolio_summary = result.get("portfolio_summary", {})
    if isinstance(portfolio_summary, Mapping) and portfolio_summary:
        cols = st.columns(4)
        cols[0].metric("Nilai pasar", f"Rp {_finite(portfolio_summary.get('market_value_idr')):,.0f}")
        cols[1].metric("Unrealized P/L", f"Rp {_finite(portfolio_summary.get('unrealized_pnl_idr')):,.0f}")
        cols[2].metric("Open risk", f"Rp {_finite(portfolio_summary.get('open_risk_idr')):,.0f}")
        cols[3].metric("Estimasi equity", f"Rp {_finite(portfolio_summary.get('estimated_equity_idr')):,.0f}")
    if isinstance(portfolio_analysis, pd.DataFrame) and not portfolio_analysis.empty:
        st.dataframe(_safe_display(portfolio_analysis), width="stretch", hide_index=True)
    else:
        st.info("Portfolio CSV belum diunggah.")

    with st.expander("Ticker belum masuk ranking"):
        leader_all = focus.get("next_leaders_all", pd.DataFrame())
        swing_all = focus.get("swing_ready_all", pd.DataFrame())
        pending_frames = []
        if isinstance(leader_all, pd.DataFrame) and not leader_all.empty:
            leader_mask = leader_all["rank_eligible"].fillna(False).astype(bool) if "rank_eligible" in leader_all.columns else pd.Series(False, index=leader_all.index)
            local = leader_all.loc[~leader_mask].copy()
            if not local.empty:
                local["model"] = "THE_NEXT_LEADER"
                pending_frames.append(local)
        if isinstance(swing_all, pd.DataFrame) and not swing_all.empty:
            swing_mask = swing_all["rank_eligible"].fillna(False).astype(bool) if "rank_eligible" in swing_all.columns else pd.Series(False, index=swing_all.index)
            local = swing_all.loc[~swing_mask].copy()
            if not local.empty:
                local["model"] = "SWING_READY"
                pending_frames.append(local)
        if pending_frames:
            concat_frames = [frame.dropna(axis=1, how="all") for frame in pending_frames]
            pending = pd.concat(concat_frames, ignore_index=True, sort=False)
        else:
            pending = pd.DataFrame()
        st.dataframe(_safe_display(pending), width="stretch", hide_index=True)
    with st.expander("Scoring contract"):
        st.dataframe(_safe_display(focus.get("production_scoring_audit", pd.DataFrame())), width="stretch", hide_index=True)
    with st.expander("Evidence per ticker"):
        evidence = _safe_display(focus.get("production_evidence_detail", pd.DataFrame()))
        st.dataframe(evidence, width="stretch", hide_index=True)
        if not evidence.empty:
            st.download_button("Download Evidence CSV", evidence.to_csv(index=False).encode("utf-8-sig"), "v9_evidence.csv", "text/csv")
    with st.expander("Data coverage dan pipeline"):
        st.dataframe(_safe_display(result.get("scan_coverage_summary", pd.DataFrame())), width="stretch", hide_index=True)
        st.dataframe(_safe_display(result.get("scanner_data_contract_audit", pd.DataFrame())), width="stretch", hide_index=True)
        st.dataframe(_safe_display(result.get("stage_timings", pd.DataFrame())), width="stretch", hide_index=True)
        st.dataframe(_safe_display(result.get("two_stage_coverage_audit", pd.DataFrame())), width="stretch", hide_index=True)
    with st.expander("OHLCV database dan provider audit"):
        ohlcv_audit = _safe_display(result.get("ohlcv_database_audit", pd.DataFrame()))
        st.dataframe(ohlcv_audit, width="stretch", hide_index=True)
        if not ohlcv_audit.empty:
            st.download_button("Download OHLCV Audit CSV", ohlcv_audit.to_csv(index=False).encode("utf-8-sig"), "v9_ohlcv_audit.csv", "text/csv")
    with st.expander("ALL_ELIGIBLE_LITE feature cache"):
        feature_audit = _safe_display(result.get("feature_cache_audit", pd.DataFrame()))
        st.dataframe(feature_audit, width="stretch", hide_index=True)
        st.dataframe(_safe_display(result.get("feature_cache_write_report", pd.DataFrame())), width="stretch", hide_index=True)
        if not feature_audit.empty:
            st.download_button("Download Feature Cache Audit CSV", feature_audit.to_csv(index=False).encode("utf-8-sig"), "v9_feature_cache_audit.csv", "text/csv")
    with st.expander("Fundamental dan database audit"):
        st.dataframe(_safe_display(result.get("fundamental_history_report", pd.DataFrame())), width="stretch", hide_index=True)
        st.dataframe(_safe_display(result.get("database_sync_report", pd.DataFrame())), width="stretch", hide_index=True)
