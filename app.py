from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import streamlit as st

APP_VERSION = "9.8.10-swing-production-gate"
APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

st.set_page_config(page_title="IDX Scanner v9.8.10", page_icon="📊", layout="wide")

REQUIRED_FILES = (
    "scanner.py", "scanner_database.py", "narrative_engine.py", "macro_engine.py",
    "simple_focus.py", "free_data_providers.py", "ihsg_direction.py",
    "decision_overlay.py", "real_money_guard.py", "fundamental_calibration.py", "v9_dashboard.py",
    "resumable_app_engine.py", "fast_scan_engine.py", "evidence_enrichment.py",
)
missing = [name for name in REQUIRED_FILES if not (APP_ROOT / name).is_file()]
if missing:
    st.error("Deployment v9.8.10 tidak lengkap.")
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


st.title("IDX Super Scanner v9.8.10 — Swing Production Gate")
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
        "Top 3 Next Leader", "Top 3 Investable", "Top 3 Swing Ready", "Top 3 Swing Watch",
    ])
    with top_leader_tab:
        if top_leaders.empty:
            st.info("Belum ada kandidat Next Leader research pada scan ini.")
        else:
            st.components.v1.html(
                render_dashboard_html(
                    top_leaders, model="NEXT_LEADER", market_regime=market_regime,
                    scan_id=scan_id, as_of=as_of,
                    completeness_note="Research ranking — bukan izin order. Lihat Real Money Gate dan Production-qualified untuk status eksekusi.",
                ),
                height=930, scrolling=True,
            )
    with top_investable_tab:
        if top_investable.empty:
            st.info("Belum ada kandidat Next Leader yang melewati portfolio suitability gate.")
        else:
            st.components.v1.html(
                render_dashboard_html(
                    top_investable, model="NEXT_LEADER", market_regime=market_regime,
                    scan_id=scan_id, as_of=as_of,
                    completeness_note="Portfolio lane — sudah melewati liquidity/quality suitability, tetapi order tetap tunduk pada Real Money Gate.",
                ),
                height=930, scrolling=True,
            )
    with top_swing_tab:
        if top_swings.empty:
            st.info("Belum ada Swing Ready yang benar-benar actionable setelah price/execution gate.")
        else:
            st.components.v1.html(
                render_dashboard_html(
                    top_swings, model="SWING_READY", market_regime=market_regime,
                    scan_id=scan_id, as_of=as_of,
                    completeness_note="Actionable lane — hanya kandidat dengan executable entry/trigger dan guardrail produksi.",
                ),
                height=930, scrolling=True,
            )
    with top_swing_watch_tab:
        if top_swing_watch.empty:
            st.info("Belum ada kandidat Swing research/watch pada scan ini.")
        else:
            st.components.v1.html(
                render_dashboard_html(
                    top_swing_watch, model="SWING_READY", market_regime=market_regime,
                    scan_id=scan_id, as_of=as_of,
                    completeness_note="Research/watch lane — boleh dipantau tetapi belum tentu memiliki order yang valid saat ini.",
                ),
                height=930, scrolling=True,
            )

with market_tab:
    st.subheader("Market & Sector Map")
    st.caption("Macro hanya memengaruhi Macro-Sector Fit dan risk overlay. Macro tidak mengganti seleksi emiten.")
    safe_dataframe(macro_snapshot, width="stretch", hide_index=True)
    market_context = result.get("market_context", pd.DataFrame())
    if isinstance(market_context, pd.DataFrame) and not market_context.empty:
        st.markdown("#### Market context")
        safe_dataframe(market_context, width="stretch", hide_index=True)
    sector_map = result.get("sector_map", pd.DataFrame())
    if isinstance(sector_map, pd.DataFrame) and not sector_map.empty:
        st.markdown("#### Sector Opportunity Map")
        safe_dataframe(sector_map, width="stretch", hide_index=True)

with leader_tab:
    st.subheader("The Next Leader")
    if leaders.empty:
        st.info("Tidak ada kandidat Next Leader yang memenuhi minimum evidence hari ini.")
    else:
        show_cols = [
            "research_rank", "production_rank", "ticker", "thesis_archetype", "Ranking Score", "Production Score", "v9_next_leader_score", "v9_next_leader_raw_score", "score_inflation_guard_active", "score_inflation_guard_reason",
            "business_quality_score", "business_quality_coverage_pct", "future_fundamental_score", "future_fundamental_coverage_pct", "valuation_mos_score", "management_capital_allocation_score", "macro_sector_fit_score", "narrative_moneyflow_score", "technical_readiness_score",
            "silent_accumulation_score", "silent_accumulation_confidence", "silent_accumulation_state", "inventory_cycle_score", "inventory_cycle_coverage_pct", "inventory_cycle_phase", "broker_inventory_shift_state", "distribution_penalty", "strong_accumulation_flag",
            "growth_lane_rank_eligible", "turnaround_lane_rank_eligible", "growth_lane_reject_reason", "turnaround_lane_reject_reason",
            "status", "thesis_state", "execution_state", "decision_overlay_state", "real_money_authorization_state", "real_money_authorized", "real_money_gate_reasons", "actionability_score", "actionability_state", "actionability_block_reasons", "next_action",
            "entry", "entry_low", "entry_high", "trigger", "stop_loss", "tp1", "tp2", "rr1", "rr2", "recommended_lots", "recommended_allocation_idr", "allocation_action",
            "execution_entry_type", "stockbit_order_instruction", "stockbit_trigger_price", "stockbit_limit_price", "stockbit_order_lots", "order_builder_eligible",
            "research_accumulation_zone_low", "research_accumulation_zone_high", "research_accumulation_reference", "research_zone_state", "research_zone_note",
            "portfolio_rank_eligible", "portfolio_gate_reason", "production_rank_eligible", "production_gate_reason", "ranking_tier", "ranking_reason", "thesis_reason", "next_proof", "invalidation",
            "fundamental_data_state", "fundamental_freshness_class", "fundamental_statement_age_days", "fundamental_latest_period_end", "growth_period_alignment_state", "fundamental_data_grade", "fundamental_completeness",
            "current_debt", "long_term_debt", "total_debt", "cash", "cash_to_debt", "debt_equity", "gross_margin", "operating_margin", "net_margin", "roe", "roa", "cash_conversion_ttm", "fcf_yield",
            "smc_state", "smc_sweep_state", "smc_bos_state", "smc_choch_state", "smc_fvg_state", "fvg_low", "fvg_high", "order_block_low", "order_block_high", "liquidity_sweep_level", "technical_reason", "narrative_reason",
            "idx_official_coverage_pct", "idx_official_evidence_status", "idx_official_state",
            "evidence_coverage_pct", "evidence_class", "top_evidence_sources", "top_reason", "top_risk", "top_catalyst", "expected_return_1_3m_pct", "probability_1_3m_pct", "holding_horizon", "estimated_days_to_entry", "estimated_days_to_tp1", "estimated_days_to_tp2",
        ]
        safe_dataframe(leaders, show_cols, width="stretch", hide_index=True)

with swing_tab:
    st.subheader("Swing Ready")
    if swings.empty:
        st.info("Tidak ada setup swing yang memenuhi struktur + fundamental gate saat ini.")
    else:
        show_cols = [
            "research_rank", "production_rank", "ticker", "Ranking Score", "Production Score", "v9_swing_score", "v9_swing_raw_score", "score_inflation_guard_active", "score_inflation_guard_reason",
            "technical_execution_score", "macro_sector_fit_score", "narrative_moneyflow_score", "business_quality_score", "risk_data_score", "silent_accumulation_score", "silent_accumulation_confidence", "silent_accumulation_state", "inventory_cycle_score", "inventory_cycle_coverage_pct", "inventory_cycle_phase", "broker_inventory_shift_state", "distribution_penalty", "strong_accumulation_flag", "status", "setup_type", "entry_type", "thesis_state", "execution_state", "decision_overlay_state", "real_money_authorization_state", "real_money_authorized", "real_money_gate_reasons", "actionability_score", "actionability_state", "actionability_block_reasons", "next_action",
            "entry", "entry_low", "entry_high", "trigger", "stop_loss", "tp1", "tp2", "rr1", "rr2", "recommended_lots", "recommended_allocation_idr", "allocation_action",
            "execution_entry_type", "stockbit_order_instruction", "stockbit_trigger_price", "stockbit_limit_price", "stockbit_order_lots", "order_builder_eligible",
            "entry_zone_role", "entry_zone_is_executable", "research_accumulation_zone_low", "research_accumulation_zone_high", "research_accumulation_reference", "research_zone_state", "research_zone_note",
            "actionable_rank_eligible", "production_rank_eligible", "production_gate_reason", "ranking_tier", "ranking_reason", "fundamental_data_state", "fundamental_freshness_class", "fundamental_statement_age_days", "growth_period_alignment_state", "smc_state", "smc_sweep_state", "smc_bos_state", "smc_choch_state", "smc_fvg_state", "fvg_low", "fvg_high", "order_block_low", "order_block_high", "liquidity_sweep_level", "next_proof", "invalidation", "evidence_coverage_pct", "evidence_class", "top_evidence_sources", "top_reason", "top_risk", "top_catalyst", "expected_return_1_3m_pct", "probability_1_3m_pct", "holding_horizon", "estimated_days_to_entry", "estimated_days_to_tp1", "estimated_days_to_tp2",
        ]
        safe_dataframe(swings, show_cols, width="stretch", hide_index=True)

with portfolio_tab:
    st.subheader("Portfolio & Audit")
    portfolio_view = result.get("portfolio_view", pd.DataFrame())
    if isinstance(portfolio_view, pd.DataFrame) and not portfolio_view.empty:
        safe_dataframe(portfolio_view, width="stretch", hide_index=True)
    else:
        st.caption("Upload portfolio CSV untuk review posisi. Scanner universe tetap dapat berjalan tanpa portfolio.")
    st.markdown("#### Scanner Contract")
    contract = result.get("scanner_contract", pd.DataFrame())
    if isinstance(contract, pd.DataFrame) and not contract.empty:
        safe_dataframe(contract, width="stretch", hide_index=True)
    st.markdown("#### Scan Coverage")
    if isinstance(coverage_summary, pd.DataFrame) and not coverage_summary.empty:
        safe_dataframe(coverage_summary, width="stretch", hide_index=True)
    evidence_report = result.get("evidence_refresh_report", pd.DataFrame())
    if isinstance(evidence_report, pd.DataFrame) and not evidence_report.empty:
        st.markdown("#### Evidence Refresh Report")
        safe_dataframe(evidence_report, width="stretch", hide_index=True)
    backfill_report = result.get("maintenance_backfill_report", pd.DataFrame())
    if isinstance(backfill_report, pd.DataFrame) and not backfill_report.empty:
        st.markdown("#### Maintenance Backfill Report")
        safe_dataframe(backfill_report, width="stretch", hide_index=True)
    provider_audit = result.get("provider_audit", pd.DataFrame())
    if isinstance(provider_audit, pd.DataFrame) and not provider_audit.empty:
        st.markdown("#### Provider Audit")
        safe_dataframe(provider_audit, width="stretch", hide_index=True)
    gate_audit = result.get("gate_audit", pd.DataFrame())
    if isinstance(gate_audit, pd.DataFrame) and not gate_audit.empty:
        st.markdown("#### Decision Gate Audit")
        safe_dataframe(gate_audit, width="stretch", hide_index=True)
    revalidation = result.get("execution_revalidation_report", pd.DataFrame())
    if isinstance(revalidation, pd.DataFrame) and not revalidation.empty:
        st.markdown("#### Execution Revalidation")
        safe_dataframe(revalidation, width="stretch", hide_index=True)
    data_quality_report = result.get("data_quality_report", pd.DataFrame())
    if isinstance(data_quality_report, pd.DataFrame) and not data_quality_report.empty:
        st.markdown("#### Data Quality")
        safe_dataframe(data_quality_report, width="stretch", hide_index=True)
    database_sync = result.get("database_sync_report", pd.DataFrame())
    if isinstance(database_sync, pd.DataFrame) and not database_sync.empty:
        st.markdown("#### Database Sync Report")
        safe_dataframe(database_sync, width="stretch", hide_index=True)
    database_detail = result.get("database_sync_detail", pd.DataFrame())
    if isinstance(database_detail, pd.DataFrame) and not database_detail.empty:
        with st.expander("Database Sync Detail", expanded=False):
            safe_dataframe(database_detail, width="stretch", hide_index=True)
    stage_timings = result.get("stage_timings", pd.DataFrame())
    if isinstance(stage_timings, pd.DataFrame) and not stage_timings.empty:
        st.markdown("#### Stage Timings")
        safe_dataframe(stage_timings, width="stretch", hide_index=True)
