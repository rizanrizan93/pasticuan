from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import streamlit as st


APP_VERSION = "9.1.0-database-first"
APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

st.set_page_config(
    page_title="IDX Scanner v9 Macro-First",
    page_icon="📊",
    layout="wide",
)

REQUIRED_FILES = (
    "scanner.py",
    "scanner_database.py",
    "narrative_engine.py",
    "macro_engine.py",
    "simple_focus.py",
    "two_stage_pipeline.py",
    "free_data_providers.py",
    "ihsg_direction.py",
    "idx_trading_calendar.py",
    "incremental_store.py",
    "research_maintenance.py",
    "selector_engine.py",
    "database_first.py",
)
missing = [name for name in REQUIRED_FILES if not (APP_ROOT / name).is_file()]
if missing:
    st.error("Deployment v9 tidak lengkap.")
    st.code("\n".join(missing), language="text")
    st.stop()

from scanner import (  # noqa: E402
    ScanConfig,
    ScanEngine,
    analyze_portfolio_positions,
    apply_execution_snapshot_gate,
    apply_fundamental_gate,
    apply_independent_price_gate,
    apply_market_status_gate,
    apply_news_gate,
    apply_universe_integrity_gate,
    apply_validation_gate,
    attach_backtest_stats,
    attach_fundamentals,
    attach_ohlcv_source_lineage,
    attach_position_sizing,
    build_independent_price_validation,
    combine_fundamental_history,
    download_benchmark,
    download_ohlcv,
    enrich_fundamentals_with_history,
    fetch_automatic_independent_prices,
    fetch_execution_snapshots,
    fetch_idx_fundamental_history,
    fetch_resilient_fundamentals,
    fetch_resilient_market_status,
    fetch_resilient_news_review,
    fetch_yahoo_fundamental_history,
    finalize_execution_integrity,
    parse_portfolio_csv,
    parse_ticker_csv,
    read_cached_fundamental_history,
    read_cached_fundamentals,
    read_cached_market_status,
    read_cached_news_review,
    run_adaptive_walkforward_validation,
    select_yahoo_fundamental_tickers,
)
from scanner_database import ScannerDatabaseBridge  # noqa: E402
from narrative_engine import build_narrative_intelligence  # noqa: E402
from ihsg_direction import IHSGDirectionConfig, analyze_ihsg_direction  # noqa: E402
from macro_engine import (  # noqa: E402
    MACRO_ENGINE_VERSION,
    build_macro_regime,
    fetch_macro_series,
)
from simple_focus import SIMPLE_FOCUS_VERSION, build_simple_focus  # noqa: E402
from two_stage_pipeline import (  # noqa: E402
    ShortlistConfig,
    build_enrichment_shortlist,
    build_lightweight_preliminary_focus,
    build_two_stage_coverage_audit,
)
from database_first import (  # noqa: E402
    DATABASE_FIRST_VERSION,
    DatabaseReadinessPolicy,
    build_database_coverage,
    estimate_remaining_passes,
    readiness_summary,
    select_database_refresh_queue,
)


st.markdown(
    """
    <style>
      .block-container {padding-top: 1rem; padding-bottom: 2.5rem;}
      [data-testid="stMetricValue"] {font-size: 1.35rem;}
      .v9-note {border:1px solid #334155; border-radius:10px; padding:12px 14px; background:#0f172a;}
      .small-muted {font-size:.84rem; color:#94a3b8;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _finite(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _merge_prefer_primary(primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    frames = [frame.copy() for frame in (primary, fallback) if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0].reset_index(drop=True)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "ticker" not in combined.columns:
        return combined
    combined["ticker"] = combined["ticker"].astype(str).str.upper().str.strip()
    combined["_priority"] = np.concatenate([np.full(len(frames[0]), 2), np.full(len(frames[1]), 1)])
    combined = combined.sort_values(["ticker", "_priority"], ascending=[True, False], kind="stable")
    return combined.drop_duplicates("ticker", keep="first").drop(columns="_priority").reset_index(drop=True)


def _bounded_histories(histories: Mapping[str, pd.DataFrame], *, max_bars: int) -> dict[str, pd.DataFrame]:
    limit = max(260, int(max_bars))
    return {
        str(ticker): frame.tail(limit).copy()
        for ticker, frame in histories.items()
        if isinstance(frame, pd.DataFrame) and not frame.empty
    }


def _primary_reference(histories: Mapping[str, pd.DataFrame]) -> dict[str, tuple[object, float]]:
    output: dict[str, tuple[object, float]] = {}
    for ticker, frame in histories.items():
        if not isinstance(frame, pd.DataFrame) or frame.empty or "Close" not in frame:
            continue
        close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
        if close.empty:
            continue
        output[str(ticker).upper()] = (frame.index[-1], float(close.iloc[-1]))
    return output


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


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_market_data(tickers: tuple[str, ...], period: str, itick_token: str):
    histories, report = download_ohlcv(tickers, period=period, itick_api_token=itick_token)
    benchmark = download_benchmark(period=period)
    return histories, report, benchmark


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_macro_data():
    return fetch_macro_series(period="6mo", timeout=12)


@st.cache_data(ttl=900, show_spinner=False)
def _cached_baseline_fundamentals(tickers: tuple[str, ...]):
    cfg = ScanConfig()
    bridge = ScannerDatabaseBridge()
    try:
        db_snapshot, db_snapshot_audit = bridge.read_fundamental_cache(list(tickers))
    except Exception as exc:
        db_snapshot, db_snapshot_audit = pd.DataFrame(), pd.DataFrame([{
            "provider": "SUPABASE_DATABASE_FIRST", "status": "READ_FAIL_SOFT",
            "error": f"{type(exc).__name__}: {str(exc)[:160]}",
        }])
    try:
        db_history, db_history_audit = bridge.read_fundamental_history_cache(list(tickers))
    except Exception as exc:
        db_history, db_history_audit = pd.DataFrame(), pd.DataFrame([{
            "provider": "SUPABASE_DATABASE_FIRST", "status": "READ_FAIL_SOFT",
            "error": f"{type(exc).__name__}: {str(exc)[:160]}",
        }])
    local_snapshot = read_cached_fundamentals(tickers, cfg)
    if not local_snapshot.empty and "fundamental_score_eligible" in local_snapshot:
        local_snapshot = local_snapshot.loc[local_snapshot["fundamental_score_eligible"].fillna(False).astype(bool)].copy()
    local_history = read_cached_fundamental_history(tickers)
    snapshot = _merge_prefer_primary(local_snapshot, db_snapshot)
    history = combine_fundamental_history(db_history, local_history)
    enriched = enrich_fundamentals_with_history(snapshot, history)
    cache_audit = pd.DataFrame([{
        "provider": "LOCAL_CACHE_ONLY", "status": "BASELINE_CACHE_READ",
        "requested_tickers": len(tickers),
        "snapshot_tickers": int(enriched["ticker"].nunique()) if not enriched.empty and "ticker" in enriched else 0,
        "history_tickers": int(history["ticker"].nunique()) if not history.empty and "ticker" in history else 0,
    }])
    reports = [frame for frame in (db_snapshot_audit, db_history_audit, cache_audit)
               if isinstance(frame, pd.DataFrame) and not frame.empty]
    return enriched, history, pd.concat(reports, ignore_index=True, sort=False) if reports else pd.DataFrame()



@st.cache_data(ttl=900, show_spinner=False)
def _cached_narrative_memory(tickers: tuple[str, ...]):
    bridge = ScannerDatabaseBridge()
    try:
        events = bridge.read_narrative_events(list(tickers))
    except Exception:
        events = pd.DataFrame()
    try:
        outcomes = bridge.read_narrative_event_outcomes(list(tickers))
    except Exception:
        outcomes = pd.DataFrame()
    return events, outcomes

def _enrich_shortlist_fundamentals(
    *,
    all_tickers: tuple[str, ...],
    shortlist: tuple[str, ...],
    official_refresh: tuple[str, ...],
    baseline_snapshot: pd.DataFrame,
    baseline_history: pd.DataFrame,
    baseline_report: pd.DataFrame,
    cfg: ScanConfig,
    yahoo_limit: int = 24,
):
    live_snapshot = fetch_resilient_fundamentals(shortlist, cfg)
    if not live_snapshot.empty and "fundamental_score_eligible" in live_snapshot:
        usable_live = live_snapshot.loc[live_snapshot["fundamental_score_eligible"].fillna(False).astype(bool)].copy()
    else:
        usable_live = live_snapshot
    snapshot = _merge_prefer_primary(usable_live, baseline_snapshot)
    idx_history, idx_report = fetch_idx_fundamental_history(
        official_refresh,
        max_tickers=len(official_refresh),
        years_back=max(1, int(getattr(cfg, "idx_fundamental_years_back", 3))),
    )
    history_before_yahoo = combine_fundamental_history(baseline_history, idx_history)
    yahoo_targets = select_yahoo_fundamental_tickers(
        shortlist,
        history_before_yahoo,
        max_tickers=max(0, min(int(yahoo_limit), len(shortlist))),
        crosscheck_top_n=min(8, len(shortlist)),
    )
    yahoo_history, yahoo_report = fetch_yahoo_fundamental_history(yahoo_targets, max_tickers=len(yahoo_targets))
    history = combine_fundamental_history(baseline_history, idx_history, yahoo_history)
    enriched = enrich_fundamentals_with_history(snapshot, history)
    snapshot_set = set(enriched.get("ticker", pd.Series(dtype=str)).dropna().astype(str)) if not enriched.empty else set()
    history_set = set(history.get("ticker", pd.Series(dtype=str)).dropna().astype(str)) if not history.empty else set()
    refresh_state_rows = []
    for ticker in shortlist:
        refresh_state_rows.extend([
            {
                "ticker": ticker, "provider": "DATABASE_DELTA_REFRESH", "scope": "FUNDAMENTAL_SNAPSHOT",
                "status": "CURRENT" if ticker in snapshot_set else "REFRESH_UNRESOLVED",
                "database_read_state": "LIVE_OR_CACHE_CURRENT" if ticker in snapshot_set else "REFRESH_UNRESOLVED",
                "refresh_required": ticker not in snapshot_set, "rows": int(ticker in snapshot_set),
            },
            {
                "ticker": ticker, "provider": "DATABASE_DELTA_REFRESH", "scope": "FUNDAMENTAL_HISTORY",
                "status": "CURRENT" if ticker in history_set else "REFRESH_UNRESOLVED",
                "database_read_state": "LIVE_OR_CACHE_CURRENT" if ticker in history_set else "REFRESH_UNRESOLVED",
                "refresh_required": ticker not in history_set, "rows": int(ticker in history_set),
            },
        ])
    reports = [frame for frame in (
        baseline_report, idx_report, yahoo_report,
        pd.DataFrame(refresh_state_rows),
        pd.DataFrame([{
            "provider": "DATABASE_FIRST_PLANNER", "status": "DELTA_REFRESH",
            "universe_tickers": len(all_tickers), "refresh_tickers": len(shortlist),
            "official_refresh_tickers": len(official_refresh), "yahoo_history_tickers": len(yahoo_targets),
        }]),
    ) if isinstance(frame, pd.DataFrame) and not frame.empty]
    return enriched, history, pd.concat(reports, ignore_index=True, sort=False) if reports else pd.DataFrame()


def _simple_contract_audit(
    tickers: list[str],
    prepared: Mapping[str, pd.DataFrame],
    fundamentals: pd.DataFrame,
    focus: Mapping[str, pd.DataFrame],
    macro_issuer: pd.DataFrame,
) -> pd.DataFrame:
    fund_set = set(fundamentals.get("ticker", pd.Series(dtype=str)).astype(str)) if not fundamentals.empty else set()
    leader = focus.get("next_leaders", pd.DataFrame())
    swing = focus.get("swing_ready", pd.DataFrame())
    leader_map = leader.set_index("ticker").to_dict("index") if not leader.empty else {}
    swing_map = swing.set_index("ticker").to_dict("index") if not swing.empty else {}
    macro_set = set(macro_issuer.get("ticker", pd.Series(dtype=str)).astype(str)) if not macro_issuer.empty else set()
    rows = []
    for ticker in tickers:
        key = str(ticker).upper()
        rows.append({
            "ticker": key,
            "ohlcv_ready": key in prepared,
            "fundamental_ready": key in fund_set,
            "macro_issuer_ready": key in macro_set,
            "next_leader_score": leader_map.get(key, {}).get("v9_next_leader_score", np.nan),
            "next_leader_status": leader_map.get(key, {}).get("status", "DATA_PENDING"),
            "swing_score": swing_map.get(key, {}).get("v9_swing_score", np.nan),
            "swing_status": swing_map.get(key, {}).get("status", "DATA_PENDING"),
        })
    return pd.DataFrame(rows)



def _run_database_backfill(
    *,
    all_tickers: list[str],
    portfolio_tickers: list[str],
    batch_size: int,
    yahoo_limit: int,
    cfg: ScanConfig,
) -> dict[str, object]:
    started = time.perf_counter()
    baseline_fundamentals, baseline_history, baseline_report = _cached_baseline_fundamentals(tuple(all_tickers))
    cached_market_status = read_cached_market_status(all_tickers, cfg)
    cached_news_review = read_cached_news_review(all_tickers, lookback_days=45, config=cfg)
    existing_narrative_events, existing_narrative_outcomes = _cached_narrative_memory(tuple(all_tickers))
    coverage_before = build_database_coverage(
        all_tickers,
        fundamentals=baseline_fundamentals,
        fundamental_history=baseline_history,
        market_status=cached_market_status,
        news_review=cached_news_review,
        fundamental_report=baseline_report,
    )
    summary_before = readiness_summary(coverage_before)
    refresh_tickers, refresh_audit = select_database_refresh_queue(
        all_tickers,
        coverage_before,
        batch_size=max(0, int(batch_size)),
        portfolio_tickers=portfolio_tickers,
    )
    if not refresh_tickers:
        return {
            "state": "DATABASE_ALREADY_CURRENT",
            "coverage_before": coverage_before,
            "coverage_after": coverage_before,
            "summary_before": summary_before,
            "summary_after": summary_before,
            "refresh_queue": refresh_audit,
            "fundamentals": baseline_fundamentals,
            "fundamental_history": baseline_history,
            "market_status": cached_market_status,
            "news_review": cached_news_review,
            "fundamental_report": baseline_report,
            "database_sync_report": pd.DataFrame(),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "remaining_passes": 0,
        }

    live_snapshot = fetch_resilient_fundamentals(refresh_tickers, cfg)
    if not live_snapshot.empty and "fundamental_score_eligible" in live_snapshot.columns:
        usable_live = live_snapshot.loc[live_snapshot["fundamental_score_eligible"].fillna(False).astype(bool)].copy()
    else:
        usable_live = live_snapshot
    snapshot = _merge_prefer_primary(usable_live, baseline_fundamentals)

    idx_history, idx_report = fetch_idx_fundamental_history(
        refresh_tickers,
        max_tickers=len(refresh_tickers),
        years_back=max(1, int(getattr(cfg, "idx_fundamental_years_back", 3))),
    )
    history_before_yahoo = combine_fundamental_history(baseline_history, idx_history)
    yahoo_targets = select_yahoo_fundamental_tickers(
        refresh_tickers,
        history_before_yahoo,
        max_tickers=max(0, min(int(yahoo_limit), len(refresh_tickers))),
        crosscheck_top_n=min(8, len(refresh_tickers)),
    )
    yahoo_history, yahoo_report = fetch_yahoo_fundamental_history(
        yahoo_targets,
        max_tickers=len(yahoo_targets),
    )
    history = combine_fundamental_history(baseline_history, idx_history, yahoo_history)
    fundamentals = enrich_fundamentals_with_history(snapshot, history)

    live_market_status = fetch_resilient_market_status(refresh_tickers, cfg)
    live_news_review = fetch_resilient_news_review(refresh_tickers, lookback_days=45, config=cfg)
    market_status = _merge_prefer_primary(live_market_status, cached_market_status)
    news_review = _merge_prefer_primary(live_news_review, cached_news_review)

    snapshot_set = set(fundamentals.get("ticker", pd.Series(dtype=str)).dropna().astype(str)) if not fundamentals.empty else set()
    history_set = set(history.get("ticker", pd.Series(dtype=str)).dropna().astype(str)) if not history.empty else set()
    refresh_state_rows = []
    for ticker in refresh_tickers:
        refresh_state_rows.extend([
            {
                "ticker": ticker, "provider": "DATABASE_FIRST_BACKFILL", "scope": "FUNDAMENTAL_SNAPSHOT",
                "status": "CURRENT" if ticker in snapshot_set else "REFRESH_UNRESOLVED",
                "database_read_state": "LIVE_OR_CACHE_CURRENT" if ticker in snapshot_set else "REFRESH_UNRESOLVED",
                "refresh_required": ticker not in snapshot_set, "rows": int(ticker in snapshot_set),
            },
            {
                "ticker": ticker, "provider": "DATABASE_FIRST_BACKFILL", "scope": "FUNDAMENTAL_HISTORY",
                "status": "CURRENT" if ticker in history_set else "REFRESH_UNRESOLVED",
                "database_read_state": "LIVE_OR_CACHE_CURRENT" if ticker in history_set else "REFRESH_UNRESOLVED",
                "refresh_required": ticker not in history_set, "rows": int(ticker in history_set),
            },
        ])
    report_frames = [
        frame for frame in (
            baseline_report,
            idx_report,
            yahoo_report,
            pd.DataFrame(refresh_state_rows),
            pd.DataFrame([{
                "provider": "DATABASE_FIRST_BACKFILL",
                "status": "REFRESH_BATCH_COMPLETED",
                "requested_tickers": len(all_tickers),
                "refresh_tickers": len(refresh_tickers),
                "yahoo_history_tickers": len(yahoo_targets),
                "scope": "DATABASE_BACKFILL",
            }]),
        )
        if isinstance(frame, pd.DataFrame) and not frame.empty
    ]
    fundamental_report = pd.concat(report_frames, ignore_index=True, sort=False) if report_frames else pd.DataFrame()

    narrative = build_narrative_intelligence(
        prepared={ticker: pd.DataFrame() for ticker in refresh_tickers},
        fundamentals=fundamentals.loc[fundamentals["ticker"].isin(refresh_tickers)].copy() if not fundamentals.empty and "ticker" in fundamentals else fundamentals,
        news_review=news_review.loc[news_review["ticker"].isin(refresh_tickers)].copy() if not news_review.empty and "ticker" in news_review else news_review,
        market_status=market_status.loc[market_status["ticker"].isin(refresh_tickers)].copy() if not market_status.empty and "ticker" in market_status else market_status,
        existing_events=existing_narrative_events,
        existing_outcomes=existing_narrative_outcomes,
        scan_config=cfg,
    )

    bridge = ScannerDatabaseBridge()
    persist_result = {
        "mode": "database_first_backfill",
        "scanner_version": APP_VERSION,
        "fundamentals": fundamentals,
        "fundamental_history": history,
        "fundamental_history_report": fundamental_report,
        "database_read_report": baseline_report,
        "market_status": market_status,
        "news_review": news_review,
        "narrative_events": narrative.get("events", pd.DataFrame()),
        "narrative_event_outcomes": narrative.get("outcomes", pd.DataFrame()),
        "narrative_profiles": narrative.get("profiles", pd.DataFrame()),
        "focus_screens": {
            "narrative_events": narrative.get("events", pd.DataFrame()),
            "narrative_event_outcomes": narrative.get("outcomes", pd.DataFrame()),
            "narrative_profiles": narrative.get("profiles", pd.DataFrame()),
        },
    }
    try:
        database_sync_report = bridge.persist_scan_result(persist_result)
    except Exception as exc:
        database_sync_report = pd.DataFrame([{
            "state": "DATABASE_FAIL_SOFT",
            "detail": f"{type(exc).__name__}: {str(exc)[:300]}",
        }])

    coverage_after = build_database_coverage(
        all_tickers,
        fundamentals=fundamentals,
        fundamental_history=history,
        market_status=market_status,
        news_review=news_review,
        fundamental_report=fundamental_report,
    )
    summary_after = readiness_summary(coverage_after)
    try:
        _cached_baseline_fundamentals.clear()
        _cached_narrative_memory.clear()
    except Exception:
        pass
    return {
        "state": str(summary_after.iloc[0].get("database_state", "BACKFILL_REQUIRED")),
        "coverage_before": coverage_before,
        "coverage_after": coverage_after,
        "summary_before": summary_before,
        "summary_after": summary_after,
        "refresh_queue": refresh_audit,
        "refreshed_tickers": pd.DataFrame({"ticker": refresh_tickers}),
        "fundamentals": fundamentals,
        "fundamental_history": history,
        "market_status": market_status,
        "news_review": news_review,
        "fundamental_report": fundamental_report,
        "narrative_engine_audit": narrative.get("audit", pd.DataFrame()),
        "database_sync_report": database_sync_report,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "remaining_passes": estimate_remaining_passes(coverage_after, batch_size),
    }


def _render_database_backfill(result: Mapping[str, object]) -> None:
    summary_before = result.get("summary_before", pd.DataFrame())
    summary_after = result.get("summary_after", pd.DataFrame())
    before = summary_before.iloc[0] if isinstance(summary_before, pd.DataFrame) and not summary_before.empty else {}
    after = summary_after.iloc[0] if isinstance(summary_after, pd.DataFrame) and not summary_after.empty else {}
    cols = st.columns(5)
    cols[0].metric("Snapshot", f"{_finite(after.get('fundamental_snapshot_pct')):.1f}%", f"{_finite(after.get('fundamental_snapshot_pct')) - _finite(before.get('fundamental_snapshot_pct')):+.1f} pp")
    cols[1].metric("History", f"{_finite(after.get('fundamental_history_pct')):.1f}%", f"{_finite(after.get('fundamental_history_pct')) - _finite(before.get('fundamental_history_pct')):+.1f} pp")
    cols[2].metric("Market status", f"{_finite(after.get('market_status_pct')):.1f}%")
    cols[3].metric("News review", f"{_finite(after.get('news_review_pct')):.1f}%")
    cols[4].metric("Estimasi proses tersisa", int(_finite(result.get("remaining_passes"), 0)))
    state = str(result.get("state", "BACKFILL_REQUIRED"))
    if state == "READY_FOR_DAILY_SCAN" or state == "DATABASE_ALREADY_CURRENT":
        st.success("Database telah memenuhi ambang Daily Scan.")
    else:
        st.warning("Database belum memenuhi ambang Daily Scan. Jalankan backfill kembali sampai status READY_FOR_DAILY_SCAN.")
    st.dataframe(_safe_display(summary_after), width="stretch", hide_index=True)
    with st.expander("Refresh queue"):
        st.dataframe(_safe_display(result.get("refresh_queue", pd.DataFrame())), width="stretch", hide_index=True)
    with st.expander("Coverage per ticker"):
        st.dataframe(_safe_display(result.get("coverage_after", pd.DataFrame())), width="stretch", hide_index=True)
    with st.expander("Provider dan database audit"):
        st.dataframe(_safe_display(result.get("fundamental_report", pd.DataFrame())), width="stretch", hide_index=True)
        st.dataframe(_safe_display(result.get("narrative_engine_audit", pd.DataFrame())), width="stretch", hide_index=True)
        st.dataframe(_safe_display(result.get("database_sync_report", pd.DataFrame())), width="stretch", hide_index=True)
    st.caption(f"Waktu backfill: {_finite(result.get('elapsed_seconds')):.1f} detik")

st.title("IDX Super Scanner v9 — Macro-First")
st.caption(
    f"{APP_VERSION} • database-first {DATABASE_FIRST_VERSION} • macro {MACRO_ENGINE_VERSION} • decision {SIMPLE_FOCUS_VERSION}"
)
st.markdown(
    """
    <div class="v9-note">
      <b>Alur keputusan disederhanakan</b><br>
      Database persisten diisi terlebih dahulu. Daily Scan membaca evidence seluruh universe dari database,
      hanya memperbarui data MISSING/STALE, lalu menjalankan macro → sektor → kualitas bisnis dan future fundamental
      → narrative–money flow → SMC/ICT execution. Output tetap <b>The Next Leader</b> dan <b>Swing Ready</b>.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Input")
    ticker_file = st.file_uploader("Universe ticker CSV", type=["csv"])
    portfolio_file = st.file_uploader("Portfolio CSV (opsional)", type=["csv"])
    operation_mode = st.radio("Mode kerja", ["Isi Database", "Scan Harian"], horizontal=True)
    account_size = float(st.number_input("Nilai akun (Rp)", min_value=0, value=5_000_000, step=500_000))
    cash_on_hand = float(st.number_input("Cash tersedia (Rp)", min_value=0, value=1_000_000, step=100_000))
    with st.expander("Pengaturan lanjutan"):
        period = st.selectbox("Riwayat OHLCV", ["3y", "5y", "10y"], index=1)
        risk_per_trade_pct = st.slider("Risiko per transaksi", 0.25, 2.00, 0.75, 0.25) / 100.0
        if operation_mode == "Isi Database":
            backfill_batch_size = st.slider("Backfill per proses", 20, 160, 80, 20)
            daily_refresh_quota = 0
        else:
            daily_refresh_quota = st.slider("Refresh MISSING/STALE per scan", 0, 60, 20, 10)
            backfill_batch_size = 0
        yahoo_history_limit = st.slider("Yahoo history fallback", 0, 40, 24, 4)
        run_oos = st.checkbox("Chronological OOS", value=False, disabled=operation_mode == "Isi Database")
        allow_partial_database = st.checkbox("Izinkan scan saat database belum siap", value=False, disabled=operation_mode == "Isi Database")
        itick_token = st.text_input("iTick API token", value=os.getenv("ITICK_API_TOKEN", ""), type="password")
        twelve_token = st.text_input("Twelve Data API key", value=os.getenv("TWELVE_DATA_API_KEY", ""), type="password")
    button_label = "Isi / Lanjutkan Database" if operation_mode == "Isi Database" else "Jalankan Daily Scan"
    run_scan = st.button(button_label, type="primary", width="stretch")


if run_scan:
    if ticker_file is None:
        st.error("Upload CSV universe terlebih dahulu.")
        st.stop()
    try:
        tickers = parse_ticker_csv(ticker_file, max_tickers=400, strict_limit=True)
    except Exception as exc:
        st.error(f"CSV ticker tidak valid: {exc}")
        st.stop()
    if not tickers:
        st.error("Tidak ada ticker IDX yang valid.")
        st.stop()

    portfolio = pd.DataFrame()
    if portfolio_file is not None:
        try:
            portfolio = parse_portfolio_csv(portfolio_file)
        except Exception as exc:
            st.error(f"Portfolio CSV tidak valid: {exc}")
            st.stop()
    portfolio_tickers = portfolio["ticker"].dropna().astype(str).drop_duplicates().tolist() if not portfolio.empty and "ticker" in portfolio else []
    max_total_tickers = 400
    if len(portfolio_tickers) > max_total_tickers:
        st.error(f"Portfolio memuat {len(portfolio_tickers)} ticker; maksimum total scanner adalah {max_total_tickers}.")
        st.stop()
    # Portfolio diprioritaskan agar seluruh posisi tetap dianalisis. Universe lain
    # mengisi sisa kapasitas sampai batas total 400, bukan 400 + portfolio.
    combined_order = list(dict.fromkeys(portfolio_tickers + tickers))
    dropped_tickers = combined_order[max_total_tickers:]
    all_tickers = combined_order[:max_total_tickers]
    if dropped_tickers:
        st.warning(
            f"Total universe + portfolio melebihi {max_total_tickers}. "
            f"Sebanyak {len(dropped_tickers)} ticker non-prioritas tidak dipindai."
        )

    cfg = ScanConfig(
        account_size_idr=account_size,
        cash_on_hand_idr=cash_on_hand,
        risk_per_trade_pct=risk_per_trade_pct,
        ai_enabled=False,
        ai_max_weight=0.0,
        selector_max_ai_weight=0.0,
        time_cycle_enabled=False,
        time_cycle_core_max_weight=0.0,
        time_cycle_multibagger_max_weight=0.0,
        eoff_enabled=False,
        eoff_ephemeris_enabled=False,
    )

    if operation_mode == "Isi Database":
        progress = st.progress(0, text="Membaca database dan menyusun refresh queue…")
        progress.progress(15, text="Mengambil hanya evidence MISSING/STALE…")
        backfill_result = _run_database_backfill(
            all_tickers=all_tickers,
            portfolio_tickers=portfolio_tickers,
            batch_size=backfill_batch_size,
            yahoo_limit=yahoo_history_limit,
            cfg=cfg,
        )
        progress.progress(100, text="Backfill database selesai")
        progress.empty()
        st.session_state["v9_database_backfill"] = backfill_result
        _render_database_backfill(backfill_result)
        st.stop()

    # Daily Scan starts from the persistent database.  Live providers are used
    # only for a bounded MISSING/STALE refresh queue.
    baseline_fundamentals, baseline_history, baseline_report = _cached_baseline_fundamentals(tuple(all_tickers))
    cached_market_status = read_cached_market_status(all_tickers, cfg)
    cached_news_review = read_cached_news_review(all_tickers, lookback_days=45, config=cfg)
    existing_narrative_events, existing_narrative_outcomes = _cached_narrative_memory(tuple(all_tickers))
    database_coverage_before = build_database_coverage(
        all_tickers,
        fundamentals=baseline_fundamentals,
        fundamental_history=baseline_history,
        market_status=cached_market_status,
        news_review=cached_news_review,
        fundamental_report=baseline_report,
    )
    database_summary_before = readiness_summary(database_coverage_before)
    database_state = str(database_summary_before.iloc[0].get("database_state", "BACKFILL_REQUIRED")) if not database_summary_before.empty else "BACKFILL_REQUIRED"
    if database_state != "READY_FOR_DAILY_SCAN" and not allow_partial_database:
        st.error("Database belum memenuhi ambang Daily Scan. Jalankan mode Isi Database terlebih dahulu.")
        st.dataframe(_safe_display(database_summary_before), width="stretch", hide_index=True)
        st.stop()
    refresh_tickers, refresh_plan = select_database_refresh_queue(
        all_tickers,
        database_coverage_before,
        batch_size=max(0, int(daily_refresh_quota)),
        portfolio_tickers=portfolio_tickers,
    )

    timings: list[dict[str, object]] = []
    started = time.perf_counter()

    def record(stage: str, stage_started: float, detail: str = "") -> None:
        timings.append({"stage": stage, "elapsed_seconds": round(time.perf_counter() - stage_started, 3), "detail": detail})

    progress = st.progress(0, text="Mengambil OHLCV, IHSG, dan macro market series…")
    stage = time.perf_counter()
    histories, download_report, benchmark = _cached_market_data(tuple(all_tickers), period, itick_token)
    macro_series, macro_source_report = _cached_macro_data()
    record("MARKET_AND_MACRO_DATA", stage, f"requested={len(all_tickers)}")

    progress.progress(18, text="Menghitung indikator, market structure, SMC/ICT, dan silent-flow inputs…")
    stage = time.perf_counter()
    daily_bar_limit = 750
    analysis_histories = histories if run_oos else _bounded_histories(histories, max_bars=daily_bar_limit)
    analysis_benchmark = benchmark if run_oos or not isinstance(benchmark, pd.DataFrame) else benchmark.tail(daily_bar_limit).copy()
    core = ScanEngine(cfg).scan(analysis_histories, analysis_benchmark)
    signals_base = apply_universe_integrity_gate(core.get("signals", pd.DataFrame()), all_tickers, core.get("prepared", {}).keys(), cfg)
    signals_base = attach_ohlcv_source_lineage(signals_base, getattr(download_report, "source_tiers", {}) or {})
    record("TECHNICAL_CORE", stage, f"prepared={len(core.get('prepared', {}))}")

    stats = pd.DataFrame()
    trades = pd.DataFrame()
    if run_oos and core.get("prepared"):
        progress.progress(27, text="Menjalankan chronological OOS…")
        stage = time.perf_counter()
        stats, trades, _ = run_adaptive_walkforward_validation(core["prepared"], cfg, initial_tickers=min(80, len(core["prepared"])))
        record("OOS", stage, f"events={len(trades)}")
    signals_base = apply_validation_gate(attach_backtest_stats(signals_base, stats), cfg)

    progress.progress(35, text="Membaca evidence persisten dan menyusun candidate verification…")
    stage = time.perf_counter()
    preliminary_signals = apply_news_gate(
        apply_market_status_gate(
            apply_fundamental_gate(attach_fundamentals(signals_base, baseline_fundamentals), cfg),
            cached_market_status, cfg,
        ),
        cached_news_review, cfg,
    )
    preliminary_focus = build_lightweight_preliminary_focus(
        core.get("prepared", {}), fundamentals=baseline_fundamentals, signals=preliminary_signals,
    )
    verification_cap = min(40, len(all_tickers), max(1, int(cfg.max_automatic_price_candidates)))
    verification_tickers, shortlist_audit = build_enrichment_shortlist(
        all_tickers,
        preliminary_focus=preliminary_focus,
        signals=preliminary_signals,
        portfolio_tickers=portfolio_tickers,
        config=ShortlistConfig(
            max_tickers=verification_cap,
            multibagger_quota=max(1, verification_cap // 2),
            core_quota=max(1, verification_cap // 3),
            technical_rescue_quota=max(2, verification_cap // 6),
        ),
    )
    record(
        "DATABASE_READ_AND_ROUTING", stage,
        f"database_state={database_state}; refresh={len(refresh_tickers)}; verification={len(verification_tickers)}",
    )

    progress.progress(50, text="Memperbarui hanya evidence MISSING/STALE…")
    stage = time.perf_counter()
    if refresh_tickers:
        fundamentals, fundamental_history, fundamental_report = _enrich_shortlist_fundamentals(
            all_tickers=tuple(all_tickers),
            shortlist=tuple(refresh_tickers),
            official_refresh=tuple(refresh_tickers),
            baseline_snapshot=baseline_fundamentals,
            baseline_history=baseline_history,
            baseline_report=baseline_report,
            cfg=cfg,
            yahoo_limit=min(int(yahoo_history_limit), len(refresh_tickers)),
        )
        live_market_status = fetch_resilient_market_status(refresh_tickers, cfg)
        live_news_review = fetch_resilient_news_review(refresh_tickers, lookback_days=45, config=cfg)
        market_status = _merge_prefer_primary(live_market_status, cached_market_status)
        news_review = _merge_prefer_primary(live_news_review, cached_news_review)
    else:
        fundamentals = baseline_fundamentals
        fundamental_history = baseline_history
        fundamental_report = baseline_report
        market_status = cached_market_status
        news_review = cached_news_review
    signals = apply_news_gate(
        apply_market_status_gate(
            apply_fundamental_gate(attach_fundamentals(signals_base, fundamentals), cfg),
            market_status, cfg,
        ),
        news_review, cfg,
    )
    record("DATABASE_DELTA_REFRESH", stage, f"refresh={len(refresh_tickers)}")

    progress.progress(66, text="Memverifikasi execution snapshot dan atomic entry plan…")
    stage = time.perf_counter()
    verification_tickers = verification_tickers[: max(1, int(cfg.max_automatic_price_candidates))]
    snapshots = fetch_execution_snapshots(verification_tickers)
    signals = apply_execution_snapshot_gate(signals, snapshots, cfg)
    independent_data, independent_report = fetch_automatic_independent_prices(
        verification_tickers,
        twelve_data_api_key=twelve_token,
        itick_api_token=itick_token,
        primary_reference=_primary_reference(histories),
        primary_source_tiers=getattr(download_report, "source_tiers", {}) or {},
        config=cfg,
    )
    price_validation = build_independent_price_validation(
        histories,
        independent_data,
        config=cfg,
        primary_source_tiers=getattr(download_report, "source_tiers", {}) or {},
    )
    signals = finalize_execution_integrity(
        attach_position_sizing(apply_independent_price_gate(signals, price_validation, cfg), cfg), cfg
    )
    record("EXECUTION_INTEGRITY", stage)

    progress.progress(78, text="Membangun macro regime, sector map, The Next Leader, dan Swing Ready…")
    stage = time.perf_counter()
    macro_result = build_macro_regime(
        benchmark=analysis_benchmark,
        prepared=core.get("prepared", {}),
        fundamentals=fundamentals,
        macro_series=macro_series,
        source_report=macro_source_report,
    )
    focus = build_simple_focus(
        core.get("prepared", {}),
        fundamentals=fundamentals,
        signals=signals,
        news_review=news_review,
        market_status=market_status,
        benchmark=analysis_benchmark,
        macro_result=macro_result,
        config=cfg,
        existing_narrative_events=existing_narrative_events,
        existing_narrative_outcomes=existing_narrative_outcomes,
    )
    record("MACRO_FIRST_DECISION", stage)

    portfolio_analysis, portfolio_summary = analyze_portfolio_positions(
        portfolio,
        histories,
        fundamentals=fundamentals,
        signals=signals,
        account_equity_idr=account_size,
        cash_on_hand_idr=cash_on_hand,
        config=cfg,
    )
    two_stage_coverage = build_two_stage_coverage_audit(
        all_tickers,
        shortlist=verification_tickers,
        fundamentals=fundamentals,
        news_review=news_review,
        market_status=market_status,
    )
    ihsg_direction = analyze_ihsg_direction(analysis_benchmark, core.get("prepared", {}), config=IHSGDirectionConfig())
    data_contract = _simple_contract_audit(all_tickers, core.get("prepared", {}), fundamentals, focus, macro_result.issuer_map)
    database_coverage_after = build_database_coverage(
        all_tickers,
        fundamentals=fundamentals,
        fundamental_history=fundamental_history,
        market_status=market_status,
        news_review=news_review,
        fundamental_report=fundamental_report,
    )
    database_summary_after = readiness_summary(database_coverage_after)
    db_after = database_summary_after.iloc[0] if not database_summary_after.empty else {}
    coverage_summary = pd.DataFrame([{
        "requested_tickers": len(all_tickers),
        "ohlcv_ready": int(data_contract.get("ohlcv_ready", pd.Series(dtype=bool)).fillna(False).sum()) if not data_contract.empty else 0,
        "fundamental_ready": int(data_contract.get("fundamental_ready", pd.Series(dtype=bool)).fillna(False).sum()) if not data_contract.empty else 0,
        "macro_issuer_ready": int(data_contract.get("macro_issuer_ready", pd.Series(dtype=bool)).fillna(False).sum()) if not data_contract.empty else 0,
        "next_leader_scored": int(pd.to_numeric(data_contract.get("next_leader_score", pd.Series(dtype=float)), errors="coerce").notna().sum()) if not data_contract.empty else 0,
        "swing_scored": int(pd.to_numeric(data_contract.get("swing_score", pd.Series(dtype=float)), errors="coerce").notna().sum()) if not data_contract.empty else 0,
        "execution_verification_tickers": len(verification_tickers),
        "database_delta_refresh_tickers": len(refresh_tickers),
        "database_state": db_after.get("database_state", "BACKFILL_REQUIRED"),
        "database_ready_pct": db_after.get("database_ready_pct", np.nan),
        "fundamental_history_pct": db_after.get("fundamental_history_pct", np.nan),
        "dropped_by_total_cap": len(dropped_tickers),
    }])

    result = {
        **core,
        "mode": "database_first_daily_scan",
        "scanner_version": APP_VERSION,
        "signals": signals,
        "fundamentals": fundamentals,
        "fundamental_history": fundamental_history,
        "fundamental_history_report": fundamental_report,
        "market_status": market_status,
        "news_review": news_review,
        "execution_snapshots": snapshots,
        "independent_price_data": independent_data,
        "independent_provider_report": independent_report,
        "price_validation": price_validation,
        "validation_stats": stats,
        "validation_trades": trades,
        "focus_screens": focus,
        "portfolio": portfolio,
        "portfolio_analysis": portfolio_analysis,
        "portfolio_summary": portfolio_summary,
        "scanner_data_contract_audit": data_contract,
        "scan_coverage_summary": coverage_summary,
        "database_coverage_before": database_coverage_before,
        "database_coverage_after": database_coverage_after,
        "database_summary_before": database_summary_before,
        "database_summary_after": database_summary_after,
        "two_stage_shortlist": shortlist_audit,
        "two_stage_refresh_plan": refresh_plan,
        "two_stage_coverage_audit": two_stage_coverage,
        "stage_timings": pd.DataFrame(timings),
        "download_report": download_report,
        "benchmark": benchmark,
        "ihsg_direction": ihsg_direction,
        "macro_snapshot": macro_result.snapshot,
        "macro_sector_map": macro_result.sector_map,
        "macro_issuer_map": macro_result.issuer_map,
        "macro_source_report": macro_result.source_report,
        "scan_elapsed_seconds": round(time.perf_counter() - started, 2),
    }

    progress.progress(93, text="Menyimpan snapshot dan audit trail…")
    stage = time.perf_counter()
    bridge = ScannerDatabaseBridge()
    try:
        result["database_sync_report"] = bridge.persist_scan_result(result)
    except Exception as exc:
        result["database_sync_report"] = pd.DataFrame([{
            "state": "DATABASE_FAIL_SOFT",
            "detail": f"{type(exc).__name__}: {str(exc)[:300]}",
        }])
    record("DATABASE_PERSIST", stage)
    result["stage_timings"] = pd.DataFrame(timings)
    result["scan_elapsed_seconds"] = round(time.perf_counter() - started, 2)
    st.session_state["v9_scan_result"] = result
    progress.progress(100, text="Scan macro-first selesai")
    progress.empty()


result = st.session_state.get("v9_scan_result")
if not result:
    st.info("Upload universe ticker lalu jalankan scanner. Maksimum 400 ticker.")
    st.stop()

focus = result.get("focus_screens", {})
leaders = focus.get("next_leaders", pd.DataFrame()).copy()
swings = focus.get("swing_ready", pd.DataFrame()).copy()
macro_snapshot = result.get("macro_snapshot", pd.DataFrame())
elapsed = _finite(result.get("scan_elapsed_seconds"), 0.0)

metrics = st.columns(5)
metrics[0].metric("Ticker", len(result.get("prepared", {})))
metrics[1].metric("Macro regime", str(macro_snapshot.iloc[0].get("macro_regime", "DATA_PENDING")) if not macro_snapshot.empty else "DATA_PENDING")
metrics[2].metric("BUY ZONE", int(leaders.get("status", pd.Series(dtype=str)).eq("BUY_ZONE").sum()) if not leaders.empty else 0)
metrics[3].metric("Execution ready", int(swings.get("status", pd.Series(dtype=str)).eq("EXECUTION_READY").sum()) if not swings.empty else 0)
metrics[4].metric("Waktu", f"{elapsed:.1f} dtk")
coverage_summary = result.get("scan_coverage_summary", pd.DataFrame())
if isinstance(coverage_summary, pd.DataFrame) and not coverage_summary.empty:
    coverage_row = coverage_summary.iloc[0]
    requested_count = int(_finite(coverage_row.get("requested_tickers"), 0))
    fundamental_count = int(_finite(coverage_row.get("fundamental_ready"), 0))
    if requested_count > 0 and fundamental_count < requested_count:
        st.warning(
            f"Evidence fundamental lengkap tersedia untuk {fundamental_count}/{requested_count} ticker. "
            "Ticker lain tetap dipindai secara teknikal tetapi berstatus DATA_PENDING sampai cache/backfill terisi."
        )

market_tab, leader_tab, swing_tab, portfolio_tab = st.tabs([
    "Market Map", "The Next Leader", "Swing Ready", "Portfolio & Audit",
])

with market_tab:
    if not macro_snapshot.empty:
        row = macro_snapshot.iloc[0]
        cols = st.columns(4)
        cols[0].metric("Macro score", f"{_finite(row.get('macro_regime_score')):.1f}")
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
        "rank", "ticker", "sector", "candidate_type", "status",
        "v9_next_leader_score", "score_coverage_pct",
        "business_quality_score", "future_fundamental_score", "valuation_mos_score",
        "management_capital_score", "issuer_macro_alignment_score",
        "narrative_flow_score", "technical_readiness_score",
        "silent_accumulation_score", "retail_adoption_stage",
        "entry_low", "entry_high", "trigger", "stop_loss", "tp1", "tp2", "rr1",
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
        "rank", "ticker", "sector", "status", "v9_swing_score", "score_coverage_pct",
        "technical_execution_score", "issuer_macro_alignment_score", "narrative_flow_score",
        "business_quality_score", "risk_data_score", "next_leader_score", "strategy",
        "entry_low", "entry_high", "trigger_price", "stop_loss", "tp1", "tp2", "rr1", "rr2",
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
    with st.expander("Fundamental dan database audit"):
        st.dataframe(_safe_display(result.get("fundamental_history_report", pd.DataFrame())), width="stretch", hide_index=True)
        st.dataframe(_safe_display(result.get("database_sync_report", pd.DataFrame())), width="stretch", hide_index=True)
