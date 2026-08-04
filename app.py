from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import streamlit as st


APP_VERSION = "8.0.2-execution-integrity-source-lineage-hotfix"
APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

st.set_page_config(
    page_title="IDX Super Scanner v8.0.2 Two-Stage",
    page_icon="📈",
    layout="wide",
)

REQUIRED_FILES = (
    "scanner.py",
    "scanner_focus.py",
    "scanner_database.py",
    "selector_engine.py",
    "narrative_engine.py",
    "production_scoring.py",
    "incremental_store.py",
    "idx_trading_calendar.py",
    "ihsg_direction.py",
    "free_data_providers.py",
    "two_stage_pipeline.py",
)
missing = [name for name in REQUIRED_FILES if not (APP_ROOT / name).is_file()]
if missing:
    st.error("Deployment v8 tidak lengkap.")
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
    read_cached_fundamental_history,
    read_cached_fundamentals,
    read_cached_market_status,
    read_cached_news_review,
    finalize_execution_integrity,
    parse_portfolio_csv,
    parse_ticker_csv,
    run_adaptive_walkforward_validation,
    select_yahoo_fundamental_tickers,
)
from scanner_focus import (  # noqa: E402
    build_focus_screens,
    build_scanner_data_contract_audit,
)
from scanner_database import ScannerDatabaseBridge  # noqa: E402
from free_data_providers import fetch_reference_fx_rates  # noqa: E402
from ihsg_direction import IHSGDirectionConfig, analyze_ihsg_direction  # noqa: E402
from production_scoring import PRODUCTION_SCORING_VERSION  # noqa: E402
from two_stage_pipeline import (  # noqa: E402
    TWO_STAGE_PIPELINE_VERSION,
    ShortlistConfig,
    build_enrichment_shortlist,
    build_lightweight_preliminary_focus,
    build_two_stage_coverage_audit,
    plan_round_robin_refresh,
)


st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 3rem;}
      [data-testid="stMetricValue"] {font-size: 1.45rem;}
      .v8-note {border:1px solid #334155; border-radius:10px; padding:12px 14px; background:#0f172a;}
      .small-muted {font-size:.86rem; color:#94a3b8;}
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
    combined["_priority"] = np.concatenate(
        [np.full(len(frames[0]), 2), np.full(len(frames[1]), 1)]
    )
    combined = combined.sort_values(["ticker", "_priority"], ascending=[True, False], kind="stable")
    return combined.drop_duplicates("ticker", keep="first").drop(columns="_priority").reset_index(drop=True)


def _bounded_histories(
    histories: Mapping[str, pd.DataFrame],
    *,
    max_bars: int,
) -> dict[str, pd.DataFrame]:
    """Bound daily computation while retaining enough history for all v8 indicators.

    V8 production needs EMA200, 52-week structure, and a 620-bar selector
    lookback.  A 750-bar daily window leaves a safety buffer without repeatedly
    processing the full 5-10 year download.  Full history remains available for
    explicit chronological OOS runs.
    """
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
        out[column] = out[column].map(
            lambda value: "" if value is None else str(value)
        )
    return out.reset_index(drop=True)


def _simple_status(frame: pd.DataFrame, *, multibagger: bool = False) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=str)
    blocked = frame.get("narrative_hard_block", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    if multibagger:
        raw = frame.get("multibagger_status", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper()
        coverage = pd.to_numeric(
            frame.get("v8_production_score_coverage_pct", pd.Series(0.0, index=frame.index)),
            errors="coerce",
        ).fillna(0.0)
        research = frame.get("research_eligible", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
        return pd.Series(
            np.select(
                [blocked, coverage.lt(55.0), raw.str.startswith("DATA_NOT_SCORED"), research],
                ["REJECT", "DATA_PENDING", "DATA_PENDING", "WATCHLIST"],
                default="REJECT",
            ),
            index=frame.index,
        )
    raw = frame.get("setup_status", frame.get("status", pd.Series("", index=frame.index))).fillna("").astype(str).str.upper()
    coverage = pd.to_numeric(
        frame.get("v8_production_score_coverage_pct", pd.Series(0.0, index=frame.index)),
        errors="coerce",
    ).fillna(0.0)
    return pd.Series(
        np.select(
            [
                blocked | raw.eq("REJECT"),
                coverage.lt(55.0),
                raw.eq("EXECUTION_READY"),
                raw.isin({"ENTRY_PLAN_READY", "READY_FOR_PRICE_VERIFY", "READY_FOR_STOCKBIT_VERIFY"}),
            ],
            ["REJECT", "DATA_PENDING", "EXECUTION_READY", "ENTRY_PLAN_READY"],
            default="WATCHLIST",
        ),
        index=frame.index,
    )


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_market_data(tickers: tuple[str, ...], period: str, itick_token: str):
    histories, report = download_ohlcv(
        tickers,
        period=period,
        itick_api_token=itick_token,
    )
    benchmark = download_benchmark(period=period)
    return histories, report, benchmark


@st.cache_data(ttl=900, show_spinner=False)
def _cached_baseline_fundamentals(tickers: tuple[str, ...]):
    """Read database and local caches only; never call external providers."""
    cfg = ScanConfig()
    bridge = ScannerDatabaseBridge()
    try:
        db_snapshot, db_snapshot_audit = bridge.read_fundamental_cache(list(tickers))
    except Exception as exc:
        db_snapshot, db_snapshot_audit = pd.DataFrame(), pd.DataFrame([{
            "provider": "SUPABASE_DATABASE_FIRST",
            "status": "READ_FAIL_SOFT",
            "error": f"{type(exc).__name__}: {str(exc)[:160]}",
        }])
    try:
        db_history, db_history_audit = bridge.read_fundamental_history_cache(list(tickers))
    except Exception as exc:
        db_history, db_history_audit = pd.DataFrame(), pd.DataFrame([{
            "provider": "SUPABASE_DATABASE_FIRST",
            "status": "READ_FAIL_SOFT",
            "error": f"{type(exc).__name__}: {str(exc)[:160]}",
        }])

    local_snapshot = read_cached_fundamentals(tickers, cfg)
    if not local_snapshot.empty and "fundamental_score_eligible" in local_snapshot:
        local_snapshot = local_snapshot.loc[
            local_snapshot["fundamental_score_eligible"].fillna(False).astype(bool)
        ].copy()
    local_history = read_cached_fundamental_history(tickers)
    snapshot = _merge_prefer_primary(local_snapshot, db_snapshot)
    history = combine_fundamental_history(db_history, local_history)
    enriched = enrich_fundamentals_with_history(snapshot, history)

    cache_audit = pd.DataFrame([{
        "provider": "LOCAL_CACHE_ONLY",
        "status": "BASELINE_CACHE_READ",
        "requested_tickers": len(tickers),
        "snapshot_tickers": int(enriched["ticker"].nunique()) if not enriched.empty and "ticker" in enriched else 0,
        "history_tickers": int(history["ticker"].nunique()) if not history.empty and "ticker" in history else 0,
    }])
    reports = [
        frame for frame in (db_snapshot_audit, db_history_audit, cache_audit)
        if isinstance(frame, pd.DataFrame) and not frame.empty
    ]
    report = pd.concat(reports, ignore_index=True, sort=False) if reports else pd.DataFrame()
    return enriched, history, report


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
    """Perform bounded live enrichment after Stage-A ranking."""
    live_snapshot = fetch_resilient_fundamentals(shortlist, cfg)
    if not live_snapshot.empty and "fundamental_score_eligible" in live_snapshot:
        usable_live = live_snapshot.loc[
            live_snapshot["fundamental_score_eligible"].fillna(False).astype(bool)
        ].copy()
    else:
        usable_live = live_snapshot
    snapshot = _merge_prefer_primary(usable_live, baseline_snapshot)

    idx_history, idx_report = fetch_idx_fundamental_history(
        official_refresh,
        max_tickers=len(official_refresh),
        years_back=max(1, int(getattr(cfg, "idx_fundamental_years_back", 3))),
    )
    history_before_yahoo = combine_fundamental_history(
        baseline_history, idx_history,
    )
    yahoo_targets = select_yahoo_fundamental_tickers(
        shortlist,
        history_before_yahoo,
        max_tickers=max(0, min(int(yahoo_limit), len(shortlist))),
        crosscheck_top_n=min(8, len(shortlist)),
    )
    yahoo_history, yahoo_report = fetch_yahoo_fundamental_history(
        yahoo_targets,
        max_tickers=len(yahoo_targets),
    )
    history = combine_fundamental_history(
        baseline_history, idx_history, yahoo_history,
    )
    enriched = enrich_fundamentals_with_history(snapshot, history)
    reports = [
        frame for frame in (
            baseline_report, idx_report, yahoo_report,
            pd.DataFrame([{
                "provider": "TWO_STAGE_PLANNER",
                "status": "BOUNDED_ENRICHMENT",
                "universe_tickers": len(all_tickers),
                "shortlist_tickers": len(shortlist),
                "official_refresh_tickers": len(official_refresh),
                "yahoo_history_tickers": len(yahoo_targets),
            }]),
        ) if isinstance(frame, pd.DataFrame) and not frame.empty
    ]
    report = pd.concat(reports, ignore_index=True, sort=False) if reports else pd.DataFrame()
    return enriched, history, report


st.title("IDX Super Scanner v8.0.2 Two-Stage Production")
st.caption(
    f"Scanner {APP_VERSION} • scoring {PRODUCTION_SCORING_VERSION} • "
    f"pipeline {TWO_STAGE_PIPELINE_VERSION} • AI/EOFF/Best Buy Date berbobot 0%."
)
st.markdown(
    """
    <div class="v8-note">
      <b>Kontrak skor produksi</b><br>
      Multibagger: 55% fundamental/future fundamental, 25% narrative–issuer alignment,
      10% market/sector, 10% silent accumulation. Core Swing: 45% technical,
      15% market/sector, 20% narrative–issuer alignment, 15% flow, 5% data/OOS.
      Tidak ada additive narrative + Emir + AI + EOFF yang menghitung evidence yang sama dua kali.<br>
      Stage A memindai seluruh universe dengan OHLCV dan cache. Stage B hanya melakukan live enrichment
      pada shortlist, sementara fundamental resmi ticker lain diperbarui secara round-robin.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Input")
    ticker_file = st.file_uploader("Universe ticker CSV", type=["csv"])
    portfolio_file = st.file_uploader("Portfolio CSV (opsional)", type=["csv"])
    period = st.selectbox("Riwayat OHLCV", ["3y", "5y", "10y"], index=1)
    account_size = float(st.number_input("Nilai akun (Rp)", min_value=0, value=5_000_000, step=500_000))
    cash_on_hand = float(st.number_input("Cash tersedia (Rp)", min_value=0, value=1_000_000, step=100_000))
    risk_per_trade_pct = st.slider("Risiko per transaksi", 0.25, 2.00, 0.75, 0.25) / 100.0
    enrichment_limit = st.slider("Stage-B enrichment shortlist", 20, 100, 60, 10)
    idx_history_limit = st.slider("Official IDX/XBRL refresh per scan", 0, 80, 40, 10)
    run_oos = st.checkbox("Jalankan chronological OOS", value=False)
    with st.expander("Sumber data opsional"):
        itick_token = st.text_input(
            "iTick API token",
            value=os.getenv("ITICK_API_TOKEN", ""),
            type="password",
        )
        twelve_token = st.text_input(
            "Twelve Data API key",
            value=os.getenv("TWELVE_DATA_API_KEY", ""),
            type="password",
        )
    run_scan = st.button("Jalankan Scanner v8.0.2", type="primary", use_container_width=True)


if run_scan:
    if ticker_file is None:
        st.error("Upload CSV universe terlebih dahulu.")
        st.stop()
    try:
        tickers = parse_ticker_csv(ticker_file, max_tickers=400)
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
    portfolio_tickers = (
        portfolio["ticker"].dropna().astype(str).drop_duplicates().tolist()
        if not portfolio.empty and "ticker" in portfolio
        else []
    )
    all_tickers = list(dict.fromkeys(tickers + portfolio_tickers))

    cfg = ScanConfig(
        account_size_idr=account_size,
        cash_on_hand_idr=cash_on_hand,
        risk_per_trade_pct=risk_per_trade_pct,
        ai_enabled=False,
        ai_max_weight=0.0,
        selector_max_ai_weight=0.0,
        narrative_growth_max_adjustment_points=0.0,
        narrative_turnaround_max_adjustment_points=0.0,
        narrative_swing_max_adjustment_points=0.0,
        emir_growth_max_adjustment_points=0.0,
        emir_turnaround_max_adjustment_points=0.0,
        emir_swing_max_adjustment_points=0.0,
        time_cycle_enabled=False,
        time_cycle_core_max_weight=0.0,
        time_cycle_multibagger_max_weight=0.0,
        eoff_enabled=False,
        eoff_ephemeris_enabled=False,
    )

    stage_timings: list[dict[str, object]] = []
    started = time.perf_counter()

    def record_stage(stage: str, stage_started: float, detail: str = "") -> None:
        stage_timings.append({
            "stage": stage,
            "elapsed_seconds": round(time.perf_counter() - stage_started, 3),
            "detail": detail,
            "pipeline_version": TWO_STAGE_PIPELINE_VERSION,
        })

    progress = st.progress(0, text="Stage A: mengambil OHLCV dan IHSG…")
    stage_started = time.perf_counter()
    histories, download_report, benchmark = _cached_market_data(
        tuple(all_tickers), period, itick_token
    )
    record_stage("A1_OHLCV_BENCHMARK", stage_started, f"requested={len(all_tickers)}")

    progress.progress(16, text="Stage A: menghitung indikator, SMC/ICT, setup, dan selector…")
    stage_started = time.perf_counter()
    daily_compute_bar_limit = 750
    analysis_histories = (
        histories if run_oos else
        _bounded_histories(histories, max_bars=daily_compute_bar_limit)
    )
    analysis_benchmark = (
        benchmark if run_oos or not isinstance(benchmark, pd.DataFrame) else
        benchmark.tail(daily_compute_bar_limit).copy()
    )
    core = ScanEngine(cfg).scan(analysis_histories, analysis_benchmark)
    signals_base = apply_universe_integrity_gate(
        core.get("signals", pd.DataFrame()),
        tickers,
        core.get("prepared", {}).keys(),
        cfg,
    )
    signals_base = attach_ohlcv_source_lineage(
        signals_base, getattr(download_report, "source_tiers", {}) or {}
    )
    record_stage(
        "A2_CORE_LOCAL_SCAN", stage_started,
        f"prepared={len(core.get('prepared', {}))}; bars={'FULL_OOS' if run_oos else daily_compute_bar_limit}",
    )

    stats = pd.DataFrame()
    trades = pd.DataFrame()
    validation_universe = pd.DataFrame()
    if run_oos and core.get("prepared"):
        progress.progress(25, text="Stage A: menjalankan chronological OOS…")
        stage_started = time.perf_counter()
        stats, trades, validation_universe = run_adaptive_walkforward_validation(
            core["prepared"], cfg, initial_tickers=min(80, len(core["prepared"]))
        )
        record_stage("A3_OOS_OPTIONAL", stage_started, f"events={len(trades)}")
    signals_base = attach_backtest_stats(signals_base, stats)
    signals_base = apply_validation_gate(signals_base, cfg)

    progress.progress(34, text="Stage A: membaca database dan cache tanpa live request massal…")
    stage_started = time.perf_counter()
    baseline_fundamentals, baseline_history, baseline_report = _cached_baseline_fundamentals(
        tuple(all_tickers)
    )
    cached_market_status = read_cached_market_status(all_tickers, cfg)
    cached_news_review = read_cached_news_review(all_tickers, lookback_days=30, config=cfg)
    preliminary_signals = attach_fundamentals(signals_base, baseline_fundamentals)
    preliminary_signals = apply_fundamental_gate(preliminary_signals, cfg)
    preliminary_signals = apply_market_status_gate(preliminary_signals, cached_market_status, cfg)
    preliminary_signals = apply_news_gate(preliminary_signals, cached_news_review, cfg)
    preliminary_focus = build_lightweight_preliminary_focus(
        core.get("prepared", {}),
        fundamentals=baseline_fundamentals,
        signals=preliminary_signals,
    )
    record_stage("A4_CACHE_FIRST_PRELIMINARY_RANKING", stage_started)

    progress.progress(48, text="Memilih shortlist Stage B dan merencanakan refresh round-robin…")
    stage_started = time.perf_counter()
    shortlist_cap = min(int(enrichment_limit), len(all_tickers))
    portfolio_priority_count = len(set(portfolio_tickers) & set(all_tickers))
    ranked_capacity = max(0, shortlist_cap - portfolio_priority_count)
    rescue_quota = min(ranked_capacity, max(5, shortlist_cap // 5))
    remaining_ranked = max(0, ranked_capacity - rescue_quota)
    multibagger_quota = (remaining_ranked + 1) // 2
    core_quota = remaining_ranked - multibagger_quota
    shortlist, shortlist_audit = build_enrichment_shortlist(
        all_tickers,
        preliminary_focus=preliminary_focus,
        signals=preliminary_signals,
        portfolio_tickers=portfolio_tickers,
        config=ShortlistConfig(
            max_tickers=shortlist_cap,
            multibagger_quota=multibagger_quota,
            core_quota=core_quota,
            technical_rescue_quota=rescue_quota,
        ),
    )
    official_refresh, refresh_plan = plan_round_robin_refresh(
        all_tickers,
        priority_tickers=shortlist,
        max_tickers=min(int(idx_history_limit), len(all_tickers)),
        priority_quota=min(24, int(idx_history_limit)),
    )
    record_stage(
        "A5_SHORTLIST_AND_REFRESH_PLAN",
        stage_started,
        f"shortlist={len(shortlist)}; official_refresh={len(official_refresh)}",
    )

    progress.progress(56, text="Stage B: memperkaya fundamental shortlist dan cohort official…")
    stage_started = time.perf_counter()
    fundamentals, fundamental_history, fundamental_report = _enrich_shortlist_fundamentals(
        all_tickers=tuple(all_tickers),
        shortlist=tuple(shortlist),
        official_refresh=tuple(official_refresh),
        baseline_snapshot=baseline_fundamentals,
        baseline_history=baseline_history,
        baseline_report=baseline_report,
        cfg=cfg,
        yahoo_limit=min(24, len(shortlist)),
    )
    record_stage("B1_BOUNDED_FUNDAMENTAL_ENRICHMENT", stage_started)

    progress.progress(66, text="Stage B: memperbarui market status dan disclosure/news shortlist…")
    stage_started = time.perf_counter()
    live_market_status = fetch_resilient_market_status(shortlist, cfg)
    live_news_review = fetch_resilient_news_review(shortlist, lookback_days=30, config=cfg)
    market_status = _merge_prefer_primary(live_market_status, cached_market_status)
    news_review = _merge_prefer_primary(live_news_review, cached_news_review)

    signals = attach_fundamentals(signals_base, fundamentals)
    signals = apply_fundamental_gate(signals, cfg)
    signals = apply_market_status_gate(signals, market_status, cfg)
    signals = apply_news_gate(signals, news_review, cfg)
    record_stage("B2_BOUNDED_CONTEXT_ENRICHMENT", stage_started)

    progress.progress(75, text="Stage B: memverifikasi snapshot dan harga kandidat teratas…")
    stage_started = time.perf_counter()
    verification_tickers = shortlist[: max(1, int(cfg.max_automatic_price_candidates))]
    snapshots = fetch_execution_snapshots(shortlist)
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
    signals = apply_independent_price_gate(signals, price_validation, cfg)
    signals = attach_position_sizing(signals, cfg)
    signals = finalize_execution_integrity(signals, cfg)
    record_stage(
        "B3_EXECUTION_VERIFICATION",
        stage_started,
        f"snapshots={len(shortlist)}; independent_price={len(verification_tickers)}",
    )

    progress.progress(84, text="Membangun ranking final Multibagger dan Core Swing untuk shortlist…")
    stage_started = time.perf_counter()
    shortlist_set = set(shortlist)
    final_prepared = {
        ticker: frame for ticker, frame in core.get("prepared", {}).items()
        if ticker in shortlist_set
    }
    final_fundamentals = (
        fundamentals.loc[fundamentals["ticker"].astype(str).isin(shortlist_set)].copy()
        if isinstance(fundamentals, pd.DataFrame) and not fundamentals.empty and "ticker" in fundamentals
        else pd.DataFrame()
    )
    final_signals = (
        signals.loc[signals["ticker"].astype(str).isin(shortlist_set)].copy()
        if isinstance(signals, pd.DataFrame) and not signals.empty and "ticker" in signals
        else pd.DataFrame()
    )
    final_news_review = (
        news_review.loc[news_review["ticker"].astype(str).isin(shortlist_set)].copy()
        if isinstance(news_review, pd.DataFrame) and not news_review.empty and "ticker" in news_review
        else pd.DataFrame()
    )
    final_market_status = (
        market_status.loc[market_status["ticker"].astype(str).isin(shortlist_set)].copy()
        if isinstance(market_status, pd.DataFrame) and not market_status.empty and "ticker" in market_status
        else pd.DataFrame()
    )
    reference_fx = fetch_reference_fx_rates()
    focus = build_focus_screens(
        final_prepared,
        fundamentals=final_fundamentals,
        core_signals=final_signals,
        news_review=final_news_review,
        market_status=final_market_status,
        benchmark=analysis_benchmark,
        config=cfg,
        validation_events=trades,
        ai_memory=pd.DataFrame(),
        reference_fx=reference_fx,
    )
    record_stage(
        "B4_FINAL_PRODUCTION_SCORING", stage_started,
        f"final_universe={len(final_prepared)}",
    )

    portfolio_analysis, portfolio_summary = analyze_portfolio_positions(
        portfolio,
        histories,
        fundamentals=fundamentals,
        signals=signals,
        account_equity_idr=account_size,
        cash_on_hand_idr=cash_on_hand,
        config=cfg,
    )

    data_contract = build_scanner_data_contract_audit(
        tickers,
        histories=histories,
        prepared=core.get("prepared", {}),
        fundamentals=fundamentals,
        fundamental_history=fundamental_history,
        selector=focus.get("stock_selector", pd.DataFrame()),
        multibagger=focus.get("multibagger", pd.DataFrame()),
        core_signals=signals,
        order_builder=focus.get("profit_order_builder", pd.DataFrame()),
        order_builder_coverage=focus.get("profit_data_coverage_audit", pd.DataFrame()),
    )
    two_stage_coverage = build_two_stage_coverage_audit(
        all_tickers,
        shortlist=shortlist,
        fundamentals=fundamentals,
        news_review=news_review,
        market_status=market_status,
    )

    ihsg_direction = analyze_ihsg_direction(
        benchmark,
        core.get("prepared", {}),
        config=IHSGDirectionConfig(),
    )

    result = {
        **core,
        "mode": "scanner_two_stage",
        "scanner_version": APP_VERSION,
        "two_stage_pipeline_version": TWO_STAGE_PIPELINE_VERSION,
        "daily_compute_bar_limit": 0 if run_oos else daily_compute_bar_limit,
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
        "validation_universe_audit": validation_universe,
        "focus_screens": focus,
        "portfolio": portfolio,
        "portfolio_analysis": portfolio_analysis,
        "portfolio_summary": portfolio_summary,
        "scanner_data_contract_audit": data_contract,
        "two_stage_shortlist": shortlist_audit,
        "two_stage_refresh_plan": refresh_plan,
        "two_stage_coverage_audit": two_stage_coverage,
        "two_stage_stage_timings": pd.DataFrame(stage_timings),
        "download_report": download_report,
        "benchmark": benchmark,
        "ihsg_direction": ihsg_direction,
        "scan_elapsed_seconds": round(time.perf_counter() - started, 2),
    }

    progress.progress(94, text="Menyimpan hasil dan cache database…")
    stage_started = time.perf_counter()
    bridge = ScannerDatabaseBridge()
    try:
        result["database_sync_report"] = bridge.persist_scan_result(result)
    except Exception as exc:
        result["database_sync_report"] = pd.DataFrame([{
            "state": "DATABASE_FAIL_SOFT",
            "detail": f"{type(exc).__name__}: {str(exc)[:300]}",
        }])
    record_stage("C1_DATABASE_PERSIST", stage_started)
    result["two_stage_stage_timings"] = pd.DataFrame(stage_timings)
    result["scan_elapsed_seconds"] = round(time.perf_counter() - started, 2)

    st.session_state["v8_scan_result"] = result
    progress.progress(100, text="Scan v8.0.2 two-stage selesai")
    progress.empty()


result = st.session_state.get("v8_scan_result")
if not result:
    st.info("Upload universe ticker lalu jalankan scanner. Maksimum 400 ticker per scan.")
    st.stop()

focus = result.get("focus_screens", {})
multibagger = focus.get("multibagger", pd.DataFrame()).copy()
core_builder = focus.get("profit_order_builder", pd.DataFrame()).copy()
shortlist_frame = result.get("two_stage_shortlist", pd.DataFrame())
coverage_frame = result.get("two_stage_coverage_audit", pd.DataFrame())
shortlisted_tickers = (
    set(shortlist_frame["ticker"].dropna().astype(str))
    if isinstance(shortlist_frame, pd.DataFrame) and not shortlist_frame.empty and "ticker" in shortlist_frame
    else set()
)
if not multibagger.empty:
    multibagger["stage_b_enriched"] = multibagger.get("ticker", pd.Series("", index=multibagger.index)).astype(str).isin(shortlisted_tickers)
    multibagger["production_status"] = _simple_status(multibagger, multibagger=True)
if not core_builder.empty:
    core_builder["stage_b_enriched"] = core_builder.get("ticker", pd.Series("", index=core_builder.index)).astype(str).isin(shortlisted_tickers)
    core_builder["production_status"] = _simple_status(core_builder)

elapsed = _finite(result.get("scan_elapsed_seconds"), 0.0)
shortlist_count = len(shortlist_frame) if isinstance(shortlist_frame, pd.DataFrame) else 0
enrichment_complete = (
    int(coverage_frame.get("expensive_enrichment_complete", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    if isinstance(coverage_frame, pd.DataFrame) and not coverage_frame.empty else 0
)
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Ticker diminta", len(result.get("all_histories", result.get("prepared", {}))))
col2.metric("Siap indikator", len(result.get("prepared", {})))
col3.metric("Stage-B shortlist", shortlist_count)
col4.metric("Enrichment lengkap", enrichment_complete)
col5.metric("Waktu scan", f"{elapsed:.1f} dtk")

tab_scan, tab_portfolio, tab_audit = st.tabs(["Scanner", "Portfolio", "Evidence & Audit"])

with tab_scan:
    sub_mb, sub_core, sub_market = st.tabs(["Multibagger", "Core Swing", "Arah IHSG"])
    with sub_mb:
        mb_columns = [
            "ticker", "stage_b_enriched", "production_status", "multibagger_lane",
            "v8_strategic_score", "v8_production_score_coverage_pct",
            "v8_fundamental_future_score", "v8_narrative_alignment_score",
            "v8_market_sector_score", "v8_flow_score",
            "multibagger_status", "research_recommendation_status",
            "effective_silent_accumulation_score", "technical_entry_state",
            "entry_low", "entry_high", "entry", "trigger", "stop_loss", "tp1", "tp2",
            "rr1", "rr2", "allocation_action", "recommended_allocation_idr",
            "selected_reason", "primary_risk",
        ]
        mb_view = _safe_display(multibagger, mb_columns)
        if mb_view.empty:
            st.warning("Belum ada kandidat Multibagger yang dapat ditampilkan.")
        else:
            st.dataframe(mb_view, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Multibagger CSV",
                mb_view.to_csv(index=False).encode("utf-8-sig"),
                "v8_multibagger.csv",
                "text/csv",
            )
    with sub_core:
        core_columns = [
            "ticker", "stage_b_enriched", "production_status", "strategy", "core_priority_score",
            "v8_production_score_coverage_pct", "v8_technical_score",
            "v8_market_sector_score", "v8_narrative_alignment_score",
            "v8_flow_score", "v8_data_validation_score",
            "setup_status", "decision_state", "entry", "trigger_price", "stop_loss",
            "tp1", "tp2", "rr1", "rr2", "stockbit_order_lots",
            "next_action", "selected_reason", "primary_risk",
        ]
        core_view = _safe_display(core_builder, core_columns)
        if core_view.empty:
            st.warning("Belum ada Core Swing yang melewati gate minimum.")
        else:
            st.dataframe(core_view, use_container_width=True, hide_index=True)
            st.download_button(
                "Download Core Swing CSV",
                core_view.to_csv(index=False).encode("utf-8-sig"),
                "v8_core_swing.csv",
                "text/csv",
            )
    with sub_market:
        ihsg = result.get("ihsg_direction") or {}
        if isinstance(ihsg, Mapping):
            metrics = st.columns(4)
            metrics[0].metric("Regime", str(ihsg.get("regime", "UNKNOWN")))
            metrics[1].metric(
                "Direction", str(ihsg.get("consensus_direction", "NO_EDGE")),
            )
            metrics[2].metric(
                "Confidence",
                f"{_finite(ihsg.get('consensus_confidence'), 0.0):.1f}%",
            )
            metrics[3].metric(
                "Risk multiplier",
                f"{_finite(ihsg.get('risk_budget_multiplier'), 0.5):.2f}x",
            )
            st.write(str(ihsg.get("regime_reason", "Belum ada alasan terstruktur.")))

with tab_portfolio:
    portfolio_analysis = result.get("portfolio_analysis", pd.DataFrame())
    portfolio_summary = result.get("portfolio_summary", {})
    if isinstance(portfolio_summary, Mapping) and portfolio_summary:
        summary_cols = st.columns(4)
        summary_cols[0].metric(
            "Nilai pasar",
            f"Rp {_finite(portfolio_summary.get('market_value_idr')):,.0f}",
        )
        summary_cols[1].metric(
            "Unrealized P/L",
            f"Rp {_finite(portfolio_summary.get('unrealized_pnl_idr')):,.0f}",
            f"{_finite(portfolio_summary.get('unrealized_pnl_pct')) * 100.0:.2f}%",
        )
        summary_cols[2].metric(
            "Open risk",
            f"Rp {_finite(portfolio_summary.get('open_risk_idr')):,.0f}",
        )
        summary_cols[3].metric(
            "Estimasi equity",
            f"Rp {_finite(portfolio_summary.get('estimated_equity_idr')):,.0f}",
        )
    elif isinstance(portfolio_summary, pd.DataFrame) and not portfolio_summary.empty:
        st.dataframe(_safe_display(portfolio_summary), use_container_width=True, hide_index=True)
    if isinstance(portfolio_analysis, pd.DataFrame) and not portfolio_analysis.empty:
        portfolio_columns = [
            "ticker", "portfolio_action", "position_value_idr", "unrealized_pnl_pct",
            "fundamental_score", "silent_accumulation_score", "active_setup",
            "entry", "stop_loss", "tp1", "tp2", "position_risk_idr",
            "portfolio_reason", "portfolio_warning",
        ]
        st.dataframe(_safe_display(portfolio_analysis, portfolio_columns), use_container_width=True, hide_index=True)
    else:
        st.info("Portfolio CSV belum diunggah.")

with tab_audit:
    audit_tabs = st.tabs([
        "Scoring Contract", "Evidence", "Two-Stage",
        "Data Contract", "Fundamental", "Selector/OOS", "Database",
    ])
    with audit_tabs[0]:
        st.dataframe(
            _safe_display(focus.get("production_scoring_audit", pd.DataFrame())),
            use_container_width=True,
            hide_index=True,
        )
    with audit_tabs[1]:
        evidence = _safe_display(
            focus.get("production_evidence_detail", pd.DataFrame()),
        )
        st.dataframe(evidence, use_container_width=True, hide_index=True)
        if not evidence.empty:
            st.download_button(
                "Download Production Evidence CSV",
                evidence.to_csv(index=False).encode("utf-8-sig"),
                "v8_production_evidence.csv",
                "text/csv",
            )
    with audit_tabs[2]:
        st.caption(
            "Shortlist menentukan ticker yang menerima live enrichment; ticker lain tetap diranking dari cache dan diperbarui bertahap."
        )
        st.dataframe(
            _safe_display(result.get("two_stage_stage_timings", pd.DataFrame())),
            use_container_width=True,
            hide_index=True,
        )
        st.dataframe(
            _safe_display(result.get("two_stage_shortlist", pd.DataFrame())),
            use_container_width=True,
            hide_index=True,
        )
        st.dataframe(
            _safe_display(result.get("two_stage_refresh_plan", pd.DataFrame())),
            use_container_width=True,
            hide_index=True,
        )
        two_stage_view = _safe_display(result.get("two_stage_coverage_audit", pd.DataFrame()))
        st.dataframe(two_stage_view, use_container_width=True, hide_index=True)
        if not two_stage_view.empty:
            st.download_button(
                "Download Two-Stage Coverage CSV",
                two_stage_view.to_csv(index=False).encode("utf-8-sig"),
                "v8_0_1_two_stage_coverage.csv",
                "text/csv",
            )
    with audit_tabs[3]:
        st.dataframe(
            _safe_display(result.get("scanner_data_contract_audit", pd.DataFrame())),
            use_container_width=True,
            hide_index=True,
        )
    with audit_tabs[4]:
        st.dataframe(
            _safe_display(result.get("fundamental_history_report", pd.DataFrame())),
            use_container_width=True,
            hide_index=True,
        )
    with audit_tabs[5]:
        selector_audit = focus.get("selector_model_audit", pd.DataFrame())
        if isinstance(selector_audit, pd.DataFrame) and not selector_audit.empty:
            st.dataframe(_safe_display(selector_audit), use_container_width=True, hide_index=True)
        stats = result.get("validation_stats", pd.DataFrame())
        if isinstance(stats, pd.DataFrame) and not stats.empty:
            st.dataframe(_safe_display(stats), use_container_width=True, hide_index=True)
        else:
            st.caption("Chronological OOS tidak dijalankan pada scan ini.")
    with audit_tabs[6]:
        st.dataframe(
            _safe_display(result.get("database_sync_report", pd.DataFrame())),
            use_container_width=True,
            hide_index=True,
        )
