from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="IDX Super Scanner Decision Dashboard v6.6.7", page_icon="🛡️", layout="wide")

# Streamlit runs the selected entrypoint from its deployment workspace. Keep
# the application directory explicit on sys.path and validate that the whole
# core module was uploaded, so a partial GitHub upload produces a useful UI
# message instead of a ModuleNotFoundError traceback.
APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

REQUIRED_SCANNER_FILES = ("scanner.py", "scanner_specialty.py", "ai_engine.py", "time_cycle.py", "eoff_reconstruction.py", "dashboard_v660.py")
missing_source_files = [name for name in REQUIRED_SCANNER_FILES if not (APP_ROOT / name).is_file()]
if missing_source_files:
    st.error("Deployment tidak lengkap: modul scanner belum lengkap di root repository.")
    st.code(
        "Root repository harus berisi:\n"
        "app.py\nscanner.py\nscanner_specialty.py\nai_engine.py\ntime_cycle.py\neoff_reconstruction.py\ndashboard_v660.py\nrequirements.txt",
        language="text",
    )
    st.write("File yang belum ditemukan:", ", ".join(missing_source_files))
    st.info("Ekstrak ZIP revisi, lalu upload seluruh isinya—bukan hanya app.py dan requirements.txt—ke branch yang dideploy.")
    st.stop()

from scanner import (
    ScanConfig,
    ScanEngine,
    apply_fundamental_gate,
    apply_market_status_gate,
    apply_news_gate,
    apply_validation_gate,
    apply_execution_snapshot_gate,
    apply_independent_price_gate,
    apply_universe_integrity_gate,
    attach_backtest_stats,
    attach_broker_summary,
    parse_broker_summary_csv,
    parse_orderbook_snapshot_csv,
    apply_ara_external_confirmation,
    attach_fundamentals,
    attach_position_sizing,
    enforce_portfolio_execution_budget,
    apply_analyst_fusion_gate,
    enforce_analyst_portfolio_budget,
    finalize_execution_integrity,
    download_benchmark,
    download_ohlcv,
    fetch_resilient_market_status,
    fetch_resilient_news_review,
    fetch_resilient_fundamentals,
    select_yahoo_fundamental_tickers,
    fetch_yahoo_fundamental_history,
    fetch_idx_fundamental_history,
    fetch_twelve_data_fundamental_history,
    parse_fundamental_history_csv,
    parse_project_management_csv,
    collect_automatic_forward_quality,
    merge_project_management_reviews,
    combine_fundamental_history,
    enrich_fundamentals_with_history,
    fetch_execution_snapshots,
    fetch_automatic_independent_prices,
    make_signal_chart,
    parse_ticker_csv,
    parse_portfolio_csv,
    analyze_portfolio_positions,
    run_walkforward_validation,
    download_intraday_ohlcv,
    specialty_intraday_shortlist,
    build_specialty_screens,
    build_daily_opportunity_board,
    build_profit_order_builder,
    build_independent_price_validation,
    build_source_quorum_audit,
    idx_daily_bar_is_final,
    idx_regular_decision_window,
    normalize_idx_ticker,
    safe_number,
    safe_text,
)
from time_cycle import (
    TimeCycleConfig, analyze_time_cycle, enrich_core_signals_with_time_cycle,
    enrich_swing_specialty_with_time_cycle, make_time_cycle_chart, TIME_CYCLE_VERSION,
    EOFF_VERSION,
)

from ai_engine import (
    LocalAIConfig, load_memory, update_outcome_memory, resolved_memory_events,
    memory_summary, parse_memory_csv, AI_VERSION,
)
from dashboard_v660 import (
    DASHBOARD_VERSION, build_top20_ranking, render_top20_dashboard, render_time_cycle_main_tab,
)


st.markdown(
    """
    <style>
      .block-container {padding-top: 1.35rem; padding-bottom: 2.5rem;}
      [data-testid="stMetricValue"] {font-size: 1.55rem;}
      .scanner-note {border:1px solid #2a3345; border-radius:12px; padding:12px 14px; background:#101723;}
      .small-muted {color:#9aa7b8; font-size:.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=600, show_spinner=False)
def cached_market_data(
    tickers: tuple[str, ...],
    period: str,
    itick_enabled: bool = False,
    _itick_api_token: str = "",
):
    histories, report = download_ohlcv(
        tickers, period=period,
        itick_api_token=_itick_api_token if itick_enabled else "",
    )
    benchmark = download_benchmark(period=period)
    report.benchmark_ok = not benchmark.empty
    return histories, report, benchmark


@st.cache_data(ttl=600, show_spinner=False)
def cached_portfolio_market_data(
    tickers: tuple[str, ...],
    period: str,
    itick_enabled: bool = False,
    _itick_api_token: str = "",
):
    """Portfolio-only path: download holdings without requiring a universe CSV."""
    return download_ohlcv(
        tickers, period=period,
        itick_api_token=_itick_api_token if itick_enabled else "",
    )


@st.cache_data(ttl=60, show_spinner=False)
def cached_intraday_data(
    tickers: tuple[str, ...],
    period: str = "5d",
    interval: str = "5m",
    itick_enabled: bool = False,
    _itick_api_token: str = "",
):
    return download_intraday_ohlcv(
        tickers, period=period, interval=interval,
        itick_api_token=_itick_api_token if itick_enabled else "",
    )


@st.cache_data(ttl=21_600, show_spinner=False)
def cached_fundamentals(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_resilient_fundamentals(tickers)


@st.cache_data(ttl=21_600, show_spinner=False)
def cached_yahoo_fundamental_history(
    tickers: tuple[str, ...], max_tickers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return fetch_yahoo_fundamental_history(tickers, max_tickers=max_tickers)


@st.cache_data(ttl=86_400, show_spinner=False)
def cached_idx_fundamental_history(
    tickers: tuple[str, ...], max_tickers: int, years_back: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return fetch_idx_fundamental_history(
        tickers, max_tickers=max_tickers, years_back=years_back,
    )


@st.cache_data(ttl=21_600, show_spinner=False)
def cached_twelve_fundamental_history(
    tickers: tuple[str, ...], enabled: bool, max_tickers: int,
    _twelve_data_api_key: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Even when the live API is disabled, the provider function may recover
    # previously validated statements from the durable local cache.
    return fetch_twelve_data_fundamental_history(
        tickers, api_key=_twelve_data_api_key if enabled else "", max_tickers=max_tickers,
    )


@st.cache_data(ttl=900, show_spinner=False)
def cached_automatic_market_status(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_resilient_market_status(tickers)


@st.cache_data(ttl=900, show_spinner=False)
def cached_automatic_news(tickers: tuple[str, ...], lookback_days: int) -> pd.DataFrame:
    return fetch_resilient_news_review(tickers, lookback_days=lookback_days)


@st.cache_data(ttl=300, show_spinner=False)
def cached_execution_snapshots(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_execution_snapshots(tickers)


def configured_twelve_data_key() -> str:
    """Read an optional deployment secret without requiring per-scan input."""
    key = os.environ.get("TWELVE_DATA_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("TWELVE_DATA_API_KEY", "")).strip()
    except Exception:
        return ""


def configured_itick_token() -> str:
    """Read the optional no-cost iTick fallback token from deployment secrets."""
    token = os.environ.get("ITICK_API_TOKEN", "").strip()
    if token:
        return token
    try:
        return str(st.secrets.get("ITICK_API_TOKEN", "")).strip()
    except Exception:
        return ""


@st.cache_data(ttl=900, show_spinner=False)
def cached_automatic_independent_prices(
    tickers: tuple[str, ...],
    reference_date: str,
    primary_reference: tuple[tuple[str, str, float], ...],
    primary_source_tiers: tuple[tuple[str, str], ...],
    config: ScanConfig,
    _twelve_data_api_key: str = "",
    _itick_api_token: str = "",
):
    return fetch_automatic_independent_prices(
        tickers,
        reference_date=reference_date,
        twelve_data_api_key=_twelve_data_api_key,
        itick_api_token=_itick_api_token,
        primary_reference={ticker: (date, close) for ticker, date, close in primary_reference},
        primary_source_tiers=dict(primary_source_tiers),
        config=config,
    )


def upload_fingerprint(*files: object) -> str:
    digest = hashlib.sha256()
    for uploaded_file in files:
        if uploaded_file is None:
            digest.update(b"<none>")
        else:
            digest.update(getattr(uploaded_file, "name", "upload").encode("utf-8", errors="ignore"))
            digest.update(uploaded_file.getvalue())
        digest.update(b"\x00")
    return digest.hexdigest()


def enrich_fundamental_shortlist(
    fundamentals: pd.DataFrame,
    tickers: tuple[str, ...],
    uploaded_history: pd.DataFrame,
    config: ScanConfig,
    twelve_enabled: bool,
    twelve_api_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch bounded statement history and return auditable enriched scores."""
    if not tickers:
        combined = combine_fundamental_history(uploaded_history)
        return enrich_fundamentals_with_history(fundamentals, combined), combined, pd.DataFrame()
    if fundamentals is None or fundamentals.empty:
        fundamentals = pd.DataFrame({"ticker": list(dict.fromkeys(tickers))})
    # Official-first policy: fetch durable IDX XBRL facts before contacting
    # aggregators. Yahoo is then limited to unresolved names plus a small top
    # cross-check cohort, substantially reducing crumb/rate-limit exposure.
    idx_history, idx_report = cached_idx_fundamental_history(
        tickers,
        max_tickers=int(config.idx_fundamental_top_n),
        years_back=int(config.idx_fundamental_years_back),
    )
    yahoo_names = select_yahoo_fundamental_tickers(
        tickers, idx_history,
        max_tickers=int(config.fundamental_history_top_n),
        crosscheck_top_n=int(getattr(config, "fundamental_crosscheck_top_n", 8)),
        min_official_periods=4,
    )
    yahoo_history, yahoo_report = cached_yahoo_fundamental_history(
        yahoo_names, max_tickers=int(config.fundamental_history_top_n),
    )
    twelve_history, twelve_report = cached_twelve_fundamental_history(
        tickers,
        enabled=bool(twelve_enabled and twelve_api_key),
        max_tickers=int(config.twelve_fundamental_top_n),
        _twelve_data_api_key=twelve_api_key,
    )
    combined = combine_fundamental_history(
        uploaded_history, yahoo_history, idx_history, twelve_history,
    )
    enriched = enrich_fundamentals_with_history(fundamentals, combined)
    reports = [
        frame for frame in (yahoo_report, idx_report, twelve_report)
        if frame is not None and not frame.empty
    ]
    if uploaded_history is not None and not uploaded_history.empty:
        reports.append(pd.DataFrame([{
            "ticker": "ALL_UPLOAD",
            "provider": "IDX_REFERENCE_UPLOAD",
            "status": "OK",
            "rows": len(uploaded_history),
            "error": "",
        }]))
    report = pd.concat(reports, ignore_index=True, sort=False) if reports else pd.DataFrame()
    return enriched, combined, report


def rupiah(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"Rp{float(value):,.0f}".replace(",", ".")


def prepare_display(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()
    if {"historical_events", "bayes_probability", "entry_fill_rate_5d"}.issubset(out.columns):
        reliable = pd.to_numeric(out["historical_events"], errors="coerce").fillna(0) >= 30
        out["probability_estimate"] = pd.to_numeric(out["bayes_probability"], errors="coerce").where(reliable)
        out["entry_fill_estimate"] = pd.to_numeric(out["entry_fill_rate_5d"], errors="coerce").where(reliable)
    else:
        out["probability_estimate"] = np.nan
        out["entry_fill_estimate"] = np.nan
    # Streamlit NumberColumn uses printf-style formatting and does not
    # multiply fractional ratios automatically. Convert only display copies.
    for ratio_column in (
        "stop_pct", "quote_spread_pct", "broksum_net_ratio",
        "independent_price_divergence_pct",
    ):
        if ratio_column in out:
            out[ratio_column] = pd.to_numeric(out[ratio_column], errors="coerce") * 100.0
    columns = [
        "ticker",
        "status",
        "execution_policy",
        "setup_state",
        "account_order_state",
        "manual_execution_candidate",
        "signal_risk_grade",
        "signal_risk_warnings",
        "signal_execution_blockers",
        "setup",
        "grade",
        "quality_score",
        "composite_score",
        "analyst_fusion_score",
        "analyst_order_mode",
        "execution_mode",
        "autopilot_verified",
        "autopilot_score",
        "autopilot_blockers",
        "autopilot_primary_setup",
        "confluence_setup_count",
        "confluence_setups",
        "strict_execution_ready",
        "requires_stockbit_price_check",
        "analyst_candidate_reason",
        "analyst_decision_basis",
        "analyst_hard_blockers",
        "strict_primary_execution_blocker",
        "strict_execution_gate_failures",
        "execution_integrity_score",
        "execution_confidence_score",
        "projected_completeness_with_independent_price",
        "projected_confidence_with_independent_price",
        "execution_readiness_pct",
        "primary_execution_blocker",
        "execution_gate_failures",
        "data_completeness_score",
        "data_completeness_tier",
        "data_missing_layers",
        "technical_data_coverage",
        "risk_data_coverage",
        "fundamental_data_coverage",
        "validation_data_coverage",
        "market_status_data_coverage",
        "news_data_coverage",
        "quote_data_coverage",
        "universe_data_coverage",
        "evidence_state",
        "validation_gate_score",
        "validation_tier",
        "probability_estimate",
        "entry_fill_estimate",
        "historical_events",
        "median_fill_bars",
        "median_time_to_tp1_bars",
        "last_price",
        "entry_type",
        "entry_low",
        "entry_high",
        "entry",
        "stop_loss",
        "tp1",
        "tp2",
        "tp1_basis",
        "tp2_basis",
        "rr1",
        "rr2",
        "stop_pct",
        "distance_atr",
        "volume_ratio",
        "adtv20_idr",
        "fundamental_score",
        "fundamental_score_10",
        "fundamental_data_grade",
        "fundamental_source_count",
        "fundamental_source_families",
        "fundamental_official_verified",
        "fundamental_consensus_score",
        "fundamental_conflicts",
        "fundamental_coverage",
        "statement_age_days",
        "silent_accumulation_score",
        "up_down_value_ratio20",
        "quote_last_price",
        "quote_spread_pct",
        "quote_market_state",
        "independent_price_state",
        "independent_source",
        "independent_source_family",
        "independent_asof",
        "independent_last_price",
        "independent_price_divergence_pct",
        "independent_overlap_bars",
        "independent_return_correlation",
        "suggested_lots",
        "sizing_is_informational",
        "proposed_order_instruction",
        "order_instruction",
        "stockbit_trigger_price",
        "stockbit_limit_price",
        "execution_timing",
        "stockbit_order_price",
        "stockbit_order_lots",
        "execution_rank",
        "capital_required_idr",
        "max_loss_idr",
        "max_loss_pct_account",
        "broksum_signal",
        "broksum_net_ratio",
        "verified_catalyst_count",
        "catalyst_summary",
        "market_status_coverage",
        "market_status_confidence",
        "news_review_status",
        "news_confidence",
        "fundamental_confidence",
        "quote_confidence",
        "universe_confidence",
        "automation_decision",
        "critical_blockers",
        "evidence_warnings",
        "portfolio_blockers",
        "market_regime",
        "action",
        "valid_until",
        "blockers",
        "reason",
    ]
    return out[[c for c in columns if c in out.columns]]


def sort_signals(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals
    out = signals.copy()
    if "composite_score" not in out:
        out["composite_score"] = out["quality_score"]
    out["status_rank"] = out["status"].map({
        "EXECUTION_READY": 0,
        "READY_FOR_STOCKBIT_VERIFY": 1,
        "SIGNAL_READY": 2,
        "ENTRY_PLAN_READY": 3,
        "READY_NOT_SELECTED": 4,
        "READY_FOR_PRICE_VERIFY": 5,
        "PENDING_CLOSE": 6,
        "PENDING_DATA": 7,
        "WATCHLIST_ENTRY": 8,
        "BLOCKED_CONTEXT": 9,
        "REJECT": 10,
    }).fillna(99)
    return out.sort_values(
        ["status_rank", "composite_score", "quality_score", "rr2", "adtv20_idr"],
        ascending=[True, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def execution_funnel_summary(signals: pd.DataFrame) -> pd.DataFrame:
    if signals is None or signals.empty:
        return pd.DataFrame(columns=["Tahap", "Ticker unik"])
    ticker = signals.get("ticker", pd.Series(signals.index.astype(str), index=signals.index)).astype(str)
    actionable = signals.get("analyst_order_mode", pd.Series("WATCH_ONLY", index=signals.index)).fillna("WATCH_ONLY").ne("WATCH_ONLY")
    no_hard = signals.get("analyst_hard_blockers", pd.Series("", index=signals.index)).fillna("").astype(str).str.strip().eq("")
    final_bar = ~signals.get("pending_close", pd.Series(False, index=signals.index)).fillna(False).astype(bool)
    valid_signal = signals.get("setup_valid_signal", pd.Series(False, index=signals.index)).fillna(False).astype(bool)
    def unique_count(mask: pd.Series) -> int:
        return int(ticker[mask].nunique())
    return pd.DataFrame([
        {"Tahap": "Setup terdeteksi", "Ticker unik": int(ticker.nunique())},
        {"Tahap": "Order mode actionable", "Ticker unik": unique_count(actionable)},
        {"Tahap": "Tidak ada invalidasi struktural/data", "Ticker unik": unique_count(actionable & no_hard)},
        {"Tahap": "Candle EOD final", "Ticker unik": unique_count(actionable & no_hard & final_bar)},
        {"Tahap": "Setup valid Signal-First", "Ticker unik": unique_count(valid_signal)},
        {"Tahap": "SIGNAL_READY", "Ticker unik": unique_count(signals["status"].eq("SIGNAL_READY"))},
        {"Tahap": "ENTRY_PLAN_READY", "Ticker unik": unique_count(signals["status"].eq("ENTRY_PLAN_READY"))},
        {"Tahap": "Harga independen satu sesi", "Ticker unik": unique_count(signals.get("independent_price_verified", pd.Series(False, index=signals.index)).fillna(False).astype(bool))},
        {"Tahap": "READY_FOR_STOCKBIT_VERIFY", "Ticker unik": unique_count(signals["status"].eq("READY_FOR_STOCKBIT_VERIFY"))},
        {"Tahap": "CORE_PLAN_VERIFIED", "Ticker unik": unique_count(signals.get("autopilot_verified", pd.Series(False, index=signals.index)).fillna(False).astype(bool))},
        {"Tahap": "EXECUTION_READY", "Ticker unik": unique_count(signals["status"].eq("EXECUTION_READY"))},
    ])


def result_table(df: pd.DataFrame) -> None:
    st.dataframe(
        prepare_display(df),
        width="stretch",
        hide_index=True,
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", pinned=True),
            "status": st.column_config.TextColumn("Status", pinned=True),
            "execution_policy": st.column_config.TextColumn("Policy"),
            "setup_state": st.column_config.TextColumn("Setup state"),
            "manual_execution_candidate": st.column_config.CheckboxColumn("Verify Stockbit"),
            "signal_risk_grade": st.column_config.TextColumn("Risk grade"),
            "signal_risk_warnings": st.column_config.TextColumn("Risk warnings"),
            "setup": "Setup",
            "quality_score": st.column_config.NumberColumn("Technical", format="%.1f"),
            "composite_score": st.column_config.NumberColumn("Composite", format="%.1f"),
            "analyst_fusion_score": st.column_config.NumberColumn("Signal score", format="%.1f"),
            "analyst_order_mode": st.column_config.TextColumn("Analyst order mode"),
            "execution_mode": st.column_config.TextColumn("Execution mode"),
            "autopilot_verified": st.column_config.CheckboxColumn("Autopilot"),
            "autopilot_score": st.column_config.NumberColumn("Autopilot score", format="%.1f%%"),
            "autopilot_primary_setup": st.column_config.CheckboxColumn("Primary order"),
            "confluence_setup_count": st.column_config.NumberColumn("Setup count", format="%d"),
            "strict_execution_ready": st.column_config.CheckboxColumn("Strict ready"),
            "requires_stockbit_price_check": st.column_config.CheckboxColumn("Check Stockbit"),
            "execution_integrity_score": st.column_config.NumberColumn("Execution confidence", format="%.1f%%"),
            "execution_confidence_score": st.column_config.NumberColumn("Confidence", format="%.1f%%"),
            "data_completeness_score": st.column_config.NumberColumn("Data completeness", format="%.1f%%"),
            "data_completeness_tier": st.column_config.TextColumn("Completeness tier"),
            "technical_data_coverage": st.column_config.NumberColumn("Tech data", format="%.0f%%"),
            "risk_data_coverage": st.column_config.NumberColumn("Risk data", format="%.0f%%"),
            "fundamental_data_coverage": st.column_config.NumberColumn("Fund. data", format="%.0f%%"),
            "validation_data_coverage": st.column_config.NumberColumn("OOS data", format="%.0f%%"),
            "market_status_data_coverage": st.column_config.NumberColumn("IDX status data", format="%.0f%%"),
            "news_data_coverage": st.column_config.NumberColumn("News data", format="%.0f%%"),
            "quote_data_coverage": st.column_config.NumberColumn("Quote data", format="%.0f%%"),
            "universe_data_coverage": st.column_config.NumberColumn("Universe data", format="%.0f%%"),
            "validation_gate_score": st.column_config.NumberColumn("OOS quality gate", format="%.0f%%"),
            "probability_estimate": st.column_config.NumberColumn("Setup OOS P(TP1<SL)*", format="%.1f%%"),
            "entry_fill_estimate": st.column_config.NumberColumn("P(fill≤5d)*", format="%.1f%%"),
            "historical_events": st.column_config.NumberColumn("Sample OOS", format="%d"),
            "median_fill_bars": st.column_config.NumberColumn("Median fill", format="%.1f bar"),
            "median_time_to_tp1_bars": st.column_config.NumberColumn("Median TP1", format="%.1f bar"),
            "last_price": st.column_config.NumberColumn("Last", format="Rp %.0f"),
            "entry_low": st.column_config.NumberColumn("Zone low", format="Rp %.0f"),
            "entry_high": st.column_config.NumberColumn("Zone high", format="Rp %.0f"),
            "entry": st.column_config.NumberColumn("Entry", format="Rp %.0f"),
            "stop_loss": st.column_config.NumberColumn("SL", format="Rp %.0f"),
            "tp1": st.column_config.NumberColumn("TP1", format="Rp %.0f"),
            "tp2": st.column_config.NumberColumn("TP2", format="Rp %.0f"),
            "tp1_basis": st.column_config.TextColumn("TP1 basis"),
            "tp2_basis": st.column_config.TextColumn("TP2 basis"),
            "target_model": st.column_config.TextColumn("Target model"),
            "target_structure": st.column_config.TextColumn("Target structure"),
            "rr1": st.column_config.NumberColumn("RR1", format="%.2f"),
            "rr2": st.column_config.NumberColumn("RR2", format="%.2f"),
            "stop_pct": st.column_config.NumberColumn("Risk", format="%.1f%%"),
            "distance_atr": st.column_config.NumberColumn("Dist. ATR", format="%.2f"),
            "volume_ratio": st.column_config.NumberColumn("Vol x", format="%.2f"),
            "adtv20_idr": st.column_config.NumberColumn("ADTV20", format="Rp %.0f"),
            "fundamental_score": st.column_config.NumberColumn("Fund.", format="%.1f"),
            "fundamental_score_10": st.column_config.NumberColumn("Fund. /10", format="%.2f"),
            "fundamental_data_grade": st.column_config.TextColumn("Fund. data grade"),
            "fundamental_source_count": st.column_config.NumberColumn("Fund. sources", format="%d"),
            "fundamental_official_verified": st.column_config.CheckboxColumn("IDX/XBRL verified"),
            "fundamental_consensus_score": st.column_config.NumberColumn("Fund. consensus", format="%.1f"),
            "fundamental_coverage": st.column_config.NumberColumn("Fund. coverage", format="%.0f%%"),
            "statement_age_days": st.column_config.NumberColumn("FS age", format="%.0f d"),
            "silent_accumulation_score": st.column_config.NumberColumn("Accumulation", format="%.0f"),
            "up_down_value_ratio20": st.column_config.NumberColumn("Up/Down value", format="%.2f"),
            "quote_last_price": st.column_config.NumberColumn("Quote", format="Rp %.0f"),
            "independent_last_price": st.column_config.NumberColumn("Independent", format="Rp %.0f"),
            "independent_price_divergence_pct": st.column_config.NumberColumn("Price diff", format="%.2f%%"),
            "independent_return_correlation": st.column_config.NumberColumn("Return corr.", format="%.3f"),
            "quote_spread_pct": st.column_config.NumberColumn("Spread", format="%.2f%%"),
            "suggested_lots": st.column_config.NumberColumn("Lot", format="%d"),
            "stockbit_order_price": st.column_config.NumberColumn("Order price", format="Rp %.0f"),
            "stockbit_order_lots": st.column_config.NumberColumn("Order lot", format="%d"),
            "execution_rank": st.column_config.NumberColumn("Rank", format="%.0f"),
            "capital_required_idr": st.column_config.NumberColumn("Modal order", format="Rp %.0f"),
            "max_loss_idr": st.column_config.NumberColumn("Max loss est.", format="Rp %.0f"),
            "max_loss_pct_account": st.column_config.NumberColumn("Risk akun", format="%.2f%%"),
            "broksum_net_ratio": st.column_config.NumberColumn("Broksum net", format="%.1f%%"),
            "valid_until": st.column_config.DatetimeColumn("Valid until", format="DD MMM YYYY"),
        },
    )



def specialty_table(df: pd.DataFrame, height: int = 430) -> None:
    if df is None or df.empty:
        st.info("Tidak ada kandidat yang memenuhi filter minimum.")
        return
    out = df.copy()
    fractional_pct_columns = [
        "risk_pct", "daily_return_pct", "room_to_ara_pct", "opening_gap_pct",
        "revenue_growth", "earnings_growth", "roe", "roa", "net_margin",
        "debt_equity", "cash_to_debt", "peg_ratio", "fcf_yield", "roc60",
        "roc120", "relative_strength60", "distance_52w_high",
        "max_position_pct_equity", "proxy_signed_volume_imbalance", "proxy_clv_pressure",
        "proxy_vwap_hold_ratio", "proxy_late_buy_imbalance", "proxy_late_volume_share",
        "proxy_directional_efficiency", "proxy_ara_lock_ratio", "proxy_max_unlock_pct",
        "proxy_range_compression", "historical_next_day_positive_rate",
        "historical_next_day_strong_rate", "historical_next_day_ara_rate",
        "historical_median_next_day_return", "historical_pre_ara_hit_rate",
        "orderbook_imbalance", "orderbook_spread_pct",
        "positive_ocf_ratio", "positive_earnings_ratio", "margin_stability",
        "share_dilution_yoy", "roic_proxy", "car", "npl_gross", "ldr",
    ]
    for column in fractional_pct_columns:
        if column in out:
            # Ratios such as DER, cash/debt, and PEG are not percentages.
            if column not in {"debt_equity", "cash_to_debt", "peg_ratio"}:
                out[column] = pd.to_numeric(out[column], errors="coerce") * 100.0
    st.dataframe(
        out,
        width="stretch",
        hide_index=True,
        height=height,
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", pinned=True),
            "last_price": st.column_config.NumberColumn("Last", format="Rp %.0f"),
            "intraday_last": st.column_config.NumberColumn("Intraday", format="Rp %.0f"),
            "entry": st.column_config.NumberColumn("Entry", format="Rp %.0f"),
            "entry_reference": st.column_config.NumberColumn("Entry ref", format="Rp %.0f"),
            "sniper_entry": st.column_config.NumberColumn("Sniper entry", format="Rp %.0f"),
            "stop_loss": st.column_config.NumberColumn("SL", format="Rp %.0f"),
            "hard_stop": st.column_config.NumberColumn("Hard SL", format="Rp %.0f"),
            "sniper_stop": st.column_config.NumberColumn("Sniper SL", format="Rp %.0f"),
            "morning_tp1": st.column_config.NumberColumn("Morning TP1", format="Rp %.0f"),
            "morning_tp2": st.column_config.NumberColumn("Morning TP2", format="Rp %.0f"),
            "day_tp1": st.column_config.NumberColumn("Day TP1", format="Rp %.0f"),
            "day_tp2": st.column_config.NumberColumn("Day TP2", format="Rp %.0f"),
            "sniper_tp1": st.column_config.NumberColumn("Sniper TP1", format="Rp %.0f"),
            "sniper_tp2": st.column_config.NumberColumn("Sniper TP2", format="Rp %.0f"),
            "tp1_basis": st.column_config.TextColumn("TP1 basis"),
            "tp2_basis": st.column_config.TextColumn("TP2 basis"),
            "target_model": st.column_config.TextColumn("Target model"),
            "target_structure": st.column_config.TextColumn("Target structure"),
            "ara_price": st.column_config.NumberColumn("ARA price", format="Rp %.0f"),
            "ara_tp1": st.column_config.NumberColumn("ARA TP1", format="Rp %.0f"),
            "ara_tp2": st.column_config.NumberColumn("ARA TP2", format="Rp %.0f"),
            "adtv20_idr": st.column_config.NumberColumn("ADTV20", format="Rp %.0f"),
            "value_today_idr": st.column_config.NumberColumn("Value today", format="Rp %.0f"),
            "market_cap": st.column_config.NumberColumn("Market cap", format="Rp %.0f"),
            "capital_required_idr": st.column_config.NumberColumn("Capital", format="Rp %.0f"),
            "risk_pct": st.column_config.NumberColumn("Risk", format="%.2f%%"),
            "daily_return_pct": st.column_config.NumberColumn("Daily return", format="%.2f%%"),
            "room_to_ara_pct": st.column_config.NumberColumn("Room to ARA", format="%.2f%%"),
            "opening_gap_pct": st.column_config.NumberColumn("Opening gap", format="%.2f%%"),
            "revenue_growth": st.column_config.NumberColumn("Revenue growth", format="%.1f%%"),
            "earnings_growth": st.column_config.NumberColumn("Earnings growth", format="%.1f%%"),
            "roe": st.column_config.NumberColumn("ROE", format="%.1f%%"),
            "roa": st.column_config.NumberColumn("ROA", format="%.1f%%"),
            "net_margin": st.column_config.NumberColumn("Net margin", format="%.1f%%"),
            "fcf_yield": st.column_config.NumberColumn("FCF yield", format="%.1f%%"),
            "roc60": st.column_config.NumberColumn("ROC60", format="%.1f%%"),
            "roc120": st.column_config.NumberColumn("ROC120", format="%.1f%%"),
            "relative_strength60": st.column_config.NumberColumn("RS60", format="%.1f%%"),
            "distance_52w_high": st.column_config.NumberColumn("From 52W high", format="%.1f%%"),
            "max_position_pct_equity": st.column_config.NumberColumn("Max allocation", format="%.1f%%"),
            "capital_priority_rank": st.column_config.NumberColumn("Capital rank", format="%d"),
            "capital_conviction_score": st.column_config.NumberColumn("Capital conviction", format="%.1f"),
            "strategic_target_weight_pct": st.column_config.NumberColumn("Target weight", format="%.2f%%"),
            "deploy_now_weight_pct": st.column_config.NumberColumn("Deploy now", format="%.2f%%"),
            "allocation_cap_pct": st.column_config.NumberColumn("Name cap", format="%.1f%%"),
            "strategic_target_amount_idr": st.column_config.NumberColumn("Strategic target", format="Rp %.0f"),
            "recommended_allocation_idr": st.column_config.NumberColumn("Deploy budget", format="Rp %.0f"),
            "estimated_order_value_idr": st.column_config.NumberColumn("Order value", format="Rp %.0f"),
            "recommended_lots": st.column_config.NumberColumn("Lots", format="%d"),
            "multibagger_cash_reserve_idr": st.column_config.NumberColumn("Cash reserve", format="Rp %.0f"),
            "fundamental_coverage": st.column_config.NumberColumn("Fund. coverage", format="%.0f%%"),
            "fundamental_score_10": st.column_config.NumberColumn("Fund. /10", format="%.2f"),
            "fundamental_data_grade": st.column_config.TextColumn("Data grade"),
            "fundamental_source_count": st.column_config.NumberColumn("Sources", format="%d"),
            "fundamental_history_quarters": st.column_config.NumberColumn("Quarter history", format="%d"),
            "fundamental_history_years": st.column_config.NumberColumn("Annual history", format="%d"),
            "fundamental_history_coverage": st.column_config.NumberColumn("History coverage", format="%.0f%%"),
            "fundamental_consensus_score": st.column_config.NumberColumn("Consensus", format="%.1f"),
            "earnings_quality_score": st.column_config.NumberColumn("Earnings quality", format="%.1f"),
            "cash_conversion_ttm": st.column_config.NumberColumn("OCF / laba TTM", format="%.2f"),
            "positive_ocf_ratio": st.column_config.NumberColumn("Positive OCF", format="%.1f%%"),
            "positive_earnings_ratio": st.column_config.NumberColumn("Positive earnings", format="%.1f%%"),
            "margin_stability": st.column_config.NumberColumn("Margin stability", format="%.1f%%"),
            "share_dilution_yoy": st.column_config.NumberColumn("Dilution YoY", format="%.1f%%"),
            "roic_proxy": st.column_config.NumberColumn("ROIC proxy", format="%.1f%%"),
            "net_debt_ebitda": st.column_config.NumberColumn("Net debt / EBITDA", format="%.2f"),
            "interest_coverage": st.column_config.NumberColumn("Interest cover", format="%.1f"),
            "car": st.column_config.NumberColumn("CAR", format="%.1f%%"),
            "npl_gross": st.column_config.NumberColumn("NPL gross", format="%.1f%%"),
            "ldr": st.column_config.NumberColumn("LDR", format="%.1f%%"),
            "time_cycle_score": st.column_config.NumberColumn("Time-cycle", format="%.1f"),
            "time_cycle_effective_weight_pct": st.column_config.NumberColumn("Cycle weight", format="%.2f%%"),
            "eoff_reconstruction_score": st.column_config.NumberColumn("EOFF score", format="%.1f"),
            "eoff_strength_label": st.column_config.TextColumn("EOFF strength"),
            "eoff_signal_active": st.column_config.CheckboxColumn("EOFF active"),
            "eoff_fib_cluster_count": st.column_config.NumberColumn("Fib cluster", format="%d"),
            "eoff_historical_hit_rate": st.column_config.NumberColumn("EOFF hist. hit", format="%.1f%%"),
            "eoff_astro_score": st.column_config.NumberColumn("Astro score", format="%.1f"),
            "eoff_adaptive_total_weight_pct": st.column_config.NumberColumn("Secondary astro share", format="%.2f%%"),
            "eoff_validation_path": st.column_config.TextColumn("EOFF validation path"),
            "eoff_reversal_date": st.column_config.TextColumn("EOFF date"),
            "eoff_moon_declination_deg": st.column_config.NumberColumn("Moon decl.", format="%.2f°"),
        },
    )


def render_specialty_download(label: str, df: pd.DataFrame, filename: str) -> None:
    st.download_button(
        label,
        (df if df is not None else pd.DataFrame()).to_csv(index=False).encode("utf-8"),
        filename,
        "text/csv",
        width="stretch",
        disabled=df is None or df.empty,
    )



def build_tradingview_bridge(
    core_signals: pd.DataFrame,
    specialty_screens: dict[str, pd.DataFrame] | None,
) -> pd.DataFrame:
    """Normalize scanner outputs into levels that can be copied to TradingView.

    Pine Script cannot read a local Streamlit dataframe directly. This bridge
    exports a stable schema so the selected row can be pasted into the
    indicator's MANUAL SCANNER LEVELS inputs without reinterpreting prices.
    """
    columns = [
        "source", "ticker", "tv_symbol", "setup", "status", "timeframe",
        "entry_low", "entry_high", "entry", "stop_loss", "tp1", "tp2",
        "rr1", "rr2", "valid_until", "market_regime", "quality_score",
        "data_completeness_score", "execution_confidence_score", "scanner_note",
    ]
    rows: list[dict[str, object]] = []

    def clean_ticker(value: object) -> str:
        ticker = str(value or "").strip().upper()
        return ticker[:-3] if ticker.endswith(".JK") else ticker

    def add_rows(
        frame: pd.DataFrame | None,
        *,
        source: str,
        setup_default: str,
        status_column: str,
        timeframe: str,
        entry_column: str = "entry",
        stop_column: str = "stop_loss",
        tp1_column: str = "tp1",
        tp2_column: str = "tp2",
        setup_column: str = "setup",
        entry_low_column: str = "entry_low",
        entry_high_column: str = "entry_high",
        note_column: str = "blockers",
    ) -> None:
        if frame is None or frame.empty or "ticker" not in frame:
            return
        for _, row in frame.iterrows():
            ticker = clean_ticker(row.get("ticker"))
            if not ticker:
                continue
            entry = pd.to_numeric(pd.Series([row.get(entry_column)]), errors="coerce").iloc[0]
            entry_low = pd.to_numeric(pd.Series([row.get(entry_low_column)]), errors="coerce").iloc[0]
            entry_high = pd.to_numeric(pd.Series([row.get(entry_high_column)]), errors="coerce").iloc[0]
            if pd.isna(entry_low):
                entry_low = entry
            if pd.isna(entry_high):
                entry_high = entry
            setup_value = str(row.get(setup_column) or setup_default)
            status_value = str(row.get(status_column) or "NOT_EVALUATED")
            note_parts = [
                str(row.get(note_column) or "").strip(),
                str(row.get("reason") or "").strip(),
                str(row.get("action") or "").strip(),
            ]
            rows.append({
                "source": source,
                "ticker": f"{ticker}.JK",
                "tv_symbol": f"IDX:{ticker}",
                "setup": setup_value,
                "status": status_value,
                "timeframe": timeframe,
                "entry_low": entry_low,
                "entry_high": entry_high,
                "entry": entry,
                "stop_loss": pd.to_numeric(pd.Series([row.get(stop_column)]), errors="coerce").iloc[0],
                "tp1": pd.to_numeric(pd.Series([row.get(tp1_column)]), errors="coerce").iloc[0],
                "tp2": pd.to_numeric(pd.Series([row.get(tp2_column)]), errors="coerce").iloc[0],
                "rr1": pd.to_numeric(pd.Series([row.get("rr1")]), errors="coerce").iloc[0],
                "rr2": pd.to_numeric(pd.Series([row.get("rr2")]), errors="coerce").iloc[0],
                "valid_until": row.get("valid_until", pd.NaT),
                "market_regime": row.get("market_regime", ""),
                "quality_score": row.get("quality_score", row.get("sniper_score", np.nan)),
                "data_completeness_score": row.get("data_completeness_score", np.nan),
                "execution_confidence_score": row.get("execution_confidence_score", np.nan),
                "scanner_note": " | ".join(dict.fromkeys(part for part in note_parts if part)),
            })

    add_rows(
        core_signals,
        source="CORE",
        setup_default="CORE",
        status_column="status",
        timeframe="1D",
    )
    specialty = specialty_screens or {}
    add_rows(
        specialty.get("sniper"), source="SNIPER", setup_default="ICT SNIPER",
        status_column="sniper_status", timeframe="1D", entry_column="sniper_entry",
        stop_column="sniper_stop", tp1_column="sniper_tp1", tp2_column="sniper_tp2",
    )
    add_rows(
        specialty.get("bpjs"), source="BPJS", setup_default="BPJS",
        status_column="bpjs_status", timeframe="5m", tp1_column="day_tp1", tp2_column="day_tp2",
    )
    add_rows(
        specialty.get("bsjp"), source="BSJP", setup_default="BSJP",
        status_column="bsjp_status", timeframe="5m", tp1_column="morning_tp1", tp2_column="morning_tp2",
    )
    add_rows(
        specialty.get("multibagger"), source="MULTIBAGGER", setup_default="MULTIBAGGER",
        status_column="multibagger_status", timeframe="1W / 1D", setup_column="active_setup",
        note_column="red_flags",
    )
    add_rows(
        specialty.get("ara_hunter"), source="ARA_HUNTER", setup_default="ARA HUNTER",
        status_column="ara_hunter_status", timeframe="5m / 1D", entry_column="entry_reference",
        stop_column="hard_stop", tp1_column="ara_tp1", tp2_column="ara_tp2",
    )
    bridge = pd.DataFrame(rows, columns=columns)
    if bridge.empty:
        return bridge
    numeric = ["entry_low", "entry_high", "entry", "stop_loss", "tp1", "tp2", "rr1", "rr2"]
    for column in numeric:
        bridge[column] = pd.to_numeric(bridge[column], errors="coerce")
    bridge = bridge.drop_duplicates(["source", "ticker", "setup", "status"], keep="first")
    status_rank = {
        "EXECUTION_READY": 0, "SNIPER_ORDER_READY": 0, "BPJS_ORDER_READY": 0, "BSJP_ORDER_READY": 0,
        "READY_FOR_STOCKBIT_VERIFY": 1, "READY_NOT_SELECTED": 1, "MULTIBAGGER_A_CANDIDATE": 1,
        "PRE_ARA_ORDER_READY": 0, "ARA_CONTINUATION_ORDER_READY": 0,
        "SIGNAL_READY": 2, "ENTRY_PLAN_READY": 3,
        "SNIPER_SIGNAL_READY": 1, "BPJS_SIGNAL_READY": 1, "BSJP_SIGNAL_READY": 1,
        "ARA_CONTINUATION_FLOW_VERIFIED_SIGNAL": 1, "PRE_ARA_SIGNAL_READY": 1, "ARA_CONTINUATION_SIGNAL_READY": 2,
        "PRE_ARA_CANDIDATE": 2, "ARA_CONTINUATION_CANDIDATE": 2, "ARA_CONFIRMED_ONLY": 3,
        "READY_FOR_PRICE_VERIFY": 2, "WAIT_SNIPER_RETRACE": 3,
        "BPJS_WATCHLIST": 3, "BSJP_WATCHLIST": 3, "MULTIBAGGER_B_CANDIDATE": 3,
        "BPJS_DAILY_RADAR": 4, "BSJP_DAILY_RADAR": 4, "PRE_ARA_DAILY_RADAR": 4,
        "PRE_ARA_WATCHLIST": 4,
    }
    bridge["_rank"] = bridge["status"].map(status_rank).fillna(9)
    return bridge.sort_values(["_rank", "source", "ticker"]).drop(columns="_rank").reset_index(drop=True)


def render_portfolio_panel(result: dict) -> None:
    portfolio_analysis: pd.DataFrame = result.get("portfolio_analysis", pd.DataFrame())
    portfolio_summary: dict = result.get("portfolio_summary", {})
    if portfolio_analysis.empty:
        st.info("Upload portfolio CSV untuk memperoleh keputusan portfolio.")
        return

    st.subheader("Portfolio decision engine")
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.metric("Nilai posisi", rupiah(portfolio_summary.get("market_value_idr", np.nan)))
    p2.metric("Unrealized P/L", rupiah(portfolio_summary.get("unrealized_pnl_idr", np.nan)))
    p3.metric("P/L %", f"{portfolio_summary.get('unrealized_pnl_pct', 0):.1%}")
    p4.metric("Open risk", rupiah(portfolio_summary.get("open_risk_idr", np.nan)))
    p5.metric("Cash", rupiah(portfolio_summary.get("cash_on_hand_idr", np.nan)))
    p6.metric("Equity basis", rupiah(portfolio_summary.get("estimated_equity_idr", np.nan)))

    inferred = float(portfolio_summary.get("inferred_equity_idr", 0.0) or 0.0)
    equity = float(portfolio_summary.get("estimated_equity_idr", 0.0) or 0.0)
    source = str(portfolio_summary.get("equity_source", ""))
    if source == "ACCOUNT_EQUITY_INPUT" and inferred > 0 and equity > inferred * 1.50:
        st.warning(
            "Equity akun yang diinput jauh lebih besar daripada nilai posisi + cash. "
            "Bobot posisi menggunakan equity input. Pastikan angka Equity akun dan Cash on hand sesuai Stockbit terbaru."
        )

    portfolio_columns = [
        "ticker", "position_action", "action_reason", "lots", "avg_price", "last_price",
        "unrealized_pnl_idr", "unrealized_pnl_pct", "position_weight_pct",
        "open_risk_idr", "open_risk_pct_equity_pct", "existing_stop_loss",
        "structural_stop_loss", "suggested_stop_loss", "suggested_tp1", "suggested_tp2",
        "tp1_basis", "tp2_basis", "scanner_setup", "scanner_status", "portfolio_add_setup",
        "avg_down_lots", "avg_down_price", "new_average_after_avg", "trend_up",
        "long_term_structure_intact", "flow_positive", "stop_breached",
        "confirmed_structure_breakdown", "fundamental_distress",
    ]
    portfolio_view = portfolio_analysis[[c for c in portfolio_columns if c in portfolio_analysis.columns]].copy()
    if "unrealized_pnl_pct" in portfolio_view:
        portfolio_view["unrealized_pnl_pct"] = pd.to_numeric(
            portfolio_view["unrealized_pnl_pct"], errors="coerce"
        ) * 100.0
    st.dataframe(
        portfolio_view, width="stretch", hide_index=True,
        column_config={
            "avg_price": st.column_config.NumberColumn("Average", format="Rp %.0f"),
            "last_price": st.column_config.NumberColumn("Last", format="Rp %.0f"),
            "unrealized_pnl_idr": st.column_config.NumberColumn("Unrealized P/L", format="Rp %.0f"),
            "unrealized_pnl_pct": st.column_config.NumberColumn("P/L %", format="%.1f%%"),
            "position_weight_pct": st.column_config.NumberColumn("Weight", format="%.1f%%"),
            "open_risk_idr": st.column_config.NumberColumn("Open risk", format="Rp %.0f"),
            "open_risk_pct_equity_pct": st.column_config.NumberColumn("Risk equity", format="%.2f%%"),
            "existing_stop_loss": st.column_config.NumberColumn("Existing SL", format="Rp %.0f"),
            "structural_stop_loss": st.column_config.NumberColumn("Structural SL", format="Rp %.0f"),
            "suggested_stop_loss": st.column_config.NumberColumn("Suggested SL", format="Rp %.0f"),
            "suggested_tp1": st.column_config.NumberColumn("TP1", format="Rp %.0f"),
            "suggested_tp2": st.column_config.NumberColumn("TP2", format="Rp %.0f"),
            "avg_down_price": st.column_config.NumberColumn("Avg-down price", format="Rp %.0f"),
            "new_average_after_avg": st.column_config.NumberColumn("New average", format="Rp %.0f"),
        },
    )
    st.caption(
        "Tidak adanya setup entry baru bersifat netral untuk posisi lama. CUT_LOSS hanya muncul saat stop tersentuh, "
        "breakdown multi-faktor terkonfirmasi, atau distress berat disertai kerusakan struktur."
    )
    source_quorum = result.get("source_quorum_audit", pd.DataFrame())
    if not source_quorum.empty:
        with st.expander("Audit source quorum portfolio"):
            st.dataframe(source_quorum, hide_index=True, width="stretch")
    st.download_button(
        "Download portfolio action plan",
        portfolio_analysis.to_csv(index=False).encode("utf-8"),
        "stockbit_portfolio_action_plan.csv",
        "text/csv",
        width="stretch",
    )


def run_single_ticker_deep_dive(
    ticker: str,
    lookback: str,
    config: ScanConfig,
    *,
    twelve_data_api_key: str = "",
    itick_api_token: str = "",
) -> dict:
    """Run a bounded full daily review for one ticker without a universe CSV."""
    audit_warnings: list[str] = []
    normalized = normalize_idx_ticker(ticker)
    histories, download_report = download_ohlcv(
        (normalized,), period=lookback, itick_api_token=itick_api_token,
    )
    history = histories.get(normalized)
    if history is None or history.empty:
        reason = getattr(download_report, "failed", {}).get(normalized, "OHLCV tidak tersedia")
        raise ValueError(reason)
    benchmark = download_benchmark(period=lookback)
    mini = ScanEngine(config).scan(histories, benchmark)
    signals = mini.get("signals", pd.DataFrame()).copy()
    if not signals.empty:
        signals["technical_setup_ready"] = signals.get("status", pd.Series(index=signals.index, dtype=str)).eq("EXECUTION_READY")
        source_tiers = getattr(download_report, "source_tiers", {}) or {}
        signals["ohlcv_source_tier"] = signals["ticker"].map(source_tiers).fillna("UNAVAILABLE")
        signals = apply_universe_integrity_gate(signals, [normalized], mini.get("prepared", {}).keys(), config)

    fundamentals = cached_fundamentals((normalized,))
    fundamental_history = pd.DataFrame()
    fundamental_report = pd.DataFrame()
    if fundamentals is not None and not fundamentals.empty:
        try:
            fundamentals, fundamental_history, fundamental_report = enrich_fundamental_shortlist(
                fundamentals,
                (normalized,),
                pd.DataFrame(),
                config,
                bool(twelve_data_api_key),
                twelve_data_api_key,
            )
        except Exception as exc:
            audit_warnings.append(f"Fundamental history enrichment gagal: {exc}")

    automatic_pm = pd.DataFrame()
    automatic_forward_report = pd.DataFrame()
    if fundamentals is not None and not fundamentals.empty and bool(getattr(config, "automatic_forward_quality_enabled", True)):
        try:
            automatic_pm, automatic_forward_report = collect_automatic_forward_quality(
                fundamentals, [normalized], config, force_refresh=False,
            )
        except Exception as exc:
            automatic_pm, automatic_forward_report = pd.DataFrame(), pd.DataFrame()
            audit_warnings.append(f"Forward project/management collection gagal: {exc}")

    signals = attach_fundamentals(signals, fundamentals)
    signals = apply_fundamental_gate(signals, config)
    market_status = cached_automatic_market_status((normalized,))
    news_review = cached_automatic_news((normalized,), int(getattr(config, "min_news_lookback_days", 30)))
    signals = apply_market_status_gate(signals, market_status, config)
    signals = apply_news_gate(signals, news_review, config)

    snapshots = cached_execution_snapshots((normalized,))
    signals = apply_execution_snapshot_gate(signals, snapshots, config)
    source_tiers = getattr(download_report, "source_tiers", {}) or {}
    independent_data = pd.DataFrame()
    independent_report = pd.DataFrame()
    try:
        last_date = pd.Timestamp(history.index[-1]).strftime("%Y-%m-%d")
        last_close = float(pd.to_numeric(history["Close"], errors="coerce").dropna().iloc[-1])
        independent_data, independent_report = fetch_automatic_independent_prices(
            (normalized,),
            reference_date=last_date,
            twelve_data_api_key=twelve_data_api_key,
            itick_api_token=itick_api_token,
            primary_reference={normalized: (last_date, last_close)},
            primary_source_tiers={normalized: str(source_tiers.get(normalized, "UNKNOWN"))},
            config=config,
        )
    except Exception as exc:
        independent_data, independent_report = pd.DataFrame(), pd.DataFrame()
        audit_warnings.append(f"Harga independen otomatis gagal: {exc}")
    price_validation = build_independent_price_validation(
        histories, independent_data, config, primary_source_tiers=source_tiers,
    )
    signals = apply_independent_price_gate(signals, price_validation, config)
    signals = attach_position_sizing(signals, config)
    signals = apply_analyst_fusion_gate(signals, config)
    signals = finalize_execution_integrity(signals, config)
    signals = sort_signals(signals)

    tc_config = TimeCycleConfig(
        min_bars=int(getattr(config, "time_cycle_min_history_bars", 260)),
        lunar_enabled=bool(getattr(config, "time_cycle_lunar_enabled", True)),
        eoff_enabled=bool(getattr(config, "eoff_enabled", True)),
        eoff_ephemeris_enabled=bool(getattr(config, "eoff_ephemeris_enabled", True)),
        eoff_min_fib_cluster=int(getattr(config, "eoff_min_fib_cluster", 4)),
        eoff_aspect_orb_deg=float(getattr(config, "eoff_aspect_orb_deg", 3.0)),
        eoff_require_astro_fib_confluence=bool(getattr(config, "eoff_require_astro_fib_confluence", True)),
    )
    signals = enrich_core_signals_with_time_cycle(
        signals,
        mini.get("prepared", {}),
        enabled=bool(getattr(config, "time_cycle_enabled", True)),
        max_weight=float(getattr(config, "time_cycle_core_max_weight", 0.10)),
        min_confidence=float(getattr(config, "time_cycle_min_confidence", 55.0)),
        config=tc_config,
    )
    time_cycle = analyze_time_cycle(history, config=tc_config)

    specialty = build_specialty_screens(
        mini.get("prepared", {}),
        fundamentals=fundamentals,
        core_signals=signals,
        project_management=automatic_pm,
        market_context=mini.get("market_context"),
        intraday={},
        config=config,
        now=pd.Timestamp.now(tz="Asia/Jakarta"),
        current_positions=0,
        current_open_risk_idr=0.0,
        cash_on_hand_idr=float(getattr(config, "cash_on_hand_idr", 0.0)),
    )
    specialty["sniper"] = enrich_swing_specialty_with_time_cycle(
        specialty.get("sniper", pd.DataFrame()),
        mini.get("prepared", {}),
        enabled=bool(getattr(config, "time_cycle_enabled", True)),
        max_weight=float(getattr(config, "time_cycle_core_max_weight", 0.10)),
        min_confidence=float(getattr(config, "time_cycle_min_confidence", 55.0)),
        config=tc_config,
    )
    try:
        specialty["profit_order_builder"] = build_profit_order_builder(
            signals, specialty, config, validation_events=pd.DataFrame(), ai_memory=pd.DataFrame(),
        )
    except Exception as exc:
        specialty["profit_order_builder"] = pd.DataFrame()
        audit_warnings.append(f"Profit Order Builder gagal: {exc}")

    detail_result = {
        "mode": "single_ticker",
        "ticker": normalized,
        "signals": signals,
        "prepared": mini.get("prepared", {}),
        "all_histories": histories,
        "specialty_screens": specialty,
        "fundamentals": fundamentals,
    }
    ranking = build_top20_ranking(detail_result, limit=5)
    if not ranking.empty:
        row = ranking.iloc[0].to_dict()
        summary = {
            "decision": row.get("decision"),
            "score": row.get("combined_score"),
            "best_buy_date": row.get("best_buy_date"),
            "entry_low": row.get("entry_low"),
            "entry_high": row.get("entry_high"),
            "trigger": row.get("trigger"),
            "stop_loss": row.get("stop_loss"),
            "tp1": row.get("tp1"),
            "tp2": row.get("tp2"),
            "time_cycle_score": row.get("time_cycle_score"),
            "reason": row.get("reason"),
        }
    else:
        summary = {
            "decision": time_cycle.get("quick_buy_action", "WAIT"),
            "score": time_cycle.get("best_buy_score", 0.0),
            "best_buy_date": time_cycle.get("best_buy_date", ""),
            "entry_low": time_cycle.get("best_buy_entry_low"),
            "entry_high": time_cycle.get("best_buy_entry_high"),
            "trigger": time_cycle.get("best_buy_trigger"),
            "stop_loss": time_cycle.get("best_buy_stop_loss"),
            "tp1": time_cycle.get("best_buy_tp1"),
            "tp2": time_cycle.get("best_buy_tp2"),
            "time_cycle_score": time_cycle.get("time_cycle_score"),
            "reason": time_cycle.get("best_buy_reason"),
        }
    multibagger = specialty.get("multibagger", pd.DataFrame())
    if isinstance(multibagger, pd.DataFrame) and not multibagger.empty:
        summary["multibagger_score"] = _finite_or_default(
            multibagger.iloc[0].get("capital_conviction_score", multibagger.iloc[0].get("multibagger_score")), 0.0,
        )
    else:
        summary["multibagger_score"] = 0.0
    return {
        "ticker": normalized,
        "summary": summary,
        "signals": signals,
        "multibagger": multibagger,
        "time_cycle": time_cycle,
        "history": history,
        "fundamentals": fundamentals,
        "fundamental_history": fundamental_history,
        "fundamental_report": fundamental_report,
        "forward_report": automatic_forward_report,
        "independent_report": independent_report,
        "audit_warnings": audit_warnings,
    }


def _finite_or_default(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


st.title("IDX Super Scanner — Best Buy & EOFF Top-20 v6.6.7")
st.caption("Rule engine menjaga validitas struktur; AI lokal gratis belajar dari walk-forward dan outcome OHLCV scanner. Tanpa bukti teresolusi dan skill OOS, AI tidak boleh mengubah ranking.")

with st.sidebar:
    twelve_data_api_key = configured_twelve_data_key()
    itick_api_token = configured_itick_token()
    st.header("Execution policy")
    execution_policy = st.radio(
        "Mode keputusan",
        ["SIGNAL_FIRST", "ACCOUNT_GUARDED"],
        index=0,
        horizontal=True,
        help=(
            "SIGNAL_FIRST menerbitkan SIGNAL_READY tanpa menyebutnya siap eksekusi; risiko tetap terlihat. "
            "ACCOUNT_GUARDED menambahkan verifikasi harga, sizing, cash, slot posisi, dan portfolio heat sebelum ORDER_READY."
        ),
    )
    if execution_policy == "SIGNAL_FIRST":
        st.info("Signal-First aktif: SIGNAL_READY adalah radar, bukan instruksi beli. READY_FOR_STOCKBIT_VERIFY berarti semua gate non-akun lolos dan harga/spread/gap masih wajib dicek di Stockbit.")
    else:
        st.warning("Account-Guarded aktif: ORDER_READY dapat diblokir oleh cash, sizing, regime, likuiditas, RR, dan portfolio heat.")
    real_money_mode = True
    with st.expander("Akun dan risiko — informasional / Account-Guarded", expanded=execution_policy == "ACCOUNT_GUARDED"):
        account_size = st.number_input("Equity akun (Rp)", 1_000_000, 10_000_000_000, 5_000_000, 500_000)
        cash_on_hand = st.number_input("Cash on hand (Rp)", 0, 10_000_000_000, 5_000_000, 100_000)
        risk_per_trade = st.slider("Risiko maksimum per transaksi", 0.25, 1.00, 0.50, 0.05) / 100
        max_positions = st.number_input("Maksimum posisi bersamaan", 1, 10, 3, 1)
        max_position_pct = st.slider("Maksimum modal per saham", 10, 50, 35, 5) / 100
        current_positions_manual = st.number_input("Posisi yang sedang terbuka", 0, 20, 0, 1)
        current_invested_manual = float(st.number_input("Modal sedang terpakai (Rp)", 0, 10_000_000_000, 0, 100_000))
        current_open_risk_manual = float(st.number_input("Open risk saat ini (Rp)", 0, 1_000_000_000, 0, 10_000))
        portfolio_equity_mode = st.selectbox(
            "Dasar bobot portfolio",
            ["Gunakan Equity akun", "Estimasi nilai posisi + cash"],
            index=0,
            help="Jika portfolio CSV diunggah, nilai posisi aktual menggantikan input manual.",
        )
    period = st.selectbox("Riwayat OHLCV", ["3y", "5y", "2y"], index=0)
    min_adtv_b = st.number_input("Minimum ADTV20 (Rp miliar)", 0.1, 100.0, 2.0, 0.5)
    min_score = st.slider("Minimum quality score", 50, 90, 72)
    execution_score = st.slider("Execution-ready technical score", min_score, 95, max(82, min_score))
    min_execution_confidence = st.slider("Minimum total execution confidence", 75, 95, 82)
    min_data_completeness = st.slider(
        "Minimum data completeness", 70, 95, 80,
        help="BUY_LIMIT tidak akan diterbitkan bila evidence coverage di bawah batas ini.",
    )
    max_stop_pct = st.slider("Maksimum jarak SL", 3.0, 10.0, 7.0, 0.5) / 100
    min_rr2 = st.slider("Minimum RR ke TP2", 2.0, 4.0, 2.7, 0.1)
    validate = st.checkbox(
        "Chronological OOS validation",
        value=True,
        help="Dipakai pada scanner universe; tidak diwajibkan untuk portfolio-only review.",
    )
    use_fundamentals = st.checkbox(
        "Ambil fundamental kandidat dan portfolio",
        value=True,
        help="Real-money BUY_LIMIT memerlukan fundamental coverage minimal 45%; bila dimatikan, kandidat tetap menjadi watchlist/PENDING_DATA.",
    )
    multibagger_full_universe = st.checkbox(
        "Multibagger: fundamental seluruh universe",
        value=True,
        disabled=not use_fundamentals,
        help="Fokus compounding memerlukan coverage universe luas. Scan pertama lebih lambat; snapshot fundamental dicache maksimal 21 hari; histori resmi tetap dicache dan direfresh berdasarkan periode laporan.",
    )
    fundamental_n = st.slider("Jumlah kandidat fundamental", 10, 300, 60, disabled=not use_fundamentals or multibagger_full_universe)
    fundamental_history_n = st.slider(
        "Histori laporan untuk shortlist Multibagger", 10, 80, 40, 5,
        disabled=not use_fundamentals,
        help="Mengambil laporan kuartalan/tahunan untuk kandidat snapshot terbaik dan posisi portfolio. Bounded agar full-universe scan tidak terkena rate limit.",
    )
    idx_fundamental_n = st.slider(
        "IDX/XBRL otomatis (maksimum emiten)", 5, 40, 20, 5,
        disabled=not use_fundamentals,
        help=(
            "Tanpa upload dan tanpa API key. Scanner mencari serta memparse filing XBRL resmi untuk shortlist. "
            "Endpoint halaman publik IDX dapat berubah, sehingga kegagalan akan fail-soft dan terlihat di audit provider."
        ),
    )
    enable_twelve_fundamentals = st.checkbox(
        "Twelve Data: laporan fundamental kedua",
        value=bool(twelve_data_api_key),
        disabled=(not use_fundamentals) or (not bool(twelve_data_api_key)),
        help="Memerlukan TWELVE_DATA_API_KEY dan paket yang mencakup endpoint income statement, balance sheet, serta cash flow.",
    )
    with st.expander("Compounding Multibagger — Capital Allocation", expanded=True):
        multibagger_base_capital = float(st.number_input("Modal pokok bucket Multibagger (Rp)", 0, 10_000_000_000, 0, 100_000))
        realized_profit_to_compound = float(st.number_input("Profit trading terealisasi untuk dipindahkan (Rp)", 0, 10_000_000_000, 0, 50_000))
        profit_allocation_pct = st.slider("Porsi profit ke bucket Multibagger", 0, 100, 100, 5) / 100
        multibagger_max_holdings = st.slider("Maksimum saham Multibagger inti", 2, 10, 5, 1)
        multibagger_min_conviction = st.slider("Minimum capital conviction", 60, 90, 72, 1)
        effective_multibagger_budget = multibagger_base_capital + realized_profit_to_compound * profit_allocation_pct
        st.caption(
            f"Budget efektif: {rupiah(effective_multibagger_budget)}. Dana terbesar diarahkan ke conviction tertinggi; "
            "bagian kandidat yang belum masuk zona entry tetap menjadi cash reserve."
        )
    with st.expander("AI Lokal Gratis — Hybrid Learning", expanded=True):
        ai_enabled = st.checkbox(
            "Aktifkan AI lokal", value=True,
            help="Tidak memakai API berbayar. Model belajar dari chronological walk-forward dan outcome memory scanner.",
        )
        ai_mode = st.selectbox(
            "Mode AI", ["HYBRID_GUARDED", "SHADOW_ONLY", "RULE_ONLY"], index=0,
            help=(
                "HYBRID_GUARDED mengoreksi ranking maksimal sesuai confidence data. "
                "SHADOW_ONLY hanya menampilkan prediksi AI tanpa mengubah urutan. RULE_ONLY mematikan AI."
            ),
        )
        ai_max_weight_pct = st.slider(
            "Bobot maksimum AI dalam conviction", 5, 35, 35, 5,
            disabled=not ai_enabled or ai_mode == "RULE_ONLY",
            help="Bobot aktual menjadi nol tanpa outcome, tanpa dukungan strategi, jika coverage fitur rendah, atau bila model tidak mengalahkan baseline OOS.",
        )
        ai_min_training_events = st.slider(
            "Minimum event untuk model statistik", 20, 100, 30, 5,
            disabled=not ai_enabled or ai_mode == "RULE_ONLY",
        )
        ai_memory_uploaded = st.file_uploader(
            "Import AI outcome memory (opsional)", type=["csv"], key="ai_memory_upload",
            help=(
                "Streamlit gratis dapat menghapus disk saat sleep/redeploy. Export memory dari Order Builder dan import kembali "
                "agar pembelajaran lintas sesi tidak hilang."
            ),
        )
        st.caption(
            "AI = validated regularized logistic untuk ranking; similarity/KNN + Bayesian prior sebagai diagnostik/shadow; chronological evaluation + drift/coverage guard. "
            "No evidence = no influence; rule engine tetap memegang invalidasi struktur, harga, dan data."
        )
    with st.expander("Time-Cycle Intelligence — Swing/Core & Multibagger", expanded=True):
        time_cycle_enabled = st.checkbox(
            "Aktifkan objective Astronacci-style time cycle", value=True,
            help=(
                "Menerapkan swing timing, Fibonacci time, autocorrelation, spectral cycle, dan moon phase tervalidasi. "
                "Hanya memengaruhi core/swing dan Multibagger; tidak memengaruhi BPJS, BSJP, ARA, atau intraday."
            ),
        )
        time_cycle_core_weight_pct = st.slider(
            "Bobot maksimum time-cycle pada ranking core", 0, 10, 10, 1, disabled=not time_cycle_enabled,
        )
        time_cycle_multibagger_weight_pct = st.slider(
            "Bobot maksimum time-cycle pada capital conviction", 0, 5, 5, 1, disabled=not time_cycle_enabled,
        )
        time_cycle_min_confidence = st.slider(
            "Minimum confidence cycle agar berpengaruh", 45, 75, 55, 5, disabled=not time_cycle_enabled,
        )
        time_cycle_lunar_enabled = st.checkbox(
            "Gunakan Full/New Moon sebagai time marker eksperimental", value=True, disabled=not time_cycle_enabled,
            help="Moon phase tidak menentukan arah. Bobotnya hanya aktif bila kedekatan pivot historis mengalahkan baseline.",
        )
        st.markdown("**Clean-room Eye-of-Future reconstruction**")
        eoff_enabled = st.checkbox(
            "Aktifkan EOFF reconstruction lengkap", value=True, disabled=not time_cycle_enabled,
            help=(
                "Menggabungkan geocentric ephemeris, Moon declination, aspect, ingress/retrograde, "
                "Sun annual cycle, ≥4 multi-anchor Fibonacci time projections, price, pattern, dan momentum."
            ),
        )
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            eoff_min_fib_cluster = st.slider(
                "Minimum Fibonacci projection cluster", 3, 8, 4, 1,
                disabled=not time_cycle_enabled or not eoff_enabled,
            )
        with ec2:
            eoff_aspect_orb_deg = st.slider(
                "Planetary aspect orb (derajat)", 1.0, 6.0, 3.0, 0.5,
                disabled=not time_cycle_enabled or not eoff_enabled,
            )
        with ec3:
            eoff_require_astro_fib_confluence = st.checkbox(
                "Wajib astro + Fibonacci cluster", value=True,
                disabled=not time_cycle_enabled or not eoff_enabled,
                help="Tanpa confluence ini, EOFF hanya menjadi shadow diagnostic dan tidak memengaruhi ranking.",
            )
        eoff_ephemeris_enabled = st.checkbox(
            "Aktifkan ephemeris geosentris offline", value=True,
            disabled=not time_cycle_enabled or not eoff_enabled,
            help="Menggunakan PyEphem lokal; tidak membutuhkan API atau file ephemeris eksternal.",
        )
        st.caption(
            "Implementasi clean-room ini tidak mengklaim formula proprietary asli. EOFF hanya masuk ke daily core/swing "
            "dan timing Multibagger setelah confluence serta historical evidence lolos; intraday, BPJS, BSJP, dan ARA tetap 0%."
        )
    st.subheader("Data otomatis")
    st.caption(
        "OHLCV gratis: cache current → Yahoo → IDX Stock Summary EOD → iTick free bila token tersedia. "
        "Harga kedua memakai cache sama-session → IDX → Google Finance → iTick → Twelve Data opsional. "
        "Fundamental shortlist: IDX/XBRL official-first + Yahoo fallback/cross-check; upload hanya fallback."
    )
    if itick_api_token:
        st.success("Fallback OHLCV iTick free terkonfigurasi; rate guard internal maksimum 4 call/menit.")
    else:
        st.caption("iTick belum dikonfigurasi. Scanner tetap berjalan gratis dengan cache, Yahoo, dan IDX resmi.")
    if twelve_data_api_key:
        st.success("Twelve Data terkonfigurasi untuk harga independen dan, bila paket mendukung, laporan fundamental kedua.")
    else:
        st.caption("Twelve Data tidak dikonfigurasi; IDX dan Google tetap berjalan tanpa API key.")
    enable_intraday_specialty = st.checkbox(
        "Ambil intraday 5m untuk BSJP/BPJS & ARA Hunter",
        value=True,
        help="Shortlist dibagi khusus untuk BPJS, BSJP, ARA, dan core. Daily Radar tetap tersedia di luar shortlist; ORDER_READY memerlukan data intraday.",
    )
    intraday_shortlist_n = st.slider(
        "Maksimum shortlist intraday", 20, 300, 200, 10,
        disabled=not enable_intraday_specialty,
    )
    st.divider()
    st.caption("Semua setup tetap ditampilkan. Core plan verified membawa lot/template, tetapi submit Stockbit tetap manual setelah revalidasi broker.")

sample_csv = b"ticker\nADRO\nANTM\nBRMS\nMDKA\nTAPG\n"
portfolio_sample_csv = (
    b"ticker,lots,avg_price,stop_loss,take_profit,notes\n"
    b"ADRO,10,2150,,,Core position\n"
    b"ANTM,5,1860,,,Trading position\n"
)
fundamental_history_sample_csv = (
    b"ticker,period_end,period_type,statement_basis,source_url,currency,unit_multiplier,shares_multiplier,revenue,net_income,operating_cash_flow,capex,total_assets,total_liabilities,equity,total_debt,cash,shares_outstanding,operating_income,ebit,ebitda,interest_expense,car,npl_gross,ldr\n"
    b"ANTM,2025-03-31,Q1,YTD_CUMULATIVE,https://www.idx.co.id/id/perusahaan-tercatat/laporan-keuangan-dan-tahunan,IDR,1000000,1,1000,100,120,20,5000,2000,3000,800,400,24000000000,150,150,180,20,,,\n"
    b"ANTM,2025-06-30,Q2,YTD_CUMULATIVE,https://www.idx.co.id/id/perusahaan-tercatat/laporan-keuangan-dan-tahunan,IDR,1000000,1,2200,230,260,45,5300,2100,3200,780,470,24000000000,330,330,390,42,,,\n"
)
project_management_sample_csv = (
    b"ticker,as_of,source_url,source_verified,project_name,project_stage,project_completion_pct,project_capex_idr,funding_secured_pct,offtake_secured_pct,expected_revenue_idr,expected_ebitda_idr,expected_cod,ownership_pct,project_delay_months,cost_overrun_pct,strategic_project,project_risk,ceo_name,ceo_tenure_years,board_avg_tenure_years,management_revenue_cagr,management_roic_change_pct,capital_allocation_score,governance_score,board_turnover_3y,insider_ownership_pct,audit_clean,related_party_risk,legal_governance_flags,management_source_url,management_verified\n"
    b"ANTM,2026-07-01,https://www.idx.co.id/,TRUE,Smelter Expansion,CONSTRUCTION,65,2500000000000,100,70,1800000000000,350000000000,2027-06-30,100,0,3,TRUE,MEDIUM,Example CEO,4,5,0.12,0.03,82,80,1,0.10,TRUE,LOW,,https://www.idx.co.id/,TRUE\n"
)
left, right = st.columns([3, 1])
with left:
    uploaded = st.file_uploader(
        "Upload CSV universe ticker IDX",
        type=["csv", "txt"],
        help="Diperlukan hanya untuk menjalankan scanner universe.",
    )
with right:
    st.write("")
    st.write("")
    st.download_button("Unduh contoh universe", sample_csv, "sample_tickers.csv", "text/csv", width="stretch")

p1, p2 = st.columns([3, 1])
with p1:
    portfolio_uploaded = st.file_uploader(
        "Upload snapshot portfolio Stockbit CSV",
        type=["csv", "txt"],
        help="Kolom wajib: ticker, lots, avg_price. Dapat dianalisis tanpa upload universe ticker.",
        key="portfolio_upload",
    )
with p2:
    st.write("")
    st.write("")
    st.download_button(
        "Unduh template portfolio", portfolio_sample_csv, "stockbit_portfolio_template.csv", "text/csv", width="stretch"
    )

fundamental_history_uploaded = None
with st.expander("Fallback manual laporan fundamental — biasanya tidak diperlukan"):
    st.caption(
        "Default v6.6.7 mengambil IDX/XBRL official-first, lalu Yahoo fallback/cross-check terbatas. Gunakan CSV hanya bila endpoint IDX sedang berubah/terblokir, "
        "emiten tidak menyediakan XBRL yang dapat diparse, atau Anda ingin menambah hasil rekonsiliasi sendiri."
    )
    f1, f2 = st.columns([3, 1])
    with f1:
        fundamental_history_uploaded = st.file_uploader(
            "Upload histori laporan IDX/XBRL (fallback opsional)",
            type=["csv", "txt"],
            key="fundamental_history_upload",
            help=(
                "Gunakan minimal 8 kuartal. source_url harus HTTPS pada idx.co.id agar provenance dikenali. "
                "Pilih YTD_CUMULATIVE untuk laporan interim kumulatif; scanner mengubahnya menjadi kuartal mandiri."
            ),
        )
    with f2:
        st.write("")
        st.write("")
        st.download_button(
            "Template fallback", fundamental_history_sample_csv,
            "idx_fundamental_history_template.csv", "text/csv", width="stretch",
        )
    st.caption(
        "Upload tidak otomatis dianggap benar hanya karena diberi label IDX: domain sumber, identitas akuntansi, "
        "staleness, dan perbedaan antar-provider tetap diperiksa."
    )

with st.expander("Forward Intelligence Otomatis — Project, Management & Future Fundamental", expanded=False):
    st.caption(
        "Default: scanner mencari dokumen IDX/OJK dan investor-relations emiten, mengekstrak proyek serta manajemen, "
        "memeriksa quorum sumber, lalu menghitung skenario dampak ke revenue, EBITDA, laba, FCF, dan utang. "
        "Upload manual hanya override opsional bila Anda memiliki bukti yang lebih lengkap."
    )
    automatic_forward_quality_enabled = st.checkbox(
        "Aktifkan pencarian project & management otomatis", value=True,
        help="Hanya kandidat fundamental teratas yang dicari. Hasil disimpan cache agar scan berikutnya lebih cepat."
    )
    af1, af2, af3 = st.columns(3)
    with af1:
        automatic_forward_quality_top_n = st.slider("Emiten forward review", 5, 30, 12, 1)
    with af2:
        automatic_forward_quality_cache_days = st.slider("Cache review (hari)", 3, 30, 14, 1)
    with af3:
        automatic_forward_quality_max_documents = st.slider("Dokumen/emiten", 2, 8, 5, 1)
    pm1, pm2 = st.columns([3, 1])
    with pm1:
        project_management_uploaded = st.file_uploader(
            "Project & Management Review CSV (opsional)",
            type=["csv", "txt"],
            key="project_management_upload",
            help=(
                "Satu ticker boleh memiliki beberapa baris proyek. Field utama: project_stage, completion, funding, offtake, delay, cost overrun, "
                "CEO/board tenure, revenue CAGR/ROIC di bawah manajemen, capital allocation, governance, dan source verification."
            ),
        )
    with pm2:
        st.write("")
        st.write("")
        st.download_button(
            "Template Project/Management", project_management_sample_csv,
            "project_management_review_template.csv", "text/csv", width="stretch",
        )

broksum_sample_csv = (
    b"ticker,date,broker_code,buy_value,sell_value\n"
    b"ANTM,2026-07-15,YP,15000000000,5000000000\n"
    b"ANTM,2026-07-15,CC,7000000000,9000000000\n"
)
orderbook_sample_csv = (
    b"ticker,timestamp,level,bid_price,bid_lots,offer_price,offer_lots\n"
    b"ANTM,2026-07-15 15:55:00,1,3200,250000,3210,50000\n"
    b"ANTM,2026-07-15 15:55:00,2,3190,180000,3220,40000\n"
)
with st.expander("Konfirmasi opsional: Broker Summary dan Order Book"):
    st.caption(
        "Tidak wajib. Tanpa upload, scanner memakai proxy orderflow/queue dari OHLCV intraday. "
        "Upload data aktual hanya untuk menaikkan atau menurunkan conviction ARA."
    )
    bcol1, bcol2 = st.columns(2)
    with bcol1:
        broksum_uploaded = st.file_uploader(
            "Broker Summary CSV (opsional)", type=["csv", "txt"], key="broksum_upload",
            help="Kolom: ticker, date, broker_code, dan buy_value/sell_value atau buy_volume/sell_volume.",
        )
        st.download_button("Template Broker Summary", broksum_sample_csv, "broker_summary_template.csv", "text/csv")
    with bcol2:
        orderbook_uploaded = st.file_uploader(
            "Order Book snapshot CSV (opsional)", type=["csv", "txt"], key="orderbook_upload",
            help="Kolom: ticker, timestamp, level, bid_price, bid_lots, offer_price, offer_lots.",
        )
        st.download_button("Template Order Book", orderbook_sample_csv, "orderbook_snapshot_template.csv", "text/csv")

st.success(
    "Scanner otomatis menghitung orderflow proxy dan queue proxy dari OHLCV intraday. Proxy tersebut tidak diklaim "
    "sebagai broker summary atau antrean bid aktual. Data Stockbit/market depth dapat diunggah sebagai konfirmasi opsional."
)

now_jkt_ui = pd.Timestamp.now(tz="Asia/Jakarta")
if idx_regular_decision_window(now_jkt_ui):
    st.warning(
        "Jam reguler IDX masih berada dalam window candle berjalan. Kandidat teknikal ditampilkan sebagai "
        "PENDING_CLOSE dan perlu di-refresh setelah 16:20 WIB."
    )
elif now_jkt_ui.weekday() < 5 and not idx_daily_bar_is_final(now_jkt_ui):
    st.success(
        "Mode PRE-MARKET: scanner memakai completed EOD hari bursa sebelumnya. Kandidat yang lulus seluruh gate "
        "adalah PLAN_VERIFIED; harga pembukaan dan spread Stockbit tetap harus diperiksa setelah pasar mulai."
    )

st.info(
    "Begitu CSV universe diunggah, scan pertama dimulai otomatis. Tombol refresh dipakai untuk mengambil snapshot "
    "terbaru. Analisis portfolio saja tetap tersedia tanpa universe ticker."
)

b1, b2 = st.columns(2)
manual_refresh = b1.button("Scan ulang / refresh data", type="primary", width="stretch", disabled=uploaded is None)
run_portfolio = b2.button(
    "Analisis portfolio saja", type="secondary", width="stretch", disabled=portfolio_uploaded is None
)
scan_signature = upload_fingerprint(
    uploaded, portfolio_uploaded, fundamental_history_uploaded, project_management_uploaded,
    broksum_uploaded, orderbook_uploaded, ai_memory_uploaded,
) if uploaded is not None else ""
new_upload = bool(
    uploaded is not None
    and st.session_state.get("_last_auto_scan_signature") != scan_signature
)
run_scan = bool(uploaded is not None and (new_upload or manual_refresh) and not run_portfolio)
if new_upload and not run_portfolio:
    st.caption("CSV baru terdeteksi—scanner dimulai otomatis.")

cfg = ScanConfig().replace(
    min_adtv_idr=float(min_adtv_b) * 1_000_000_000,
    min_score=float(min_score),
    execution_score=float(execution_score),
    min_execution_confidence=float(min_execution_confidence),
    min_data_completeness=float(min_data_completeness),
    max_stop_pct=float(max_stop_pct),
    min_rr2=float(min_rr2),
    fundamental_top_n=int(fundamental_n),
    fundamental_history_top_n=int(fundamental_history_n),
    idx_fundamental_top_n=min(int(idx_fundamental_n), int(fundamental_history_n)),
    idx_fundamental_years_back=3,
    twelve_fundamental_top_n=min(20, int(fundamental_history_n)),
    real_money_mode=bool(real_money_mode),
    require_fundamentals=bool(use_fundamentals),
    require_market_status=False,
    require_news_review=False,
    require_validation=False,
    require_independent_price_verification=True,
    execution_policy=str(execution_policy),
    autopilot_enabled=bool(execution_policy == "ACCOUNT_GUARDED"),
    allow_autopilot_risk_off=False,
    account_size_idr=float(account_size),
    cash_on_hand_idr=float(cash_on_hand),
    risk_per_trade_pct=float(risk_per_trade),
    max_positions=int(max_positions),
    max_position_pct=float(max_position_pct),
    multibagger_profit_allocation_pct=float(profit_allocation_pct),
    multibagger_capital_budget_idr=float(effective_multibagger_budget),
    multibagger_max_holdings=int(multibagger_max_holdings),
    multibagger_min_capital_conviction=float(multibagger_min_conviction),
    automatic_forward_quality_enabled=bool(automatic_forward_quality_enabled),
    automatic_forward_quality_top_n=int(automatic_forward_quality_top_n),
    automatic_forward_quality_cache_days=int(automatic_forward_quality_cache_days),
    automatic_forward_quality_max_documents=int(automatic_forward_quality_max_documents),
    ai_enabled=bool(ai_enabled),
    ai_mode=str(ai_mode),
    ai_max_weight=float(ai_max_weight_pct) / 100.0,
    ai_min_training_events=int(ai_min_training_events),
    time_cycle_enabled=bool(time_cycle_enabled),
    time_cycle_core_max_weight=float(time_cycle_core_weight_pct) / 100.0,
    time_cycle_multibagger_max_weight=float(time_cycle_multibagger_weight_pct) / 100.0,
    time_cycle_min_confidence=float(time_cycle_min_confidence),
    time_cycle_lunar_enabled=bool(time_cycle_lunar_enabled),
    eoff_enabled=bool(eoff_enabled),
    eoff_ephemeris_enabled=bool(eoff_ephemeris_enabled),
    eoff_min_fib_cluster=int(eoff_min_fib_cluster),
    eoff_aspect_orb_deg=float(eoff_aspect_orb_deg),
    eoff_require_astro_fib_confluence=bool(eoff_require_astro_fib_confluence),
)
portfolio_equity_input = float(account_size) if portfolio_equity_mode == "Gunakan Equity akun" else None

uploaded_project_management = pd.DataFrame()
if project_management_uploaded is not None:
    try:
        uploaded_project_management = parse_project_management_csv(project_management_uploaded)
        st.success(
            f"Project/management review terbaca: {len(uploaded_project_management)} baris, "
            f"{uploaded_project_management['ticker'].nunique()} emiten."
        )
    except Exception as exc:
        st.error(f"Project/management CSV tidak dapat dibaca: {exc}")
        st.stop()

uploaded_fundamental_history = pd.DataFrame()
if fundamental_history_uploaded is not None:
    try:
        uploaded_fundamental_history = parse_fundamental_history_csv(fundamental_history_uploaded)
        official_rows = int(uploaded_fundamental_history["source_family"].eq("IDX_OFFICIAL_REFERENCE").sum())
        st.success(
            f"Histori fundamental terbaca: {len(uploaded_fundamental_history)} baris, "
            f"{uploaded_fundamental_history['ticker'].nunique()} emiten, {official_rows} baris merujuk domain IDX."
        )
    except Exception as exc:
        st.error(f"Histori fundamental tidak dapat dibaca: {exc}")
        st.stop()

if run_portfolio and portfolio_uploaded is not None:
    try:
        portfolio = parse_portfolio_csv(portfolio_uploaded)
    except Exception as exc:
        st.error(f"Portfolio CSV tidak dapat dibaca: {exc}")
        st.stop()
    if portfolio.empty:
        st.error("Portfolio CSV tidak memiliki posisi yang valid.")
        st.stop()

    portfolio_tickers = tuple(portfolio["ticker"].drop_duplicates().tolist())
    progress = st.progress(0, text=f"Mengunduh OHLCV {len(portfolio_tickers)} posisi portfolio…")
    histories, report = cached_portfolio_market_data(tuple(portfolio_tickers), period, bool(itick_api_token), itick_api_token)
    progress.progress(55, text="Menghitung struktur, flow, stop, target, dan bobot posisi…")
    fundamentals = cached_fundamentals(portfolio_tickers) if use_fundamentals else pd.DataFrame()
    fundamental_history = pd.DataFrame()
    fundamental_history_report = pd.DataFrame()
    if use_fundamentals and not fundamentals.empty:
        progress.progress(72, text="Mengambil IDX/XBRL official-first + Yahoo fallback/cross-check dan memeriksa kualitas laba…")
        fundamentals, fundamental_history, fundamental_history_report = enrich_fundamental_shortlist(
            fundamentals,
            portfolio_tickers,
            uploaded_fundamental_history,
            cfg,
            enable_twelve_fundamentals,
            twelve_data_api_key,
        )
    portfolio_analysis, portfolio_summary = analyze_portfolio_positions(
        portfolio,
        histories,
        fundamentals=fundamentals,
        signals=pd.DataFrame(),
        account_equity_idr=portfolio_equity_input,
        cash_on_hand_idr=float(cash_on_hand),
        config=cfg,
    )
    source_quorum_audit = build_source_quorum_audit(
        portfolio_tickers,
        source_tiers=getattr(report, "source_tiers", {}) or {},
        fundamental_history=fundamental_history,
        config=cfg,
    )
    st.session_state["scan_result"] = {
        "mode": "portfolio",
        "portfolio": portfolio,
        "portfolio_analysis": portfolio_analysis,
        "portfolio_summary": portfolio_summary,
        "fundamentals": fundamentals,
        "fundamental_history": fundamental_history,
        "fundamental_history_report": fundamental_history_report,
        "project_management_review": combined_project_management if "combined_project_management" in locals() else uploaded_project_management,
        "automatic_forward_report": automatic_forward_report if "automatic_forward_report" in locals() else pd.DataFrame(),
        "source_quorum_audit": source_quorum_audit,
        "all_histories": histories,
        "download_report": report,
    }
    progress.progress(100, text="Portfolio review selesai")
    progress.empty()

if run_scan and uploaded is not None:
    try:
        scan_tickers = parse_ticker_csv(uploaded)
    except Exception as exc:
        st.error(f"CSV universe tidak dapat dibaca: {exc}")
        st.stop()
    if not scan_tickers:
        st.error("Tidak menemukan ticker yang valid di CSV universe.")
        st.stop()

    portfolio = pd.DataFrame()
    if portfolio_uploaded is not None:
        try:
            portfolio = parse_portfolio_csv(portfolio_uploaded)
        except Exception as exc:
            st.error(f"Portfolio CSV tidak dapat dibaca: {exc}")
            st.stop()
    portfolio_tickers = portfolio["ticker"].drop_duplicates().tolist() if not portfolio.empty else []
    all_tickers = list(dict.fromkeys(scan_tickers + portfolio_tickers))

    market_status = pd.DataFrame()
    news_review = pd.DataFrame()
    broksum = pd.DataFrame()
    orderbook = pd.DataFrame()
    if broksum_uploaded is not None:
        try:
            broksum = parse_broker_summary_csv(broksum_uploaded)
        except Exception as exc:
            st.error(f"Broker Summary CSV tidak dapat dibaca: {exc}")
            st.stop()
    if orderbook_uploaded is not None:
        try:
            orderbook = parse_orderbook_snapshot_csv(orderbook_uploaded)
        except Exception as exc:
            st.error(f"Order Book CSV tidak dapat dibaca: {exc}")
            st.stop()
    progress = st.progress(0, text=f"Mengunduh OHLCV {len(all_tickers)} ticker dan IHSG…")
    histories, report, benchmark = cached_market_data(tuple(all_tickers), period, bool(itick_api_token), itick_api_token)
    progress.progress(30, text="Menghitung indikator, struktur pasar, core setup, dan posisi portfolio…")
    result = ScanEngine(cfg).scan(histories, benchmark)
    signals = result["signals"]
    if not signals.empty:
        signals["technical_setup_ready"] = signals["status"].eq("EXECUTION_READY")
    source_tier_map = getattr(report, "source_tiers", {}) or {}
    if not signals.empty and "ticker" in signals:
        signals["ohlcv_source_tier"] = signals["ticker"].map(source_tier_map).fillna("UNAVAILABLE")
    signals = apply_universe_integrity_gate(signals, scan_tickers, result["prepared"].keys(), cfg)

    stats = pd.DataFrame()
    trades = pd.DataFrame()
    if validate and result["prepared"]:
        progress.progress(45, text="Menjalankan chronological OOS validation dengan live-plan parity…")
        stats, trades = run_walkforward_validation(result["prepared"], cfg)
    signals = attach_backtest_stats(signals, stats)
    signals = apply_validation_gate(signals, cfg)

    fundamentals = pd.DataFrame()
    fundamental_history = pd.DataFrame()
    fundamental_history_report = pd.DataFrame()
    if use_fundamentals and (not signals.empty or portfolio_tickers):
        progress.progress(63, text="Mengambil fundamental kandidat dan posisi portfolio…")
        ranked_names = (
            signals.sort_values(["status_rank", "quality_score"], ascending=[True, False])["ticker"].drop_duplicates().tolist()
            if not signals.empty else []
        )
        execution_names = (
            signals.loc[signals["status"].eq("EXECUTION_READY"), "ticker"].drop_duplicates().tolist()
            if not signals.empty else []
        )
        if multibagger_full_universe:
            top_names = list(dict.fromkeys(portfolio_tickers + scan_tickers))
        else:
            top_names = list(dict.fromkeys(portfolio_tickers + execution_names + ranked_names[:fundamental_n]))
        fundamentals = cached_fundamentals(tuple(top_names))
        if not fundamentals.empty:
            progress.progress(68, text="Mengambil IDX/XBRL official-first + Yahoo fallback/cross-check, kualitas laba, dilusi, dan consensus…")
            ranking = fundamentals.copy()
            numeric_column = lambda name, default: pd.to_numeric(
                ranking[name] if name in ranking else pd.Series(default, index=ranking.index),
                errors="coerce",
            ).fillna(default)
            snapshot_score = numeric_column("fundamental_score", 0.0)
            snapshot_coverage = numeric_column("fundamental_coverage", 0.0)
            revenue_growth = numeric_column("revenue_growth", -0.25).clip(-0.25, 0.50)
            earnings_growth = numeric_column("earnings_growth", -0.25).clip(-0.25, 0.75)
            ranking["_history_priority"] = snapshot_score + 0.10 * snapshot_coverage + 8.0 * revenue_growth + 6.0 * earnings_growth
            multibagger_ranked = ranking.sort_values("_history_priority", ascending=False)["ticker"].drop_duplicates().tolist()
            uploaded_names = (
                uploaded_fundamental_history["ticker"].drop_duplicates().tolist()
                if not uploaded_fundamental_history.empty else []
            )
            history_names = tuple(dict.fromkeys(
                portfolio_tickers + execution_names + uploaded_names
                + multibagger_ranked[:int(fundamental_history_n)]
            ))
            fundamentals, fundamental_history, fundamental_history_report = enrich_fundamental_shortlist(
                fundamentals,
                history_names,
                uploaded_fundamental_history,
                cfg,
                enable_twelve_fundamentals,
                twelve_data_api_key,
            )
    automatic_project_management = pd.DataFrame()
    automatic_forward_report = pd.DataFrame()
    if use_fundamentals and automatic_forward_quality_enabled and not fundamentals.empty:
        progress.progress(72, text=f"Mencari project, management, dan forward impact untuk kandidat teratas…")
        automatic_project_management, automatic_forward_report = collect_automatic_forward_quality(
            fundamentals, all_tickers, cfg, force_refresh=False,
        )
    combined_project_management = merge_project_management_reviews(
        automatic_project_management, uploaded_project_management,
    )
    signals = attach_fundamentals(signals, fundamentals)
    signals = apply_fundamental_gate(signals, cfg)

    progress.progress(76, text="Menyelesaikan status IDX dan berita…")
    # Resolve context for every visible actionable candidate, not only rows that
    # were already EXECUTION_READY before context was fetched. Limiting context
    # to pre-ready rows created a circular dependency and left watchlist rows
    # permanently at 37.5–50% completeness.
    visible_context_names = (
        signals.loc[~signals["status"].eq("REJECT")]
        .sort_values(["status_rank", "quality_score"], ascending=[True, False])
        ["ticker"].drop_duplicates().head(60).tolist()
        if not signals.empty else []
    )
    specialty_shortlist = specialty_intraday_shortlist(
        result["prepared"], signals, max_candidates=int(intraday_shortlist_n)
    )
    context_names = list(dict.fromkeys(visible_context_names + portfolio_tickers + specialty_shortlist))
    if context_names:
        market_status = cached_automatic_market_status(tuple(context_names))
        news_review = cached_automatic_news(tuple(context_names), cfg.min_news_lookback_days)
    signals = apply_market_status_gate(signals, market_status, cfg)
    signals = apply_news_gate(signals, news_review, cfg)
    signals = attach_broker_summary(signals, broksum)

    # Specialty tables also receive explicit status/news blockers for tickers
    # that do not happen to have a core setup row.
    specialty_context = pd.DataFrame({
        "ticker": specialty_shortlist,
        "status": ["WATCHLIST_ENTRY"] * len(specialty_shortlist),
        "setup": ["SPECIALTY_CONTEXT"] * len(specialty_shortlist),
        "quality_score": [0.0] * len(specialty_shortlist),
        "composite_score": [0.0] * len(specialty_shortlist),
        "ohlcv_source_tier": [source_tier_map.get(ticker, "UNAVAILABLE") for ticker in specialty_shortlist],
    })
    if not specialty_context.empty:
        specialty_context = apply_market_status_gate(specialty_context, market_status, cfg)
        specialty_context = apply_news_gate(specialty_context, news_review, cfg)
    specialty_signal_context = pd.concat([signals, specialty_context], ignore_index=True, sort=False)

    quote_candidates = (
        signals.loc[~signals["status"].eq("REJECT")]
        .sort_values(["status_rank", "quality_score"], ascending=[True, False])
        ["ticker"].drop_duplicates().head(40).tolist()
        if not signals.empty else []
    )
    snapshots = cached_execution_snapshots(tuple(quote_candidates)) if quote_candidates else pd.DataFrame()
    signals = apply_execution_snapshot_gate(signals, snapshots, cfg)

    independent_price_data = pd.DataFrame()
    automatic_price_report = pd.DataFrame()
    if quote_candidates:
        reference_dates = [
            pd.Timestamp(histories[ticker].index[-1])
            for ticker in quote_candidates
            if ticker in histories and histories[ticker] is not None and not histories[ticker].empty
        ]
        reference_date = max(reference_dates).strftime("%Y-%m-%d") if reference_dates else now_jkt_ui.strftime("%Y-%m-%d")
        automatic_names = tuple(quote_candidates[: int(cfg.max_automatic_price_candidates)])
        primary_reference = tuple(
            (
                ticker,
                pd.Timestamp(histories[ticker].index[-1]).strftime("%Y-%m-%d"),
                float(pd.to_numeric(histories[ticker]["Close"], errors="coerce").dropna().iloc[-1]),
            )
            for ticker in automatic_names
            if ticker in histories
            and histories[ticker] is not None
            and not histories[ticker].empty
            and pd.to_numeric(histories[ticker]["Close"], errors="coerce").dropna().size > 0
        )
        progress.progress(
            84,
            text=f"Memvalidasi harga otomatis untuk {len(automatic_names)} kandidat (IDX → Google → fallback)…",
        )
        primary_source_tiers = tuple(
            (ticker, str(source_tier_map.get(ticker, "UNKNOWN")))
            for ticker in automatic_names
        )
        independent_price_data, automatic_price_report = cached_automatic_independent_prices(
            automatic_names,
            reference_date,
            primary_reference,
            primary_source_tiers,
            cfg,
            twelve_data_api_key,
            itick_api_token,
        )
    price_validation = build_independent_price_validation(
        histories, independent_price_data, cfg,
        primary_source_tiers=source_tier_map,
    )
    signals = apply_independent_price_gate(signals, price_validation, cfg)
    if not specialty_context.empty:
        specialty_context = apply_independent_price_gate(specialty_context, price_validation, cfg)
        specialty_context["independent_price_required"] = bool(
            cfg.real_money_mode and cfg.require_independent_price_verification
        )
    signals = attach_position_sizing(signals, cfg)
    signals = apply_analyst_fusion_gate(signals, cfg)

    portfolio_analysis, portfolio_summary = analyze_portfolio_positions(
        portfolio, histories, fundamentals=fundamentals, signals=signals,
        account_equity_idr=portfolio_equity_input, cash_on_hand_idr=float(cash_on_hand), config=cfg,
    )
    if not portfolio.empty:
        current_positions = int(portfolio_summary.get("positions", len(portfolio)))
        current_invested = float(portfolio_summary.get("market_value_idr", 0.0))
        current_open_risk = float(portfolio_summary.get("open_risk_idr", 0.0))
    else:
        current_positions = int(current_positions_manual)
        current_invested = float(current_invested_manual)
        current_open_risk = float(current_open_risk_manual)

    progress.progress(90, text="Meranking seluruh setup valid tanpa account-risk gate…")
    signals = enforce_analyst_portfolio_budget(
        signals, cfg, current_positions=current_positions, current_open_risk_idr=current_open_risk,
        current_invested_idr=current_invested, cash_on_hand_idr=float(cash_on_hand),
    )
    signals = finalize_execution_integrity(signals, cfg)
    signals = sort_signals(signals)
    signals = enrich_core_signals_with_time_cycle(
        signals, result["prepared"], enabled=bool(cfg.time_cycle_enabled),
        max_weight=float(cfg.time_cycle_core_max_weight),
        min_confidence=float(cfg.time_cycle_min_confidence),
        config=TimeCycleConfig(
            min_bars=int(cfg.time_cycle_min_history_bars),
            lunar_enabled=bool(cfg.time_cycle_lunar_enabled),
            eoff_enabled=bool(cfg.eoff_enabled),
            eoff_ephemeris_enabled=bool(cfg.eoff_ephemeris_enabled),
            eoff_min_fib_cluster=int(cfg.eoff_min_fib_cluster),
            eoff_aspect_orb_deg=float(cfg.eoff_aspect_orb_deg),
            eoff_require_astro_fib_confluence=bool(cfg.eoff_require_astro_fib_confluence),
        ),
    )
    specialty_signal_context = pd.concat([signals, specialty_context], ignore_index=True, sort=False)
    portfolio_analysis, portfolio_summary = analyze_portfolio_positions(
        portfolio, histories, fundamentals=fundamentals, signals=signals,
        account_equity_idr=portfolio_equity_input, cash_on_hand_idr=float(cash_on_hand), config=cfg,
    )

    intraday_histories: dict[str, pd.DataFrame] = {}
    intraday_report = None
    if enable_intraday_specialty and specialty_shortlist:
        progress.progress(93, text=f"Mengunduh intraday 5m untuk {len(specialty_shortlist)} kandidat specialty…")
        intraday_histories, intraday_report = cached_intraday_data(tuple(specialty_shortlist), "5d", "5m", bool(itick_api_token), itick_api_token)
    progress.progress(96, text="Membangun Sniper, BSJP/BPJS, Multibagger, dan ARA Hunter…")
    specialty_screens = build_specialty_screens(
        result["prepared"],
        fundamentals=fundamentals,
        core_signals=specialty_signal_context,
        project_management=combined_project_management,
        market_context=result.get("market_context"),
        intraday=intraday_histories,
        config=cfg,
        now=now_jkt_ui,
        current_positions=current_positions,
        current_open_risk_idr=current_open_risk,
        cash_on_hand_idr=float(cash_on_hand),
    )
    specialty_screens["sniper"] = enrich_swing_specialty_with_time_cycle(
        specialty_screens.get("sniper", pd.DataFrame()), result["prepared"],
        enabled=bool(cfg.time_cycle_enabled), max_weight=float(cfg.time_cycle_core_max_weight),
        min_confidence=float(cfg.time_cycle_min_confidence),
        config=TimeCycleConfig(
            min_bars=int(cfg.time_cycle_min_history_bars),
            lunar_enabled=bool(cfg.time_cycle_lunar_enabled),
            eoff_enabled=bool(cfg.eoff_enabled),
            eoff_ephemeris_enabled=bool(cfg.eoff_ephemeris_enabled),
            eoff_min_fib_cluster=int(cfg.eoff_min_fib_cluster),
            eoff_aspect_orb_deg=float(cfg.eoff_aspect_orb_deg),
            eoff_require_astro_fib_confluence=bool(cfg.eoff_require_astro_fib_confluence),
        ),
    )
    specialty_screens["ara_hunter"] = apply_ara_external_confirmation(
        specialty_screens.get("ara_hunter", pd.DataFrame()),
        broker_summary=broksum,
        orderbook=orderbook,
        now=now_jkt_ui,
    )
    specialty_screens["daily_opportunities"] = build_daily_opportunity_board(specialty_screens)
    ai_cfg = LocalAIConfig(
        enabled=bool(cfg.ai_enabled), mode=str(cfg.ai_mode), max_weight=float(cfg.ai_max_weight),
        min_training_events=int(cfg.ai_min_training_events), min_strategy_events=int(cfg.ai_min_strategy_events),
        knn_k=int(cfg.ai_knn_k), memory_entry_window_bars=int(cfg.ai_memory_entry_window_bars),
        memory_horizon_bars=int(cfg.ai_memory_horizon_bars),
    )
    try:
        imported_memory = load_memory(ai_memory_uploaded)
    except Exception as exc:
        st.warning(f"AI memory tidak dapat diimpor; memakai cache lokal: {exc}")
        imported_memory = load_memory(None)
    # Resolve old signals before using them as training evidence. Empty ranking
    # means no new signals are registered in this first pass.
    ai_memory = update_outcome_memory(pd.DataFrame(), result["prepared"], imported_memory, ai_cfg)
    profit_order_builder = build_profit_order_builder(
        signals, specialty_screens, cfg, validation_events=trades,
        ai_memory=resolved_memory_events(ai_memory),
    )
    ai_memory = update_outcome_memory(profit_order_builder, result["prepared"], ai_memory, ai_cfg)
    specialty_screens["profit_order_builder"] = profit_order_builder
    specialty_screens["profit_strategy_audit"] = profit_order_builder.attrs.get("strategy_audit", pd.DataFrame())
    specialty_screens["ai_model_audit"] = profit_order_builder.attrs.get("ai_audit", pd.DataFrame())
    specialty_screens["ai_outcome_memory"] = ai_memory
    specialty_screens["ai_memory_summary"] = memory_summary(ai_memory)

    source_quorum_audit = build_source_quorum_audit(
        all_tickers,
        source_tiers=source_tier_map,
        price_validation=price_validation,
        fundamental_history=fundamental_history,
        market_status=market_status,
        news_review=news_review,
        validation_stats=stats,
        intraday_report=intraday_report,
        broker_summary=broksum,
        orderbook=orderbook,
        config=cfg,
    )

    result.update({
        "mode": "scanner", "signals": signals, "validation_stats": stats,
        "validation_trades": trades, "ai_outcome_memory": ai_memory,
        "ai_model_audit": specialty_screens.get("ai_model_audit", pd.DataFrame()),
        "fundamentals": fundamentals,
        "fundamental_history": fundamental_history,
        "fundamental_history_report": fundamental_history_report,
        "source_quorum_audit": source_quorum_audit,
        "market_status": market_status, "news_review": news_review,
        "broker_summary": broksum, "orderbook_snapshot": orderbook, "execution_snapshots": snapshots,
        "independent_price_data": independent_price_data,
        "price_validation": price_validation,
        "independent_provider_report": automatic_price_report,
        "twelve_data_report": (
            automatic_price_report.loc[automatic_price_report.get("provider", pd.Series(dtype=str)).eq("TWELVE_DATA")].copy()
            if not automatic_price_report.empty else pd.DataFrame()
        ),
        "download_report": report, "portfolio": portfolio,
        "portfolio_analysis": portfolio_analysis, "portfolio_summary": portfolio_summary,
        "all_histories": histories,
        "specialty_screens": specialty_screens,
        "intraday_histories": intraday_histories,
        "intraday_report": intraday_report,
        "realized_profit_to_compound_idr": realized_profit_to_compound,
        "compounding_budget_idr": effective_multibagger_budget,
        "project_management_review": combined_project_management,
        "automatic_forward_report": automatic_forward_report,
    })
    st.session_state["scan_result"] = result
    st.session_state["_last_auto_scan_signature"] = scan_signature
    progress.progress(100, text="Scan dan portfolio review selesai")
    progress.empty()

if "scan_result" not in st.session_state:
    st.markdown(
        """
        <div class="scanner-note">
          <b>Alur otomatis v6.6.7 Fundamental Resilience Batch 2</b><br>
          1) Tab 1 dan Tab 3 dapat membedah ticker tanpa CSV.<br>
          2) Upload universe ticker untuk membangun Top 20 ticker unik lintas pasar dan Dashboard Lengkap.<br>
          3) Portfolio Stockbit bersifat opsional; unggah hanya bila ingin cash/heat/posisi dihitung dari snapshot nyata.
        </div>
        """,
        unsafe_allow_html=True,
    )
    result = ScanEngine(cfg).scan({}, None)
    result.update({
        "mode": "scanner",
        "signals": pd.DataFrame(columns=["ticker", "status", "setup", "quality_score"]),
        "universe": pd.DataFrame(),
        "prepared": {},
        "specialty_screens": {},
        "all_histories": {},
        "validation_stats": pd.DataFrame(),
        "validation_trades": pd.DataFrame(),
        "download_report": None,
    })
else:
    result = st.session_state["scan_result"]
if result.get("mode") == "portfolio":
    st.success("Portfolio-only review selesai. Scanner universe dan backtest tidak dijalankan.")
    render_portfolio_panel(result)
    report = result.get("download_report")
    if report is not None and getattr(report, "failed", None):
        with st.expander(f"Ticker portfolio gagal diunduh ({len(report.failed)})"):
            st.dataframe(pd.DataFrame(report.failed.items(), columns=["ticker", "error"]), hide_index=True)
    st.stop()

signals: pd.DataFrame = result["signals"]
universe: pd.DataFrame = result["universe"]
context = result["market_context"]
report = result.get("download_report")

if context.regime == "RISK_ON":
    st.success(f"Regime: {context.regime} — {context.reason}")
elif context.regime == "RISK_OFF":
    st.error(f"Regime: {context.regime} — {context.reason}")
else:
    st.warning(f"Regime: {context.regime} — {context.reason}")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Ticker valid", len(result["prepared"]))
m2.metric("Setup rows", len(signals))
m3.metric("Trigger confirmed", int(signals.loc[signals.get("analyst_order_mode", pd.Series('', index=signals.index)).eq("TRIGGER_CONFIRMED"), "ticker"].nunique()) if not signals.empty else 0)
m4.metric("Entry plans", int(signals.loc[signals["status"].eq("ENTRY_PLAN_READY"), "ticker"].nunique()) if not signals.empty else 0)
m5.metric("Core plan verified", int(signals.loc[signals.get("autopilot_verified", pd.Series(False, index=signals.index)).fillna(False).astype(bool), "ticker"].nunique()) if not signals.empty else 0)
m6, m7, m8, m9, m10 = st.columns(5)
m6.metric("Account gate", "ON" if execution_policy == "ACCOUNT_GUARDED" else "USER MANAGED")
m7.metric("Menunggu close", int(signals.loc[signals["status"].eq("PENDING_CLOSE"), "ticker"].nunique()) if not signals.empty else 0)
m8.metric("Pending data", int(signals.loc[signals["status"].eq("PENDING_DATA"), "ticker"].nunique()) if not signals.empty else 0)
m9.metric("Watchlist", int(signals.loc[signals["status"].eq("WATCHLIST_ENTRY"), "ticker"].nunique()) if not signals.empty else 0)
m10.metric("Breadth > EMA50", f"{context.breadth_ema50:.0f}%" if context.breadth_ema50 is not None else "N/A")

main_best, main_existing, main_time_cycle = st.tabs(
    [
        "1 · Saham Terbaik",
        "2 · Dashboard Lengkap",
        "3 · Time-Cycle",
    ]
)

with main_best:
    render_top20_dashboard(
        result,
        lambda ticker, lookback: run_single_ticker_deep_dive(
            ticker,
            lookback,
            cfg,
            twelve_data_api_key=twelve_data_api_key,
            itick_api_token=itick_api_token,
        ),
    )

with main_existing:
    tab_multibagger, tab_orders, tab_daily, tab_setups, tab_sniper, tab_fast, tab_ara, tab_portfolio, tab_chart, tab_bridge, tab_validation, tab_audit, tab_method = st.tabs(
        [
            "Multibagger Capital", "Order Builder", "Daily Radar", "Core Setups", "Sniper Entry",
            "BSJP / BPJS", "ARA Hunter", "Portfolio Stockbit", "Chart",
            "TradingView / Stockbit", "Validation", "Audit Universe", "Metodologi",
        ]
    )

    specialty_screens: dict[str, pd.DataFrame] = result.get("specialty_screens", {})

    with tab_orders:
        profit_orders = specialty_screens.get("profit_order_builder", pd.DataFrame())
        st.subheader("Profit Conviction Ranking")
        st.caption(
            "Urutan lintas seluruh mesin profit. Rule conviction digabung dengan AI lokal yang belajar dari chronological walk-forward "
            "dan outcome memory. Time-cycle/EOFF clean-room ikut memengaruhi daily core/swing hanya setelah confluence dan evidence lolos. "
            "Bobot AI otomatis diturunkan saat sampel lemah, kalibrasi buruk, atau terjadi feature drift; bukan jaminan profit."
        )
        if profit_orders.empty:
            st.info("Belum ada setup profit-engine yang melewati minimum conviction.")
        else:
            p1, p2, p3, p4, p5, p6 = st.columns(6)
            p1.metric("Kandidat terurut", len(profit_orders))
            top_rule = float(profit_orders.iloc[0].get('profit_conviction_score', 0.0))
            top_hybrid = float(profit_orders.iloc[0].get('hybrid_conviction_score', top_rule))
            p2.metric("Hybrid conviction", f"{top_hybrid:.1f}", delta=f"Rule {top_rule:.1f}")
            p3.metric("P(fill × TP1)", f"{float(profit_orders.iloc[0].get('ai_trade_success_probability_pct', np.nan)):.1f}%" if pd.notna(profit_orders.iloc[0].get('ai_trade_success_probability_pct')) else "N/A")
            p4.metric("Bobot AI aktual", f"{float(profit_orders.iloc[0].get('ai_effective_weight_pct', 0.0)):.1f}%")
            p5.metric("Prioritas #1", str(profit_orders.iloc[0].get('ticker', '-')))
            p6.metric("Strategi #1", str(profit_orders.iloc[0].get('strategy', '-')))
            if not bool(profit_orders.iloc[0].get('ai_can_influence_ranking', False)):
                st.info("AI masih shadow-by-evidence untuk kandidat teratas; hybrid conviction tetap sama dengan rule score. Lihat AI gate reasons pada tabel.")
            st.dataframe(profit_orders, hide_index=True, width="stretch", height=520)
            st.download_button(
                "Download Profit Conviction Order Builder",
                profit_orders.to_csv(index=False).encode("utf-8"),
                "profit_conviction_order_builder.csv", "text/csv", width="stretch",
            )
            ai_memory_view = specialty_screens.get("ai_outcome_memory", pd.DataFrame())
            ai_audit_view = specialty_screens.get("ai_model_audit", pd.DataFrame())
            ai_summary_view = specialty_screens.get("ai_memory_summary", pd.DataFrame())
            a1, a2 = st.columns(2)
            with a1:
                if not ai_memory_view.empty:
                    st.download_button(
                        "Export AI outcome memory", ai_memory_view.to_csv(index=False).encode("utf-8"),
                        "idx_scanner_ai_outcome_memory.csv", "text/csv", width="stretch",
                    )
            with a2:
                st.caption(f"AI runtime: {AI_VERSION}")
            with st.expander("Audit AI lokal & pembelajaran", expanded=False):
                if not ai_summary_view.empty:
                    st.dataframe(ai_summary_view, hide_index=True, width="stretch")
                if not ai_audit_view.empty:
                    st.dataframe(ai_audit_view, hide_index=True, width="stretch")
                st.info(
                    "Disk Streamlit Community Cloud tidak dijamin persisten setelah sleep/redeploy. "
                    "Export memory secara berkala lalu import kembali melalui sidebar."
                )
        strategy_audit = specialty_screens.get("profit_strategy_audit", pd.DataFrame())
        if not strategy_audit.empty:
            with st.expander("Mengapa strategi tertentu tidak muncul di ranking?", expanded=False):
                st.dataframe(strategy_audit, hide_index=True, width="stretch")
                if len(profit_orders) and profit_orders["strategy"].nunique() == 1:
                    only_strategy = str(profit_orders.iloc[0].get("strategy", "-"))
                    st.info(
                        f"Ranking saat ini hanya berisi {only_strategy} karena strategi lain tidak memiliki status eligible, "
                        "berada di bawah minimum conviction, kalah deduplikasi ticker, atau berada di luar batas Top-N. "
                        "Ini bukan pembatasan permanen terhadap strategi lain."
                    )
        st.divider()
        if signals.empty:
            st.warning("Tidak ada setup core yang valid. Audit Universe menjelaskan alasan per ticker.")
        else:
            execution = signals[signals.get("autopilot_verified", pd.Series(False, index=signals.index)).fillna(False).astype(bool)].sort_values(
                ["analyst_fusion_score", "quality_score"], ascending=False
            )
            manual_verify = signals[signals["status"].eq("READY_FOR_STOCKBIT_VERIFY")].sort_values(
                ["analyst_fusion_score", "quality_score"], ascending=False
            )
            signal_ready = signals[signals["status"].eq("SIGNAL_READY")].sort_values(
                ["analyst_fusion_score", "quality_score"], ascending=False
            )
            entry_plans = signals[signals["status"].eq("ENTRY_PLAN_READY")].sort_values(
                ["analyst_fusion_score", "quality_score"], ascending=False
            )
            confluence_alternates = signals[signals["status"].eq("READY_NOT_SELECTED")].sort_values(
                ["analyst_fusion_score", "quality_score"], ascending=False
            )
            pending_close = signals[signals["status"].eq("PENDING_CLOSE")]
            price_verify = signals[signals["status"].eq("READY_FOR_PRICE_VERIFY")]
            st.subheader("Order Builder — CORE PLAN VERIFIED")
            st.caption("Tabel ini boleh dijadikan template Stockbit, tetapi submit tetap manual setelah pemeriksaan harga, gap pembukaan, dan spread. BUY_LIT memakai trigger terpisah dari limit.")
            if execution.empty:
                if execution_policy == "SIGNAL_FIRST":
                    st.info("SIGNAL_FIRST memang tidak menerbitkan EXECUTION_READY. Gunakan tabel READY_FOR_STOCKBIT_VERIFY di bawah untuk kandidat manual yang telah lolos seluruh gate non-akun.")
                else:
                    st.info("Tidak ada order yang lolos seluruh Account Guard saat ini. Tidak melakukan transaksi adalah hasil yang valid.")
            else:
                result_table(execution)
            if not manual_verify.empty:
                st.subheader("READY_FOR_STOCKBIT_VERIFY — kandidat manual prioritas")
                st.warning("Seluruh gate otomatis non-akun telah lolos. Sebelum order: cocokkan last price, bid/offer, spread, gap, dan batas ARA/ARB di Stockbit; tentukan lot secara manual.")
                result_table(manual_verify)
            if not signal_ready.empty:
                st.subheader("SIGNAL_READY — radar, belum boleh dibeli")
                st.info("Setup dan trigger teknikal terdeteksi, tetapi masih ada gate data/risiko/konteks yang belum lolos. Lihat Risk warnings dan Signal blockers.")
                result_table(signal_ready)
            if not entry_plans.empty:
                st.subheader("ENTRY_PLAN_READY — tunggu harga/konfirmasi")
                st.info("Zona, SL, dan target sudah valid, tetapi order belum boleh dipasang otomatis sebelum harga masuk zona atau confirmation muncul.")
                result_table(entry_plans)
            if not confluence_alternates.empty:
                st.subheader("Confluence alternate — setup kedua pada ticker yang sama")
                st.info("Setup tetap ditampilkan, tetapi tidak membuat order kedua agar risiko saham yang sama tidak terhitung dua kali.")
                result_table(confluence_alternates)
            if not pending_close.empty:
                st.subheader("PENDING_CLOSE")
                st.info("Setup teknikal sudah terbentuk dari completed EOD terakhir. Refresh setelah 16:20 WIB; belum boleh disalin sebagai order.")
                result_table(pending_close)
            if not price_verify.empty:
                st.subheader("READY_FOR_PRICE_VERIFY")
                st.warning(
                    "Harga otomatis kedua belum tervalidasi pada mode Account-Guarded. "
                    "Last price dan bid/offer wajib dicocokkan sebelum kandidat dapat dinaikkan."
                )
                result_table(price_verify)
            st.subheader("Execution funnel")
            st.dataframe(execution_funnel_summary(signals), hide_index=True, width="stretch")
            blocker_series = signals.get("primary_execution_blocker", pd.Series(dtype=str)).replace("NONE", np.nan).dropna()
            if not blocker_series.empty:
                top_blockers = blocker_series.value_counts().rename_axis("Penyebab utama").reset_index(name="Jumlah")
                st.dataframe(top_blockers, hide_index=True, width="stretch")
            technical_details = (
                signals.get("blockers", pd.Series(dtype=str)).dropna().astype(str)
                .str.split(" • ").explode().str.strip().dropna()
            )
            technical_details = technical_details[technical_details.ne("")]
            if not technical_details.empty:
                st.caption("Rincian blocker detector/tradeability paling sering")
                st.dataframe(
                    technical_details.value_counts().head(12).rename_axis("Blocker").reset_index(name="Jumlah"),
                    hide_index=True, width="stretch",
                )
            c1, c2, c3 = st.columns(3)
            c1.download_button(
                "Download semua hasil CSV",
                signals.drop(columns=["status_rank"], errors="ignore").to_csv(index=False).encode("utf-8"),
                "idx_super_scanner_results.csv", "text/csv", width="stretch",
            )
            c2.download_button(
                "Download ready-for-Stockbit CSV",
                manual_verify.drop(columns=["status_rank"], errors="ignore").to_csv(index=False).encode("utf-8"),
                "idx_ready_for_stockbit_verify.csv", "text/csv", width="stretch",
                disabled=manual_verify.empty,
            )
            c3.download_button(
                "Download execution-ready CSV",
                execution.drop(columns=["status_rank"], errors="ignore").to_csv(index=False).encode("utf-8"),
                "idx_execution_ready.csv", "text/csv", width="stretch",
                disabled=execution.empty,
            )

    with tab_daily:
        daily_board = specialty_screens.get("daily_opportunities", pd.DataFrame())
        st.subheader("Daily Opportunity Board")
        st.caption("SETUP_READY/SIGNAL_READY berarti struktur valid. ORDER_READY hanya tersedia pada mode ACCOUNT_GUARDED; pada SIGNAL_FIRST ukuran order dikelola manual.")
        if daily_board.empty:
            st.info("Universe belum menghasilkan kandidat yang dapat diranking.")
        else:
            order_count = int(daily_board.get("order_ready", pd.Series(False, index=daily_board.index)).fillna(False).astype(bool).sum())
            d1, d2, d3 = st.columns(3)
            d1.metric("Kandidat harian", len(daily_board))
            d2.metric("ORDER_READY", order_count)
            d3.metric("Cash compounding", rupiah(result.get("compounding_budget_idr", 0.0)))
            st.dataframe(daily_board, hide_index=True, width="stretch")
            st.download_button("Download Daily Radar", daily_board.to_csv(index=False).encode("utf-8"), "daily_opportunity_board.csv", "text/csv", width="stretch")

    with tab_setups:
        st.caption("Setiap detector dipisahkan agar shortlist, blocker, dan level order tidak tercampur.")
        setup_specs = [
            ("Pullback Continuation", "PULLBACK_CONTINUATION", "pullback_continuation.csv"),
            ("Breakout Retest", "BREAKOUT_RETEST", "breakout_retest.csv"),
            ("Reversal Accumulation", "REVERSAL_ACCUMULATION", "reversal_accumulation.csv"),
            ("Unicorn / ICT", "UNICORN_SNIPER_ICT", "unicorn_ict.csv"),
        ]
        setup_tabs = st.tabs([item[0] for item in setup_specs])
        for setup_tab, (label, setup_code, filename) in zip(setup_tabs, setup_specs):
            with setup_tab:
                subset = signals[signals["setup"].eq(setup_code)].copy() if not signals.empty else pd.DataFrame()
                signal_count = int(subset["status"].eq("SIGNAL_READY").sum()) if not subset.empty else 0
                verify_count = int(subset["status"].eq("READY_FOR_STOCKBIT_VERIFY").sum()) if not subset.empty else 0
                ready_count = int(subset.get("autopilot_verified", pd.Series(False, index=subset.index)).fillna(False).astype(bool).sum()) if not subset.empty else 0
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Kandidat", len(subset))
                c2.metric("Signal ready", signal_count)
                c3.metric("Verify Stockbit", verify_count)
                c4.metric("Execution verified", ready_count)
                if subset.empty:
                    st.info(f"Tidak ada kandidat {label}.")
                else:
                    result_table(subset)
                    st.download_button(
                        f"Download {label}",
                        subset.drop(columns=["status_rank"], errors="ignore").to_csv(index=False).encode("utf-8"),
                        filename, "text/csv", width="stretch",
                    )

    with tab_sniper:
        sniper = specialty_screens.get("sniper", pd.DataFrame())
        st.subheader("ICT Sniper Entry — Calibrated")
        st.caption("Sweep–BOS–FVG dinilai sebagai signal layer. Tiket order tetap harus berasal dari core yang melewati account-risk gate.")
        specialty_table(sniper)
        render_specialty_download("Download Sniper Entry", sniper, "strict_sniper_entries.csv")

    with tab_fast:
        fast_bsjp, fast_bpjs = st.tabs(["BSJP — Beli Sore Jual Pagi", "BPJS — Beli Pagi Jual Sore"])
        with fast_bsjp:
            bsjp = specialty_screens.get("bsjp", pd.DataFrame())
            st.warning("BSJP_SIGNAL_READY berarti setup valid. Risiko, regime, RR, stop, dan kondisi akun ditampilkan sebagai warning; BSJP_ORDER_READY hanya dipakai pada mode ACCOUNT_GUARDED pada 14:30–15:49 WIB.")
            specialty_table(bsjp)
            render_specialty_download("Download BSJP", bsjp, "bsjp_candidates.csv")
        with fast_bpjs:
            bpjs = specialty_screens.get("bpjs", pd.DataFrame())
            st.warning("BPJS_SIGNAL_READY memerlukan ORB/VWAP dan intraday fresh. Pada SIGNAL_FIRST, risiko dan akun tidak menggugurkan setup; BPJS_ORDER_READY hanya dipakai pada ACCOUNT_GUARDED. Posisi tetap wajib ditutup sebelum market close.")
            st.caption("STALE_INTRADAY berarti candle terakhir sudah terlalu lama/sesi lama. WAIT_OPENING_BARS berarti data masuk tetapi ORB belum memiliki candle konfirmasi.")
            specialty_table(bpjs)
            render_specialty_download("Download BPJS", bpjs, "bpjs_candidates.csv")
        intraday_report = result.get("intraday_report")
        if intraday_report is not None and getattr(intraday_report, "failed", None):
            with st.expander(f"Intraday provider gagal untuk {len(intraday_report.failed)} ticker"):
                st.dataframe(pd.DataFrame(intraday_report.failed.items(), columns=["ticker", "error"]), hide_index=True)
        if intraday_report is not None and getattr(intraday_report, "warnings", None):
            with st.expander(f"Intraday memakai fallback untuk {len(intraday_report.warnings)} ticker"):
                st.dataframe(pd.DataFrame(intraday_report.warnings.items(), columns=["ticker", "warning"]), hide_index=True)

    with tab_multibagger:
        multibagger = specialty_screens.get("multibagger", pd.DataFrame())
        st.subheader("Multibagger Capital Allocation")
        budget = float(result.get("compounding_budget_idr", 0.0) or 0.0)
        accumulate_now = int(multibagger.get("compounding_state", pd.Series(dtype=str)).isin(["ACCUMULATE_NOW", "STARTER_NOW"]).sum()) if not multibagger.empty else 0
        deployed = float(pd.to_numeric(multibagger.get("estimated_order_value_idr", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not multibagger.empty else 0.0
        reserve = max(0.0, budget - deployed)
        top_destination = "CASH_RESERVE"
        top_amount = 0.0
        if not multibagger.empty and "capital_priority_rank" in multibagger:
            ranked_capital = multibagger[pd.to_numeric(multibagger["capital_priority_rank"], errors="coerce").notna()].copy()
            if not ranked_capital.empty:
                ranked_capital = ranked_capital.sort_values("capital_priority_rank")
                top_destination = str(ranked_capital.iloc[0].get("ticker", "CASH_RESERVE"))
                top_amount = float(ranked_capital.iloc[0].get("strategic_target_amount_idr", 0.0) or 0.0)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Budget Multibagger", rupiah(budget))
        c2.metric("Deploy sekarang", rupiah(deployed))
        c3.metric("Cash reserve", rupiah(reserve))
        c4.metric("Prioritas #1", top_destination, rupiah(top_amount))
        st.caption(
            "Capital conviction menimbang growth, profitability, OCF/FCF quality, solvabilitas, valuasi, "
            "integritas data, momentum, smart-money proxy, kualitas pipeline proyek, rekam jejak manajemen, dan guarded EOFF timing. "
            "Project/management otomatis memprioritaskan IDX/OJK dan IR emiten; single-source atau modelled impact diberi confidence lebih rendah. Target weight adalah bobot strategis sleeve; "
            "Deploy now hanya aktif saat zona entry valid. Kandidat terbaik menerima porsi terbesar, tetapi cap per saham mencegah all-in tunggal."
        )
        if not multibagger.empty:
            allocation_columns = [
                column for column in [
                    "capital_priority_rank", "ticker", "capital_tier", "capital_conviction_score",
                    "multibagger_status", "compounding_state", "project_pipeline_score",
                    "management_quality_score", "forward_quality_coverage", "future_fundamental_impact_score",
                    "future_impact_confidence", "future_revenue_uplift_base_pct", "future_ebitda_uplift_base_pct",
                    "future_net_profit_uplift_base_pct", "future_fcf_pressure_idr", "future_net_debt_change_pct",
                    "multibagger_time_cycle_score", "time_cycle_capital_weight_pct", "quick_buy_action",
                    "best_buy_date", "best_buy_window_start", "best_buy_window_end", "best_buy_score",
                    "best_buy_confidence", "best_buy_entry_low", "best_buy_entry_high", "best_buy_trigger",
                    "best_buy_stop_loss", "best_buy_tp1", "best_buy_tp2", "eoff_strength_label",
                    "eoff_reconstruction_score", "eoff_signal_active", "eoff_direction_bias",
                    "eoff_fib_cluster_count", "eoff_reversal_date", "eoff_astro_score", "eoff_adaptive_total_weight_pct", "eoff_validation_path",
                    "project_source_families", "project_source_quorum_verified", "project_source_urls",
                    "management_source_urls", "ceo_name", "project_names",
                    "strategic_target_weight_pct", "deploy_now_weight_pct", "strategic_target_amount_idr",
                    "recommended_allocation_idr", "recommended_lots", "allocation_action", "allocation_reason", "red_flags",
                ] if column in multibagger.columns
            ]
            st.markdown("#### Capital destination ranking")
            st.dataframe(multibagger[allocation_columns].head(10), hide_index=True, width="stretch")
        forward_report = result.get("automatic_forward_report", pd.DataFrame())
        if isinstance(forward_report, pd.DataFrame) and not forward_report.empty:
            with st.expander("Audit pencarian forward intelligence otomatis", expanded=False):
                st.dataframe(forward_report, hide_index=True, width="stretch")
                st.caption("AUTO_VERIFIED berarti minimal dua keluarga sumber resmi berbeda. AUTO_SINGLE_SOURCE tetap dipakai dengan confidence lebih rendah.")
        st.markdown("#### Full Multibagger evidence")
        specialty_table(multibagger, height=560)
        render_specialty_download("Download Multibagger Capital Plan", multibagger, "multibagger_capital_plan.csv")

    with tab_ara:
        ara = specialty_screens.get("ara_hunter", pd.DataFrame())
        st.subheader("ARA Intelligence — Pre-ARA & Continuation")
        st.info(
            "PRE_ARA mencari saham yang belum ARA. ARA_CONTINUATION menilai saham yang sudah ARA/dekat ARA "
            "untuk peluang sesi berikutnya. Orderflow dan queue otomatis adalah proxy OHLCV, bukan data broker/order book aktual."
        )
        if not ara.empty and "ara_hunter_status" in ara:
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Pre-ARA order", int(ara["ara_hunter_status"].eq("PRE_ARA_ORDER_READY").sum()))
            a2.metric(
                "Signal ready",
                int(ara["ara_hunter_status"].isin(["PRE_ARA_SIGNAL_READY", "ARA_CONTINUATION_SIGNAL_READY", "ARA_CONTINUATION_FLOW_VERIFIED_SIGNAL"]).sum()),
            )
            a3.metric("Pre-ARA candidate", int(ara["ara_hunter_status"].eq("PRE_ARA_CANDIDATE").sum()))
            a4.metric("Daily radar", int(ara["ara_hunter_status"].eq("PRE_ARA_DAILY_RADAR").sum()))
        specialty_table(ara, height=540)
        render_specialty_download("Download ARA Intelligence", ara, "ara_intelligence.csv")


    with tab_portfolio:
        render_portfolio_panel(result)

    with tab_chart:
        @st.fragment
        def render_signal_chart_fragment() -> None:
            if signals.empty:
                st.info("Belum ada setup untuk digambar.")
                return
            labels = [f"{row.ticker} · {row.setup}" for row in signals.itertuples()]
            key = "chart_setup_selector_v460"
            prior = st.session_state.get(key)
            if prior not in labels:
                st.session_state[key] = labels[0]
            selected = st.selectbox("Pilih setup", labels, key=key)
            selected_pos = labels.index(selected)
            signal = signals.iloc[selected_pos].to_dict()
            frame = result["prepared"][signal["ticker"]]
            st.plotly_chart(
                make_signal_chart(frame, signal),
                width="stretch",
                key=f"signal_chart_{signal['ticker']}_{signal['setup']}",
            )
            st.write("Evidence:", signal.get("evidence", "—"))
            if signal.get("signal_risk_warnings"):
                st.warning("Risk disclosure: " + str(signal["signal_risk_warnings"]))
            if signal.get("blockers"):
                st.caption("Detector notes: " + str(signal["blockers"]))
        render_signal_chart_fragment()


    with tab_bridge:
        st.subheader("TradingView confirmation bridge")
        st.caption(
            "Pilih kandidat dari tabel, buka simbol pada TradingView, lalu salin entry zone, order entry, SL, TP1, dan TP2 "
            "ke mode MANUAL SCANNER LEVELS. Pine Script tidak membaca dataframe Streamlit atau Stockbit secara langsung."
        )
        bridge = build_tradingview_bridge(signals, specialty_screens)
        if bridge.empty:
            st.info("Belum ada level scanner yang dapat diekspor.")
        else:
            st.dataframe(
                bridge,
                hide_index=True,
                width="stretch",
                column_config={
                    "ticker": st.column_config.TextColumn("Ticker", pinned=True),
                    "tv_symbol": st.column_config.TextColumn("TradingView"),
                    "entry_low": st.column_config.NumberColumn("Entry low", format="Rp %.0f"),
                    "entry_high": st.column_config.NumberColumn("Entry high", format="Rp %.0f"),
                    "entry": st.column_config.NumberColumn("Order entry", format="Rp %.0f"),
                    "stop_loss": st.column_config.NumberColumn("SL", format="Rp %.0f"),
                    "tp1": st.column_config.NumberColumn("TP1", format="Rp %.0f"),
                    "tp2": st.column_config.NumberColumn("TP2", format="Rp %.0f"),
                    "rr1": st.column_config.NumberColumn("RR1", format="%.2f"),
                    "rr2": st.column_config.NumberColumn("RR2", format="%.2f"),
                },
            )
            st.download_button(
                "Download TradingView bridge CSV",
                bridge.to_csv(index=False).encode("utf-8"),
                "tradingview_scanner_bridge.csv",
                "text/csv",
                width="stretch",
            )
        pine_file = APP_ROOT / "IDX_Scanner_Confirmation_v1.pine"
        stockbit_file = APP_ROOT / "STOCKBIT_SCREENER_PRESETS.md"
        c1, c2 = st.columns(2)
        if pine_file.is_file():
            c1.download_button(
                "Download Pine Script indicator",
                pine_file.read_bytes(),
                pine_file.name,
                "text/plain",
                width="stretch",
            )
        else:
            c1.warning("File Pine Script tidak ditemukan dalam deployment.")
        if stockbit_file.is_file():
            c2.download_button(
                "Download preset screener Stockbit",
                stockbit_file.read_bytes(),
                stockbit_file.name,
                "text/markdown",
                width="stretch",
            )
        else:
            c2.warning("Panduan screener Stockbit tidak ditemukan dalam deployment.")
        st.markdown(
            """
            **Urutan eksekusi:** Stockbit menyaring universe → scanner menilai setup, data, dan risiko → TradingView mengonfirmasi struktur/bar penutup → order tetap dimasukkan manual sebagai limit order di Stockbit.

            **Peringatan:** directional score pada indikator adalah skor konfluensi berbasis aturan, bukan probabilitas statistik dan bukan prediksi yang dijamin.
            """
        )

    with tab_validation:
        stats: pd.DataFrame = result.get("validation_stats", pd.DataFrame())
        trades: pd.DataFrame = result.get("validation_trades", pd.DataFrame())
        if stats.empty:
            st.info("Aktifkan chronological OOS validation sebelum menjalankan scan.")
        else:
            st.subheader("Chronological out-of-sample validation")
            st.caption("Statistik ini merupakan chronological holdout pada level setup, bukan probabilitas khusus ticker. Detector dan structural levels dibangun ulang pada setiap tanggal kandidat; entry harus tersentuh dalam 5 bar, gap yang merusak RR dibatalkan, biaya/slippage masuk, dan bar ambigu dihitung konservatif sebagai SL lebih dulu.")
            st.dataframe(stats, width="stretch", hide_index=True)
            st.download_button(
                "Download seluruh trade historis",
                trades.to_csv(index=False).encode("utf-8"),
                "walkforward_trades.csv",
                "text/csv",
            )

    with tab_audit:
        st.subheader("Audit coverage dan sumber data")
        requested_count = len(getattr(report, "requested", []) or []) if report is not None else 0
        downloaded_count = len(getattr(report, "downloaded", []) or []) if report is not None else 0
        download_coverage = 100.0 * downloaded_count / requested_count if requested_count else 0.0
        fund_audit = result.get("fundamentals", pd.DataFrame())
        fundamental_history_audit = result.get("fundamental_history", pd.DataFrame())
        fundamental_history_report = result.get("fundamental_history_report", pd.DataFrame())
        status_audit = result.get("market_status", pd.DataFrame())
        news_audit = result.get("news_review", pd.DataFrame())
        quote_audit = result.get("execution_snapshots", pd.DataFrame())
        independent_audit = result.get("price_validation", pd.DataFrame())
        provider_audit = result.get("independent_provider_report", pd.DataFrame())
        source_quorum_audit = result.get("source_quorum_audit", pd.DataFrame())
        fund_series = fund_audit.get("fundamental_coverage", pd.Series(index=fund_audit.index, dtype=float))
        fund_ok = int(pd.to_numeric(fund_series, errors="coerce").ge(45).sum()) if not fund_audit.empty else 0
        status_ok = int(status_audit.get("market_status_verified", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not status_audit.empty else 0
        news_ok = int(news_audit.get("provider_query_ok", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not news_audit.empty else 0
        quote_ok = int(quote_audit.get("quote_verified", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not quote_audit.empty else 0
        independent_ok = int(independent_audit.get("independent_price_verified", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not independent_audit.empty else 0
        provider_ok = int(provider_audit.get("status", pd.Series(dtype=str)).eq("OK").sum()) if not provider_audit.empty else 0
        a1, a2, a3, a4, a5, a6, a7 = st.columns(7)
        a1.metric("OHLCV coverage", f"{download_coverage:.1f}%")
        a2.metric("Fundamental ≥45%", fund_ok)
        a3.metric("Status IDX verified", status_ok)
        a4.metric("News query OK", news_ok)
        a5.metric("Quote verified", quote_ok)
        a6.metric("Harga independen", independent_ok)
        a7.metric("Auto provider OK", provider_ok)
        grade_series = fund_audit.get("fundamental_data_grade", pd.Series(index=fund_audit.index, dtype=str)).astype(str)
        score10_series = pd.to_numeric(fund_audit.get("fundamental_score_10", pd.Series(index=fund_audit.index, dtype=float)), errors="coerce")
        sources_series = pd.to_numeric(fund_audit.get("fundamental_source_count", pd.Series(index=fund_audit.index, dtype=float)), errors="coerce")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Fundamental data grade A", int(grade_series.eq("A").sum()))
        f2.metric("Fundamental data grade B", int(grade_series.eq("B").sum()))
        f3.metric("Fundamental score ≥8/10", int(score10_series.ge(8.0).sum()))
        f4.metric("Multi-source history", int(sources_series.ge(2).sum()))
        source_tiers = getattr(report, "source_tiers", {}) or {} if report is not None else {}
        live_yahoo_ohlcv = sum(str(tier).startswith("LIVE_YAHOO") for tier in source_tiers.values())
        live_idx_ohlcv = sum(str(tier).startswith("LIVE_IDX") for tier in source_tiers.values())
        live_itick_ohlcv = sum(str(tier).startswith("LIVE_ITICK") for tier in source_tiers.values())
        fresh_cache_ohlcv = sum(str(tier).startswith("CACHE_FRESH_VERIFIED") for tier in source_tiers.values())
        stale_cache_ohlcv = sum(str(tier) == "CACHE_FALLBACK" for tier in source_tiers.values())
        unavailable_ohlcv = sum(str(tier) == "UNAVAILABLE" for tier in source_tiers.values())
        st.caption(
            f"Tier OHLCV: {live_yahoo_ohlcv} Yahoo · {live_idx_ohlcv} IDX patch · "
            f"{live_itick_ohlcv} iTick free · {fresh_cache_ohlcv} cache current · "
            f"{stale_cache_ohlcv} cache stale · {unavailable_ohlcv} unavailable."
        )
        st.subheader("Source quorum per lapisan")
        st.caption(
            "Fallback tidak sama dengan verifikasi simultan. TWO_SOURCE hanya diberikan bila dua keluarga sumber benar-benar "
            "hadir; data regulator dan data akun memakai kebijakan authority-first. Kandidat tetap ditampilkan saat quorum parsial."
        )
        if not source_quorum_audit.empty:
            st.dataframe(source_quorum_audit, hide_index=True, width="stretch")
        st.dataframe(
            pd.DataFrame(
                [
                    ["Verified local cache", "Menghindari unduh ulang full-history ketika EOD sudah current", "Automatic cache-first"],
                    ["Yahoo Finance via yfinance", "Primary OHLCV daily/intraday, IHSG, fundamental", "Automatic bounded batch"],
                    ["Yahoo statement history", "Quarterly/annual revenue, earnings, cash flow, balance sheet, dilution", "Automatic bounded Multibagger shortlist"],
                    ["IDX official XBRL/iXBRL", "First-party quarterly/annual statement history", "Automatic, no key; bounded + cached + fail-soft"],
                    ["Manual IDX/XBRL CSV", "Fallback bila filing otomatis tidak dapat diparse", "Optional; provenance/accounting checks"],
                    ["Twelve Data fundamentals", "Third-source income statement, balance sheet, and cash flow consensus", "Optional eligible API plan; bounded shortlist"],
                    ["IDX Stock Summary API", "Menambal bar EOD terakhir saat Yahoo gagal dan cache historis tersedia", "Automatic official fallback"],
                    ["iTick free tier", "Fallback OHLCV daily/intraday untuk ticker gagal", "Optional token; ≤4 call/min internal"],
                    ["Official IDX public pages", "Daftar saham, FCA/pemantauan, suspensi, aksi korporasi, disclosure", "Automated + cache"],
                    ["Google News RSS", "Berita luas per ticker", "Automated"],
                    ["IDX Stock Summary API", "Cross-check harga EOD resmi kandidat", "Automatic primary verification"],
                    ["Google Finance public quote", "Fallback harga independen kandidat", "Automatic bounded shortlist"],
                    ["Persistent evidence cache", "Mempertahankan bukti harga provider yang masih sama-session", "Automatic; tidak mengubah identitas sumber"],
                    ["iTick free tier", "Fallback harga independen untuk shortlist yang belum terpecahkan", "Optional token; bounded"],
                    ["Twelve Data XIDX EOD", "Fallback recent close dan return path", "Automatic bila deployment secret tersedia"],
                    ["OHLCV flow model", "Proxy akumulasi/distribusi tanpa broker-summary upload", "Automatic; bukan beneficial-owner data"],
                ],
                columns=["Provider family", "Dipakai untuk", "Mode"],
            ),
            hide_index=True,
            width="stretch",
        )
        if independent_ok:
            st.success(f"{independent_ok} ticker mempunyai cross-validation harga dari keluarga sumber independen.")
        else:
            st.warning(
                "Belum ada harga independen yang terverifikasi. Setup tetap terlihat sebagai SIGNAL_READY, "
                "tetapi tidak menjadi kandidat manual atau tiket Stockbit."
            )
        if not independent_audit.empty:
            st.dataframe(independent_audit, hide_index=True, width="stretch")
        if not provider_audit.empty:
            with st.expander("Audit provider harga otomatis", expanded=bool(provider_audit["status"].ne("OK").any())):
                st.dataframe(provider_audit, hide_index=True, width="stretch")
        if not fundamental_history_report.empty:
            with st.expander(
                "Audit provider fundamental historis",
                expanded=bool(fundamental_history_report.get("status", pd.Series(dtype=str)).ne("OK").any()),
            ):
                st.dataframe(fundamental_history_report, hide_index=True, width="stretch")
        if not fundamental_history_audit.empty:
            st.caption(
                f"Statement history: {len(fundamental_history_audit)} baris · "
                f"{fundamental_history_audit['ticker'].nunique()} emiten · "
                f"{fundamental_history_audit['source_family'].nunique()} keluarga sumber."
            )
        st.subheader("Execution gate diagnostics")
        st.dataframe(execution_funnel_summary(signals), hide_index=True, width="stretch")
        st.subheader("Satu baris untuk setiap ticker")
        st.dataframe(universe, width="stretch", hide_index=True)
        if report is not None and report.failed:
            with st.expander(f"Ticker gagal diunduh ({len(report.failed)})"):
                st.dataframe(pd.DataFrame(report.failed.items(), columns=["ticker", "error"]), hide_index=True)
        if report is not None and report.warnings:
            with st.expander(f"Peringatan kualitas OHLCV ({len(report.warnings)})"):
                st.dataframe(pd.DataFrame(report.warnings.items(), columns=["ticker", "warning"]), hide_index=True)

    with tab_method:
        st.markdown(
            """
            ### Hirarki sinyal v6.6.7 Best Buy & EOFF Top-20

            1. **Profit Conviction Ranking:** Order Builder menggabungkan Pullback Continuation, Sniper, BPJS, BSJP, PRE-ARA, dan ARA Continuation. Satu ticker hanya memiliki satu rencana utama; strategi alternatif tetap dicatat.
            2. **Validitas struktur:** setup, zona entry, trigger/retest, masa berlaku, dan level harga harus masih valid.
            3. **Data minimum:** candle EOD harus final, OHLCV tidak boleh unavailable atau terlalu lama, dan suspensi/FCA tetap memblokir.
            4. **Kandidat tetap lengkap:** satu ticker boleh muncul pada beberapa setup dan seluruhnya tetap ditampilkan.
            5. **Dua kebijakan eksekusi:** default `SIGNAL_FIRST` menerbitkan `SIGNAL_READY` untuk radar dan `READY_FOR_STOCKBIT_VERIFY` hanya setelah seluruh gate non-akun lolos. `ACCOUNT_GUARDED` menambahkan sizing, cash, slot posisi, dan portfolio heat sebelum `EXECUTION_READY`.
            6. **Core setup:** Pullback dapat memakai `LIMIT_PULLBACK_ZONE`; Breakout memakai limit setelah retest/reclaim; Reversal membutuhkan CHOCH/BOS; trigger langsung memakai `TRIGGER_CONFIRMED`.
            7. **ARA Intelligence:** `DAILY_RADAR` selalu meranking kandidat; `SIGNAL_READY` berarti pola terkonfirmasi; `ORDER_READY` merupakan lapisan opsional ACCOUNT_GUARDED. Default SIGNAL_FIRST tidak menyembunyikan setup karena kondisi akun.
            8. **Verifikasi harga:** IDX EOD tanggal referensi → Google Finance → iTick opsional → Twelve Data opsional, dengan cache bukti provider yang masih sama-session.
            9. **Sesi perdagangan:** pre-market memakai EOD sebelumnya; sesi aktif menghasilkan `PENDING_CLOSE`; post-close memakai candle final hari tersebut.
            10. **Sizing:** pada `SIGNAL_FIRST`, lot bersifat informasional dan `SIGNAL_READY` tidak membawa tiket beli. `READY_FOR_STOCKBIT_VERIFY` membawa usulan harga tetapi wajib revalidasi broker. Pada `ACCOUNT_GUARDED`, order yang lolos membawa lot berbasis risiko akun.
            11. **Chart state:** pilihan ticker chart memakai fragment sehingga tidak mereset dashboard atau kembali ke tab pertama.

            `SIGNAL_READY` bukan instruksi beli. `READY_FOR_STOCKBIT_VERIFY` adalah kandidat manual, sedangkan `EXECUTION_READY` selalu mensyaratkan `autopilot_verified = True`; seluruh submit tetap dilakukan manual di Stockbit.
            """
        )


with main_time_cycle:
    render_time_cycle_main_tab(
        cfg,
        lambda tickers, lookback: download_ohlcv(
            tickers, period=lookback, itick_api_token=itick_api_token,
        ),
    )
