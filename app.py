from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="IDX Super Scanner ARA Intelligence v4.6.0", page_icon="🎯", layout="wide")

# Streamlit runs the selected entrypoint from its deployment workspace. Keep
# the application directory explicit on sys.path and validate that the whole
# core module was uploaded, so a partial GitHub upload produces a useful UI
# message instead of a ModuleNotFoundError traceback.
APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

REQUIRED_SCANNER_FILES = ("scanner.py",)
missing_source_files = [name for name in REQUIRED_SCANNER_FILES if not (APP_ROOT / name).is_file()]
if missing_source_files:
    st.error("Deployment tidak lengkap: modul `scanner.py` belum ada di root repository.")
    st.code(
        "Root repository harus berisi:\n"
        "app.py\nscanner.py\nrequirements.txt",
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
    build_independent_price_validation,
    idx_daily_bar_is_final,
    idx_regular_decision_window,
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
):
    return fetch_automatic_independent_prices(
        tickers,
        reference_date=reference_date,
        twelve_data_api_key=_twelve_data_api_key,
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
        "stop_pct", "quote_spread_pct", "max_loss_pct_account", "broksum_net_ratio",
        "independent_price_divergence_pct",
    ):
        if ratio_column in out:
            out[ratio_column] = pd.to_numeric(out[ratio_column], errors="coerce") * 100.0
    columns = [
        "ticker",
        "status",
        "setup",
        "grade",
        "quality_score",
        "composite_score",
        "analyst_fusion_score",
        "analyst_order_mode",
        "execution_mode",
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
        "order_instruction",
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
        "READY_NOT_SELECTED": 1,
        "READY_FOR_PRICE_VERIFY": 2,
        "PENDING_CLOSE": 3,
        "PENDING_DATA": 4,
        "WATCHLIST_ENTRY": 5,
        "BLOCKED_CONTEXT": 6,
        "REJECT": 7,
    }).fillna(99)
    return out.sort_values(
        ["status_rank", "composite_score", "quality_score", "rr2", "adtv20_idr"],
        ascending=[True, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def execution_funnel_summary(signals: pd.DataFrame) -> pd.DataFrame:
    if signals is None or signals.empty:
        return pd.DataFrame(columns=["Tahap", "Kandidat tersisa"])
    total = pd.Series(True, index=signals.index)
    actionable = signals.get("analyst_order_mode", pd.Series("WATCH_ONLY", index=signals.index)).fillna("WATCH_ONLY").ne("WATCH_ONLY")
    no_hard = signals.get("analyst_hard_blockers", pd.Series("", index=signals.index)).fillna("").astype(str).str.strip().eq("")
    final_bar = ~signals.get("pending_close", pd.Series(False, index=signals.index)).fillna(False).astype(bool)
    valid_signal = signals.get("setup_valid_signal", pd.Series(False, index=signals.index)).fillna(False).astype(bool)
    return pd.DataFrame([
        {"Tahap": "Setup terdeteksi", "Kandidat tersisa": int(total.sum())},
        {"Tahap": "Order mode actionable", "Kandidat tersisa": int(actionable.sum())},
        {"Tahap": "Tidak ada invalidasi struktural/data", "Kandidat tersisa": int((actionable & no_hard).sum())},
        {"Tahap": "Candle EOD final", "Kandidat tersisa": int((actionable & no_hard & final_bar).sum())},
        {"Tahap": "Setup valid Signal-First", "Kandidat tersisa": int(valid_signal.sum())},
        {"Tahap": "EXECUTION_READY", "Kandidat tersisa": int(signals["status"].eq("EXECUTION_READY").sum())},
    ])


def result_table(df: pd.DataFrame) -> None:
    st.dataframe(
        prepare_display(df),
        width="stretch",
        hide_index=True,
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", pinned=True),
            "status": st.column_config.TextColumn("Status", pinned=True),
            "setup": "Setup",
            "quality_score": st.column_config.NumberColumn("Technical", format="%.1f"),
            "composite_score": st.column_config.NumberColumn("Composite", format="%.1f"),
            "analyst_fusion_score": st.column_config.NumberColumn("Signal score", format="%.1f"),
            "analyst_order_mode": st.column_config.TextColumn("Analyst order mode"),
            "execution_mode": st.column_config.TextColumn("Execution mode"),
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
            "rr1": st.column_config.NumberColumn("RR1", format="%.2f"),
            "rr2": st.column_config.NumberColumn("RR2", format="%.2f"),
            "stop_pct": st.column_config.NumberColumn("Risk", format="%.1f%%"),
            "distance_atr": st.column_config.NumberColumn("Dist. ATR", format="%.2f"),
            "volume_ratio": st.column_config.NumberColumn("Vol x", format="%.2f"),
            "adtv20_idr": st.column_config.NumberColumn("ADTV20", format="Rp %.0f"),
            "fundamental_score": st.column_config.NumberColumn("Fund.", format="%.1f"),
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
            "ara_price": st.column_config.NumberColumn("ARA price", format="Rp %.0f"),
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
            "fundamental_coverage": st.column_config.NumberColumn("Fund. coverage", format="%.0f%%"),
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
        stop_column="hard_stop", tp1_column="ara_price", tp2_column="ara_price",
    )
    bridge = pd.DataFrame(rows, columns=columns)
    if bridge.empty:
        return bridge
    numeric = ["entry_low", "entry_high", "entry", "stop_loss", "tp1", "tp2", "rr1", "rr2"]
    for column in numeric:
        bridge[column] = pd.to_numeric(bridge[column], errors="coerce")
    bridge = bridge.drop_duplicates(["source", "ticker", "setup", "status"], keep="first")
    status_rank = {
        "EXECUTION_READY": 0, "SNIPER_READY": 0, "BPJS_READY": 0, "BSJP_READY": 0,
        "READY_NOT_SELECTED": 1, "MULTIBAGGER_A_CANDIDATE": 1,
        "ARA_CONTINUATION_VERIFIED_ORDERFLOW": 0, "PRE_ARA_READY": 0, "ARA_CONTINUATION_READY": 1,
        "PRE_ARA_CANDIDATE": 2, "ARA_CONTINUATION_CANDIDATE": 2, "ARA_CONFIRMED_ONLY": 3,
        "READY_FOR_PRICE_VERIFY": 2, "WAIT_SNIPER_RETRACE": 3,
        "BPJS_WATCHLIST": 3, "BSJP_WATCHLIST": 3, "MULTIBAGGER_B_CANDIDATE": 3,
        "PRE_ARA_WATCHLIST": 3,
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
    st.download_button(
        "Download portfolio action plan",
        portfolio_analysis.to_csv(index=False).encode("utf-8"),
        "stockbit_portfolio_action_plan.csv",
        "text/csv",
        width="stretch",
    )

st.title("IDX Super Scanner — ARA Intelligence v4.6.0")
st.caption("Signal-First: setup valid diterbitkan tanpa dibatasi cash, lot, risiko per trade, jumlah posisi, atau alokasi portfolio. Risiko tetap ditampilkan sebagai informasi; harga dan bid/offer final dikonfirmasi di Stockbit.")

with st.sidebar:
    st.header("Signal-First scanner")
    st.info("Account-risk gate NONAKTIF: cash, lot, open risk, jumlah posisi, dan batas modal tidak menggugurkan sinyal.")
    real_money_mode = True
    with st.expander("Input portfolio opsional — hanya untuk analisis posisi", expanded=False):
        account_size = st.number_input("Equity akun (Rp)", 1_000_000, 10_000_000_000, 5_000_000, 500_000)
        cash_on_hand = st.number_input("Cash on hand (Rp)", 0, 10_000_000_000, 5_000_000, 100_000)
        portfolio_equity_mode = st.selectbox(
            "Dasar bobot portfolio",
            ["Gunakan Equity akun", "Estimasi nilai posisi + cash"],
            index=0,
            help="Hanya memengaruhi tab Portfolio, bukan status sinyal scanner.",
        )
    risk_per_trade = 0.005
    max_positions = 999
    max_position_pct = 1.0
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
        value=False,
        disabled=not use_fundamentals,
        help="Aktifkan hanya untuk riset Multibagger. Default off mengurangi rate limit dan memprioritaskan fundamental kandidat core.",
    )
    fundamental_n = st.slider("Jumlah kandidat fundamental", 10, 300, 60, disabled=not use_fundamentals or multibagger_full_universe)
    st.subheader("Data otomatis")
    twelve_data_api_key = configured_twelve_data_key()
    itick_api_token = configured_itick_token()
    st.caption(
        "OHLCV gratis: cache current → Yahoo → IDX Stock Summary EOD → iTick free bila token tersedia. "
        "Harga kedua tetap memakai IDX → Google Finance → Twelve Data opsional."
    )
    if itick_api_token:
        st.success("Fallback OHLCV iTick free terkonfigurasi; rate guard internal maksimum 4 call/menit.")
    else:
        st.caption("iTick belum dikonfigurasi. Scanner tetap berjalan gratis dengan cache, Yahoo, dan IDX resmi.")
    if twelve_data_api_key:
        st.success("Fallback harga independen Twelve Data terkonfigurasi.")
    else:
        st.caption("Twelve Data tidak dikonfigurasi; IDX dan Google tetap berjalan tanpa API key.")
    enable_intraday_specialty = st.checkbox(
        "Ambil intraday 5m untuk BSJP/BPJS & ARA Hunter",
        value=True,
        help="Hanya shortlist liquid/momentum yang diunduh agar scan 300 saham tetap terkendali.",
    )
    intraday_shortlist_n = st.slider(
        "Maksimum shortlist intraday", 20, 120, 70, 10,
        disabled=not enable_intraday_specialty,
    )
    current_positions_manual = 0
    current_invested_manual = 0.0
    current_open_risk_manual = 0.0
    st.divider()
    st.caption("Signal-First: semua kandidat valid tetap ditampilkan. Lot ditentukan pengguna di Stockbit.")

sample_csv = b"ticker\nADRO\nANTM\nBRMS\nMDKA\nTAPG\n"
portfolio_sample_csv = (
    b"ticker,lots,avg_price,stop_loss,take_profit,notes\n"
    b"ADRO,10,2150,,,Core position\n"
    b"ANTM,5,1860,,,Trading position\n"
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
        "boleh menjadi EXECUTION_READY untuk rencana limit order hari ini."
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
scan_signature = upload_fingerprint(uploaded, portfolio_uploaded, broksum_uploaded, orderbook_uploaded) if uploaded is not None else ""
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
    real_money_mode=bool(real_money_mode),
    require_fundamentals=bool(use_fundamentals),
    require_market_status=False,
    require_news_review=False,
    require_validation=False,
    require_independent_price_verification=True,
    account_size_idr=float(account_size),
    cash_on_hand_idr=float(cash_on_hand),
    risk_per_trade_pct=float(risk_per_trade),
    max_positions=int(max_positions),
    max_position_pct=float(max_position_pct),
)
portfolio_equity_input = float(account_size) if portfolio_equity_mode == "Gunakan Equity akun" else None

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
    portfolio_analysis, portfolio_summary = analyze_portfolio_positions(
        portfolio,
        histories,
        fundamentals=fundamentals,
        signals=pd.DataFrame(),
        account_equity_idr=portfolio_equity_input,
        cash_on_hand_idr=float(cash_on_hand),
        config=cfg,
    )
    st.session_state["scan_result"] = {
        "mode": "portfolio",
        "portfolio": portfolio,
        "portfolio_analysis": portfolio_analysis,
        "portfolio_summary": portfolio_summary,
        "fundamentals": fundamentals,
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
        market_context=result.get("market_context"),
        intraday=intraday_histories,
        config=cfg,
    )
    specialty_screens["ara_hunter"] = apply_ara_external_confirmation(
        specialty_screens.get("ara_hunter", pd.DataFrame()),
        broker_summary=broksum,
        orderbook=orderbook,
        now=now_jkt_ui,
    )

    result.update({
        "mode": "scanner", "signals": signals, "validation_stats": stats,
        "validation_trades": trades, "fundamentals": fundamentals,
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
    })
    st.session_state["scan_result"] = result
    st.session_state["_last_auto_scan_signature"] = scan_signature
    progress.progress(100, text="Scan dan portfolio review selesai")
    progress.empty()

if "scan_result" not in st.session_state:
    st.markdown(
        """
        <div class="scanner-note">
          <b>Alur otomatis v4.6.0 ARA Intelligence</b><br>
          1) Upload universe ticker—scan dan pengumpulan evidence langsung dimulai.<br>
          2) Portfolio Stockbit bersifat opsional; unggah hanya bila ingin cash/heat/posisi dihitung dari snapshot nyata.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

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

m1, m2, m3, m4, m5, m6, m7, m8, m9 = st.columns(9)
m1.metric("Ticker valid", len(result["prepared"]))
m2.metric("Setup terdeteksi", len(signals))
m3.metric("Execution ready", int(signals["status"].eq("EXECUTION_READY").sum()) if not signals.empty else 0)
m4.metric("Strict ready", int(signals.get("strict_execution_ready", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not signals.empty else 0)
m5.metric("Account gate", "OFF")
m6.metric("Menunggu close", int(signals["status"].eq("PENDING_CLOSE").sum()) if not signals.empty else 0)
m7.metric("Pending data", int(signals["status"].eq("PENDING_DATA").sum()) if not signals.empty else 0)
m8.metric("Watchlist", int(signals["status"].eq("WATCHLIST_ENTRY").sum()) if not signals.empty else 0)
m9.metric("Breadth > EMA50", f"{context.breadth_ema50:.0f}%" if context.breadth_ema50 is not None else "N/A")

tab_orders, tab_setups, tab_sniper, tab_fast, tab_multibagger, tab_ara, tab_portfolio, tab_chart, tab_bridge, tab_validation, tab_audit, tab_method = st.tabs(
    [
        "Order Builder", "Core Setups", "Sniper Entry", "BSJP / BPJS",
        "Multibagger", "ARA Hunter", "Portfolio Stockbit", "Chart",
        "TradingView / Stockbit", "Validation", "Audit Universe", "Metodologi",
    ]
)

specialty_screens: dict[str, pd.DataFrame] = result.get("specialty_screens", {})

with tab_orders:
    if signals.empty:
        st.warning("Tidak ada setup yang valid. Audit Universe menjelaskan alasan per ticker.")
    else:
        execution = signals[signals["status"].eq("EXECUTION_READY")]
        pending_close = signals[signals["status"].eq("PENDING_CLOSE")]
        price_verify = signals[signals["status"].eq("READY_FOR_PRICE_VERIFY")]
        st.subheader("Order Builder — execution-ready")
        st.caption("EXECUTION_READY berarti setup valid menurut Signal-First dan tidak dibatasi kondisi akun. Lot ditentukan pengguna; cocokkan harga terakhir dan bid/offer di Stockbit sebelum submit. Market order tetap dihindari.")
        if execution.empty:
            st.info("Tidak ada direct-execution order dari core setup saat ini. Lihat funnel di bawah untuk gate yang menggugurkan kandidat.")
        else:
            result_table(execution)
        if not pending_close.empty:
            st.subheader("PENDING_CLOSE")
            st.info("Setup teknikal sudah terbentuk dari completed EOD terakhir. Refresh setelah 16:20 WIB; belum boleh disalin sebagai order.")
            result_table(pending_close)
        if not price_verify.empty:
            st.subheader("READY_FOR_PRICE_VERIFY")
            st.warning(
                "Harga otomatis kedua belum tervalidasi. Setup tetap dapat dinilai pada mode Signal-First, tetapi "
                "last price dan bid/offer wajib dicocokkan di Stockbit sebelum memasang limit order."
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
        c1, c2 = st.columns(2)
        c1.download_button(
            "Download semua hasil CSV",
            signals.drop(columns=["status_rank"], errors="ignore").to_csv(index=False).encode("utf-8"),
            "idx_super_scanner_results.csv", "text/csv", width="stretch",
        )
        c2.download_button(
            "Download execution-ready CSV",
            execution.drop(columns=["status_rank"], errors="ignore").to_csv(index=False).encode("utf-8"),
            "idx_execution_ready.csv", "text/csv", width="stretch",
            disabled=execution.empty,
        )

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
            ready_count = int(subset["status"].eq("EXECUTION_READY").sum()) if not subset.empty else 0
            c1, c2 = st.columns(2)
            c1.metric("Kandidat", len(subset))
            c2.metric("Execution ready", ready_count)
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
    st.subheader("Strict Sniper Entry")
    st.caption("Subset paling ketat dari sweep–BOS–FVG: menuntut jarak entry dekat, volume displacement, silent accumulation, RR, dan stop yang sempit.")
    specialty_table(sniper)
    render_specialty_download("Download Sniper Entry", sniper, "strict_sniper_entries.csv")

with tab_fast:
    fast_bsjp, fast_bpjs = st.tabs(["BSJP — Beli Sore Jual Pagi", "BPJS — Beli Pagi Jual Sore"])
    with fast_bsjp:
        bsjp = specialty_screens.get("bsjp", pd.DataFrame())
        st.warning("Strategi overnight berisiko gap. BSJP_READY memerlukan data intraday sesi hari ini yang masih fresh, regime IHSG layak, dan window 14:30–15:49 WIB. Hasil tetap MANUAL_REVIEW_ONLY.")
        specialty_table(bsjp)
        render_specialty_download("Download BSJP", bsjp, "bsjp_candidates.csv")
    with fast_bpjs:
        bpjs = specialty_screens.get("bpjs", pd.DataFrame())
        st.warning("BPJS adalah day trade manual. BPJS_READY hanya dapat muncul pada 09:20–10:45 WIB setelah ORB 15 menit, VWAP, opening volume, freshness, regime, dan batas gap lulus. Posisi wajib ditutup sebelum market close.")
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
    st.subheader("Multibagger Radar")
    st.caption("Ranking 12–36 bulan berdasarkan pertumbuhan, profitabilitas, cash flow/utang, valuasi, momentum, dan akumulasi. Bukan prediksi bahwa harga pasti berlipat.")
    specialty_table(multibagger, height=500)
    render_specialty_download("Download Multibagger Radar", multibagger, "multibagger_radar.csv")

with tab_ara:
    ara = specialty_screens.get("ara_hunter", pd.DataFrame())
    st.subheader("ARA Intelligence — Pre-ARA & Continuation")
    st.info(
        "PRE_ARA mencari saham yang belum ARA. ARA_CONTINUATION menilai saham yang sudah ARA/dekat ARA "
        "untuk peluang sesi berikutnya. Orderflow dan queue otomatis adalah proxy OHLCV, bukan data broker/order book aktual."
    )
    if not ara.empty and "ara_hunter_status" in ara:
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Pre-ARA ready", int(ara["ara_hunter_status"].eq("PRE_ARA_READY").sum()))
        a2.metric(
            "Continuation ready",
            int(ara["ara_hunter_status"].isin(["ARA_CONTINUATION_READY", "ARA_CONTINUATION_VERIFIED_ORDERFLOW"]).sum()),
        )
        a3.metric("Pre-ARA candidate", int(ara["ara_hunter_status"].eq("PRE_ARA_CANDIDATE").sum()))
        a4.metric("ARA confirmed only", int(ara["ara_hunter_status"].eq("ARA_CONFIRMED_ONLY").sum()))
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
    status_audit = result.get("market_status", pd.DataFrame())
    news_audit = result.get("news_review", pd.DataFrame())
    quote_audit = result.get("execution_snapshots", pd.DataFrame())
    independent_audit = result.get("price_validation", pd.DataFrame())
    provider_audit = result.get("independent_provider_report", pd.DataFrame())
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
    st.dataframe(
        pd.DataFrame(
            [
                ["Verified local cache", "Menghindari unduh ulang full-history ketika EOD sudah current", "Automatic cache-first"],
                ["Yahoo Finance via yfinance", "Primary OHLCV daily/intraday, IHSG, fundamental", "Automatic bounded batch"],
                ["IDX Stock Summary API", "Menambal bar EOD terakhir saat Yahoo gagal dan cache historis tersedia", "Automatic official fallback"],
                ["iTick free tier", "Fallback OHLCV daily/intraday untuk ticker gagal", "Optional token; ≤4 call/min internal"],
                ["Official IDX public pages", "Daftar saham, FCA/pemantauan, suspensi, aksi korporasi, disclosure", "Automated + cache"],
                ["Google News RSS", "Berita luas per ticker", "Automated"],
                ["IDX Stock Summary API", "Cross-check harga EOD resmi kandidat", "Automatic primary verification"],
                ["Google Finance public quote", "Fallback harga independen kandidat", "Automatic bounded shortlist"],
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
            "Belum ada harga independen yang terverifikasi. Signal-First tetap dapat menerbitkan setup valid, "
            "tetapi harga terakhir dan bid/offer wajib dicocokkan di Stockbit sebelum memasang limit order."
        )
    if not independent_audit.empty:
        st.dataframe(independent_audit, hide_index=True, width="stretch")
    if not provider_audit.empty:
        with st.expander("Audit provider harga otomatis", expanded=bool(provider_audit["status"].ne("OK").any())):
            st.dataframe(provider_audit, hide_index=True, width="stretch")
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
        ### Hirarki sinyal v4.6.0 ARA Intelligence

        1. **Validitas struktur:** setup, zona entry, trigger/retest, masa berlaku, dan level harga harus masih valid.
        2. **Data minimum:** candle EOD harus final, OHLCV tidak boleh unavailable atau terlalu lama, dan suspensi/FCA tetap memblokir.
        3. **Signal-First:** cash, jumlah lot, open risk, jumlah posisi, dan batas modal akun tidak pernah menggugurkan setup valid.
        4. **Risk disclosure:** stop lebar, RR rendah, likuiditas tipis, volatilitas ekstrem, regime risk-off, berita, dan fundamental ditampilkan sebagai grade/peringatan—bukan account gate.
        5. **Core setup:** Pullback dapat memakai `LIMIT_PULLBACK_ZONE`; Breakout memakai limit setelah retest/reclaim; Reversal membutuhkan CHOCH/BOS; trigger langsung memakai `TRIGGER_CONFIRMED`.
        6. **ARA Intelligence:** model dipisahkan menjadi `PRE_ARA_READY/CANDIDATE` dan `ARA_CONTINUATION_READY/CANDIDATE`. Proxy orderflow/queue otomatis dapat diperkuat dengan upload Broker Summary dan Order Book.
        7. **Verifikasi broker:** bila harga independen belum tersedia, `requires_stockbit_price_check` aktif. Cocokkan last price dan bid/offer di Stockbit.
        8. **Sesi perdagangan:** pre-market memakai EOD sebelumnya; sesi aktif menghasilkan `PENDING_CLOSE`; post-close memakai candle final hari tersebut.
        9. **Sizing:** lot hanya kalkulator informasional. `EXECUTION_READY` menggunakan `BUY_LIMIT_USER_SIZE` dan pengguna menentukan lot.
        10. **Chart state:** pilihan ticker chart memakai fragment sehingga tidak mereset dashboard atau kembali ke tab pertama.

        `EXECUTION_READY` pada versi ini berarti setup dinilai valid secara struktur dan data minimum. Label tersebut bukan jaminan profit dan bukan rekomendasi market order.
        """
    )
