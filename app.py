from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="IDX Super Scanner Resilient v4.0", page_icon="🎯", layout="wide")

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
    apply_universe_integrity_gate,
    attach_backtest_stats,
    attach_broker_summary,
    attach_fundamentals,
    attach_position_sizing,
    enforce_portfolio_execution_budget,
    finalize_execution_integrity,
    download_benchmark,
    download_ohlcv,
    fetch_fundamentals,
    fetch_resilient_market_status,
    fetch_resilient_news_review,
    fetch_resilient_fundamentals,
    fetch_execution_snapshots,
    make_signal_chart,
    parse_broker_summary_csv,
    parse_market_status_csv,
    parse_news_review_csv,
    parse_ticker_csv,
    parse_portfolio_csv,
    analyze_portfolio_positions,
    run_walkforward_validation,
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


@st.cache_data(ttl=1_800, show_spinner=False)
def cached_market_data(tickers: tuple[str, ...], period: str):
    histories, report = download_ohlcv(tickers, period=period)
    benchmark = download_benchmark(period=period)
    report.benchmark_ok = not benchmark.empty
    return histories, report, benchmark


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


def rupiah(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"Rp{float(value):,.0f}".replace(",", ".")


def prepare_display(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()
    if "historical_events" in out:
        reliable = out["historical_events"].fillna(0) >= 30
        out["probability_estimate"] = out["bayes_probability"].where(reliable)
        out["entry_fill_estimate"] = out["entry_fill_rate_5d"].where(reliable)
    else:
        out["probability_estimate"] = np.nan
        out["entry_fill_estimate"] = np.nan
    # Streamlit NumberColumn uses printf-style formatting and does not
    # multiply fractional ratios automatically. Convert only display copies.
    for ratio_column in ("stop_pct", "quote_spread_pct", "max_loss_pct_account", "broksum_net_ratio"):
        if ratio_column in out:
            out[ratio_column] = pd.to_numeric(out[ratio_column], errors="coerce") * 100.0
    columns = [
        "ticker",
        "status",
        "setup",
        "grade",
        "quality_score",
        "composite_score",
        "execution_integrity_score",
        "execution_confidence_score",
        "data_completeness_score",
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
    out["status_rank"] = out["status"].map({"EXECUTION_READY": 0, "PENDING_DATA": 1, "WATCHLIST_ENTRY": 2, "BLOCKED_CONTEXT": 3, "REJECT": 4}).fillna(99)
    return out.sort_values(
        ["status_rank", "composite_score", "quality_score", "rr2", "adtv20_idr"],
        ascending=[True, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def result_table(df: pd.DataFrame) -> None:
    st.dataframe(
        prepare_display(df),
        use_container_width=True,
        hide_index=True,
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", pinned=True),
            "status": st.column_config.TextColumn("Status", pinned=True),
            "setup": "Setup",
            "quality_score": st.column_config.NumberColumn("Technical", format="%.1f"),
            "composite_score": st.column_config.NumberColumn("Composite", format="%.1f"),
            "execution_integrity_score": st.column_config.NumberColumn("Execution confidence", format="%.1f%%"),
            "execution_confidence_score": st.column_config.NumberColumn("Confidence", format="%.1f%%"),
            "data_completeness_score": st.column_config.NumberColumn("Data completeness", format="%.1f%%"),
            "validation_gate_score": st.column_config.NumberColumn("OOS gate", format="%.0f%%"),
            "probability_estimate": st.column_config.NumberColumn("P(TP1<SL)*", format="%.1f%%"),
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
            "stop_pct": st.column_config.NumberColumn("Risk", format="%.1%%"),
            "distance_atr": st.column_config.NumberColumn("Dist. ATR", format="%.2f"),
            "volume_ratio": st.column_config.NumberColumn("Vol x", format="%.2f"),
            "adtv20_idr": st.column_config.NumberColumn("ADTV20", format="Rp %.0f"),
            "fundamental_score": st.column_config.NumberColumn("Fund.", format="%.1f"),
            "fundamental_coverage": st.column_config.NumberColumn("Fund. coverage", format="%.0f%%"),
            "statement_age_days": st.column_config.NumberColumn("FS age", format="%.0f d"),
            "silent_accumulation_score": st.column_config.NumberColumn("Accumulation", format="%.0f"),
            "up_down_value_ratio20": st.column_config.NumberColumn("Up/Down value", format="%.2f"),
            "quote_last_price": st.column_config.NumberColumn("Quote", format="Rp %.0f"),
            "quote_spread_pct": st.column_config.NumberColumn("Spread", format="%.2%%"),
            "suggested_lots": st.column_config.NumberColumn("Lot", format="%d"),
            "stockbit_order_price": st.column_config.NumberColumn("Order price", format="Rp %.0f"),
            "stockbit_order_lots": st.column_config.NumberColumn("Order lot", format="%d"),
            "execution_rank": st.column_config.NumberColumn("Rank", format="%.0f"),
            "capital_required_idr": st.column_config.NumberColumn("Modal order", format="Rp %.0f"),
            "max_loss_idr": st.column_config.NumberColumn("Max loss est.", format="Rp %.0f"),
            "max_loss_pct_account": st.column_config.NumberColumn("Risk akun", format="%.2f%%"),
            "broksum_net_ratio": st.column_config.NumberColumn("Broksum net", format="%.1%%"),
            "valid_until": st.column_config.DatetimeColumn("Valid until", format="DD MMM YYYY"),
        },
    )


st.title("IDX Super Scanner — Resilient Evidence v4.0")
st.caption("Resilient evidence · critical blockers · portfolio action engine · actual cash/risk · Stockbit limit-order plan")

with st.sidebar:
    st.header("Filter live money")
    real_money_mode = st.checkbox(
        "Real-money resilient mode",
        value=True,
        help="Critical blocker tetap menghentikan order. Gangguan satu provider hanya menurunkan confidence dan memakai cache/fallback.",
    )
    account_size = st.number_input("Equity akun (Rp)", 1_000_000, 10_000_000_000, 10_000_000, 500_000)
    cash_on_hand = st.number_input("Cash on hand (Rp)", 0, 10_000_000_000, 10_000_000, 100_000)
    risk_per_trade = st.slider("Risiko per trade", 0.25, 1.00, 0.75, 0.05) / 100
    max_positions = st.slider("Maksimum posisi bersamaan", 1, 5, 2)
    max_position_pct = st.slider("Maksimum modal per posisi", 10, 60, 40, 5) / 100
    period = st.selectbox("Riwayat OHLCV", ["3y", "5y", "2y"], index=0)
    min_adtv_b = st.number_input("Minimum ADTV20 (Rp miliar)", 0.1, 100.0, 2.0, 0.5)
    min_score = st.slider("Minimum quality score", 50, 90, 72)
    execution_score = st.slider("Execution-ready technical score", min_score, 95, max(82, min_score))
    min_execution_confidence = st.slider("Minimum total execution confidence", 75, 95, 84)
    max_stop_pct = st.slider("Maksimum jarak SL", 3.0, 10.0, 7.0, 0.5) / 100
    min_rr2 = st.slider("Minimum RR ke TP2", 2.0, 4.0, 2.7, 0.1)
    validate = st.checkbox(
        "Chronological OOS validation",
        value=True,
        help="Validasi lemah menurunkan confidence. Edge OOS negatif yang sudah cukup sampel tetap memblokir order.",
    )
    use_fundamentals = st.checkbox("Ambil fundamental kandidat dan portfolio", value=True)
    fundamental_n = st.slider("Jumlah kandidat fundamental", 10, 150, 80, disabled=not use_fundamentals)
    st.subheader("Fallback bila tanpa portfolio CSV")
    current_positions_manual = st.number_input("Jumlah posisi aktif", 0, 20, 0, 1)
    current_invested_manual = st.number_input("Nilai posisi aktif (Rp)", 0, 10_000_000_000, 0, 100_000)
    current_open_risk_manual = st.number_input("Open risk total ke SL (Rp)", 0, 1_000_000_000, 0, 25_000)
    st.divider()
    st.caption("Bullish-only. Suspensi/FCA, breakdown, berita negatif material, level invalid, candle belum final, atau portfolio risk penuh tetap memblokir order.")

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
        help=(
            "Kolom boleh bernama ticker/symbol/kode/emiten, atau gunakan kolom pertama. "
            ".JK ditambahkan otomatis; maksimal 1.200 kode."
        ),
    )
with right:
    st.write("")
    st.write("")
    st.download_button("Unduh contoh universe", sample_csv, "sample_tickers.csv", "text/csv", use_container_width=True)

p1, p2 = st.columns([3, 1])
with p1:
    portfolio_uploaded = st.file_uploader(
        "Opsional: upload snapshot portfolio Stockbit CSV",
        type=["csv", "txt"],
        help="Kolom wajib: ticker, lots, avg_price. Stop loss dan take profit boleh kosong.",
        key="portfolio_upload",
    )
with p2:
    st.write("")
    st.write("")
    st.download_button(
        "Unduh template portfolio", portfolio_sample_csv, "stockbit_portfolio_template.csv", "text/csv", use_container_width=True
    )

st.info(
    "Mesin evidence memakai sumber live, retry, cache terakhir yang masih layak, dan fallback OHLCV/quote. "
    "Gangguan satu sumber tidak otomatis menghapus setup. Bukti negatif eksplisit tetap menjadi critical blocker."
)

run = st.button("Jalankan scanner", type="primary", use_container_width=True, disabled=uploaded is None)

if run and uploaded is not None:
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

    cfg = ScanConfig().replace(
        min_adtv_idr=float(min_adtv_b) * 1_000_000_000,
        min_score=float(min_score),
        execution_score=float(execution_score),
        min_execution_confidence=float(min_execution_confidence),
        max_stop_pct=float(max_stop_pct),
        min_rr2=float(min_rr2),
        fundamental_top_n=int(fundamental_n),
        real_money_mode=bool(real_money_mode),
        require_fundamentals=False,
        require_market_status=False,
        require_news_review=False,
        require_validation=False,
        account_size_idr=float(account_size),
        cash_on_hand_idr=float(cash_on_hand),
        risk_per_trade_pct=float(risk_per_trade),
        max_positions=int(max_positions),
        max_position_pct=float(max_position_pct),
    )
    market_status = pd.DataFrame()
    news_review = pd.DataFrame()
    broksum = pd.DataFrame()
    progress = st.progress(0, text=f"Mengunduh OHLCV {len(all_tickers)} ticker dan IHSG…")
    histories, report, benchmark = cached_market_data(tuple(all_tickers), period)
    progress.progress(30, text="Menghitung indikator, struktur pasar, empat setup, dan seluruh posisi portfolio…")
    result = ScanEngine(cfg).scan(histories, benchmark)
    signals = result["signals"]
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
        progress.progress(63, text="Mengambil fundamental kandidat dan posisi portfolio dengan cache fallback…")
        ranked_names = (
            signals.sort_values(["status_rank", "quality_score"], ascending=[True, False])["ticker"].drop_duplicates().tolist()
            if not signals.empty else []
        )
        execution_names = (
            signals.loc[signals["status"].eq("EXECUTION_READY"), "ticker"].drop_duplicates().tolist()
            if not signals.empty else []
        )
        top_names = list(dict.fromkeys(portfolio_tickers + execution_names + ranked_names[:fundamental_n]))
        fundamentals = cached_fundamentals(tuple(top_names))
    signals = attach_fundamentals(signals, fundamentals)
    signals = apply_fundamental_gate(signals, cfg)

    progress.progress(76, text="Menyelesaikan status IDX dan berita melalui live source, retry, dan cache…")
    potential = (
        signals.loc[signals["status"].eq("EXECUTION_READY"), "ticker"].drop_duplicates().tolist()
        if not signals.empty else []
    )
    context_names = list(dict.fromkeys(potential + portfolio_tickers))
    if context_names:
        market_status = cached_automatic_market_status(tuple(context_names))
        news_review = cached_automatic_news(tuple(context_names), cfg.min_news_lookback_days)
    signals = apply_market_status_gate(signals, market_status, cfg)
    signals = apply_news_gate(signals, news_review, cfg)

    quote_candidates = (
        signals.loc[signals["status"].eq("EXECUTION_READY"), "ticker"].drop_duplicates().tolist()
        if not signals.empty else []
    )
    snapshots = cached_execution_snapshots(tuple(quote_candidates)) if quote_candidates else pd.DataFrame()
    signals = apply_execution_snapshot_gate(signals, snapshots, cfg)
    signals = attach_position_sizing(signals, cfg)

    # First portfolio pass derives current market value and structural open risk.
    portfolio_analysis, portfolio_summary = analyze_portfolio_positions(
        portfolio,
        histories,
        fundamentals=fundamentals,
        signals=signals,
        account_equity_idr=float(account_size),
        cash_on_hand_idr=float(cash_on_hand),
        config=cfg,
    )
    if not portfolio.empty:
        current_positions = int(portfolio_summary.get("positions", len(portfolio)))
        current_invested = float(portfolio_summary.get("market_value_idr", 0.0))
        current_open_risk = float(portfolio_summary.get("open_risk_idr", 0.0))
    else:
        current_positions = int(current_positions_manual)
        current_invested = float(current_invested_manual)
        current_open_risk = float(current_open_risk_manual)

    progress.progress(90, text="Menghitung lot, actual cash, dan risiko portofolio agregat…")
    signals = enforce_portfolio_execution_budget(
        signals,
        cfg,
        current_positions=current_positions,
        current_open_risk_idr=current_open_risk,
        current_invested_idr=current_invested,
        cash_on_hand_idr=float(cash_on_hand),
    )
    signals = finalize_execution_integrity(signals, cfg)
    signals = sort_signals(signals)

    # Re-evaluate holdings against final scanner state.
    portfolio_analysis, portfolio_summary = analyze_portfolio_positions(
        portfolio,
        histories,
        fundamentals=fundamentals,
        signals=signals,
        account_equity_idr=float(account_size),
        cash_on_hand_idr=float(cash_on_hand),
        config=cfg,
    )

    result["signals"] = signals
    result["validation_stats"] = stats
    result["validation_trades"] = trades
    result["fundamentals"] = fundamentals
    result["market_status"] = market_status
    result["news_review"] = news_review
    result["broker_summary"] = broksum
    result["execution_snapshots"] = snapshots
    result["download_report"] = report
    result["portfolio"] = portfolio
    result["portfolio_analysis"] = portfolio_analysis
    result["portfolio_summary"] = portfolio_summary
    result["all_histories"] = histories
    st.session_state["scan_result"] = result
    progress.progress(100, text="Scan dan portfolio review selesai")
    progress.empty()

if "scan_result" not in st.session_state:
    st.markdown(
        """
        <div class="scanner-note">
          <b>Cara kerja singkat</b><br>
          Upload ticker → unduh OHLCV → cek regime & liquidity → deteksi setup → hitung level pada fraksi harga IDX → validasi historis → keluarkan order plan.<br>
          <span class="small-muted">EXECUTION_READY adalah instruksi limit order; PENDING_DATA tetap ditampilkan dan tidak dibuang.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

result = st.session_state["scan_result"]
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

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Ticker valid", len(result["prepared"]))
m2.metric("Setup terdeteksi", len(signals))
m3.metric("Execution ready", int(signals["status"].eq("EXECUTION_READY").sum()) if not signals.empty else 0)
m4.metric("Pending data", int(signals["status"].eq("PENDING_DATA").sum()) if not signals.empty else 0)
m5.metric("Watchlist", int(signals["status"].eq("WATCHLIST_ENTRY").sum()) if not signals.empty else 0)
m6.metric("Breadth > EMA50", f"{context.breadth_ema50:.0f}%" if context.breadth_ema50 is not None else "N/A")

tab_orders, tab_portfolio, tab_chart, tab_validation, tab_audit, tab_method = st.tabs(
    ["Setup & Order Builder", "Portfolio Stockbit", "Chart", "Validation", "Audit Universe", "Metodologi"]
)

with tab_orders:
    if signals.empty:
        st.warning("Tidak ada setup yang valid. Audit Universe menjelaskan alasan per ticker.")
    else:
        execution = signals[signals["status"].eq("EXECUTION_READY")]
        if not execution.empty:
            st.subheader("Order Builder — execution-ready")
            st.caption("Hanya BUY_LIMIT dengan status EXECUTION_READY dan tanpa critical blocker yang boleh disalin ke Stockbit. Market order dilarang.")
            result_table(execution)
        st.subheader("Semua setup, pending evidence, dan watchlist")
        result_table(signals)
        st.caption("*Gangguan satu provider menurunkan confidence, bukan otomatis menghapus setup. Probability tetap statistik OOS pooled per setup, bukan jaminan hasil individual.")
        c1, c2 = st.columns(2)
        c1.download_button(
            "Download semua hasil CSV",
            signals.drop(columns=["status_rank"], errors="ignore").to_csv(index=False).encode("utf-8"),
            "idx_super_scanner_results.csv",
            "text/csv",
            use_container_width=True,
        )
        c2.download_button(
            "Download execution-ready CSV",
            execution.drop(columns=["status_rank"], errors="ignore").to_csv(index=False).encode("utf-8"),
            "idx_execution_ready.csv",
            "text/csv",
            use_container_width=True,
            disabled=execution.empty,
        )


with tab_portfolio:
    portfolio_analysis: pd.DataFrame = result.get("portfolio_analysis", pd.DataFrame())
    portfolio_summary: dict = result.get("portfolio_summary", {})
    if portfolio_analysis.empty:
        st.info("Upload portfolio CSV untuk memperoleh keputusan HOLD, TAKE_PROFIT, REDUCE, CUT_LOSS, atau AVG_DOWN_ALLOWED.")
    else:
        st.subheader("Portfolio decision engine")
        p1, p2, p3, p4, p5 = st.columns(5)
        p1.metric("Nilai posisi", rupiah(portfolio_summary.get("market_value_idr", np.nan)))
        p2.metric("Unrealized P/L", rupiah(portfolio_summary.get("unrealized_pnl_idr", np.nan)))
        p3.metric("P/L %", f"{portfolio_summary.get('unrealized_pnl_pct', 0):.1%}")
        p4.metric("Open risk", rupiah(portfolio_summary.get("open_risk_idr", np.nan)))
        p5.metric("Cash", rupiah(portfolio_summary.get("cash_on_hand_idr", np.nan)))
        portfolio_columns = [
            "ticker", "position_action", "action_reason", "lots", "avg_price", "last_price",
            "unrealized_pnl_idr", "unrealized_pnl_pct", "position_weight", "open_risk_idr",
            "suggested_stop_loss", "suggested_tp1", "suggested_tp2", "scanner_setup", "scanner_status",
            "avg_down_lots", "avg_down_price", "new_average_after_avg", "trend_up",
            "long_term_structure_intact", "flow_positive", "fundamental_distress",
        ]
        portfolio_view = portfolio_analysis[[c for c in portfolio_columns if c in portfolio_analysis.columns]].copy()
        for ratio_column in ("unrealized_pnl_pct", "position_weight"):
            if ratio_column in portfolio_view:
                portfolio_view[ratio_column] = pd.to_numeric(portfolio_view[ratio_column], errors="coerce") * 100.0
        st.dataframe(
            portfolio_view, use_container_width=True, hide_index=True,
            column_config={
                "avg_price": st.column_config.NumberColumn("Average", format="Rp %.0f"),
                "last_price": st.column_config.NumberColumn("Last", format="Rp %.0f"),
                "unrealized_pnl_idr": st.column_config.NumberColumn("Unrealized P/L", format="Rp %.0f"),
                "unrealized_pnl_pct": st.column_config.NumberColumn("P/L %", format="%.1%%"),
                "position_weight": st.column_config.NumberColumn("Weight", format="%.1%%"),
                "open_risk_idr": st.column_config.NumberColumn("Open risk", format="Rp %.0f"),
                "suggested_stop_loss": st.column_config.NumberColumn("Structural SL", format="Rp %.0f"),
                "suggested_tp1": st.column_config.NumberColumn("TP1", format="Rp %.0f"),
                "suggested_tp2": st.column_config.NumberColumn("TP2", format="Rp %.0f"),
                "avg_down_price": st.column_config.NumberColumn("Avg-down price", format="Rp %.0f"),
                "new_average_after_avg": st.column_config.NumberColumn("New average", format="Rp %.0f"),
            },
        )
        st.caption("AVG_DOWN_ALLOWED hanya muncul bila struktur jangka panjang bertahan, harga dekat support, flow positif, bobot aman, cash tersedia, dan setup scanner aktif. Harga turun saja tidak cukup.")
        st.download_button(
            "Download portfolio action plan",
            portfolio_analysis.to_csv(index=False).encode("utf-8"),
            "stockbit_portfolio_action_plan.csv",
            "text/csv",
        )

with tab_chart:
    if signals.empty:
        st.info("Belum ada setup untuk digambar.")
    else:
        labels = [f"{row.ticker} · {row.setup}" for row in signals.itertuples()]
        selected = st.selectbox("Pilih setup", labels)
        selected_pos = labels.index(selected)
        signal = signals.iloc[selected_pos].to_dict()
        frame = result["prepared"][signal["ticker"]]
        st.plotly_chart(make_signal_chart(frame, signal), use_container_width=True)
        st.write("Evidence:", signal.get("evidence", "—"))
        if signal.get("blockers"):
            st.warning(signal["blockers"])

with tab_validation:
    stats: pd.DataFrame = result.get("validation_stats", pd.DataFrame())
    trades: pd.DataFrame = result.get("validation_trades", pd.DataFrame())
    if stats.empty:
        st.info("Aktifkan chronological OOS validation sebelum menjalankan scan.")
    else:
        st.subheader("Chronological out-of-sample validation")
        st.caption("Detector dan structural levels live dibangun ulang pada setiap tanggal kandidat. Buy-stop/limit harus benar-benar tersentuh dalam 5 bar; gap yang merusak RR dibatalkan; biaya dan slippage masuk; bila SL/TP ambigu pada daily bar, SL dianggap lebih dulu.")
        st.dataframe(stats, use_container_width=True, hide_index=True)
        st.download_button(
            "Download seluruh trade historis",
            trades.to_csv(index=False).encode("utf-8"),
            "walkforward_trades.csv",
            "text/csv",
        )

with tab_audit:
    st.subheader("Satu baris untuk setiap ticker")
    st.dataframe(universe, use_container_width=True, hide_index=True)
    if report is not None and report.failed:
        with st.expander(f"Ticker gagal diunduh ({len(report.failed)})"):
            st.dataframe(pd.DataFrame(report.failed.items(), columns=["ticker", "error"]), hide_index=True)
    if report is not None and report.warnings:
        with st.expander(f"Peringatan kualitas OHLCV ({len(report.warnings)})"):
            st.dataframe(pd.DataFrame(report.warnings.items(), columns=["ticker", "warning"]), hide_index=True)

with tab_method:
    st.markdown(
        """
        ### Hirarki sinyal v4.0

        1. **Critical tradeability:** data harga, likuiditas, ATR, fraksi IDX, structural stop, RR, expiry, dan anti-ARA chase.
        2. **Market structure:** regime IHSG, trend, relative strength, sweep, BOS/CHOCH, pullback/retest, FVG, serta money-flow proxy.
        3. **Resilient evidence:** sumber live dicoba ulang, hasil layak disimpan dalam cache, lalu quote/OHLCV dipakai sebagai fallback yang diberi confidence lebih rendah.
        4. **Weighted decision:** teknikal 35%, risk mechanics 20%, status pasar 10%, berita 8%, fundamental 10%, OOS 7%, quote 5%, dan breadth 5%.
        5. **Critical blockers:** suspensi/FCA, corporate action yang belum disesuaikan, berita negatif material, edge OOS negatif, fundamental distress, konflik quote, candle belum final, dan portfolio risk penuh.
        6. **Portfolio engine:** setiap posisi dinilai menjadi HOLD, HOLD_NO_AVG, HOLD_TIGHT_STOP, TAKE_PROFIT_PARTIAL, REDUCE, CUT_LOSS, DO_NOT_AVG_DOWN, atau AVG_DOWN_ALLOWED.

        `PENDING_DATA` berarti setup teknikal tetap tersimpan tetapi total confidence belum mencapai threshold order. `EXECUTION_READY` hanya menghasilkan `BUY_LIMIT`; scanner tidak mengirim order otomatis ke Stockbit.
        """
    )
