from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="IDX Super Scanner Hardened v2.1 Flat", page_icon="🎯", layout="wide")

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
    attach_backtest_stats,
    attach_broker_summary,
    attach_fundamentals,
    attach_position_sizing,
    download_benchmark,
    download_ohlcv,
    fetch_fundamentals,
    make_signal_chart,
    parse_broker_summary_csv,
    parse_market_status_csv,
    parse_news_review_csv,
    parse_ticker_csv,
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
    return fetch_fundamentals(tickers)


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
    columns = [
        "ticker",
        "status",
        "setup",
        "grade",
        "quality_score",
        "composite_score",
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
        "suggested_lots",
        "capital_required_idr",
        "max_loss_idr",
        "max_loss_pct_account",
        "broksum_signal",
        "broksum_net_ratio",
        "verified_catalyst_count",
        "catalyst_summary",
        "market_status_coverage",
        "news_review_status",
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
    out["status_rank"] = out["status"].map({"EXECUTION_READY": 0, "WATCHLIST_ENTRY": 1, "REJECT": 2})
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
            "suggested_lots": st.column_config.NumberColumn("Lot", format="%d"),
            "capital_required_idr": st.column_config.NumberColumn("Modal order", format="Rp %.0f"),
            "max_loss_idr": st.column_config.NumberColumn("Max loss est.", format="Rp %.0f"),
            "max_loss_pct_account": st.column_config.NumberColumn("Risk akun", format="%.2f%%"),
            "broksum_net_ratio": st.column_config.NumberColumn("Broksum net", format="%.1%%"),
            "valid_until": st.column_config.DatetimeColumn("Valid until", format="DD MMM YYYY"),
        },
    )


st.title("IDX Super Scanner — Hardened v2.1 Flat")
st.caption("Evidence-based trend & momentum · IDX execution rules · SMC/ICT timing · fundamental quality gate")

with st.sidebar:
    st.header("Filter live money")
    real_money_mode = st.checkbox(
        "Real-money fail-closed mode",
        value=True,
        help="Tanpa fundamental, status IDX, dan news review yang valid, status otomatis diturunkan.",
    )
    account_size = st.number_input("Equity akun (Rp)", 1_000_000, 10_000_000_000, 10_000_000, 500_000)
    risk_per_trade = st.slider("Risiko per trade", 0.50, 1.50, 1.00, 0.05) / 100
    max_positions = st.slider("Maksimum posisi bersamaan", 1, 5, 2)
    max_position_pct = st.slider("Maksimum modal per posisi", 10, 60, 40, 5) / 100
    period = st.selectbox("Riwayat OHLCV", ["3y", "5y", "2y"], index=0)
    min_adtv_b = st.number_input("Minimum ADTV20 (Rp miliar)", 0.1, 100.0, 2.0, 0.5)
    min_score = st.slider("Minimum quality score", 50, 90, 70)
    execution_score = st.slider("Execution-ready score", min_score, 95, max(78, min_score))
    max_stop_pct = st.slider("Maksimum jarak SL", 3.0, 12.0, 8.0, 0.5) / 100
    min_rr2 = st.slider("Minimum RR ke TP2", 2.0, 4.0, 2.5, 0.1)
    validate = st.checkbox(
        "Expanding-window OOS validation",
        value=True,
        help="Membangun ulang plan live pada tanggal historis, mensimulasikan conditional fill, biaya, dan test folds setelah initial train window.",
    )
    use_fundamentals = st.checkbox("Ambil fundamental top candidates", value=True)
    fundamental_gate = st.checkbox("Aktifkan fundamental hard gate", value=True, disabled=not use_fundamentals)
    fundamental_n = st.slider("Jumlah kandidat fundamental", 10, 100, 50, disabled=not use_fundamentals)
    st.divider()
    st.caption("Scanner bersifat bullish-only. RISK_OFF/unknown, ARA chase, data stale, atau likuiditas rendah tidak dapat berstatus execution-ready.")

sample_csv = b"ticker\nADRO\nANTM\nBRMS\nMDKA\nTAPG\n"
status_template = (
    b"ticker,as_of,suspended,special_monitoring,fca,special_notation,corporate_action,sharia,source_url\n"
    b"ADRO,2026-07-13,false,false,false,,false,true,https://www.idx.co.id/\n"
)
news_template = (
    b"ticker,reviewed_at,review_status,title,event_date,sentiment,materiality,verified,source_url\n"
    b"ADRO,2026-07-13,COMPLETE,,,,LOW,false,\n"
)
broksum_template = (
    b"ticker,date,broker_code,buy_value,sell_value\n"
    b"ADRO,2026-07-13,XX,0,0\n"
)
left, right = st.columns([3, 1])
with left:
    uploaded = st.file_uploader(
        "Upload CSV ticker IDX",
        type=["csv", "txt"],
        help=(
            "Kolom boleh bernama ticker/symbol/kode/emiten, atau gunakan kolom pertama. "
            ".JK ditambahkan otomatis; maksimal 1.200 kode agar seluruh universe IDX dapat dimuat."
        ),
    )
with right:
    st.write("")
    st.write("")
    st.download_button("Unduh contoh CSV", sample_csv, "sample_tickers.csv", "text/csv", use_container_width=True)

with st.expander("Context files untuk real-money gate", expanded=real_money_mode):
    st.caption("Gunakan snapshot resmi/hasil review terbaru. File kosong tidak dianggap coverage.")
    market_status_upload = st.file_uploader(
        "Market status IDX CSV", type=["csv"], key="market_status_upload"
    )
    news_review_upload = st.file_uploader(
        "News & catalyst review CSV", type=["csv"], key="news_review_upload"
    )
    broksum_upload = st.file_uploader(
        "Broker summary export CSV (opsional)", type=["csv"], key="broksum_upload"
    )
    d1, d2, d3 = st.columns(3)
    d1.download_button("Template status", status_template, "market_status_template.csv", "text/csv")
    d2.download_button("Template news", news_template, "news_review_template.csv", "text/csv")
    d3.download_button("Template broksum", broksum_template, "broker_summary_template.csv", "text/csv")

run = st.button("Jalankan scanner", type="primary", use_container_width=True, disabled=uploaded is None)

if run and uploaded is not None:
    try:
        tickers = parse_ticker_csv(uploaded)
    except Exception as exc:
        st.error(f"CSV tidak dapat dibaca: {exc}")
        st.stop()
    if not tickers:
        st.error("Tidak menemukan ticker yang valid di CSV.")
        st.stop()
    cfg = ScanConfig().replace(
        min_adtv_idr=float(min_adtv_b) * 1_000_000_000,
        min_score=float(min_score),
        execution_score=float(execution_score),
        max_stop_pct=float(max_stop_pct),
        min_rr2=float(min_rr2),
        fundamental_top_n=int(fundamental_n),
        real_money_mode=bool(real_money_mode),
        require_fundamentals=bool(real_money_mode or fundamental_gate),
        require_market_status=bool(real_money_mode),
        require_news_review=bool(real_money_mode),
        account_size_idr=float(account_size),
        risk_per_trade_pct=float(risk_per_trade),
        max_positions=int(max_positions),
        max_position_pct=float(max_position_pct),
    )
    try:
        market_status = (
            parse_market_status_csv(market_status_upload)
            if market_status_upload is not None
            else pd.DataFrame()
        )
        news_review = (
            parse_news_review_csv(news_review_upload)
            if news_review_upload is not None
            else pd.DataFrame()
        )
        broksum = (
            parse_broker_summary_csv(broksum_upload)
            if broksum_upload is not None
            else pd.DataFrame()
        )
    except Exception as exc:
        st.error(f"Context CSV tidak valid: {exc}")
        st.stop()
    progress = st.progress(0, text=f"Mengunduh OHLCV {len(tickers)} ticker dan IHSG…")
    histories, report, benchmark = cached_market_data(tuple(tickers), period)
    progress.progress(35, text="Menghitung indikator, struktur pasar, dan empat setup…")
    result = ScanEngine(cfg).scan(histories, benchmark)
    signals = result["signals"]
    stats = pd.DataFrame()
    trades = pd.DataFrame()
    if validate and result["prepared"]:
        progress.progress(50, text="Menjalankan expanding-window OOS validation dengan live-plan parity…")
        stats, trades = run_walkforward_validation(result["prepared"], cfg)
        signals = attach_backtest_stats(signals, stats)
    else:
        signals = attach_backtest_stats(signals, stats)
    fundamentals = pd.DataFrame()
    if use_fundamentals and not signals.empty:
        progress.progress(72, text="Mengambil fundamental seluruh execution candidate dan ranking teratas…")
        ranked_names = signals.sort_values(
            ["status_rank", "quality_score"], ascending=[True, False]
        )["ticker"].drop_duplicates().tolist()
        execution_names = signals.loc[
            signals["status"].eq("EXECUTION_READY"), "ticker"
        ].drop_duplicates().tolist()
        top_names = list(dict.fromkeys(execution_names + ranked_names[:fundamental_n]))
        fundamentals = cached_fundamentals(tuple(top_names))
        signals = attach_fundamentals(signals, fundamentals)
    else:
        signals = attach_fundamentals(signals, fundamentals)
    if fundamental_gate or real_money_mode:
        signals = apply_fundamental_gate(signals, cfg)
    progress.progress(84, text="Menerapkan status IDX, news/catalyst, broker summary, dan sizing…")
    signals = apply_market_status_gate(signals, market_status, cfg, result.get("asof"))
    signals = apply_news_gate(signals, news_review, cfg, result.get("asof"))
    signals = attach_broker_summary(signals, broksum)
    signals = attach_position_sizing(signals, cfg)
    signals = sort_signals(signals)
    result["signals"] = signals
    result["validation_stats"] = stats
    result["validation_trades"] = trades
    result["fundamentals"] = fundamentals
    result["market_status"] = market_status
    result["news_review"] = news_review
    result["broker_summary"] = broksum
    result["download_report"] = report
    st.session_state["scan_result"] = result
    progress.progress(100, text="Scan selesai")
    progress.empty()

if "scan_result" not in st.session_state:
    st.markdown(
        """
        <div class="scanner-note">
          <b>Cara kerja singkat</b><br>
          Upload ticker → unduh OHLCV → cek regime & liquidity → deteksi setup → hitung level pada fraksi harga IDX → validasi historis → keluarkan order plan.<br>
          <span class="small-muted">Entry selalu conditional. WATCHLIST_ENTRY bukan instruksi beli.</span>
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

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Ticker valid", len(result["prepared"]))
m2.metric("Setup terdeteksi", len(signals))
m3.metric("Execution ready", int(signals["status"].eq("EXECUTION_READY").sum()) if not signals.empty else 0)
m4.metric("Watchlist", int(signals["status"].eq("WATCHLIST_ENTRY").sum()) if not signals.empty else 0)
m5.metric("Breadth > EMA50", f"{context.breadth_ema50:.0f}%" if context.breadth_ema50 is not None else "N/A")

tab_orders, tab_chart, tab_validation, tab_audit, tab_method = st.tabs(
    ["Setup & Order Builder", "Chart", "Validation", "Audit Universe", "Metodologi"]
)

with tab_orders:
    if signals.empty:
        st.warning("Tidak ada setup yang valid. Audit Universe menjelaskan alasan per ticker.")
    else:
        execution = signals[signals["status"].eq("EXECUTION_READY")]
        if not execution.empty:
            st.subheader("Order Builder — execution-ready")
            st.caption("Gunakan trigger setelah pasar buka normal; batalkan bila gap membuka di atas entry zone/target atau level validasi berubah.")
            result_table(execution)
        st.subheader("Semua setup valid dan watchlist")
        result_table(signals)
        st.caption("*P(fill) dan P(TP1 sebelum SL) hanya ditampilkan jika minimal 30 filled OOS events. Interval dan sample tetap wajib dibaca; order-book slippage aktual belum tersedia.")
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
        st.info("Aktifkan expanding-window OOS validation sebelum menjalankan scan.")
    else:
        st.subheader("Expanding-window out-of-sample validation")
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
        ### Hirarki sinyal

        1. **Tradeability gate:** data segar, ADTV, hari tanpa transaksi, ATR, dan anti-ARA chase.
        2. **Market regime:** IHSG plus breadth universe. RISK_OFF/unknown tidak boleh execution-ready.
        3. **Evidence layer:** tren, momentum 3–12 bulan, proximity to 52-week high, profitability/quality bila data tersedia.
        4. **Timing layer:** pullback continuation, breakout-retest, reversal setelah CHOCH, atau ICT sweep–BOS–FVG.
        5. **Risk engine:** level dibulatkan ke fraksi IDX, SL struktural, TP resistance-confirmed/minimum-R, expiry, distance, dan stale-zone rejection.
        6. **Real-money context:** fundamental fail-closed, status IDX/FCA/notasi, review berita, optional broker summary, lalu lot sizing setelah fee/slippage.

        **Batas data:** OHLCV/CMF/OBV hanyalah proxy akumulasi. Broker summary hanya digunakan bila CSV dilampirkan dan tetap bukan identitas beneficial owner. Scanner tidak memiliki antrean bid-offer atau spread riil. Fundamental Yahoo dapat kosong/terlambat; coverage yang kurang akan memblokir execution-ready dalam real-money mode.

        **Prinsip eksekusi:** `EXECUTION_READY` berarti seluruh data yang diwajibkan tersedia dan semua gate lolos saat snapshot—bukan jaminan profit. `WATCHLIST_ENTRY` berarti setup ada tetapi trigger/context belum lengkap. Periksa order book Stockbit sebelum mengirim limit order dan jangan mengejar zona yang sudah lewat.
        """
    )
