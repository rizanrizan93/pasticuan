from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

from scanner.backtest import attach_backtest_stats, run_walkforward_validation
from scanner.charts import make_signal_chart
from scanner.config import ScanConfig
from scanner.data import download_benchmark, download_ohlcv, parse_ticker_csv
from scanner.engine import ScanEngine
from scanner.fundamentals import attach_fundamentals, fetch_fundamentals


st.set_page_config(page_title="IDX Super Scanner", page_icon="🎯", layout="wide")
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
    return histories, report, benchmark


@st.cache_data(ttl=21_600, show_spinner=False)
def cached_fundamentals(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_fundamentals(tickers)


def rupiah(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"Rp{float(value):,.0f}".replace(",", ".")


def apply_fundamental_gate(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty or "fundamental_score" not in signals:
        return signals
    out = signals.copy()
    usable = out["fundamental_coverage"].fillna(0) >= 45
    red = usable & (
        (out["fundamental_score"].fillna(100) < 45)
        | out["fundamental_red_flags"].fillna("").str.contains("OCF negatif|DER tinggi", regex=True)
    )
    downgrade = red & out["status"].eq("EXECUTION_READY")
    out.loc[downgrade, "status"] = "WATCHLIST_ENTRY"
    for idx in out.index[red]:
        prior = str(out.at[idx, "blockers"] or "").strip()
        message = "Fundamental hard gate gagal"
        out.at[idx, "blockers"] = f"{prior} • {message}" if prior else message
        prior_count = pd.to_numeric(out.at[idx, "blocker_count"], errors="coerce")
        out.at[idx, "blocker_count"] = int(prior_count) + 1 if pd.notna(prior_count) else 1
    out["status_rank"] = out["status"].map({"EXECUTION_READY": 0, "WATCHLIST_ENTRY": 1, "REJECT": 2})
    return out


def prepare_display(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()
    if "historical_events" in out:
        out["probability_estimate"] = out["bayes_probability"].where(out["historical_events"] >= 20)
    else:
        out["probability_estimate"] = np.nan
    columns = [
        "ticker",
        "status",
        "setup",
        "grade",
        "quality_score",
        "composite_score",
        "probability_estimate",
        "historical_events",
        "last_price",
        "entry_type",
        "entry_low",
        "entry_high",
        "entry",
        "stop_loss",
        "tp1",
        "tp2",
        "rr1",
        "rr2",
        "stop_pct",
        "distance_atr",
        "volume_ratio",
        "adtv20_idr",
        "fundamental_score",
        "fundamental_coverage",
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
            "probability_estimate": st.column_config.NumberColumn("Est. hit rate*", format="%.1f%%"),
            "historical_events": st.column_config.NumberColumn("Sample", format="%d"),
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
            "valid_until": st.column_config.DatetimeColumn("Valid until", format="DD MMM YYYY"),
        },
    )


st.title("IDX Super Scanner")
st.caption("Evidence-based trend & momentum · IDX execution rules · SMC/ICT timing · fundamental quality gate")

with st.sidebar:
    st.header("Filter live money")
    period = st.selectbox("Riwayat OHLCV", ["3y", "5y", "2y"], index=0)
    min_adtv_b = st.number_input("Minimum ADTV20 (Rp miliar)", 0.1, 100.0, 2.0, 0.5)
    min_score = st.slider("Minimum quality score", 50, 90, 70)
    execution_score = st.slider("Execution-ready score", min_score, 95, max(78, min_score))
    max_stop_pct = st.slider("Maksimum jarak SL", 3.0, 15.0, 9.0, 0.5) / 100
    min_rr2 = st.slider("Minimum RR ke TP2", 1.5, 4.0, 2.0, 0.1)
    validate = st.checkbox("Walk-forward validation", value=True, help="Menggunakan sinyal historis tanpa look-ahead dan biaya transaksi.")
    use_fundamentals = st.checkbox("Ambil fundamental top candidates", value=True)
    fundamental_gate = st.checkbox("Aktifkan fundamental hard gate", value=True, disabled=not use_fundamentals)
    fundamental_n = st.slider("Jumlah kandidat fundamental", 5, 50, 20, disabled=not use_fundamentals)
    st.divider()
    st.caption("Scanner bersifat bullish-only. RISK_OFF/unknown, ARA chase, data stale, atau likuiditas rendah tidak dapat berstatus execution-ready.")

sample_csv = b"ticker\nADRO\nANTM\nBRMS\nMDKA\nTAPG\n"
left, right = st.columns([3, 1])
with left:
    uploaded = st.file_uploader(
        "Upload CSV ticker IDX",
        type=["csv", "txt"],
        help="Kolom boleh bernama ticker/symbol/kode/emiten, atau gunakan kolom pertama. .JK ditambahkan otomatis.",
    )
with right:
    st.write("")
    st.write("")
    st.download_button("Unduh contoh CSV", sample_csv, "sample_tickers.csv", "text/csv", use_container_width=True)

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
    )
    progress = st.progress(0, text=f"Mengunduh OHLCV {len(tickers)} ticker dan IHSG…")
    histories, report, benchmark = cached_market_data(tuple(tickers), period)
    progress.progress(35, text="Menghitung indikator, struktur pasar, dan empat setup…")
    result = ScanEngine(cfg).scan(histories, benchmark)
    signals = result["signals"]
    stats = pd.DataFrame()
    trades = pd.DataFrame()
    if validate and result["prepared"]:
        progress.progress(55, text="Menjalankan walk-forward event study…")
        stats, trades = run_walkforward_validation(result["prepared"], cfg)
        signals = attach_backtest_stats(signals, stats)
    else:
        signals = attach_backtest_stats(signals, stats)
    fundamentals = pd.DataFrame()
    if use_fundamentals and not signals.empty:
        progress.progress(75, text="Mengambil fundamental kandidat teratas…")
        top_names = (
            signals.sort_values(["status_rank", "quality_score"], ascending=[True, False])["ticker"]
            .drop_duplicates()
            .head(fundamental_n)
            .tolist()
        )
        fundamentals = cached_fundamentals(tuple(top_names))
        signals = attach_fundamentals(signals, fundamentals)
        if fundamental_gate:
            signals = apply_fundamental_gate(signals)
    else:
        signals = attach_fundamentals(signals, fundamentals)
    signals = sort_signals(signals)
    result["signals"] = signals
    result["validation_stats"] = stats
    result["validation_trades"] = trades
    result["fundamentals"] = fundamentals
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
        st.caption("*Estimasi hit rate hanya ditampilkan jika sampel walk-forward ≥20; bukan probabilitas pasti dan belum mencakup slippage order book.")
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
        st.info("Aktifkan Walk-forward validation sebelum menjalankan scan.")
    else:
        st.subheader("Walk-forward event study")
        st.caption("Sinyal dihitung dari informasi yang tersedia saat itu; entry next-open; target 2R; stop ATR; biaya pulang-pergi masuk; bila SL dan TP tersentuh pada candle yang sama, SL dianggap lebih dulu.")
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

with tab_method:
    st.markdown(
        """
        ### Hirarki sinyal

        1. **Tradeability gate:** data segar, ADTV, hari tanpa transaksi, ATR, dan anti-ARA chase.
        2. **Market regime:** IHSG plus breadth universe. RISK_OFF/unknown tidak boleh execution-ready.
        3. **Evidence layer:** tren, momentum 3–12 bulan, proximity to 52-week high, profitability/quality bila data tersedia.
        4. **Timing layer:** pullback continuation, breakout-retest, reversal setelah CHOCH, atau ICT sweep–BOS–FVG.
        5. **Risk engine:** level dibulatkan ke fraksi harga IDX, SL struktural, TP berbasis R, expiry, distance-to-entry, dan stale-zone rejection.

        **Batas data:** OHLCV/CMF/OBV hanyalah proxy akumulasi. Scanner tidak memiliki broker summary, antrean bid-offer, beneficial-owner flow, maupun spread riil; karena itu ia tidak mengklaim mendeteksi bandar secara pasti. Fundamental Yahoo untuk IDX dapat kosong atau terlambat—coverage selalu ditampilkan.

        **Prinsip eksekusi:** status `EXECUTION_READY` berarti semua gate saat scan lolos, bukan jaminan profit. `WATCHLIST_ENTRY` berarti setup ada tetapi trigger/retest/gate belum lengkap. Jangan memasang limit order pada entry lama jika harga sudah lebih dari 2 ATR dari zona.
        """
    )
