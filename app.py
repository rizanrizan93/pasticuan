"""IDX Momentum / Trend Scanner for Streamlit

Features
- Paste tickers manually or upload CSV (100-300+ rows)
- Auto-convert IDX tickers to Yahoo Finance format (.JK) by default
- Uses trend-following + Minervini-style template + CAN SLIM-lite leadership filters
- Scores each stock and ranks candidates
- Optional comparison against IHSG / benchmark index

Install
    pip install streamlit pandas numpy yfinance plotly openpyxl

Run
    streamlit run idx_momentum_scanner_streamlit.py

Notes
- This scanner is intentionally selective. Empty results are acceptable in weak markets.
- IDX fundamental data can be patchy via free data sources, so the core logic is price/volume based.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


# =========================
# Page config
# =========================
st.set_page_config(
    page_title="IDX Momentum Scanner",
    page_icon="📈",
    layout="wide",
)


# =========================
# Helpers
# =========================

def clean_ticker(text: str) -> str:
    """Normalize ticker input."""
    if text is None:
        return ""
    t = str(text).strip().upper()
    t = re.sub(r"\s+", "", t)
    t = t.replace("/", "-")
    return t


def to_yahoo_ticker(ticker: str, assume_idx: bool = True) -> str:
    """Convert a raw ticker into a Yahoo Finance ticker.

    Examples
    --------
    BBCA -> BBCA.JK
    BBCA.JK -> BBCA.JK
    ^JKSE -> ^JKSE
    """
    t = clean_ticker(ticker)
    if not t:
        return ""
    if t.startswith("^"):
        return t
    if "." in t:
        return t
    if assume_idx:
        return f"{t}.JK"
    return t


def parse_tickers_text(text: str, assume_idx: bool = True) -> List[str]:
    if not text:
        return []
    raw = re.split(r"[\n,;\t ]+", text)
    tickers = []
    seen = set()
    for item in raw:
        t = clean_ticker(item)
        if not t:
            continue
        yt = to_yahoo_ticker(t, assume_idx=assume_idx)
        if yt not in seen:
            tickers.append(yt)
            seen.add(yt)
    return tickers


def guess_ticker_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "ticker", "tickers", "symbol", "symbols", "kode", "kode_saham", "stock",
        "stocks", "saham", "code", "emiten", "issuer"
    ]
    lower_cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_cols:
            return lower_cols[c]
    return None


def extract_tickers_from_df(df: pd.DataFrame, column: str, assume_idx: bool = True) -> List[str]:
    if column not in df.columns:
        return []
    tickers = []
    seen = set()
    for v in df[column].astype(str).fillna("").tolist():
        t = clean_ticker(v)
        if not t or t in {"NAN", "NONE", "NULL"}:
            continue
        yt = to_yahoo_ticker(t, assume_idx=assume_idx)
        if yt not in seen:
            tickers.append(yt)
            seen.add(yt)
    return tickers


def chunked(seq: Sequence[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield list(seq[i:i + n])


@st.cache_data(ttl=60 * 30, show_spinner=False)
def download_history_batch(tickers: tuple[str, ...], period: str = "2y") -> pd.DataFrame:
    """Download OHLCV for a ticker batch with yfinance.

    Returns a wide dataframe with MultiIndex columns when multiple tickers are provided.
    """
    data = yf.download(
        tickers=list(tickers),
        period=period,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
        prepost=False,
    )
    return data


@st.cache_data(ttl=60 * 30, show_spinner=False)
def download_single_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = yf.download(
        tickers=ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        prepost=False,
        threads=False,
    )
    return df


def fetch_history(tickers: List[str], period: str = "2y", batch_size: int = 25) -> dict[str, pd.DataFrame]:
    """Fetch history for each ticker and return a dictionary of dataframes."""
    out: dict[str, pd.DataFrame] = {}
    if not tickers:
        return out

    for batch in chunked(tickers, batch_size):
        try:
            data = download_history_batch(tuple(batch), period=period)
        except Exception:
            data = pd.DataFrame()

        # Multi-ticker download returns MultiIndex columns.
        if isinstance(data.columns, pd.MultiIndex):
            for t in batch:
                if t in data.columns.get_level_values(0):
                    sub = data[t].copy()
                    sub = sub.dropna(how="all")
                    if not sub.empty:
                        out[t] = sub
                else:
                    # fallback single request
                    try:
                        sub = download_single_history(t, period=period)
                        if not sub.empty:
                            out[t] = sub.dropna(how="all")
                    except Exception:
                        pass
        else:
            # Single ticker batch or failed shape
            if len(batch) == 1:
                t = batch[0]
                sub = data.copy()
                if not sub.empty:
                    out[t] = sub.dropna(how="all")
            else:
                for t in batch:
                    try:
                        sub = download_single_history(t, period=period)
                        if not sub.empty:
                            out[t] = sub.dropna(how="all")
                    except Exception:
                        pass
    return out


def safe_last(series: pd.Series, n: int = 1):
    if series is None or series.empty or len(series) < n:
        return np.nan
    return series.iloc[-n]


def pct_change_over(series: pd.Series, n: int) -> float:
    if series is None or len(series.dropna()) <= n:
        return np.nan
    recent = series.dropna()
    a = recent.iloc[-1]
    b = recent.iloc[-(n + 1)]
    if b == 0 or pd.isna(a) or pd.isna(b):
        return np.nan
    return (a / b - 1) * 100


def compute_indicators(df: pd.DataFrame, benchmark: Optional[pd.DataFrame] = None) -> Optional[dict]:
    """Compute scanner metrics for one stock history dataframe."""
    if df is None or df.empty:
        return None

    d = df.copy()
    d = d.dropna(how="all")
    if d.empty or "Close" not in d.columns:
        return None

    # standardize columns if needed
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in d.columns:
            d[col] = np.nan

    d = d.dropna(subset=["Close"])
    if len(d) < 220:
        # Need enough bars for 200-day trend logic.
        return None

    close = d["Close"]
    volume = d["Volume"] if "Volume" in d.columns else pd.Series(index=d.index, dtype=float)

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma150 = close.rolling(150).mean()
    sma200 = close.rolling(200).mean()

    avg_vol20 = volume.rolling(20).mean()
    dollar_vol20 = (close * volume).rolling(20).mean()

    high_252 = close.rolling(252).max()
    low_252 = close.rolling(252).min()
    breakout_20 = close > close.rolling(20).max().shift(1)

    last_close = float(close.iloc[-1])
    last_vol = float(volume.iloc[-1]) if not pd.isna(safe_last(volume)) else np.nan
    last_sma20 = float(safe_last(sma20))
    last_sma50 = float(safe_last(sma50))
    last_sma150 = float(safe_last(sma150))
    last_sma200 = float(safe_last(sma200))
    last_avg_vol20 = float(safe_last(avg_vol20))
    last_dollar_vol20 = float(safe_last(dollar_vol20))
    last_high_252 = float(safe_last(high_252))
    last_low_252 = float(safe_last(low_252))

    # benchmark relative strength
    rs_6m = np.nan
    rs_12m = np.nan
    rs_line_above_50 = np.nan
    if benchmark is not None and not benchmark.empty and "Close" in benchmark.columns:
        b = benchmark.copy().dropna(subset=["Close"])
        if len(b) >= 220:
            common = pd.concat([close.rename("stock"), b["Close"].rename("bench")], axis=1).dropna()
            if len(common) >= 220:
                stock_6m = pct_change_over(common["stock"], 126)
                bench_6m = pct_change_over(common["bench"], 126)
                stock_12m = pct_change_over(common["stock"], 252)
                bench_12m = pct_change_over(common["bench"], 252)
                rs_6m = stock_6m - bench_6m if pd.notna(stock_6m) and pd.notna(bench_6m) else np.nan
                rs_12m = stock_12m - bench_12m if pd.notna(stock_12m) and pd.notna(bench_12m) else np.nan
                rs_series = common["stock"] / common["bench"]
                rs_sma50 = rs_series.rolling(50).mean()
                if len(rs_series.dropna()) >= 51 and not pd.isna(safe_last(rs_sma50)):
                    rs_line_above_50 = float(safe_last(rs_series) / safe_last(rs_sma50) - 1) * 100

    # core returns
    ret_20d = pct_change_over(close, 20)
    ret_50d = pct_change_over(close, 50)
    ret_126d = pct_change_over(close, 126)
    ret_252d = pct_change_over(close, 252)

    # breakout / distance metrics
    dist_52w_high = (last_close / last_high_252 - 1) * 100 if last_high_252 and not pd.isna(last_high_252) else np.nan
    dist_52w_low = (last_close / last_low_252 - 1) * 100 if last_low_252 and not pd.isna(last_low_252) else np.nan
    dist_sma50 = (last_close / last_sma50 - 1) * 100 if last_sma50 and not pd.isna(last_sma50) else np.nan
    dist_sma150 = (last_close / last_sma150 - 1) * 100 if last_sma150 and not pd.isna(last_sma150) else np.nan
    dist_sma200 = (last_close / last_sma200 - 1) * 100 if last_sma200 and not pd.isna(last_sma200) else np.nan
    vol_ratio = last_vol / last_avg_vol20 if last_avg_vol20 and not pd.isna(last_avg_vol20) else np.nan

    # Minervini-style trend template
    cond_price_above_50 = last_close > last_sma50 if pd.notna(last_sma50) else False
    cond_price_above_150 = last_close > last_sma150 if pd.notna(last_sma150) else False
    cond_price_above_200 = last_close > last_sma200 if pd.notna(last_sma200) else False
    cond_50_above_150 = last_sma50 > last_sma150 if pd.notna(last_sma50) and pd.notna(last_sma150) else False
    cond_150_above_200 = last_sma150 > last_sma200 if pd.notna(last_sma150) and pd.notna(last_sma200) else False
    cond_200_rising = sma200.iloc[-1] > sma200.iloc[-20] if len(sma200.dropna()) >= 20 else False
    cond_near_high = dist_52w_high >= -15 if pd.notna(dist_52w_high) else False
    cond_breakout = bool(breakout_20.iloc[-1]) if len(breakout_20.dropna()) else False

    # Liquidity / demand filters
    cond_liquid = bool(last_dollar_vol20 >= 5_000_000) if pd.notna(last_dollar_vol20) else False
    cond_volume_surge = bool(vol_ratio >= 1.5) if pd.notna(vol_ratio) else False

    # Score build-up
    score = 0

    # Trend: 40 points
    score += 8 if cond_price_above_50 else 0
    score += 8 if cond_price_above_150 else 0
    score += 8 if cond_price_above_200 else 0
    score += 6 if cond_50_above_150 else 0
    score += 4 if cond_150_above_200 else 0
    score += 4 if cond_200_rising else 0
    score += 2 if cond_near_high else 0

    # Leadership / relative strength: 20 points
    if pd.notna(rs_6m):
        score += 8 if rs_6m > 0 else 0
        score += 4 if rs_6m > 10 else 0
    if pd.notna(rs_12m):
        score += 4 if rs_12m > 0 else 0
        score += 4 if rs_12m > 15 else 0
    if pd.notna(rs_line_above_50):
        score += 4 if rs_line_above_50 > 0 else 0

    # Demand / volume: 20 points
    score += 10 if cond_volume_surge else 0
    score += 4 if pd.notna(vol_ratio) and vol_ratio >= 1.2 else 0
    score += 6 if cond_breakout else 0

    # Liquidity / tradability: 10 points
    if pd.notna(last_dollar_vol20):
        if last_dollar_vol20 >= 25_000_000:
            score += 10
        elif last_dollar_vol20 >= 10_000_000:
            score += 7
        elif last_dollar_vol20 >= 5_000_000:
            score += 4
        elif last_dollar_vol20 >= 2_000_000:
            score += 2

    # Conservative penalty for weakness
    if pd.notna(ret_252d) and ret_252d < 0:
        score -= 5
    if pd.notna(dist_52w_high) and dist_52w_high < -25:
        score -= 5

    score = max(0, min(100, int(round(score))))

    # Final label
    if score >= 80 and cond_price_above_50 and cond_price_above_150 and cond_price_above_200 and cond_50_above_150:
        label = "BUY WATCH"
    elif score >= 65:
        label = "SETUP"
    elif score >= 50:
        label = "WEAK LEADER"
    else:
        label = "PASS"

    return {
        "last_close": last_close,
        "last_vol": last_vol,
        "avg_vol20": last_avg_vol20,
        "dollar_vol20": last_dollar_vol20,
        "sma20": last_sma20,
        "sma50": last_sma50,
        "sma150": last_sma150,
        "sma200": last_sma200,
        "dist_sma50_pct": dist_sma50,
        "dist_sma150_pct": dist_sma150,
        "dist_sma200_pct": dist_sma200,
        "dist_52w_high_pct": dist_52w_high,
        "dist_52w_low_pct": dist_52w_low,
        "ret_20d_pct": ret_20d,
        "ret_50d_pct": ret_50d,
        "ret_126d_pct": ret_126d,
        "ret_252d_pct": ret_252d,
        "rs_6m_vs_benchmark_pct": rs_6m,
        "rs_12m_vs_benchmark_pct": rs_12m,
        "rs_line_vs_sma50_pct": rs_line_above_50,
        "vol_ratio": vol_ratio,
        "cond_price_above_50": cond_price_above_50,
        "cond_price_above_150": cond_price_above_150,
        "cond_price_above_200": cond_price_above_200,
        "cond_50_above_150": cond_50_above_150,
        "cond_150_above_200": cond_150_above_200,
        "cond_200_rising": cond_200_rising,
        "cond_near_high": cond_near_high,
        "cond_breakout": cond_breakout,
        "cond_liquid": cond_liquid,
        "cond_volume_surge": cond_volume_surge,
        "score": score,
        "label": label,
    }


def score_to_badge(score: int) -> str:
    if score >= 80:
        return "🟢"
    if score >= 65:
        return "🟡"
    if score >= 50:
        return "🟠"
    return "🔴"


def make_chart(stock_df: pd.DataFrame, ticker: str):
    d = stock_df.copy().dropna(subset=["Close"]) if stock_df is not None else pd.DataFrame()
    if d.empty:
        return None
    d["SMA50"] = d["Close"].rolling(50).mean()
    d["SMA150"] = d["Close"].rolling(150).mean()
    d["SMA200"] = d["Close"].rolling(200).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d.index, y=d["Close"], name="Close", mode="lines"))
    fig.add_trace(go.Scatter(x=d.index, y=d["SMA50"], name="SMA50", mode="lines"))
    fig.add_trace(go.Scatter(x=d.index, y=d["SMA150"], name="SMA150", mode="lines"))
    fig.add_trace(go.Scatter(x=d.index, y=d["SMA200"], name="SMA200", mode="lines"))
    fig.update_layout(
        title=f"{ticker} Price & Trend",
        height=480,
        xaxis_title="Date",
        yaxis_title="Price",
        legend_orientation="h",
    )
    return fig


def build_results_table(results: List[dict]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    columns = [
        "ticker", "score", "label", "last_close", "ret_20d_pct", "ret_50d_pct", "ret_126d_pct", "ret_252d_pct",
        "dist_52w_high_pct", "dist_sma50_pct", "dist_sma150_pct", "dist_sma200_pct",
        "vol_ratio", "dollar_vol20", "rs_6m_vs_benchmark_pct", "rs_12m_vs_benchmark_pct",
        "cond_price_above_50", "cond_price_above_150", "cond_price_above_200",
        "cond_50_above_150", "cond_150_above_200", "cond_200_rising", "cond_near_high", "cond_breakout",
    ]
    for c in columns:
        if c not in df.columns:
            df[c] = np.nan
    df = df[columns]
    return df.sort_values(["score", "dollar_vol20"], ascending=[False, False]).reset_index(drop=True)


# =========================
# UI
# =========================
st.title("📈 IDX Momentum / Trend Scanner")
st.caption(
    "Scanner sederhana untuk saham IHSG/IDX dengan logika trend following, Minervini-style template, dan CAN SLIM-lite."
)

with st.sidebar:
    st.header("Input saham")
    input_mode = st.radio(
        "Mode scan",
        ["Paste ticker", "Upload CSV"],
        index=0,
    )
    assume_idx = st.checkbox("Anggap ticker tanpa suffix sebagai IDX (.JK)", value=True)
    use_benchmark = st.checkbox("Bandingkan dengan benchmark", value=True)
    benchmark_ticker = st.text_input("Benchmark ticker", value="^JKSE")

    st.divider()
    st.header("Filter")
    min_score = st.slider("Minimum score", 0, 100, 65, 1)
    min_dollar_volume = st.number_input("Min avg dollar volume 20D", min_value=0, value=5_000_000, step=500_000)
    max_dist_high = st.slider("Maks jarak ke 52w high (%)", -50, 0, -15, 1)
    min_vol_ratio = st.slider("Min volume ratio vs avg20", 0.5, 5.0, 1.5, 0.1)
    require_breakout = st.checkbox("Wajib breakout 20D", value=False)

    st.divider()
    st.header("Periode data")
    period = st.selectbox("Lookback", ["1y", "2y", "3y"], index=1)
    batch_size = st.slider("Batch size download", 5, 50, 25, 5)

st.subheader("Daftar ticker")
manual_text = ""
uploaded_df = None
csv_ticker_col = None

if input_mode == "Paste ticker":
    manual_text = st.text_area(
        "Masukkan ticker dipisah koma / spasi / baris baru",
        value="BBCA, BBRI, BMRI, ASII, TLKM",
        height=120,
    )
else:
    uploaded = st.file_uploader("Upload CSV berisi 100-300 saham atau lebih", type=["csv"])
    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
            st.write("Preview CSV")
            st.dataframe(uploaded_df.head(10), use_container_width=True)
            guessed = guess_ticker_column(uploaded_df)
            options = list(uploaded_df.columns)
            default_idx = options.index(guessed) if guessed in options else 0
            csv_ticker_col = st.selectbox("Pilih kolom ticker", options, index=default_idx)
        except Exception as e:
            st.error(f"Gagal membaca CSV: {e}")

scan_button = st.button("Scan sekarang", type="primary")

if scan_button:
    # Build ticker list
    tickers: List[str] = []
    if input_mode == "Paste ticker":
        tickers = parse_tickers_text(manual_text, assume_idx=assume_idx)
    else:
        if uploaded_df is not None and csv_ticker_col is not None:
            tickers = extract_tickers_from_df(uploaded_df, csv_ticker_col, assume_idx=assume_idx)

    tickers = [t for t in tickers if t]
    tickers = list(dict.fromkeys(tickers))

    if not tickers:
        st.warning("Belum ada ticker valid untuk discan.")
        st.stop()

    if use_benchmark:
        bench = to_yahoo_ticker(benchmark_ticker, assume_idx=False)
    else:
        bench = None

    st.info(f"Ticker valid: {len(tickers)}")

    with st.spinner("Mengunduh data harga dan menghitung skor..."):
        stock_histories = fetch_history(tickers, period=period, batch_size=batch_size)
        benchmark_history = None
        if use_benchmark and bench:
            try:
                benchmark_history = download_single_history(bench, period=period)
            except Exception:
                benchmark_history = None

        results = []
        for ticker in tickers:
            hist = stock_histories.get(ticker)
            if hist is None or hist.empty:
                continue
            metrics = compute_indicators(hist, benchmark=benchmark_history)
            if metrics is None:
                continue
            metrics["ticker"] = ticker
            metrics["badge"] = score_to_badge(metrics["score"])
            if metrics["dollar_vol20"] is not None and pd.notna(metrics["dollar_vol20"]):
                if metrics["dollar_vol20"] < min_dollar_volume:
                    continue
            if pd.notna(metrics["dist_52w_high_pct"]) and metrics["dist_52w_high_pct"] < max_dist_high:
                continue
            if pd.notna(metrics["vol_ratio"]) and metrics["vol_ratio"] < min_vol_ratio:
                continue
            if require_breakout and not bool(metrics["cond_breakout"]):
                continue
            if metrics["score"] < min_score:
                continue
            results.append(metrics)

    results_df = build_results_table(results)

    if results_df.empty:
        st.error("Tidak ada saham yang lolos filter. Coba turunkan threshold atau longgarkan kondisi pasar.")
        st.stop()

    display_df = results_df.copy()
    display_df.insert(0, "signal", display_df["score"].apply(lambda x: f"{score_to_badge(int(x))} {int(x)}"))

    rename_map = {
        "ticker": "Ticker",
        "signal": "Signal",
        "score": "Score",
        "label": "Label",
        "last_close": "Last",
        "ret_20d_pct": "Ret 20D %",
        "ret_50d_pct": "Ret 50D %",
        "ret_126d_pct": "Ret 6M %",
        "ret_252d_pct": "Ret 12M %",
        "dist_52w_high_pct": "Dist 52W High %",
        "dist_sma50_pct": "Dist SMA50 %",
        "dist_sma150_pct": "Dist SMA150 %",
        "dist_sma200_pct": "Dist SMA200 %",
        "vol_ratio": "Vol Ratio",
        "dollar_vol20": "Avg $Vol20",
        "rs_6m_vs_benchmark_pct": "RS 6M vs Bench %",
        "rs_12m_vs_benchmark_pct": "RS 12M vs Bench %",
    }
    display_df = display_df.rename(columns=rename_map)

    st.subheader("Hasil scan")
    st.dataframe(
        display_df[
            [
                "Signal", "Ticker", "Label", "Last", "Score",
                "Ret 20D %", "Ret 50D %", "Ret 6M %", "Ret 12M %",
                "Dist 52W High %", "Dist SMA50 %", "Dist SMA150 %", "Dist SMA200 %",
                "Vol Ratio", "Avg $Vol20", "RS 6M vs Bench %", "RS 12M vs Bench %",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download hasil CSV",
        data=csv_bytes,
        file_name="idx_momentum_scanner_results.csv",
        mime="text/csv",
    )

    st.subheader("Detail chart")
    selected = st.selectbox("Pilih ticker", display_df["Ticker"].tolist())
    if selected:
        hist = stock_histories.get(selected)
        if hist is not None and not hist.empty:
            fig = make_chart(hist, selected)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

            last_row = results_df.loc[results_df["ticker"] == selected].iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Score", int(last_row["score"]))
            c2.metric("Label", last_row["label"])
            c3.metric("Price", f"{last_row['last_close']:.2f}")
            c4.metric("Vol Ratio", f"{last_row['vol_ratio']:.2f}" if pd.notna(last_row["vol_ratio"]) else "n/a")

            st.markdown("**Checklist utama**")
            checklist = pd.DataFrame([
                ["Price > SMA50", bool(last_row["cond_price_above_50"])],
                ["Price > SMA150", bool(last_row["cond_price_above_150"])],
                ["Price > SMA200", bool(last_row["cond_price_above_200"])],
                ["SMA50 > SMA150", bool(last_row["cond_50_above_150"])],
                ["SMA150 > SMA200", bool(last_row["cond_150_above_200"])],
                ["SMA200 rising", bool(last_row["cond_200_rising"])],
                ["Near 52W high", bool(last_row["cond_near_high"])],
                ["Breakout 20D", bool(last_row["cond_breakout"])],
                ["Volume surge", bool(last_row["cond_volume_surge"])],
            ], columns=["Rule", "Pass"])
            st.dataframe(checklist, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(
        "**Interpretasi cepat:**\n"
        "- **BUY WATCH** = kandidat terbaik, tetapi tetap butuh validasi entry.\n"
        "- **SETUP** = mulai kuat, belum selalu siap beli.\n"
        "- **WEAK LEADER** = bisa dipantau, tapi belum ideal.\n"
        "- **PASS** = jangan dipaksa."
    )

else:
    st.info("Masukkan ticker lalu klik **Scan sekarang**.")

st.markdown("---")
st.caption(
    "Versi awal scanner. Untuk hasil terbaik, pakai universe saham liquid, benchmark aktif, dan jangan paksa sinyal saat IHSG lemah."
)
