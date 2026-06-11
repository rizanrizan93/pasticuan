"""IDX Momentum / Trend Scanner for Streamlit.

A simple, selective scanner for Indonesian stocks using:
- trend-following / Minervini-style template
- CAN SLIM-lite leadership and volume filters
- optional benchmark comparison (e.g. ^JKSE)
- paste tickers or upload CSV

Run:
    streamlit run app.py

Dependencies:
    streamlit
    pandas
    numpy
    yfinance

Notes:
- This scanner is intentionally selective. Empty results are valid.
- If benchmark data is unavailable or rate-limited, the scanner still works.
"""

from __future__ import annotations

import re
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="IDX Momentum Scanner", page_icon="📈", layout="wide")

# -----------------------------
# Ticker helpers
# -----------------------------
def clean_ticker(value: str) -> str:
    t = str(value or "").strip().upper()
    t = re.sub(r"\s+", "", t)
    return t


def to_yahoo_ticker(ticker: str, assume_idx: bool = True) -> str:
    t = clean_ticker(ticker)
    if not t:
        return ""
    if t.startswith("^"):
        return t
    if "." in t:
        return t
    return f"{t}.JK" if assume_idx else t


def parse_tickers_text(text: str, assume_idx: bool = True) -> List[str]:
    if not text:
        return []
    raw_items = re.split(r"[\n,;\t ]+", text)
    out: List[str] = []
    seen = set()
    for item in raw_items:
        t = clean_ticker(item)
        if not t:
            continue
        yt = to_yahoo_ticker(t, assume_idx=assume_idx)
        if yt and yt not in seen:
            out.append(yt)
            seen.add(yt)
    return out


def guess_ticker_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "ticker", "tickers", "symbol", "symbols", "kode", "kode_saham",
        "saham", "stock", "stocks", "code", "emiten"
    ]
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def extract_tickers_from_df(df: pd.DataFrame, column: str, assume_idx: bool = True) -> List[str]:
    if df is None or df.empty or column not in df.columns:
        return []
    out: List[str] = []
    seen = set()
    for val in df[column].astype(str).fillna(""):
        t = clean_ticker(val)
        if not t or t in {"NAN", "NONE", "NULL"}:
            continue
        yt = to_yahoo_ticker(t, assume_idx=assume_idx)
        if yt and yt not in seen:
            out.append(yt)
            seen.add(yt)
    return out


def chunked(seq: Sequence[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


# -----------------------------
# Data download helpers
# -----------------------------
def normalize_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # yfinance can return MultiIndex columns or lowercase-ish variants
    if isinstance(out.columns, pd.MultiIndex):
        # not expected for single-ticker download, but handle it anyway
        if "Close" in out.columns.get_level_values(0):
            out = out["Close"].copy()
        else:
            # try the first level with a ticker subframe
            out = out.copy()

    rename_map = {}
    for col in out.columns:
        c = str(col).strip()
        if c.lower() == "adj close":
            continue
        if c.lower() == "open":
            rename_map[col] = "Open"
        elif c.lower() == "high":
            rename_map[col] = "High"
        elif c.lower() == "low":
            rename_map[col] = "Low"
        elif c.lower() == "close":
            rename_map[col] = "Close"
        elif c.lower() == "volume":
            rename_map[col] = "Volume"
    out = out.rename(columns=rename_map)

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in out.columns:
            out[col] = np.nan

    out = out[["Open", "High", "Low", "Close", "Volume"]].copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["Close"])
    return out


@st.cache_data(ttl=60 * 20, show_spinner=False)
def download_single_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    try:
        df = yf.download(
            tickers=ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception:
        return pd.DataFrame()
    return normalize_ohlcv_frame(df)


@st.cache_data(ttl=60 * 20, show_spinner=False)
def download_batch_history(tickers: Tuple[str, ...], period: str = "2y") -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    if not tickers:
        return out

    try:
        raw = yf.download(
            tickers=list(tickers),
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="ticker",
        )
    except Exception:
        raw = pd.DataFrame()

    if raw is not None and not raw.empty and isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            try:
                if t in raw.columns.get_level_values(0):
                    sub = raw[t].copy()
                    sub = normalize_ohlcv_frame(sub)
                    if not sub.empty:
                        out[t] = sub
            except Exception:
                continue

    # Fallback per ticker for any missing items
    for t in tickers:
        if t in out:
            continue
        try:
            sub = download_single_history(t, period=period)
            if not sub.empty:
                out[t] = sub
        except Exception:
            continue

    return out


def fetch_benchmark(ticker: str, period: str = "2y") -> Optional[pd.DataFrame]:
    if not ticker:
        return None
    t = ticker.strip()
    if not t:
        return None
    df = download_single_history(t, period=period)
    return df if not df.empty else None


# -----------------------------
# Indicators & scoring
# -----------------------------
def pct_change_over(series: pd.Series, n: int) -> float:
    s = series.dropna()
    if len(s) <= n:
        return np.nan
    a = s.iloc[-1]
    b = s.iloc[-(n + 1)]
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return (a / b - 1.0) * 100.0


def compute_indicators(df: pd.DataFrame, benchmark: Optional[pd.DataFrame] = None) -> Optional[dict]:
    if df is None or df.empty or "Close" not in df.columns:
        return None

    d = normalize_ohlcv_frame(df)
    if d.empty or len(d) < 220:
        return None

    close = d["Close"]
    volume = d["Volume"].fillna(0)

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
    last_vol = float(volume.iloc[-1]) if not pd.isna(volume.iloc[-1]) else np.nan
    last_sma50 = float(sma50.iloc[-1]) if pd.notna(sma50.iloc[-1]) else np.nan
    last_sma150 = float(sma150.iloc[-1]) if pd.notna(sma150.iloc[-1]) else np.nan
    last_sma200 = float(sma200.iloc[-1]) if pd.notna(sma200.iloc[-1]) else np.nan
    last_avg_vol20 = float(avg_vol20.iloc[-1]) if pd.notna(avg_vol20.iloc[-1]) else np.nan
    last_dollar_vol20 = float(dollar_vol20.iloc[-1]) if pd.notna(dollar_vol20.iloc[-1]) else np.nan
    last_high_252 = float(high_252.iloc[-1]) if pd.notna(high_252.iloc[-1]) else np.nan
    last_low_252 = float(low_252.iloc[-1]) if pd.notna(low_252.iloc[-1]) else np.nan

    dist_sma50 = (last_close / last_sma50 - 1.0) * 100.0 if pd.notna(last_sma50) and last_sma50 != 0 else np.nan
    dist_sma150 = (last_close / last_sma150 - 1.0) * 100.0 if pd.notna(last_sma150) and last_sma150 != 0 else np.nan
    dist_sma200 = (last_close / last_sma200 - 1.0) * 100.0 if pd.notna(last_sma200) and last_sma200 != 0 else np.nan
    dist_52w_high = (last_close / last_high_252 - 1.0) * 100.0 if pd.notna(last_high_252) and last_high_252 != 0 else np.nan
    dist_52w_low = (last_close / last_low_252 - 1.0) * 100.0 if pd.notna(last_low_252) and last_low_252 != 0 else np.nan
    vol_ratio = (last_vol / last_avg_vol20) if pd.notna(last_avg_vol20) and last_avg_vol20 != 0 else np.nan

    cond_price_above_50 = pd.notna(last_sma50) and last_close > last_sma50
    cond_price_above_150 = pd.notna(last_sma150) and last_close > last_sma150
    cond_price_above_200 = pd.notna(last_sma200) and last_close > last_sma200
    cond_50_above_150 = pd.notna(last_sma50) and pd.notna(last_sma150) and last_sma50 > last_sma150
    cond_150_above_200 = pd.notna(last_sma150) and pd.notna(last_sma200) and last_sma150 > last_sma200
    cond_200_rising = sma200.iloc[-1] > sma200.iloc[-20] if len(sma200.dropna()) >= 20 else False
    cond_near_high = pd.notna(dist_52w_high) and dist_52w_high >= -15
    cond_breakout = bool(breakout_20.iloc[-1]) if len(breakout_20) else False
    cond_volume_surge = pd.notna(vol_ratio) and vol_ratio >= 1.5
    cond_liquid = pd.notna(last_dollar_vol20) and last_dollar_vol20 >= 5_000_000

    # Relative strength vs benchmark, if available
    rs_6m = np.nan
    rs_12m = np.nan
    rs_line_vs_sma50 = np.nan
    if benchmark is not None and not benchmark.empty and "Close" in benchmark.columns:
        b = normalize_ohlcv_frame(benchmark)
        if not b.empty:
            common = pd.concat([close.rename("stock"), b["Close"].rename("bench")], axis=1).dropna()
            if len(common) >= 220:
                stock_6m = pct_change_over(common["stock"], 126)
                bench_6m = pct_change_over(common["bench"], 126)
                stock_12m = pct_change_over(common["stock"], 252)
                bench_12m = pct_change_over(common["bench"], 252)
                if pd.notna(stock_6m) and pd.notna(bench_6m):
                    rs_6m = stock_6m - bench_6m
                if pd.notna(stock_12m) and pd.notna(bench_12m):
                    rs_12m = stock_12m - bench_12m

                rs = common["stock"] / common["bench"]
                rs_sma50 = rs.rolling(50).mean()
                if pd.notna(rs.iloc[-1]) and pd.notna(rs_sma50.iloc[-1]) and rs_sma50.iloc[-1] != 0:
                    rs_line_vs_sma50 = (rs.iloc[-1] / rs_sma50.iloc[-1] - 1.0) * 100.0

    ret_20d = pct_change_over(close, 20)
    ret_50d = pct_change_over(close, 50)
    ret_126d = pct_change_over(close, 126)
    ret_252d = pct_change_over(close, 252)

    score = 0
    score += 8 if cond_price_above_50 else 0
    score += 8 if cond_price_above_150 else 0
    score += 8 if cond_price_above_200 else 0
    score += 6 if cond_50_above_150 else 0
    score += 4 if cond_150_above_200 else 0
    score += 4 if cond_200_rising else 0
    score += 2 if cond_near_high else 0

    if pd.notna(rs_6m):
        score += 8 if rs_6m > 0 else 0
        score += 4 if rs_6m > 10 else 0
    if pd.notna(rs_12m):
        score += 4 if rs_12m > 0 else 0
        score += 4 if rs_12m > 15 else 0
    if pd.notna(rs_line_vs_sma50):
        score += 4 if rs_line_vs_sma50 > 0 else 0

    score += 10 if cond_volume_surge else 0
    score += 4 if pd.notna(vol_ratio) and vol_ratio >= 1.2 else 0
    score += 6 if cond_breakout else 0

    if pd.notna(last_dollar_vol20):
        if last_dollar_vol20 >= 25_000_000:
            score += 10
        elif last_dollar_vol20 >= 10_000_000:
            score += 7
        elif last_dollar_vol20 >= 5_000_000:
            score += 4
        elif last_dollar_vol20 >= 2_000_000:
            score += 2

    if pd.notna(ret_252d) and ret_252d < 0:
        score -= 5
    if pd.notna(dist_52w_high) and dist_52w_high < -25:
        score -= 5

    score = max(0, min(100, int(round(score))))

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
        "rs_line_vs_sma50_pct": rs_line_vs_sma50,
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


def badge(score: int) -> str:
    if score >= 80:
        return "🟢"
    if score >= 65:
        return "🟡"
    if score >= 50:
        return "🟠"
    return "🔴"


def scan_universe(
    tickers: List[str],
    period: str,
    benchmark_df: Optional[pd.DataFrame],
    min_score: int,
    min_dollar_volume: float,
    max_dist_high_pct: float,
    min_vol_ratio: float,
    require_breakout: bool,
    batch_size: int = 25,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    histories = download_batch_history(tuple(tickers), period=period) if tickers else {}
    results: List[dict] = []

    for t in tickers:
        hist = histories.get(t)
        if hist is None or hist.empty:
            continue
        metrics = compute_indicators(hist, benchmark=benchmark_df)
        if metrics is None:
            continue
        metrics["ticker"] = t
        metrics["signal"] = f"{badge(metrics['score'])} {metrics['score']}"

        if pd.notna(metrics["dollar_vol20"]) and metrics["dollar_vol20"] < min_dollar_volume:
            continue
        if pd.notna(metrics["dist_52w_high_pct"]) and metrics["dist_52w_high_pct"] < max_dist_high_pct:
            continue
        if pd.notna(metrics["vol_ratio"]) and metrics["vol_ratio"] < min_vol_ratio:
            continue
        if require_breakout and not bool(metrics["cond_breakout"]):
            continue
        if metrics["score"] < min_score:
            continue

        results.append(metrics)

    if not results:
        return pd.DataFrame(), histories

    df = pd.DataFrame(results).sort_values(["score", "dollar_vol20"], ascending=[False, False]).reset_index(drop=True)
    return df, histories


# -----------------------------
# UI
# -----------------------------
st.title("📈 IDX Momentum / Trend Scanner")
st.caption("Scanner sederhana, selektif, dan fokus ke leader — bukan banyak sinyal.")

with st.sidebar:
    st.header("Input")
    input_mode = st.radio("Mode scan", ["Paste ticker", "Upload CSV"], index=0)
    assume_idx = st.checkbox("Ticker tanpa suffix dianggap IDX (.JK)", value=True)
    use_benchmark = st.checkbox("Bandingkan dengan benchmark", value=True)
    benchmark_ticker = st.text_input("Benchmark ticker", value="^JKSE")

    st.divider()
    st.header("Filter")
    min_score = st.slider("Minimum score", 0, 100, 65, 1)
    min_dollar_volume = st.number_input("Min avg dollar volume 20D", min_value=0, value=5_000_000, step=500_000)
    max_dist_high_pct = st.slider("Maks jarak ke 52w high (%)", -50, 0, -15, 1)
    min_vol_ratio = st.slider("Min volume ratio vs avg20", 0.5, 5.0, 1.5, 0.1)
    require_breakout = st.checkbox("Wajib breakout 20D", value=False)
    period = st.selectbox("Lookback", ["1y", "2y", "3y"], index=1)

    st.divider()
    st.header("Download")
    st.caption("Kalau scanning banyak ticker, pakai CSV dan mulai dari 100-300 saham.")

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
    uploaded = st.file_uploader("Upload CSV berisi ticker", type=["csv"])
    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
            st.write("Preview CSV")
            st.dataframe(uploaded_df.head(10), width="stretch", hide_index=True)
            guessed = guess_ticker_column(uploaded_df)
            options = list(uploaded_df.columns)
            default_idx = options.index(guessed) if guessed in options else 0
            csv_ticker_col = st.selectbox("Pilih kolom ticker", options, index=default_idx)
        except Exception as e:
            st.error(f"Gagal membaca CSV: {e}")

scan_button = st.button("Scan sekarang", type="primary")

if scan_button:
    tickers: List[str] = []
    if input_mode == "Paste ticker":
        tickers = parse_tickers_text(manual_text, assume_idx=assume_idx)
    elif uploaded_df is not None and csv_ticker_col is not None:
        tickers = extract_tickers_from_df(uploaded_df, csv_ticker_col, assume_idx=assume_idx)

    tickers = list(dict.fromkeys([t for t in tickers if t]))

    if not tickers:
        st.warning("Belum ada ticker valid untuk discan.")
        st.stop()

    benchmark_df = None
    if use_benchmark and benchmark_ticker.strip():
        with st.spinner("Mengunduh benchmark..."):
            try:
                benchmark_df = fetch_benchmark(benchmark_ticker.strip(), period=period)
            except Exception:
                benchmark_df = None
        if benchmark_df is None or benchmark_df.empty:
            st.warning("Benchmark tidak berhasil diambil. Scanner tetap jalan tanpa benchmark.")

    with st.spinner("Mengunduh data dan menghitung skor..."):
        results_df, histories = scan_universe(
            tickers=tickers,
            period=period,
            benchmark_df=benchmark_df,
            min_score=min_score,
            min_dollar_volume=min_dollar_volume,
            max_dist_high_pct=max_dist_high_pct,
            min_vol_ratio=min_vol_ratio,
            require_breakout=require_breakout,
        )

    st.info(f"Ticker valid: {len(tickers)}")

    if results_df.empty:
        st.error("Tidak ada saham yang lolos filter. Coba turunkan threshold atau longgarkan kondisi.")
        st.stop()

    show_cols = [
        "signal", "ticker", "label", "last_close", "score",
        "ret_20d_pct", "ret_50d_pct", "ret_126d_pct", "ret_252d_pct",
        "dist_52w_high_pct", "dist_sma50_pct", "dist_sma150_pct", "dist_sma200_pct",
        "vol_ratio", "dollar_vol20", "rs_6m_vs_benchmark_pct", "rs_12m_vs_benchmark_pct",
    ]
    display_df = results_df.copy()
    display_df = display_df.rename(columns={
        "ticker": "ticker",
        "last_close": "last_close",
        "score": "score",
    })

    st.subheader("Hasil scan")
    st.dataframe(
        display_df[show_cols].rename(columns={
            "signal": "Signal",
            "ticker": "Ticker",
            "label": "Label",
            "last_close": "Last",
            "score": "Score",
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
        }),
        width="stretch",
        hide_index=True,
    )

    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download hasil CSV",
        data=csv_bytes,
        file_name="idx_momentum_scanner_results.csv",
        mime="text/csv",
    )

    st.subheader("Detail ticker")
    selected = st.selectbox("Pilih ticker", display_df["ticker"].tolist())
    if selected:
        hist = histories.get(selected)
        if hist is not None and not hist.empty:
            chart_df = pd.DataFrame({
                "Close": hist["Close"],
                "SMA50": hist["Close"].rolling(50).mean(),
                "SMA150": hist["Close"].rolling(150).mean(),
                "SMA200": hist["Close"].rolling(200).mean(),
            })
            st.line_chart(chart_df, width="stretch")

            row = results_df.loc[results_df["ticker"] == selected].iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Score", int(row["score"]))
            c2.metric("Label", row["label"])
            c3.metric("Price", f"{row['last_close']:.2f}")
            c4.metric("Vol Ratio", f"{row['vol_ratio']:.2f}" if pd.notna(row["vol_ratio"]) else "n/a")

            st.markdown("**Checklist utama**")
            checklist = pd.DataFrame([
                ["Price > SMA50", bool(row["cond_price_above_50"])],
                ["Price > SMA150", bool(row["cond_price_above_150"])],
                ["Price > SMA200", bool(row["cond_price_above_200"])],
                ["SMA50 > SMA150", bool(row["cond_50_above_150"])],
                ["SMA150 > SMA200", bool(row["cond_150_above_200"])],
                ["SMA200 rising", bool(row["cond_200_rising"])],
                ["Near 52W high", bool(row["cond_near_high"])],
                ["Breakout 20D", bool(row["cond_breakout"])],
                ["Volume surge", bool(row["cond_volume_surge"])],
                ["Liquid enough", bool(row["cond_liquid"])],
            ], columns=["Rule", "Pass"])
            st.dataframe(checklist, width="stretch", hide_index=True)

    st.markdown("---")
    st.markdown(
        "**Interpretasi cepat:**\n"
        "- **BUY WATCH** = kandidat terbaik, tetap tunggu entry yang rapi.\n"
        "- **SETUP** = mulai kuat, bisa dipantau.\n"
        "- **WEAK LEADER** = ada tenaga, tapi belum ideal.\n"
        "- **PASS** = jangan dipaksa."
    )
else:
    st.info("Masukkan ticker lalu klik **Scan sekarang**.")

st.caption("Jika benchmark tidak tersedia karena rate-limit, scanner tetap jalan dengan logika trend, volume, dan liquidity.")
