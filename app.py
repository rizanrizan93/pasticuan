
# pullback_continuation_scanner_streamlit_v2.py
# Bullish pullback-continuation scanner for Streamlit.
# Focus: better entry quality for small capital, with liquidity, RS, trend, and trigger quality filters.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]


@dataclass
class ScanParams:
    period: str = "2y"
    interval: str = "1d"
    auto_adjust: bool = True

    ema_fast: int = 20
    ema_slow: int = 50
    ema_trend: int = 200

    pivot_left: int = 3
    pivot_right: int = 3
    lookback_swing: int = 40

    min_pullback_pct: float = 0.03
    max_pullback_pct: float = 0.18
    max_extension_from_pivot_pct: float = 0.02
    min_reclaim_buffer_pct: float = 0.0015
    max_bars_after_pullback_low: int = 10

    use_volume_filter: bool = True
    volume_ma: int = 20
    pullback_vol_max_mult: float = 0.90
    breakout_vol_min_mult: float = 1.10
    min_avg_volume: float = 0.0
    min_avg_turnover_idr: float = 0.0

    min_price: float = 0.0
    max_price: float = 1e18

    atr_period: int = 14
    atr_pct_min: float = 0.015
    atr_pct_max: float = 0.12
    sl_buffer_atr: float = 0.20
    tp_rr: float = 2.0

    require_close_in_upper_half: bool = True
    min_close_location_value: float = 0.65

    benchmark_symbol: str = ""
    rs_lookback: int = 20
    min_rs_outperformance_pct: float = 0.0  # percent outperformance over lookback

    min_setup_score: float = 75.0


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Flatten multiindex if present.
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = ["_".join([str(x) for x in tup if str(x) != ""]).strip() for tup in out.columns]

    rename_map = {}
    for c in out.columns:
        lc = str(c).strip().lower()
        if lc == "adj close":
            rename_map[c] = "Adj Close"
        elif lc == "volume":
            rename_map[c] = "Volume"
        elif lc == "open":
            rename_map[c] = "Open"
        elif lc == "high":
            rename_map[c] = "High"
        elif lc == "low":
            rename_map[c] = "Low"
        elif lc == "close":
            rename_map[c] = "Close"
    if rename_map:
        out = out.rename(columns=rename_map)

    for c in REQUIRED_COLS:
        if c not in out.columns:
            out[c] = np.nan

    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = _standardize_columns(df)
    if df.empty:
        return df

    for col in REQUIRED_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop broken candles.
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[(df["High"] >= df[["Open", "Close", "Low"]].max(axis=1))]
    df = df[(df["Low"] <= df[["Open", "Close", "High"]].min(axis=1))]
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0.0)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ohlcv_yfinance(symbol: str, period: str, interval: str, auto_adjust: bool) -> pd.DataFrame:
    symbol = symbol.strip()
    if not symbol:
        return pd.DataFrame()
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False,
            group_by="column",
            threads=False,
        )
    except Exception:
        return pd.DataFrame()
    return clean_ohlcv(df)


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def pivot_high(series: pd.Series, left: int, right: int) -> pd.Series:
    window = left + right + 1
    return series.eq(series.rolling(window=window, center=True, min_periods=window).max())


def pivot_low(series: pd.Series, left: int, right: int) -> pd.Series:
    window = left + right + 1
    return series.eq(series.rolling(window=window, center=True, min_periods=window).min())


def clv(df: pd.DataFrame) -> pd.Series:
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    return ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / rng


def compute_rs_score(close: pd.Series, benchmark_close: pd.Series, lookback: int) -> pd.Series:
    ratio = close / benchmark_close.replace(0, np.nan)
    rs = ratio / ratio.shift(lookback) - 1.0
    return rs


def compute_signal_score(flags: Dict[str, bool], pullback_depth: float, atr_pct: float, rs_pct: float) -> float:
    score = 0.0
    weights = {
        "trend": 22.0,
        "structure": 18.0,
        "reclaim": 18.0,
        "volume": 12.0,
        "liquidity": 12.0,
        "quality": 8.0,
        "rs": 10.0,
    }
    for k, w in weights.items():
        if flags.get(k, False):
            score += w

    # Prefer moderate pullbacks.
    if 0.04 <= pullback_depth <= 0.10:
        score += 5.0
    elif 0.10 < pullback_depth <= 0.15:
        score += 2.5

    # Prefer not-too-dead and not-too-wild ranges.
    if 0.02 <= atr_pct <= 0.06:
        score += 3.0
    elif 0.06 < atr_pct <= 0.09:
        score += 1.5

    # Reward positive relative strength.
    if rs_pct >= 0.0:
        score += min(5.0, rs_pct * 40.0)
    else:
        score += max(-5.0, rs_pct * 40.0)

    return max(0.0, min(100.0, score))


def scan_pullback_continuation(df: pd.DataFrame, params: ScanParams, benchmark_df: pd.DataFrame | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), df

    data = df.copy()
    data["EMA_FAST"] = ema(data["Close"], params.ema_fast)
    data["EMA_SLOW"] = ema(data["Close"], params.ema_slow)
    data["EMA_TREND"] = ema(data["Close"], params.ema_trend)
    data["ATR"] = atr(data, params.atr_period)
    data["VOL_MA"] = data["Volume"].rolling(params.volume_ma, min_periods=params.volume_ma).mean()
    data["PIVOT_HIGH"] = pivot_high(data["High"], params.pivot_left, params.pivot_right)
    data["PIVOT_LOW"] = pivot_low(data["Low"], params.pivot_left, params.pivot_right)
    data["CLV"] = clv(data)

    if benchmark_df is not None and not benchmark_df.empty:
        b = benchmark_df.copy()
        b["Close"] = pd.to_numeric(b["Close"], errors="coerce")
        b = b.dropna(subset=["Close"]).sort_index()
        merged = pd.DataFrame(index=data.index)
        merged["Close"] = data["Close"]
        merged["BenchmarkClose"] = b["Close"].reindex(data.index).ffill()
        data["RS"] = compute_rs_score(merged["Close"], merged["BenchmarkClose"], params.rs_lookback)
    else:
        data["RS"] = np.nan

    signals: List[Dict] = []
    start_idx = max(params.ema_trend, params.volume_ma, params.lookback_swing, params.pivot_left + params.pivot_right + 10)

    rows = data.reset_index()
    index_col = rows.columns[0]

    for i in range(start_idx, len(rows)):
        row = rows.iloc[i]
        close = float(row["Close"])
        high = float(row["High"])
        low = float(row["Low"])
        vol = float(row["Volume"]) if pd.notna(row["Volume"]) else 0.0

        if not np.isfinite(close) or close <= 0:
            continue
        if close < params.min_price or close > params.max_price:
            continue

        atr_val = float(row["ATR"]) if pd.notna(row["ATR"]) else np.nan
        if not np.isfinite(atr_val) or atr_val <= 0:
            continue

        atr_pct = atr_val / close
        if atr_pct < params.atr_pct_min or atr_pct > params.atr_pct_max:
            continue

        ema_fast_val = float(row["EMA_FAST"]) if pd.notna(row["EMA_FAST"]) else np.nan
        ema_slow_val = float(row["EMA_SLOW"]) if pd.notna(row["EMA_SLOW"]) else np.nan
        ema_trend_val = float(row["EMA_TREND"]) if pd.notna(row["EMA_TREND"]) else np.nan
        vol_ma_val = float(row["VOL_MA"]) if pd.notna(row["VOL_MA"]) else np.nan
        clv_val = float(row["CLV"]) if pd.notna(row["CLV"]) else np.nan
        rs_val = float(row["RS"]) if pd.notna(row["RS"]) else np.nan

        if not np.isfinite(ema_fast_val) or not np.isfinite(ema_slow_val) or not np.isfinite(ema_trend_val):
            continue

        avg_vol_20 = float(rows["Volume"].iloc[max(0, i - params.volume_ma):i].mean())
        avg_turnover = float((rows["Close"].iloc[max(0, i - params.volume_ma):i] * rows["Volume"].iloc[max(0, i - params.volume_ma):i]).mean())

        if params.min_avg_volume > 0 and avg_vol_20 < params.min_avg_volume:
            continue
        if params.min_avg_turnover_idr > 0 and avg_turnover < params.min_avg_turnover_idr:
            continue

        # Trend quality: stacked EMAs + price above trend.
        trend_ok = (
            close > ema_trend_val
            and ema_fast_val > ema_slow_val > ema_trend_val
            and ema_fast_val > data["EMA_FAST"].iloc[max(0, i - 5)]
            and ema_slow_val > data["EMA_SLOW"].iloc[max(0, i - 5)]
        )
        if not trend_ok:
            continue

        # Pullback must come after a valid pivot high.
        recent = rows.iloc[max(0, i - params.lookback_swing):i]
        pivots_high = recent[recent["PIVOT_HIGH"] == True]
        if pivots_high.empty:
            continue

        last_pivot_high = pivots_high.iloc[-1]
        pivot_high_idx = int(last_pivot_high.name)
        pivot_high_price = float(last_pivot_high["High"])

        after_pivot = rows.iloc[pivot_high_idx + 1:i + 1]
        if after_pivot.empty:
            continue

        low_idx = int(after_pivot["Low"].idxmin())
        pullback_low_row = rows.loc[low_idx]
        pullback_low = float(pullback_low_row["Low"])
        if low_idx <= pivot_high_idx:
            continue

        retrace_depth = (pivot_high_price - pullback_low) / pivot_high_price
        retrace_ok = params.min_pullback_pct <= retrace_depth <= params.max_pullback_pct
        if not retrace_ok:
            continue

        # Support quality: pullback should respect fast/slow EMA area.
        support_ref = min(ema_fast_val, ema_slow_val)
        support_touch_ok = pullback_low <= support_ref * 1.01
        if not support_touch_ok:
            continue

        # Avoid chasing too far above the pivot high.
        extension_from_pivot = (close - pivot_high_price) / pivot_high_price
        if extension_from_pivot > params.max_extension_from_pivot_pct:
            continue

        signal_age = i - low_idx
        if signal_age > params.max_bars_after_pullback_low:
            continue

        reclaim_level = pivot_high_price * (1.0 + params.min_reclaim_buffer_pct)
        reclaim_ok = close >= reclaim_level

        # Candle quality on the trigger bar.
        candle_range = max(high - low, 1e-9)
        close_location_value = (close - low) / candle_range
        quality_ok = True
        if params.require_close_in_upper_half:
            quality_ok = quality_ok and (close_location_value >= params.min_close_location_value)
        quality_ok = quality_ok and np.isfinite(clv_val) and clv_val >= 0.4

        # Volume pattern: contraction on pullback, expansion on trigger.
        pullback_slice = rows.iloc[pivot_high_idx + 1:low_idx + 1]
        breakout_slice = rows.iloc[max(0, i - 2):i + 1]
        pullback_vol_mean = float(pullback_slice["Volume"].mean()) if not pullback_slice.empty else np.nan
        breakout_vol_mean = float(breakout_slice["Volume"].mean()) if not breakout_slice.empty else np.nan
        volume_ok = True
        if params.use_volume_filter:
            if np.isfinite(vol_ma_val) and np.isfinite(pullback_vol_mean):
                volume_ok = volume_ok and (pullback_vol_mean <= params.pullback_vol_max_mult * vol_ma_val)
            if np.isfinite(vol_ma_val) and np.isfinite(breakout_vol_mean):
                volume_ok = volume_ok and (breakout_vol_mean >= params.breakout_vol_min_mult * vol_ma_val)

        rs_pct = 0.0
        rs_ok = True
        if np.isfinite(rs_val):
            rs_pct = rs_val
            rs_ok = rs_val >= params.min_rs_outperformance_pct

        flags = {
            "trend": trend_ok,
            "structure": support_touch_ok and retrace_ok,
            "reclaim": reclaim_ok,
            "volume": volume_ok,
            "liquidity": True,
            "quality": quality_ok,
            "rs": rs_ok,
        }

        score = compute_signal_score(flags, retrace_depth, atr_pct, rs_pct)
        if score < params.min_setup_score:
            continue

        entry = close
        stop = min(pullback_low, ema_slow_val) - params.sl_buffer_atr * atr_val
        risk = max(entry - stop, 1e-9)
        target = entry + params.tp_rr * risk

        signals.append(
            {
                "Date": row[index_col],
                "TickerScore": round(score, 1),
                "Entry": round(entry, 4),
                "Stop": round(stop, 4),
                "Target": round(target, 4),
                "Risk": round(risk, 4),
                "RR": round((target - entry) / risk, 2),
                "PivotHigh": round(pivot_high_price, 4),
                "PullbackLow": round(pullback_low, 4),
                "RetracePct": round(retrace_depth * 100.0, 2),
                "ATRpct": round(atr_pct * 100.0, 2),
                "RSpct": round(rs_pct * 100.0, 2) if np.isfinite(rs_pct) else np.nan,
                "SignalAgeBars": int(signal_age),
                "CloseLocationValue": round(close_location_value, 2),
                "AvgVol20": round(avg_vol_20, 2),
                "AvgTurnover20": round(avg_turnover, 2),
                "VolumeOK": bool(volume_ok),
                "ReclaimOK": bool(reclaim_ok),
                "QualityOK": bool(quality_ok),
            }
        )

    signals_df = pd.DataFrame(signals)
    if not signals_df.empty:
        signals_df = signals_df.sort_values(["TickerScore", "RR"], ascending=[False, False]).reset_index(drop=True)

    return signals_df, data


def parse_tickers(text: str) -> List[str]:
    raw = [x.strip().upper() for x in text.replace("\n", ",").split(",")]
    tickers = []
    for t in raw:
        if not t:
            continue
        if "." not in t and not t.startswith("^"):
            t = f"{t}.JK"
        tickers.append(t)
    return list(dict.fromkeys(tickers))


def load_tickers_from_csv(uploaded_file) -> List[str]:
    if uploaded_file is None:
        return []
    df = pd.read_csv(uploaded_file)
    for col in df.columns:
        if col.lower() in {"ticker", "symbol", "code", "saham"}:
            vals = df[col].astype(str).str.strip().str.upper().tolist()
            return parse_tickers(",".join(vals))
    return []


def load_ohlcv_from_csv(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    df = pd.read_csv(uploaded_file)
    for c in df.columns:
        if str(c).lower() in {"date", "datetime", "time"}:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            df = df.dropna(subset=[c]).set_index(c)
            break
    return clean_ohlcv(df)


def app():
    st.set_page_config(page_title="Pullback Continuation Scanner", layout="wide")
    st.title("Pullback Continuation Scanner")
    st.caption("Bullish continuation scanner tuned for better entry quality on small capital.")

    with st.sidebar:
        st.header("Data")
        source_mode = st.radio("Source", ["YFinance", "Upload OHLCV CSV"], index=0)
        period = st.selectbox("Period", ["6mo", "1y", "2y", "5y", "max"], index=2)
        interval = st.selectbox("Interval", ["1d", "1wk", "1h", "30m", "15m", "5m"], index=0)
        auto_adjust = st.checkbox("Auto-adjust prices", value=True)
        benchmark_symbol = st.text_input("Benchmark symbol (optional)", value="")
        benchmark_period = st.selectbox("Benchmark period", ["6mo", "1y", "2y", "5y", "max"], index=2)

        st.header("Universe")
        ticker_text = st.text_area("Tickers", value="BBCA, BMRI, BBRI, TLKM, ASII", height=110)
        ticker_csv = st.file_uploader("Upload ticker CSV", type=["csv"])
        ohlcv_csv = None
        if source_mode == "Upload OHLCV CSV":
            ohlcv_csv = st.file_uploader("Upload OHLCV CSV", type=["csv"])

        st.header("Entry quality")
        min_price = st.number_input("Min price", min_value=0.0, value=0.0, step=1.0)
        max_price = st.number_input("Max price", min_value=0.0, value=1_000_000.0, step=100.0)
        min_setup_score = st.slider("Minimum setup score", 0.0, 100.0, 78.0, 1.0)

        st.header("Liquidity for small capital")
        min_avg_volume = st.number_input("Min avg volume", min_value=0.0, value=100000.0, step=10000.0)
        min_avg_turnover_idr = st.number_input("Min avg turnover (IDR)", min_value=0.0, value=1000000000.0, step=50000000.0)

        st.header("Trend")
        ema_fast = st.number_input("EMA fast", min_value=2, max_value=200, value=20, step=1)
        ema_slow = st.number_input("EMA slow", min_value=5, max_value=300, value=50, step=1)
        ema_trend = st.number_input("EMA trend", min_value=20, max_value=400, value=200, step=1)
        lookback_swing = st.number_input("Swing lookback", min_value=10, max_value=120, value=40, step=1)

        st.header("Pullback / trigger")
        min_pullback_pct = st.slider("Min pullback %", 0.0, 0.30, 0.03, 0.005)
        max_pullback_pct = st.slider("Max pullback %", 0.01, 0.60, 0.18, 0.005)
        max_extension_from_pivot_pct = st.slider("Max extension above pivot %", 0.0, 0.10, 0.02, 0.001)
        min_reclaim_buffer_pct = st.slider("Reclaim buffer %", 0.0, 0.02, 0.0015, 0.0005)
        max_bars_after_pullback_low = st.number_input("Max bars after low", min_value=1, max_value=30, value=10, step=1)

        st.header("Volume / volatility")
        use_volume_filter = st.checkbox("Use volume filter", value=True)
        volume_ma = st.number_input("Volume MA", min_value=5, max_value=100, value=20, step=1)
        pullback_vol_max_mult = st.slider("Pullback vol max xMA", 0.2, 2.0, 0.9, 0.05)
        breakout_vol_min_mult = st.slider("Breakout vol min xMA", 0.2, 3.0, 1.1, 0.05)
        atr_pct_min = st.slider("Min ATR %", 0.0, 0.20, 0.015, 0.005)
        atr_pct_max = st.slider("Max ATR %", 0.01, 0.40, 0.12, 0.005)
        sl_buffer_atr = st.slider("Stop buffer ATR", 0.0, 2.0, 0.20, 0.05)
        tp_rr = st.slider("Take profit RR", 0.5, 5.0, 2.0, 0.1)

        st.header("Relative strength")
        rs_lookback = st.number_input("RS lookback", min_value=5, max_value=120, value=20, step=1)
        min_rs_outperformance_pct = st.slider("Min RS outperformance %", -0.10, 0.30, 0.00, 0.005)

        run = st.button("Scan", type="primary")

    params = ScanParams(
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        ema_fast=int(ema_fast),
        ema_slow=int(ema_slow),
        ema_trend=int(ema_trend),
        pivot_left=3,
        pivot_right=3,
        lookback_swing=int(lookback_swing),
        min_pullback_pct=float(min_pullback_pct),
        max_pullback_pct=float(max_pullback_pct),
        max_extension_from_pivot_pct=float(max_extension_from_pivot_pct),
        min_reclaim_buffer_pct=float(min_reclaim_buffer_pct),
        max_bars_after_pullback_low=int(max_bars_after_pullback_low),
        use_volume_filter=bool(use_volume_filter),
        volume_ma=int(volume_ma),
        pullback_vol_max_mult=float(pullback_vol_max_mult),
        breakout_vol_min_mult=float(breakout_vol_min_mult),
        min_avg_volume=float(min_avg_volume),
        min_avg_turnover_idr=float(min_avg_turnover_idr),
        min_price=float(min_price),
        max_price=float(max_price),
        atr_pct_min=float(atr_pct_min),
        atr_pct_max=float(atr_pct_max),
        sl_buffer_atr=float(sl_buffer_atr),
        tp_rr=float(tp_rr),
        benchmark_symbol=benchmark_symbol.strip().upper(),
        rs_lookback=int(rs_lookback),
        min_rs_outperformance_pct=float(min_rs_outperformance_pct),
        min_setup_score=float(min_setup_score),
    )

    if not run:
        st.info("Set the parameters and scan.")
        return

    if source_mode == "Upload OHLCV CSV":
        df = load_ohlcv_from_csv(ohlcv_csv)
        if df.empty:
            st.error("Upload a valid OHLCV CSV with Date/Datetime and Open, High, Low, Close, Volume columns.")
            return
        bench_df = pd.DataFrame()
        if params.benchmark_symbol:
            bench_df = fetch_ohlcv_yfinance(params.benchmark_symbol, benchmark_period, interval, auto_adjust)
        signals_df, enriched = scan_pullback_continuation(df, params, bench_df if not bench_df.empty else None)

        if signals_df.empty:
            st.warning("No setups found.")
            st.dataframe(enriched.tail(50), use_container_width=True)
            return

        st.subheader("Signals")
        st.dataframe(signals_df, use_container_width=True, hide_index=True)
        st.download_button("Download signals", signals_df.to_csv(index=False).encode("utf-8"), "signals.csv", "text/csv")
        st.line_chart(enriched[["Close", "EMA_FAST", "EMA_SLOW", "EMA_TREND"]].tail(200))
        return

    tickers = load_tickers_from_csv(ticker_csv) if ticker_csv else []
    if not tickers:
        tickers = parse_tickers(ticker_text)

    if not tickers:
        st.error("No tickers found.")
        return

    bench_df = pd.DataFrame()
    if params.benchmark_symbol:
        bench_df = fetch_ohlcv_yfinance(params.benchmark_symbol, benchmark_period, interval, auto_adjust)

    all_rows: List[Dict] = []
    data_cache: Dict[str, pd.DataFrame] = {}
    empty_tickers: List[str] = []

    progress = st.progress(0)
    status = st.empty()

    for idx, ticker in enumerate(tickers, start=1):
        status.write(f"Processing {ticker} ({idx}/{len(tickers)})")
        df = fetch_ohlcv_yfinance(ticker, params.period, params.interval, params.auto_adjust)
        data_cache[ticker] = df

        if df.empty or len(df) < max(params.ema_trend, params.volume_ma, params.lookback_swing) + 20:
            empty_tickers.append(ticker)
            progress.progress(idx / len(tickers))
            continue

        signals_df, enriched = scan_pullback_continuation(df, params, bench_df if not bench_df.empty else None)
        if not signals_df.empty:
            top = signals_df.iloc[0].to_dict()
            top["Ticker"] = ticker
            top["Bars"] = len(enriched)
            top["LastClose"] = float(enriched["Close"].iloc[-1])
            top["AvgVol20"] = float(enriched["Volume"].tail(20).mean()) if len(enriched) >= 20 else float(enriched["Volume"].mean())
            top["AvgTurnover20"] = float((enriched["Close"].tail(20) * enriched["Volume"].tail(20)).mean()) if len(enriched) >= 20 else float((enriched["Close"] * enriched["Volume"]).mean())
            all_rows.append(top)

        progress.progress(idx / len(tickers))

    progress.empty()
    status.empty()

    result = pd.DataFrame(all_rows)
    if result.empty:
        st.warning("No setups found with the current settings.")
        if empty_tickers:
            st.caption("Skipped / insufficient data: " + ", ".join(empty_tickers))
        return

    result = result.sort_values(["TickerScore", "RR"], ascending=[False, False]).reset_index(drop=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Setups found", len(result))
    c2.metric("Scanned", len(tickers))
    c3.metric("Top score", f"{result['TickerScore'].max():.1f}")
    c4.metric("Median RR", f"{result['RR'].median():.2f}")

    st.subheader("Ranked signals")
    show_cols = [
        "Ticker", "TickerScore", "RR", "Entry", "Stop", "Target", "RetracePct", "ATRpct",
        "RSpct", "CloseLocationValue", "SignalAgeBars", "AvgTurnover20", "LastClose", "Bars"
    ]
    existing_cols = [c for c in show_cols if c in result.columns]
    st.dataframe(result[existing_cols], use_container_width=True, hide_index=True)

    st.download_button(
        "Download ranked signals",
        result.to_csv(index=False).encode("utf-8"),
        file_name="pullback_continuation_ranked_signals.csv",
        mime="text/csv",
    )

    st.subheader("Detail")
    selected = st.selectbox("Ticker", result["Ticker"].tolist())
    selected_df = data_cache.get(selected, pd.DataFrame())
    if not selected_df.empty:
        signals_df, enriched = scan_pullback_continuation(selected_df, params, bench_df if not bench_df.empty else None)
        st.line_chart(enriched[["Close", "EMA_FAST", "EMA_SLOW", "EMA_TREND"]].tail(200))
        st.dataframe(signals_df.head(10), use_container_width=True, hide_index=True)

    if empty_tickers:
        st.caption("Insufficient / failed fetch: " + ", ".join(empty_tickers))


if __name__ == "__main__":
    app()
