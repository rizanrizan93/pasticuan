from __future__ import annotations

import numpy as np
import pandas as pd


OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["Close"].shift(1)
    return pd.concat(
        [(df["High"] - df["Low"]), (df["High"] - prev).abs(), (df["Low"] - prev).abs()],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.where(loss.ne(0), 100.0).fillna(50.0)


def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    up = df["High"].diff()
    down = -df["Low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr_ = atr(df, length).replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr_
    minus_di = 100 * minus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / length, adjust=False, min_periods=length).mean().fillna(0.0)


def cmf(df: pd.DataFrame, length: int = 20) -> pd.Series:
    spread = (df["High"] - df["Low"]).replace(0, np.nan)
    multiplier = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / spread
    money_flow = multiplier.fillna(0.0) * df["Volume"]
    return money_flow.rolling(length).sum() / df["Volume"].rolling(length).sum().replace(0, np.nan)


def mfi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    raw = typical * df["Volume"]
    direction = typical.diff()
    positive = raw.where(direction > 0, 0.0).rolling(length).sum()
    negative = raw.where(direction < 0, 0.0).rolling(length).sum()
    ratio = positive / negative.replace(0, np.nan)
    return (100 - 100 / (1 + ratio)).fillna(50.0)


def obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["Close"].diff()).fillna(0.0)
    return (sign * df["Volume"]).cumsum()


def confirmed_pivot(series: pd.Series, left: int = 3, right: int = 3, mode: str = "high") -> pd.Series:
    window = left + right + 1
    roll = series.rolling(window, center=True, min_periods=window)
    raw = series.eq(roll.max()) if mode == "high" else series.eq(roll.min())
    # Place the pivot value on its confirmation bar. No future observation is
    # available to a historical signal before this shifted timestamp.
    return series.where(raw).shift(right)


def prepare_indicators(df: pd.DataFrame, benchmark: pd.DataFrame | None = None) -> pd.DataFrame:
    out = df.copy()
    for col in OHLCV:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
    out["Volume"] = out["Volume"].fillna(0.0).clip(lower=0)

    for length in (10, 20, 50, 100, 200):
        out[f"EMA{length}"] = ema(out["Close"], length)
    out["ATR14"] = atr(out, 14)
    out["ATR_PCT"] = out["ATR14"] / out["Close"].replace(0, np.nan)
    out["RSI14"] = rsi(out["Close"], 14)
    out["ADX14"] = adx(out, 14)
    macd = ema(out["Close"], 12) - ema(out["Close"], 26)
    signal = ema(macd, 9)
    out["MACD"] = macd
    out["MACD_HIST"] = macd - signal
    out["CMF20"] = cmf(out, 20)
    out["MFI14"] = mfi(out, 14)
    out["OBV"] = obv(out)
    out["OBV_SLOPE10"] = out["OBV"].diff(10) / out["Volume"].rolling(20).mean().replace(0, np.nan)

    out["VOL_MA20"] = out["Volume"].rolling(20).mean()
    out["VOL_RATIO"] = out["Volume"] / out["VOL_MA20"].replace(0, np.nan)
    out["VALUE"] = out["Close"] * out["Volume"]
    out["ADTV20"] = out["VALUE"].rolling(20).mean()
    out["ZERO_VOL20"] = out["Volume"].eq(0).rolling(20).mean()
    typical = (out["High"] + out["Low"] + out["Close"]) / 3
    out["VWAP20"] = (typical * out["Volume"]).rolling(20).sum() / out["Volume"].rolling(20).sum().replace(0, np.nan)

    for length in (20, 60, 120):
        out[f"ROC{length}"] = out["Close"].pct_change(length)
    out["HIGH20_PREV"] = out["High"].shift(1).rolling(20).max()
    out["HIGH55_PREV"] = out["High"].shift(1).rolling(55).max()
    out["HIGH252"] = out["High"].rolling(252, min_periods=120).max()
    out["LOW20_PREV"] = out["Low"].shift(1).rolling(20).min()
    out["LOW55_PREV"] = out["Low"].shift(1).rolling(55).min()
    out["DIST_52W_HIGH"] = out["Close"] / out["HIGH252"].replace(0, np.nan) - 1

    out["PIVOT_HIGH"] = confirmed_pivot(out["High"], 3, 3, "high")
    out["PIVOT_LOW"] = confirmed_pivot(out["Low"], 3, 3, "low")
    out["LAST_PIVOT_HIGH"] = out["PIVOT_HIGH"].ffill()
    out["LAST_PIVOT_LOW"] = out["PIVOT_LOW"].ffill()

    body = (out["Close"] - out["Open"]).abs()
    candle_range = (out["High"] - out["Low"]).replace(0, np.nan)
    out["BODY_ATR"] = body / out["ATR14"].replace(0, np.nan)
    out["CLOSE_LOCATION"] = (out["Close"] - out["Low"]) / candle_range
    out["BULL_CANDLE"] = out["Close"] > out["Open"]
    out["BEAR_CANDLE"] = out["Close"] < out["Open"]
    out["BULL_REJECTION"] = (out["CLOSE_LOCATION"] > 0.65) & (out["Close"] > out["Open"])
    out["RANGE_CONTRACTION20"] = out["ATR14"] / out["ATR14"].rolling(60).median().replace(0, np.nan)

    # Bullish fair-value gap with displacement. The condition is known at t.
    out["BULL_FVG"] = (
        (out["Low"] > out["High"].shift(2))
        & (out["Close"].shift(1) > out["Open"].shift(1))
        & (out["BODY_ATR"].shift(1) >= 0.65)
        & (out["VOL_RATIO"].shift(1) >= 1.15)
    )
    out["FVG_LOW"] = out["High"].shift(2).where(out["BULL_FVG"])
    out["FVG_HIGH"] = out["Low"].where(out["BULL_FVG"])

    if benchmark is not None and not benchmark.empty and "Close" in benchmark:
        bench_close = benchmark["Close"].reindex(out.index).ffill()
        out["BENCH_CLOSE"] = bench_close
        out["BENCH_EMA50"] = ema(bench_close, 50)
        out["BENCH_EMA200"] = ema(bench_close, 200)
        out["BENCH_ROC20"] = bench_close.pct_change(20)
        out["REL_STRENGTH60"] = out["ROC60"] - bench_close.pct_change(60)
    else:
        out["BENCH_CLOSE"] = np.nan
        out["BENCH_EMA50"] = np.nan
        out["BENCH_EMA200"] = np.nan
        out["BENCH_ROC20"] = np.nan
        out["REL_STRENGTH60"] = np.nan
    return out.replace([np.inf, -np.inf], np.nan)
