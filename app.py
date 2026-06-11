
# Free-tier optimization notes:
# - Keep universe <= 100 tickers per scan
# - Use max_workers 2-4
# - Retrain manually, not continuously
# - Prefer cached downloads


import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split


st.set_page_config(page_title="IDX Profit-First Scanner ML", layout="wide")


# =========================================================
# Helpers
# =========================================================

def clip(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, x)))


def to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and not x.strip():
            return None
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return None


def normalize_ticker(raw: str, auto_suffix: bool = True) -> str:
    t = (raw or "").strip().upper()
    if not t:
        return ""
    if t.startswith("^"):
        return t
    if "." in t:
        return t
    return f"{t}.JK" if auto_suffix else t


def detect_ticker_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["ticker", "symbol", "code", "kode", "saham", "stock", "asset"]
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    return None


def round_down_to_lot(shares: float, lot_size: int = 100) -> int:
    if shares <= 0:
        return 0
    return int(math.floor(shares / lot_size) * lot_size)


# =========================================================
# Data loading
# =========================================================

@st.cache_data(show_spinner=False, ttl=3600)
def load_history(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=False,
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    needed = ["open", "high", "low", "close", "volume"]
    if any(c not in df.columns for c in needed):
        return pd.DataFrame()

    df = df[needed].dropna()
    if df.empty:
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index)
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def load_fundamentals(ticker: str) -> Dict:
    try:
        info = yf.Ticker(ticker).get_info()
        return info if isinstance(info, dict) else {}
    except Exception:
        return {}


# =========================================================
# Indicators
# =========================================================

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1 / window, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / window, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    mu = series.rolling(window).mean()
    sigma = series.rolling(window).std(ddof=0)
    return (series - mu) / sigma.replace(0, np.nan)


def cmf(df: pd.DataFrame, window: int = 20) -> pd.Series:
    hl_range = (df["high"] - df["low"]).replace(0, np.nan)
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / hl_range
    mfv = mfm * df["volume"]
    return mfv.rolling(window).sum() / df["volume"].rolling(window).sum().replace(0, np.nan)


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["close"]
    h = out["high"]
    l = out["low"]
    v = out["volume"]

    out["ema20"] = ema(c, 20)
    out["ema50"] = ema(c, 50)
    out["ema100"] = ema(c, 100)
    out["ema200"] = ema(c, 200)

    out["sma20"] = sma(c, 20)
    out["sma50"] = sma(c, 50)
    out["sma200"] = sma(c, 200)

    out["rsi14"] = rsi(c, 14)
    out["atr14"] = atr(out, 14)
    out["atr_pct"] = out["atr14"] / c.replace(0, np.nan)
    out["atr_pct_sma20"] = sma(out["atr_pct"], 20)
    out["atr_pct_sma60"] = sma(out["atr_pct"], 60)

    out["ret1"] = c.pct_change(1)
    out["ret5"] = c.pct_change(5)
    out["ret20"] = c.pct_change(20)
    out["ret60"] = c.pct_change(60)
    out["ret252"] = c.pct_change(252)

    out["vol_sma20"] = sma(v, 20)
    out["vol_sma50"] = sma(v, 50)
    out["vol_rvol20"] = v / out["vol_sma20"].replace(0, np.nan)
    out["vol_z20"] = rolling_zscore(v, 20)
    out["ret_z20"] = rolling_zscore(out["ret1"], 20)

    out["cmf20"] = cmf(out, 20)
    out["obv"] = obv(out)
    out["obv_ema20"] = ema(out["obv"], 20)
    out["obv_ema50"] = ema(out["obv"], 50)

    out["high20"] = h.rolling(20).max()
    out["high55"] = h.rolling(55).max()
    out["low5"] = l.rolling(5).min()
    out["low20"] = l.rolling(20).min()
    out["low55"] = l.rolling(55).min()

    lookback = 22
    cmax = c.rolling(lookback).max()
    out["vixfix"] = ((cmax - l) / cmax.replace(0, np.nan)) * 100.0
    out["vixfix_ma"] = out["vixfix"].rolling(20).mean()
    out["vixfix_std"] = out["vixfix"].rolling(20).std(ddof=0)
    out["vixfix_upper"] = out["vixfix_ma"] + 2.0 * out["vixfix_std"]
    out["vixfix_panic"] = out["vixfix"] > out["vixfix_upper"]

    out["ema_stack"] = (out["ema20"] > out["ema50"]) & (out["ema50"] > out["ema100"]) & (out["ema100"] > out["ema200"])
    out["trend_stack"] = (out["ema20"] > out["ema50"]) & (out["ema50"] > out["sma200"])
    out["bull_stack"] = (c > out["ema20"]) & (out["ema20"] > out["ema50"]) & (out["ema50"] > out["ema100"])
    out["above_sma50"] = c > out["sma50"]
    out["above_sma200"] = c > out["sma200"]
    out["near_20d_high"] = c >= out["high20"] * 0.985
    out["near_55d_high"] = c >= out["high55"] * 0.975
    out["compression"] = (out["atr_pct"] < out["atr_pct_sma60"] * 0.90) & (out["atr_pct"] < out["atr_pct_sma20"])
    out["higher_low_5d"] = out["low5"] > out["low20"] * 0.985

    return out


def ticker_feature_row(df: pd.DataFrame) -> Dict[str, float]:
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    feats = {
        "close": float(last["close"]),
        "rsi14": float(last["rsi14"]),
        "atr_pct": float(last["atr_pct"]),
        "ret1": float(last["ret1"]),
        "ret5": float(last["ret5"]),
        "ret20": float(last["ret20"]),
        "ret60": float(last["ret60"]),
        "vol_rvol20": float(last["vol_rvol20"]),
        "vol_z20": float(last["vol_z20"]),
        "ret_z20": float(last["ret_z20"]),
        "cmf20": float(last["cmf20"]),
        "obv_ema20": float(last["obv_ema20"]),
        "obv_ema50": float(last["obv_ema50"]),
        "ema20": float(last["ema20"]),
        "ema50": float(last["ema50"]),
        "ema100": float(last["ema100"]),
        "ema200": float(last["ema200"]),
        "sma50": float(last["sma50"]),
        "sma200": float(last["sma200"]),
        "ema_gap20_50": float((last["ema20"] / last["ema50"]) - 1.0) if pd.notna(last["ema50"]) and last["ema50"] != 0 else np.nan,
        "ema_gap50_200": float((last["ema50"] / last["ema200"]) - 1.0) if pd.notna(last["ema200"]) and last["ema200"] != 0 else np.nan,
        "dist_high20": float((last["close"] / last["high20"]) - 1.0) if pd.notna(last["high20"]) and last["high20"] != 0 else np.nan,
        "dist_high55": float((last["close"] / last["high55"]) - 1.0) if pd.notna(last["high55"]) and last["high55"] != 0 else np.nan,
        "vixfix": float(last["vixfix"]),
        "vixfix_panic": float(bool(last["vixfix_panic"])),
        "compression": float(bool(last["compression"])),
        "above_sma50": float(bool(last["above_sma50"])),
        "above_sma200": float(bool(last["above_sma200"])),
        "ema_stack": float(bool(last["ema_stack"])),
        "trend_stack": float(bool(last["trend_stack"])),
        "bull_stack": float(bool(last["bull_stack"])),
        "near_20d_high": float(bool(last["near_20d_high"])),
        "near_55d_high": float(bool(last["near_55d_high"])),
        "higher_low_5d": float(bool(last["higher_low_5d"])),
        "rsi_above_50": float(last["rsi14"] > 50),
        "rsi_bull_zone": float(55 <= last["rsi14"] <= 78),
        "rsi_pullback_zone": float(40 <= last["rsi14"] <= 60),
        "volume_up": float(last["close"] > prev["close"]),
    }
    return feats


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema_gap20_50"] = (out["ema20"] / out["ema50"]) - 1.0
    out["ema_gap50_200"] = (out["ema50"] / out["ema200"]) - 1.0
    out["dist_high20"] = (out["close"] / out["high20"]) - 1.0
    out["dist_high55"] = (out["close"] / out["high55"]) - 1.0
    out["vixfix_panic"] = out["vixfix_panic"].astype(float)
    out["compression"] = out["compression"].astype(float)
    out["above_sma50"] = out["above_sma50"].astype(float)
    out["above_sma200"] = out["above_sma200"].astype(float)
    out["ema_stack"] = out["ema_stack"].astype(float)
    out["trend_stack"] = out["trend_stack"].astype(float)
    out["bull_stack"] = out["bull_stack"].astype(float)
    out["near_20d_high"] = out["near_20d_high"].astype(float)
    out["near_55d_high"] = out["near_55d_high"].astype(float)
    out["higher_low_5d"] = out["higher_low_5d"].astype(float)
    out["rsi_above_50"] = (out["rsi14"] > 50).astype(float)
    out["rsi_bull_zone"] = out["rsi14"].between(55, 78).astype(float)
    out["rsi_pullback_zone"] = out["rsi14"].between(40, 60).astype(float)
    out["volume_up"] = (out["close"] > out["close"].shift(1)).astype(float)
    return out


def score_market(df: pd.DataFrame) -> Tuple[float, str, Dict[str, float]]:
    if df.empty:
        return np.nan, "Unknown", {}
    last = df.iloc[-1]
    score = 0.0
    detail = {}
    conds = [
        (last["close"] > last["sma200"], 40, "above_sma200"),
        (last["ema20"] > last["ema50"], 25, "ema20_above_ema50"),
        (last["close"] > last["sma50"], 15, "above_sma50"),
        (last["ret20"] > 0, 10, "ret20_positive"),
        (45 <= last["rsi14"] <= 70, 10, "rsi_reasonable"),
    ]
    for cond, pts, key in conds:
        detail[key] = 1.0 if cond else 0.0
        if cond:
            score += pts
    score = clip(score)
    label = "RISK ON" if score >= 70 else "NEUTRAL" if score >= 50 else "RISK OFF"
    return score, label, detail


def score_liquidity(df: pd.DataFrame, min_avg_dollar_vol: float, min_price: float) -> Tuple[float, Dict[str, float]]:
    last = df.iloc[-1]
    avg_dollar_vol_20 = float((df["close"] * df["volume"]).tail(20).mean())
    avg_vol_20 = float(df["volume"].tail(20).mean())
    price = float(last["close"])
    atr_pct = float(last.get("atr_pct", np.nan))

    score = 0.0
    if np.isfinite(avg_dollar_vol_20):
        score += 40 if avg_dollar_vol_20 >= min_avg_dollar_vol else 40 * clip(avg_dollar_vol_20 / max(min_avg_dollar_vol, 1), 0, 1)
    if price >= min_price:
        score += 20
    else:
        score += 20 * clip(price / max(min_price, 1e-9), 0, 1)
    if avg_vol_20 >= 100_000:
        score += 20
    elif avg_vol_20 >= 25_000:
        score += 12
    elif avg_vol_20 >= 10_000:
        score += 6
    else:
        score += 2
    if np.isfinite(atr_pct):
        if atr_pct <= 0.04:
            score += 20
        elif atr_pct <= 0.07:
            score += 12
        elif atr_pct <= 0.10:
            score += 6
    return clip(score), {
        "avg_dollar_vol_20": avg_dollar_vol_20,
        "avg_vol_20": avg_vol_20,
        "price": price,
        "atr_pct": atr_pct,
    }


def score_quality(info: Dict) -> Tuple[float, Dict[str, float]]:
    if not info:
        return np.nan, {}

    roe = to_float(info.get("returnOnEquity"))
    pm = to_float(info.get("profitMargins"))
    om = to_float(info.get("operatingMargins"))
    de = to_float(info.get("debtToEquity"))
    rg = to_float(info.get("revenueGrowth"))
    eg = to_float(info.get("earningsGrowth"))
    mc = to_float(info.get("marketCap"))

    parts = {}
    if roe is not None:
        parts["roe"] = clip(roe * 100.0, 0, 25)
    if pm is not None:
        parts["profit_margin"] = clip(pm * 100.0, 0, 20)
    if om is not None:
        parts["operating_margin"] = clip(om * 100.0, 0, 15)
    if de is not None:
        if de <= 25:
            parts["debt"] = 15
        elif de <= 50:
            parts["debt"] = 12
        elif de <= 100:
            parts["debt"] = 7
        elif de <= 200:
            parts["debt"] = 3
        else:
            parts["debt"] = 0
    if rg is not None:
        parts["revenue_growth"] = clip(rg * 100.0, 0, 20)
    if eg is not None:
        parts["earnings_growth"] = clip(eg * 100.0, 0, 15)
    if mc is not None:
        if mc >= 10_000_000_000:
            parts["market_cap"] = 10
        elif mc >= 2_000_000_000:
            parts["market_cap"] = 7
        elif mc >= 500_000_000:
            parts["market_cap"] = 4
        else:
            parts["market_cap"] = 1

    if not parts:
        return np.nan, {}

    raw = sum(parts.values())
    max_possible = 25 + 20 + 15 + 15 + 20 + 15 + 10
    return clip((raw / max_possible) * 100.0), parts


def score_trend(df: pd.DataFrame, benchmark_df: Optional[pd.DataFrame]) -> Tuple[float, str, Dict[str, float]]:
    last = df.iloc[-1]
    score = 0.0
    detail = {}
    rs20 = np.nan
    rs60 = np.nan

    if benchmark_df is not None and not benchmark_df.empty and len(benchmark_df) >= 60:
        b = benchmark_df.iloc[-1]
        rs20 = float(last["ret20"] - b["ret20"])
        rs60 = float(last["ret60"] - b["ret60"])

    conds = [
        (last["ema_stack"], 25, "ema_stack"),
        (last["bull_stack"], 20, "bull_stack"),
        (last["close"] > last["sma200"], 10, "above_sma200"),
        (last["ret20"] > 0, 10, "ret20_positive"),
        (last["ret60"] > 0, 10, "ret60_positive"),
        (last["near_20d_high"] or last["near_55d_high"], 10, "near_high"),
        (55 <= last["rsi14"] <= 78, 10, "rsi_bull_zone"),
        (last["vol_rvol20"] >= 1.1, 5, "volume_support"),
    ]
    for cond, pts, key in conds:
        detail[key] = 1.0 if cond else 0.0
        if cond:
            score += pts

    if np.isfinite(rs20):
        detail["rs20"] = float(rs20)
        if rs20 > 0:
            score += 5
        if rs20 > 0.05:
            score += 5
    if np.isfinite(rs60):
        detail["rs60"] = float(rs60)
        if rs60 > 0:
            score += 5

    score = clip(score)
    label = "Strong Trend" if score >= 70 else "Trend" if score >= 50 else "Weak Trend"
    return score, label, detail


def score_institution_flow(df: pd.DataFrame) -> Tuple[float, Dict[str, float]]:
    last = df.iloc[-1]
    score = 0.0
    detail = {}

    rvol = float(last.get("vol_rvol20", np.nan))
    if np.isfinite(rvol):
        detail["rvol20"] = rvol
        if rvol >= 2.0:
            score += 35
        elif rvol >= 1.5:
            score += 25
        elif rvol >= 1.2:
            score += 15
        elif rvol >= 1.0:
            score += 8

    obv_trend = bool(last.get("obv_ema20", np.nan) > last.get("obv_ema50", np.nan))
    detail["obv_trend"] = 1.0 if obv_trend else 0.0
    if obv_trend:
        score += 20

    cmf20 = float(last.get("cmf20", np.nan))
    detail["cmf20"] = cmf20
    if np.isfinite(cmf20):
        if cmf20 > 0.05:
            score += 15
        elif cmf20 > 0:
            score += 10
        elif cmf20 > -0.05:
            score += 5

    last20 = df.tail(20)
    up_vol = float(last20.loc[last20["close"] > last20["close"].shift(1), "volume"].sum())
    down_vol = float(last20.loc[last20["close"] < last20["close"].shift(1), "volume"].sum())
    acc_ratio = up_vol / down_vol if down_vol > 0 else np.nan
    detail["acc_ratio"] = acc_ratio
    if np.isfinite(acc_ratio):
        if acc_ratio >= 1.5:
            score += 15
        elif acc_ratio >= 1.2:
            score += 10
        elif acc_ratio >= 1.0:
            score += 5

    return clip(score), detail


def score_setup(df: pd.DataFrame) -> Tuple[float, str, Dict[str, float]]:
    last = df.iloc[-1]
    prev = df.iloc[-2]
    breakout_score = 0.0
    pullback_score = 0.0
    detail = {}

    breakout_conds = [
        (last["near_20d_high"], 25, "near_20d_high"),
        (last["near_55d_high"], 10, "near_55d_high"),
        (last["vol_rvol20"] >= 1.25, 20, "rvol_expansion"),
        (last["vol_z20"] >= 1.0, 10, "vol_z_positive"),
        (last["compression"], 15, "compression"),
        (last["ema_stack"], 10, "ema_stack"),
        (55 <= last["rsi14"] <= 78, 10, "rsi_breakout_zone"),
    ]
    for cond, pts, key in breakout_conds:
        detail[f"breakout_{key}"] = 1.0 if cond else 0.0
        if cond:
            breakout_score += pts

    panic = bool(last["vixfix_panic"])
    rebound = bool(last["close"] > prev["close"] and last["close"] > last["low5"] * 1.01)
    pullback_conds = [
        (last["above_sma50"], 15, "above_sma50"),
        (last["ema20"] > last["ema50"], 10, "ema20_above_ema50"),
        (40 <= last["rsi14"] <= 60, 15, "rsi_pullback_zone"),
        (panic, 20, "vixfix_panic"),
        (rebound, 20, "rebound_confirmation"),
        (last["vol_rvol20"] >= 0.8, 10, "liquidity_ok"),
        (last["compression"], 10, "compression"),
    ]
    for cond, pts, key in pullback_conds:
        detail[f"pullback_{key}"] = 1.0 if cond else 0.0
        if cond:
            pullback_score += pts

    if breakout_score >= pullback_score:
        return clip(breakout_score), "Breakout", detail
    return clip(pullback_score), "Pullback", detail


def combine_scores(regime: float, trend: float, flow: float, setup: float, liquidity: float, quality: Optional[float], ml_prob: Optional[float]) -> float:
    w_quality = 0.05 if quality is not None and np.isfinite(quality) else 0.0
    w_ml = 0.15 if ml_prob is not None and np.isfinite(ml_prob) else 0.0
    weights = {
        "regime": 0.25,
        "trend": 0.20,
        "flow": 0.20,
        "setup": 0.15,
        "liquidity": 0.10,
        "quality": w_quality,
        "ml": w_ml,
    }
    total = sum(weights.values())
    value = regime * weights["regime"]
    value += trend * weights["trend"]
    value += flow * weights["flow"]
    value += setup * weights["setup"]
    value += liquidity * weights["liquidity"]
    if w_quality > 0:
        value += quality * weights["quality"]
    if w_ml > 0:
        value += (ml_prob * 100.0) * weights["ml"]
    return clip(value / total) if total > 0 else 0.0


# =========================================================
# Entry plan
# =========================================================

def build_entry_plan(df: pd.DataFrame, setup_type: str, account_size: float, risk_pct: float, lot_rounding: bool) -> Dict[str, float]:
    last = df.iloc[-1]
    atr14 = float(last["atr14"])
    close = float(last["close"])
    high20 = float(last["high20"])
    low5 = float(last["low5"])
    ema20 = float(last["ema20"])
    ema50 = float(last["ema50"])

    if setup_type == "Pullback":
        entry_trigger = max(float(df["high"].tail(5).max()), ema20 * 1.002)
        stop_loss = min(low5, ema20) - 0.8 * atr14
        invalidation = min(low5, ema20)
        target1_rr = 1.5
        target2_rr = 2.5
        note = "Wait for reclaim / rebound confirmation."
    else:
        entry_trigger = max(high20 + 0.10 * atr14, close * 1.005)
        stop_loss = min(low5, ema20) - 0.8 * atr14
        invalidation = min(low5, ema20)
        target1_rr = 1.5
        target2_rr = 3.0
        note = "Use stop-buy only after volume confirms."

    if np.isnan(stop_loss) or stop_loss <= 0:
        stop_loss = close * 0.95
    if entry_trigger <= stop_loss:
        entry_trigger = max(close, stop_loss * 1.01)

    risk_per_share = max(entry_trigger - stop_loss, 1e-9)
    risk_budget = account_size * (risk_pct / 100.0)
    raw_shares = risk_budget / risk_per_share
    shares = int(raw_shares)
    if lot_rounding:
        shares = round_down_to_lot(shares, 100)
    if shares < 100 and lot_rounding:
        shares = 0

    lots = shares // 100
    position_value = shares * entry_trigger
    total_risk = shares * risk_per_share
    target1 = entry_trigger + target1_rr * risk_per_share
    target2 = entry_trigger + target2_rr * risk_per_share

    return {
        "entry_trigger": float(entry_trigger),
        "stop_loss": float(stop_loss),
        "invalidation": float(invalidation),
        "target1": float(target1),
        "target2": float(target2),
        "risk_per_share": float(risk_per_share),
        "risk_budget": float(risk_budget),
        "shares": int(shares),
        "lots": int(lots),
        "position_value": float(position_value),
        "total_risk": float(total_risk),
        "rr_t1": float(target1_rr),
        "rr_t2": float(target2_rr),
        "setup_note": note,
        "close": float(close),
        "ema50": float(ema50),
    }


# =========================================================
# ML dataset and model
# =========================================================

FEATURE_COLUMNS = [
    "close", "rsi14", "atr_pct", "ret1", "ret5", "ret20", "ret60",
    "vol_rvol20", "vol_z20", "ret_z20", "cmf20",
    "obv_ema20", "obv_ema50", "ema20", "ema50", "ema100", "ema200",
    "sma50", "sma200", "ema_gap20_50", "ema_gap50_200",
    "dist_high20", "dist_high55", "vixfix", "vixfix_panic",
    "compression", "above_sma50", "above_sma200", "ema_stack", "trend_stack",
    "bull_stack", "near_20d_high", "near_55d_high", "higher_low_5d",
    "rsi_above_50", "rsi_bull_zone", "rsi_pullback_zone", "volume_up",
]


def generate_trade_label(df: pd.DataFrame, i: int, horizon: int, target_pct: float, stop_pct: float) -> int:
    entry = float(df["close"].iloc[i])
    target = entry * (1.0 + target_pct)
    stop = entry * (1.0 - stop_pct)
    end = min(len(df) - 1, i + horizon)

    # Sequential barrier approximation using daily high/low.
    for j in range(i + 1, end + 1):
        if float(df["low"].iloc[j]) <= stop:
            return 0
        if float(df["high"].iloc[j]) >= target:
            return 1
    return 0


def build_ml_dataset(
    tickers: List[str],
    period: str,
    horizon: int,
    target_pct: float,
    stop_pct: float,
    max_samples_total: int,
    sample_stride: int,
    min_history: int = 260,
) -> Tuple[pd.DataFrame, pd.Series]:
    rows = []

    for tkr in tickers:
        df = load_history(tkr, period=period)
        if df.empty or len(df) < min_history:
            continue
        df = build_indicators(df)
        feat_df = build_feature_matrix(df)

        usable_idx = list(range(220, len(feat_df) - horizon, max(sample_stride, 1)))
        for i in usable_idx:
            row = feat_df.iloc[i]
            if row[FEATURE_COLUMNS].isna().all():
                continue
            label = generate_trade_label(feat_df, i, horizon=horizon, target_pct=target_pct, stop_pct=stop_pct)
            sample = {c: row.get(c, np.nan) for c in FEATURE_COLUMNS}
            sample["date"] = feat_df.index[i]
            sample["ticker"] = tkr
            sample["label"] = label
            rows.append(sample)

            if max_samples_total > 0 and len(rows) >= max_samples_total:
                break
        if max_samples_total > 0 and len(rows) >= max_samples_total:
            break

    if not rows:
        return pd.DataFrame(), pd.Series(dtype=int)

    X = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    y = X.pop("label").astype(int)
    X = X.drop(columns=["ticker", "date"], errors="ignore")
    return X, y


@dataclass
class MLArtifact:
    model: object
    feature_columns: List[str]
    metrics: Dict[str, float]


def train_ml_model(X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[MLArtifact], Dict[str, float]]:
    if X.empty or len(X) < 500 or y.nunique() < 2:
        return None, {"error": "Data training belum cukup atau label hanya satu kelas."}

    # Time-aware split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=4,
        max_iter=220,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )
    model.fit(X_train[FEATURE_COLUMNS], y_train)

    prob = model.predict_proba(X_test[FEATURE_COLUMNS])[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_test, prob))
    except Exception:
        metrics["auc"] = np.nan

    artifact = MLArtifact(model=model, feature_columns=FEATURE_COLUMNS.copy(), metrics=metrics)
    return artifact, metrics


def ml_predict_prob(artifact: Optional[MLArtifact], row_dict: Dict[str, float]) -> float:
    if artifact is None:
        return np.nan
    X = pd.DataFrame([{c: row_dict.get(c, np.nan) for c in artifact.feature_columns}])
    try:
        prob = artifact.model.predict_proba(X)[0, 1]
        return float(prob)
    except Exception:
        return np.nan



def evaluate_signal_snapshot(
    hist_df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame],
    min_avg_dollar_vol: float,
    min_price: float,
    ml_artifact: Optional[MLArtifact] = None,
    ml_enabled: bool = False,
    use_gates: bool = True,
) -> Optional[Dict]:
    """
    Evaluate the scanner at a single historical bar using only data available up to that bar.
    This is used by the backtest engine and avoids lookahead.
    """
    if hist_df is None or hist_df.empty or len(hist_df) < 220:
        return None

    last = hist_df.iloc[-1]
    market_score, market_label, market_detail = score_market(benchmark_df) if benchmark_df is not None and not benchmark_df.empty else (np.nan, "Unknown", {})
    regime_score, regime_label, regime_detail = score_market(hist_df)
    trend_score, trend_label, trend_detail = score_trend(hist_df, benchmark_df)
    flow_score, flow_detail = score_institution_flow(hist_df)
    setup_score, setup_label, setup_detail = score_setup(hist_df)
    liquidity_score, liquidity_detail = score_liquidity(hist_df, min_avg_dollar_vol=min_avg_dollar_vol, min_price=min_price)

    feat_row = ticker_feature_row(hist_df)
    ml_prob = ml_predict_prob(ml_artifact, feat_row) if ml_enabled else np.nan

    final_score = combine_scores(
        regime=regime_score,
        trend=trend_score,
        flow=flow_score,
        setup=setup_score,
        liquidity=liquidity_score,
        quality=None,
        ml_prob=ml_prob if np.isfinite(ml_prob) else None,
    )

    gate_pass = (
        (market_label != "RISK OFF") if use_gates else True
    ) and (
        regime_score >= 50 if use_gates else True
    ) and (
        trend_score >= 45 if use_gates else True
    ) and (
        flow_score >= 40 if use_gates else True
    ) and (
        liquidity_score >= 45 if use_gates else True
    ) and (
        final_score >= 60 if use_gates else True
    )

    if final_score >= 85 and gate_pass:
        grade = "A+"
    elif final_score >= 75:
        grade = "A"
    elif final_score >= 65:
        grade = "B"
    else:
        grade = "C"

    if grade in {"A+", "A"} and gate_pass and (not ml_enabled or (np.isfinite(ml_prob) and ml_prob >= 0.55)):
        decision = "BUY"
    elif grade == "B":
        decision = "WATCH"
    else:
        decision = "AVOID"

    return {
        "market_score": round(float(market_score), 2) if np.isfinite(market_score) else np.nan,
        "market_label": market_label,
        "market_detail": market_detail,
        "regime_score": round(float(regime_score), 2),
        "regime_label": regime_label,
        "regime_detail": regime_detail,
        "trend_score": round(float(trend_score), 2),
        "trend_label": trend_label,
        "trend_detail": trend_detail,
        "flow_score": round(float(flow_score), 2),
        "flow_detail": flow_detail,
        "setup_score": round(float(setup_score), 2),
        "setup_label": setup_label,
        "setup_detail": setup_detail,
        "liquidity_score": round(float(liquidity_score), 2),
        "liquidity_detail": liquidity_detail,
        "ml_prob": round(float(ml_prob), 4) if np.isfinite(ml_prob) else np.nan,
        "final_score": round(float(final_score), 2),
        "grade": grade,
        "decision": decision,
        "gate_pass": gate_pass,
        "close": float(last["close"]),
        "rsi14": float(last["rsi14"]),
        "ret20": float(last["ret20"]),
        "ret60": float(last["ret60"]),
        "vol_rvol20": float(last["vol_rvol20"]),
        "vol_z20": float(last["vol_z20"]),
        "atr_pct": float(last["atr_pct"]),
        "vixfix_panic": bool(last["vixfix_panic"]),
    }


def simulate_trade_from_signal(
    df: pd.DataFrame,
    signal_idx: int,
    entry_trigger: float,
    stop_loss: float,
    target1: float,
    target2: float,
    max_hold_days: int = 20,
    entry_window_days: int = 3,
    partial_split: float = 0.5,
) -> Optional[Dict]:
    """
    Simulate a single trade from a scanner signal.
    Conservative fill order:
    1) stop
    2) target 1
    3) target 2
    If a bar hits both stop and target, stop wins.
    """
    if df is None or df.empty or signal_idx >= len(df) - 2:
        return None

    start = signal_idx + 1
    end_entry = min(len(df) - 1, signal_idx + entry_window_days)
    entry_idx = None

    for j in range(start, end_entry + 1):
        if float(df["high"].iloc[j]) >= entry_trigger:
            entry_idx = j
            break

    if entry_idx is None:
        return None

    risk_per_share = max(entry_trigger - stop_loss, 1e-9)
    r1 = (target1 - entry_trigger) / risk_per_share
    r2 = (target2 - entry_trigger) / risk_per_share

    t1_hit = False
    realized_r = 0.0
    exit_idx = min(len(df) - 1, entry_idx + max_hold_days)
    exit_reason = "time"
    trade_closed = False

    for k in range(entry_idx, exit_idx + 1):
        high = float(df["high"].iloc[k])
        low = float(df["low"].iloc[k])

        # Conservative: stop first.
        if low <= stop_loss:
            if t1_hit:
                realized_r += (1.0 - partial_split) * (-1.0)
            else:
                realized_r = -1.0
            exit_reason = "stop"
            exit_idx = k
            trade_closed = True
            break

        if not t1_hit and high >= target1:
            realized_r += partial_split * r1
            t1_hit = True

        if high >= target2:
            if t1_hit:
                realized_r += (1.0 - partial_split) * r2
            else:
                realized_r += r2
            exit_reason = "target2"
            exit_idx = k
            trade_closed = True
            break

        if k == exit_idx:
            close = float(df["close"].iloc[k])
            close_r = (close - entry_trigger) / risk_per_share
            if t1_hit:
                realized_r += (1.0 - partial_split) * close_r
            else:
                realized_r = close_r
            exit_reason = "time"
            trade_closed = True
            break

    if not trade_closed:
        return None

    return {
        "entry_idx": int(entry_idx),
        "exit_idx": int(exit_idx),
        "entry_date": df.index[entry_idx],
        "exit_date": df.index[exit_idx],
        "entry_price": float(entry_trigger),
        "stop_loss": float(stop_loss),
        "target1": float(target1),
        "target2": float(target2),
        "risk_per_share": float(risk_per_share),
        "realized_r": float(realized_r),
        "exit_reason": exit_reason,
        "t1_hit": bool(t1_hit),
        "bars_held": int(exit_idx - entry_idx + 1),
    }


def summarize_trade_list(trades: List[Dict], risk_pct: float = 1.0) -> Dict[str, float]:
    if not trades:
        return {
            "trades": 0,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "expectancy_r": np.nan,
            "max_drawdown": np.nan,
            "gross_profit_r": np.nan,
            "gross_loss_r": np.nan,
            "avg_win_r": np.nan,
            "avg_loss_r": np.nan,
            "median_r": np.nan,
            "avg_bars_held": np.nan,
            "total_return_pct": np.nan,
        }

    rs = np.array([float(t["realized_r"]) for t in trades], dtype=float)
    wins = rs[rs > 0]
    losses = rs[rs < 0]
    win_rate = float((rs > 0).mean())
    gross_profit_r = float(wins.sum()) if len(wins) else 0.0
    gross_loss_r = float(abs(losses.sum())) if len(losses) else 0.0
    profit_factor = float(gross_profit_r / gross_loss_r) if gross_loss_r > 0 else np.inf
    expectancy_r = float(rs.mean())
    avg_win_r = float(wins.mean()) if len(wins) else np.nan
    avg_loss_r = float(losses.mean()) if len(losses) else np.nan
    median_r = float(np.median(rs))
    avg_bars_held = float(np.mean([t.get("bars_held", np.nan) for t in trades]))

    # Approximate equity curve assuming fixed risk_pct per trade.
    risk_frac = max(risk_pct, 0.0) / 100.0
    equity = [1.0]
    for r in rs:
        equity.append(max(1e-9, equity[-1] * (1.0 + r * risk_frac)))
    equity = np.array(equity[1:], dtype=float)
    peaks = np.maximum.accumulate(equity)
    drawdown = np.where(peaks > 0, (peaks - equity) / peaks, 0.0)
    max_dd = float(drawdown.max()) if len(drawdown) else np.nan
    total_return_pct = float((equity[-1] - 1.0) * 100.0) if len(equity) else np.nan

    return {
        "trades": int(len(trades)),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy_r": expectancy_r,
        "max_drawdown": max_dd,
        "gross_profit_r": gross_profit_r,
        "gross_loss_r": gross_loss_r,
        "avg_win_r": avg_win_r,
        "avg_loss_r": avg_loss_r,
        "median_r": median_r,
        "avg_bars_held": avg_bars_held,
        "total_return_pct": total_return_pct,
    }


def run_rule_based_backtest(
    universe: List[str],
    period: str,
    benchmark_ticker: str,
    min_avg_dollar_vol: float,
    min_price: float,
    ml_artifact: Optional[MLArtifact],
    ml_enabled: bool,
    max_hold_days: int,
    entry_window_days: int,
    cooldown_days: int,
    risk_pct: float,
    max_signals_per_ticker: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    benchmark_df = build_benchmark(benchmark_ticker, period)
    if not benchmark_df.empty:
        benchmark_df = build_indicators(benchmark_df)

    trades = []
    signal_rows = []

    for raw_tkr in universe:
        tkr = normalize_ticker(raw_tkr, auto_suffix=True)
        df = load_history(tkr, period=period)
        if df.empty or len(df) < 260:
            continue
        df = build_indicators(df)

        i = 220
        signals_this_ticker = 0
        while i < len(df) - max_hold_days - 1:
            hist = df.iloc[: i + 1]
            bench_hist = benchmark_df.loc[: hist.index[-1]].copy() if not benchmark_df.empty else None
            snap = evaluate_signal_snapshot(
                hist_df=hist,
                benchmark_df=bench_hist,
                min_avg_dollar_vol=min_avg_dollar_vol,
                min_price=min_price,
                ml_artifact=ml_artifact,
                ml_enabled=ml_enabled,
            )
            if snap is None:
                i += 1
                continue

            if snap["decision"] == "BUY":
                plan = build_entry_plan(
                    hist,
                    snap["setup_label"],
                    account_size=100_000_000.0,
                    risk_pct=max(risk_pct, 0.1),
                    lot_rounding=False,
                )
                trade = simulate_trade_from_signal(
                    df=df,
                    signal_idx=i,
                    entry_trigger=plan["entry_trigger"],
                    stop_loss=plan["stop_loss"],
                    target1=plan["target1"],
                    target2=plan["target2"],
                    max_hold_days=max_hold_days,
                    entry_window_days=entry_window_days,
                )
                signal_rows.append(
                    {
                        "ticker": raw_tkr,
                        "ticker_norm": tkr,
                        "signal_date": hist.index[-1],
                        "final_score": snap["final_score"],
                        "grade": snap["grade"],
                        "decision": snap["decision"],
                        "setup": snap["setup_label"],
                        "ml_prob": snap["ml_prob"],
                        "entry_trigger": plan["entry_trigger"],
                        "stop_loss": plan["stop_loss"],
                        "target1": plan["target1"],
                        "target2": plan["target2"],
                    }
                )
                signals_this_ticker += 1

                if trade is not None:
                    trade.update(
                        {
                            "ticker": raw_tkr,
                            "ticker_norm": tkr,
                            "signal_date": hist.index[-1],
                            "grade": snap["grade"],
                            "decision": snap["decision"],
                            "setup": snap["setup_label"],
                            "final_score": snap["final_score"],
                            "ml_prob": snap["ml_prob"],
                        }
                    )
                    trades.append(trade)
                    i = trade["exit_idx"] + cooldown_days
                else:
                    i += 1

                if max_signals_per_ticker > 0 and signals_this_ticker >= max_signals_per_ticker:
                    break
            else:
                i += 1

    trade_df = pd.DataFrame(trades)
    metrics = summarize_trade_list(trades, risk_pct=risk_pct)
    signal_df = pd.DataFrame(signal_rows).sort_values(["signal_date", "final_score"], ascending=[False, False]) if signal_rows else pd.DataFrame()
    return trade_df, metrics, signal_df


def run_walk_forward_validation(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if X.empty or len(X) < 500 or y.nunique() < 2:
        return pd.DataFrame(), {"error": "Data tidak cukup untuk walk-forward validation."}

    n_splits = max(2, min(int(n_splits), 10))
    splitter = TimeSeriesSplit(n_splits=n_splits)

    fold_rows = []
    oof_prob = np.full(len(X), np.nan, dtype=float)

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        X_train = X.iloc[train_idx][FEATURE_COLUMNS].copy()
        y_train = y.iloc[train_idx].copy()
        X_test = X.iloc[test_idx][FEATURE_COLUMNS].copy()
        y_test = y.iloc[test_idx].copy()

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            continue

        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=4,
            max_iter=220,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=42,
        )
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        oof_prob[test_idx] = prob

        row = {
            "fold": fold,
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
        }
        try:
            row["auc"] = float(roc_auc_score(y_test, prob))
        except Exception:
            row["auc"] = np.nan

        fold_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)
    valid = ~np.isnan(oof_prob)
    summary = {}
    if valid.any():
        y_valid = y.iloc[valid]
        prob_valid = oof_prob[valid]
        pred_valid = (prob_valid >= 0.5).astype(int)
        summary = {
            "folds_used": int(len(fold_df)),
            "oof_accuracy": float(accuracy_score(y_valid, pred_valid)),
            "oof_precision": float(precision_score(y_valid, pred_valid, zero_division=0)),
            "oof_recall": float(recall_score(y_valid, pred_valid, zero_division=0)),
        }
        try:
            summary["oof_auc"] = float(roc_auc_score(y_valid, prob_valid))
        except Exception:
            summary["oof_auc"] = np.nan
    else:
        summary = {"error": "Tidak ada fold valid."}

    if not fold_df.empty:
        summary["mean_fold_accuracy"] = float(fold_df["accuracy"].mean())
        summary["mean_fold_precision"] = float(fold_df["precision"].mean())
        summary["mean_fold_recall"] = float(fold_df["recall"].mean())
        summary["mean_fold_auc"] = float(fold_df["auc"].mean()) if "auc" in fold_df.columns else np.nan

    return fold_df, summary


def artifact_to_bytes(artifact: MLArtifact) -> bytes:
    buff = BytesIO()
    joblib.dump({"model": artifact.model, "feature_columns": artifact.feature_columns, "metrics": artifact.metrics}, buff)
    return buff.getvalue()


# =========================================================
# Parsing
# =========================================================

def parse_manual_tickers(raw: str, auto_suffix: bool) -> List[str]:
    parts = []
    for item in (raw or "").replace("\n", ",").split(","):
        t = item.strip().upper()
        if t:
            parts.append(normalize_ticker(t, auto_suffix=auto_suffix))
    out = []
    seen = set()
    for t in parts:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def parse_uploaded_csv(file, auto_suffix: bool) -> List[str]:
    try:
        df = pd.read_csv(file)
    except Exception:
        return []
    if df.empty:
        return []

    col = detect_ticker_column(df) or df.columns[0]
    tickers = []
    for val in df[col].astype(str).tolist():
        t = val.strip().upper()
        if t and t.lower() != "nan":
            tickers.append(normalize_ticker(t, auto_suffix=auto_suffix))

    out = []
    seen = set()
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# =========================================================
# Evaluation
# =========================================================

def evaluate_ticker(
    ticker: str,
    period: str,
    min_avg_dollar_vol: float,
    min_price: float,
    use_fundamentals: bool,
    benchmark_df: Optional[pd.DataFrame],
    ml_artifact: Optional[MLArtifact],
    ml_enabled: bool,
) -> Optional[Dict]:
    raw_ticker = ticker
    df = load_history(ticker, period=period)
    if df.empty or len(df) < 220:
        return None

    df = build_indicators(df)
    last = df.iloc[-1]
    if not np.isfinite(last.get("sma200", np.nan)):
        return None

    market_score, market_label, market_detail = score_market(benchmark_df) if benchmark_df is not None and not benchmark_df.empty else (np.nan, "Unknown", {})
    regime_score, regime_label, regime_detail = score_market(df)
    trend_score, trend_label, trend_detail = score_trend(df, benchmark_df)
    flow_score, flow_detail = score_institution_flow(df)
    setup_score, setup_label, setup_detail = score_setup(df)
    liquidity_score, liquidity_detail = score_liquidity(df, min_avg_dollar_vol=min_avg_dollar_vol, min_price=min_price)

    info = load_fundamentals(ticker) if use_fundamentals else {}
    quality_score, quality_detail = score_quality(info) if use_fundamentals else (np.nan, {})

    feat_row = ticker_feature_row(df)
    ml_prob = ml_predict_prob(ml_artifact, feat_row) if ml_enabled else np.nan

    final_score = combine_scores(
        regime=regime_score,
        trend=trend_score,
        flow=flow_score,
        setup=setup_score,
        liquidity=liquidity_score,
        quality=quality_score if np.isfinite(quality_score) else None,
        ml_prob=ml_prob if np.isfinite(ml_prob) else None,
    )

    gate_pass = (
        market_label != "RISK OFF"
        and regime_score >= 50
        and trend_score >= 45
        and flow_score >= 40
        and liquidity_score >= 45
        and final_score >= 60
    )

    if final_score >= 85 and gate_pass:
        grade = "A+"
    elif final_score >= 75:
        grade = "A"
    elif final_score >= 65:
        grade = "B"
    else:
        grade = "C"

    if grade in {"A+", "A"} and gate_pass and (not ml_enabled or (np.isfinite(ml_prob) and ml_prob >= 0.55)):
        decision = "BUY"
    elif grade == "B":
        decision = "WATCH"
    else:
        decision = "AVOID"

    result = {
        "ticker": raw_ticker,
        "ticker_norm": ticker,
        "final_score": round(final_score, 2),
        "grade": grade,
        "decision": decision,
        "market_score": round(market_score, 2) if np.isfinite(market_score) else np.nan,
        "market_label": market_label,
        "regime_score": round(regime_score, 2),
        "regime_label": regime_label,
        "trend_score": round(trend_score, 2),
        "trend_label": trend_label,
        "flow_score": round(flow_score, 2),
        "setup_score": round(setup_score, 2),
        "setup_label": setup_label,
        "liquidity_score": round(liquidity_score, 2),
        "quality_score": round(float(quality_score), 2) if np.isfinite(quality_score) else np.nan,
        "ml_prob": round(float(ml_prob), 4) if np.isfinite(ml_prob) else np.nan,
        "gate_pass": gate_pass,
        "close": float(last["close"]),
        "rsi14": float(last["rsi14"]),
        "ret20": float(last["ret20"]),
        "ret60": float(last["ret60"]),
        "vol_rvol20": float(last["vol_rvol20"]),
        "vol_z20": float(last["vol_z20"]),
        "cmf20": float(last["cmf20"]),
        "atr_pct": float(last["atr_pct"]),
        "vixfix_panic": bool(last["vixfix_panic"]),
        "avg_dollar_vol_20": liquidity_detail.get("avg_dollar_vol_20", np.nan),
        "avg_vol_20": liquidity_detail.get("avg_vol_20", np.nan),
        "market_cap": to_float(info.get("marketCap")) if info else np.nan,
        "sector": info.get("sector", "") if info else "",
        "industry": info.get("industry", "") if info else "",
        "_df": df,
        "_details": {
            "market_detail": market_detail,
            "regime_detail": regime_detail,
            "trend_detail": trend_detail,
            "flow_detail": flow_detail,
            "setup_detail": setup_detail,
            "liquidity_detail": liquidity_detail,
            "quality_detail": quality_detail,
        },
    }
    return result


def parse_universe(source_mode: str, csv_file, manual_text: str, auto_suffix: bool) -> List[str]:
    if source_mode == "Upload CSV":
        if csv_file is None:
            return []
        return parse_uploaded_csv(csv_file, auto_suffix=auto_suffix)
    return parse_manual_tickers(manual_text, auto_suffix=auto_suffix)


# =========================================================
# UI
# =========================================================

st.title("IDX Profit-First Scanner ML")
st.caption("Scanner IDX + ML ranking + entry plan. Fokus pada market health, relative strength, volume expansion, VCP, dan probabilitas trade berhasil.")

with st.sidebar:
    st.header("Universe")
    source_mode = st.radio("Sumber ticker", ["Paste ticker", "Upload CSV"], index=0)
    auto_suffix = st.checkbox("Auto tambah .JK", value=True)
    manual_text = st.text_area("Ticker (pisahkan koma / baris baru)", value="BBRI, BMRI, BBCA, ASII, ADRO", height=130, disabled=(source_mode != "Paste ticker"))
    csv_file = st.file_uploader("Upload CSV universe", type=["csv"], disabled=(source_mode != "Upload CSV"))

    st.divider()
    st.header("Scanner settings")
    benchmark_ticker = st.text_input("Benchmark IHSG", value="^JKSE")
    period = st.selectbox("History period", ["1y", "2y", "3y", "5y"], index=1)
    min_price = st.number_input("Min harga", min_value=0.0, value=500.0, step=50.0)
    min_avg_dollar_vol = st.number_input("Min avg dollar volume 20D", min_value=0.0, value=5_000_000_000.0, step=500_000_000.0, format="%.0f")
    use_fundamentals = st.checkbox("Pakai fundamentals jika tersedia", value=False)
    max_workers = st.slider("Parallel workers", 1, 4, 2)
    min_final_score = st.slider("Min final score", 0, 100, 65)
    min_regime_score = st.slider("Min regime score", 0, 100, 50)
    min_trend_score = st.slider("Min trend score", 0, 100, 45)
    min_flow_score = st.slider("Min flow score", 0, 100, 40)
    min_liquidity_score = st.slider("Min liquidity score", 0, 100, 45)
    top_n = st.slider("Top N hasil", 5, 100, 25)

    st.divider()
    st.header("Entry plan")
    account_size = st.number_input("Account size", min_value=0.0, value=100_000_000.0, step=5_000_000.0, format="%.0f")
    risk_pct = st.slider("Risk per trade (%)", 0.1, 5.0, 1.0, 0.1)
    lot_rounding = st.checkbox("Round to lots (100 shares)", value=True)

    st.divider()
    st.header("Machine Learning")
    ml_enabled = st.checkbox("Enable ML ranking", value=True)
    train_period = st.selectbox("ML training period", ["2y", "3y", "5y"], index=1)
    horizon = st.slider("ML horizon (days)", 5, 20, 10)
    target_pct = st.slider("ML target (%)", 3.0, 20.0, 8.0, 0.5) / 100.0
    stop_pct = st.slider("ML stop (%)", 1.0, 15.0, 4.0, 0.5) / 100.0
    sample_stride = st.slider("Sample stride", 1, 5, 2)
    max_samples_total = st.slider("Max training samples", 1000, 20000, 6000, 500)
    max_train_tickers = st.slider("Max training tickers", 10, 200, 60, 5)
    train_btn = st.button("Train ML model", type="secondary")

    st.divider()
    st.header("Backtest & validation")
    backtest_period = st.selectbox("Backtest period", ["5y", "10y"], index=0)
    backtest_use_ml = st.checkbox("Use ML in backtest", value=False)
    max_backtest_tickers = st.slider("Max backtest tickers", 5, 100, 20, 5)
    max_hold_days = st.slider("Max hold days", 5, 40, 20, 1)
    entry_window_days = st.slider("Entry window days", 1, 10, 3, 1)
    cooldown_days = st.slider("Cooldown after exit (days)", 0, 10, 2, 1)
    backtest_risk_pct = st.slider("Backtest risk per trade (%)", 0.1, 5.0, 1.0, 0.1)
    wf_splits = st.slider("Walk-forward folds", 2, 8, 5, 1)
    backtest_btn = st.button("Run backtest", type="secondary")

    scan_btn = st.button("Scan sekarang", type="primary")


@st.cache_data(show_spinner=False, ttl=1800)
def build_benchmark(ticker: str, period: str) -> pd.DataFrame:
    df = load_history(ticker, period=period)
    if df.empty:
        return df
    return build_indicators(df)


if "ml_artifact" not in st.session_state:
    st.session_state["ml_artifact"] = None
if "ml_metrics" not in st.session_state:
    st.session_state["ml_metrics"] = {}
if "ml_training_info" not in st.session_state:
    st.session_state["ml_training_info"] = {}

if "backtest_results" not in st.session_state:
    st.session_state["backtest_results"] = {}
if "backtest_trades" not in st.session_state:
    st.session_state["backtest_trades"] = pd.DataFrame()
if "backtest_signals" not in st.session_state:
    st.session_state["backtest_signals"] = pd.DataFrame()
if "wf_summary" not in st.session_state:
    st.session_state["wf_summary"] = {}
if "wf_folds" not in st.session_state:
    st.session_state["wf_folds"] = pd.DataFrame()


if train_btn:
    universe = parse_universe(source_mode, csv_file, manual_text, auto_suffix=auto_suffix)
    universe = universe[:max_train_tickers]
    if not universe:
        st.warning("Daftar ticker kosong untuk training.")
        st.stop()

    st.subheader("Training ML")
    prog = st.progress(0)
    stat = st.empty()

    train_rows = []

    for i, tkr in enumerate(universe, start=1):
        stat.write(f"Training data: {tkr} ({i}/{len(universe)})")
        df = load_history(tkr, period=train_period)
        if df is not None and not df.empty and len(df) >= 260:
            df = build_indicators(df)
            feat_df = build_feature_matrix(df)
            usable_idx = list(range(220, len(feat_df) - horizon, max(sample_stride, 1)))
            for idx in usable_idx:
                row = feat_df.iloc[idx]
                if row[FEATURE_COLUMNS].isna().all():
                    continue
                label = generate_trade_label(feat_df, idx, horizon=horizon, target_pct=target_pct, stop_pct=stop_pct)
                sample = {c: row.get(c, np.nan) for c in FEATURE_COLUMNS}
                sample["date"] = feat_df.index[idx]
                sample["ticker"] = tkr
                sample["label"] = label
                train_rows.append(sample)
                if max_samples_total > 0 and len(train_rows) >= max_samples_total:
                    break
        prog.progress(i / len(universe))
        if max_samples_total > 0 and len(train_rows) >= max_samples_total:
            break

    prog.empty()
    stat.empty()

    if not train_rows:
        st.error("Tidak ada data training yang cukup.")
    else:
        X = pd.DataFrame(train_rows).sort_values("date").reset_index(drop=True)
        y = X.pop("label").astype(int)
        X = X.drop(columns=["ticker", "date"], errors="ignore")

        artifact, metrics = train_ml_model(X, y)
        if artifact is None:
            st.error(metrics.get("error", "Training gagal."))
        else:
            st.session_state["ml_artifact"] = artifact
            st.session_state["ml_metrics"] = metrics
            st.session_state["ml_training_info"] = {
                "tickers_used": len(universe),
                "samples": len(X),
                "horizon": horizon,
                "target_pct": target_pct,
                "stop_pct": stop_pct,
                "period": train_period,
            }

            wf_folds, wf_summary = run_walk_forward_validation(X, y, n_splits=wf_splits)
            st.session_state["wf_folds"] = wf_folds
            st.session_state["wf_summary"] = wf_summary

            st.success("ML model berhasil dilatih.")


if backtest_btn:
    universe = parse_universe(source_mode, csv_file, manual_text, auto_suffix=auto_suffix)
    universe = universe[:max_backtest_tickers]
    if not universe:
        st.warning("Daftar ticker kosong untuk backtest.")
        st.stop()

    st.subheader("Running backtest")
    prog = st.progress(0)
    stat = st.empty()

    # Optional ML from current session; if not enabled, backtest runs rule-based only.
    ml_for_backtest = st.session_state.get("ml_artifact") if backtest_use_ml else None

    trade_df, bt_metrics, signal_df = run_rule_based_backtest(
        universe=universe,
        period=backtest_period,
        benchmark_ticker=benchmark_ticker,
        min_avg_dollar_vol=min_avg_dollar_vol,
        min_price=min_price,
        ml_artifact=ml_for_backtest,
        ml_enabled=backtest_use_ml,
        max_hold_days=max_hold_days,
        entry_window_days=entry_window_days,
        cooldown_days=cooldown_days,
        risk_pct=backtest_risk_pct,
    )

    st.session_state["backtest_results"] = bt_metrics
    st.session_state["backtest_trades"] = trade_df
    st.session_state["backtest_signals"] = signal_df

    st.success("Backtest selesai.")
    prog.empty()
    stat.empty()

if scan_btn:
    tickers = parse_universe(source_mode, csv_file, manual_text, auto_suffix=auto_suffix)
    if not tickers:
        st.warning("Daftar ticker kosong.")
        st.stop()

    benchmark_df = build_benchmark(benchmark_ticker, period)
    benchmark_score, benchmark_label, benchmark_detail = score_market(benchmark_df) if not benchmark_df.empty else (np.nan, "Unknown", {})
    ml_artifact = st.session_state.get("ml_artifact")

    st.subheader("Market health")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Benchmark", benchmark_ticker)
    m2.metric("Regime", benchmark_label)
    m3.metric("Score", f"{benchmark_score:.1f}" if np.isfinite(benchmark_score) else "N/A")
    m4.metric("Universe", len(tickers))

    if st.session_state.get("ml_metrics"):
        mm1, mm2, mm3, mm4 = st.columns(4)
        mm1.metric("ML Accuracy", f"{st.session_state['ml_metrics'].get('accuracy', np.nan):.3f}")
        mm2.metric("ML Precision", f"{st.session_state['ml_metrics'].get('precision', np.nan):.3f}")
        mm3.metric("ML Recall", f"{st.session_state['ml_metrics'].get('recall', np.nan):.3f}")
        mm4.metric("ML AUC", f"{st.session_state['ml_metrics'].get('auc', np.nan):.3f}" if np.isfinite(st.session_state['ml_metrics'].get('auc', np.nan)) else "N/A")

    rows = []
    progress = st.progress(0)
    status = st.empty()

    def worker(tkr: str):
        try:
            return evaluate_ticker(
                ticker=tkr,
                period=period,
                min_avg_dollar_vol=min_avg_dollar_vol,
                min_price=min_price,
                use_fundamentals=use_fundamentals,
                benchmark_df=benchmark_df,
                ml_artifact=ml_artifact,
                ml_enabled=ml_enabled,
            )
        except Exception as e:
            return {
                "ticker": tkr,
                "ticker_norm": normalize_ticker(tkr, auto_suffix=auto_suffix),
                "final_score": np.nan,
                "grade": "",
                "decision": "ERROR",
                "market_score": np.nan,
                "market_label": "Error",
                "regime_score": np.nan,
                "regime_label": "Error",
                "trend_score": np.nan,
                "trend_label": "Error",
                "flow_score": np.nan,
                "setup_score": np.nan,
                "setup_label": "Error",
                "liquidity_score": np.nan,
                "quality_score": np.nan,
                "ml_prob": np.nan,
                "gate_pass": False,
                "close": np.nan,
                "rsi14": np.nan,
                "ret20": np.nan,
                "ret60": np.nan,
                "vol_rvol20": np.nan,
                "vol_z20": np.nan,
                "cmf20": np.nan,
                "atr_pct": np.nan,
                "vixfix_panic": False,
                "avg_dollar_vol_20": np.nan,
                "avg_vol_20": np.nan,
                "market_cap": np.nan,
                "sector": "",
                "industry": "",
                "error": str(e),
            }

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, t): t for t in tickers}
        done = 0
        for fut in as_completed(futures):
            done += 1
            status.write(f"Scanning {done}/{len(tickers)}...")
            rows.append(fut.result())
            progress.progress(done / len(tickers))

    progress.empty()
    status.empty()

    if not rows:
        st.error("Tidak ada hasil valid dari scanner.")
        st.stop()

    df = pd.DataFrame(rows)
    sort_cols = [c for c in ["final_score", "regime_score", "trend_score", "flow_score", "liquidity_score", "ml_prob"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=False, na_position="last")

    filtered = df.copy()
    filtered = filtered[
        (filtered["final_score"].fillna(-1) >= min_final_score)
        & (filtered["regime_score"].fillna(-1) >= min_regime_score)
        & (filtered["trend_score"].fillna(-1) >= min_trend_score)
        & (filtered["flow_score"].fillna(-1) >= min_flow_score)
        & (filtered["liquidity_score"].fillna(-1) >= min_liquidity_score)
    ]

    breadth_above_sma50 = np.nan
    breadth_above_sma200 = np.nan
    above_50 = []
    above_200 = []
    for _, row in df.iterrows():
        hist = row.get("_df")
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            last = hist.iloc[-1]
            above_50.append(bool(last.get("close", np.nan) > last.get("sma50", np.nan)))
            above_200.append(bool(last.get("close", np.nan) > last.get("sma200", np.nan)))
    if above_50:
        breadth_above_sma50 = float(np.mean(above_50) * 100.0)
    if above_200:
        breadth_above_sma200 = float(np.mean(above_200) * 100.0)

    sector_rank = (
        df[df["sector"].astype(str).str.len() > 0]
        .groupby("sector", dropna=True)["final_score"]
        .agg(["mean", "count"])
        .sort_values(["mean", "count"], ascending=False)
        .reset_index()
        if "sector" in df.columns else pd.DataFrame()
    )

    tabs = st.tabs(["Market", "Scanner", "Entry Plan", "ML", "Backtest", "Sector", "Export"])

    with tabs[0]:
        st.subheader("Market breadth")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Benchmark regime", benchmark_label)
        c2.metric("Above SMA50", f"{breadth_above_sma50:.1f}%" if np.isfinite(breadth_above_sma50) else "N/A")
        c3.metric("Above SMA200", f"{breadth_above_sma200:.1f}%" if np.isfinite(breadth_above_sma200) else "N/A")
        c4.metric("Risk mode", "BUYING OK" if benchmark_label == "RISK ON" else "WAIT")

        st.caption("Aturan praktis: jika market risk-off, fokus ke watchlist; jika risk-on, hanya ambil kandidat terbaik.")
        if benchmark_detail:
            st.dataframe(pd.DataFrame([benchmark_detail]), use_container_width=True, hide_index=True)

    with tabs[1]:
        st.subheader("Scanner results")
        a, b, c, d = st.columns(4)
        a.metric("Total ticker", len(df))
        b.metric("Lolos filter", len(filtered))
        c.metric("A+/A", int(df["grade"].isin(["A+", "A"]).sum()))
        d.metric("BUY candidates", int((df["decision"] == "BUY").sum()))

        show_cols = [
            "ticker", "grade", "decision", "final_score", "ml_prob",
            "regime_score", "trend_score", "flow_score", "setup_score",
            "liquidity_score", "quality_score", "close", "rsi14", "ret20",
            "ret60", "vol_rvol20", "cmf20", "atr_pct", "vixfix_panic",
            "avg_dollar_vol_20", "sector", "industry",
        ]
        show_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(filtered[show_cols].head(top_n).reset_index(drop=True), use_container_width=True, hide_index=True)

        st.markdown("**Interpretasi:** A+ / A = kandidat utama, B = monitor, C = abaikan.")

        selected = st.selectbox("Pilih ticker untuk detail", options=df["ticker"].tolist())
        picked = df[df["ticker"] == selected].iloc[0]
        st.write(f"**{picked['ticker']}** — {picked['decision']} | Grade {picked['grade']} | Final {picked['final_score']:.1f}")

        cols = st.columns(5)
        cols[0].metric("Regime", f"{picked['regime_score']:.1f}")
        cols[1].metric("Trend", f"{picked['trend_score']:.1f}")
        cols[2].metric("Flow", f"{picked['flow_score']:.1f}")
        cols[3].metric("Setup", f"{picked['setup_score']:.1f}")
        cols[4].metric("ML Prob", f"{picked['ml_prob']:.2f}" if np.isfinite(picked.get("ml_prob", np.nan)) else "N/A")

        hist = picked.get("_df")
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            st.line_chart(hist[["close", "ema20", "ema50", "ema100", "ema200"]].dropna().tail(180))
            with st.expander("Latest indicator snapshot"):
                last = hist.iloc[-1]
                snapshot = pd.DataFrame(
                    {
                        "metric": ["close", "ema20", "ema50", "ema100", "ema200", "sma200", "rsi14", "atr14", "atr_pct", "ret20", "ret60", "vol_rvol20", "vol_z20", "cmf20", "vixfix", "vixfix_upper"],
                        "value": [last["close"], last["ema20"], last["ema50"], last["ema100"], last["ema200"], last["sma200"], last["rsi14"], last["atr14"], last["atr_pct"], last["ret20"], last["ret60"], last["vol_rvol20"], last["vol_z20"], last["cmf20"], last["vixfix"], last["vixfix_upper"]],
                    }
                )
                st.dataframe(snapshot, use_container_width=True, hide_index=True)

    with tabs[2]:
        st.subheader("Entry plan")
        selected = st.selectbox("Ticker for entry plan", options=df["ticker"].tolist(), key="entry_plan_ticker")
        picked = df[df["ticker"] == selected].iloc[0]
        hist = picked.get("_df")
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            plan = build_entry_plan(hist, picked["setup_label"], account_size=account_size, risk_pct=risk_pct, lot_rounding=lot_rounding)
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Entry", f"{plan['entry_trigger']:.2f}")
            p2.metric("Stop", f"{plan['stop_loss']:.2f}")
            p3.metric("Target 1", f"{plan['target1']:.2f}")
            p4.metric("Target 2", f"{plan['target2']:.2f}")

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Risk/share", f"{plan['risk_per_share']:.2f}")
            r2.metric("Shares", f"{plan['shares']}")
            r3.metric("Lots", f"{plan['lots']}")
            r4.metric("R:R T2", f"{plan['rr_t2']:.2f}R")

            st.markdown(f"**Setup:** {picked['setup_label']}")
            st.markdown(f"**Plan:** {plan['setup_note']}")

            plan_df = pd.DataFrame([{
                "ticker": selected,
                "setup": picked["setup_label"],
                "decision": picked["decision"],
                "entry_trigger": plan["entry_trigger"],
                "stop_loss": plan["stop_loss"],
                "invalidation": plan["invalidation"],
                "target1": plan["target1"],
                "target2": plan["target2"],
                "risk_per_share": plan["risk_per_share"],
                "risk_budget": plan["risk_budget"],
                "shares": plan["shares"],
                "lots": plan["lots"],
                "position_value": plan["position_value"],
                "total_risk": plan["total_risk"],
                "rr_t1": plan["rr_t1"],
                "rr_t2": plan["rr_t2"],
            }])
            st.dataframe(plan_df, use_container_width=True, hide_index=True)

            st.markdown(
                """
                **Trade rule singkat**
                - Entry hanya jika harga mengonfirmasi level trigger.
                - Stop loss wajib; jangan dipindah menjauh saat rugi.
                - Ambil partial profit di target 1, sisakan untuk target 2 bila trend lanjut.
                - Jika market berubah risk-off, kurangi size atau skip trade.
                """
            )
        else:
            st.info("Data chart belum cukup untuk membuat entry plan.")

    with tabs[3]:
        st.subheader("Machine Learning")
        if st.session_state.get("ml_artifact") is None:
            st.info("Belum ada model. Klik **Train ML model** di sidebar.")
        else:
            st.success("Model aktif.")
            info = st.session_state.get("ml_training_info", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tickers used", info.get("tickers_used", "N/A"))
            c2.metric("Samples", info.get("samples", "N/A"))
            c3.metric("Horizon", info.get("horizon", "N/A"))
            c4.metric("Target / Stop", f"{info.get('target_pct', 0)*100:.1f}% / {info.get('stop_pct', 0)*100:.1f}%")
            metrics = st.session_state.get("ml_metrics", {})
            st.dataframe(pd.DataFrame([metrics]), use_container_width=True, hide_index=True)

            wf_summary = st.session_state.get("wf_summary", {})
            if wf_summary:
                st.markdown("**Walk-forward summary**")
                st.dataframe(pd.DataFrame([wf_summary]), use_container_width=True, hide_index=True)

            artifact = st.session_state["ml_artifact"]
            model_bytes = artifact_to_bytes(artifact)
            st.download_button(
                "Download ML model (.joblib)",
                model_bytes,
                file_name="idx_ml_model.joblib",
                mime="application/octet-stream",
            )

            st.markdown(
                """
                **Cara kerja ML**
                - Label = apakah trade mencapai target sebelum stop dalam horizon tertentu.
                - Model dipakai sebagai ranking layer, bukan menggantikan logika market/regime.
                - Retrain berkala lebih aman daripada update liar setiap hari.
                """
            )

    with tabs[4]:
        st.subheader("Backtest")
        backtest_results = st.session_state.get("backtest_results", {})
        backtest_trades = st.session_state.get("backtest_trades", pd.DataFrame())
        backtest_signals = st.session_state.get("backtest_signals", pd.DataFrame())

        if not backtest_results:
            st.info("Klik **Run backtest** di sidebar untuk menghitung metrik 5–10 tahun.")
        else:
            b1, b2, b3, b4, b5, b6 = st.columns(6)
            b1.metric("Trades", backtest_results.get("trades", 0))
            wr = backtest_results.get("win_rate", np.nan)
            b2.metric("Win rate", f"{wr*100:.1f}%" if np.isfinite(wr) else "N/A")
            pf = backtest_results.get("profit_factor", np.nan)
            b3.metric("Profit factor", f"{pf:.2f}" if np.isfinite(pf) else "N/A")
            er = backtest_results.get("expectancy_r", np.nan)
            b4.metric("Expectancy (R)", f"{er:.2f}" if np.isfinite(er) else "N/A")
            dd = backtest_results.get("max_drawdown", np.nan)
            b5.metric("Max drawdown", f"{dd*100:.1f}%" if np.isfinite(dd) else "N/A")
            tr = backtest_results.get("total_return_pct", np.nan)
            b6.metric("Total return", f"{tr:.1f}%" if np.isfinite(tr) else "N/A")

            summary_df = pd.DataFrame([backtest_results])
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            if isinstance(backtest_trades, pd.DataFrame) and not backtest_trades.empty:
                st.markdown("**Trade log**")
                trade_view_cols = [c for c in [
                    "ticker", "signal_date", "entry_date", "exit_date", "decision", "grade", "setup",
                    "final_score", "ml_prob", "entry_price", "stop_loss", "target1", "target2",
                    "realized_r", "exit_reason", "bars_held"
                ] if c in backtest_trades.columns]
                st.dataframe(backtest_trades[trade_view_cols].head(200), use_container_width=True, hide_index=True)
                st.download_button(
                    "Download backtest trades CSV",
                    backtest_trades.to_csv(index=False).encode("utf-8"),
                    file_name="idx_backtest_trades.csv",
                    mime="text/csv",
                )

            if isinstance(backtest_signals, pd.DataFrame) and not backtest_signals.empty:
                st.markdown("**Signals (entries that qualified)**")
                signal_view_cols = [c for c in [
                    "ticker", "signal_date", "decision", "grade", "setup", "final_score",
                    "ml_prob", "entry_trigger", "stop_loss", "target1", "target2"
                ] if c in backtest_signals.columns]
                st.dataframe(backtest_signals[signal_view_cols].head(200), use_container_width=True, hide_index=True)
                st.download_button(
                    "Download backtest signals CSV",
                    backtest_signals.to_csv(index=False).encode("utf-8"),
                    file_name="idx_backtest_signals.csv",
                    mime="text/csv",
                )

        wf_summary = st.session_state.get("wf_summary", {})
        wf_folds = st.session_state.get("wf_folds", pd.DataFrame())
        st.divider()
        st.subheader("Walk-forward validation")
        if not wf_summary:
            st.info("Train ML model dulu untuk melihat walk-forward validation.")
        else:
            w1, w2, w3, w4 = st.columns(4)
            w1.metric("OOF Accuracy", f"{wf_summary.get('oof_accuracy', np.nan):.3f}" if np.isfinite(wf_summary.get('oof_accuracy', np.nan)) else "N/A")
            w2.metric("OOF Precision", f"{wf_summary.get('oof_precision', np.nan):.3f}" if np.isfinite(wf_summary.get('oof_precision', np.nan)) else "N/A")
            w3.metric("OOF Recall", f"{wf_summary.get('oof_recall', np.nan):.3f}" if np.isfinite(wf_summary.get('oof_recall', np.nan)) else "N/A")
            w4.metric("OOF AUC", f"{wf_summary.get('oof_auc', np.nan):.3f}" if np.isfinite(wf_summary.get('oof_auc', np.nan)) else "N/A")
            st.dataframe(pd.DataFrame([wf_summary]), use_container_width=True, hide_index=True)
            if isinstance(wf_folds, pd.DataFrame) and not wf_folds.empty:
                st.dataframe(wf_folds, use_container_width=True, hide_index=True)

    with tabs[5]:
        st.subheader("Sector ranking")
        if sector_rank.empty:
            st.info("Sector data tidak tersedia dari data provider.")
        else:
            st.dataframe(sector_rank, use_container_width=True, hide_index=True)
        st.caption("Ranking sektor di sini adalah proxy berbasis score rata-rata kandidat yang berhasil dibaca sektornya.")

    with tabs[6]:
        st.subheader("Export")
        export_cols = [c for c in [
            "ticker", "ticker_norm", "decision", "grade", "final_score", "ml_prob",
            "regime_score", "trend_score", "flow_score", "setup_score", "liquidity_score",
            "quality_score", "close", "rsi14", "ret20", "ret60", "vol_rvol20",
            "cmf20", "atr_pct", "vixfix_panic", "avg_dollar_vol_20", "avg_vol_20",
            "market_cap", "sector", "industry"
        ] if c in df.columns]
        st.download_button(
            "Download full results CSV",
            df[export_cols].to_csv(index=False).encode("utf-8"),
            file_name="idx_profit_first_scanner_results.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download filtered results CSV",
            filtered[export_cols].to_csv(index=False).encode("utf-8"),
            file_name="idx_profit_first_scanner_filtered.csv",
            mime="text/csv",
        )

        template = pd.DataFrame({"ticker": ["BBRI", "BMRI", "BBCA", "ASII", "ADRO"]})
        st.write("Template CSV:")
        st.dataframe(template, use_container_width=True, hide_index=True)
        st.download_button(
            "Download template CSV",
            template.to_csv(index=False).encode("utf-8"),
            file_name="idx_universe_template.csv",
            mime="text/csv",
        )

    with st.expander("Scanner logic summary"):
        st.markdown(
            """
            - **Market health** memutuskan apakah market layak di-trade.
            - **Trend** menilai struktur harga dan relative strength.
            - **Institutional flow** mencari volume, OBV, CMF, dan accumulation.
            - **Setup** memilih breakout atau pullback/shakeout yang lebih kuat.
            - **Liquidity** mencegah saham yang sulit dieksekusi.
            - **ML** memberi ranking probabilistik pada kandidat yang sudah lolos filter.
            - **Entry plan** memberi level entry, stop, target, dan sizing otomatis.
            """
        )
else:
    st.info("Pilih sumber ticker di sidebar, lalu klik **Scan sekarang**.")
    st.markdown(
        """
        Mode yang didukung:
        - **Paste ticker**: cocok untuk daftar pendek.
        - **Upload CSV**: cocok untuk scan universe IHSG penuh.

        CSV cukup punya kolom `ticker`, `symbol`, atau `code`. Jika ticker IDX belum ada suffix, app otomatis menambah `.JK`.
        """
    )
