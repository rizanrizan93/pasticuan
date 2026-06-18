import math
import time
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

REQUIRED_COLS = ["ticker", "date", "open", "high", "low", "close", "volume"]


def normalize_ticker_id(x: str) -> str:
    s = str(x).strip().upper()
    if s and "." not in s:
        s = f"{s}.JK"
    return s


def normalize_display_ticker(x: str) -> str:
    s = str(x).strip().upper()
    return s[:-3] if s.endswith(".JK") else s


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {"symbol": "ticker", "code": "ticker", "datetime": "date", "timestamp": "date", "time": "date", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
    return df


def standardize_ticker_list(df: pd.DataFrame) -> List[str]:
    df = clean_columns(df)
    if "ticker" in df.columns:
        col = "ticker"
    elif "symbol" in df.columns:
        col = "symbol"
    else:
        col = df.columns[0]
    vals = df[col].dropna().astype(str).map(normalize_ticker_id).map(str.strip)
    return list(dict.fromkeys([v for v in vals.tolist() if v]))


def standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_columns(df)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV wajib punya kolom: {', '.join(REQUIRED_COLS)}. Kolom hilang: {', '.join(missing)}")
    out = df[REQUIRED_COLS].copy()
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=REQUIRED_COLS)
    out = out.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    return out.reset_index(drop=True)


def tr(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    return pd.concat([(df["high"] - df["low"]), (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return tr(df).rolling(n, min_periods=max(3, n // 3)).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def relative_volume(df: pd.DataFrame, n: int = 20) -> pd.Series:
    return df["volume"] / df["volume"].rolling(n, min_periods=max(5, n // 3)).mean()


def candle_body(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs()


def candle_efficiency(df: pd.DataFrame) -> float:
    r = (df["high"] - df["low"]).replace(0, np.nan)
    b = candle_body(df)
    if len(df) == 0 or pd.isna(r.iloc[-1]) or r.iloc[-1] <= 0:
        return 0.0
    return float(min(1.0, b.iloc[-1] / r.iloc[-1]))


def trend_state(df: pd.DataFrame) -> str:
    if len(df) < 60:
        return "neutral"
    e20 = ema(df["close"], 20)
    e50 = ema(df["close"], 50)
    c = df["close"]
    if e20.iloc[-1] > e50.iloc[-1] and c.iloc[-1] > e20.iloc[-1]:
        return "uptrend"
    if e20.iloc[-1] < e50.iloc[-1] and c.iloc[-1] < e20.iloc[-1]:
        return "downtrend"
    return "range"


def swing_levels(df: pd.DataFrame, w: int = 3) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    sh = df["high"].rolling(w * 2 + 1, center=True).max().eq(df["high"])
    sl = df["low"].rolling(w * 2 + 1, center=True).min().eq(df["low"])
    highs = [(i, float(df["high"].iloc[i])) for i in np.where(sh.fillna(False).values)[0]]
    lows = [(i, float(df["low"].iloc[i])) for i in np.where(sl.fillna(False).values)[0]]
    return highs, lows


def liquidity_levels(df: pd.DataFrame) -> List[float]:
    highs, lows = swing_levels(df, 3)
    levels = [v for _, v in highs] + [v for _, v in lows]
    levels += [float(df["high"].iloc[-20:].max()), float(df["low"].iloc[-20:].min())]
    return sorted(set(round(float(x), 4) for x in levels if pd.notna(x)))


def nearest_future_level(levels: List[float], price: float, bullish: bool = True) -> Optional[float]:
    vals = [float(x) for x in levels if pd.notna(x)]
    if bullish:
        cands = [x for x in vals if x > price]
        return min(cands) if cands else None
    cands = [x for x in vals if x < price]
    return max(cands) if cands else None


def rr(entry: float, stop: float, target: float, bullish: bool = True) -> float:
    risk = entry - stop if bullish else stop - entry
    reward = target - entry if bullish else entry - target
    if risk <= 0:
        return np.nan
    return reward / risk if reward > 0 else 0.0


def fill_probability(close: float, entry: float, atrv: float, age_bars: int, dir_strength: float = 0.0) -> float:
    if any(pd.isna(v) for v in [close, entry, atrv]):
        return 0.0
    atrv = max(float(atrv), 1e-9)
    dist = abs(close - entry) / atrv
    base = 100.0 * math.exp(-1.05 * dist)
    age_penalty = max(0.35, 1.0 - 0.06 * max(age_bars, 0))
    mom = 1.0 + max(-0.15, min(0.25, dir_strength))
    return float(max(0.0, min(100.0, base * age_penalty * mom)))


def score_quality(rr2: float, fill_prob: float, confluence: float, structure: float) -> float:
    rr2 = 0.0 if pd.isna(rr2) else rr2
    score = min(50.0, rr2 * 16.0) + min(25.0, fill_prob * 0.25) + min(15.0, confluence * 15.0) + min(10.0, structure * 10.0)
    return float(min(100.0, score))


def volume_expansion(df: pd.DataFrame) -> float:
    rv = relative_volume(df, 20)
    return 0.0 if pd.isna(rv.iloc[-1]) else float(max(0.0, min(2.0, rv.iloc[-1])))


def detect_timeframe_label(df: pd.DataFrame) -> str:
    if len(df) < 5:
        return "unknown"
    diffs = df["date"].sort_values().diff().dropna()
    if diffs.empty:
        return "unknown"
    med = diffs.dt.total_seconds().median() / 60.0
    if med <= 2:
        return "M1"
    if med <= 10:
        return "M5"
    if med <= 20:
        return "M15"
    if med <= 90:
        return "H1"
    if med <= 240:
        return "H4"
    return "D1"


def load_ohlcv(ticker: str, period: str, interval: str) -> pd.DataFrame:
    y = normalize_ticker_id(ticker)
    data = yf.download(y, period=period, interval=interval, auto_adjust=False, progress=False, threads=False)
    if data is None or len(data) == 0:
        return pd.DataFrame()
    data = data.reset_index()
    date_col = "Date" if "Date" in data.columns else "Datetime" if "Datetime" in data.columns else data.columns[0]
    mapping = {}
    for c in data.columns:
        lc = str(c).lower()
        if lc in ["open", "high", "low", "close", "volume"]:
            mapping[c] = lc
        elif c == date_col or lc in ["date", "datetime"]:
            mapping[c] = "date"
    data = data.rename(columns=mapping)
    keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in data.columns]
    if len(keep) < 6:
        return pd.DataFrame()
    out = data[keep].copy()
    out["ticker"] = normalize_display_ticker(ticker)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    out = out.sort_values("date").drop_duplicates(["date"], keep="last")
    return out[["ticker", "date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def sweep_pattern(df: pd.DataFrame, bullish: bool = True) -> Optional[int]:
    highs, lows = swing_levels(df, 3)
    if bullish:
        for i in range(len(df) - 2, max(2, len(df) - 12), -1):
            prev_lows = [x[1] for x in lows if x[0] < i - 1]
            if not prev_lows:
                continue
            ref = prev_lows[-1]
            if df["low"].iloc[i] < ref and df["close"].iloc[i] > ref and i + 1 < len(df) and df["close"].iloc[i + 1] > df["open"].iloc[i + 1]:
                return i
    else:
        for i in range(len(df) - 2, max(2, len(df) - 12), -1):
            prev_highs = [x[1] for x in highs if x[0] < i - 1]
            if not prev_highs:
                continue
            ref = prev_highs[-1]
            if df["high"].iloc[i] > ref and df["close"].iloc[i] < ref and i + 1 < len(df) and df["close"].iloc[i + 1] < df["open"].iloc[i + 1]:
                return i
    return None


def engine_unicorn_sniper(df: pd.DataFrame) -> Dict:
    if len(df) < 60:
        return {"valid": False, "reason": "data kurang"}
    a = atr(df, 14)
    atrv = float(a.iloc[-1]) if pd.notna(a.iloc[-1]) else np.nan
    c = float(df["close"].iloc[-1])
    t = trend_state(df)
    lvls = liquidity_levels(df)

    i = sweep_pattern(df, bullish=True)
    if i is not None:
        entry = float(df["close"].iloc[min(i + 1, len(df) - 1)])
        stop = float(df["low"].iloc[i]) - (0.15 * atrv if pd.notna(atrv) else 0.0)
        tp1 = nearest_future_level([x for x in lvls if x > entry], entry, True) or (entry + 1.5 * (entry - stop))
        tp2 = nearest_future_level([x for x in lvls if x > tp1], tp1, True) or (entry + 2.5 * (entry - stop))
        r1 = rr(entry, stop, tp1, True)
        r2 = rr(entry, stop, tp2, True)
        fp = fill_probability(c, entry, atrv, len(df) - i - 1, 0.15 if t == "uptrend" else 0.0)
        conf = min(1.0, 0.55 + 0.15 * volume_expansion(df) + 0.15 * candle_efficiency(df))
        score = score_quality(r2, fp, conf, 1.0 if t in ("uptrend", "range") else 0.7)
        return {"valid": True, "setup_type": "Unicorn/Sniper", "direction": "Bullish", "entry": round(entry, 2), "stoploss": round(stop, 2), "tp1": round(tp1, 2), "tp2": round(tp2, 2), "rr1": round(r1, 2) if pd.notna(r1) else None, "rr2": round(r2, 2) if pd.notna(r2) else None, "fill_prob": round(fp, 1), "score": round(score, 1), "reason": "sweep + displacement + retrace", "invalidation": f"close below sweep low {round(float(df['low'].iloc[i]), 2)}"}

    i = sweep_pattern(df, bullish=False)
    if i is not None:
        entry = float(df["close"].iloc[min(i + 1, len(df) - 1)])
        stop = float(df["high"].iloc[i]) + (0.15 * atrv if pd.notna(atrv) else 0.0)
        tp1 = nearest_future_level([x for x in lvls if x < entry], entry, False) or (entry - 1.5 * (stop - entry))
        tp2 = nearest_future_level([x for x in lvls if x < tp1], tp1, False) or (entry - 2.5 * (stop - entry))
        r1 = rr(entry, stop, tp1, False)
        r2 = rr(entry, stop, tp2, False)
        fp = fill_probability(c, entry, atrv, len(df) - i - 1, 0.15 if t == "downtrend" else 0.0)
        conf = min(1.0, 0.55 + 0.15 * volume_expansion(df) + 0.15 * candle_efficiency(df))
        score = score_quality(r2, fp, conf, 1.0 if t in ("downtrend", "range") else 0.7)
        return {"valid": True, "setup_type": "Unicorn/Sniper", "direction": "Bearish", "entry": round(entry, 2), "stoploss": round(stop, 2), "tp1": round(tp1, 2), "tp2": round(tp2, 2), "rr1": round(r1, 2) if pd.notna(r1) else None, "rr2": round(r2, 2) if pd.notna(r2) else None, "fill_prob": round(fp, 1), "score": round(score, 1), "reason": "sweep + displacement + retrace", "invalidation": f"close above sweep high {round(float(df['high'].iloc[i]), 2)}"}

    return {"valid": False, "reason": "no sweep + displacement found"}


def engine_pullback_continuation(df: pd.DataFrame) -> Dict:
    if len(df) < 60:
        return {"valid": False, "reason": "data kurang"}
    a = atr(df, 14)
    atrv = float(a.iloc[-1]) if pd.notna(a.iloc[-1]) else np.nan
    e20 = ema(df["close"], 20)
    e50 = ema(df["close"], 50)
    c = float(df["close"].iloc[-1])
    t = trend_state(df)
    lvls = liquidity_levels(df)

    if t == "uptrend":
        low8 = float(df["low"].iloc[-8:].min())
        if low8 <= e20.iloc[-1] + 0.35 * atrv and df["close"].iloc[-1] >= e50.iloc[-1] and df["close"].iloc[-1] > df["open"].iloc[-1]:
            entry = c
            stop = low8 - (0.15 * atrv if pd.notna(atrv) else 0.0)
            tp1 = nearest_future_level([x for x in lvls if x > entry], entry, True) or (entry + 1.5 * (entry - stop))
            tp2 = nearest_future_level([x for x in lvls if x > tp1], tp1, True) or (entry + 2.5 * (entry - stop))
            r1 = rr(entry, stop, tp1, True)
            r2 = rr(entry, stop, tp2, True)
            fp = fill_probability(c, entry, atrv, 0, 0.08)
            score = score_quality(r2, fp, min(1.0, 0.5 + 0.2 * volume_expansion(df)), 1.0)
            return {"valid": True, "setup_type": "Pullback Continuation", "direction": "Bullish", "entry": round(entry, 2), "stoploss": round(stop, 2), "tp1": round(tp1, 2), "tp2": round(tp2, 2), "rr1": round(r1, 2) if pd.notna(r1) else None, "rr2": round(r2, 2) if pd.notna(r2) else None, "fill_prob": round(fp, 1), "score": round(score, 1), "reason": "uptrend intact + pullback + reclaim", "invalidation": f"close below pullback low {round(low8, 2)}"}

    if t == "downtrend":
        high8 = float(df["high"].iloc[-8:].max())
        if high8 >= e20.iloc[-1] - 0.35 * atrv and df["close"].iloc[-1] <= e50.iloc[-1] and df["close"].iloc[-1] < df["open"].iloc[-1]:
            entry = c
            stop = high8 + (0.15 * atrv if pd.notna(atrv) else 0.0)
            tp1 = nearest_future_level([x for x in lvls if x < entry], entry, False) or (entry - 1.5 * (stop - entry))
            tp2 = nearest_future_level([x for x in lvls if x < tp1], tp1, False) or (entry - 2.5 * (stop - entry))
            r1 = rr(entry, stop, tp1, False)
            r2 = rr(entry, stop, tp2, False)
            fp = fill_probability(c, entry, atrv, 0, 0.08)
            score = score_quality(r2, fp, min(1.0, 0.5 + 0.2 * volume_expansion(df)), 1.0)
            return {"valid": True, "setup_type": "Pullback Continuation", "direction": "Bearish", "entry": round(entry, 2), "stoploss": round(stop, 2), "tp1": round(tp1, 2), "tp2": round(tp2, 2), "rr1": round(r1, 2) if pd.notna(r1) else None, "rr2": round(r2, 2) if pd.notna(r2) else None, "fill_prob": round(fp, 1), "score": round(score, 1), "reason": "downtrend intact + pullback + reclaim", "invalidation": f"close above pullback high {round(high8, 2)}"}

    return {"valid": False, "reason": "trend / reclaim / pullback criteria not met"}


def engine_breakout_retest(df: pd.DataFrame) -> Dict:
    if len(df) < 50:
        return {"valid": False, "reason": "data kurang"}
    a = atr(df, 14)
    atrv = float(a.iloc[-1]) if pd.notna(a.iloc[-1]) else np.nan
    rv = relative_volume(df, 20)
    vr = float(rv.iloc[-1]) if pd.notna(rv.iloc[-1]) else 0.0
    c = float(df["close"].iloc[-1])
    lvls = liquidity_levels(df)
    t = trend_state(df)

    recent = df.iloc[-12:]
    rh = float(recent["high"].max())
    rl = float(recent["low"].min())
    width = rh - rl
    avg_atr = float(a.iloc[-20:].mean()) if pd.notna(a.iloc[-20:].mean()) else np.nan
    compression = pd.notna(avg_atr) and width <= 2.0 * avg_atr
    breakout_up = df["close"].iloc[-1] > rh and df["close"].iloc[-2] <= rh
    breakout_down = df["close"].iloc[-1] < rl and df["close"].iloc[-2] >= rl

    if compression and breakout_up and vr >= 1.1:
        entry = rh
        stop = rl - (0.12 * atrv if pd.notna(atrv) else 0.0)
        tp1 = nearest_future_level([x for x in lvls if x > entry], entry, True) or (rh + width)
        tp2 = max(tp1, entry + 2.0 * (entry - stop))
        r1 = rr(entry, stop, tp1, True)
        r2 = rr(entry, stop, tp2, True)
        fp = fill_probability(c, entry, atrv, 0, 0.1 if t == "uptrend" else 0.0)
        score = score_quality(r2, fp, min(1.0, 0.45 + 0.25 * min(2.0, vr) + 0.1 * candle_efficiency(df)), 1.0)
        return {"valid": True, "setup_type": "Breakout Retest", "direction": "Bullish", "entry": round(entry, 2), "stoploss": round(stop, 2), "tp1": round(tp1, 2), "tp2": round(tp2, 2), "rr1": round(r1, 2) if pd.notna(r1) else None, "rr2": round(r2, 2) if pd.notna(r2) else None, "fill_prob": round(fp, 1), "score": round(score, 1), "reason": "compressed base + breakout + volume expansion", "invalidation": f"close back below base high {round(rh, 2)}"}

    if compression and breakout_down and vr >= 1.1:
        entry = rl
        stop = rh + (0.12 * atrv if pd.notna(atrv) else 0.0)
        tp1 = nearest_future_level([x for x in lvls if x < entry], entry, False) or (rl - width)
        tp2 = min(tp1, entry - 2.0 * (stop - entry))
        r1 = rr(entry, stop, tp1, False)
        r2 = rr(entry, stop, tp2, False)
        fp = fill_probability(c, entry, atrv, 0, 0.1 if t == "downtrend" else 0.0)
        score = score_quality(r2, fp, min(1.0, 0.45 + 0.25 * min(2.0, vr) + 0.1 * candle_efficiency(df)), 1.0)
        return {"valid": True, "setup_type": "Breakout Retest", "direction": "Bearish", "entry": round(entry, 2), "stoploss": round(stop, 2), "tp1": round(tp1, 2), "tp2": round(tp2, 2), "rr1": round(r1, 2) if pd.notna(r1) else None, "rr2": round(r2, 2) if pd.notna(r2) else None, "fill_prob": round(fp, 1), "score": round(score, 1), "reason": "compressed base + breakout + volume expansion", "invalidation": f"close back above base low {round(rl, 2)}"}

    return {"valid": False, "reason": "no valid breakout + retest context"}


def engine_reversal_accumulation(df: pd.DataFrame) -> Dict:
    if len(df) < 60:
        return {"valid": False, "reason": "data kurang"}
    a = atr(df, 14)
    atrv = float(a.iloc[-1]) if pd.notna(a.iloc[-1]) else np.nan
    c = float(df["close"].iloc[-1])
    rv = relative_volume(df, 20)
    vr = float(rv.iloc[-1]) if pd.notna(rv.iloc[-1]) else 0.0
    lvls = liquidity_levels(df)
    t = trend_state(df)

    recent = df.iloc[-15:]
    rh = float(recent["high"].max())
    rl = float(recent["low"].min())
    width = rh - rl
    base_ok = pd.notna(atrv) and width <= 2.4 * atrv
    selloff = df["close"].iloc[-10] < df["close"].iloc[-20] and df["close"].iloc[-5] < df["close"].iloc[-10] and df["low"].iloc[-1] >= df["low"].iloc[-3]
    reclaim = df["close"].iloc[-1] > rh and df["close"].iloc[-1] > df["open"].iloc[-1]
    bullish_div = df["close"].iloc[-1] > df["close"].iloc[-3] and df["low"].iloc[-1] <= df["low"].iloc[-2]

    if base_ok and (selloff or t == "downtrend") and (reclaim or bullish_div):
        entry = rh
        stop = rl - (0.15 * atrv if pd.notna(atrv) else 0.0)
        tp1 = nearest_future_level([x for x in lvls if x > entry], entry, True) or (entry + 1.25 * (entry - stop))
        tp2 = max(entry + 2.0 * (entry - stop), entry + 1.5 * width)
        r1 = rr(entry, stop, tp1, True)
        r2 = rr(entry, stop, tp2, True)
        fp = fill_probability(c, entry, atrv, 0, 0.06 if t == "downtrend" else 0.0)
        score = score_quality(r2, fp, min(1.0, 0.35 + 0.2 * min(2.0, vr) + 0.15 * candle_efficiency(df)), 1.0)
        return {"valid": True, "setup_type": "Reversal Accumulation", "direction": "Bullish", "entry": round(entry, 2), "stoploss": round(stop, 2), "tp1": round(tp1, 2), "tp2": round(tp2, 2), "rr1": round(r1, 2) if pd.notna(r1) else None, "rr2": round(r2, 2) if pd.notna(r2) else None, "fill_prob": round(fp, 1), "score": round(score, 1), "reason": "selloff/base + absorption + reclaim", "invalidation": f"close below accumulation low {round(rl, 2)}"}

    buyoff = df["close"].iloc[-10] > df["close"].iloc[-20] and df["close"].iloc[-5] > df["close"].iloc[-10] and df["high"].iloc[-1] <= df["high"].iloc[-3]
    reclaim = df["close"].iloc[-1] < rl and df["close"].iloc[-1] < df["open"].iloc[-1]
    bearish_div = df["close"].iloc[-1] < df["close"].iloc[-3] and df["high"].iloc[-1] >= df["high"].iloc[-2]

    if base_ok and (buyoff or t == "uptrend") and (reclaim or bearish_div):
        entry = rl
        stop = rh + (0.15 * atrv if pd.notna(atrv) else 0.0)
        tp1 = nearest_future_level([x for x in lvls if x < entry], entry, False) or (entry - 1.25 * (stop - entry))
        tp2 = min(entry - 2.0 * (stop - entry), entry - 1.5 * width)
        r1 = rr(entry, stop, tp1, False)
        r2 = rr(entry, stop, tp2, False)
        fp = fill_probability(c, entry, atrv, 0, 0.06 if t == "uptrend" else 0.0)
        score = score_quality(r2, fp, min(1.0, 0.35 + 0.2 * min(2.0, vr) + 0.15 * candle_efficiency(df)), 1.0)
        return {"valid": True, "setup_type": "Reversal Accumulation", "direction": "Bearish", "entry": round(entry, 2), "stoploss": round(stop, 2), "tp1": round(tp1, 2), "tp2": round(tp2, 2), "rr1": round(r1, 2) if pd.notna(r1) else None, "rr2": round(r2, 2) if pd.notna(r2) else None, "fill_prob": round(fp, 1), "score": round(score, 1), "reason": "buyoff/base + absorption + reclaim", "invalidation": f"close above accumulation high {round(rh, 2)}"}

    return {"valid": False, "reason": "no accumulation reversal pattern confirmed"}


SETUP_ENGINES = [engine_unicorn_sniper, engine_pullback_continuation, engine_breakout_retest, engine_reversal_accumulation]


def scan_ticker(df: pd.DataFrame, ticker: str) -> List[Dict]:
    x = df[df["ticker"] == ticker].copy()
    if len(x) < 30:
        return []
    rows = []
    for engine in SETUP_ENGINES:
        try:
            r = engine(x)
        except Exception as e:
            r = {"valid": False, "reason": f"engine error: {type(e).__name__}: {e}"}
        if r.get("valid"):
            rows.append({"ticker": ticker, "bars": len(x), "timeframe": detect_timeframe_label(x), "trend_state": trend_state(x), **r})
    return rows


def detect_timeframe_label(df: pd.DataFrame) -> str:
    if len(df) < 5:
        return "unknown"
    diffs = df["date"].sort_values().diff().dropna()
    if diffs.empty:
        return "unknown"
    m = diffs.dt.total_seconds().median() / 60.0
    if m <= 2:
        return "M1"
    if m <= 10:
        return "M5"
    if m <= 20:
        return "M15"
    if m <= 90:
        return "H1"
    if m <= 240:
        return "H4"
    return "D1"


def build_chart(df: pd.DataFrame, ticker: str, row: pd.Series):
    x = df[df["ticker"] == ticker].copy()
    fig = go.Figure(go.Candlestick(x=x["date"], open=x["open"], high=x["high"], low=x["low"], close=x["close"], name=ticker))
    for name, val in [("Entry", row.get("entry")), ("Stop", row.get("stoploss")), ("TP1", row.get("tp1")), ("TP2", row.get("tp2"))]:
        if val is not None and pd.notna(val):
            fig.add_hline(y=float(val), line_dash="dash", annotation_text=f"{name}: {val}", annotation_position="top left")
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=40, b=10), title=f"{ticker} — {row.get('setup_type')} ({row.get('direction')})")
    return fig



def _standardize_yf_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw is None or len(raw) == 0:
        return pd.DataFrame()

    data = raw.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [
            "_".join([str(x) for x in col if str(x) and str(x) != "nan"]).strip("_")
            for col in data.columns
        ]

    data = data.reset_index()
    date_col = "Date" if "Date" in data.columns else "Datetime" if "Datetime" in data.columns else data.columns[0]
    mapping = {}
    for c in data.columns:
        lc = str(c).lower()
        if lc in ["open", "high", "low", "close", "volume"]:
            mapping[c] = lc
        elif c == date_col or lc in ["date", "datetime", "index"]:
            mapping[c] = "date"
    data = data.rename(columns=mapping)
    keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in data.columns]
    if len(keep) < 6:
        return pd.DataFrame()

    out = data[keep].copy()
    out["ticker"] = normalize_display_ticker(ticker)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    out = out.sort_values("date").drop_duplicates(["date"], keep="last")
    return out[["ticker", "date", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def _extract_batch_symbol(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw is None or len(raw) == 0:
        return pd.DataFrame()

    key_variants = [normalize_ticker_id(ticker), normalize_display_ticker(ticker)]
    if isinstance(raw.columns, pd.MultiIndex):
        for level in range(raw.columns.nlevels):
            level_values = [str(x).upper() for x in raw.columns.get_level_values(level)]
            for key in key_variants:
                key_u = key.upper()
                if key_u in level_values:
                    try:
                        return raw.xs(key_u, axis=1, level=level).copy()
                    except Exception:
                        try:
                            return raw.xs(key, axis=1, level=level).copy()
                        except Exception:
                            pass
        return pd.DataFrame()

    return raw.copy()


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_ohlcv_for_ticker(ticker: str, period: str, interval: str) -> pd.DataFrame:
    raw = yf.download(
        normalize_ticker_id(ticker),
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
        timeout=20,
    )
    return _standardize_yf_frame(raw, ticker)


def fetch_ohlcv_for_tickers(tickers: List[str], period: str, interval: str, batch_size: int = 25) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    results: Dict[str, pd.DataFrame] = {}
    failed: List[str] = []
    ordered = list(dict.fromkeys([normalize_ticker_id(t) for t in tickers]))

    for start in range(0, len(ordered), batch_size):
        batch = ordered[start:start + batch_size]
        batch_query = batch if len(batch) > 1 else batch[0]
        raw = None
        try:
            raw = yf.download(
                batch_query,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
                group_by="ticker",
                timeout=20,
            )
        except Exception:
            raw = None

        for t in batch:
            key = normalize_ticker_id(t)
            if key in results:
                continue
            sub = _extract_batch_symbol(raw, key)
            df = _standardize_yf_frame(sub, key)
            if not df.empty:
                results[key] = df

        missing = [t for t in batch if t not in results]
        for t in missing:
            key = normalize_ticker_id(t)
            df = pd.DataFrame()
            for attempt in range(3):
                try:
                    df = fetch_ohlcv_for_ticker(key, period=period, interval=interval)
                except Exception:
                    df = pd.DataFrame()
                if not df.empty:
                    results[key] = df
                    break
                time.sleep(0.8 * (attempt + 1))
            if key not in results:
                failed.append(key)

    return results, failed


st.set_page_config(page_title="Indonesia Setup Scanner", layout="wide")
st.title("Indonesia Setup Scanner")
st.caption("Upload CSV ticker saja. Scanner otomatis fetch OHLCV lalu langsung scan setup entry.")

if "ohlcv_data" not in st.session_state:
    st.session_state["ohlcv_data"] = pd.DataFrame()

with st.sidebar:
    st.header("Kontrol")
    period = st.selectbox("Period data", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
    interval = st.selectbox("Interval data", ["1d", "1h", "30m", "15m"], index=0)
    limit_rows = st.number_input("Batas ticker diproses", 1, 1000, 200, 1)
    min_bars = st.number_input("Minimum bar per ticker", 30, 1000, 60, 5)
    min_rr = st.number_input("Minimum RR2", 0.0, 20.0, 1.5, 0.1)
    min_fill = st.number_input("Minimum Fill Probability", 0.0, 100.0, 20.0, 1.0)
    top_n = st.number_input("Top N hasil", 10, 1000, 100, 10)
    direction_filter = st.selectbox("Filter arah", ["All", "Bullish", "Bearish"])
    setup_filter = st.selectbox("Filter setup", ["All", "Unicorn/Sniper", "Pullback Continuation", "Breakout Retest", "Reversal Accumulation"])

ticker_upload = st.file_uploader("Upload CSV ticker", type=["csv"])

if ticker_upload is None:
    st.info("CSV ticker minimal berisi kolom `ticker` atau `symbol` atau satu kolom ticker.")
    st.code("ticker\nBBCA\nTLKM\nASII")
    st.stop()

try:
    ticker_df = pd.read_csv(ticker_upload)
    tickers = standardize_ticker_list(ticker_df)[: int(limit_rows)]
except Exception as e:
    st.error(f"File ticker tidak valid: {type(e).__name__}: {e}")
    st.stop()

st.write(f"Ticker terdeteksi: **{len(tickers)}**")
st.dataframe(pd.DataFrame({"ticker": [normalize_display_ticker(t) for t in tickers]}), width="stretch", hide_index=True)


if st.button("Fetch OHLCV + Scan", type="primary"):
    if not tickers:
        st.error("Tidak ada ticker valid di CSV.")
        st.stop()

    with st.spinner("Mengunduh OHLCV dari Yahoo Finance..."):
        progress = st.progress(0)
        status = st.empty()
        ohlcv_map, failed = fetch_ohlcv_for_tickers(tickers, period=period, interval=interval, batch_size=25)

        # progress visual after the batch download completes
        for i, t in enumerate(tickers, start=1):
            status.write(f"Menyiapkan {normalize_display_ticker(t)} ({i}/{len(tickers)})")
            progress.progress(i / max(1, len(tickers)))
        status.empty()
        progress.empty()

    ohlcv_chunks = [ohlcv_map[t] for t in tickers if t in ohlcv_map and not ohlcv_map[t].empty]

    if not ohlcv_chunks:
        st.error("Tidak ada data OHLCV yang berhasil diunduh dari Yahoo Finance.")
        if failed:
            with st.expander("Ticker yang gagal diunduh"):
                st.write([normalize_display_ticker(x) for x in failed])
        st.info("Coba ubah period ke 1y/2y, interval ke 1d, atau cek apakah ticker masih aktif di Yahoo Finance.")
        st.stop()

    ohlcv = standardize_ohlcv(pd.concat(ohlcv_chunks, ignore_index=True))
    st.session_state["ohlcv_data"] = ohlcv

    st.success(f"OHLCV berhasil diunduh: {len(ohlcv)} bar dari {ohlcv['ticker'].nunique()} ticker")
    if failed:
        with st.expander("Ticker gagal diunduh"):
            st.write([normalize_display_ticker(x) for x in failed])

    st.download_button("Download OHLCV CSV", data=ohlcv.to_csv(index=False).encode("utf-8"), file_name=f"ohlcv_{period}_{interval}.csv", mime="text/csv")

    scan_rows = []
    for t in sorted(ohlcv["ticker"].unique().tolist()):
        x = ohlcv[ohlcv["ticker"] == t]
        if len(x) < min_bars:
            continue
        rows = scan_ticker(ohlcv, t)
        for r in rows:
            if r.get("rr2") is not None and r.get("rr2") >= min_rr and r.get("fill_prob") >= min_fill:
                scan_rows.append(r)

    if not scan_rows:
        st.warning("OHLCV berhasil diunduh, tetapi tidak ada setup yang lolos filter scan.")
        st.stop()

    res = pd.DataFrame(scan_rows)
    res = res[res["valid"] == True].copy()
    if direction_filter != "All":
        res = res[res["direction"] == direction_filter].copy()
    if setup_filter != "All":
        res = res[res["setup_type"] == setup_filter].copy()
    if "score" in res.columns:
        res = res.sort_values(["score", "rr2", "fill_prob"], ascending=[False, False, False])
    res = res.head(int(top_n)).reset_index(drop=True)

    st.subheader("Hasil Scan")
    cols_show = ["ticker", "setup_type", "direction", "timeframe", "trend_state", "entry", "stoploss", "tp1", "tp2", "rr1", "rr2", "fill_prob", "score", "reason", "invalidation"]
    existing = [c for c in cols_show if c in res.columns]
    st.dataframe(res[existing], width="stretch", hide_index=True)
    st.download_button("Download hasil scan CSV", data=res.to_csv(index=False).encode("utf-8"), file_name="setup_scan_results.csv", mime="text/csv")

    st.subheader("Preview Chart")
    selected = st.selectbox("Pilih ticker", res["ticker"].tolist())
    row = res[res["ticker"] == selected].iloc[0]
    fig = build_chart(ohlcv, selected, row)
    st.plotly_chart(fig, width="stretch")

    with st.expander("Lihat data mentah ticker terpilih"):
        st.dataframe(ohlcv[ohlcv["ticker"] == selected].tail(80), width="stretch", hide_index=True)

    st.markdown("""

**Catatan**
- Tidak ada tab download terpisah.
- User cukup upload CSV ticker lalu klik **Fetch OHLCV + Scan**.
- Data OHLCV hasil fetch langsung dipakai untuk logika entry setup.
- Untuk ticker Indonesia, suffix `.JK` dipakai otomatis.
""")
