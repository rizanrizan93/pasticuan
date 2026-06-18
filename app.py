import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


REQUIRED_COLS = ["ticker", "date", "open", "high", "low", "close", "volume"]


# ============================================================
# Helpers
# ============================================================
def normalize_ticker_id(id_str: str) -> str:
    s = str(id_str).strip().upper()
    if not s:
        return s
    # For Indonesia equities on Yahoo Finance, append .JK if missing.
    if "." not in s:
        s = f"{s}.JK"
    return s


def normalize_display_ticker(id_str: str) -> str:
    s = str(id_str).strip().upper()
    if s.endswith(".JK"):
        s = s[:-3]
    return s


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    aliases = {
        "symbol": "ticker",
        "code": "ticker",
        "datetime": "date",
        "timestamp": "date",
        "time": "date",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    }
    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)
    return df


def standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_columns(df)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV wajib punya kolom: {', '.join(REQUIRED_COLS)}. Kolom hilang: {', '.join(missing)}"
        )

    out = df[REQUIRED_COLS].copy()
    out["ticker"] = out["ticker"].astype(str).str.strip().str.upper()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=REQUIRED_COLS)
    out = out.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    return out.reset_index(drop=True)


def standardize_ticker_list(df: pd.DataFrame) -> List[str]:
    df = clean_columns(df)
    if "ticker" in df.columns:
        col = "ticker"
    elif "symbol" in df.columns:
        col = "symbol"
    else:
        # accept the first column if user uploads a single-column CSV
        col = df.columns[0]

    tickers = (
        df[col]
        .dropna()
        .astype(str)
        .map(lambda x: normalize_ticker_id(x))
        .map(lambda x: x.strip())
    )
    tickers = [t for t in tickers.tolist() if t]
    return list(dict.fromkeys(tickers))


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).rolling(period, min_periods=max(3, period // 3)).mean()


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rolling_swing_high(series_high: pd.Series, window: int = 3) -> pd.Series:
    return series_high.rolling(window * 2 + 1, center=True).max().eq(series_high)


def rolling_swing_low(series_low: pd.Series, window: int = 3) -> pd.Series:
    return series_low.rolling(window * 2 + 1, center=True).min().eq(series_low)


def candle_body(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs()


def candle_range(df: pd.DataFrame) -> pd.Series:
    return (df["high"] - df["low"]).replace(0, np.nan)


def relative_volume(df: pd.DataFrame, period: int = 20) -> pd.Series:
    return df["volume"] / df["volume"].rolling(period, min_periods=max(5, period // 3)).mean()


def trend_state(df: pd.DataFrame) -> str:
    if len(df) < 60:
        return "neutral"
    e20 = ema(df["close"], 20)
    e50 = ema(df["close"], 50)
    c = df["close"]

    up = (e20.iloc[-1] > e50.iloc[-1]) and (c.iloc[-1] > e20.iloc[-1])
    down = (e20.iloc[-1] < e50.iloc[-1]) and (c.iloc[-1] < e20.iloc[-1])

    if up:
        return "uptrend"
    if down:
        return "downtrend"
    return "range"


def last_swing_levels(df: pd.DataFrame, swing_window: int = 3) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    sw_high = rolling_swing_high(df["high"], swing_window)
    sw_low = rolling_swing_low(df["low"], swing_window)
    highs = [(i, float(df["high"].iloc[i])) for i in np.where(sw_high.fillna(False).values)[0]]
    lows = [(i, float(df["low"].iloc[i])) for i in np.where(sw_low.fillna(False).values)[0]]
    return highs, lows


def liquidity_levels(df: pd.DataFrame) -> List[float]:
    highs, lows = last_swing_levels(df, swing_window=3)
    levels = [h for _, h in highs] + [l for _, l in lows]
    levels += [float(df["high"].iloc[-20:].max()), float(df["low"].iloc[-20:].min())]
    return sorted(set([round(x, 4) for x in levels if pd.notna(x)]))


def nearest_future_level(levels: List[float], current_price: float, bullish: bool = True) -> Optional[float]:
    levels = [float(x) for x in levels if pd.notna(x)]
    if bullish:
        cand = [x for x in levels if x > current_price]
        return min(cand) if cand else None
    cand = [x for x in levels if x < current_price]
    return max(cand) if cand else None


def rr(entry: float, stop: float, target: float, bullish: bool = True) -> float:
    risk = entry - stop if bullish else stop - entry
    reward = target - entry if bullish else entry - target
    if risk <= 0:
        return np.nan
    return reward / risk if reward > 0 else 0.0


def fill_probability(current_close: float, entry: float, atr_value: float, age_bars: int, direction_strength: float = 0.0) -> float:
    if any(pd.isna(v) for v in [current_close, entry, atr_value]):
        return 0.0
    atr_value = max(float(atr_value), 1e-9)
    dist = abs(current_close - entry) / atr_value
    base = 100.0 * math.exp(-1.05 * dist)
    age_penalty = max(0.35, 1.0 - 0.06 * max(age_bars, 0))
    momentum_bonus = 1.0 + max(-0.15, min(0.25, direction_strength))
    prob = base * age_penalty * momentum_bonus
    return float(max(0.0, min(100.0, prob)))


def score_quality(rr2: float, fill_prob: float, confluence: float, structure: float) -> float:
    rr2 = 0.0 if pd.isna(rr2) else rr2
    rr_component = min(50.0, max(0.0, rr2 * 16.0))
    fill_component = min(25.0, max(0.0, fill_prob * 0.25))
    conf_component = min(15.0, max(0.0, confluence * 15.0))
    struct_component = min(10.0, max(0.0, structure * 10.0))
    return float(min(100.0, rr_component + fill_component + conf_component + struct_component))


def candle_efficiency(df: pd.DataFrame) -> float:
    r = candle_range(df)
    b = candle_body(df)
    if len(df) == 0 or pd.isna(r.iloc[-1]) or r.iloc[-1] <= 0:
        return 0.0
    return float(min(1.0, b.iloc[-1] / r.iloc[-1]))


def volume_expansion(df: pd.DataFrame, period: int = 20) -> float:
    rv = relative_volume(df, period=period)
    if pd.isna(rv.iloc[-1]):
        return 0.0
    return float(max(0.0, min(2.0, rv.iloc[-1])))


def detect_timeframe_label(df: pd.DataFrame) -> str:
    if len(df) < 5:
        return "unknown"
    diffs = df["date"].sort_values().diff().dropna()
    if diffs.empty:
        return "unknown"
    med_minutes = diffs.dt.total_seconds().median() / 60.0
    if med_minutes <= 2:
        return "M1"
    if med_minutes <= 10:
        return "M5"
    if med_minutes <= 20:
        return "M15"
    if med_minutes <= 90:
        return "H1"
    if med_minutes <= 240:
        return "H4"
    return "D1"


# ============================================================
# Setup Engines
# ============================================================
def engine_unicorn_sniper(df: pd.DataFrame) -> Dict:
    if len(df) < 60:
        return {"valid": False, "reason": "data kurang"}

    a = atr(df, 14)
    t = trend_state(df)
    c = float(df["close"].iloc[-1])
    atrv = float(a.iloc[-1]) if pd.notna(a.iloc[-1]) else np.nan
    highs, lows = last_swing_levels(df, swing_window=3)
    lvls = liquidity_levels(df)

    # Bullish sweep + displacement
    bullish_sweep = None
    for i in range(len(df) - 2, max(2, len(df) - 12), -1):
        prev_sw_lows = [x[1] for x in lows if x[0] < i - 1]
        if not prev_sw_lows:
            continue
        ref_low = prev_sw_lows[-1]
        swept = df["low"].iloc[i] < ref_low and df["close"].iloc[i] > ref_low
        displacement = (df["close"].iloc[i + 1] - df["open"].iloc[i + 1]) > 0 if i + 1 < len(df) else False
        if swept and displacement:
            bullish_sweep = i
            break

    if bullish_sweep is not None:
        idx = min(bullish_sweep + 1, len(df) - 1)
        entry = float(df["close"].iloc[idx])
        sweep_low = float(df["low"].iloc[bullish_sweep])
        stop = sweep_low - 0.15 * atrv if pd.notna(atrv) else sweep_low * 0.995
        tp1 = nearest_future_level([x for x in lvls if x > entry], entry, bullish=True)
        tp2 = nearest_future_level([x for x in lvls if x > (tp1 if tp1 else entry)], (tp1 if tp1 else entry), bullish=True)
        if tp1 is None:
            tp1 = entry + 1.5 * (entry - stop)
        if tp2 is None:
            tp2 = entry + 2.5 * (entry - stop)
        r1 = rr(entry, stop, tp1, bullish=True)
        r2 = rr(entry, stop, tp2, bullish=True)
        fp = fill_probability(c, entry, atrv, len(df) - bullish_sweep - 1, direction_strength=0.15 if t == "uptrend" else 0.0)
        conf = min(1.0, 0.55 + 0.15 * volume_expansion(df) + 0.15 * candle_efficiency(df))
        struct = 1.0 if t in ("uptrend", "range") else 0.7
        score = score_quality(r2, fp, conf, struct)
        return {
            "valid": True,
            "setup_type": "Unicorn/Sniper",
            "direction": "Bullish",
            "entry": round(entry, 2),
            "stoploss": round(stop, 2),
            "tp1": round(tp1, 2),
            "tp2": round(tp2, 2),
            "rr1": round(r1, 2) if pd.notna(r1) else None,
            "rr2": round(r2, 2) if pd.notna(r2) else None,
            "fill_prob": round(fp, 1),
            "score": round(score, 1),
            "reason": "liquidity sweep + displacement + retrace",
            "invalidation": f"close below sweep low {round(sweep_low, 2)}",
        }

    bearish_sweep = None
    for i in range(len(df) - 2, max(2, len(df) - 12), -1):
        prev_sw_highs = [x[1] for x in highs if x[0] < i - 1]
        if not prev_sw_highs:
            continue
        ref_high = prev_sw_highs[-1]
        swept = df["high"].iloc[i] > ref_high and df["close"].iloc[i] < ref_high
        displacement = (df["close"].iloc[i + 1] - df["open"].iloc[i + 1]) < 0 if i + 1 < len(df) else False
        if swept and displacement:
            bearish_sweep = i
            break

    if bearish_sweep is not None:
        idx = min(bearish_sweep + 1, len(df) - 1)
        entry = float(df["close"].iloc[idx])
        sweep_high = float(df["high"].iloc[bearish_sweep])
        stop = sweep_high + 0.15 * atrv if pd.notna(atrv) else sweep_high * 1.005
        tp1 = nearest_future_level([x for x in lvls if x < entry], entry, bullish=False)
        tp2 = nearest_future_level([x for x in lvls if x < (tp1 if tp1 else entry)], (tp1 if tp1 else entry), bullish=False)
        if tp1 is None:
            tp1 = entry - 1.5 * (stop - entry)
        if tp2 is None:
            tp2 = entry - 2.5 * (stop - entry)
        r1 = rr(entry, stop, tp1, bullish=False)
        r2 = rr(entry, stop, tp2, bullish=False)
        fp = fill_probability(c, entry, atrv, len(df) - bearish_sweep - 1, direction_strength=0.15 if t == "downtrend" else 0.0)
        conf = min(1.0, 0.55 + 0.15 * volume_expansion(df) + 0.15 * candle_efficiency(df))
        struct = 1.0 if t in ("downtrend", "range") else 0.7
        score = score_quality(r2, fp, conf, struct)
        return {
            "valid": True,
            "setup_type": "Unicorn/Sniper",
            "direction": "Bearish",
            "entry": round(entry, 2),
            "stoploss": round(stop, 2),
            "tp1": round(tp1, 2),
            "tp2": round(tp2, 2),
            "rr1": round(r1, 2) if pd.notna(r1) else None,
            "rr2": round(r2, 2) if pd.notna(r2) else None,
            "fill_prob": round(fp, 1),
            "score": round(score, 1),
            "reason": "liquidity sweep + displacement + retrace",
            "invalidation": f"close above sweep high {round(sweep_high, 2)}",
        }

    return {"valid": False, "reason": "no sweep + displacement found"}


def engine_pullback_continuation(df: pd.DataFrame) -> Dict:
    if len(df) < 60:
        return {"valid": False, "reason": "data kurang"}

    a = atr(df, 14)
    e20 = ema(df["close"], 20)
    e50 = ema(df["close"], 50)
    c = float(df["close"].iloc[-1])
    atrv = float(a.iloc[-1]) if pd.notna(a.iloc[-1]) else np.nan
    t = trend_state(df)
    lvls = liquidity_levels(df)

    if t == "uptrend":
        min_recent = df["low"].iloc[-8:].min()
        in_zone = min_recent <= e20.iloc[-1] + 0.35 * atrv if pd.notna(atrv) else True
        not_broken = df["close"].iloc[-1] >= e50.iloc[-1]
        reclaim = df["close"].iloc[-1] > df["open"].iloc[-1] and df["close"].iloc[-1] > df["close"].iloc[-2]
        if in_zone and not_broken and reclaim:
            entry = float(df["close"].iloc[-1])
            pullback_low = float(df["low"].iloc[-8:].min())
            stop = pullback_low - 0.15 * atrv if pd.notna(atrv) else pullback_low * 0.995
            tp1 = nearest_future_level([x for x in lvls if x > entry], entry, bullish=True) or (entry + 1.5 * (entry - stop))
            tp2 = nearest_future_level([x for x in lvls if x > tp1], tp1, bullish=True) or (entry + 2.5 * (entry - stop))
            r1 = rr(entry, stop, tp1, bullish=True)
            r2 = rr(entry, stop, tp2, bullish=True)
            fp = fill_probability(c, entry, atrv, 0, direction_strength=0.08)
            conf = min(1.0, 0.5 + 0.2 * volume_expansion(df) + 0.1 * candle_efficiency(df))
            score = score_quality(r2, fp, conf, 1.0)
            return {
                "valid": True,
                "setup_type": "Pullback Continuation",
                "direction": "Bullish",
                "entry": round(entry, 2),
                "stoploss": round(stop, 2),
                "tp1": round(tp1, 2),
                "tp2": round(tp2, 2),
                "rr1": round(r1, 2) if pd.notna(r1) else None,
                "rr2": round(r2, 2) if pd.notna(r2) else None,
                "fill_prob": round(fp, 1),
                "score": round(score, 1),
                "reason": "uptrend intact + pullback + reclaim",
                "invalidation": f"close below pullback low {round(pullback_low, 2)}",
            }

    if t == "downtrend":
        max_recent = df["high"].iloc[-8:].max()
        in_zone = max_recent >= e20.iloc[-1] - 0.35 * atrv if pd.notna(atrv) else True
        not_broken = df["close"].iloc[-1] <= e50.iloc[-1]
        reclaim = df["close"].iloc[-1] < df["open"].iloc[-1] and df["close"].iloc[-1] < df["close"].iloc[-2]
        if in_zone and not_broken and reclaim:
            entry = float(df["close"].iloc[-1])
            pullback_high = float(df["high"].iloc[-8:].max())
            stop = pullback_high + 0.15 * atrv if pd.notna(atrv) else pullback_high * 1.005
            tp1 = nearest_future_level([x for x in lvls if x < entry], entry, bullish=False) or (entry - 1.5 * (stop - entry))
            tp2 = nearest_future_level([x for x in lvls if x < tp1], tp1, bullish=False) or (entry - 2.5 * (stop - entry))
            r1 = rr(entry, stop, tp1, bullish=False)
            r2 = rr(entry, stop, tp2, bullish=False)
            fp = fill_probability(c, entry, atrv, 0, direction_strength=0.08)
            conf = min(1.0, 0.5 + 0.2 * volume_expansion(df) + 0.1 * candle_efficiency(df))
            score = score_quality(r2, fp, conf, 1.0)
            return {
                "valid": True,
                "setup_type": "Pullback Continuation",
                "direction": "Bearish",
                "entry": round(entry, 2),
                "stoploss": round(stop, 2),
                "tp1": round(tp1, 2),
                "tp2": round(tp2, 2),
                "rr1": round(r1, 2) if pd.notna(r1) else None,
                "rr2": round(r2, 2) if pd.notna(r2) else None,
                "fill_prob": round(fp, 1),
                "score": round(score, 1),
                "reason": "downtrend intact + pullback + reclaim",
                "invalidation": f"close above pullback high {round(pullback_high, 2)}",
            }

    return {"valid": False, "reason": "trend / reclaim / pullback criteria not met"}


def engine_breakout_retest(df: pd.DataFrame) -> Dict:
    if len(df) < 50:
        return {"valid": False, "reason": "data kurang"}

    a = atr(df, 14)
    c = float(df["close"].iloc[-1])
    atrv = float(a.iloc[-1]) if pd.notna(a.iloc[-1]) else np.nan
    rv = relative_volume(df, 20)
    vr = float(rv.iloc[-1]) if pd.notna(rv.iloc[-1]) else 0.0
    lvls = liquidity_levels(df)
    t = trend_state(df)

    base_n = 12
    recent = df.iloc[-base_n:]
    range_high = float(recent["high"].max())
    range_low = float(recent["low"].min())
    range_width = range_high - range_low
    avg_atr = float(a.iloc[-20:].mean()) if pd.notna(a.iloc[-20:].mean()) else np.nan
    compression = pd.notna(avg_atr) and range_width <= 2.0 * avg_atr

    breakout_up = df["close"].iloc[-1] > range_high and df["close"].iloc[-2] <= range_high
    breakout_down = df["close"].iloc[-1] < range_low and df["close"].iloc[-2] >= range_low

    if compression and breakout_up and vr >= 1.1:
        entry = range_high
        stop = range_low - 0.12 * atrv if pd.notna(atrv) else range_low * 0.995
        measured = range_high + range_width
        tp1 = nearest_future_level([x for x in lvls if x > entry], entry, bullish=True) or measured
        tp2 = max(measured, entry + 2.0 * (entry - stop))
        r1 = rr(entry, stop, tp1, bullish=True)
        r2 = rr(entry, stop, tp2, bullish=True)
        fp = fill_probability(c, entry, atrv, 0, direction_strength=0.1 if t == "uptrend" else 0.0)
        conf = min(1.0, 0.45 + 0.25 * min(2.0, vr) + 0.1 * candle_efficiency(df))
        score = score_quality(r2, fp, conf, 1.0 if t in ("uptrend", "range") else 0.65)
        return {
            "valid": True,
            "setup_type": "Breakout Retest",
            "direction": "Bullish",
            "entry": round(entry, 2),
            "stoploss": round(stop, 2),
            "tp1": round(tp1, 2),
            "tp2": round(tp2, 2),
            "rr1": round(r1, 2) if pd.notna(r1) else None,
            "rr2": round(r2, 2) if pd.notna(r2) else None,
            "fill_prob": round(fp, 1),
            "score": round(score, 1),
            "reason": "compressed base + breakout + volume expansion",
            "invalidation": f"close back below base high {round(range_high, 2)}",
        }

    if compression and breakout_down and vr >= 1.1:
        entry = range_low
        stop = range_high + 0.12 * atrv if pd.notna(atrv) else range_high * 1.005
        measured = range_low - range_width
        tp1 = nearest_future_level([x for x in lvls if x < entry], entry, bullish=False) or measured
        tp2 = min(measured, entry - 2.0 * (stop - entry))
        r1 = rr(entry, stop, tp1, bullish=False)
        r2 = rr(entry, stop, tp2, bullish=False)
        fp = fill_probability(c, entry, atrv, 0, direction_strength=0.1 if t == "downtrend" else 0.0)
        conf = min(1.0, 0.45 + 0.25 * min(2.0, vr) + 0.1 * candle_efficiency(df))
        score = score_quality(r2, fp, conf, 1.0 if t in ("downtrend", "range") else 0.65)
        return {
            "valid": True,
            "setup_type": "Breakout Retest",
            "direction": "Bearish",
            "entry": round(entry, 2),
            "stoploss": round(stop, 2),
            "tp1": round(tp1, 2),
            "tp2": round(tp2, 2),
            "rr1": round(r1, 2) if pd.notna(r1) else None,
            "rr2": round(r2, 2) if pd.notna(r2) else None,
            "fill_prob": round(fp, 1),
            "score": round(score, 1),
            "reason": "compressed base + breakout + volume expansion",
            "invalidation": f"close back above base low {round(range_low, 2)}",
        }

    return {"valid": False, "reason": "no valid breakout + retest context"}


def engine_reversal_accumulation(df: pd.DataFrame) -> Dict:
    if len(df) < 60:
        return {"valid": False, "reason": "data kurang"}

    a = atr(df, 14)
    c = float(df["close"].iloc[-1])
    atrv = float(a.iloc[-1]) if pd.notna(a.iloc[-1]) else np.nan
    rv = relative_volume(df, 20)
    vr = float(rv.iloc[-1]) if pd.notna(rv.iloc[-1]) else 0.0
    lvls = liquidity_levels(df)
    t = trend_state(df)

    recent = df.iloc[-15:]
    range_high = float(recent["high"].max())
    range_low = float(recent["low"].min())
    range_width = range_high - range_low
    base_ok = pd.notna(atrv) and range_width <= 2.4 * atrv

    selloff = (
        df["close"].iloc[-10] < df["close"].iloc[-20]
        and df["close"].iloc[-5] < df["close"].iloc[-10]
        and df["low"].iloc[-1] >= df["low"].iloc[-3]
    )
    reclaim = df["close"].iloc[-1] > range_high and df["close"].iloc[-1] > df["open"].iloc[-1]
    bullish_div = df["close"].iloc[-1] > df["close"].iloc[-3] and df["low"].iloc[-1] <= df["low"].iloc[-2]

    if base_ok and (selloff or t == "downtrend") and (reclaim or bullish_div):
        entry = range_high
        stop = range_low - 0.15 * atrv if pd.notna(atrv) else range_low * 0.995
        tp1 = nearest_future_level([x for x in lvls if x > entry], entry, bullish=True) or (entry + 1.25 * (entry - stop))
        tp2 = max(entry + 2.0 * (entry - stop), entry + range_width * 1.5)
        r1 = rr(entry, stop, tp1, bullish=True)
        r2 = rr(entry, stop, tp2, bullish=True)
        fp = fill_probability(c, entry, atrv, 0, direction_strength=0.06 if t == "downtrend" else 0.0)
        conf = min(1.0, 0.35 + 0.2 * min(2.0, vr) + 0.15 * candle_efficiency(df))
        score = score_quality(r2, fp, conf, 1.0 if t in ("downtrend", "range") else 0.7)
        return {
            "valid": True,
            "setup_type": "Reversal Accumulation",
            "direction": "Bullish",
            "entry": round(entry, 2),
            "stoploss": round(stop, 2),
            "tp1": round(tp1, 2),
            "tp2": round(tp2, 2),
            "rr1": round(r1, 2) if pd.notna(r1) else None,
            "rr2": round(r2, 2) if pd.notna(r2) else None,
            "fill_prob": round(fp, 1),
            "score": round(score, 1),
            "reason": "selloff/base + absorption + reclaim",
            "invalidation": f"close below accumulation low {round(range_low, 2)}",
        }

    buyoff = (
        df["close"].iloc[-10] > df["close"].iloc[-20]
        and df["close"].iloc[-5] > df["close"].iloc[-10]
        and df["high"].iloc[-1] <= df["high"].iloc[-3]
    )
    reclaim = df["close"].iloc[-1] < range_low and df["close"].iloc[-1] < df["open"].iloc[-1]
    bearish_div = df["close"].iloc[-1] < df["close"].iloc[-3] and df["high"].iloc[-1] >= df["high"].iloc[-2]

    if base_ok and (buyoff or t == "uptrend") and (reclaim or bearish_div):
        entry = range_low
        stop = range_high + 0.15 * atrv if pd.notna(atrv) else range_high * 1.005
        tp1 = nearest_future_level([x for x in lvls if x < entry], entry, bullish=False) or (entry - 1.25 * (stop - entry))
        tp2 = min(entry - 2.0 * (stop - entry), entry - range_width * 1.5)
        r1 = rr(entry, stop, tp1, bullish=False)
        r2 = rr(entry, stop, tp2, bullish=False)
        fp = fill_probability(c, entry, atrv, 0, direction_strength=0.06 if t == "uptrend" else 0.0)
        conf = min(1.0, 0.35 + 0.2 * min(2.0, vr) + 0.15 * candle_efficiency(df))
        score = score_quality(r2, fp, conf, 1.0 if t in ("uptrend", "range") else 0.7)
        return {
            "valid": True,
            "setup_type": "Reversal Accumulation",
            "direction": "Bearish",
            "entry": round(entry, 2),
            "stoploss": round(stop, 2),
            "tp1": round(tp1, 2),
            "tp2": round(tp2, 2),
            "rr1": round(r1, 2) if pd.notna(r1) else None,
            "rr2": round(r2, 2) if pd.notna(r2) else None,
            "fill_prob": round(fp, 1),
            "score": round(score, 1),
            "reason": "buyoff/base + absorption + reclaim",
            "invalidation": f"close above accumulation high {round(range_high, 2)}",
        }

    return {"valid": False, "reason": "no accumulation reversal pattern confirmed"}


SETUP_ENGINES = [
    engine_unicorn_sniper,
    engine_pullback_continuation,
    engine_breakout_retest,
    engine_reversal_accumulation,
]


def scan_ticker(df: pd.DataFrame, ticker: str) -> List[Dict]:
    x = df[df["ticker"] == ticker].copy()
    if len(x) < 30:
        return []

    rows = []
    for engine in SETUP_ENGINES:
        try:
            res = engine(x)
        except Exception as e:
            res = {"valid": False, "reason": f"engine error: {type(e).__name__}: {e}"}
        if res.get("valid"):
            rows.append(
                {
                    "ticker": ticker,
                    "bars": len(x),
                    "timeframe": detect_timeframe_label(x),
                    "trend_state": trend_state(x),
                    **res,
                }
            )
    return rows


def build_chart(df: pd.DataFrame, ticker: str, row: pd.Series):
    x = df[df["ticker"] == ticker].copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=x["date"],
        open=x["open"],
        high=x["high"],
        low=x["low"],
        close=x["close"],
        name=ticker,
    ))

    for name, val in [("Entry", row.get("entry")), ("Stop", row.get("stoploss")), ("TP1", row.get("tp1")), ("TP2", row.get("tp2"))]:
        if val is not None and pd.notna(val):
            fig.add_hline(y=float(val), line_dash="dash", annotation_text=f"{name}: {val}", annotation_position="top left")

    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        title=f"{ticker} — {row.get('setup_type')} ({row.get('direction')})",
    )
    return fig


def fetch_ohlcv_for_ticker(ticker: str, period: str, interval: str) -> pd.DataFrame:
    yf_ticker = normalize_ticker_id(ticker)
    data = yf.download(
        yf_ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=False,
    )

    if data is None or len(data) == 0:
        return pd.DataFrame()

    data = data.reset_index()
    date_col = "Date" if "Date" in data.columns else "Datetime" if "Datetime" in data.columns else data.columns[0]

    # Handle multiindex / standard columns from yfinance
    cols = [str(c).lower() for c in data.columns]
    mapping = {}
    for c in data.columns:
        lc = str(c).lower()
        if lc in ["open", "high", "low", "close", "volume"]:
            mapping[c] = lc
        elif lc in ["adj close", "adj_close", "adjclose"]:
            mapping[c] = "adj_close"
        elif c == date_col:
            mapping[c] = "date"
        elif lc in ["date", "datetime"]:
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
    out = out[["ticker", "date", "open", "high", "low", "close", "volume"]]
    return out.reset_index(drop=True)


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="Indonesia Setup Scanner", layout="wide")
st.title("Indonesia Setup Scanner")
st.caption("Tab 1: scan setup entry. Tab 2: unduh OHLCV ticker untuk dipakai di tab 1.")

if "ohlcv_data" not in st.session_state:
    st.session_state["ohlcv_data"] = pd.DataFrame()
if "ticker_list" not in st.session_state:
    st.session_state["ticker_list"] = []
if "scan_results" not in st.session_state:
    st.session_state["scan_results"] = pd.DataFrame()

tab1, tab2 = st.tabs(["Scan Setup", "Download OHLCV"])

with tab2:
    st.subheader("Unduh OHLCV dari daftar ticker")
    st.write("Upload CSV ticker saja. Kolom yang didukung: `ticker` / `symbol` / satu kolom tunggal.")

    ticker_upload = st.file_uploader("Upload CSV ticker", type=["csv"], key="ticker_csv_upload")
    c1, c2, c3 = st.columns(3)
    with c1:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)
    with c2:
        interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m"], index=0)
    with c3:
        limit_rows = st.number_input("Batas ticker diproses", min_value=1, max_value=1000, value=200, step=1)

    if ticker_upload is not None:
        try:
            ticker_df = pd.read_csv(ticker_upload)
            ticker_list = standardize_ticker_list(ticker_df)
            ticker_list = ticker_list[: int(limit_rows)]
            st.session_state["ticker_list"] = ticker_list

            st.write(f"Ticker terdeteksi: **{len(ticker_list)}**")
            st.dataframe(pd.DataFrame({"ticker": ticker_list}), use_container_width=True, hide_index=True)

            if st.button("Download OHLCV", type="primary"):
                progress = st.progress(0)
                status = st.empty()
                all_ohlcv = []

                for i, t in enumerate(ticker_list, start=1):
                    status.write(f"Mengunduh {t} ({i}/{len(ticker_list)})")
                    try:
                        df = fetch_ohlcv_for_ticker(t, period=period, interval=interval)
                        if not df.empty:
                            all_ohlcv.append(df)
                    except Exception as e:
                        st.warning(f"Gagal unduh {t}: {type(e).__name__}: {e}")
                    progress.progress(i / max(1, len(ticker_list)))

                status.empty()
                progress.empty()

                if all_ohlcv:
                    merged = pd.concat(all_ohlcv, ignore_index=True)
                    merged = standardize_ohlcv(merged)
                    st.session_state["ohlcv_data"] = merged
                    st.success(f"OHLCV berhasil diunduh: {len(merged)} bar dari {merged['ticker'].nunique()} ticker")
                    st.dataframe(merged.tail(100), use_container_width=True, hide_index=True)
                    csv_bytes = merged.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download OHLCV CSV",
                        data=csv_bytes,
                        file_name=f"ohlcv_{period}_{interval}.csv",
                        mime="text/csv",
                    )
                else:
                    st.error("Tidak ada data OHLCV yang berhasil diunduh.")
        except Exception as e:
            st.error(f"File ticker tidak valid: {type(e).__name__}: {e}")
    else:
        st.info("Upload CSV ticker untuk mulai unduh OHLCV.")

with tab1:
    st.subheader("Scan setup entry")
    st.write("Tab ini bisa memakai OHLCV dari upload langsung atau dari hasil unduhan tab 2 yang tersimpan di session.")

    use_downloaded = st.checkbox("Gunakan OHLCV hasil tab 2 jika tersedia", value=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        min_bars = st.number_input("Minimum bar per ticker", min_value=30, max_value=1000, value=60, step=5)
    with c2:
        min_rr = st.number_input("Minimum RR2", min_value=0.0, max_value=20.0, value=1.5, step=0.1)
    with c3:
        min_fill = st.number_input("Minimum Fill Probability", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
    with c4:
        top_n = st.number_input("Top N hasil", min_value=10, max_value=1000, value=100, step=10)

    d1, d2 = st.columns(2)
    with d1:
        direction_filter = st.selectbox("Filter arah", ["All", "Bullish", "Bearish"])
    with d2:
        setup_filter = st.selectbox(
            "Filter setup",
            ["All", "Unicorn/Sniper", "Pullback Continuation", "Breakout Retest", "Reversal Accumulation"],
        )

    scan_upload = st.file_uploader("Upload OHLCV CSV untuk scan", type=["csv"], key="ohlcv_csv_upload")

    source_df = pd.DataFrame()
    source_label = ""

    if scan_upload is not None:
        try:
            source_df = standardize_ohlcv(pd.read_csv(scan_upload))
            source_label = "upload langsung"
        except Exception as e:
            st.error(f"CSV OHLCV tidak valid: {type(e).__name__}: {e}")
            st.stop()
    elif use_downloaded and not st.session_state["ohlcv_data"].empty:
        source_df = st.session_state["ohlcv_data"].copy()
        source_label = "hasil unduhan tab 2"

    if source_df.empty:
        st.info("Upload OHLCV CSV atau gunakan OHLCV dari tab 2.")
        st.stop()

    tickers = sorted(source_df["ticker"].unique().tolist())
    st.write(f"Sumber data: **{source_label}** | Ticker: **{len(tickers)}** | Bar total: **{len(source_df)}**")

    all_rows = []
    progress = st.progress(0)
    status = st.empty()

    for i, t in enumerate(tickers, start=1):
        status.write(f"Scanning {t} ({i}/{len(tickers)})")
        x = source_df[source_df["ticker"] == t].copy()
        if len(x) < min_bars:
            progress.progress(i / len(tickers))
            continue

        try:
            rows = scan_ticker(source_df, t)
            for r in rows:
                if r.get("rr2") is not None and r.get("rr2") >= min_rr and r.get("fill_prob") >= min_fill:
                    all_rows.append(r)
        except Exception as e:
            all_rows.append(
                {
                    "ticker": t,
                    "bars": len(x),
                    "timeframe": detect_timeframe_label(x),
                    "trend_state": trend_state(x),
                    "valid": False,
                    "setup_type": "error",
                    "direction": "NA",
                    "reason": f"{type(e).__name__}: {e}",
                }
            )

        progress.progress(i / len(tickers))

    status.empty()
    progress.empty()

    if not all_rows:
        st.warning("Tidak ada setup yang lolos filter. Coba turunkan threshold RR/fill atau cek kualitas data.")
        st.stop()

    res = pd.DataFrame(all_rows)
    res = res[res["valid"] == True].copy()

    if direction_filter != "All":
        res = res[res["direction"] == direction_filter].copy()
    if setup_filter != "All":
        res = res[res["setup_type"] == setup_filter].copy()

    if "score" in res.columns:
        res = res.sort_values(["score", "rr2", "fill_prob"], ascending=[False, False, False])

    res = res.head(int(top_n)).reset_index(drop=True)
    st.session_state["scan_results"] = res.copy()

    st.subheader("Hasil Scan")
    cols_show = [
        "ticker", "setup_type", "direction", "timeframe", "trend_state",
        "entry", "stoploss", "tp1", "tp2", "rr1", "rr2", "fill_prob", "score", "reason", "invalidation"
    ]
    existing = [c for c in cols_show if c in res.columns]
    st.dataframe(res[existing], use_container_width=True, hide_index=True)

    csv_out = res.to_csv(index=False).encode("utf-8")
    st.download_button("Download hasil scan CSV", data=csv_out, file_name="setup_scan_results.csv", mime="text/csv")

    st.subheader("Preview Chart")
    selected = st.selectbox("Pilih ticker", res["ticker"].tolist())
    selected_row = res[res["ticker"] == selected].iloc[0]
    fig = build_chart(source_df, selected, selected_row)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Lihat data mentah ticker terpilih"):
        st.dataframe(source_df[source_df["ticker"] == selected].tail(80), use_container_width=True, hide_index=True)

    st.markdown(
        """
**Catatan**
- Tab 2 mengunduh OHLCV dari Yahoo Finance untuk ticker Indonesia dengan suffix `.JK`.
- Hasil unduhan disimpan di session dan bisa langsung dipakai tab 1.
- Kalau kamu upload OHLCV sendiri di tab 1, data itu akan dipakai sebagai sumber scan.
"""
    )
