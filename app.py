
from __future__ import annotations

import io
import json
import math
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


st.set_page_config(
    page_title="IDX Profit Scanner",
    page_icon="📈",
    layout="wide",
)

APP_TITLE = "IDX Profit Scanner"
APP_SUBTITLE = "Scanner ringkas untuk IDX/IHSG: fokus pada trend continuation, pullback/retest, likuiditas realistis, dan ranking kandidat."
DEFAULT_BENCHMARK = "^JKSE"
ENABLE_LIVE_FUNDAMENTALS = True
MAX_SAFE_WORKERS = 4
OHLCV_MIN_REQUEST_GAP = 0.85
_OHLCV_LOCK = threading.Lock()
_OHLCV_LAST_REQUEST_AT = 0.0
FUNDAMENTAL_MIN_REQUEST_GAP = 3.0
_FUNDAMENTAL_LOCK = threading.Lock()
_FUNDAMENTAL_LAST_REQUEST_AT = 0.0
FUNDAMENTAL_CACHE_TTL = 7 * 24 * 60 * 60
IDX_FUNDAMENTAL_UPLOAD_NAME = "idx_fundamentals.csv"


def normalize_ticker(symbol: str) -> str:
    s = str(symbol or "").strip().upper()
    if not s or s == "NAN":
        return ""
    if s in {"TICKER", "SYMBOL", "STOCK", "SAHAM", "CODE", "KODE", "HEADER", "NAMA", "NAMA_SAHAM"}:
        return ""
    if s.startswith("^"):
        return s
    if s.endswith(".JK"):
        return s
    return f"{s}.JK"


def _ticker_candidates(symbol: str) -> list[str]:
    base = str(symbol or "").strip().upper()
    if not base or base == "NAN":
        return []
    out = []
    for candidate in [base, base.replace(".JK", ""), normalize_ticker(base)]:
        candidate = str(candidate).strip().upper()
        if candidate and candidate not in out:
            out.append(candidate)
    return out



def _clean_number(value: Any) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    if isinstance(value, str):
        s = value.strip().replace(",", "").replace("%", "")
        if not s:
            return np.nan
        try:
            return float(s)
        except Exception:
            return np.nan
    return np.nan


def _normalize_fundamental_row(row: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "revenuegrowth": "revenueGrowth",
        "salesgrowth": "revenueGrowth",
        "netincomegrowth": "earningsGrowth",
        "earningsgrowth": "earningsGrowth",
        "profitmargins": "profitMargins",
        "netmargin": "profitMargins",
        "trailingpe": "trailingPE",
        "pe": "trailingPE",
        "pricetobook": "priceToBook",
        "pb": "priceToBook",
        "marketcap": "marketCap",
        "mcap": "marketCap",
        "ticker": "ticker",
        "symbol": "ticker",
        "kode": "ticker",
        "code": "ticker",
        "nama": "name",
        "name": "name",
        "source": "source",
        "updatedat": "updatedAt",
        "lastupdated": "updatedAt",
    }
    out: dict[str, Any] = {}
    for k, v in row.items():
        key = str(k).strip()
        norm = aliases.get(key.lower().replace(" ", "").replace("_", ""), key)
        if norm in {"revenueGrowth", "earningsGrowth", "profitMargins", "trailingPE", "priceToBook", "marketCap"}:
            out[norm] = _clean_number(v)
        else:
            out[norm] = v
    return out


def load_idx_fundamentals(uploaded_file_bytes: bytes | None = None) -> pd.DataFrame:
    paths = []
    if uploaded_file_bytes is not None:
        paths.append(io.BytesIO(uploaded_file_bytes))
    local_path = Path(IDX_FUNDAMENTAL_UPLOAD_NAME)
    if local_path.exists():
        paths.append(str(local_path))

    frames = []
    for source in paths:
        try:
            if hasattr(source, "read"):
                source.seek(0)
                df = pd.read_csv(source)
            else:
                df = pd.read_csv(source)
            if df is not None and not df.empty:
                frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df.columns = [str(c).strip() for c in df.columns]
    ticker_col = None
    for candidate in ["Ticker", "ticker", "Symbol", "symbol", "Kode", "kode", "Code", "code"]:
        if candidate in df.columns:
            ticker_col = candidate
            break
    if ticker_col is None:
        return pd.DataFrame()

    df = df.copy()
    df["Ticker"] = df[ticker_col].map(normalize_ticker)
    df = df[df["Ticker"].str.len() > 0].copy()
    if df.empty:
        return pd.DataFrame()

    norm_rows = []
    for _, row in df.iterrows():
        norm_rows.append(_normalize_fundamental_row(row.to_dict()))
    out = pd.DataFrame(norm_rows)
    if "Ticker" not in out.columns and "ticker" in out.columns:
        out["Ticker"] = out["ticker"]
    if "Ticker" not in out.columns:
        return pd.DataFrame()
    out["Ticker"] = out["Ticker"].map(normalize_ticker)
    return out.drop_duplicates("Ticker").reset_index(drop=True)


def _merge_fundamental_sources(idx_row: dict[str, Any] | None, yf_info: dict[str, Any] | None) -> dict[str, Any]:
    idx_row = idx_row or {}
    yf_info = yf_info or {}
    merged = dict(idx_row)
    for k in ["revenueGrowth", "earningsGrowth", "profitMargins", "trailingPE", "priceToBook", "marketCap", "shortName", "longName"]:
        if k not in merged or pd.isna(merged.get(k)) or merged.get(k) in {None, ""}:
            if k in yf_info:
                merged[k] = yf_info.get(k)
    return merged


@st.cache_data(ttl=FUNDAMENTAL_CACHE_TTL, show_spinner=False)
def load_yf_info(symbol: str) -> dict[str, Any]:
    if not ENABLE_LIVE_FUNDAMENTALS:
        return {}

    for candidate in _ticker_candidates(symbol):
        for attempt in range(2):
            try:
                with _FUNDAMENTAL_LOCK:
                    elapsed = time.monotonic() - _FUNDAMENTAL_LAST_REQUEST_AT
                    if elapsed < FUNDAMENTAL_MIN_REQUEST_GAP:
                        time.sleep(FUNDAMENTAL_MIN_REQUEST_GAP - elapsed)
                    globals()["_FUNDAMENTAL_LAST_REQUEST_AT"] = time.monotonic()
                ticker = yf.Ticker(candidate)
                info = ticker.get_info()
                if isinstance(info, dict) and info:
                    return info
            except Exception:
                if attempt == 0:
                    time.sleep(0.75)
                continue
    return {}


@st.cache_data(ttl=FUNDAMENTAL_CACHE_TTL, show_spinner=False)
def load_fundamental_snapshot(symbol: str, use_live: bool = False, idx_frame: pd.DataFrame | None = None) -> dict[str, Any]:
    ticker = normalize_ticker(symbol)
    idx_row: dict[str, Any] = {}
    if idx_frame is not None and not idx_frame.empty and "Ticker" in idx_frame.columns:
        hit = idx_frame[idx_frame["Ticker"] == ticker]
        if not hit.empty:
            idx_row = hit.iloc[0].to_dict()

    yf_info = load_yf_info(symbol) if use_live else {}
    merged = _merge_fundamental_sources(idx_row, yf_info)
    quality = _fundamental_quality(merged)

    source_parts = []
    if idx_row:
        source_parts.append("IDX")
    if yf_info:
        source_parts.append("YF")
    quality["fund_source"] = "+".join(source_parts) if source_parts else "NONE"
    quality["fund_available"] = bool(quality.get("fund_available", 0.0))
    quality["fund_live_used"] = bool(yf_info)
    quality["fund_idx_used"] = bool(idx_row)
    return quality


def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:

    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        if "Open" in out.columns.get_level_values(0):
            out.columns = out.columns.get_level_values(0)
        else:
            out.columns = out.columns.get_level_values(-1)

    out.columns = [str(c).strip() for c in out.columns]
    rename_map = {}
    if "Adj Close" not in out.columns and "AdjClose" in out.columns:
        rename_map["AdjClose"] = "Adj Close"
    if rename_map:
        out = out.rename(columns=rename_map)

    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not set(needed).issubset(set(out.columns)):
        return pd.DataFrame()

    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index, errors="coerce")
        except Exception:
            return pd.DataFrame()

    out = out[~out.index.isna()].copy()
    out = out.loc[:, ~out.columns.duplicated()].copy()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")].copy()

    for col in needed + (["Adj Close"] if "Adj Close" in out.columns else []):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=needed).copy()
    return out if not out.empty else pd.DataFrame()

def _throttled_yf_download(candidate: str, period: str) -> pd.DataFrame:
    global _OHLCV_LAST_REQUEST_AT
    with _OHLCV_LOCK:
        elapsed = time.monotonic() - _OHLCV_LAST_REQUEST_AT
        if elapsed < OHLCV_MIN_REQUEST_GAP:
            time.sleep(OHLCV_MIN_REQUEST_GAP - elapsed)
        _OHLCV_LAST_REQUEST_AT = time.monotonic()

        return yf.download(
            candidate,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            timeout=20,
        )


@st.cache_data(ttl=900, show_spinner=False)
def load_price_history(symbol: str, months: int = 12) -> pd.DataFrame:
    try:
        months = max(3, int(months))
    except Exception:
        months = 12

    candidates = _ticker_candidates(symbol)
    if not candidates:
        return pd.DataFrame()

    periods = [f"{months}mo"]
    if months > 6:
        periods.append("6mo")
    if months > 3:
        periods.append("3mo")

    for candidate in candidates:
        for period in periods:
            try:
                df = _throttled_yf_download(candidate, period)
                df = _standardize_ohlcv(df)
                if not df.empty:
                    df.attrs["source"] = f"yfinance:{candidate}:{period}"
                    try:
                        df.attrs["last_date"] = df.index.max().isoformat()
                    except Exception:
                        df.attrs["last_date"] = ""
                    df.attrs["rows"] = int(len(df))
                    return df
            except Exception:
                continue
    empty = pd.DataFrame()
    empty.attrs["source"] = "unavailable"
    return empty


@st.cache_data(ttl=FUNDAMENTAL_CACHE_TTL, show_spinner=False)
def load_yf_info(symbol: str) -> dict[str, Any]:
    if not ENABLE_LIVE_FUNDAMENTALS:
        return {}

    for candidate in _ticker_candidates(symbol):
        for attempt in range(2):
            try:
                ticker = yf.Ticker(candidate)
                info = ticker.get_info()
                if isinstance(info, dict) and info:
                    return info
            except Exception:
                if attempt == 0:
                    time.sleep(0.5)
                continue
    return {}


@st.cache_data(ttl=FUNDAMENTAL_CACHE_TTL, show_spinner=False)
def load_fundamental_snapshot(symbol: str, use_live: bool = False, idx_frame: pd.DataFrame | None = None) -> dict[str, Any]:
    ticker = normalize_ticker(symbol)
    idx_row: dict[str, Any] = {}
    if idx_frame is not None and not idx_frame.empty and "Ticker" in idx_frame.columns:
        hit = idx_frame[idx_frame["Ticker"] == ticker]
        if not hit.empty:
            idx_row = hit.iloc[0].to_dict()

    yf_info = load_yf_info(symbol) if use_live else {}
    merged = _merge_fundamental_sources(idx_row, yf_info)
    return _fundamental_quality(merged)


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=int(length), adjust=False, min_periods=max(1, int(length) // 2)).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return true_range(df).rolling(length, min_periods=length).mean()


def adx(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = true_range(df)
    atr_val = tr.rolling(length, min_periods=length).mean().replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(length, min_periods=length).sum() / atr_val
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(length, min_periods=length).sum() / atr_val
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.rolling(length, min_periods=length).mean().fillna(0.0)


def slope(series: pd.Series, length: int = 10) -> float:
    s = series.dropna().tail(length)
    if len(s) < 3:
        return 0.0
    x = np.arange(len(s), dtype=float)
    y = s.astype(float).values
    try:
        coef = np.polyfit(x, y, 1)[0]
        return float(coef)
    except Exception:
        return 0.0


def score_bucket(value: float | int | None, lo: float, hi: float, invert: bool = False) -> float:
    if value is None or pd.isna(value):
        return 50.0
    if hi == lo:
        return 50.0
    x = (float(value) - lo) / (hi - lo)
    x = float(np.clip(x, 0.0, 1.0))
    if invert:
        x = 1.0 - x
    return float(np.clip(x * 100.0, 0.0, 100.0))


def weighted_score(parts: list[tuple[float, float | int | None]]) -> float:
    num = 0.0
    den = 0.0
    for weight, value in parts:
        if value is None or pd.isna(value):
            continue
        num += float(weight) * float(value)
        den += float(weight)
    return float(num / den) if den > 0 else np.nan


def _safe_latest(series: pd.Series, default=np.nan):
    try:
        val = series.dropna().iloc[-1]
        return float(val)
    except Exception:
        return default


def _safe_prev(series: pd.Series, n: int = 1, default=np.nan):
    try:
        val = series.dropna().iloc[-(n + 1)]
        return float(val)
    except Exception:
        return default


def _fundamental_quality(info: dict[str, Any]) -> dict[str, float]:
    if not isinstance(info, dict) or not info:
        return {
            "fund_score": np.nan,
            "fund_available": 0.0,
            "revenue_growth": np.nan,
            "earnings_growth": np.nan,
            "profit_margins": np.nan,
            "pe": np.nan,
            "pb": np.nan,
            "market_cap": np.nan,
        }
    revenue_growth = _clean_number(info.get("revenueGrowth", info.get("revenuegrowth", info.get("salesGrowth"))))
    earnings_growth = _clean_number(info.get("earningsGrowth", info.get("earningsgrowth", info.get("netIncomeGrowth"))))
    profit_margins = _clean_number(info.get("profitMargins", info.get("profitmargins", info.get("netMargin"))))
    pe = _clean_number(info.get("trailingPE", info.get("pe")))
    pb = _clean_number(info.get("priceToBook", info.get("pb")))
    market_cap = _clean_number(info.get("marketCap", info.get("marketcap", info.get("mcap"))))

    parts = []
    if isinstance(revenue_growth, (int, float)) and not pd.isna(revenue_growth):
        parts.append(score_bucket(revenue_growth, -0.2, 0.3))
    if isinstance(earnings_growth, (int, float)) and not pd.isna(earnings_growth):
        parts.append(score_bucket(earnings_growth, -0.2, 0.35))
    if isinstance(profit_margins, (int, float)) and not pd.isna(profit_margins):
        parts.append(score_bucket(profit_margins, -0.05, 0.25))
    if isinstance(pe, (int, float)) and pe > 0:
        parts.append(score_bucket(pe, 7, 35, invert=True))
    if isinstance(pb, (int, float)) and pb > 0:
        parts.append(score_bucket(pb, 0.6, 8, invert=True))

    fund_score = float(np.clip(np.mean(parts), 0.0, 100.0)) if parts else np.nan

    return {
        "fund_score": fund_score,
        "fund_available": 1.0 if parts else 0.0,
        "revenue_growth": float(revenue_growth) if isinstance(revenue_growth, (int, float)) else np.nan,
        "earnings_growth": float(earnings_growth) if isinstance(earnings_growth, (int, float)) else np.nan,
        "profit_margins": float(profit_margins) if isinstance(profit_margins, (int, float)) else np.nan,
        "pe": float(pe) if isinstance(pe, (int, float)) else np.nan,
        "pb": float(pb) if isinstance(pb, (int, float)) else np.nan,
        "market_cap": float(market_cap) if isinstance(market_cap, (int, float)) else np.nan,
    }


@dataclass
class AnalysisResult:
    ticker: str
    name: str
    setup: str
    action: str
    score: float
    entry: float
    stoploss: float
    takeprofit1: float
    takeprofit2: float
    rr_to_tp1: float
    rr_to_tp2: float
    close: float
    ema20: float
    ema50: float
    ema200: float
    rsi14: float
    adx14: float
    atr14: float
    avg_volume20: float
    avg_rupiah_volume20: float
    rel_volume20: float
    trend_score: float
    structure_score: float
    liquidity_score: float
    rr_score: float
    rs_score: float
    fund_score: float
    setup_score: float
    reasons: str
    valid: bool


def analyze_symbol(symbol: str, months: int, benchmark_df: pd.DataFrame | None = None, universe_tag: str = "") -> dict[str, Any]:
    symbol = str(symbol or "").strip().upper()
    if not symbol:
        return {}
    df = load_price_history(symbol, months=months)
    if df.empty or len(df) < 60:
        return {}

    ohlcv_source = str(df.attrs.get("source", "yfinance"))
    ohlcv_last_date = str(df.attrs.get("last_date", ""))
    ohlcv_rows = int(df.attrs.get("rows", len(df)))

    df = df.copy()
    df["EMA20"] = ema(df["Close"], 20)
    df["EMA50"] = ema(df["Close"], 50)
    df["EMA200"] = ema(df["Close"], 200)
    df["RSI14"] = rsi(df["Close"], 14)
    df["ATR14"] = atr(df, 14)
    df["ADX14"] = adx(df, 14)
    df["VolSMA20"] = df["Volume"].rolling(20, min_periods=20).mean()
    df["High20"] = df["High"].rolling(20, min_periods=20).max().shift(1)
    df["Low20"] = df["Low"].rolling(20, min_periods=20).min().shift(1)
    df["High55"] = df["High"].rolling(55, min_periods=55).max().shift(1)
    df["Low55"] = df["Low"].rolling(55, min_periods=55).min().shift(1)
    df["Range20"] = (df["High"].rolling(20).max() - df["Low"].rolling(20).min()) / df["Close"]
    df["AvgRupiahVol20"] = df["Close"] * df["VolSMA20"]
    last = df.iloc[-1]
    prev = df.iloc[-2]

    close = float(last["Close"])
    ema20_v = float(last["EMA20"]) if pd.notna(last["EMA20"]) else np.nan
    ema50_v = float(last["EMA50"]) if pd.notna(last["EMA50"]) else np.nan
    ema200_v = float(last["EMA200"]) if pd.notna(last["EMA200"]) else np.nan
    rsi_v = float(last["RSI14"]) if pd.notna(last["RSI14"]) else 50.0
    adx_v = float(last["ADX14"]) if pd.notna(last["ADX14"]) else 0.0
    atr_v = float(last["ATR14"]) if pd.notna(last["ATR14"]) else np.nan
    avg_vol20 = float(last["VolSMA20"]) if pd.notna(last["VolSMA20"]) else np.nan
    avg_rupiah20 = float(last["AvgRupiahVol20"]) if pd.notna(last["AvgRupiahVol20"]) else np.nan
    rel_vol20 = float(last["Volume"] / avg_vol20) if pd.notna(avg_vol20) and avg_vol20 > 0 else np.nan

    trend_ok = all([
        pd.notna(ema20_v), pd.notna(ema50_v),
        close > ema20_v > ema50_v,
    ])
    long_term_ok = bool(pd.notna(ema200_v) and close > ema200_v) if not pd.isna(ema200_v) else False
    ema20_slope = slope(df["EMA20"], 8)
    ema50_slope = slope(df["EMA50"], 8)
    ema200_slope = slope(df["EMA200"], 8)

    recent_high20 = float(last["High20"]) if pd.notna(last["High20"]) else np.nan
    recent_high55 = float(last["High55"]) if pd.notna(last["High55"]) else np.nan
    recent_low20 = float(last["Low20"]) if pd.notna(last["Low20"]) else np.nan
    recent_low55 = float(last["Low55"]) if pd.notna(last["Low55"]) else np.nan
    range20 = float(last["Range20"]) if pd.notna(last["Range20"]) else np.nan

    # Base / structure measures
    base_tightness = 100.0 - score_bucket(range20, 0.04, 0.25)
    if pd.isna(base_tightness):
        base_tightness = 50.0
    distance_to_ema20_atr = (close - ema20_v) / atr_v if pd.notna(atr_v) and atr_v > 0 and pd.notna(ema20_v) else np.nan
    distance_to_ema50_atr = (close - ema50_v) / atr_v if pd.notna(atr_v) and atr_v > 0 and pd.notna(ema50_v) else np.nan
    breakout_distance_atr = (close - recent_high20) / atr_v if pd.notna(atr_v) and atr_v > 0 and pd.notna(recent_high20) else np.nan

    # Setup classification
    pullback_like = False
    breakout_like = False
    pullback_strength = 0.0
    breakout_strength = 0.0

    if trend_ok:
        if pd.notna(distance_to_ema20_atr) and -0.85 <= distance_to_ema20_atr <= 0.85:
            pullback_like = True
            pullback_strength += 35
        if pd.notna(distance_to_ema50_atr) and -0.95 <= distance_to_ema50_atr <= 0.55:
            pullback_strength += 18
        if last["Close"] > last["Open"] and last["Close"] > prev["Close"]:
            pullback_strength += 10
        if rsi_v >= 45 and rsi_v <= 68:
            pullback_strength += 12
        if pd.notna(last["Low20"]) and abs(close - recent_low20) / close <= 0.12:
            pullback_strength += 5

        if pd.notna(breakout_distance_atr) and 0 <= breakout_distance_atr <= 0.55:
            breakout_like = True
            breakout_strength += 30
        if pd.notna(recent_high55) and close > recent_high55:
            breakout_strength += 25
        if rel_vol20 is not np.nan and pd.notna(rel_vol20) and rel_vol20 >= 1.15:
            breakout_strength += 10
        if last["Close"] > prev["Close"]:
            breakout_strength += 8
        if rsi_v >= 52:
            breakout_strength += 8

    setup = "NONE"
    setup_score = 0.0
    if pullback_like and pullback_strength >= 40:
        setup = "TREND_PULLBACK"
        setup_score = float(np.clip(pullback_strength, 0, 100))
    if breakout_like and breakout_strength > setup_score:
        setup = "BREAKOUT_RETEST"
        setup_score = float(np.clip(breakout_strength, 0, 100))
    if setup == "NONE" and trend_ok and long_term_ok:
        setup = "TREND_CONTINUATION"
        setup_score = 38.0 + float(np.clip((rsi_v - 45) * 1.5, 0, 22))

    # Entry / stop / target rules
    if setup == "TREND_PULLBACK":
        entry = float(min(close, ema20_v if pd.notna(ema20_v) else close))
        structural_low = float(min(recent_low20 if pd.notna(recent_low20) else close, last["Low"]))
        stoploss = float(min(structural_low - (0.45 * atr_v if pd.notna(atr_v) else 0.0), entry - (1.15 * atr_v if pd.notna(atr_v) else 0.0)))
        risk = max(entry - stoploss, 1e-9)
        tp1 = float(entry + 2.0 * risk)
        tp2 = float(entry + 3.2 * risk)
        action = "BUY_ON_PULLBACK"
    elif setup == "BREAKOUT_RETEST":
        entry = float(max(close, recent_high20 if pd.notna(recent_high20) else close))
        structural_low = float(min(recent_low20 if pd.notna(recent_low20) else close, last["Low"]))
        stoploss = float(min(entry - (1.20 * atr_v if pd.notna(atr_v) else 0.0), structural_low - (0.25 * atr_v if pd.notna(atr_v) else 0.0)))
        risk = max(entry - stoploss, 1e-9)
        tp1 = float(entry + 2.2 * risk)
        tp2 = float(entry + 3.6 * risk)
        action = "BUY_ON_BREAKOUT_RETEST"
    elif setup == "TREND_CONTINUATION":
        entry = float(close)
        stoploss = float(min(entry - (1.25 * atr_v if pd.notna(atr_v) else 0.0), ema50_v if pd.notna(ema50_v) else entry * 0.95))
        risk = max(entry - stoploss, 1e-9)
        tp1 = float(entry + 2.0 * risk)
        tp2 = float(entry + 3.0 * risk)
        action = "WATCH"
    else:
        entry = float(close)
        stoploss = float(close - (1.35 * atr_v if pd.notna(atr_v) else close * 0.05))
        risk = max(entry - stoploss, 1e-9)
        tp1 = float(entry + 2.0 * risk)
        tp2 = float(entry + 3.0 * risk)
        action = "SKIP"

    rr1 = float((tp1 - entry) / max(entry - stoploss, 1e-9))
    rr2 = float((tp2 - entry) / max(entry - stoploss, 1e-9))

    # Scores
    liquidity_score = score_bucket(avg_rupiah20 if pd.notna(avg_rupiah20) else np.nan, 300_000_000, 5_000_000_000)
    if pd.notna(avg_rupiah20) and avg_rupiah20 < 300_000_000:
        liquidity_score *= 0.65

    trend_score = 0.0
    trend_score += 30 if trend_ok else 0
    trend_score += 10 if long_term_ok else 0
    trend_score += score_bucket(ema20_slope, -10, 10)
    trend_score += score_bucket(ema50_slope, -8, 8)
    trend_score += score_bucket(ema200_slope, -4, 4)
    trend_score += score_bucket(rsi_v, 42, 72)
    trend_score += score_bucket(adx_v, 12, 34)
    trend_score = float(np.clip(trend_score / 6.0, 0, 100))

    rs_score = 50.0
    if benchmark_df is not None and not benchmark_df.empty and symbol != normalize_ticker(DEFAULT_BENCHMARK):
        try:
            benchmark_df = benchmark_df.copy()
            bench = benchmark_df.reindex(df.index).ffill().bfill()
            if "Close" in bench.columns:
                bench_close = bench["Close"]
            else:
                bench_close = bench.iloc[:, 0]
            rs_series = df["Close"] / bench_close.replace(0, np.nan)
            rs_score = score_bucket(slope(rs_series, 10), -0.02, 0.03)
        except Exception:
            rs_score = 50.0
    else:
        rs_score = score_bucket(slope(df["Close"] / df["Close"].rolling(20).mean().replace(0, np.nan), 10), -0.01, 0.02)

    # Fundamental dipakai lebih agresif untuk kandidat investasi/growth;
    # untuk ticker yang lebih trading-oriented, kita pakai cache/IDX snapshot dulu dan live lookup hanya bila diperlukan.
    use_live_fundamental = ENABLE_LIVE_FUNDAMENTALS and (str(universe_tag).upper() == "GROWTHWATCH")
    idx_snapshot = None
    try:
        idx_snapshot = st.session_state.get("idx_fundamentals_snapshot")
    except Exception:
        idx_snapshot = None
    fund = load_fundamental_snapshot(symbol, use_live=use_live_fundamental, idx_frame=idx_snapshot)
    fund_score = fund.get("fund_score", np.nan)
    fund_available = bool(fund.get("fund_available", False))
    fund_source = str(fund.get("fund_source", "NONE"))

    rr_score = 0.0
    rr_score += score_bucket(rr1, 1.5, 4.0)
    rr_score += score_bucket(rr2, 2.5, 6.0)
    rr_score /= 2.0

    structure_score = 0.0
    structure_score += score_bucket(base_tightness, 35, 95)
    if pd.notna(distance_to_ema20_atr):
        structure_score += score_bucket(abs(distance_to_ema20_atr), 0.0, 1.6, invert=True)
    if pd.notna(distance_to_ema50_atr):
        structure_score += score_bucket(abs(distance_to_ema50_atr), 0.0, 2.0, invert=True)
    if pd.notna(breakout_distance_atr):
        structure_score += score_bucket(abs(breakout_distance_atr), 0.0, 1.2, invert=True)
    structure_score /= 3.0

    # Small-cap / momentum penalty when liquidity is thin
    penalty = 0.0
    if pd.notna(avg_rupiah20) and avg_rupiah20 < 200_000_000:
        penalty += 14
    if rsi_v > 82:
        penalty += 8
    if adx_v < 12:
        penalty += 10
    if setup == "NONE":
        penalty += 18

    total_score = weighted_score([
        (0.22, trend_score),
        (0.18, structure_score),
        (0.16, liquidity_score),
        (0.16, rr_score),
        (0.14, rs_score),
        (0.10, fund_score if fund_available else np.nan),
        (0.04, setup_score),
    ])
    if pd.isna(total_score):
        total_score = 0.0
    total_score = float(np.clip(total_score - penalty, 0, 100))

    reasons = []
    if trend_ok:
        reasons.append("trend_up")
    if long_term_ok:
        reasons.append("above_ema200")
    if setup == "TREND_PULLBACK":
        reasons.append("pullback_to_trend")
    if setup == "BREAKOUT_RETEST":
        reasons.append("breakout_retest")
    if pd.notna(avg_rupiah20):
        reasons.append(f"avg_rp_vol20={avg_rupiah20/1e6:.0f}jt")
    if pd.notna(rel_vol20):
        reasons.append(f"rel_vol={rel_vol20:.2f}")
    if pd.notna(rsi_v):
        reasons.append(f"rsi={rsi_v:.1f}")
    if pd.notna(adx_v):
        reasons.append(f"adx={adx_v:.1f}")

    valid = bool(
        setup in {"TREND_PULLBACK", "BREAKOUT_RETEST", "TREND_CONTINUATION"} and
        pd.notna(avg_rupiah20) and avg_rupiah20 >= 300_000_000 and
        total_score >= 55 and
        rr2 >= 2.2
    )

    name = ""
    try:
        name = str(yf_info.get("shortName") or yf_info.get("longName") or symbol.replace(".JK", "")).strip()
    except Exception:
        name = symbol.replace(".JK", "")

    return {
        "Ticker": symbol,
        "Name": name,
        "Setup": setup,
        "Action": action,
        "Score": round(total_score, 2),
        "OHLCVSource": ohlcv_source,
        "OHLCVLastDate": ohlcv_last_date,
        "OHLCVRows": ohlcv_rows,
        "FundSource": fund_source,
        "FundAvailable": fund_available,
        "Entry": round(entry, 4),
        "Stoploss": round(stoploss, 4),
        "TakeProfit1": round(tp1, 4),
        "TakeProfit2": round(tp2, 4),
        "RR_to_TP1": round(rr1, 2),
        "RR_to_TP2": round(rr2, 2),
        "Close": round(close, 4),
        "EMA20": round(ema20_v, 4) if pd.notna(ema20_v) else np.nan,
        "EMA50": round(ema50_v, 4) if pd.notna(ema50_v) else np.nan,
        "EMA200": round(ema200_v, 4) if pd.notna(ema200_v) else np.nan,
        "RSI14": round(rsi_v, 2),
        "ADX14": round(adx_v, 2),
        "ATR14": round(atr_v, 4) if pd.notna(atr_v) else np.nan,
        "AvgVolume20": round(avg_vol20, 2) if pd.notna(avg_vol20) else np.nan,
        "AvgRupiahVolume20": round(avg_rupiah20, 2) if pd.notna(avg_rupiah20) else np.nan,
        "RelVolume20": round(rel_vol20, 2) if pd.notna(rel_vol20) else np.nan,
        "TrendScore": round(trend_score, 2),
        "StructureScore": round(structure_score, 2),
        "LiquidityScore": round(liquidity_score, 2),
        "RRScore": round(rr_score, 2),
        "RSScore": round(rs_score, 2),
        "FundScore": round(float(fund_score), 2) if pd.notna(fund_score) else np.nan,
        "SetupScore": round(setup_score, 2),
        "Reasons": ", ".join(reasons),
        "Valid": valid,
        "RevenueGrowth": fund["revenue_growth"],
        "EarningsGrowth": fund["earnings_growth"],
        "ProfitMargins": fund["profit_margins"],
        "PE": fund["pe"],
        "PB": fund["pb"],
        "MarketCap": fund["market_cap"],
    }


def parse_universe_csv(uploaded_file) -> list[str]:
    if uploaded_file is None:
        return []
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, header=None)
        except Exception:
            return []
    if df is None or df.empty:
        return []
    col = None
    for candidate in ["Ticker", "ticker", "Symbol", "symbol", "Kode", "kode", "Saham", "saham"]:
        if candidate in df.columns:
            col = candidate
            break
    if col is None:
        col = df.columns[0]
    tickers = []
    for raw in df[col].astype(str).tolist():
        t = normalize_ticker(raw)
        if t and t not in tickers:
            tickers.append(t)
    return tickers


def scan_universe(tickers: list[tuple[str, str]] | list[str], months: int, benchmark_symbol: str, workers: int) -> pd.DataFrame:
    benchmark_df = load_price_history(benchmark_symbol, months=max(months, 12))
    results = []
    total = len(tickers)
    if total == 0:
        return pd.DataFrame()

    workers = max(1, min(int(workers), MAX_SAFE_WORKERS))
    normalized = []
    for item in tickers:
        if isinstance(item, tuple) and len(item) >= 2:
            normalized.append((item[0], item[1]))
        else:
            normalized.append((str(item), ""))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(analyze_symbol, ticker, months, benchmark_df, tag): ticker for ticker, tag in normalized}
        for idx, future in enumerate(as_completed(futures), start=1):
            try:
                row = future.result()
                if row:
                    results.append(row)
            except Exception as exc:
                ticker = futures.get(future, "")
                results.append({"Ticker": ticker, "Name": "", "Setup": "ERROR", "Action": "ERROR", "Score": 0.0, "Reasons": str(exc), "Valid": False})
            progress = idx / total
            st.session_state["_scan_progress"] = progress
    if not results:
        return pd.DataFrame()
    out = pd.DataFrame(results)
    out = out.sort_values(["Valid", "Score", "RR_to_TP2", "LiquidityScore"], ascending=[False, False, False, False]).reset_index(drop=True)
    return out


def default_universe_df() -> pd.DataFrame:
    # Curated starter universe; this is a broad watchlist, not a claim that every row is a current index constituent.
    rows = [
        # core leaders from current official indices
        ("AADI","CoreLeader"),("ACES","CoreLeader"),("ADMR","CoreLeader"),("ADRO","CoreLeader"),("AKRA","CoreLeader"),
        ("AMMN","CoreLeader"),("AMRT","CoreLeader"),("ANTM","CoreLeader"),("ARCI","CoreLeader"),("ARTO","CoreLeader"),
        ("ASII","CoreLeader"),("AVIA","CoreLeader"),("BBCA","CoreLeader"),("BBNI","CoreLeader"),("BBRI","CoreLeader"),
        ("BBTN","CoreLeader"),("BBYB","CoreLeader"),("BKSL","CoreLeader"),("BMRI","CoreLeader"),("BREN","CoreLeader"),
        ("BRIS","CoreLeader"),("BRMS","CoreLeader"),("BRPT","CoreLeader"),("BSDE","CoreLeader"),("BTPS","CoreLeader"),
        ("BUKA","CoreLeader"),("BULL","CoreLeader"),("BUMI","CoreLeader"),("BUVA","CoreLeader"),("CBDK","CoreLeader"),
        ("CMRY","CoreLeader"),("CPIN","CoreLeader"),("CTRA","CoreLeader"),("CUAN","CoreLeader"),("DEWA","CoreLeader"),
        ("DSNG","CoreLeader"),("DSSA","CoreLeader"),("ELSA","CoreLeader"),("EMTK","CoreLeader"),("ENRG","CoreLeader"),
        ("ERAA","CoreLeader"),("ESSA","CoreLeader"),("EXCL","CoreLeader"),("FILM","CoreLeader"),("GOTO","CoreLeader"),
        ("HEAL","CoreLeader"),("HMSP","CoreLeader"),("HRTA","CoreLeader"),("HRUM","CoreLeader"),("ICBP","CoreLeader"),
        ("IMPC","CoreLeader"),("INCO","CoreLeader"),("INDF","CoreLeader"),("INDY","CoreLeader"),("INET","CoreLeader"),
        ("INKP","CoreLeader"),("INTP","CoreLeader"),("ISAT","CoreLeader"),("ITMG","CoreLeader"),("JPFA","CoreLeader"),
        ("JSMR","CoreLeader"),("KIJA","CoreLeader"),("KLBF","CoreLeader"),("KPIG","CoreLeader"),("MAPA","CoreLeader"),
        ("MAPI","CoreLeader"),("MBMA","CoreLeader"),("MDKA","CoreLeader"),("MEDC","CoreLeader"),("MIKA","CoreLeader"),
        ("MTEL","CoreLeader"),("MYOR","CoreLeader"),("NCKL","CoreLeader"),("PANI","CoreLeader"),("PGAS","CoreLeader"),
        ("PGEO","CoreLeader"),("PNLF","CoreLeader"),("PSAB","CoreLeader"),("PTBA","CoreLeader"),("PTRO","CoreLeader"),
        ("PWON","CoreLeader"),("RAJA","CoreLeader"),("RATU","CoreLeader"),("SCMA","CoreLeader"),("SGER","CoreLeader"),
        ("SIDO","CoreLeader"),("SMGR","CoreLeader"),("SMIL","CoreLeader"),("SMRA","CoreLeader"),("SSIA","CoreLeader"),
        ("TAPG","CoreLeader"),("TCPI","CoreLeader"),("TINS","CoreLeader"),("TLKM","CoreLeader"),("TOBA","CoreLeader"),
        ("TOWR","CoreLeader"),("TPIA","CoreLeader"),("UNTR","CoreLeader"),("UNVR","CoreLeader"),("WIFI","CoreLeader"),
        ("WIRG","CoreLeader"),

        # more growth/liquid names from current market watchlists and sectors
        ("AALI","GrowthWatch"),("ACST","GrowthWatch"),("ADHI","GrowthWatch"),("ADMF","GrowthWatch"),("AGII","GrowthWatch"),
        ("AGRO","GrowthWatch"),("APEX","GrowthWatch"),("ARNA","GrowthWatch"),("ASGR","GrowthWatch"),("ASJT","GrowthWatch"),
        ("AUTO","GrowthWatch"),("BACA","GrowthWatch"),("BDMN","GrowthWatch"),("BFIN","GrowthWatch"),("BIRD","GrowthWatch"),
        ("BISI","GrowthWatch"),("BJBR","GrowthWatch"),("BJTM","GrowthWatch"),("BNGA","GrowthWatch"),("BNLI","GrowthWatch"),
        ("BOGA","GrowthWatch"),("BOSS","GrowthWatch"),("BRAM","GrowthWatch"),("BSSR","GrowthWatch"),("CASS","GrowthWatch"),
        ("CFIN","GrowthWatch"),("CMNP","GrowthWatch"),("CLEO","GrowthWatch"),("COCO","GrowthWatch"),("DILD","GrowthWatch"),
        ("DMAS","GrowthWatch"),("DOID","GrowthWatch"),("ELTY","GrowthWatch"),("GEMS","GrowthWatch"),("GJTL","GrowthWatch"),
        ("HDFA","GrowthWatch"),("HRME","GrowthWatch"),("IMAS","GrowthWatch"),("INTA","GrowthWatch"),("KBLI","GrowthWatch"),
        ("KKGI","GrowthWatch"),("LSIP","GrowthWatch"),("MIDI","GrowthWatch"),("MLPL","GrowthWatch"),("MPMX","GrowthWatch"),
        ("MTDL","GrowthWatch"),("NISP","GrowthWatch"),("NOBU","GrowthWatch"),("PNBN","GrowthWatch"),("PPRE","GrowthWatch"),
        ("PPRO","GrowthWatch"),("PRDA","GrowthWatch"),("RALS","GrowthWatch"),("SMDR","GrowthWatch"),("SPTO","GrowthWatch"),
        ("TKIM","GrowthWatch"),("TBLA","GrowthWatch"),("TRIS","GrowthWatch"),("TSPC","GrowthWatch"),("UNIC","GrowthWatch"),
        ("WTON","GrowthWatch"),("WIKA","GrowthWatch"),("WINS","GrowthWatch"),("WEGE","GrowthWatch"),("UNTD","GrowthWatch"),
        ("ROTI","GrowthWatch"),("SKLT","GrowthWatch"),("MERK","GrowthWatch"),("PEHA","GrowthWatch"),("PYFA","GrowthWatch"),
        ("KAEF","GrowthWatch"),("AVIA","GrowthWatch"),("ANJT","GrowthWatch"),("BEEF","GrowthWatch"),("BMAS","GrowthWatch"),
        ("BSIM","GrowthWatch"),("BSWD","GrowthWatch"),("CNKO","GrowthWatch"),("EAST","GrowthWatch"),("FISH","GrowthWatch"),
        ("INAF","GrowthWatch"),("JAWA","GrowthWatch"),("JATI","GrowthWatch"),("JECC","GrowthWatch"),("KMTR","GrowthWatch"),
        ("MARI","GrowthWatch"),("PNBS","GrowthWatch"),("SMAR","GrowthWatch"),("SMCB","GrowthWatch"),("TOTO","GrowthWatch"),
        ("VOKS","GrowthWatch"),("WAPO","GrowthWatch"),("WIIM","GrowthWatch"),("MBAP","GrowthWatch"),("SULI","GrowthWatch"),
        ("SRAJ","GrowthWatch"),("GGRM","GrowthWatch"),("BREN","GrowthWatch"),("BRNA","GrowthWatch"),("BIRD","GrowthWatch"),("BNII","GrowthWatch"),("POWR","GrowthWatch"),
    ]

    df = pd.DataFrame(rows, columns=["Ticker", "UniverseTag"])
    df["Ticker"] = df["Ticker"].map(normalize_ticker)
    df = df[df["Ticker"].str.len() > 0].drop_duplicates("Ticker").reset_index(drop=True)
    return df


def render_table_download(df: pd.DataFrame, filename: str, label: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv_bytes, file_name=filename, mime="text/csv")


def main():
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)

    with st.sidebar:
        st.header("Pengaturan Scan")
        universe_mode = st.radio("Universe source", ["Starter CSV", "Upload CSV"], index=0)
        universe_preset = "Combined 200"
        uploaded = None
        if universe_mode == "Starter CSV":
            universe_preset = st.selectbox(
                "Starter preset",
                ["Combined 200", "Trading only", "Multibagger only"],
                index=0,
            )
        else:
            uploaded = st.file_uploader("Upload CSV ticker", type=["csv"])
        idx_upload = st.file_uploader("Upload IDX fundamental CSV (opsional)", type=["csv"])
        months = st.slider("Lookback months", 3, 36, 12)
        benchmark_symbol = st.text_input("Benchmark", value=DEFAULT_BENCHMARK)
        min_rp_volume = st.number_input("Min avg rupiah volume 20D", value=300_000_000, step=50_000_000, min_value=0)
        min_price = st.number_input("Min price", value=100, step=50, min_value=0)
        max_price = st.number_input("Max price", value=50_000, step=100, min_value=0)
        min_score = st.slider("Min score", 0, 100, 55)
        workers = st.slider("Parallel workers", 1, 6, 3)
        max_scan = st.slider("Max ticker to scan", 20, 300, 200)
        show_all = st.checkbox("Show all scanned rows", value=False)
        if st.button("Load starter universe"):
            st.session_state["starter_universe"] = default_universe_df()

    starter_universe = st.session_state.get("starter_universe", default_universe_df())
    idx_snapshot = load_idx_fundamentals(idx_upload.getvalue() if idx_upload is not None else None)
    st.session_state["idx_fundamentals_snapshot"] = idx_snapshot
    if universe_mode == "Upload CSV" and uploaded is not None:
        tickers = parse_universe_csv(uploaded)
        universe_df = pd.DataFrame({"Ticker": tickers, "UniverseTag": ["Uploaded"] * len(tickers)})
    else:
        universe_df = starter_universe.copy()
        if universe_preset == "Trading only":
            universe_df = universe_df[universe_df["UniverseTag"].eq("CoreLeader")].copy()
        elif universe_preset == "Multibagger only":
            universe_df = universe_df[universe_df["UniverseTag"].eq("GrowthWatch")].copy()

    universe_df["Ticker"] = universe_df["Ticker"].map(normalize_ticker)
    universe_df = universe_df[universe_df["Ticker"].str.len() > 0].drop_duplicates("Ticker").reset_index(drop=True)
    if max_scan and len(universe_df) > max_scan:
        universe_df = universe_df.head(max_scan).copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Ticker in universe", len(universe_df))
    c2.metric("Min avg rupiah volume", f"Rp{int(min_rp_volume):,}".replace(",", "."))
    c3.metric("Benchmark", benchmark_symbol)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Scan", "Universe 200", "Trading CSV", "Multibagger CSV", "Method"])

    with tab1:
        st.subheader("Scanner")
        if st.button("Run scan", type="primary"):
            with st.spinner("Scanning universe..."):
                st.session_state["_scan_progress"] = 0.0
                results = scan_universe(list(zip(universe_df["Ticker"].tolist(), universe_df["UniverseTag"].fillna("").tolist())), months=months, benchmark_symbol=benchmark_symbol, workers=workers)
                if results.empty:
                    st.warning("Tidak ada hasil.")
                    st.session_state["scan_results"] = pd.DataFrame()
                else:
                    # lightweight filters suitable for small capital
                    filtered = results.copy()
                    filtered = filtered[(filtered["Close"] >= min_price) & (filtered["Close"] <= max_price)]
                    filtered = filtered[filtered["AvgRupiahVolume20"].fillna(0) >= float(min_rp_volume)]
                    filtered = filtered[filtered["Score"] >= float(min_score)]
                    st.session_state["scan_results"] = filtered.reset_index(drop=True)

        progress = float(st.session_state.get("_scan_progress", 0.0))
        if progress > 0:
            st.progress(min(1.0, max(0.0, progress)))

        results = st.session_state.get("scan_results", pd.DataFrame())
        if isinstance(results, pd.DataFrame) and not results.empty:
            top = results.copy()
            top = top.sort_values(["Valid", "Score", "RR_to_TP2", "LiquidityScore"], ascending=[False, False, False, False]).reset_index(drop=True)
            top.insert(0, "Rank", np.arange(1, len(top) + 1))
            if not show_all:
                valid_top = top[top["Valid"] == True].copy()
                top = valid_top.head(20).copy() if not valid_top.empty else top.head(20).copy()
            st.write(f"Rows: {len(top)}")
            st.dataframe(
                top[
                    [
                        "Rank","Ticker","Name","Setup","Action","Score","Close","Entry","Stoploss",
                        "TakeProfit1","TakeProfit2","RR_to_TP1","RR_to_TP2",
                        "AvgRupiahVolume20","RelVolume20","RSI14","ADX14",
                        "TrendScore","StructureScore","LiquidityScore","RRScore","RSScore","FundScore","Reasons"
                    ]
                ],
                width="stretch",
                hide_index=True,
            )
            render_table_download(top, "idx_profit_scanner_results.csv", "Download hasil scan CSV")
        else:
            st.info("Jalankan scan untuk melihat hasil.")

    with tab2:
        st.subheader("Starter universe 200")
        st.write("Universe ini dibuat sebagai watchlist awal yang lebih lebar. Bukan klaim bahwa semua nama pasti lolos kualitas saat ini.")
        st.dataframe(universe_df, width="stretch", hide_index=True)
        render_table_download(universe_df, "idx_starter_universe_200.csv", "Download starter universe CSV")

    with tab3:
        st.subheader("Trading universe CSV")
        trading_df = universe_df[universe_df["UniverseTag"].eq("CoreLeader")].copy()
        st.write("Daftar awal untuk trading: lebih condong ke leader yang relatif lebih likuid dan lebih mudah dieksekusi pada modal kecil.")
        st.dataframe(trading_df, width="stretch", hide_index=True)
        render_table_download(trading_df, "idx_trading_universe.csv", "Download trading CSV")

    with tab4:
        st.subheader("Multibagger universe CSV")
        multibagger_df = universe_df[universe_df["UniverseTag"].eq("GrowthWatch")].copy()
        st.write("Daftar awal untuk investasi: lebih condong ke growth/turnaround/watchlist yang belum tentu cocok untuk entry cepat.")
        st.dataframe(multibagger_df, width="stretch", hide_index=True)
        render_table_download(multibagger_df, "idx_multibagger_universe.csv", "Download multibagger CSV")

    with tab5:
        st.subheader("Logika scanner")
        st.markdown(
            """
Scanner ini sengaja dipersempit ke ide utama:

- trend continuation
- pullback ke EMA / struktur
- breakout yang retest
- likuiditas IDR 20D yang cukup untuk modal kecil
- skor hanya untuk ranking

Aturan ringkas yang dipakai:
- prioritas setup: **TREND_PULLBACK** > **BREAKOUT_RETEST** > **TREND_CONTINUATION**
- stoploss berbasis struktur harga + ATR, bukan ATR tunggal
- target minimal RR sehat; trade yang RR terlalu kecil akan turun prioritasnya
- fundamental live dimatikan secara default agar scan lebih stabil dan tidak mudah kena rate limit; bila dinyalakan, bobotnya tetap kecil
            """
        )
        st.caption("Sumber harga memakai yfinance .JK sebagai baseline gratis. Bila data kosong, ticker akan di-skip.")

    st.divider()
    st.caption("Catatan: universe ini adalah starter watchlist. Untuk eksekusi riil, tetap verifikasi tiap kandidat di chart dan data broker Anda.")


if __name__ == "__main__":
    main()
