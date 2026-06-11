import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


st.set_page_config(page_title="IDX Simple Trend Scanner", layout="wide")


# =========================================================
# Utilities
# =========================================================

def clip(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, x)))


def normalize_ticker(raw: str, auto_suffix: bool = True) -> str:
    t = (raw or "").strip().upper()
    if not t:
        return ""
    if t.startswith("^"):
        return t
    if "." in t:
        return t
    return f"{t}.JK" if auto_suffix else t


def round_down_to_lot(shares: float, lot_size: int = 100) -> int:
    if shares <= 0:
        return 0
    return int(math.floor(shares / lot_size) * lot_size)


def detect_ticker_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["ticker", "symbol", "code", "kode", "saham", "stock", "asset"]
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    return None


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


def dedupe_tickers(tickers: List[str]) -> List[str]:
    out = []
    seen = set()
    for t in tickers:
        t = (t or "").strip().upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def safe_float(value, default: float = np.nan) -> float:
    try:
        if value is None:
            return default
        x = float(value)
        if np.isfinite(x):
            return float(x)
    except Exception:
        pass
    return default


def normalize_ratio(value) -> float:
    x = safe_float(value, np.nan)
    if not np.isfinite(x):
        return np.nan
    # yfinance may return ratios either as decimal (0.23) or percent-like (23).
    if abs(x) > 1.5:
        x = x / 100.0
    return float(x)


def _get_statement_df(tkr: yf.Ticker, attr_names: List[str]) -> pd.DataFrame:
    for attr in attr_names:
        try:
            df = getattr(tkr, attr, None)
            if isinstance(df, pd.DataFrame) and not df.empty:
                out = df.copy()
                try:
                    out.columns = pd.to_datetime(out.columns)
                    out = out.reindex(sorted(out.columns), axis=1)
                except Exception:
                    pass
                return out
        except Exception:
            continue
    return pd.DataFrame()


def _growth_from_statement(stmt: pd.DataFrame, row_candidates: List[str]) -> float:
    if stmt is None or stmt.empty:
        return np.nan
    row_name = None
    lower_index = {str(idx).strip().lower(): idx for idx in stmt.index}
    for cand in row_candidates:
        key = str(cand).strip().lower()
        if key in lower_index:
            row_name = lower_index[key]
            break
    if row_name is None:
        return np.nan

    ser = pd.to_numeric(stmt.loc[row_name], errors="coerce").dropna()
    if ser.empty:
        return np.nan

    ser = ser.sort_index()
    latest = safe_float(ser.iloc[-1], np.nan)
    if not np.isfinite(latest):
        return np.nan

    # Prefer year-over-year quarterly comparison when possible.
    if len(ser) >= 5:
        prev = safe_float(ser.iloc[-5], np.nan)
        if np.isfinite(prev) and prev != 0:
            return float(latest / prev - 1.0)

    if len(ser) >= 2:
        prev = safe_float(ser.iloc[-2], np.nan)
        if np.isfinite(prev) and prev != 0:
            return float(latest / prev - 1.0)

    return np.nan


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_fundamentals(ticker: str) -> Dict[str, float]:
    """
    Lightweight CAN SLIM-style fundamental snapshot.
    For many IDX names, Yahoo data can be incomplete, so we combine info fields
    with quarterly statements when available.
    """
    out = {
        "revenue_growth": np.nan,
        "earnings_growth": np.nan,
        "roe": np.nan,
        "profit_margin": np.nan,
        "institutional_ownership": np.nan,
        "quarterly_revenue_growth": np.nan,
        "quarterly_earnings_growth": np.nan,
    }

    try:
        tkr = yf.Ticker(ticker)
    except Exception:
        return out

    info = {}
    try:
        info = tkr.info or {}
    except Exception:
        info = {}

    out["revenue_growth"] = normalize_ratio(info.get("revenueGrowth"))
    out["earnings_growth"] = normalize_ratio(
        info.get("earningsGrowth") if info.get("earningsGrowth") is not None else info.get("earningsQuarterlyGrowth")
    )
    out["roe"] = normalize_ratio(info.get("returnOnEquity"))
    out["profit_margin"] = normalize_ratio(info.get("profitMargins"))
    out["institutional_ownership"] = normalize_ratio(
        info.get("heldPercentInstitutions")
        if info.get("heldPercentInstitutions") is not None
        else info.get("institutionPercentHeld")
    )

    q_income = _get_statement_df(tkr, ["quarterly_income_stmt", "quarterly_financials"])
    if q_income.empty:
        q_income = _get_statement_df(tkr, ["quarterly_income_statement", "quarterly_income_stmt"])
    if not q_income.empty:
        out["quarterly_revenue_growth"] = _growth_from_statement(
            q_income,
            ["Total Revenue", "TotalRevenue", "Revenue", "Operating Revenue"],
        )
        out["quarterly_earnings_growth"] = _growth_from_statement(
            q_income,
            [
                "Net Income",
                "NetIncome",
                "Net Income Common Stockholders",
                "Net Income From Continuing Operation Net Minority Interest",
                "Operating Income",
            ],
        )

    # Use statement-derived growth as fallback when info is unavailable.
    if not np.isfinite(out["revenue_growth"]) and np.isfinite(out["quarterly_revenue_growth"]):
        out["revenue_growth"] = out["quarterly_revenue_growth"]
    if not np.isfinite(out["earnings_growth"]) and np.isfinite(out["quarterly_earnings_growth"]):
        out["earnings_growth"] = out["quarterly_earnings_growth"]

    return out


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_idx_universe(limit: int = 800) -> List[str]:
    urls = [
        "https://www.idx.co.id/en/market-data/stocks-data/stock-list/",
        "https://www.idx.co.id/id/data-pasar/data-saham/daftar-saham",
    ]
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    }

    for url in urls:
        try:
            # Try direct HTML read first. If the site blocks it, fall back to requests.
            try:
                tables = pd.read_html(url)
            except Exception:
                import requests

                resp = requests.get(url, headers=headers, timeout=20)
                resp.raise_for_status()
                tables = pd.read_html(resp.text)

            for df in tables:
                if df.empty:
                    continue
                cols = [str(c).strip().lower() for c in df.columns]
                code_cols = [i for i, c in enumerate(cols) if c in {"code", "kode", "ticker", "symbol"} or "code" in c or "kode" in c]
                if not code_cols:
                    continue
                col = df.columns[code_cols[0]]
                raw_codes = []
                for val in df[col].astype(str).tolist():
                    code = val.strip().upper()
                    if not code or code in {"NAN", "NONE"}:
                        continue
                    if re.fullmatch(r"[A-Z0-9]{1,5}", code):
                        raw_codes.append(code)
                raw_codes = dedupe_tickers(raw_codes)
                if raw_codes:
                    return [normalize_ticker(code, auto_suffix=True) for code in raw_codes[:limit]]
        except Exception:
            continue

    return []


# =========================================================
# Data
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
def build_benchmark(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    if not ticker:
        return pd.DataFrame()
    candidates = [ticker.strip()]
    if not ticker.strip().startswith("^"):
        candidates.append(normalize_ticker(ticker, auto_suffix=False))
    seen = set()
    for sym in candidates:
        if not sym or sym in seen:
            continue
        seen.add(sym)
        df = load_history(sym, period=period, interval=interval)
        if not df.empty:
            return df
    return pd.DataFrame()


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



def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    c = out["close"]
    h = out["high"]
    l = out["low"]
    v = out["volume"]

    out["ema20"] = ema(c, 20)
    out["ema50"] = ema(c, 50)
    out["ema150"] = ema(c, 150)
    out["ema200"] = ema(c, 200)
    out["sma50"] = sma(c, 50)
    out["sma200"] = sma(c, 200)
    out["rsi14"] = rsi(c, 14)
    out["atr14"] = atr(out, 14)
    out["atr_pct"] = out["atr14"] / c.replace(0, np.nan)
    out["ret20"] = c.pct_change(20)
    out["ret60"] = c.pct_change(60)
    out["ret120"] = c.pct_change(120)
    out["vol_sma20"] = sma(v, 20)
    out["vol_sma50"] = sma(v, 50)
    out["vol_rvol20"] = v / out["vol_sma20"].replace(0, np.nan)
    out["high20"] = h.rolling(20).max()
    out["high55"] = h.rolling(55).max()
    out["high252"] = h.rolling(252).max()
    out["low5"] = l.rolling(5).min()
    out["low252"] = l.rolling(252).min()
    out["ema50_20ago"] = out["ema50"].shift(20)
    out["ema150_20ago"] = out["ema150"].shift(20)
    out["ema200_20ago"] = out["ema200"].shift(20)
    out["sma200_20ago"] = out["sma200"].shift(20)
    out["ema50_rising"] = out["ema50"] > out["ema50_20ago"]
    out["ema150_rising"] = out["ema150"] > out["ema150_20ago"]
    out["ema200_rising"] = out["ema200"] > out["ema200_20ago"]
    out["sma200_rising"] = out["sma200"] > out["sma200_20ago"]
    out["near_52w_high_pct"] = out["close"] / out["high252"].replace(0, np.nan)
    out["dist_52w_high_pct"] = out["high252"] / out["close"].replace(0, np.nan) - 1.0
    out["atr20"] = sma(out["atr14"], 20)
    out["tightness_pct"] = out["atr14"] / out["close"].replace(0, np.nan)
    return out



# =========================================================
# Minervini / CAN SLIM scanner
# =========================================================

def score_benchmark(df: pd.DataFrame) -> Tuple[float, str, Dict[str, float]]:
    if df.empty or len(df) < 220:
        return np.nan, "Unknown", {}
    d = build_indicators(df)
    last = d.iloc[-1]
    score = 0.0
    detail = {
        "close_above_sma200": float(last["close"] > last["sma200"]),
        "ema50_above_sma200": float(last["ema50"] > last["sma200"]),
        "close_above_ema50": float(last["close"] > last["ema50"]),
        "rsi_above_50": float(last["rsi14"] > 50),
        "sma200_rising": float(last["sma200_rising"]),
    }
    if last["close"] > last["sma200"]:
        score += 30
    if last["ema50"] > last["sma200"]:
        score += 20
    if last["close"] > last["ema50"]:
        score += 20
    if last["rsi14"] > 50:
        score += 15
    if last["sma200_rising"]:
        score += 15
    score = clip(score)
    label = "RISK ON" if score >= 70 else "NEUTRAL" if score >= 50 else "RISK OFF"
    return score, label, detail


def _fundamental_checks(fund: Dict[str, float]) -> Dict[str, bool]:
    revenue_growth = safe_float(fund.get("revenue_growth"), np.nan)
    earnings_growth = safe_float(fund.get("earnings_growth"), np.nan)
    quarterly_revenue_growth = safe_float(fund.get("quarterly_revenue_growth"), np.nan)
    quarterly_earnings_growth = safe_float(fund.get("quarterly_earnings_growth"), np.nan)
    roe = safe_float(fund.get("roe"), np.nan)
    profit_margin = safe_float(fund.get("profit_margin"), np.nan)
    inst = safe_float(fund.get("institutional_ownership"), np.nan)

    c_ok = bool(
        (np.isfinite(revenue_growth) and revenue_growth >= 0.25)
        or (np.isfinite(earnings_growth) and earnings_growth >= 0.25)
        or (np.isfinite(quarterly_revenue_growth) and quarterly_revenue_growth >= 0.20)
        or (np.isfinite(quarterly_earnings_growth) and quarterly_earnings_growth >= 0.20)
    )
    a_ok = bool(
        (np.isfinite(roe) and roe >= 0.17)
        or (np.isfinite(earnings_growth) and earnings_growth >= 0.15)
        or (np.isfinite(profit_margin) and profit_margin >= 0.10)
    )
    i_ok = bool(np.isfinite(inst) and inst >= 0.10)

    return {
        "c_ok": c_ok,
        "a_ok": a_ok,
        "i_ok": i_ok,
    }


def evaluate_ticker(
    ticker: str,
    period: str,
    min_avg_dollar_vol: float,
    min_price: float,
    benchmark_df: Optional[pd.DataFrame],
) -> Optional[Dict]:
    df = load_history(ticker, period=period)
    if df.empty or len(df) < 220:
        return None

    df = build_indicators(df)
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last

    if not np.isfinite(last.get("sma200", np.nan)):
        return None

    # Benchmark context
    if benchmark_df is not None and not benchmark_df.empty and len(benchmark_df) >= 220:
        b = build_indicators(benchmark_df)
        b_last = b.iloc[-1]
        benchmark_close = float(b_last["close"])
        benchmark_sma200 = float(b_last["sma200"])
        benchmark_ema50 = float(b_last["ema50"])
        benchmark_risk_on = bool(benchmark_close > benchmark_sma200 and benchmark_ema50 > benchmark_sma200 and b_last["sma200_rising"])
    else:
        benchmark_close = np.nan
        benchmark_sma200 = np.nan
        benchmark_ema50 = np.nan
        benchmark_risk_on = True

    # Core liquidity filters
    avg_dollar_vol_20 = float((df["close"] * df["volume"]).tail(20).mean())
    avg_vol_20 = float(df["volume"].tail(20).mean())
    price = float(last["close"])
    liquidity_ok = bool(
        np.isfinite(avg_dollar_vol_20)
        and avg_dollar_vol_20 >= min_avg_dollar_vol
        and avg_vol_20 >= 100_000
        and price >= min_price
    )

    # Minervini-style structure
    price_structure_ok = bool(
        last["close"] > last["ema50"] > last["ema150"] > last["ema200"]
        and last["close"] > last["sma200"]
        and last["ema150"] > last["ema200"]
        and last["ema200_rising"]
        and last["sma200_rising"]
    )

    near_high_ok = bool(np.isfinite(last["near_52w_high_pct"]) and last["near_52w_high_pct"] >= 0.75)
    trend_quality_ok = bool(
        price_structure_ok
        and near_high_ok
        and np.isfinite(last["rsi14"])
        and 50 <= float(last["rsi14"]) <= 80
        and np.isfinite(last["atr_pct"])
        and float(last["atr_pct"]) <= 0.15
    )

    ret20 = float(last["ret20"])
    ret60 = float(last["ret60"])
    ret120 = float(last["ret120"]) if np.isfinite(last["ret120"]) else np.nan
    if benchmark_df is not None and not benchmark_df.empty and len(benchmark_df) >= 220:
        bench_ret20 = float(b_last["ret20"])
        bench_ret60 = float(b_last["ret60"])
        bench_ret120 = float(b_last["ret120"])
    else:
        bench_ret20 = np.nan
        bench_ret60 = np.nan
        bench_ret120 = np.nan
    rs20 = float(ret20 - bench_ret20) if np.isfinite(bench_ret20) else np.nan
    rs60 = float(ret60 - bench_ret60) if np.isfinite(bench_ret60) else np.nan
    rs120 = float(ret120 - bench_ret120) if np.isfinite(bench_ret120) else np.nan
    relative_strength_ok = bool(
        (np.isfinite(rs20) and rs20 > 0)
        and (np.isfinite(rs60) and rs60 > 0)
    )

    vol_rvol20 = float(last["vol_rvol20"])
    volume_ok = bool(np.isfinite(vol_rvol20) and vol_rvol20 >= 1.5 and last["close"] > prev["close"])

    breakout_ok = bool(
        last["close"] >= last["high20"] * 0.99
        or last["close"] >= last["high55"] * 0.975
        or last["close"] >= last["high252"] * 0.95
    )

    # Technical score (Minervini core)
    tech_checks = {
        "market_ok": benchmark_risk_on,
        "liquidity_ok": liquidity_ok,
        "price_structure_ok": price_structure_ok,
        "trend_quality_ok": trend_quality_ok,
        "relative_strength_ok": relative_strength_ok,
        "volume_ok": volume_ok,
        "breakout_ok": breakout_ok,
    }
    tech_score = int(sum(int(v) for v in tech_checks.values()))

    # Fundamental loading is expensive, so keep it cached and separate.
    fund = load_fundamentals(ticker)
    fund_checks = _fundamental_checks(fund)
    fundamental_score = int(sum(int(v) for v in fund_checks.values()))
    can_slim_ok = bool(sum(int(v) for v in fund_checks.values()) >= 2 and fund_checks["c_ok"] and fund_checks["a_ok"])

    final_score = int(tech_score + fundamental_score)

    if benchmark_risk_on and tech_checks["price_structure_ok"] and tech_checks["relative_strength_ok"] and tech_checks["breakout_ok"] and fundamental_score >= 2 and (fund_checks["c_ok"] or fund_checks["a_ok"]):
        decision = "BUY"
        grade = "ELITE"
    elif benchmark_risk_on and tech_score >= 5 and fundamental_score >= 1:
        decision = "WATCH"
        grade = "STRONG"
    elif tech_score >= 5:
        decision = "WATCH"
        grade = "WATCHLIST"
    else:
        decision = "IGNORE"
        grade = "IGNORE"

    dist_ema50_pct = float(price / last["ema50"] - 1.0) if np.isfinite(last["ema50"]) and last["ema50"] != 0 else np.nan

    return {
        "ticker": ticker,
        "final_score": final_score,
        "tech_score": tech_score,
        "fundamental_score": fundamental_score,
        "grade": grade,
        "decision": decision,
        "benchmark_risk_on": benchmark_risk_on,
        "liquidity_ok": liquidity_ok,
        "price_structure_ok": price_structure_ok,
        "trend_quality_ok": trend_quality_ok,
        "relative_strength_ok": relative_strength_ok,
        "volume_ok": volume_ok,
        "breakout_ok": breakout_ok,
        "minervini_ok": bool(benchmark_risk_on and price_structure_ok and trend_quality_ok and relative_strength_ok and volume_ok and breakout_ok),
        "can_slim_ok": can_slim_ok,
        "market_ok": benchmark_risk_on,
        "close": price,
        "rsi14": float(last["rsi14"]),
        "ret20": ret20,
        "ret60": ret60,
        "ret120": ret120,
        "rs20": rs20,
        "rs60": rs60,
        "rs120": rs120,
        "vol_rvol20": vol_rvol20,
        "atr_pct": float(last["atr_pct"]),
        "avg_dollar_vol_20": avg_dollar_vol_20,
        "avg_vol_20": avg_vol_20,
        "high20": float(last["high20"]),
        "high55": float(last["high55"]),
        "high252": float(last["high252"]),
        "ema20": float(last["ema20"]),
        "ema50": float(last["ema50"]),
        "ema150": float(last["ema150"]),
        "ema200": float(last["ema200"]),
        "sma200": float(last["sma200"]),
        "dist_ema50_pct": dist_ema50_pct,
        "revenue_growth": safe_float(fund.get("revenue_growth"), np.nan),
        "earnings_growth": safe_float(fund.get("earnings_growth"), np.nan),
        "roe": safe_float(fund.get("roe"), np.nan),
        "profit_margin": safe_float(fund.get("profit_margin"), np.nan),
        "institutional_ownership": safe_float(fund.get("institutional_ownership"), np.nan),
        "quarterly_revenue_growth": safe_float(fund.get("quarterly_revenue_growth"), np.nan),
        "quarterly_earnings_growth": safe_float(fund.get("quarterly_earnings_growth"), np.nan),
        "_df": df,
        "_checks": {**tech_checks, **fund_checks},
    }


def build_entry_plan(df: pd.DataFrame, account_size: float, risk_pct: float, lot_rounding: bool) -> Dict[str, float]:
    last = df.iloc[-1]
    close = float(last["close"])
    atr14 = float(last["atr14"])
    ema20_v = float(last["ema20"])
    low5 = float(last["low5"])
    high20 = float(last["high20"])
    high55 = float(last["high55"])

    entry_trigger = max(high20 + 0.10 * atr14, high55 * 1.001, close * 1.003)
    stop_anchor = min(low5, ema20_v)
    stop_loss = stop_anchor - 0.8 * atr14
    invalidation = stop_anchor

    if not np.isfinite(stop_loss) or stop_loss <= 0:
        stop_loss = close * 0.95
    if entry_trigger <= stop_loss:
        entry_trigger = max(close, stop_loss * 1.01)

    risk_per_share = max(entry_trigger - stop_loss, 1e-9)
    risk_budget = account_size * (risk_pct / 100.0)
    raw_shares = risk_budget / risk_per_share
    shares = int(raw_shares)
    if lot_rounding:
        shares = round_down_to_lot(shares, 100)
        if shares < 100:
            shares = 0

    lots = shares // 100
    position_value = shares * entry_trigger
    total_risk = shares * risk_per_share
    target1 = entry_trigger + 1.5 * risk_per_share
    target2 = entry_trigger + 3.0 * risk_per_share

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
        "rr_t1": 1.5,
        "rr_t2": 3.0,
        "close": float(close),
        "setup_note": "Minervini-style breakout: wait for close above trigger with volume confirmation.",
    }


# =========================================================
# Scan / summaries
# =========================================================

def summarize_universe(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {"breadth_above_sma50": np.nan, "breadth_above_sma200": np.nan}
    above_50 = []
    above_200 = []
    for _, row in df.iterrows():
        hist = row.get("_df")
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            last = hist.iloc[-1]
            above_50.append(bool(last.get("close", np.nan) > last.get("ema50", np.nan)))
            above_200.append(bool(last.get("close", np.nan) > last.get("ema200", np.nan)))
    return {
        "breadth_above_sma50": float(np.mean(above_50) * 100.0) if above_50 else np.nan,
        "breadth_above_sma200": float(np.mean(above_200) * 100.0) if above_200 else np.nan,
    }


def reset_scanner_state():
    for key in ["scan_context", "selected_ticker", "entry_ticker"]:
        st.session_state.pop(key, None)


def ensure_selectbox_value(key: str, options: List[str], fallback_index: int = 0) -> str:
    if not options:
        return ""
    fallback_index = max(0, min(fallback_index, len(options) - 1))
    fallback_value = options[fallback_index]
    if key not in st.session_state or st.session_state.get(key) not in options:
        st.session_state[key] = fallback_value
    return st.session_state[key]


def render_dashboard(context: Dict):
    df = context["df"]
    filtered = context["filtered"]
    benchmark_ticker = context["benchmark_ticker"]
    benchmark_score = context["benchmark_score"]
    benchmark_label = context["benchmark_label"]
    benchmark_detail = context["benchmark_detail"]
    breadth_above_sma50 = context["breadth_above_sma50"]
    breadth_above_sma200 = context["breadth_above_sma200"]

    tabs = st.tabs(["Market", "Scanner", "Entry Plan", "Export"])

    with tabs[0]:
        st.subheader("Market filter")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Benchmark", benchmark_ticker)
        c2.metric("Regime", benchmark_label)
        c3.metric("Score", f"{benchmark_score:.1f}" if np.isfinite(benchmark_score) else "N/A")
        c4.metric("Action", "Trade allowed" if benchmark_label == "RISK ON" else "No trade")

        st.caption("Scanner aktif saat market risk-on. Buy list diprioritaskan hanya untuk setup Minervini + CAN SLIM yang kuat.")
        if benchmark_detail:
            st.dataframe(pd.DataFrame([benchmark_detail]), use_container_width=True, hide_index=True)

        b1, b2 = st.columns(2)
        b1.metric("Breadth above EMA50", f"{breadth_above_sma50:.1f}%" if np.isfinite(breadth_above_sma50) else "N/A")
        b2.metric("Breadth above EMA200", f"{breadth_above_sma200:.1f}%" if np.isfinite(breadth_above_sma200) else "N/A")

    with tabs[1]:
        st.subheader("Scanner result")
        a, b, c, d = st.columns(4)
        a.metric("Universe", len(df))
        b.metric("Filtered", len(filtered))
        c.metric("BUY", int((df["decision"] == "BUY").sum()))
        d.metric("WATCH", int((df["decision"] == "WATCH").sum()))

        show_cols = [
            "ticker",
            "grade",
            "decision",
            "final_score",
            "tech_score",
            "fundamental_score",
            "close",
            "rsi14",
            "ret20",
            "ret60",
            "rs20",
            "rs60",
            "vol_rvol20",
            "atr_pct",
            "avg_dollar_vol_20",
            "minervini_ok",
            "can_slim_ok",
            "liquidity_ok",
            "price_structure_ok",
            "relative_strength_ok",
            "volume_ok",
            "breakout_ok",
        ]
        show_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(filtered[show_cols].head(context["top_n"]).reset_index(drop=True), use_container_width=True, hide_index=True)

        options = df["ticker"].tolist()
        selected = ensure_selectbox_value("selected_ticker", options, 0)
        picked = df[df["ticker"] == selected].iloc[0]
        st.write(f"**{picked['ticker']}** — {picked['decision']} | {picked['grade']} | score {picked['final_score']}")

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Market", "OK" if picked["market_ok"] else "Fail")
        m2.metric("Minervini", "OK" if picked["minervini_ok"] else "Fail")
        m3.metric("CAN SLIM", "OK" if picked["can_slim_ok"] else "Partial")
        m4.metric("Liquidity", "OK" if picked["liquidity_ok"] else "Fail")
        m5.metric("RS", "OK" if picked["relative_strength_ok"] else "Fail")
        m6.metric("Breakout", "OK" if picked["breakout_ok"] else "Fail")

        hist = picked.get("_df")
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            st.line_chart(hist[["close", "ema20", "ema50", "ema150", "ema200"]].dropna().tail(180))
            st.bar_chart(hist[["volume"]].tail(90))
            with st.expander("Latest snapshot"):
                last = hist.iloc[-1]
                snapshot = pd.DataFrame(
                    {
                        "metric": [
                            "close",
                            "ema20",
                            "ema50",
                            "ema150",
                            "ema200",
                            "sma200",
                            "high252",
                            "rsi14",
                            "atr14",
                            "atr_pct",
                            "ret20",
                            "ret60",
                            "ret120",
                            "vol_rvol20",
                            "revenue_growth",
                            "earnings_growth",
                            "roe",
                            "institutional_ownership",
                        ],
                        "value": [
                            last["close"],
                            last["ema20"],
                            last["ema50"],
                            last["ema150"],
                            last["ema200"],
                            last["sma200"],
                            last["high252"],
                            last["rsi14"],
                            last["atr14"],
                            last["atr_pct"],
                            last["ret20"],
                            last["ret60"],
                            last["ret120"],
                            last["vol_rvol20"],
                            picked.get("revenue_growth", np.nan),
                            picked.get("earnings_growth", np.nan),
                            picked.get("roe", np.nan),
                            picked.get("institutional_ownership", np.nan),
                        ],
                    }
                )
                st.dataframe(snapshot, use_container_width=True, hide_index=True)

    with tabs[2]:
        st.subheader("Entry plan")
        options = df["ticker"].tolist()
        selected = ensure_selectbox_value("entry_ticker", options, 0)
        picked = df[df["ticker"] == selected].iloc[0]
        hist = picked.get("_df")
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            plan = build_entry_plan(hist, account_size=context["account_size"], risk_pct=context["risk_pct"], lot_rounding=context["lot_rounding"])

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

            st.markdown(f"**Setup rule:** {plan['setup_note']}")

            plan_df = pd.DataFrame(
                [
                    {
                        "ticker": selected,
                        "decision": picked["decision"],
                        "grade": picked["grade"],
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
                    }
                ]
            )
            st.dataframe(plan_df, use_container_width=True, hide_index=True)
        else:
            st.info("Data belum cukup untuk entry plan.")

    with tabs[3]:
        st.subheader("Export")
        csv_all = df.drop(columns=["_df"], errors="ignore").to_csv(index=False).encode("utf-8")
        csv_filtered = filtered.drop(columns=["_df"], errors="ignore").to_csv(index=False).encode("utf-8")
        c1, c2 = st.columns(2)
        c1.download_button("Download all results CSV", csv_all, file_name="scanner_all_results.csv", mime="text/csv", use_container_width=True)
        c2.download_button("Download filtered results CSV", csv_filtered, file_name="scanner_filtered_results.csv", mime="text/csv", use_container_width=True)

        st.caption("Gunakan filtered results sebagai shortlist, bukan sebagai sinyal final tanpa konfirmasi chart.")


# =========================================================
# App
# =========================================================

if "scan_context" not in st.session_state:
    st.session_state["scan_context"] = None
if "selected_ticker" not in st.session_state:
    st.session_state["selected_ticker"] = ""
if "entry_ticker" not in st.session_state:
    st.session_state["entry_ticker"] = ""

st.title("IDX Minervini + CAN SLIM Scanner")
st.caption("Lebih ketat: market direction, Minervini trend structure, relative strength, volume, breakout, lalu lapisan fundamental CAN SLIM bila data tersedia.")

with st.sidebar:
    st.markdown("### Status")
    if st.session_state.get("scan_context"):
        ctx = st.session_state["scan_context"]
        st.success(f"Scan tersimpan: {len(ctx['df'])} ticker | {len(ctx['filtered'])} lolos filter")
        st.caption(f"Benchmark: {ctx['benchmark_label']} | Score {ctx['benchmark_score']:.1f}" if np.isfinite(ctx["benchmark_score"]) else f"Benchmark: {ctx['benchmark_label']}")
    else:
        st.info("Belum ada hasil scan tersimpan.")
    st.button("Reset hasil", width="stretch", on_click=reset_scanner_state)

    with st.expander("Universe", expanded=True):
        source_mode = st.radio("Sumber ticker", ["Auto IDX 800", "Paste ticker"], index=0)
        auto_suffix = st.checkbox("Auto tambah .JK", value=True)
        universe_limit = st.selectbox("Universe Likuid IDX", [100, 300, 500, 800], index=1)
        manual_text = st.text_area(
            "Ticker fallback (pisahkan koma / baris baru)",
            value="BBRI, BMRI, BBCA, ASII, ADRO",
            height=130,
            disabled=(source_mode != "Paste ticker"),
        )
        st.caption("Mode default akan menarik daftar saham IDX otomatis, lalu universe dipilih berdasarkan likuiditas dan dipotong sesuai Top N.")

    with st.expander("Scanner settings", expanded=True):
        benchmark_ticker = st.text_input("Benchmark IHSG", value="^JKSE")
        period = st.selectbox("History period", ["1y", "2y", "3y", "5y"], index=1)
        min_price = st.number_input("Min harga", min_value=0.0, value=500.0, step=50.0)
        min_avg_dollar_vol = st.number_input("Min avg dollar volume 20D", min_value=0.0, value=1_000_000_000.0, step=100_000_000.0, format="%.0f")
        max_workers = st.slider("Parallel workers", 1, 4, 2)
        top_n = st.slider("Top N hasil", 5, 100, 25)

    with st.expander("Entry plan", expanded=False):
        account_size = st.number_input("Account size", min_value=0.0, value=100_000_000.0, step=5_000_000.0, format="%.0f")
        risk_pct = st.slider("Risk per trade (%)", 0.1, 5.0, 1.0, 0.1)
        lot_rounding = st.checkbox("Round to lots (100 shares)", value=True)

    scan_btn = st.button("Scan sekarang", type="primary", width="stretch")

if scan_btn:
    if source_mode == "Auto IDX 800":
        tickers = load_idx_universe(limit=universe_limit)
        if not tickers:
            st.warning("Gagal memuat universe IDX otomatis. Pakai fallback ticker manual di sidebar.")
            tickers = parse_manual_tickers(manual_text, auto_suffix=auto_suffix)
    else:
        tickers = parse_manual_tickers(manual_text, auto_suffix=auto_suffix)

    if not tickers:
        st.warning("Daftar ticker kosong.")
        st.stop()

    benchmark_df = build_benchmark(benchmark_ticker, period)
    if not benchmark_df.empty:
        benchmark_score, benchmark_label, benchmark_detail = score_benchmark(benchmark_df)
    else:
        benchmark_score, benchmark_label, benchmark_detail = np.nan, "Unknown", {}

    st.subheader("Scanning...")
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
                benchmark_df=benchmark_df,
            )
        except Exception as e:
            return {
                "ticker": tkr,
                "final_score": np.nan,
                "grade": "ERROR",
                "decision": "ERROR",
                "benchmark_risk_on": False,
                "liquidity_ok": False,
                "trend_ok": False,
                "relative_strength_ok": False,
                "volume_ok": False,
                "breakout_ok": False,
                "close": np.nan,
                "rsi14": np.nan,
                "ret20": np.nan,
                "ret60": np.nan,
                "rs20": np.nan,
                "rs60": np.nan,
                "vol_rvol20": np.nan,
                "atr_pct": np.nan,
                "avg_dollar_vol_20": np.nan,
                "avg_vol_20": np.nan,
                "high20": np.nan,
                "high55": np.nan,
                "ema20": np.nan,
                "ema50": np.nan,
                "ema150": np.nan,
                "ema200": np.nan,
                "sma200": np.nan,
                "_df": pd.DataFrame(),
                "_checks": {"error": str(e)},
            }

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker, t): t for t in tickers}
        done = 0
        for fut in as_completed(futures):
            rows.append(fut.result())
            done += 1
            status.write(f"Scanning {done}/{len(tickers)} ...")
            progress.progress(done / len(tickers))

    progress.empty()
    status.empty()

    if not rows:
        st.error("Tidak ada hasil valid dari scanner.")
        st.stop()

    df = pd.DataFrame(rows)
    df = df.sort_values(["final_score", "tech_score", "vol_rvol20", "avg_dollar_vol_20"], ascending=False, na_position="last")

    filtered = df[(df["decision"].isin(["BUY", "WATCH"])) | (df["tech_score"].fillna(0) >= 6)].copy()

    breadth = summarize_universe(df)
    st.session_state["scan_context"] = {
        "df": df,
        "filtered": filtered,
        "benchmark_ticker": benchmark_ticker,
        "benchmark_score": benchmark_score,
        "benchmark_label": benchmark_label,
        "benchmark_detail": benchmark_detail,
        "breadth_above_sma50": breadth["breadth_above_sma50"],
        "breadth_above_sma200": breadth["breadth_above_sma200"],
        "top_n": top_n,
        "account_size": account_size,
        "risk_pct": risk_pct,
        "lot_rounding": lot_rounding,
    }

    tickers_list = df["ticker"].tolist()
    if tickers_list:
        if st.session_state.get("selected_ticker") not in tickers_list:
            st.session_state["selected_ticker"] = tickers_list[0]
        if st.session_state.get("entry_ticker") not in tickers_list:
            st.session_state["entry_ticker"] = tickers_list[0]

context = st.session_state.get("scan_context")
if context:
    render_dashboard(context)
else:
    st.info("Pilih universe ticker di sidebar, lalu klik **Scan sekarang**.")
    st.markdown(
        """
        Scanner ini sekarang lebih ketat:
        - Market harus **risk-on**
        - Struktur harga harus **ala Minervini**
        - Relative strength harus **positif**
        - Volume harus **menguat**
        - Fundamental CAN SLIM dipakai bila data tersedia
        - Entry tetap hanya pada **breakout**
        """
    )
