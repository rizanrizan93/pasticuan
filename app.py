import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


st.set_page_config(page_title="Predictive Stock Scanner", layout="wide")
st.title("Predictive Stock Scanner")
st.caption("Scanner probabilistik untuk ranking entry, stoploss, takeprofit, dan horizon per emiten.")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Universe")
source = st.sidebar.radio(
    "Sumber ticker",
    ["Paste tickers", "Upload CSV"],
    index=0,
)

paste_text = "BMRI\nBBCA\nASII\nTLKM\nMBMA"
uploaded = None
if source == "Paste tickers":
    paste_text = st.sidebar.text_area("Ticker (satu baris / koma)", value=paste_text, height=140)
else:
    uploaded = st.sidebar.file_uploader("Upload CSV universe", type=["csv"])

st.sidebar.header("Scan Settings")
period_months = st.sidebar.slider("Data historis (bulan)", 6, 60, 24)
max_tickers = st.sidebar.slider("Maks ticker", 20, 200, 100, step=10)
min_price = st.sidebar.number_input("Min harga (Rp)", value=100.0, step=10.0)
max_price = st.sidebar.number_input("Max harga (Rp)", value=50000.0, step=500.0)
min_avg_vol = st.sidebar.number_input("Min avg volume 20D", value=150000, step=50000)
lookbacks = st.sidebar.multiselect("Horizon prediksi (hari bursa)", [5, 10, 15, 20], default=[5, 10, 20])
mode = st.sidebar.selectbox("Mode risiko", ["Conservative", "Balanced", "Aggressive"], index=1)
workers = st.sidebar.slider("Parallel workers", 2, 12, 6)
run = st.sidebar.button("Run predictive scan", type="primary")

MODE_CFG = {
    "Conservative": {"entry_pad": 0.12, "stop_pad": 0.95, "tp_mult": 1.15, "min_rr": 2.0},
    "Balanced": {"entry_pad": 0.18, "stop_pad": 0.90, "tp_mult": 1.05, "min_rr": 1.6},
    "Aggressive": {"entry_pad": 0.25, "stop_pad": 0.82, "tp_mult": 0.95, "min_rr": 1.3},
}
CFG = MODE_CFG[mode]


# -----------------------------
# Helpers
# -----------------------------

def normalize_ticker(x: str) -> str:
    s = re.sub(r"\s+", "", str(x or "").upper())
    s = s.replace("/", "-")
    if not s:
        return ""
    if s.endswith(".JK") or s.startswith("^"):
        return s
    # IDX-friendly default
    if s.isalpha() and len(s) <= 5:
        return f"{s}.JK"
    return s


def parse_universe(text: str | None, csv_df: pd.DataFrame | None) -> List[str]:
    tickers: List[str] = []
    if csv_df is not None and not csv_df.empty:
        cols = [str(c).strip() for c in csv_df.columns]
        candidate = None
        for c in ["Ticker", "ticker", "Symbol", "symbol", "Kode", "code", "Stock"]:
            if c in cols:
                candidate = c
                break
        if candidate is None:
            candidate = cols[0]
        for raw in csv_df[candidate].dropna().astype(str).tolist():
            sym = normalize_ticker(raw)
            if sym and sym not in tickers:
                tickers.append(sym)
    if text:
        for raw in re.split(r"[\n,;\t ]+", text):
            if not raw.strip():
                continue
            sym = normalize_ticker(raw)
            if sym and sym not in tickers:
                tickers.append(sym)
    return tickers


def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [" ".join([str(p) for p in c if str(p) != ""]).strip() for c in out.columns]
    out.columns = [str(c).strip() for c in out.columns]
    if "Date" in out.columns:
        out = out.set_index("Date")
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].copy()
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not set(needed).issubset(set(out.columns)):
        return pd.DataFrame()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")].copy()
    for c in needed + (["Adj Close"] if "Adj Close" in out.columns else []):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=needed).copy()
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def download_one(symbol: str, period: str) -> Tuple[str, pd.DataFrame, str]:
    candidates = [symbol]
    if symbol.endswith(".JK"):
        candidates.append(symbol[:-3])
    elif not symbol.startswith("^"):
        candidates.append(f"{symbol}.JK")
    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]
    last_err = ""
    for cand in candidates:
        try:
            df = yf.download(cand, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
            df = _standardize_ohlcv(df)
            if not df.empty:
                df = df.copy()
                df["Symbol"] = symbol
                return symbol, df, ""
            last_err = "empty"
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
    return symbol, pd.DataFrame(), last_err or "failed"


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    rs = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / down.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_s = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr_s
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr_s
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def recent_swing_low(df: pd.DataFrame, n: int = 20) -> float:
    return float(df["Low"].tail(n).min()) if len(df) else np.nan


def recent_swing_high(df: pd.DataFrame, n: int = 20) -> float:
    return float(df["High"].tail(n).max()) if len(df) else np.nan


def fib_retrace_levels(low: float, high: float) -> Dict[str, float]:
    span = max(high - low, 1e-9)
    return {
        "0.236": high - 0.236 * span,
        "0.382": high - 0.382 * span,
        "0.500": high - 0.500 * span,
        "0.618": high - 0.618 * span,
    }


def finite_min(values, default=np.nan):
    vals = [float(v) for v in values if np.isfinite(v)]
    return float(min(vals)) if vals else default


def finite_max(values, default=np.nan):
    vals = [float(v) for v in values if np.isfinite(v)]
    return float(max(vals)) if vals else default


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret1"] = out["Close"].pct_change()
    out["ret3"] = out["Close"].pct_change(3)
    out["ret5"] = out["Close"].pct_change(5)
    out["vol20"] = out["Close"].pct_change().rolling(20, min_periods=20).std()
    out["ema20"] = ema(out["Close"], 20)
    out["ema50"] = ema(out["Close"], 50)
    out["ema200"] = ema(out["Close"], 200)
    out["rsi14"] = rsi(out["Close"], 14)
    out["atr14"] = atr(out, 14)
    out["atr_pct"] = out["atr14"] / out["Close"]
    out["adx14"] = adx(out, 14)
    out["vol_sma20"] = sma(out["Volume"], 20)
    out["vol_ratio20"] = out["Volume"] / out["vol_sma20"]
    out["dist_ema20"] = (out["Close"] - out["ema20"]) / out["Close"]
    out["dist_ema50"] = (out["Close"] - out["ema50"]) / out["Close"]
    out["close_pos_20"] = (out["Close"] - out["Low"].rolling(20, min_periods=20).min()) / (
        out["High"].rolling(20, min_periods=20).max() - out["Low"].rolling(20, min_periods=20).min()
    )
    out["range20"] = (out["High"].rolling(20, min_periods=20).max() - out["Low"].rolling(20, min_periods=20).min()) / out["Close"]
    out["body_ratio"] = (out["Close"] - out["Open"]).abs() / (out["High"] - out["Low"]).replace(0, np.nan)
    out["upper_wick"] = (out["High"] - out[["Open", "Close"]].max(axis=1)) / out["Close"]
    out["lower_wick"] = (out[["Open", "Close"]].min(axis=1) - out["Low"]) / out["Close"]
    out["hh20"] = out["High"].rolling(20, min_periods=20).max()
    out["ll20"] = out["Low"].rolling(20, min_periods=20).min()
    out["hh10"] = out["High"].rolling(10, min_periods=10).max()
    out["ll10"] = out["Low"].rolling(10, min_periods=10).min()
    out["trend_slope10"] = out["Close"].rolling(10, min_periods=10).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=False)
    out["trend_slope20"] = out["Close"].rolling(20, min_periods=20).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=False)
    return out


def build_dataset(all_frames: Dict[str, pd.DataFrame], horizons: List[int]) -> pd.DataFrame:
    rows = []
    for sym, raw in all_frames.items():
        if raw is None or raw.empty:
            continue
        df = compute_features(raw)
        for h in horizons:
            df[f"fwd_high_{h}"] = df["High"].rolling(h, min_periods=1).max().shift(-h) / df["Close"] - 1.0
            df[f"fwd_low_{h}"] = df["Low"].rolling(h, min_periods=1).min().shift(-h) / df["Close"] - 1.0
            df[f"fwd_close_{h}"] = df["Close"].shift(-h) / df["Close"] - 1.0
        df["Symbol"] = sym
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, axis=0)
    return out


def feature_columns() -> List[str]:
    return [
        "ret1", "ret3", "ret5", "vol20", "ema20", "ema50", "ema200",
        "rsi14", "atr14", "atr_pct", "adx14", "vol_sma20", "vol_ratio20",
        "dist_ema20", "dist_ema50", "close_pos_20", "range20", "body_ratio",
        "upper_wick", "lower_wick", "hh20", "ll20", "hh10", "ll10",
        "trend_slope10", "trend_slope20",
    ]


@dataclass
class HorizonModels:
    up_q50: Pipeline
    up_q90: Pipeline
    low_q10: Pipeline
    low_q50: Pipeline
    score: float


def make_model(q: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        (
            "gbr",
            GradientBoostingRegressor(
                loss="quantile",
                alpha=q,
                n_estimators=90,
                learning_rate=0.05,
                max_depth=3,
                min_samples_leaf=18,
                subsample=0.8,
                random_state=42,
            ),
        ),
    ])


def train_models(train_df: pd.DataFrame, horizons: List[int]) -> Dict[int, HorizonModels]:
    feats = feature_columns()
    out: Dict[int, HorizonModels] = {}
    base = train_df.dropna(subset=feats).copy()
    if base.empty:
        return out
    for h in horizons:
        d = base.dropna(subset=[f"fwd_high_{h}", f"fwd_low_{h}"]).copy()
        if len(d) < 300:
            continue
        X = d[feats]
        up = d[f"fwd_high_{h}"]
        low = d[f"fwd_low_{h}"]
        m_up50 = make_model(0.50).fit(X, up)
        m_up90 = make_model(0.90).fit(X, up)
        m_low10 = make_model(0.10).fit(X, low)
        m_low50 = make_model(0.50).fit(X, low)
        # crude model quality proxy: separation between upside median and downside median
        score = float(np.nanmean(up) - np.nanmean(low))
        out[h] = HorizonModels(m_up50, m_up90, m_low10, m_low50, score)
    return out


def classify_setup(df: pd.DataFrame) -> Tuple[str, str]:
    if df.empty:
        return "UNKNOWN", "No data"
    last = df.iloc[-1]
    close = float(last["Close"])
    ema20_v = float(last.get("ema20", np.nan))
    ema50_v = float(last.get("ema50", np.nan))
    ema200_v = float(last.get("ema200", np.nan))
    rsi_v = float(last.get("rsi14", np.nan))
    atr_v = float(last.get("atr14", np.nan))
    ll20 = float(last.get("ll20", np.nan))
    hh20 = float(last.get("hh20", np.nan))
    trend_up = close > ema20_v > ema50_v if np.isfinite([close, ema20_v, ema50_v]).all() else False
    trend_down = close < ema20_v < ema50_v if np.isfinite([close, ema20_v, ema50_v]).all() else False
    near_support = np.isfinite(ll20) and close <= ll20 * 1.08
    near_resistance = np.isfinite(hh20) and close >= hh20 * 0.92
    squeeze = np.isfinite(atr_v) and np.isfinite(last.get("range20", np.nan)) and last["atr_pct"] < 0.06 and last["range20"] < 0.18
    oversold = np.isfinite(rsi_v) and rsi_v < 40
    reclaimed = len(df) > 3 and df["Close"].iloc[-1] > df["High"].iloc[-3]
    if trend_up and near_support:
        return "PULLBACK", "Trend up, retrace to support"
    if reclaimed and oversold and not trend_down:
        return "UNICORN", "Sweep/reclaim / imbalance fill"
    if trend_up and squeeze and near_resistance:
        return "SNIPER", "Tight base near breakout"
    if trend_down and oversold:
        return "REVERSAL", "Downtrend exhaustion"
    if trend_up:
        return "PULLBACK", "Primary trend continuation"
    if reclaimed:
        return "UNICORN", "Structure reclaim"
    return "REVERSAL", "Mean reversion / reversal watch"


def build_plan(df: pd.DataFrame, pred: Dict[int, Dict[str, float]], setup: str) -> Dict[str, float | str]:
    last = df.iloc[-1]
    close = float(last["Close"])
    atr_v = float(last.get("atr14", np.nan))
    ema20_v = float(last.get("ema20", np.nan))
    ema50_v = float(last.get("ema50", np.nan))
    ll20 = float(last.get("ll20", np.nan))
    hh20 = float(last.get("hh20", np.nan))
    swing_low = recent_swing_low(df, 20)
    swing_high = recent_swing_high(df, 20)
    low_10 = recent_swing_low(df, 10)
    high_10 = recent_swing_high(df, 10)

    # Base structural anchors
    support_anchor = finite_min([ll20, ema20_v, ema50_v, swing_low, low_10], default=close * 0.97)
    resistance_anchor = finite_max([hh20, high_10], default=close * 1.03)

    # Setup-specific entry/stop behavior
    if setup == "PULLBACK":
        entry_low = np.nanmax([x for x in [support_anchor, close - 0.35 * atr_v if np.isfinite(atr_v) else np.nan] if np.isfinite(x)])
        entry_high = np.nanmin([x for x in [close - 0.05 * atr_v if np.isfinite(atr_v) else close, ema20_v * 1.01 if np.isfinite(ema20_v) else np.nan] if np.isfinite(x)])
    elif setup == "UNICORN":
        entry_low = np.nanmax([x for x in [support_anchor, close - 0.28 * atr_v if np.isfinite(atr_v) else np.nan] if np.isfinite(x)])
        entry_high = np.nanmin([x for x in [close + 0.03 * atr_v if np.isfinite(atr_v) else close, ema20_v * 1.005 if np.isfinite(ema20_v) else np.nan] if np.isfinite(x)])
    elif setup == "SNIPER":
        entry_low = np.nanmax([x for x in [support_anchor, close - 0.20 * atr_v if np.isfinite(atr_v) else np.nan] if np.isfinite(x)])
        entry_high = np.nanmin([x for x in [close - 0.02 * atr_v if np.isfinite(atr_v) else close, resistance_anchor * 1.002 if np.isfinite(resistance_anchor) else np.nan] if np.isfinite(x)])
    else:  # REVERSAL
        entry_low = np.nanmax([x for x in [low_10, close - 0.25 * atr_v if np.isfinite(atr_v) else np.nan] if np.isfinite(x)])
        entry_high = np.nanmin([x for x in [close + 0.10 * atr_v if np.isfinite(atr_v) else close, support_anchor * 1.03 if np.isfinite(support_anchor) else np.nan] if np.isfinite(x)])

    if not np.isfinite(entry_low):
        entry_low = close * 0.995
    if not np.isfinite(entry_high) or entry_high <= entry_low:
        entry_high = max(entry_low * 1.01, close)

    entry_mid = float((entry_low + entry_high) / 2.0)
    # Model-derived future range
    best_h = sorted(pred.keys(), key=lambda h: pred[h]["ev"], reverse=True)[0]
    up50 = pred[best_h]["up50"]
    up90 = pred[best_h]["up90"]
    low10 = pred[best_h]["low10"]
    low50 = pred[best_h]["low50"]

    tp1 = max(entry_mid * (1 + up50), resistance_anchor if np.isfinite(resistance_anchor) else entry_mid * 1.02)
    tp2 = max(entry_mid * (1 + up90), tp1 * CFG["tp_mult"])

    structural_stop = finite_min([support_anchor, low10 * 0.995 if np.isfinite(low10) else np.nan], default=close * 0.97)
    stop = min(entry_low * (1 - 0.5 * CFG["stop_pad"] * (atr_v / close if np.isfinite(atr_v) and close else 0.03)), structural_stop)
    stop = min(stop, entry_low * 0.998)
    if not np.isfinite(stop) or stop >= entry_low:
        stop = entry_low * 0.97

    risk = max(entry_mid - stop, 1e-9)
    rr1 = (tp1 - entry_mid) / risk
    rr2 = (tp2 - entry_mid) / risk
    return {
        "best_horizon": best_h,
        "entry_low": float(entry_low),
        "entry_high": float(entry_high),
        "entry_mid": float(entry_mid),
        "stop": float(stop),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "risk": float(risk),
        "rr1": float(rr1),
        "rr2": float(rr2),
        "up50": float(up50),
        "up90": float(up90),
        "low10": float(low10),
        "low50": float(low50),
    }


def score_plan(df: pd.DataFrame, plan: Dict[str, float | str], setup: str) -> Dict[str, float]:
    last = df.iloc[-1]
    close = float(last["Close"])
    atr_pct = float(last.get("atr_pct", np.nan))
    rsi_v = float(last.get("rsi14", np.nan))
    adx_v = float(last.get("adx14", np.nan))
    vol_ratio = float(last.get("vol_ratio20", np.nan))
    trend_slope = float(last.get("trend_slope10", np.nan))

    rr1 = float(plan["rr1"])
    rr2 = float(plan["rr2"])
    up50 = float(plan["up50"])
    up90 = float(plan["up90"])
    low10 = float(plan["low10"])
    risk = float(plan["risk"])

    trend_score = 0.0
    if np.isfinite(trend_slope):
        trend_score += np.clip((trend_slope / close) * 1000, -2, 2)
    if np.isfinite(adx_v):
        trend_score += np.clip((adx_v - 15) / 10, -1, 2)
    if np.isfinite(rsi_v):
        trend_score += np.clip((rsi_v - 45) / 12, -1, 1)

    structure_score = 2.0
    if setup == "PULLBACK":
        structure_score += 0.8
    elif setup == "UNICORN":
        structure_score += 1.0
    elif setup == "SNIPER":
        structure_score += 1.1
    else:
        structure_score += 0.5

    vol_score = 0.0 if not np.isfinite(vol_ratio) else np.clip((vol_ratio - 1.0), -0.5, 2.0)
    regime_penalty = 0.0 if not np.isfinite(atr_pct) else np.clip((atr_pct - 0.04) * 10, 0.0, 1.8)
    edge = max(up50 * 100, up90 * 80) - abs(low10 * 100) * 0.7
    score = 20 + 12 * trend_score + 9 * structure_score + 10 * vol_score + 8 * np.clip(rr1, 0, 4) + 6 * np.clip(rr2, 0, 6) + edge - 8 * regime_penalty
    score = float(np.clip(score, 0, 100))

    prob_entry = float(np.clip(0.55 + (-low10 * 6) + (0.04 - (atr_pct if np.isfinite(atr_pct) else 0.04)) * 8, 0.05, 0.95))
    prob_tp1 = float(np.clip(0.30 + up50 * 6 + np.clip(rr1 / 4, 0, 0.5), 0.05, 0.98))
    prob_tp2 = float(np.clip(0.15 + up90 * 5 + np.clip(rr2 / 6, 0, 0.4), 0.01, 0.95))
    ev = float((prob_tp1 * (plan["tp1"] - plan["entry_mid"])) - ((1 - prob_tp1) * (plan["entry_mid"] - plan["stop"])))

    return {
        "score": score,
        "prob_entry": prob_entry,
        "prob_tp1": prob_tp1,
        "prob_tp2": prob_tp2,
        "ev": ev,
    }


def predict_for_ticker(df: pd.DataFrame, models: Dict[int, HorizonModels], horizons: List[int]) -> Dict:
    feat = compute_features(df)
    last = feat.iloc[-1]
    X = feat[feature_columns()].iloc[[-1]]
    pred: Dict[int, Dict[str, float]] = {}
    for h in horizons:
        if h not in models:
            continue
        m = models[h]
        up50 = float(m.up_q50.predict(X)[0])
        up90 = float(m.up_q90.predict(X)[0])
        low10 = float(m.low_q10.predict(X)[0])
        low50 = float(m.low_q50.predict(X)[0])
        # Simple EV proxy for horizon selection
        risk = max(abs(low10), 1e-6)
        ev = up50 - 0.8 * risk
        pred[h] = {"up50": up50, "up90": up90, "low10": low10, "low50": low50, "ev": ev}
    if not pred:
        return {}
    setup, setup_note = classify_setup(feat)
    plan = build_plan(feat, pred, setup)
    metrics = score_plan(feat, plan, setup)
    close = float(last["Close"])
    atr_v = float(last.get("atr14", np.nan))
    return {
        "symbol": str(last.get("Symbol", "")),
        "date": feat.index[-1],
        "close": close,
        "atr": atr_v,
        "setup": setup,
        "setup_note": setup_note,
        **plan,
        **metrics,
        "ema20": float(last.get("ema20", np.nan)),
        "ema50": float(last.get("ema50", np.nan)),
        "ema200": float(last.get("ema200", np.nan)),
        "rsi14": float(last.get("rsi14", np.nan)),
        "adx14": float(last.get("adx14", np.nan)),
        "vol_ratio20": float(last.get("vol_ratio20", np.nan)),
        "atr_pct": float(last.get("atr_pct", np.nan)),
        "trend_slope10": float(last.get("trend_slope10", np.nan)),
    }


def chart_ticker(df: pd.DataFrame, row: Dict):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC"))
    for name, value, color in [
        ("Entry low", row["entry_low"], "#2ecc71"),
        ("Entry high", row["entry_high"], "#27ae60"),
        ("Stop", row["stop"], "#e74c3c"),
        ("TP1", row["tp1"], "#3498db"),
        ("TP2", row["tp2"], "#8e44ad"),
    ]:
        fig.add_hline(y=float(value), line_width=1, line_dash="dot", line_color=color, annotation_text=name)
    fig.update_layout(height=540, margin=dict(l=10, r=10, t=30, b=10), xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# Main workflow
# -----------------------------

csv_df = None
if uploaded is not None:
    try:
        csv_df = pd.read_csv(uploaded)
    except Exception:
        csv_df = pd.DataFrame()

tickers = parse_universe(paste_text if source == "Paste tickers" else None, csv_df)
tickers = tickers[:max_tickers]

if run:
    if not tickers:
        st.error("Tidak ada ticker valid.")
        st.stop()

    period = f"{period_months}mo"
    st.write(f"Scanning {len(tickers)} ticker...")

    # Download data
    frames: Dict[str, pd.DataFrame] = {}
    errs: Dict[str, str] = {}
    prog = st.progress(0)
    total = len(tickers)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_one, sym, period): sym for sym in tickers}
        done = 0
        for fut in as_completed(futures):
            sym, df, err = fut.result()
            if not df.empty:
                frames[sym] = df.tail(700).copy()
            else:
                errs[sym] = err
            done += 1
            prog.progress(done / total)
    prog.empty()

    if not frames:
        st.error("Tidak ada data yang berhasil diunduh.")
        st.stop()

    # Apply price/volume filter on latest snapshot
    filtered_frames: Dict[str, pd.DataFrame] = {}
    for sym, df in frames.items():
        feat = compute_features(df)
        last = feat.iloc[-1]
        close = float(last["Close"])
        avg_vol = float(feat["Volume"].tail(20).mean())
        if close < min_price or close > max_price:
            continue
        if np.isfinite(avg_vol) and avg_vol < min_avg_vol:
            continue
        filtered_frames[sym] = df

    if not filtered_frames:
        st.warning("Semua ticker tersaring oleh harga/volume filter.")
        st.stop()

    # Build pooled dataset and train quantile models
    pooled = build_dataset(filtered_frames, lookbacks)
    if pooled.empty:
        st.error("Dataset training kosong.")
        st.stop()

    models = train_models(pooled, lookbacks)
    if not models:
        st.error("Tidak cukup data untuk melatih model.")
        st.stop()

    # Predict current rows
    rows = []
    for sym, df in filtered_frames.items():
        res = predict_for_ticker(df, models, lookbacks)
        if not res:
            continue
        rows.append(res)

    if not rows:
        st.warning("Tidak ada hasil prediksi.")
        st.stop()

    out = pd.DataFrame(rows)
    out = out.sort_values(["score", "prob_tp1", "rr1"], ascending=False).reset_index(drop=True)
    top20 = out.head(20).copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scanned", len(filtered_frames))
    c2.metric("Valid setups", len(out))
    c3.metric("Top score", f"{top20.iloc[0]['score']:.1f}")
    c4.metric("Median RR1", f"{out['rr1'].median():.2f}")

    st.subheader("Top 20 predictive setups")
    show_cols = [
        "symbol", "setup", "score", "close", "entry_low", "entry_high", "stop", "tp1", "tp2",
        "best_horizon", "prob_entry", "prob_tp1", "prob_tp2", "rr1", "rr2", "setup_note",
    ]
    st.dataframe(top20[show_cols], use_container_width=True, hide_index=True)

    pick = st.selectbox("Lihat detail ticker", top20["symbol"].tolist())
    row = top20[top20["symbol"] == pick].iloc[0].to_dict()
    st.subheader(f"Detail {pick}")
    st.write(
        f"Setup: **{row['setup']}** | Horizon terbaik: **{int(row['best_horizon'])} hari** | "
        f"Entry: **{row['entry_low']:.2f} - {row['entry_high']:.2f}** | Stop: **{row['stop']:.2f}** | "
        f"TP1: **{row['tp1']:.2f}** | TP2: **{row['tp2']:.2f}**"
    )
    st.write(f"Skor: **{row['score']:.1f}** | Prob TP1: **{row['prob_tp1']:.2f}** | Prob TP2: **{row['prob_tp2']:.2f}**")
    chart_ticker(filtered_frames[pick].tail(180), row)

    # Execution note
    st.info(
        "Gunakan entry zone sebagai limit/alert area di Stockbit. "
        "TP1 cocok untuk partial take profit, TP2 untuk runner, dan stop untuk invalidation struktural."
    )

    # raw table and exports
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("Download full scan CSV", csv, file_name="predictive_scan.csv", mime="text/csv")

else:
    st.info("Isi universe ticker lalu klik 'Run predictive scan'.")
    st.write("Contoh output yang dicari: entry zone, stoploss, TP1/TP2, horizon terbaik, dan ranking probabilistik.")

