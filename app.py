import math
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


def parse_universe(source_mode: str, csv_file, manual_text: str, auto_suffix: bool) -> List[str]:
    if source_mode == "Upload CSV":
        if csv_file is None:
            return []
        return parse_uploaded_csv(csv_file, auto_suffix=auto_suffix)
    return parse_manual_tickers(manual_text, auto_suffix=auto_suffix)


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
    out["vol_sma20"] = sma(v, 20)
    out["vol_rvol20"] = v / out["vol_sma20"].replace(0, np.nan)
    out["high20"] = h.rolling(20).max()
    out["high55"] = h.rolling(55).max()
    out["low5"] = l.rolling(5).min()
    out["ema200_20ago"] = out["ema200"].shift(20)
    out["ema200_rising"] = out["ema200"] > out["ema200_20ago"]
    return out


# =========================================================
# Simple risk-on / trend / liquidity scanner
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
    }
    if last["close"] > last["sma200"]:
        score += 40
    if last["ema50"] > last["sma200"]:
        score += 25
    if last["close"] > last["ema50"]:
        score += 20
    if last["rsi14"] > 50:
        score += 15
    score = clip(score)
    label = "RISK ON" if score >= 70 else "NEUTRAL" if score >= 50 else "RISK OFF"
    return score, label, detail


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
        benchmark_ret20 = float(b_last["ret20"])
        benchmark_ret60 = float(b_last["ret60"])
        benchmark_risk_on = bool(benchmark_close > benchmark_sma200 and benchmark_ema50 > benchmark_sma200)
    else:
        benchmark_close = np.nan
        benchmark_sma200 = np.nan
        benchmark_ema50 = np.nan
        benchmark_ret20 = np.nan
        benchmark_ret60 = np.nan
        benchmark_risk_on = True

    # Core filters
    avg_dollar_vol_20 = float((df["close"] * df["volume"]).tail(20).mean())
    avg_vol_20 = float(df["volume"].tail(20).mean())
    price = float(last["close"])
    liquidity_ok = bool(
        np.isfinite(avg_dollar_vol_20)
        and avg_dollar_vol_20 >= min_avg_dollar_vol
        and avg_vol_20 >= 100_000
        and price >= min_price
    )

    trend_ok = bool(
        last["close"] > last["ema50"] > last["ema200"]
        and last["close"] > last["sma200"]
        and bool(last["ema200_rising"])
    )

    ret20 = float(last["ret20"])
    ret60 = float(last["ret60"])
    rs20 = float(ret20 - benchmark_ret20) if np.isfinite(benchmark_ret20) else np.nan
    rs60 = float(ret60 - benchmark_ret60) if np.isfinite(benchmark_ret60) else np.nan
    relative_strength_ok = bool(
        np.isfinite(rs20) and np.isfinite(rs60) and rs20 > 0 and rs60 > 0
    )

    vol_rvol20 = float(last["vol_rvol20"])
    volume_ok = bool(np.isfinite(vol_rvol20) and vol_rvol20 >= 1.5 and last["close"] > prev["close"])

    breakout_ok = bool(
        last["close"] >= last["high20"] * 0.99
        or last["close"] >= last["high55"] * 0.975
    )

    strong_setup = bool(
        breakout_ok
        and last["close"] > last["ema20"]
        and 50 <= last["rsi14"] <= 78
    )

    checks = {
        "risk_on": benchmark_risk_on,
        "liquidity_ok": liquidity_ok,
        "trend_ok": trend_ok,
        "relative_strength_ok": relative_strength_ok,
        "volume_ok": volume_ok,
        "breakout_ok": strong_setup,
    }
    score = int(sum(int(v) for v in checks.values()))

    if score == 6:
        decision = "BUY"
        grade = "ELITE"
    elif score == 5:
        decision = "WATCH"
        grade = "STRONG"
    elif score == 4:
        decision = "WATCH"
        grade = "WATCHLIST"
    else:
        decision = "IGNORE"
        grade = "IGNORE"

    return {
        "ticker": ticker,
        "final_score": score,
        "grade": grade,
        "decision": decision,
        "benchmark_risk_on": benchmark_risk_on,
        "liquidity_ok": liquidity_ok,
        "trend_ok": trend_ok,
        "relative_strength_ok": relative_strength_ok,
        "volume_ok": volume_ok,
        "breakout_ok": strong_setup,
        "close": price,
        "rsi14": float(last["rsi14"]),
        "ret20": ret20,
        "ret60": ret60,
        "rs20": rs20,
        "rs60": rs60,
        "vol_rvol20": vol_rvol20,
        "atr_pct": float(last["atr_pct"]),
        "avg_dollar_vol_20": avg_dollar_vol_20,
        "avg_vol_20": avg_vol_20,
        "high20": float(last["high20"]),
        "high55": float(last["high55"]),
        "ema20": float(last["ema20"]),
        "ema50": float(last["ema50"]),
        "ema150": float(last["ema150"]),
        "ema200": float(last["ema200"]),
        "sma200": float(last["sma200"]),
        "_df": df,
        "_checks": checks,
    }


def build_entry_plan(df: pd.DataFrame, account_size: float, risk_pct: float, lot_rounding: bool) -> Dict[str, float]:
    last = df.iloc[-1]
    close = float(last["close"])
    atr14 = float(last["atr14"])
    ema20_v = float(last["ema20"])
    low5 = float(last["low5"])
    high20 = float(last["high20"])

    entry_trigger = max(high20 + 0.10 * atr14, close * 1.003)
    stop_loss = min(low5, ema20_v) - 0.8 * atr14
    invalidation = min(low5, ema20_v)

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
        "setup_note": "Breakout only: wait for close above trigger with volume confirmation.",
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

        st.caption("Scanner hanya aktif saat market risk-on.")
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
            "close",
            "rsi14",
            "ret20",
            "ret60",
            "rs20",
            "rs60",
            "vol_rvol20",
            "atr_pct",
            "avg_dollar_vol_20",
            "liquidity_ok",
            "trend_ok",
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
        m1.metric("Risk-on", "Yes" if picked["benchmark_risk_on"] else "No")
        m2.metric("Trend", "OK" if picked["trend_ok"] else "Fail")
        m3.metric("Liquidity", "OK" if picked["liquidity_ok"] else "Fail")
        m4.metric("RS", "OK" if picked["relative_strength_ok"] else "Fail")
        m5.metric("Volume", "OK" if picked["volume_ok"] else "Fail")
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
                            "rsi14",
                            "atr14",
                            "atr_pct",
                            "ret20",
                            "ret60",
                            "vol_rvol20",
                            "high20",
                            "high55",
                        ],
                        "value": [
                            last["close"],
                            last["ema20"],
                            last["ema50"],
                            last["ema150"],
                            last["ema200"],
                            last["sma200"],
                            last["rsi14"],
                            last["atr14"],
                            last["atr_pct"],
                            last["ret20"],
                            last["ret60"],
                            last["vol_rvol20"],
                            last["high20"],
                            last["high55"],
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

st.title("IDX Simple Trend + Liquidity Scanner")
st.caption("Versi ketat untuk modal real: risk-on, liquidity, trend, relative strength, volume, breakout. Tidak ada ML dan tidak ada score rumit.")

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
        source_mode = st.radio("Sumber ticker", ["Paste ticker", "Upload CSV"], index=0)
        auto_suffix = st.checkbox("Auto tambah .JK", value=True)
        manual_text = st.text_area("Ticker (pisahkan koma / baris baru)", value="BBRI, BMRI, BBCA, ASII, ADRO", height=130, disabled=(source_mode != "Paste ticker"))
        csv_file = st.file_uploader("Upload CSV universe", type=["csv"], disabled=(source_mode != "Upload CSV"))

    with st.expander("Scanner settings", expanded=True):
        benchmark_ticker = st.text_input("Benchmark IHSG", value="^JKSE")
        period = st.selectbox("History period", ["1y", "2y", "3y", "5y"], index=1)
        min_price = st.number_input("Min harga", min_value=0.0, value=500.0, step=50.0)
        min_avg_dollar_vol = st.number_input("Min avg dollar volume 20D", min_value=0.0, value=5_000_000_000.0, step=500_000_000.0, format="%.0f")
        max_workers = st.slider("Parallel workers", 1, 4, 2)
        top_n = st.slider("Top N hasil", 5, 100, 25)

    with st.expander("Entry plan", expanded=False):
        account_size = st.number_input("Account size", min_value=0.0, value=100_000_000.0, step=5_000_000.0, format="%.0f")
        risk_pct = st.slider("Risk per trade (%)", 0.1, 5.0, 1.0, 0.1)
        lot_rounding = st.checkbox("Round to lots (100 shares)", value=True)

    scan_btn = st.button("Scan sekarang", type="primary", width="stretch")

if scan_btn:
    tickers = parse_universe(source_mode, csv_file, manual_text, auto_suffix=auto_suffix)
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
    df = df.sort_values(["final_score", "vol_rvol20", "avg_dollar_vol_20"], ascending=False, na_position="last")

    filtered = df[df["final_score"].fillna(-1) >= 5].copy()

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
        Scanner ini sengaja dibuat ketat dan sederhana:
        - Market harus **risk-on**
        - Saham harus **likuid**
        - Trend harus **bullish**
        - Relative strength harus **mengalahkan benchmark**
        - Volume harus **menguat**
        - Entry hanya pada **breakout**
        """
    )
