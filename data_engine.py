from __future__ import annotations

import random
import time

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


def normalize_ticker(symbol: str) -> str:
    s = str(symbol).strip().upper()
    if not s or s == "NAN":
        return ""
    if s.startswith("^"):
        return s
    return s if s.endswith(".JK") else f"{s}.JK"


def make_flow_score(flow_mode: str) -> float:
    mapping = {
        "Big Akumulasi": 95.0,
        "Small Akumulasi": 75.0,
        "Netral": 50.0,
        "Small Distribusi": 30.0,
        "Big Distribusi": 10.0,
    }
    return mapping.get(flow_mode, 50.0)


def map_flow_to_score(flow_mode: str) -> float:
    """Backward-compatible alias kept for older call sites."""
    return make_flow_score(flow_mode)


def _ticker_candidates(symbol: str) -> list[str]:
    base = str(symbol).strip().upper()
    if not base or base == "NAN":
        return []

    candidates: list[str] = []

    def add(candidate: str) -> None:
        candidate = str(candidate).strip().upper()
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    add(base)
    if base.startswith("^"):
        return candidates

    if base.endswith(".JK"):
        add(base[:-3])
    else:
        add(f"{base}.JK")

    return candidates


def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        top = list(out.columns.get_level_values(0))
        if any(col in top for col in ["Open", "High", "Low", "Close", "Volume"]):
            out.columns = out.columns.get_level_values(0)
        else:
            out.columns = out.columns.get_level_values(-1)

    out = out.loc[:, ~out.columns.duplicated()].copy()
    needed = {"Open", "High", "Low", "Close", "Volume"}
    if not needed.issubset(set(out.columns)):
        return pd.DataFrame()

    out = out.dropna(how="all").copy()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    for col in needed:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=list(needed)).copy()
    return out if not out.empty else pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def load_ticker_data(symbol: str, months: int) -> pd.DataFrame:
    """Load OHLCV history with conservative retries and strong caching.

    A single failed request is common for some IDX tickers and should not be
    amplified into repeated hits. This function makes at most one primary
    request and one fallback request per candidate symbol, which materially
    reduces Yahoo rate-limit pressure in Streamlit deployments.
    """
    try:
        months = int(months)
    except Exception:
        months = 12
    months = max(1, months)

    end = pd.Timestamp.now(tz="UTC").tz_localize(None)
    start = end - pd.DateOffset(months=months)

    base = str(symbol).strip()
    candidates: list[str] = []
    if base:
        candidates.append(base)
        if base.endswith(".JK"):
            candidates.append(base[:-3])
        elif not base.startswith("^"):
            candidates.append(f"{base}.JK")

    seen: set[str] = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    for candidate in candidates:
        try:
            # Primary request.
            df = yf.download(
                candidate,
                period=f"{months}mo",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            df = _standardize_ohlcv(df)
            if not df.empty:
                return df

            # Fallback request using explicit dates. We keep this to exactly one
            # additional call to avoid hammering Yahoo on bad symbols.
            df = yf.download(
                candidate,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            df = _standardize_ohlcv(df)
            if not df.empty:
                return df
        except Exception:
            # One lightweight retry with a tiny jitter can recover from transient
            # network hiccups, but we do not keep looping on a failing symbol.
            try:
                time.sleep(0.15 + random.random() * 0.1)
                df = yf.download(
                    candidate,
                    period=f"{months}mo",
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    threads=False,
                )
                df = _standardize_ohlcv(df)
                if not df.empty:
                    return df
            except Exception:
                pass

    return pd.DataFrame()


@st.cache_data(ttl=21600, show_spinner=False)
def load_yf_info(symbol: str) -> dict:
    """Cached, low-noise Yahoo Finance metadata fetch.

    We prefer fast_info first because it is materially lighter and often avoids
    crumb / 401 failures seen on the full quoteSummary endpoint. The full info
    endpoint is only attempted when fast_info yields nothing useful.
    """
    base = str(symbol).strip()
    if not base:
        return {}

    for candidate in _ticker_candidates(base):
        try:
            ticker = yf.Ticker(candidate)
            merged: dict = {}

            # Prefer the lightweight path first.
            try:
                fast_info = getattr(ticker, "fast_info", None)
                if fast_info is not None:
                    fast_dict = dict(fast_info)
                    if isinstance(fast_dict, dict) and fast_dict:
                        merged.update(fast_dict)
            except Exception:
                pass

            # Only fall back to the heavy info endpoint if fast_info was empty.
            if not merged:
                info = {}
                try:
                    info = ticker.get_info() or {}
                except Exception:
                    try:
                        info = ticker.info or {}
                    except Exception:
                        info = {}

                if isinstance(info, dict) and info:
                    merged.update(info)

            if merged:
                merged["_resolved_symbol"] = candidate
                merged["_info_source"] = "fast_info" if merged.get("lastPrice") is not None or merged.get("currency") is not None else "info"
                return merged
        except Exception:
            continue

    return {}


def parse_universe_text(text: str) -> list[str]:
    tokens: list[str] = []
    for line in text.splitlines():
        line = line.strip().upper()
        if not line:
            continue
        parts = [p.strip().upper() for p in line.replace(";", ",").split(",")]
        tokens.extend([p for p in parts if p])

    cleaned = []
    for t in tokens:
        norm = normalize_ticker(t)
        if norm:
            cleaned.append(norm)
    return list(dict.fromkeys(cleaned))


def load_universe_from_csv(source) -> list[str]:
    if source is None:
        return []
    try:
        dfu = pd.read_csv(source)
    except Exception:
        return []

    if dfu.empty:
        return []

    ticker_col = next(
        (
            col
            for col in dfu.columns
            if str(col).strip().lower() in {"ticker", "symbol", "kode", "code", "stock", "saham"}
        ),
        dfu.columns[0],
    )

    vals = dfu[ticker_col].astype(str).str.upper().str.strip().tolist()
    out = []
    for v in vals:
        norm = normalize_ticker(v)
        if norm:
            out.append(norm)
    return list(dict.fromkeys(out))
