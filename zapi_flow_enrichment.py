from __future__ import annotations

"""Pasticuan-owned ZAPI foreign-flow enrichment with bounded scoring influence.

The module implements the upstream ZAPI contract locally while preserving
an explicit evidence boundary: foreign flow can confirm accumulation/smart-money
hypotheses, but it never identifies a broker or beneficial owner and it never
replaces price-structure/SMC evidence.
"""

from io import BytesIO
from typing import Any, Iterable, Mapping
import os
import time

import numpy as np
import pandas as pd
import requests

from idx_trading_calendar import (
    latest_expected_completed_session,
    previous_idx_session,
    trading_session_age,
)


ZAPI_FLOW_ENRICHMENT_VERSION = "1.1.0-pasticuan-owned-idx-session-coverage"
ZAPI_FOREIGN_FLOW_URL = "https://api.zpi.web.id/v1/finance:idx/foreign-flow"
ZAPI_STOCK_SUMMARY_URL = "https://api.zpi.web.id/v1/finance:idx/stock-summary"
OWNED_CACHE_URL = "https://raw.githubusercontent.com/rizanrizan93/pasticuan/main/data/zapi_foreign_flow_60d.csv.gz"

_HISTORY_CACHE: dict[str, tuple[float, pd.DataFrame, dict[str, Any]]] = {}
_FEATURE_CACHE: dict[tuple[str, ...], tuple[float, pd.DataFrame, dict[str, Any]]] = {}


def _canonical(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text[:-3] if text.endswith(".JK") else text


def _ticker_column(frame: pd.DataFrame) -> str:
    for name in ("ticker", "symbol", "code", "stock_code", "stockcode"):
        if name in frame.columns:
            return name
    return ""


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _clip(value: Any, low: float = 0.0, high: float = 100.0) -> float:
    number = _finite(value, np.nan)
    if not np.isfinite(number):
        return np.nan
    return float(np.clip(number, low, high))


def _secret(name: str) -> str:
    value = str(os.getenv(name, "") or "").strip()
    if value:
        return value
    try:
        import streamlit as st  # imported lazily so tests do not require a UI context
        return str(st.secrets.get(name, "") or "").strip()
    except Exception:
        return ""


def _http_get(url: str, *, params: Mapping[str, Any] | None = None, api_key: str = "", timeout: float = 12.0, raw: bool = False) -> Any:
    headers = {
        "Accept": "application/json,text/plain,*/*",
        "User-Agent": "Mozilla/5.0 (compatible; IDX-ZAPI-Flow-Enrichment/1.0)",
    }
    if api_key:
        headers["x-api-key"] = api_key
    try:
        from curl_cffi import requests as curl_requests
        response = curl_requests.get(
            url,
            params=dict(params or {}),
            headers=headers,
            timeout=timeout,
            impersonate="chrome",
        )
    except Exception:
        response = requests.get(url, params=dict(params or {}), headers=headers, timeout=timeout)
    status = int(getattr(response, "status_code", 0) or 0)
    if status in {401, 403}:
        raise RuntimeError(f"ZAPI_AUTH_HTTP_{status}")
    if status == 429:
        raise RuntimeError("ZAPI_RATE_LIMIT_HTTP_429")
    response.raise_for_status()
    return response.content if raw else response.json()


def _unwrap_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    current = payload
    for _ in range(3):
        nested = current.get("data")
        if isinstance(nested, dict) and any(key in nested for key in ("data", "recordsTotal", "total", "date")):
            current = nested
        else:
            break
    return current


def _normalize_foreign_payload(payload: Any, trade_date: Any, universe: set[str]) -> pd.DataFrame:
    root = _unwrap_payload(payload)
    rows = root.get("data")
    if not isinstance(rows, list):
        return pd.DataFrame()
    unit = str(root.get("unit") or "shares").strip().lower()
    if unit not in {"share", "shares", "lembar", "lembar saham"}:
        raise RuntimeError(f"ZAPI_UNEXPECTED_FOREIGN_UNIT_{unit}")
    fallback = pd.to_datetime(trade_date, errors="coerce")
    response_date = pd.to_datetime(root.get("date"), errors="coerce")
    day = response_date.normalize() if pd.notna(response_date) else fallback.normalize()
    normalized: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        ticker = _canonical(item.get("code"))
        if not ticker or ticker not in universe:
            continue
        buy = _finite(item.get("foreignBuyShares"), np.nan)
        sell = _finite(item.get("foreignSellShares"), np.nan)
        raw_net = _finite(item.get("netForeignShares"), np.nan)
        net = raw_net if np.isfinite(raw_net) else (buy - sell if np.isfinite(buy) and np.isfinite(sell) else np.nan)
        if not any(np.isfinite(value) for value in (buy, sell, net)):
            continue
        normalized.append({
            "ticker": ticker,
            "trade_date": day,
            "foreign_buy_shares": buy,
            "foreign_sell_shares": sell,
            "foreign_net_shares": net,
            "volume": _finite(item.get("volume"), np.nan),
            "value": _finite(item.get("value"), np.nan),
            "source": "ZAPI_IDX_FOREIGN_FLOW",
            "flow_unit": "SHARES",
        })
    return pd.DataFrame(normalized)


def _normalize_stock_summary_payload(payload: Any, trade_date: Any, universe: set[str]) -> pd.DataFrame:
    root = _unwrap_payload(payload)
    rows = root.get("data")
    if not isinstance(rows, list):
        return pd.DataFrame()
    fallback = pd.to_datetime(trade_date, errors="coerce")
    normalized: list[dict[str, Any]] = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        ticker = _canonical(item.get("StockCode"))
        if not ticker or ticker not in universe:
            continue
        item_date = pd.to_datetime(item.get("Date"), errors="coerce")
        day = item_date.normalize() if pd.notna(item_date) else fallback.normalize()
        buy = _finite(item.get("ForeignBuy"), np.nan)
        sell = _finite(item.get("ForeignSell"), np.nan)
        if not any(np.isfinite(value) for value in (buy, sell)):
            continue
        net = buy - sell if np.isfinite(buy) and np.isfinite(sell) else np.nan
        normalized.append({
            "ticker": ticker,
            "trade_date": day,
            "foreign_buy_shares": buy,
            "foreign_sell_shares": sell,
            "foreign_net_shares": net,
            "volume": _finite(item.get("Volume"), np.nan),
            "value": _finite(item.get("Value"), np.nan),
            "frequency": _finite(item.get("Frequency"), np.nan),
            "bid": _finite(item.get("Bid"), np.nan),
            "offer": _finite(item.get("Offer"), np.nan),
            "bid_volume": _finite(item.get("BidVolume"), np.nan),
            "offer_volume": _finite(item.get("OfferVolume"), np.nan),
            "listed_shares": _finite(item.get("ListedShares"), np.nan),
            "tradable_shares": _finite(item.get("TradebleShares"), np.nan),
            "source": "ZAPI_IDX_STOCK_SUMMARY_FALLBACK",
            "flow_unit": "SHARES",
        })
    return pd.DataFrame(normalized)


def _paginate_day(
    url: str,
    trade_date: Any,
    universe: Iterable[Any],
    *,
    api_key: str,
    normalizer: Any,
    page_size: int = 1000,
    max_pages: int = 6,
) -> pd.DataFrame:
    names = {_canonical(value) for value in universe if _canonical(value)}
    if not names or not api_key:
        return pd.DataFrame()
    day = pd.Timestamp(trade_date).date().isoformat()
    start = 0
    parts: list[pd.DataFrame] = []
    total: int | None = None
    for _ in range(max(1, int(max_pages))):
        payload = _http_get(
            url,
            params={"date": day, "length": max(1, min(int(page_size), 1000)), "start": start},
            api_key=api_key,
            timeout=12.0,
        )
        root = _unwrap_payload(payload)
        raw_rows = root.get("data") if isinstance(root.get("data"), list) else []
        frame = normalizer(payload, trade_date, names)
        if not frame.empty:
            parts.append(frame)
        candidates = [root.get("total"), root.get("recordsFiltered"), root.get("recordsTotal")]
        totals = [pd.to_numeric(value, errors="coerce") for value in candidates]
        raw_total = next((value for value in totals if pd.notna(value)), None)
        returned = len(raw_rows)
        total = int(raw_total) if raw_total is not None else start + returned
        if returned <= 0 or total <= start + returned:
            break
        start += returned
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True, sort=False)
    return out.drop_duplicates(["ticker", "trade_date"], keep="first").reset_index(drop=True)


def _fetch_direct_day(universe: Iterable[Any], trade_date: Any, api_key: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not api_key:
        return pd.DataFrame(), {"state": "NO_ZAPI_KEY", "provider": "ZAPI_DIRECT"}
    try:
        foreign = _paginate_day(
            ZAPI_FOREIGN_FLOW_URL,
            trade_date,
            universe,
            api_key=api_key,
            normalizer=_normalize_foreign_payload,
        )
        if not foreign.empty:
            return foreign, {"state": "OK", "provider": "ZAPI_IDX_FOREIGN_FLOW", "rows": len(foreign)}
    except Exception as foreign_exc:
        foreign_error = f"{type(foreign_exc).__name__}: {str(foreign_exc)[:160]}"
    else:
        foreign_error = "NO_DATA"
    try:
        fallback = _paginate_day(
            ZAPI_STOCK_SUMMARY_URL,
            trade_date,
            universe,
            api_key=api_key,
            normalizer=_normalize_stock_summary_payload,
        )
        if not fallback.empty:
            return fallback, {
                "state": "OK_FALLBACK",
                "provider": "ZAPI_IDX_STOCK_SUMMARY_FALLBACK",
                "rows": len(fallback),
                "foreign_flow_error": foreign_error,
            }
        return pd.DataFrame(), {
            "state": "NO_DATA",
            "provider": "ZAPI_DIRECT",
            "foreign_flow_error": foreign_error,
        }
    except Exception as fallback_exc:
        return pd.DataFrame(), {
            "state": "FAIL_SOFT",
            "provider": "ZAPI_DIRECT",
            "foreign_flow_error": foreign_error,
            "fallback_error": f"{type(fallback_exc).__name__}: {str(fallback_exc)[:160]}",
        }


def _normalize_history(frame: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "ticker", "trade_date", "foreign_buy_shares", "foreign_sell_shares",
        "foreign_net_shares", "volume", "value", "source", "flow_unit",
    ]
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(columns=columns)
    out = frame.copy()
    renames = {
        "date": "trade_date",
        "foreign_buy": "foreign_buy_shares",
        "foreign_sell": "foreign_sell_shares",
        "foreign_net": "foreign_net_shares",
        "net_foreign_shares": "foreign_net_shares",
    }
    for old, new in renames.items():
        if old in out.columns and new not in out.columns:
            out = out.rename(columns={old: new})
    ticker_col = _ticker_column(out)
    if ticker_col and ticker_col != "ticker":
        out = out.rename(columns={ticker_col: "ticker"})
    if "ticker" not in out.columns or "trade_date" not in out.columns:
        return pd.DataFrame(columns=columns)
    out["ticker"] = out["ticker"].map(_canonical)
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    for name in (
        "foreign_buy_shares", "foreign_sell_shares", "foreign_net_shares",
        "volume", "value", "bid", "offer", "bid_volume", "offer_volume",
    ):
        if name in out.columns:
            out[name] = pd.to_numeric(out[name], errors="coerce")
    if "foreign_net_shares" not in out.columns:
        buy = pd.to_numeric(
            out["foreign_buy_shares"] if "foreign_buy_shares" in out.columns else pd.Series(np.nan, index=out.index),
            errors="coerce",
        )
        sell = pd.to_numeric(
            out["foreign_sell_shares"] if "foreign_sell_shares" in out.columns else pd.Series(np.nan, index=out.index),
            errors="coerce",
        )
        out["foreign_net_shares"] = buy - sell
    if "source" not in out.columns:
        out["source"] = "ZAPI_CACHE_SOURCE_UNKNOWN"
    if "flow_unit" not in out.columns:
        out["flow_unit"] = "SHARES"
    out = out.dropna(subset=["ticker", "trade_date"])
    out = out[out["ticker"].ne("")]
    return out.sort_values(["ticker", "trade_date"], kind="stable").drop_duplicates(
        ["ticker", "trade_date"], keep="last"
    ).reset_index(drop=True)


def _load_owned_cache() -> tuple[pd.DataFrame, dict[str, Any]]:
    cached = _HISTORY_CACHE.get("owned")
    now = time.monotonic()
    if cached and cached[0] > now:
        return cached[1].copy(), dict(cached[2])
    try:
        payload = _http_get(OWNED_CACHE_URL, timeout=10.0, raw=True)
        frame = pd.read_csv(BytesIO(payload), compression="gzip")
        frame = _normalize_history(frame)
        meta = {
            "state": "OK" if not frame.empty else "EMPTY",
            "provider": "PASTICUAN_OWNED_ZAPI_CACHE",
            "rows": int(len(frame)),
            "latest_trade_date": (
                pd.to_datetime(frame["trade_date"], errors="coerce").max().date().isoformat()
                if not frame.empty else ""
            ),
        }
    except Exception as exc:
        frame = pd.DataFrame()
        meta = {
            "state": "FAIL_SOFT",
            "provider": "PASTICUAN_OWNED_ZAPI_CACHE",
            "rows": 0,
            "error": f"{type(exc).__name__}: {str(exc)[:180]}",
        }
    _HISTORY_CACHE["owned"] = (now + 900.0, frame.copy(), dict(meta))
    return frame, meta


def _expected_idx_sessions(anchor: Any = None, count: int = 20) -> list[pd.Timestamp]:
    """Return completed IDX sessions newest-first; unknown calendars fail closed."""
    current = latest_expected_completed_session(anchor)
    values: list[pd.Timestamp] = []
    while len(values) < max(1, int(count)):
        values.append(pd.Timestamp(current).normalize())
        current = previous_idx_session(current, include_date=False)
    return values


def _recent_weekdays(anchor: pd.Timestamp | None = None, count: int = 3) -> list[pd.Timestamp]:
    """Compatibility name: refresh dates are actual IDX sessions, not weekdays."""
    return _expected_idx_sessions(anchor, count)


def _load_history_for_universe(tickers: Iterable[Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    names = sorted({_canonical(value) for value in tickers if _canonical(value)})
    owned, meta = _load_owned_cache()
    history = owned.loc[owned["ticker"].isin(names)].copy() if not owned.empty else pd.DataFrame()
    latest = pd.to_datetime(history.get("trade_date"), errors="coerce").max() if not history.empty else pd.NaT
    recent = _recent_weekdays(count=3)
    fresh_enough = pd.notna(latest) and latest >= recent[0]
    api_key = _secret("ZAPI_KEY")
    direct_meta: dict[str, Any] = {
        "state": "NOT_NEEDED" if fresh_enough else ("NOT_ATTEMPTED" if api_key else "NO_ZAPI_KEY"),
        "provider": "ZAPI_DIRECT",
    }
    if not fresh_enough and api_key:
        for day in recent:
            direct, direct_meta = _fetch_direct_day(names, day, api_key)
            if not direct.empty:
                direct = _normalize_history(direct)
                frames = [frame for frame in (history, direct) if isinstance(frame, pd.DataFrame) and not frame.empty]
                history = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
                history = _normalize_history(history)
                break
    combined_meta = {
        "state": "OK" if not history.empty else (direct_meta.get("state") or meta.get("state") or "NO_DATA"),
        "owned_cache_state": meta.get("state"),
        "owned_cache_provider": meta.get("provider"),
        "direct_state": direct_meta.get("state"),
        "direct_provider": direct_meta.get("provider"),
        "zapi_key_available": bool(api_key),
        "rows": int(len(history)),
        "latest_trade_date": (
            pd.to_datetime(history["trade_date"], errors="coerce").max().date().isoformat()
            if not history.empty else ""
        ),
    }
    return history, combined_meta


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return pd.to_numeric(numerator, errors="coerce").div(
        pd.to_numeric(denominator, errors="coerce").replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)


def _ratio_abs_score(series: pd.Series) -> pd.Series:
    # Foreign net participation of +/- 2% of daily volume is already material.
    return (50.0 + 2500.0 * pd.to_numeric(series, errors="coerce")).clip(0.0, 100.0)


def _cross_section_score(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.notna()
    out = pd.Series(np.nan, index=series.index, dtype=float)
    if int(valid.sum()) >= 3:
        out.loc[valid] = numeric.loc[valid].rank(method="average", pct=True) * 100.0
    elif int(valid.sum()) > 0:
        out.loc[valid] = 50.0
    return out


def score_foreign_history(
    history: pd.DataFrame,
    universe: Iterable[Any] | None = None,
    *,
    as_of: Any = None,
) -> pd.DataFrame:
    frame = _normalize_history(history)
    names = sorted({_canonical(value) for value in (universe or frame.get("ticker", [])) if _canonical(value)})
    if not names:
        return pd.DataFrame()
    expected = _expected_idx_sessions(as_of, 20)
    expected_set = set(expected)
    expected5 = set(expected[:5])
    expected10 = set(expected[:10])
    latest_expected = expected[0]
    expected_dates = "|".join(day.date().isoformat() for day in expected)
    rows: list[dict[str, Any]] = []
    for ticker in names:
        ticker_history = frame.loc[frame["ticker"].eq(ticker)].copy()
        local = ticker_history.loc[ticker_history["trade_date"].isin(expected_set)].sort_values("trade_date").copy()
        observed = int(local["trade_date"].nunique()) if not local.empty else 0
        coverage_ratio = observed / 20.0
        sources = sorted(set(str(value) for value in local.get("source", pd.Series(dtype=str)).dropna() if str(value).strip()))
        source_text = " | ".join(sources)
        verified_sources = {"ZAPI_IDX_FOREIGN_FLOW", "ZAPI_IDX_STOCK_SUMMARY_FALLBACK"}
        provenance = "VERIFIED_UPSTREAM_ZAPI" if sources and set(sources).issubset(verified_sources) else ("UNVERIFIED_SOURCE" if sources else "MISSING")
        if local.empty:
            rows.append({
                "ticker": ticker,
                "zapi_foreign_latest_trade_date": np.nan,
                "zapi_foreign_observed_days": 0,
                "zapi_foreign_net_shares_1d": np.nan,
                "zapi_foreign_net_shares_5d": np.nan,
                "zapi_foreign_net_shares_10d": np.nan,
                "zapi_foreign_net_shares_20d": np.nan,
                "zapi_foreign_net_participation_1d": np.nan,
                "zapi_foreign_net_participation_5d": np.nan,
                "zapi_foreign_net_participation_10d": np.nan,
                "zapi_foreign_net_participation_20d": np.nan,
                "zapi_foreign_positive_days_ratio_5d": np.nan,
                "zapi_foreign_positive_days_ratio_20d": np.nan,
                "zapi_foreign_buy_ratio_20d": np.nan,
                "zapi_foreign_buy_ratio_5d": np.nan,
                "zapi_flow_source": source_text,
                "zapi_flow_unit": "SHARES",
                "foreign_source": source_text,
                "foreign_latest_session": "",
                "foreign_expected_sessions": 20,
                "foreign_expected_session_dates": expected_dates,
                "foreign_observed_sessions": 0,
                "foreign_coverage_ratio": 0.0,
                "foreign_freshness_state": "MISSING",
                "foreign_provenance": provenance,
                "foreign_window_state": "MISSING",
                "_foreign_freshness_factor": 0.0,
            })
            continue
        latest = local.iloc[-1]
        recent5 = local.loc[local["trade_date"].isin(expected5)]
        recent10 = local.loc[local["trade_date"].isin(expected10)]
        buy20 = pd.to_numeric(local.get("foreign_buy_shares"), errors="coerce")
        sell20 = pd.to_numeric(local.get("foreign_sell_shares"), errors="coerce")
        net20 = pd.to_numeric(local.get("foreign_net_shares"), errors="coerce")
        vol20 = pd.to_numeric(local.get("volume"), errors="coerce")
        net5 = pd.to_numeric(recent5.get("foreign_net_shares"), errors="coerce")
        net10 = pd.to_numeric(recent10.get("foreign_net_shares"), errors="coerce")
        vol5 = pd.to_numeric(recent5.get("volume"), errors="coerce")
        vol10 = pd.to_numeric(recent10.get("volume"), errors="coerce")
        gross20 = buy20.fillna(0.0) + sell20.fillna(0.0)
        gross5 = pd.to_numeric(recent5.get("foreign_buy_shares"), errors="coerce").fillna(0.0) + pd.to_numeric(recent5.get("foreign_sell_shares"), errors="coerce").fillna(0.0)
        latest_volume = _finite(latest.get("volume"), np.nan)
        latest_net = _finite(latest.get("foreign_net_shares"), np.nan)
        latest_session = pd.Timestamp(latest.get("trade_date")).normalize()
        lag = trading_session_age(latest_session, latest_expected)
        if lag == 0:
            freshness_state, freshness_factor = "FRESH", 1.0
        elif lag == 1:
            freshness_state, freshness_factor = "LAGGING_1_SESSION", 0.85
        elif lag == 2:
            freshness_state, freshness_factor = "LAGGING_2_SESSIONS", 0.55
        else:
            freshness_state, freshness_factor = "STALE", 0.0
        window_state = "SUFFICIENT_20D" if observed == 20 else "PARTIAL"
        rows.append({
            "ticker": ticker,
            "zapi_foreign_latest_trade_date": latest_session,
            "zapi_foreign_observed_days": observed,
            "zapi_foreign_net_shares_1d": latest_net,
            "zapi_foreign_net_shares_5d": float(net5.sum(min_count=1)),
            "zapi_foreign_net_shares_10d": float(net10.sum(min_count=1)),
            "zapi_foreign_net_shares_20d": float(net20.sum(min_count=1)),
            "zapi_foreign_net_participation_1d": latest_net / latest_volume if np.isfinite(latest_net) and np.isfinite(latest_volume) and latest_volume > 0 else np.nan,
            "zapi_foreign_net_participation_5d": float(net5.sum(min_count=1) / vol5.sum(min_count=1)) if vol5.sum(min_count=1) > 0 else np.nan,
            "zapi_foreign_net_participation_10d": float(net10.sum(min_count=1) / vol10.sum(min_count=1)) if vol10.sum(min_count=1) > 0 else np.nan,
            "zapi_foreign_net_participation_20d": float(net20.sum(min_count=1) / vol20.sum(min_count=1)) if vol20.sum(min_count=1) > 0 else np.nan,
            "zapi_foreign_positive_days_ratio_5d": float(net5.gt(0).mean()) if net5.notna().any() else np.nan,
            "zapi_foreign_positive_days_ratio_20d": float(net20.gt(0).mean()) if net20.notna().any() else np.nan,
            "zapi_foreign_buy_ratio_20d": float(buy20.sum(min_count=1) / gross20.sum(min_count=1)) if gross20.sum(min_count=1) > 0 else np.nan,
            "zapi_foreign_buy_ratio_5d": float(pd.to_numeric(recent5.get("foreign_buy_shares"), errors="coerce").sum(min_count=1) / gross5.sum(min_count=1)) if gross5.sum(min_count=1) > 0 else np.nan,
            "zapi_flow_source": source_text,
            "zapi_flow_unit": "SHARES",
            "foreign_source": source_text,
            "foreign_latest_session": latest_session.date().isoformat(),
            "foreign_expected_sessions": 20,
            "foreign_expected_session_dates": expected_dates,
            "foreign_observed_sessions": observed,
            "foreign_coverage_ratio": coverage_ratio,
            "foreign_freshness_state": freshness_state,
            "foreign_provenance": provenance,
            "foreign_window_state": window_state,
            "_foreign_freshness_factor": freshness_factor,
        })
    out = pd.DataFrame(rows)
    ratio20 = pd.to_numeric(out["zapi_foreign_net_participation_20d"], errors="coerce")
    ratio5 = pd.to_numeric(out["zapi_foreign_net_participation_5d"], errors="coerce")
    ratio20_rank = _cross_section_score(ratio20)
    ratio5_rank = _cross_section_score(ratio5)
    ratio20_abs = _ratio_abs_score(ratio20)
    ratio5_abs = _ratio_abs_score(ratio5)
    score20 = 0.55 * pd.to_numeric(ratio20_rank, errors="coerce") + 0.45 * pd.to_numeric(ratio20_abs, errors="coerce")
    score5 = 0.55 * pd.to_numeric(ratio5_rank, errors="coerce") + 0.45 * pd.to_numeric(ratio5_abs, errors="coerce")
    persistence20_score = 100.0 * pd.to_numeric(out["zapi_foreign_positive_days_ratio_20d"], errors="coerce")
    persistence5_score = 100.0 * pd.to_numeric(out["zapi_foreign_positive_days_ratio_5d"], errors="coerce")
    buy_score = 100.0 * pd.to_numeric(out["zapi_foreign_buy_ratio_20d"], errors="coerce")

    components = pd.concat(
        [score20.rename("score20"), score5.rename("score5"), persistence20_score.rename("p20"), persistence5_score.rename("p5"), buy_score.rename("buy")],
        axis=1,
    )
    weights = pd.Series({"score20": 0.30, "score5": 0.30, "p20": 0.18, "p5": 0.12, "buy": 0.10})
    numerator = components.mul(weights, axis=1).sum(axis=1, min_count=1)
    denominator = components.notna().mul(weights, axis=1).sum(axis=1)
    flow_score = numerator.div(denominator.replace(0.0, np.nan)).clip(0.0, 100.0)

    freshness = pd.to_numeric(out.pop("_foreign_freshness_factor"), errors="coerce").fillna(0.0)
    history_cov = pd.to_numeric(out["foreign_coverage_ratio"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    coverage = (100.0 * history_cov * freshness).clip(0.0, 100.0)

    out["zapi_foreign_flow_score"] = flow_score.round(1)
    out["zapi_foreign_flow_coverage_pct"] = pd.Series(coverage, index=out.index).round(1)
    out["zapi_accumulation_confirmation_score"] = (
        0.70 * flow_score + 0.20 * persistence20_score.fillna(50.0) + 0.10 * persistence5_score.fillna(50.0)
    ).clip(0.0, 100.0).round(1)
    out["zapi_smart_money_confirmation_score"] = (
        0.75 * flow_score + 0.25 * persistence20_score.fillna(50.0)
    ).clip(0.0, 100.0).round(1)
    # This is intentionally a flow confirmation for an independently derived
    # market structure, never a replacement SMC score.
    out["zapi_smc_flow_confirmation_score"] = (
        0.80 * flow_score + 0.20 * score5.fillna(50.0)
    ).clip(0.0, 100.0).round(1)
    ratio20 = pd.to_numeric(out["zapi_foreign_net_participation_20d"], errors="coerce")
    out["zapi_foreign_state"] = np.select(
        [ratio20.ge(0.003), ratio20.le(-0.003)],
        ["NET_ACCUMULATION", "NET_DISTRIBUTION"],
        default="MIXED_NEUTRAL",
    )
    out["zapi_flow_evidence_type"] = "ZAPI_FOREIGN_FLOW_CONFIRMATION_NOT_BROKER_IDENTITY"
    out["zapi_flow_enrichment_version"] = ZAPI_FLOW_ENRICHMENT_VERSION
    return out


def get_zapi_features(tickers: Iterable[Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    codes = tuple(sorted({_canonical(value) for value in tickers if _canonical(value)}))
    if not codes:
        return pd.DataFrame(), {"state": "EMPTY_UNIVERSE"}
    cached = _FEATURE_CACHE.get(codes)
    now = time.monotonic()
    if cached and cached[0] > now:
        return cached[1].copy(), dict(cached[2])
    history, meta = _load_history_for_universe(codes)
    features = score_foreign_history(history, codes)
    if not features.empty:
        features["zapi_flow_meta_state"] = str(meta.get("state") or "UNKNOWN")
        features["zapi_owned_cache_state"] = str(meta.get("owned_cache_state") or "")
        features["zapi_direct_state"] = str(meta.get("direct_state") or "")
    _FEATURE_CACHE[codes] = (now + 600.0, features.copy(), dict(meta))
    while len(_FEATURE_CACHE) > 8:
        _FEATURE_CACHE.pop(next(iter(_FEATURE_CACHE)))
    return features, meta


def _blend(base_score: Any, base_coverage: Any, zapi_score: Any, zapi_coverage: Any, *, max_weight: float, zapi_only_coverage_cap: float = 55.0) -> tuple[float, float, float]:
    base = _finite(base_score, np.nan)
    base_cov = _clip(base_coverage)
    zapi = _finite(zapi_score, np.nan)
    zapi_cov = _clip(zapi_coverage)
    if not np.isfinite(zapi) or not np.isfinite(zapi_cov) or zapi_cov <= 0:
        return base, base_cov if np.isfinite(base_cov) else 0.0, 0.0
    z_weight = float(np.clip(float(max_weight) * zapi_cov / 100.0, 0.0, max_weight))
    if np.isfinite(base):
        blended = (1.0 - z_weight) * base + z_weight * zapi
        effective_base_cov = base_cov if np.isfinite(base_cov) else 0.0
        combined_cov = effective_base_cov + (100.0 - effective_base_cov) * zapi_cov / 100.0 * max_weight
        return float(np.clip(blended, 0.0, 100.0)), float(np.clip(combined_cov, 0.0, 100.0)), 100.0 * z_weight
    # Foreign-only evidence is useful research evidence but cannot masquerade as
    # a fully observed smart-money/accumulation profile.
    return float(np.clip(zapi, 0.0, 100.0)), float(min(zapi_only_coverage_cap, zapi_cov * max_weight)), 100.0 * z_weight


def _merge_features(frame: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or features.empty or "ticker" not in frame.columns:
        return frame.copy()
    left = frame.copy()
    left["_zapi_key"] = left["ticker"].map(_canonical)
    right = features.copy()
    right["_zapi_key"] = right["ticker"].map(_canonical)
    right = right.drop(columns=["ticker"], errors="ignore").drop_duplicates("_zapi_key", keep="last")
    duplicate = [column for column in right.columns if column != "_zapi_key" and column in left.columns]
    if duplicate:
        left = left.drop(columns=duplicate)
    return left.merge(right, on="_zapi_key", how="left").drop(columns=["_zapi_key"])


def _first_num(row: Mapping[str, Any], names: Iterable[str]) -> float:
    for name in names:
        value = _finite(row.get(name), np.nan)
        if np.isfinite(value):
            return value
    return np.nan


def enrich_super_universe(universe: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(universe, pd.DataFrame) or universe.empty or "ticker" not in universe.columns:
        return universe.copy() if isinstance(universe, pd.DataFrame) else pd.DataFrame()
    features, _ = get_zapi_features(universe["ticker"].tolist())
    out = _merge_features(universe, features)
    if out.empty:
        return out
    rows: list[dict[str, Any]] = []
    for _, row in out.iterrows():
        base = _first_num(row, ("flow_silent_accumulation_score", "sig_silent_accumulation_score"))
        base_cov = _first_num(row, ("flow_silent_accumulation_confidence", "flow_silent_accumulation_data_coverage", "sig_silent_accumulation_confidence"))
        zapi = _first_num(row, ("zapi_accumulation_confirmation_score",))
        zapi_cov = _first_num(row, ("zapi_foreign_flow_coverage_pct",))
        blended, coverage, weight_pct = _blend(base, base_cov, zapi, zapi_cov, max_weight=0.35)
        rows.append({
            "flow_silent_accumulation_score": blended,
            "flow_silent_accumulation_confidence": coverage,
            "flow_silent_accumulation_data_coverage": coverage,
            "zapi_confirmation_weight_pct": weight_pct,
            "zapi_super_original_silent_score": base,
            "zapi_super_original_silent_coverage_pct": base_cov,
            "zapi_super_flow_basis": "PRICE_VOLUME_SILENT_ACCUM_PLUS_BOUNDED_ZAPI_FOREIGN_CONFIRMATION",
        })
    overlay = pd.DataFrame(rows, index=out.index)
    for column in overlay.columns:
        out[column] = overlay[column]
    return out


def enrich_emir_radar(radar: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(radar, pd.DataFrame) or radar.empty or "ticker" not in radar.columns:
        return radar.copy() if isinstance(radar, pd.DataFrame) else pd.DataFrame()
    features, _ = get_zapi_features(radar["ticker"].tolist())
    out = _merge_features(radar, features)
    if out.empty:
        return out
    smart_values: list[float] = []
    smart_coverages: list[float] = []
    smart_weights: list[float] = []
    adjusted_conviction: list[float] = []
    deltas: list[float] = []
    for _, row in out.iterrows():
        smart = _first_num(row, ("smart_money_score",))
        smart_cov = _first_num(row, ("smart_money_coverage_pct",))
        zapi_smart = _first_num(row, ("zapi_smart_money_confirmation_score",))
        zapi_cov = _first_num(row, ("zapi_foreign_flow_coverage_pct",))
        blended_smart, blended_cov, weight_pct = _blend(smart, smart_cov, zapi_smart, zapi_cov, max_weight=0.30)
        smart_values.append(blended_smart)
        smart_coverages.append(blended_cov)
        smart_weights.append(weight_pct)

        conviction = _first_num(row, ("emir_conviction_score", "emir_final_score"))
        flow_score = _first_num(row, ("zapi_foreign_flow_score",))
        if np.isfinite(conviction) and np.isfinite(flow_score) and np.isfinite(zapi_cov) and zapi_cov > 0:
            directional = float(np.clip((flow_score - 50.0) / 50.0, -1.0, 1.0))
            delta = float(np.clip(2.5 * directional * zapi_cov / 100.0, -2.5, 2.5))
            adjusted = float(np.clip(conviction + delta, 0.0, 100.0))
        else:
            delta = 0.0
            adjusted = conviction
        deltas.append(delta)
        adjusted_conviction.append(adjusted)
    out["smart_money_score"] = smart_values
    out["smart_money_coverage_pct"] = smart_coverages
    out["zapi_smart_money_confirmation_weight_pct"] = smart_weights
    out["zapi_emir_conviction_delta"] = deltas
    if "emir_conviction_score" in out.columns:
        out["emir_conviction_score"] = adjusted_conviction
    if "emir_final_score" in out.columns:
        out["emir_final_score"] = adjusted_conviction
    out["zapi_emir_flow_basis"] = "EXISTING_EMIR_FLOW_PLUS_BOUNDED_ZAPI_FOREIGN_CONFIRMATION"
    return out


def blend_emir_dashboard_output(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    out = frame.copy()
    flow_values: list[float] = []
    silent_values: list[float] = []
    dominance_values: list[float] = []
    for _, row in out.iterrows():
        zapi = _first_num(row, ("zapi_foreign_flow_score",))
        zapi_acc = _first_num(row, ("zapi_accumulation_confirmation_score",))
        cov = _first_num(row, ("zapi_foreign_flow_coverage_pct",))
        base_flow = _first_num(row, ("dashboard_flow_score",))
        base_silent = _first_num(row, ("dashboard_silent_accum_score",))
        flow, _, _ = _blend(base_flow, 100.0 if np.isfinite(base_flow) else 0.0, zapi, cov, max_weight=0.30)
        silent, _, _ = _blend(base_silent, 100.0 if np.isfinite(base_silent) else 0.0, zapi_acc, cov, max_weight=0.20)
        distribution = _first_num(row, ("distribution_score",))
        if np.isfinite(flow) and np.isfinite(silent):
            parts = [(flow, 0.45), (silent, 0.35)]
            if np.isfinite(distribution):
                parts.append((100.0 - np.clip(distribution, 0.0, 100.0), 0.20))
            denominator = sum(weight for _, weight in parts)
            dominance = sum(value * weight for value, weight in parts) / denominator
        else:
            dominance = _first_num(row, ("dashboard_accumulation_dominance_pct",))
        flow_values.append(flow)
        silent_values.append(silent)
        dominance_values.append(dominance)
    out["dashboard_flow_score"] = flow_values
    out["dashboard_silent_accum_score"] = silent_values
    out["dashboard_accumulation_dominance_pct"] = dominance_values
    return out


__all__ = [
    "ZAPI_FLOW_ENRICHMENT_VERSION",
    "score_foreign_history",
    "get_zapi_features",
    "enrich_super_universe",
    "enrich_emir_radar",
    "blend_emir_dashboard_output",
]
