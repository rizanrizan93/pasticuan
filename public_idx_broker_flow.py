from __future__ import annotations

"""Public IDX participant-flow enrichment."""

from datetime import date, datetime, timedelta
from io import BytesIO
import gzip
import os
import re
import tempfile
from typing import Any, Iterable
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests

from idx_trade_detail_discovery import (
    DiscoveryAttempt,
    discover_trade_detail_url as _discover_trade_detail_url,
    download_trade_detail as _download_trade_detail,
)

PUBLIC_INDEX_URL = (
    "https://www.idxdata3.co.id/INET_Specification/Market_Summary/Market_Indices/"
    "IX200720.TXT?directory=.%2FIDX+Reporting+PSPP%2FRevitalisasi%2FPUBLIK%2F"
)
PUBLIC_CACHE_URL = (
    "https://raw.githubusercontent.com/rizanrizan93/pasticuan/main/"
    "data/public_broker_flow_30d.csv.gz"
)
VERSION = "1.0.0-public-idx-trade-detail-participant-flow"
SOURCE_NAME = "IDX_PUBLIC_TRADE_DETAIL_PUBLIK"


def _canon(value: Any) -> str:
    text = str(value or "").strip().upper()
    return text[:-3] if text.endswith(".JK") else text


def _num(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _latest_weekday(anchor: date | None = None) -> date:
    current = anchor or datetime.now().date()
    while current.weekday() >= 5:
        current -= timedelta(days=1)
    return current


def _headers() -> dict[str, str]:
    return {"User-Agent": "Mozilla/5.0 (compatible; IDX-Public-Broker-Collector/1.0)", "Accept": "text/csv,text/plain,*/*"}


def discover_trade_detail_url(
    trade_date: date,
    timeout: int = 20,
    diagnostics: list[DiscoveryAttempt] | None = None,
) -> str:
    return _discover_trade_detail_url(trade_date, timeout=timeout, diagnostics=diagnostics)


def download_trade_detail(
    trade_date: date,
    timeout: int = 45,
    diagnostics: list[DiscoveryAttempt] | None = None,
) -> tuple[str, str]:
    path, url = _download_trade_detail(trade_date, timeout=timeout, diagnostics=diagnostics)
    return str(path), url


def _read_trade_chunks(path: str, universe: set[str] | None = None, chunksize: int = 250_000):
    usecols = {"asset", "participant_buy", "participant_sell", "volume", "value", "tradingdate"}
    for sep in ("|", ","):
        try:
            iterator = pd.read_csv(path, sep=sep, usecols=lambda c: str(c).strip().lower() in usecols, chunksize=chunksize, low_memory=False, dtype=str)
            yielded = False
            for chunk in iterator:
                chunk.columns = [str(c).strip().lower() for c in chunk.columns]
                renames = {"seccode": "asset", "code": "asset", "ticker": "asset", "brokersellid": "participant_sell", "brokerbuyid": "participant_buy", "sellbrokerid": "participant_sell", "buybrokerid": "participant_buy", "quantity": "volume", "tradedate": "tradingdate"}
                for old, new in renames.items():
                    if old in chunk.columns and new not in chunk.columns:
                        chunk = chunk.rename(columns={old: new})
                required = {"asset", "participant_buy", "participant_sell", "volume", "value"}
                if not required.issubset(chunk.columns):
                    continue
                chunk["asset"] = chunk["asset"].map(_canon)
                if universe:
                    chunk = chunk[chunk["asset"].isin(universe)]
                if chunk.empty:
                    continue
                chunk["participant_buy"] = chunk["participant_buy"].astype(str).str.strip().str.upper()
                chunk["participant_sell"] = chunk["participant_sell"].astype(str).str.strip().str.upper()
                chunk["volume"] = pd.to_numeric(chunk["volume"], errors="coerce").fillna(0.0)
                chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce").fillna(0.0)
                yielded = True
                yield chunk
            if yielded:
                return
        except Exception:
            continue
    raise RuntimeError("IDX_TRADE_DETAIL_PARSE_FAILED")


def aggregate_trade_detail(path: str, trade_date: date, universe: Iterable[Any] | None = None) -> pd.DataFrame:
    names = {_canon(value) for value in (universe or []) if _canon(value)}
    buy_parts: list[pd.DataFrame] = []
    sell_parts: list[pd.DataFrame] = []
    for chunk in _read_trade_chunks(path, names or None):
        buy_parts.append(chunk[chunk["participant_buy"].ne("")].groupby(["asset", "participant_buy"], dropna=False, as_index=False).agg(buy_value=("value", "sum"), buy_volume=("volume", "sum")).rename(columns={"participant_buy": "broker_code"}))
        sell_parts.append(chunk[chunk["participant_sell"].ne("")].groupby(["asset", "participant_sell"], dropna=False, as_index=False).agg(sell_value=("value", "sum"), sell_volume=("volume", "sum")).rename(columns={"participant_sell": "broker_code"}))
    if not buy_parts and not sell_parts:
        return pd.DataFrame()
    buy = pd.concat(buy_parts, ignore_index=True).groupby(["asset", "broker_code"], as_index=False).sum()
    sell = pd.concat(sell_parts, ignore_index=True).groupby(["asset", "broker_code"], as_index=False).sum()
    out = buy.merge(sell, on=["asset", "broker_code"], how="outer").fillna(0.0).rename(columns={"asset": "ticker"})
    out["trade_date"] = pd.Timestamp(trade_date)
    out["buy_avg"] = out["buy_value"].div(out["buy_volume"].replace(0.0, np.nan))
    out["sell_avg"] = out["sell_value"].div(out["sell_volume"].replace(0.0, np.nan))
    out["net_value"] = out["buy_value"] - out["sell_value"]
    out["net_volume"] = out["buy_volume"] - out["sell_volume"]
    out["gross_value"] = out["buy_value"] + out["sell_value"]
    out["source"] = SOURCE_NAME
    out["source_verified"] = True
    out["provenance_state"] = "OFFICIAL_IDX_PUBLIC_EOD_TRADE_DETAIL_PARTICIPANT_FLOW_NOT_BENEFICIAL_OWNER"
    return out[["trade_date", "ticker", "broker_code", "buy_value", "sell_value", "buy_volume", "sell_volume", "buy_avg", "sell_avg", "net_value", "net_volume", "gross_value", "source", "source_verified", "provenance_state"]]


def trim_daily_top_flow(daily: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if not isinstance(daily, pd.DataFrame) or daily.empty:
        return pd.DataFrame()
    frame = daily.copy()
    positive, negative = frame[frame["net_value"] > 0].copy(), frame[frame["net_value"] < 0].copy()
    positive["side"], negative["side"] = "TOP_NET_BUYER", "TOP_NET_SELLER"
    positive["net_rank"] = positive.groupby(["trade_date", "ticker"])["net_value"].rank(method="first", ascending=False)
    negative["net_rank"] = negative.groupby(["trade_date", "ticker"])["net_value"].rank(method="first", ascending=True)
    out = pd.concat([positive[positive["net_rank"] <= top_n], negative[negative["net_rank"] <= top_n]], ignore_index=True)
    if out.empty:
        return out
    out["broker_flow_version"] = VERSION
    return out.sort_values(["trade_date", "ticker", "side", "net_rank"]).reset_index(drop=True)


def load_public_cache() -> pd.DataFrame:
    try:
        response = requests.get(PUBLIC_CACHE_URL, headers=_headers(), timeout=20)
        response.raise_for_status()
        return _normalize_cache(pd.read_csv(BytesIO(gzip.decompress(response.content))))
    except Exception:
        return _normalize_cache(pd.DataFrame())


def _normalize_cache(frame: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["trade_date", "ticker", "broker_code", "buy_value", "sell_value", "buy_volume", "sell_volume", "buy_avg", "sell_avg", "net_value", "net_volume", "gross_value", "source", "source_verified", "provenance_state", "side", "net_rank"]
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(columns=columns)
    out = frame.copy()
    out["ticker"] = out["ticker"].map(_canon)
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    for column in ("buy_value", "sell_value", "buy_volume", "sell_volume", "buy_avg", "sell_avg", "net_value", "net_volume", "gross_value", "net_rank"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.dropna(subset=["ticker", "trade_date"]).drop_duplicates(["trade_date", "ticker", "broker_code", "side"], keep="last").reset_index(drop=True)


def _cross_section(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.notna()
    score = pd.Series(np.nan, index=values.index, dtype=float)
    if int(valid.sum()) >= 3:
        score.loc[valid] = values.loc[valid].rank(pct=True) * 100.0
    elif bool(valid.any()):
        score.loc[valid] = 50.0
    return score


def score_broker_history(history: pd.DataFrame, universe: Iterable[Any]) -> pd.DataFrame:
    frame = _normalize_cache(history)
    names = sorted({_canon(value) for value in universe if _canon(value)})
    rows: list[dict[str, Any]] = []
    if "ticker" in frame.columns and not frame.empty:
        frame = frame[frame["ticker"].isin(names)]
    for ticker in names:
        local = frame[frame["ticker"].eq(ticker)].copy() if "ticker" in frame.columns else pd.DataFrame()
        if local.empty:
            rows.append({"ticker": ticker, "broker_flow_coverage_pct": 0.0, "broker_accumulation_state": "NO_DATA", "broker_flow_version": VERSION})
            continue
        dates = sorted(local["trade_date"].dropna().unique())[-20:]
        recent = local[local["trade_date"].isin(dates)]
        buyers, sellers = recent[recent["side"].eq("TOP_NET_BUYER")], recent[recent["side"].eq("TOP_NET_SELLER")]
        top3 = buyers.sort_values(["trade_date", "net_rank"]).groupby("trade_date").head(3)
        counts = top3["broker_code"].value_counts() if not top3.empty else pd.Series(dtype=float)
        top_broker = str(counts.index[0]) if not counts.empty else ""
        persistence = float(counts.iloc[0]) / max(1, len(dates)) * 100.0 if not counts.empty else np.nan
        broker_net = buyers.groupby("broker_code")["net_value"].sum().sort_values(ascending=False) if not buyers.empty else pd.Series(dtype=float)
        top_net = float(broker_net.iloc[0]) if len(broker_net) else np.nan
        positive_total = float(buyers["net_value"].clip(lower=0).sum()) if not buyers.empty else 0.0
        concentration = top_net / positive_total * 100.0 if top_net > 0 and positive_total > 0 else np.nan
        positive, negative = float(buyers["net_value"].sum()) if not buyers.empty else 0.0, float(sellers["net_value"].abs().sum()) if not sellers.empty else 0.0
        dominance_score = float(np.clip(50.0 + 15.0 * (positive / negative if negative > 0 else 5.0), 0, 100)) if positive > 0 else 30.0
        latest_date = max(dates) if dates else pd.NaT
        latest = local[local["trade_date"].eq(latest_date)] if pd.notna(latest_date) else pd.DataFrame()
        latest_buy = latest[latest["side"].eq("TOP_NET_BUYER")].sort_values("net_rank")
        latest_broker = str(latest_buy.iloc[0]["broker_code"]) if not latest_buy.empty else ""
        latest_avg = _num(latest_buy.iloc[0].get("buy_avg")) if not latest_buy.empty else np.nan
        rows.append({"ticker": ticker, "broker_flow_observed_days": int(recent["trade_date"].nunique()), "broker_flow_latest_date": latest_date, "broker_top_buyer_code": top_broker, "broker_latest_top_buyer_code": latest_broker, "broker_top3_buyer_persistence_20d_pct": persistence, "broker_top_buyer_net_value_20d": top_net, "broker_buyer_concentration_pct": concentration, "broker_buy_sell_dominance_score": dominance_score, "broker_latest_top_buyer_buy_avg": latest_avg, "broker_flow_coverage_pct": min(100.0, 100.0 * len(dates) / 20.0), "broker_flow_source": SOURCE_NAME, "broker_flow_provenance": "OFFICIAL_IDX_PUBLIC_EOD_TRADE_DETAIL_PARTICIPANT_FLOW_NOT_BENEFICIAL_OWNER", "broker_flow_version": VERSION})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if "broker_top_buyer_net_value_20d" not in out.columns:
        out["broker_net_score"] = np.nan
        out["broker_accumulation_score"] = np.nan
        out["broker_smart_money_confirmation_score"] = np.nan
        return out
    out["broker_net_score"] = _cross_section(out["broker_top_buyer_net_value_20d"])
    persistence = pd.to_numeric(out["broker_top3_buyer_persistence_20d_pct"], errors="coerce").clip(0, 100)
    concentration = pd.to_numeric(out["broker_buyer_concentration_pct"], errors="coerce").clip(0, 100)
    dominance = pd.to_numeric(out["broker_buy_sell_dominance_score"], errors="coerce").clip(0, 100)
    out["broker_accumulation_score"] = (0.35 * out["broker_net_score"].fillna(50.0) + 0.25 * persistence.fillna(0) + 0.20 * concentration.fillna(0) + 0.20 * dominance.fillna(50)).clip(0, 100).round(1)
    out["broker_smart_money_confirmation_score"] = (0.70 * out["broker_accumulation_score"] + 0.30 * out["broker_net_score"].fillna(50)).clip(0, 100).round(1)
    out["broker_accumulation_state"] = np.select([out["broker_accumulation_score"].ge(70), out["broker_accumulation_score"].le(35)], ["PARTICIPANT_ACCUMULATION", "PARTICIPANT_DISTRIBUTION"], default="PARTICIPANT_MIXED")
    return out


def get_broker_features(tickers: Iterable[Any], *, live_fallback: bool | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    names = sorted({_canon(value) for value in tickers if _canon(value)})
    history = load_public_cache()
    meta = {"source": "GITHUB_ROLLING_PUBLIC_IDX_BROKER_CACHE" if not history.empty else "NONE", "state": "OK" if not history.empty else "NO_CACHE", "observed_dates": int(history["trade_date"].nunique()) if not history.empty else 0}
    if live_fallback is None:
        live_fallback = str(os.getenv("IDX_BROKER_LIVE_FALLBACK", "1")).strip().lower() in {"1", "true", "yes", "on"}
    latest = pd.to_datetime(history.get("trade_date"), errors="coerce").max() if not history.empty else pd.NaT
    target = pd.Timestamp(_latest_weekday())
    if live_fallback and (pd.isna(latest) or latest < target) and names:
        path = None
        try:
            path, source_url = download_trade_detail(target.date())
            daily = trim_daily_top_flow(aggregate_trade_detail(path, target.date(), names))
            if not daily.empty:
                history = pd.concat([history, daily], ignore_index=True) if not history.empty else daily
                history = _normalize_cache(history)
                meta.update({"source": SOURCE_NAME, "state": "OK_LIVE_REFRESH", "latest_source_url": source_url, "observed_dates": int(history["trade_date"].nunique())})
        except Exception as exc:
            meta.update({"live_error": f"{type(exc).__name__}: {str(exc)[:180]}"})
        finally:
            if path:
                try: os.remove(path)
                except OSError: pass
    features = score_broker_history(history, names)
    if not features.empty:
        features["broker_flow_meta_state"] = meta.get("state", "UNKNOWN")
    return features, meta


def enrich_super_broker(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "ticker" not in frame.columns:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    features, _ = get_broker_features(frame["ticker"].tolist())
    if features.empty:
        return frame.copy()
    out, right = frame.copy(), features.copy()
    out["_key"], right["_key"] = out["ticker"].map(_canon), right["ticker"].map(_canon)
    right = right.drop(columns=["ticker"]).drop_duplicates("_key")
    duplicate = [c for c in right.columns if c != "_key" and c in out.columns]
    if duplicate: out = out.drop(columns=duplicate)
    out = out.merge(right, on="_key", how="left").drop(columns=["_key"])
    base = pd.to_numeric(out.get("flow_silent_accumulation_score", pd.Series(np.nan, index=out.index)), errors="coerce")
    broker = pd.to_numeric(out.get("broker_accumulation_score"), errors="coerce")
    coverage = pd.to_numeric(out.get("broker_flow_coverage_pct"), errors="coerce").clip(0, 100)
    weight = 0.20 * coverage.fillna(0) / 100.0
    out["broker_confirmation_weight_pct"] = (100 * weight).round(1)
    out["broker_pre_confirmation_accumulation_score"] = base
    out["broker_post_confirmation_accumulation_score"] = ((1 - weight) * base + weight * broker).where(base.notna() & broker.notna(), base).clip(0, 100).round(1)
    out["flow_silent_accumulation_score"] = out["broker_post_confirmation_accumulation_score"]
    out["broker_accumulation_delta"] = (out["broker_post_confirmation_accumulation_score"] - base).round(1)
    return out

__all__ = ["VERSION", "PUBLIC_CACHE_URL", "discover_trade_detail_url", "download_trade_detail", "aggregate_trade_detail", "trim_daily_top_flow", "load_public_cache", "score_broker_history", "get_broker_features", "enrich_super_broker"]
