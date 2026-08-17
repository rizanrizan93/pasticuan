from __future__ import annotations

"""Public IDX participant-flow enrichment.

Uses the official public EOD Trade Detail Publik report to reconstruct
participant-by-stock flow. This is participant evidence, not beneficial-owner
identity. The module is deliberately fail-soft and cache-first.
"""

from dataclasses import dataclass
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
    return {
        "User-Agent": "Mozilla/5.0 (compatible; IDX-Public-Broker-Collector/1.0)",
        "Accept": "text/csv,text/plain,*/*",
    }


def discover_trade_detail_url(trade_date: date, timeout: int = 20) -> str:
    filename = f"Trade-Detail-Publik_{trade_date:%Y%m%d}.csv"
    try:
        response = requests.get(PUBLIC_INDEX_URL, headers=_headers(), timeout=timeout)
        response.raise_for_status()
        matches = re.findall(r"href=[\"']([^\"']*" + re.escape(filename) + r")[\"']", response.text, flags=re.I)
        for href in matches:
            candidate = urljoin(response.url, href)
            if filename.lower() in candidate.lower():
                return candidate
    except Exception:
        pass

    direct_candidates = [
        f"https://www.idxdata3.co.id/IDX%20Reporting%20PSPP/Revitalisasi/PUBLIK/{filename}",
        f"https://idxdata3.co.id/IDX%20Reporting%20PSPP/Revitalisasi/PUBLIK/{filename}",
        f"https://www.idxdata3.co.id/Market_Summary/Market_Summary/{filename}",
    ]
    for candidate in direct_candidates:
        try:
            probe = requests.get(candidate, headers=_headers(), timeout=timeout, stream=True)
            content_type = str(probe.headers.get("content-type", "")).lower()
            if probe.ok and ("text" in content_type or "csv" in content_type or "octet" in content_type):
                probe.close()
                return candidate
            probe.close()
        except Exception:
            continue
    raise RuntimeError(f"IDX_TRADE_DETAIL_URL_NOT_FOUND:{filename}")


def download_trade_detail(trade_date: date, timeout: int = 45) -> tuple[str, str]:
    url = discover_trade_detail_url(trade_date, timeout=timeout)
    suffix = ".csv"
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    path = handle.name
    handle.close()
    response = requests.get(url, headers=_headers(), timeout=timeout, stream=True)
    response.raise_for_status()
    with open(path, "wb") as output:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                output.write(chunk)
    return path, url


def _read_trade_chunks(path: str, universe: set[str] | None = None, chunksize: int = 250_000):
    usecols = ["asset", "participant_buy", "participant_sell", "volume", "value", "tradingdate"]
    for sep in ("|", ","):
        try:
            iterator = pd.read_csv(
                path,
                sep=sep,
                usecols=lambda c: str(c).strip().lower() in usecols,
                chunksize=chunksize,
                low_memory=False,
                dtype=str,
            )
            for chunk in iterator:
                chunk.columns = [str(c).strip().lower() for c in chunk.columns]
                renames = {
                    "seccode": "asset",
                    "code": "asset",
                    "ticker": "asset",
                    "brokersellid": "participant_sell",
                    "brokerbuyid": "participant_buy",
                    "sellbrokerid": "participant_sell",
                    "buybrokerid": "participant_buy",
                    "quantity": "volume",
                    "tradedate": "tradingdate",
                }
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
                yield chunk
            return
        except Exception:
            continue
    raise RuntimeError("IDX_TRADE_DETAIL_PARSE_FAILED")


def aggregate_trade_detail(path: str, trade_date: date, universe: Iterable[Any] | None = None) -> pd.DataFrame:
    names = {_canon(value) for value in (universe or []) if _canon(value)}
    buy_parts: list[pd.DataFrame] = []
    sell_parts: list[pd.DataFrame] = []
    for chunk in _read_trade_chunks(path, names or None):
        buy = (
            chunk[chunk["participant_buy"].ne("")]
            .groupby(["asset", "participant_buy"], dropna=False, as_index=False)
            .agg(buy_value=("value", "sum"), buy_volume=("volume", "sum"))
            .rename(columns={"participant_buy": "broker_code"})
        )
        sell = (
            chunk[chunk["participant_sell"].ne("")]
            .groupby(["asset", "participant_sell"], dropna=False, as_index=False)
            .agg(sell_value=("value", "sum"), sell_volume=("volume", "sum"))
            .rename(columns={"participant_sell": "broker_code"})
        )
        buy_parts.append(buy)
        sell_parts.append(sell)

    if not buy_parts and not sell_parts:
        return pd.DataFrame()
    buy = pd.concat(buy_parts, ignore_index=True).groupby(["asset", "broker_code"], as_index=False).sum()
    sell = pd.concat(sell_parts, ignore_index=True).groupby(["asset", "broker_code"], as_index=False).sum()
    out = buy.merge(sell, on=["asset", "broker_code"], how="outer").fillna(0.0)
    out = out.rename(columns={"asset": "ticker"})
    out["trade_date"] = pd.Timestamp(trade_date)
    out["buy_avg"] = out["buy_value"].div(out["buy_volume"].replace(0.0, np.nan))
    out["sell_avg"] = out["sell_value"].div(out["sell_volume"].replace(0.0, np.nan))
    out["net_value"] = out["buy_value"] - out["sell_value"]
    out["net_volume"] = out["buy_volume"] - out["sell_volume"]
    out["gross_value"] = out["buy_value"] + out["sell_value"]
    out["source"] = SOURCE_NAME
    out["source_verified"] = True
    out["provenance_state"] = "OFFICIAL_IDX_PUBLIC_EOD_TRADE_DETAIL_PARTICIPANT_FLOW_NOT_BENEFICIAL_OWNER"
    return out[[
        "trade_date", "ticker", "broker_code", "buy_value", "sell_value", "buy_volume", "sell_volume",
        "buy_avg", "sell_avg", "net_value", "net_volume", "gross_value", "source", "source_verified", "provenance_state",
    ]]


def trim_daily_top_flow(daily: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if not isinstance(daily, pd.DataFrame) or daily.empty:
        return pd.DataFrame()
    frame = daily.copy()
    frame["abs_net"] = pd.to_numeric(frame["net_value"], errors="coerce").abs()
    positive = frame[frame["net_value"] > 0].copy()
    negative = frame[frame["net_value"] < 0].copy()
    positive["side"] = "TOP_NET_BUYER"
    negative["side"] = "TOP_NET_SELLER"
    positive["net_rank"] = positive.groupby(["trade_date", "ticker"])["net_value"].rank(method="first", ascending=False)
    negative["net_rank"] = negative.groupby(["trade_date", "ticker"])["net_value"].rank(method="first", ascending=True)
    out = pd.concat([
        positive[positive["net_rank"] <= top_n],
        negative[negative["net_rank"] <= top_n],
    ], ignore_index=True)
    if out.empty:
        return out
    out["broker_flow_version"] = VERSION
    return out.drop(columns=["abs_net"], errors="ignore").sort_values(["trade_date", "ticker", "side", "net_rank"]).reset_index(drop=True)


def load_public_cache() -> pd.DataFrame:
    try:
        response = requests.get(PUBLIC_CACHE_URL, headers=_headers(), timeout=20)
        response.raise_for_status()
        raw = gzip.decompress(response.content)
        frame = pd.read_csv(BytesIO(raw))
        return _normalize_cache(frame)
    except Exception:
        return pd.DataFrame()


def _normalize_cache(frame: pd.DataFrame | None) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    out["ticker"] = out["ticker"].map(_canon)
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce").dt.normalize()
    for column in ("buy_value", "sell_value", "buy_volume", "sell_volume", "buy_avg", "sell_avg", "net_value", "net_volume", "gross_value", "net_rank"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.dropna(subset=["ticker", "trade_date"]).drop_duplicates(
        ["trade_date", "ticker", "broker_code", "side"], keep="last"
    ).reset_index(drop=True)


def _cross_section(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    valid = values.notna()
    score = pd.Series(np.nan, index=values.index, dtype=float)
    if int(valid.sum()) >= 3:
        score.loc[valid] = values.loc[valid].rank(pct=True) * 100.0
    elif int(valid.sum()):
        score.loc[valid] = 50.0
    return score


def score_broker_history(history: pd.DataFrame, universe: Iterable[Any]) -> pd.DataFrame:
    frame = _normalize_cache(history)
    names = sorted({_canon(value) for value in universe if _canon(value)})
    rows: list[dict[str, Any]] = []
    if not frame.empty:
        frame = frame[frame["ticker"].isin(names)]
    for ticker in names:
        local = frame[frame["ticker"].eq(ticker)].copy()
        if local.empty:
            rows.append({"ticker": ticker})
            continue
        dates = sorted(local["trade_date"].dropna().unique())
        recent = local[local["trade_date"].isin(dates[-20:])].copy()
        buyers = recent[recent["side"].eq("TOP_NET_BUYER")].copy()
        sellers = recent[recent["side"].eq("TOP_NET_SELLER")].copy()
        buyer_daily_top = buyers.sort_values(["trade_date", "net_rank"]).groupby("trade_date").head(3)
        top_broker_counts = buyer_daily_top["broker_code"].value_counts() if not buyer_daily_top.empty else pd.Series(dtype=float)
        persistent_broker = str(top_broker_counts.index[0]) if not top_broker_counts.empty else ""
        persistent_days = float(top_broker_counts.iloc[0]) if not top_broker_counts.empty else 0.0
        broker_net = buyers.groupby("broker_code")["net_value"].sum().sort_values(ascending=False) if not buyers.empty else pd.Series(dtype=float)
        top_net = float(broker_net.iloc[0]) if len(broker_net) else np.nan
        total_positive = float(pd.to_numeric(buyers["net_value"], errors="coerce").clip(lower=0.0).sum()) if not buyers.empty else np.nan
        concentration = top_net / total_positive if np.isfinite(top_net) and total_positive > 0 else np.nan
        positive_total = float(pd.to_numeric(buyers["net_value"], errors="coerce").sum()) if not buyers.empty else 0.0
        negative_total = float(pd.to_numeric(sellers["net_value"], errors="coerce").abs().sum()) if not sellers.empty else 0.0
        dominance = positive_total / negative_total if negative_total > 0 else (5.0 if positive_total > 0 else np.nan)
        latest_date = max(dates) if dates else pd.NaT
        latest = local[local["trade_date"].eq(latest_date)] if pd.notna(latest_date) else pd.DataFrame()
        latest_buyers = latest[latest["side"].eq("TOP_NET_BUYER")].sort_values("net_rank")
        latest_broker = str(latest_buyers.iloc[0]["broker_code"]) if not latest_buyers.empty else ""
        latest_avg = _num(latest_buyers.iloc[0].get("buy_avg")) if not latest_buyers.empty else np.nan
        rows.append({
            "ticker": ticker,
            "broker_flow_observed_days": int(recent["trade_date"].nunique()),
            "broker_flow_latest_date": latest_date,
            "broker_top_buyer_code": persistent_broker,
            "broker_latest_top_buyer_code": latest_broker,
            "broker_top3_buyer_persistence_20d_pct": 100.0 * persistent_days / max(1, len(dates[-20:])),
            "broker_top_buyer_net_value_20d": top_net,
            "broker_buyer_concentration_pct": 100.0 * concentration if np.isfinite(concentration) else np.nan,
            "broker_buy_sell_dominance_ratio": dominance,
            "broker_latest_top_buyer_buy_avg": latest_avg,
            "broker_flow_coverage_pct": min(100.0, 100.0 * int(recent["trade_date"].nunique()) / 20.0),
            "broker_flow_source": SOURCE_NAME,
            "broker_flow_provenance": "OFFICIAL_IDX_PUBLIC_EOD_TRADE_DETAIL_PARTICIPANT_FLOW_NOT_BENEFICIAL_OWNER",
        })
    out = pd.DataFrame(rows)
    out["broker_net_score"] = _cross_section(out["broker_top_buyer_net_value_20d"])
    persistence = pd.to_numeric(out["broker_top3_buyer_persistence_20d_pct"], errors="coerce").clip(0, 100)
    concentration = pd.to_numeric(out["broker_buyer_concentration_pct"], errors="coerce").clip(0, 100)
    dominance = pd.to_numeric(out["broker_buy_sell_dominance_ratio"], errors="coerce")
    dominance_score = (50.0 + 15.0 * dominance.fillna(1.0)).clip(0, 100)
    out["broker_accumulation_score"] = (
        0.35 * out["broker_net_score"].fillna(50.0)
        + 0.25 * persistence.fillna(0.0)
        + 0.20 * concentration.fillna(0.0)
        + 0.20 * dominance_score
    ).clip(0, 100).round(1)
    out["broker_smart_money_confirmation_score"] = (
        0.70 * out["broker_accumulation_score"] + 0.30 * out["broker_net_score"].fillna(50.0)
    ).clip(0, 100).round(1)
    out["broker_accumulation_state"] = np.select(
        [out["broker_accumulation_score"].ge(70), out["broker_accumulation_score"].le(35)],
        ["PARTICIPANT_ACCUMULATION", "PARTICIPANT_DISTRIBUTION"],
        default="PARTICIPANT_MIXED",
    )
    out["broker_flow_version"] = VERSION
    return out


def get_broker_features(tickers: Iterable[Any], *, live_fallback: bool | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    names = sorted({_canon(value) for value in tickers if _canon(value)})
    history = load_public_cache()
    meta = {
        "source": "GITHUB_ROLLING_PUBLIC_IDX_BROKER_CACHE" if not history.empty else "NONE",
        "state": "OK" if not history.empty else "NO_CACHE",
        "observed_dates": int(history["trade_date"].nunique()) if not history.empty else 0,
    }
    if live_fallback is None:
        live_fallback = str(os.getenv("IDX_BROKER_LIVE_FALLBACK", "1")).strip().lower() in {"1", "true", "yes", "on"}
    latest = pd.to_datetime(history.get("trade_date"), errors="coerce").max() if not history.empty else pd.NaT
    target = pd.Timestamp(_latest_weekday())
    if live_fallback and (pd.isna(latest) or latest < target) and names:
        path = None
        try:
            path, source_url = download_trade_detail(target.date())
            daily = aggregate_trade_detail(path, target.date(), names)
            daily = trim_daily_top_flow(daily)
            if not daily.empty:
                history = pd.concat([history, daily], ignore_index=True) if not history.empty else daily
                history = _normalize_cache(history)
                meta.update({"source": SOURCE_NAME, "state": "OK_LIVE_REFRESH", "latest_source_url": source_url, "observed_dates": int(history["trade_date"].nunique())})
        except Exception as exc:
            meta.update({"live_error": f"{type(exc).__name__}: {str(exc)[:180]}"})
        finally:
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass
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
    out = frame.copy()
    right = features.copy()
    out["_key"] = out["ticker"].map(_canon)
    right["_key"] = right["ticker"].map(_canon)
    right = right.drop(columns=["ticker"]).drop_duplicates("_key")
    duplicate = [c for c in right.columns if c != "_key" and c in out.columns]
    if duplicate:
        out = out.drop(columns=duplicate)
    out = out.merge(right, on="_key", how="left").drop(columns=["_key"])
    base = pd.to_numeric(out.get("flow_silent_accumulation_score", pd.Series(np.nan, index=out.index)), errors="coerce")
    broker = pd.to_numeric(out.get("broker_accumulation_score"), errors="coerce")
    coverage = pd.to_numeric(out.get("broker_flow_coverage_pct"), errors="coerce").clip(0, 100)
    weight = 0.20 * coverage.fillna(0.0) / 100.0
    out["broker_confirmation_weight_pct"] = (100.0 * weight).round(1)
    out["broker_pre_confirmation_accumulation_score"] = base
    out["broker_post_confirmation_accumulation_score"] = ((1.0 - weight) * base + weight * broker).where(base.notna() & broker.notna(), base).clip(0, 100).round(1)
    out["flow_silent_accumulation_score"] = out["broker_post_confirmation_accumulation_score"]
    out["broker_accumulation_delta"] = (out["broker_post_confirmation_accumulation_score"] - base).round(1)
    return out


def enrich_emir_broker(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "ticker" not in frame.columns:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    features, _ = get_broker_features(frame["ticker"].tolist())
    if features.empty:
        return frame.copy()
    out = frame.copy()
    right = features.copy()
    out["_key"] = out["ticker"].map(_canon)
    right["_key"] = right["ticker"].map(_canon)
    right = right.drop(columns=["ticker"]).drop_duplicates("_key")
    duplicate = [c for c in right.columns if c != "_key" and c in out.columns]
    if duplicate:
        out = out.drop(columns=duplicate)
    out = out.merge(right, on="_key", how="left").drop(columns=["_key"])
    base = pd.to_numeric(out.get("smart_money_score", pd.Series(np.nan, index=out.index)), errors="coerce")
    broker = pd.to_numeric(out.get("broker_smart_money_confirmation_score"), errors="coerce")
    coverage = pd.to_numeric(out.get("broker_flow_coverage_pct"), errors="coerce").clip(0, 100)
    weight = 0.20 * coverage.fillna(0.0) / 100.0
    blended = ((1.0 - weight) * base + weight * broker).where(base.notna() & broker.notna(), base).clip(0, 100)
    out["broker_confirmation_weight_pct"] = (100.0 * weight).round(1)
    out["broker_pre_confirmation_smart_money_score"] = base
    out["broker_post_confirmation_smart_money_score"] = blended.round(1)
    out["smart_money_score"] = blended.round(1)
    pre_conviction = pd.to_numeric(out.get("emir_conviction_score", pd.Series(np.nan, index=out.index)), errors="coerce")
    directional = ((broker.fillna(50.0) - 50.0) / 50.0).clip(-1.0, 1.0)
    delta = (1.5 * directional * coverage.fillna(0.0) / 100.0).clip(-1.5, 1.5)
    out["broker_emir_conviction_delta"] = delta.round(3)
    if "emir_conviction_score" in out.columns:
        out["emir_conviction_score"] = (pre_conviction + delta).clip(0, 100).round(3)
    if "emir_final_score" in out.columns:
        out["emir_final_score"] = (pd.to_numeric(out["emir_final_score"], errors="coerce") + delta).clip(0, 100).round(3)
    out["broker_flow_identity_policy"] = "PARTICIPANT_FLOW_NOT_BENEFICIAL_OWNER_IDENTITY"
    return out


__all__ = [
    "VERSION", "PUBLIC_CACHE_URL", "discover_trade_detail_url", "download_trade_detail",
    "aggregate_trade_detail", "trim_daily_top_flow", "load_public_cache", "score_broker_history",
    "get_broker_features", "enrich_super_broker", "enrich_emir_broker",
]
