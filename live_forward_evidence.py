from __future__ import annotations

"""Lightweight forward-evidence acquisition used during production scans.

This module deliberately separates *collection coverage* from *scoring coverage*.
A successful source check with no material forward event is recorded as a valid
check, but never receives a positive Future Fundamental score. Google News RSS
is research evidence only; strict production evidence still requires the
existing HTTPS/entity/quorum governed-evidence path.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
from html import unescape
from typing import Any, Iterable
from urllib.parse import quote_plus
import re
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests

LIVE_FORWARD_EVIDENCE_VERSION = "1.0.0"

_FORWARD_RULES: tuple[tuple[str, tuple[str, ...], float, float], ...] = (
    ("PROJECT_OR_CONTRACT", ("KONTRAK", "CONTRACT", "BACKLOG", "ORDER BOOK", "ORDERBOOK", "OFFTAKE", "TENDER"), 68.0, 70.0),
    ("CAPACITY_OR_EXPANSION", ("EKSPANSI", "EXPANSION", "KAPASITAS", "CAPACITY", "PABRIK", "PLANT", "SMELTER", "COMMISSIONING", "COMMERCIAL OPERATION"), 66.0, 68.0),
    ("GUIDANCE_OR_TARGET", ("GUIDANCE", "TARGET PENDAPATAN", "TARGET REVENUE", "TARGET LABA", "TARGET PROFIT", "PRODUCTION TARGET", "TARGET PRODUKSI"), 61.0, 64.0),
    ("PRODUCT_OR_NEW_MARKET", ("PRODUK BARU", "NEW PRODUCT", "PASAR BARU", "NEW MARKET", "PELUNCURAN", "LAUNCH"), 58.0, 61.0),
    ("STRATEGIC_JV_OR_MA", ("JOINT VENTURE", " JV ", "AKUISISI", "ACQUISITION", "MERGER", "INVESTOR STRATEGIS", "STRATEGIC INVESTOR"), 64.0, 66.0),
    ("CAPEX", ("CAPEX", "BELANJA MODAL", "CAPITAL EXPENDITURE"), 58.0, 62.0),
)
_ADVERSE_RULES: tuple[tuple[str, tuple[str, ...], float, float], ...] = (
    ("PROJECT_DELAY_OR_CANCEL", ("KONTRAK DIBATALKAN", "CONTRACT CANCELLED", "CONTRACT TERMINATED", "PROYEK DITUNDA", "PROJECT DELAY", "COD MUNDUR"), 32.0, 28.0),
    ("GUIDANCE_CUT", ("GUIDANCE CUT", "GUIDANCE DITURUNKAN", "TARGET DITURUNKAN"), 30.0, 26.0),
)


def _ticker(value: Any) -> str:
    text = str(value or "").upper().strip()
    return text if text.endswith(".JK") else f"{text}.JK" if text else ""


def _clean_title(value: Any) -> str:
    text = unescape(re.sub(r"<[^>]+>", " ", str(value or "")))
    return re.sub(r"\s+", " ", text).strip()


def _classify(text: str) -> tuple[str, float, float] | None:
    upper = f" {str(text or '').upper()} "
    for category, tokens, pipeline, impact in _ADVERSE_RULES:
        if any(token in upper for token in tokens):
            return category, pipeline, impact
    for category, tokens, pipeline, impact in _FORWARD_RULES:
        if any(token in upper for token in tokens):
            return category, pipeline, impact
    return None


def _published(value: Any) -> pd.Timestamp | pd.NaT:
    try:
        stamp = pd.Timestamp(parsedate_to_datetime(str(value)))
    except Exception:
        stamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(stamp):
        return pd.NaT
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _one(ticker: str, *, lookback_days: int, timeout: float) -> list[dict[str, Any]]:
    symbol = _ticker(ticker)
    bare = symbol.removesuffix(".JK")
    checked = pd.Timestamp.now(tz="UTC")
    query = quote_plus(f'"{bare}" IDX saham')
    url = f"https://news.google.com/rss/search?q={query}&hl=id&gl=ID&ceid=ID:id"
    base = {
        "ticker": symbol,
        "review_origin": "LIVE_GOOGLE_NEWS_FORWARD_RESEARCH",
        "project_source_families": "GOOGLE_NEWS_RSS",
        "project_source_quorum_verified": False,
        "source_quorum_count": 0,
        "entity_match_verified": True,
        "source_checked_at": checked.isoformat(),
        "forward_collection_provider": "GOOGLE_NEWS_RSS",
        "forward_collection_version": LIVE_FORWARD_EVIDENCE_VERSION,
        "forward_research_only": True,
    }
    try:
        response = requests.get(url, timeout=max(1.0, float(timeout)), headers={"User-Agent": "Mozilla/5.0 IDXScanner/1.0"})
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception as exc:
        return [{
            **base,
            "project_name": "FORWARD_EVIDENCE_CHECK",
            "project_stage": "COLLECTION_FAILED",
            "project_data_coverage": 0.0,
            "forward_collection_coverage_pct": 0.0,
            "forward_collection_state": "FORWARD_CHECK_FAILED_RETRYABLE",
            "project_execution_flags": f"{type(exc).__name__}: {str(exc)[:180]}",
        }]

    cutoff = checked - pd.Timedelta(days=max(7, int(lookback_days)))
    items: list[dict[str, Any]] = []
    publishers: set[str] = set()
    scanned = 0
    for item in root.findall(".//item")[:20]:
        title = _clean_title(item.findtext("title"))
        link = str(item.findtext("link") or "").strip()
        source_node = item.find("source")
        publisher = _clean_title(source_node.text if source_node is not None else "")
        published = _published(item.findtext("pubDate"))
        if pd.notna(published) and published < cutoff:
            continue
        scanned += 1
        classified = _classify(title)
        if classified is None:
            continue
        category, pipeline, impact = classified
        if publisher:
            publishers.add(publisher.upper())
        items.append({
            **base,
            "project_name": title[:500] or f"{bare} forward event",
            "project_names": title[:500],
            "project_stage": category,
            "project_pipeline_score_observed": pipeline,
            "future_fundamental_impact_score_observed": impact,
            "project_data_coverage": 48.0,
            "project_source_urls": link,
            "last_verified_at": published.isoformat() if pd.notna(published) else checked.isoformat(),
            "event_date": published.date().isoformat() if pd.notna(published) else checked.date().isoformat(),
            "project_execution_flags": "RESEARCH_DISCOVERY_NOT_STRICT_OFFICIAL_EVIDENCE",
            "forward_collection_coverage_pct": 100.0,
            "forward_collection_state": "MATERIAL_FORWARD_RESEARCH_EVIDENCE_FOUND",
            "forward_research_category": category,
            "forward_research_publisher": publisher,
        })
    if items:
        # Multiple independent publishers improve research confidence, but do not
        # become production quorum. Production quorum remains governed separately.
        publisher_count = max(1, len(publishers))
        coverage = min(72.0, 48.0 + 8.0 * min(3, publisher_count - 1))
        for row in items:
            row["project_data_coverage"] = coverage
            row["source_quorum_count"] = publisher_count
        return items

    return [{
        **base,
        "project_name": "FORWARD_EVIDENCE_CHECK",
        "project_stage": "NO_MATERIAL_FORWARD_EVENT_FOUND",
        "project_data_coverage": 0.0,
        "last_verified_at": checked.isoformat(),
        "event_date": checked.date().isoformat(),
        "project_execution_flags": f"RSS_CHECK_OK_ITEMS_SCANNED={scanned}",
        "forward_collection_coverage_pct": 100.0,
        "forward_collection_state": "FORWARD_CHECK_COMPLETED_NO_MATERIAL_EVENT",
    }]


def collect_live_forward_evidence(
    tickers: Iterable[Any],
    *,
    lookback_days: int = 120,
    max_workers: int = 12,
    timeout: float = 4.0,
) -> pd.DataFrame:
    names = list(dict.fromkeys(_ticker(value) for value in tickers if _ticker(value)))
    if not names:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    workers = max(1, min(int(max_workers), 16, len(names)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, ticker, lookback_days=lookback_days, timeout=timeout): ticker for ticker in names}
        for future in as_completed(futures):
            try:
                rows.extend(future.result())
            except Exception as exc:
                ticker = futures[future]
                rows.append({
                    "ticker": ticker,
                    "project_name": "FORWARD_EVIDENCE_CHECK",
                    "project_stage": "COLLECTION_FAILED",
                    "project_data_coverage": 0.0,
                    "forward_collection_coverage_pct": 0.0,
                    "forward_collection_state": "FORWARD_CHECK_FAILED_RETRYABLE",
                    "project_execution_flags": f"{type(exc).__name__}: {str(exc)[:180]}",
                    "review_origin": "LIVE_GOOGLE_NEWS_FORWARD_RESEARCH",
                    "project_source_families": "GOOGLE_NEWS_RSS",
                    "project_source_quorum_verified": False,
                    "forward_research_only": True,
                })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["ticker"] = frame["ticker"].map(_ticker)
    return frame


def collection_coverage(frame: pd.DataFrame | None, tickers: Iterable[Any] | None = None) -> dict[str, float]:
    names = list(dict.fromkeys(_ticker(value) for value in (tickers or []) if _ticker(value)))
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {"tickers": float(len(names)), "checked": 0.0, "coverage_pct": 0.0}
    local = frame.copy()
    local["ticker"] = local.get("ticker", pd.Series(dtype=str)).map(_ticker)
    if names:
        local = local.loc[local["ticker"].isin(names)]
    coverage = pd.to_numeric(local.get("forward_collection_coverage_pct", 0.0), errors="coerce").fillna(0.0)
    checked = local.loc[coverage.gt(0), "ticker"].nunique()
    total = len(names) if names else local["ticker"].nunique()
    return {
        "tickers": float(total),
        "checked": float(checked),
        "coverage_pct": round(100.0 * checked / max(total, 1), 1),
    }


__all__ = ["LIVE_FORWARD_EVIDENCE_VERSION", "collect_live_forward_evidence", "collection_coverage"]
