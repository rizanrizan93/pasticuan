import concurrent.futures as cf
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema, hilbert, periodogram

# =========================================================
# IDX / IHSG DUAL TAB SCANNER - FINAL VERSION
# Tab 1: Market Structure Top 20 + reversal signals
# Tab 2: Institutional Forward Score with sub-tabs + entry plan / benchmark / time analysis
# =========================================================

st.set_page_config(page_title="IDX Dual Tab Scanner", layout="wide")
st.title("📊 IDX Dual Tab Scanner")
st.caption(
    "Global watchlist untuk ranking cepat, lalu deep dive untuk bedah detail per ticker dengan institutional forward score, entry plan, dan time analysis."
)
st.markdown("---")

# =========================================================
# Sidebar
# =========================================================
st.sidebar.header("🎯 Universe Source")
universe_mode = st.sidebar.radio(
    "Pilih sumber universe",
    ["Paste tickers", "Upload CSV", "Local file midcap_universe.csv"],
    index=0,
)

paste_text = ""
uploaded_file = None
if universe_mode == "Paste tickers":
    paste_text = st.sidebar.text_area(
        "Paste tickers (satu per baris / dipisah koma)",
        value="BMRI\nBBCA\nTLKM\nASII",
        height=140,
    )
elif universe_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV universe", type=["csv"])
else:
    st.sidebar.info("Mode ini akan membaca file `midcap_universe.csv` dari folder aplikasi.")

st.sidebar.markdown("---")
st.sidebar.header("🧭 Scan Settings")
months = st.sidebar.slider("Periode data historis (bulan)", 12, 60, 24)
min_price = st.sidebar.number_input("Min harga (Rp)", value=200.0, step=10.0)
max_price = st.sidebar.number_input("Max harga (Rp)", value=25000.0, step=500.0)
min_avg_volume = st.sidebar.number_input("Min rata-rata volume 20D", value=150000, step=50000)
min_history_bars = st.sidebar.slider("Min candle valid", 60, 240, 100)

st.sidebar.markdown("---")
st.sidebar.header("🚀 Execution")
max_workers = st.sidebar.slider("Max parallel workers", 2, 12, 6)
ranking_sort_mode = st.sidebar.selectbox("Ranking order", ["Descending", "Ascending"], index=0)
run_global_scan = st.sidebar.button("Run global scan", type="primary")

GLOBAL_MODE = "Conservative"  # Profit-only mode: prioritize precision over signal count

# =========================================================
# Utilities
# =========================================================

from data_engine import *
from fundamental_analyst import *
from technical_analyst import *
from technical_analyst import _safe_float as _safe_float
from catalyst_nlp import *


def _safe_text(value) -> str:
    try:
        text = str(value or "").strip()
        return text
    except Exception:
        return ""

DECISION_CACHE_PATH = Path("scanner_decision_cache.json")


def _load_decision_cache() -> dict:
    try:
        if DECISION_CACHE_PATH.exists():
            data = json.loads(DECISION_CACHE_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _save_decision_cache(cache: dict) -> None:
    try:
        DECISION_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _apply_decision_hysteresis(symbol: str, result: dict, cache: dict) -> dict:
    """Keep buy-side decisions stable unless deterioration is broad and persistent."""
    sym = str(symbol or "").upper().strip()
    if not sym or not isinstance(result, dict):
        return result

    prev = cache.get(sym, {}) if isinstance(cache, dict) else {}
    prev_decision = str(prev.get("decision", "")).upper().strip()
    curr_decision = str(result.get("decision", "")).upper().strip()

    prev_rank = _decision_rank(prev_decision)
    curr_rank = _decision_rank(curr_decision)

    curr_score = float(result.get("score", np.nan)) if pd.notna(result.get("score", np.nan)) else np.nan
    prev_score = float(prev.get("score", np.nan)) if pd.notna(prev.get("score", np.nan)) else np.nan
    curr_ifs = float(result.get("ifs_score", np.nan)) if pd.notna(result.get("ifs_score", np.nan)) else np.nan
    prev_ifs = float(prev.get("ifs_score", np.nan)) if pd.notna(prev.get("ifs_score", np.nan)) else np.nan

    curr_market_struct = float(result.get("market_structure_score", np.nan)) if pd.notna(result.get("market_structure_score", np.nan)) else np.nan
    prev_market_struct = float(prev.get("market_structure_score", np.nan)) if pd.notna(prev.get("market_structure_score", np.nan)) else np.nan
    curr_smart_money = float(result.get("smart_money_score", np.nan)) if pd.notna(result.get("smart_money_score", np.nan)) else np.nan
    prev_smart_money = float(prev.get("smart_money_score", np.nan)) if pd.notna(prev.get("smart_money_score", np.nan)) else np.nan

    buyish_prev = prev_rank >= _decision_rank("BUY")
    downgrade = curr_rank < prev_rank

    score_drop = (prev_score - curr_score) if (np.isfinite(prev_score) and np.isfinite(curr_score)) else np.nan
    ifs_drop = (prev_ifs - curr_ifs) if (np.isfinite(prev_ifs) and np.isfinite(curr_ifs)) else np.nan
    struct_drop = (prev_market_struct - curr_market_struct) if (np.isfinite(prev_market_struct) and np.isfinite(curr_market_struct)) else np.nan
    smart_drop = (prev_smart_money - curr_smart_money) if (np.isfinite(prev_smart_money) and np.isfinite(curr_smart_money)) else np.nan

    strong_break = bool(
        curr_rank == 0
        and (
            (np.isfinite(curr_market_struct) and curr_market_struct < 46.0)
            or (np.isfinite(curr_ifs) and curr_ifs < 58.0)
            or (np.isfinite(curr_score) and curr_score < 55.0)
            or (np.isfinite(curr_smart_money) and curr_smart_money < 40.0)
        )
    )

    clear_deterioration = bool(
        (np.isfinite(score_drop) and score_drop >= 8.0)
        and (
            (np.isfinite(ifs_drop) and ifs_drop >= 6.0)
            or (np.isfinite(struct_drop) and struct_drop >= 6.0)
            or (np.isfinite(smart_drop) and smart_drop >= 6.0)
        )
    )

    weak_streak = int(prev.get("downgrade_streak", 0) or 0)

    if buyish_prev and downgrade:
        if not strong_break and not (clear_deterioration and weak_streak >= 1):
            weak_streak += 1
            preserved = prev_decision if prev_decision in {"BUY", "STRONG BUY"} else ("BUY" if curr_decision != "AVOID" else "WATCHLIST")
            result["decision"] = preserved
            notes = result.get("notes", [])
            if not isinstance(notes, list):
                notes = [str(notes)]
            notes.append(f"Decision_Hysteresis_Preserved(streak={weak_streak})")
            result["notes"] = notes
        else:
            weak_streak = 0
    else:
        weak_streak = 0

    if result.get("decision") in {"BUY", "STRONG BUY"}:
        for key in ("entry_price", "stop_price", "target_1", "target_2", "risk_reward_1", "risk_reward_2"):
            if pd.isna(result.get(key, np.nan)) and pd.notna(prev.get(key, np.nan)):
                result[key] = prev.get(key)

    result["downgrade_streak"] = int(weak_streak)
    result["decision_raw"] = curr_decision
    result["decision_prev"] = prev_decision
    return result


def _extract_news_value(item: dict, *keys, default=""):
    if not isinstance(item, dict):
        return default
    for key in keys:
        value = item.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, dict):
            for nested_key in ("text", "title", "summary", "description", "name", "url"):
                nested = value.get(nested_key)
                if nested not in (None, ""):
                    return nested
            if "content" in value and value["content"] not in (None, ""):
                return value["content"]
        else:
            return value
    return default


def _extract_news_datetime(item: dict) -> str:
    if not isinstance(item, dict):
        return ""
    for key in ("providerPublishTime", "published_at", "pubDate", "publishedAt", "date", "datetime", "time"):
        value = item.get(key)
        if value in (None, ""):
            continue
        try:
            if isinstance(value, (int, float)) and value > 10_000_000_000:
                return pd.to_datetime(value, unit="ms", utc=True).tz_convert(None).strftime("%Y-%m-%d %H:%M")
            if isinstance(value, (int, float)):
                return pd.to_datetime(value, unit="s", utc=True).tz_convert(None).strftime("%Y-%m-%d %H:%M")
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.notna(parsed):
                if getattr(parsed, "tzinfo", None) is not None:
                    parsed = parsed.tz_convert(None)
                return parsed.strftime("%Y-%m-%d %H:%M")
            return str(value)
        except Exception:
            continue
    return ""


def _parse_manual_news_lines(raw_text: str) -> list[dict]:
    items: list[dict] = []
    for line in str(raw_text or "").splitlines():
        text = line.strip()
        if not text:
            continue
        parts = [p.strip() for p in re.split(r"\s*\|\|\s*|\s*\|\s*", text) if p.strip()]
        title = parts[0] if parts else text
        source = parts[1] if len(parts) > 1 else "Manual"
        summary = parts[2] if len(parts) > 2 else ""
        items.append(
            {
                "title": title,
                "summary": summary,
                "source": source,
                "published_at": "",
                "link": "",
            }
        )
    return items




def _news_item_fingerprint(item: dict) -> str:
    if not isinstance(item, dict):
        return ""
    title = _safe_text(item.get("title") or item.get("headline") or item.get("name") or "").lower()
    source = _safe_text(item.get("source") or item.get("publisher") or item.get("provider") or item.get("site") or "").lower()
    link = _safe_text(item.get("link") or item.get("url") or "").lower()
    summary = _safe_text(item.get("summary") or item.get("description") or item.get("snippet") or "").lower()[:120]
    return "|".join([title, source, link, summary])


def _dedupe_news_items(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        fp = _news_item_fingerprint(item)
        if not fp or fp in seen:
            continue
        seen.add(fp)
        out.append(item)
    return out


def _extract_company_name_candidates(symbol: str) -> list[str]:
    base = str(symbol or "").strip()
    if not base:
        return []
    names: list[str] = []
    try:
        info_fn = globals().get("load_yf_info")
        if callable(info_fn):
            info = info_fn(base) or {}
            for key in ("shortName", "longName", "displayName", "name", "symbol"):
                value = _safe_text(info.get(key, ""))
                if value:
                    names.append(value)
    except Exception:
        pass
    cleaned: list[str] = []
    for value in names:
        value = re.sub(r"\s+", " ", value).strip()
        if value and value.lower() not in {x.lower() for x in cleaned}:
            cleaned.append(value)
    return cleaned


def _build_indonesia_news_queries(symbol: str) -> list[str]:
    base = str(symbol or "").strip()
    if not base:
        return []

    candidate_fn = globals().get("_ticker_candidates")
    candidates = candidate_fn(base) if callable(candidate_fn) else [base]
    company_candidates = _extract_company_name_candidates(base)

    local_domains = (
        "site:kontan.co.id",
        "site:bisnis.com",
        "site:cnbcindonesia.com",
        "site:cnnindonesia.com",
        "site:tempo.co",
        "site:kompas.com",
        "site:detik.com",
        "site:antaranews.com",
        "site:idx.co.id",
        "site:ojk.go.id",
        "site:bi.go.id",
    )

    queries: list[str] = []

    def add(value: str) -> None:
        value = re.sub(r"\s+", " ", str(value or "")).strip()
        if value and value not in queries:
            queries.append(value)

    # Prioritize issuer / company names because Indonesian outlets often avoid
    # pure ticker strings and use the legal name or brand name instead.
    for company in company_candidates:
        comp = re.sub(r"\s+", " ", company).strip()
        if not comp:
            continue
        for q in (
            comp,
            f"{comp} saham",
            f"{comp} emiten",
            f"{comp} berita",
            f"{comp} Indonesia",
            f'"{comp}"',
        ):
            add(q)
        for domain in local_domains[:6]:
            add(f"{comp} {domain}")

    for candidate in candidates:
        cand = str(candidate or "").strip().upper()
        if not cand:
            continue
        base_term = cand.replace(".JK", "")
        if base_term.startswith("^"):
            add(base_term)
            continue

        for q in (
            base_term,
            f"{base_term} saham",
            f"{base_term} emiten",
            f"{base_term} berita",
            f"{base_term} Indonesia",
            f'"{base_term}"',
        ):
            add(q)
        for domain in local_domains[:4]:
            add(f"{base_term} {domain}")

    return queries[:18]


def _fetch_google_news_rss(query: str) -> list[dict]:
    import requests
    from urllib.parse import quote
    import xml.etree.ElementTree as ET

    out: list[dict] = []
    rss_url = f"https://news.google.com/rss/search?q={quote(query)}&hl=id&gl=ID&ceid=ID:id"
    resp = requests.get(rss_url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
    if not resp.ok or not resp.text.strip():
        return out

    try:
        root = ET.fromstring(resp.text)
    except Exception:
        return out

    for item in root.findall(".//item"):
        title = _safe_text(item.findtext("title", default=""))
        if not title:
            continue
        out.append({
            "title": title,
            "summary": _safe_text(item.findtext("description", default="")),
            "source": _safe_text(item.findtext("source", default="Google News")) or "Google News",
            "published_at": _safe_text(item.findtext("pubDate", default="")),
            "link": _safe_text(item.findtext("link", default="")),
        })
    return out


def _fetch_indonesia_news_google_first(symbol: str, limit: int = 12) -> list[dict]:
    queries = _build_indonesia_news_queries(symbol)
    if not queries:
        return []

    all_items: list[dict] = []
    per_query_cap = max(5, min(10, int(limit)))
    for query in queries:
        try:
            items = _fetch_google_news_rss(query)
            if not items:
                continue
            all_items.extend(items[:per_query_cap])
            if len(all_items) >= max(2 * limit, 24):
                break
        except Exception:
            continue

    return _dedupe_news_items(all_items)[: max(1, int(limit))]

@st.cache_data(ttl=900, show_spinner=False)
def fetch_ticker_news_items(symbol: str, limit: int = 12) -> list[dict]:
    base = str(symbol or "").strip()
    if not base:
        return []

    candidate_fn = globals().get("_ticker_candidates")
    candidates = candidate_fn(base) if callable(candidate_fn) else [base]
    seen = set()
    ordered_candidates = []
    for candidate in candidates:
        cand = str(candidate).strip()
        if cand and cand not in seen:
            seen.add(cand)
            ordered_candidates.append(cand)

    fetched: list[dict] = []
    for candidate in ordered_candidates:
        try:
            ticker = yf.Ticker(candidate)
            raw_news = None

            try:
                getter = getattr(ticker, "get_news", None)
                if callable(getter):
                    raw_news = getter()
            except Exception:
                raw_news = None

            if not raw_news:
                try:
                    raw_news = getattr(ticker, "news", None)
                    if callable(raw_news):
                        raw_news = raw_news()
                except Exception:
                    raw_news = None

            if not raw_news:
                continue

            if isinstance(raw_news, dict):
                raw_news = raw_news.get("news") or raw_news.get("items") or []

            for item in raw_news:
                if not isinstance(item, dict):
                    continue
                title = _safe_text(_extract_news_value(item, "title", "headline", default=""))
                if not title:
                    content = item.get("content")
                    if isinstance(content, dict):
                        title = _safe_text(content.get("title") or content.get("headline") or content.get("summary"))
                    elif isinstance(content, str):
                        title = _safe_text(content)
                summary = _safe_text(
                    _extract_news_value(item, "summary", "description", default="")
                )
                if not summary:
                    content = item.get("content")
                    if isinstance(content, dict):
                        summary = _safe_text(content.get("summary") or content.get("description") or content.get("text"))
                source = _safe_text(
                    _extract_news_value(item, "publisher", "provider", "source", default="Yahoo Finance")
                )
                link = _safe_text(_extract_news_value(item, "link", "url", default=""))
                if not link:
                    content = item.get("content")
                    if isinstance(content, dict):
                        link = _safe_text(content.get("canonicalUrl", {}).get("url") if isinstance(content.get("canonicalUrl"), dict) else content.get("url"))
                fetched.append(
                    {
                        "title": title,
                        "summary": summary,
                        "source": source,
                        "published_at": _extract_news_datetime(item),
                        "link": link,
                    }
                )
            if fetched:
                break
        except Exception:
            continue


@st.cache_data(ttl=900, show_spinner=False)
def fetch_ticker_news_search_items(symbol: str, limit: int = 12) -> list[dict]:
    base = str(symbol or "").strip()
    if not base:
        return []

    candidate_fn = globals().get("_ticker_candidates")
    candidates = candidate_fn(base) if callable(candidate_fn) else [base]
    seen = set()
    ordered_candidates = []
    for candidate in candidates:
        cand = str(candidate).strip()
        if cand and cand not in seen:
            seen.add(cand)
            ordered_candidates.append(cand)

    import requests
    from urllib.parse import quote
    import xml.etree.ElementTree as ET

    fetched: list[dict] = []

    # Yahoo Finance search endpoint fallback
    for candidate in ordered_candidates:
        try:
            url = "https://query1.finance.yahoo.com/v1/finance/search"
            params = {"q": candidate, "newsCount": 20, "quotesCount": 5, "enableFuzzyQuery": "true"}
            resp = requests.get(url, params=params, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
            if resp.ok:
                payload = resp.json()
                for item in payload.get("news", []) or []:
                    content = item if isinstance(item, dict) else {}
                    title = _safe_text(content.get("title") or content.get("headline") or "")
                    summary = _safe_text(content.get("summary") or content.get("description") or "")
                    source = _safe_text(content.get("publisher") or content.get("provider") or "Yahoo Finance")
                    link = _safe_text(content.get("link") or content.get("url") or "")
                    published = ""
                    for k in ("providerPublishTime", "pubDate", "published_at", "publishedAt", "date"):
                        v = content.get(k)
                        if v not in (None, ""):
                            published = _extract_news_datetime({k: v})
                            break
                    if title:
                        fetched.append({
                            "title": title,
                            "summary": summary,
                            "source": source,
                            "published_at": published,
                            "link": link,
                        })
                if fetched:
                    break
        except Exception:
            continue

    # Yahoo RSS fallback as a second layer
    if not fetched:
        for candidate in ordered_candidates:
            for q in (candidate, candidate.replace(".JK", "")):
                try:
                    rss_url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote(q)}&region=US&lang=en-US"
                    resp = requests.get(rss_url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
                    if not resp.ok or not resp.text.strip():
                        continue
                    root = ET.fromstring(resp.text)
                    for item in root.findall(".//item"):
                        title = _safe_text(item.findtext("title", default=""))
                        summary = _safe_text(item.findtext("description", default=""))
                        source = _safe_text(item.findtext("source", default="Yahoo Finance"))
                        link = _safe_text(item.findtext("link", default=""))
                        pub = _safe_text(item.findtext("pubDate", default=""))
                        if title:
                            fetched.append({
                                "title": title,
                                "summary": summary,
                                "source": source or "Yahoo Finance",
                                "published_at": pub,
                                "link": link,
                            })
                    if fetched:
                        break
                except Exception:
                    continue
            if fetched:
                break

    return fetched[: max(1, int(limit))]


@st.cache_data(ttl=900, show_spinner=False)
def fetch_indonesia_news_items(symbol: str, limit: int = 12) -> list[dict]:
    """Fetch Indonesia-focused news using Google News RSS first.

    The collector now widens coverage beyond ticker-only queries by adding:
    - company / issuer name candidates from Yahoo metadata
    - Indonesian market aliases (saham, emiten, berita)
    - curated Indonesia-local source filters (Kontan, Bisnis, CNBC Indonesia,
      Antara, Kompas, Detik, Tempo, IDX, OJK, BI)

    Yahoo Finance remains a fallback only when Google News cannot surface any
    Indonesia-focused coverage.
    """
    base = str(symbol or "").strip()
    if not base:
        return []

    fetched = _fetch_indonesia_news_google_first(base, limit=limit)
    if not fetched:
        fetched = fetch_ticker_news_search_items(symbol, limit)

    return _dedupe_news_items(fetched)[: max(1, int(limit))]


def _news_decision_label(decision: str) -> str:
    d = str(decision or "").strip().upper()
    if d == "PASS":
        return "PASSED"
    if d == "REJECT":
        return "REJECTED"
    return "WATCH"


def _decision_rank(decision: str) -> int:
    mapping = {
        "STRONG BUY": 3,
        "BUY": 2,
        "WATCHLIST": 1,
        "AVOID": 0,
    }
    return mapping.get(str(decision or "").strip().upper(), 0)


def _sniper_rank(r: dict) -> int:
    if not isinstance(r, dict):
        return 0
    if str(r.get("unicorn_setup_status", "")).upper() == "ENTRY":
        return 2
    if str(r.get("unicorn_sniper_status", "")).upper() == "ENTRY":
        return 1
    if r.get("unicorn_sniper_valid", False):
        return 1
    if r.get("unicorn_setup_valid", False):
        return 1
    return 0


def _build_watch_df(watch_rows: list[dict], ascending: bool = False) -> pd.DataFrame:
    if not watch_rows:
        return pd.DataFrame()
    watch_df = pd.DataFrame(watch_rows)
    # Prioritize actual quality metrics first; decision labels come after.
    sort_cols = [
        "Score",
        "IFS",
        "MarketStruct",
        "SmartMoney",
        "CycleRel",
        "DecisionRank",
        "SniperRank",
    ]
    present_cols = [c for c in sort_cols if c in watch_df.columns]
    if present_cols:
        watch_df = watch_df.sort_values(
            present_cols,
            ascending=[ascending] * len(present_cols),
            na_position="last",
        )
    return watch_df.reset_index(drop=True)


def _display_watch_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df.drop(columns=["DecisionRank", "SniperRank"], errors="ignore")



def _recalibrate_global_scan_results(valid_results: list[dict]) -> list[dict]:
    """Recalibrate score / decision after the universe scan completes.

    The technical score remains the base quality view, but the final scan output
    blends it with IFS and universe-relative percentile so the ranking is stable
    across weak, sideways, and strong markets.
    """
    if not valid_results:
        return valid_results

    try:
        decision_cache = _load_decision_cache()

        base_df = pd.DataFrame(
            {
                "Score": [float(_safe_float(r.get("score"), np.nan)) for r in valid_results],
                "IFS": [float(_safe_float(r.get("ifs_score"), np.nan)) for r in valid_results],
            }
        )
        if base_df.empty:
            return valid_results

        score_pct = base_df["Score"].rank(pct=True, method="average") * 100.0
        ifs_pct = base_df["IFS"].rank(pct=True, method="average") * 100.0

        # Keep the score anchored to absolute quality, with only a light
        # percentile component so the universe ranking does not swing too much.
        composite = (
            (base_df["Score"].fillna(0.0) * 0.45)
            + (base_df["IFS"].fillna(50.0) * 0.45)
            + (score_pct.fillna(50.0) * 0.10)
        ).clip(lower=0.0, upper=100.0)

        updated_cache = dict(decision_cache)

        for idx, r in enumerate(valid_results):
            symbol = str(r.get("symbol", "")).upper().strip()
            prev = decision_cache.get(symbol, {}) if symbol else {}

            raw_score = float(_safe_float(r.get("score"), np.nan))
            ifs_score = float(_safe_float(r.get("ifs_score"), np.nan))
            comp_score = float(composite.iloc[idx])
            spct = float(score_pct.iloc[idx])
            ipct = float(ifs_pct.iloc[idx])

            prev_final_score = _safe_float(prev.get("final_score", prev.get("score", np.nan)), np.nan)
            if pd.notna(prev_final_score) and np.isfinite(prev_final_score):
                comp_score = float((comp_score * 0.72) + (float(prev_final_score) * 0.28))

            r["score_raw"] = raw_score
            r["score_pct"] = spct
            r["ifs_pct"] = ipct
            r["score"] = comp_score


            market_regime = str(r.get("market_regime", "SIDEWAYS")).strip().upper()
            # Profit-only mode: only take longs when the macro backdrop is supportive.
            if market_regime == "BEAR":
                buy_threshold, strong_threshold = 82.0, 92.0
            elif market_regime == "BULL":
                buy_threshold, strong_threshold = 78.0, 88.0
            else:
                buy_threshold, strong_threshold = 85.0, 95.0

            trend_bullish = bool(r.get("trend_ok", False)) and market_regime == "BULL"
            trend_strict = bool(r.get("trend_ok", False)) and bool(r.get("trend_ok_strict", False))
            macro_gate_ok = str(("ON" if bool(r.get("macro_gate_ok", True)) else "OFF")).strip().upper() == "ON"
            liquidity_fail = "filter_likuiditas_gagal" in str(r.get("notes", "")).lower()

            market_struct = float(_safe_float(r.get("market_structure_score"), np.nan))
            smart_money = float(_safe_float(r.get("smart_money_score"), np.nan))
            rs_score = float(_safe_float(r.get("rs_composite_score"), np.nan))
            tradeability = float(_safe_float(r.get("tradeability_score"), np.nan))
            phase_ok = str(r.get("phase", "")).strip().lower() not in {"distribution", "markdown"}

            quality_ok = bool(
                trend_bullish
                and trend_strict
                and macro_gate_ok
                and phase_ok
                and market_struct >= 62.0
                and smart_money >= 60.0
                and rs_score >= 60.0
                and tradeability >= 55.0
                and ifs_score >= 72.0
            )

            strong_setup = bool(bool(r.get("unicorn_setup_valid", False)) and bool(r.get("unicorn_sniper_valid", False)))
            setup_valid = bool(strong_setup or quality_ok)

            decision = "AVOID"

            if liquidity_fail or market_regime != "BULL":
                decision = "AVOID"
            elif strong_setup and quality_ok and comp_score >= strong_threshold:
                decision = "STRONG BUY"
            elif quality_ok and comp_score >= buy_threshold:
                decision = "BUY"
            elif quality_ok and comp_score >= (buy_threshold - 2.0):
                decision = "WATCHLIST"
            elif setup_valid and comp_score >= (buy_threshold + 2.0) and ifs_score >= 75.0:
                decision = "WATCHLIST"

            r["DecisionRaw"] = str(r.get("Decision", r.get("decision", "AVOID")))
            r["Decision"] = decision
            r["decision"] = decision

            # Rebuild the entry plan so BUY/STRONG BUY decisions get a valid trade plan.
            try:
                entry_plan = build_entry_plan_from_context(r)
                if isinstance(entry_plan, dict):
                    for k, v in entry_plan.items():
                        r[k] = v
            except Exception:
                pass

            r = _apply_decision_hysteresis(symbol, r, updated_cache)
            updated_cache[symbol] = {
                "decision": r.get("decision"),
                "score": r.get("score", np.nan),
                "final_score": r.get("score", np.nan),
                "ifs_score": r.get("ifs_score", np.nan),
                "market_structure_score": r.get("market_structure_score", np.nan),
                "smart_money_score": r.get("smart_money_score", np.nan),
                "entry_price": r.get("entry_price", np.nan),
                "stop_price": r.get("stop_price", np.nan),
                "target_1": r.get("target_1", np.nan),
                "target_2": r.get("target_2", np.nan),
                "risk_reward_1": r.get("risk_reward_1", np.nan),
                "risk_reward_2": r.get("risk_reward_2", np.nan),
                "downgrade_streak": r.get("downgrade_streak", 0),
                "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            }

            recal_note = f"CompositeScore={comp_score:.2f};IFSBlend={ifs_score:.2f};RankPct={spct:.1f}"
            notes = r.get("notes", [])
            if not isinstance(notes, list):
                notes = [str(notes)] if pd.notna(notes) and str(notes).strip() else []
            notes.append(f"GlobalRecalibration:{recal_note}")
            r["notes"] = notes

            valid_results[idx] = r

        _save_decision_cache(updated_cache)
        return valid_results
    except Exception:
        return valid_results


# =========================================================
# Universe loading
# =========================================================
if universe_mode == "Paste tickers":
    universe = parse_universe_text(paste_text)
elif universe_mode == "Upload CSV":
    universe = load_universe_from_csv(uploaded_file)
else:
    local_file = Path("midcap_universe.csv")
    universe = load_universe_from_csv(local_file) if local_file.exists() else []

if "global_scan_results" not in st.session_state:
    st.session_state.global_scan_results = []
if "global_watch_df" not in st.session_state:
    st.session_state.global_watch_df = pd.DataFrame()
if "global_watch_df_raw" not in st.session_state:
    st.session_state.global_watch_df_raw = pd.DataFrame()
if "global_unicorn_df" not in st.session_state:
    st.session_state.global_unicorn_df = pd.DataFrame()
if "global_unicorn_df_raw" not in st.session_state:
    st.session_state.global_unicorn_df_raw = pd.DataFrame()
if "global_valid_results" not in st.session_state:
    st.session_state.global_valid_results = []

flow_val = map_flow_to_score("Netral")
GLOBAL_BENCHMARK_SYMBOL = "^JKSE"
GLOBAL_BENCHMARK_DF = load_ticker_data(GLOBAL_BENCHMARK_SYMBOL, months)
GLOBAL_MACRO_CONTEXT = build_macro_liquidity_gate(GLOBAL_BENCHMARK_DF, GLOBAL_BENCHMARK_SYMBOL)

def process_symbol(symbol: str):
    try:
        d = load_ticker_data(symbol, months)
        if d.empty or len(d) < min_history_bars:
            return {"valid": False, "symbol": symbol, "reason": "Data historis tidak mencukupi"}

        fundamental = compute_fundamental_grade(symbol)
        future_context = compute_future_fundamental_grade(symbol, d, GLOBAL_MACRO_CONTEXT)
        res = score_stock_smc(
            d,
            flow_used=False,
            flow_val=50,
            min_avg_volume=min_avg_volume,
            min_price=min_price,
            max_price=max_price,
            mode=GLOBAL_MODE,
            min_history_bars=min_history_bars,
            macro_context=GLOBAL_MACRO_CONTEXT,
            future_fundamental_context=future_context,
        )
        res["entry_plan"] = build_entry_plan(res)
        res.update(res["entry_plan"])
        ifs_context = compute_institutional_forward_score(
            symbol=symbol,
            price_df=d,
            bench_df=GLOBAL_MACRO_CONTEXT.get("benchmark_df"),
            current_fundamental=fundamental,
            future_context=future_context,
            technical_context=res,
        )
        res["symbol"] = symbol
        res["fundamental_score"] = fundamental.get("fundamental_score", np.nan)
        res["fundamental_grade"] = fundamental.get("fundamental_grade", "n/a")
        res["expected_revenue_growth_next_q"] = future_context.get("expected_revenue_growth_next_q", np.nan)
        res["expected_eps_growth_next_q"] = future_context.get("expected_eps_growth_next_q", np.nan)
        res["expected_margin_next_q"] = future_context.get("expected_margin_next_q", np.nan)
        res["ifs_score"] = ifs_context.get("ifs_score", np.nan)
        res["ifs_grade"] = ifs_context.get("ifs_grade", "n/a")
        res["ifs_breakdown"] = ifs_context.get("ifs_breakdown", {})
        res["ifs_detail"] = ifs_context.get("ifs_detail", {})
        return res
    except Exception as e:
        return {"valid": False, "symbol": symbol, "reason": str(e)}

# =========================================================

def run_deep_dive_analysis(
    ticker_input: str,
    strategy_mode: str,
    bandarmology_mode: str,
    benchmark_symbol_local: str,
    show_benchmark_local: bool,
    entry_buffer_atr_local: float,
    stop_loss_atr_local: float,
    take_profit_1_atr_local: float,
    take_profit_2_atr_local: float,
) -> dict:
    """Run a single-ticker deep dive and return a reusable analysis bundle."""
    deep_ticker = normalize_ticker(ticker_input)
    flow_val_local = map_flow_to_score(bandarmology_mode)

    stock_df = load_ticker_data(deep_ticker, months)
    bench_df = load_ticker_data(benchmark_symbol_local, months) if benchmark_symbol_local else pd.DataFrame()

    macro_context = None
    if show_benchmark_local and not bench_df.empty and len(bench_df) >= min_history_bars:
        macro_context = build_macro_liquidity_gate(bench_df.copy(), benchmark_symbol_local)

    if stock_df.empty or len(stock_df) < min_history_bars:
        return {
            "symbol": deep_ticker,
            "stock_df": stock_df,
            "bench_df": bench_df,
            "macro_context": macro_context,
            "stock_res": None,
            "fundamental": None,
            "future_context": None,
            "ifs_context": None,
            "entry_plan": None,
            "error": "Data ticker tidak cukup atau gagal diunduh.",
        }

    try:
        future_fundamental_context = compute_future_fundamental_grade(deep_ticker, stock_df, macro_context)
    except Exception as exc:
        future_fundamental_context = {
            "current_fundamental_score": np.nan,
            "current_fundamental_grade": "n/a",
            "future_fundamental_score": np.nan,
            "future_fundamental_grade": "n/a",
            "future_fundamental_direction": "n/a",
            "future_fundamental_confidence": np.nan,
            "future_growth_proxy": np.nan,
            "future_fundamental_momentum_score": np.nan,
            "future_seasonal_anomaly_score": np.nan,
            "future_inflection_score": np.nan,
            "future_cash_flow_proxy": np.nan,
            "future_balance_quality": np.nan,
            "future_price_proxy": np.nan,
            "future_cycle_support": np.nan,
            "future_macro_score": np.nan,
            "future_macro_adjusted_score": np.nan,
            "future_macro_gate_ok": False,
            "future_macro_gate_reason": f"future_fundamental_error: {type(exc).__name__}",
            "expected_revenue_growth_next_q": np.nan,
            "expected_eps_growth_next_q": np.nan,
            "expected_margin_next_q": np.nan,
            "future_moat_reason": f"future_fundamental_error: {type(exc).__name__}",
            "future_reliability": np.nan,
            "future_time_to_top": np.nan,
            "future_time_to_bottom": np.nan,
            "future_phase": "Unknown",
        }
    try:
        stock_res = score_stock_smc(
            stock_df,
            flow_used=True,
            flow_val=flow_val_local,
            min_avg_volume=min_avg_volume,
            min_price=min_price,
            max_price=max_price,
            mode=strategy_mode,
            min_history_bars=min_history_bars,
            macro_context=macro_context,
            future_fundamental_context=future_fundamental_context,
        )
    except Exception as exc:
        return {
            "symbol": deep_ticker,
            "stock_df": stock_df,
            "bench_df": bench_df,
            "macro_context": macro_context,
            "stock_res": None,
            "fundamental": None,
            "future_context": future_fundamental_context,
            "ifs_context": None,
            "entry_plan": None,
            "error": f"score_stock_smc failed: {type(exc).__name__}: {exc}",
        }

    try:
        fundamental = compute_fundamental_grade(deep_ticker)
    except Exception:
        fundamental = {}
    stock_res["peg_ratio"] = fundamental.get("peg_ratio", np.nan)
    stock_res["trailing_pe"] = fundamental.get("trailing_pe", np.nan)
    stock_res["forward_pe"] = fundamental.get("forward_pe", np.nan)
    stock_res["revenue_growth"] = fundamental.get("revenue_growth", np.nan)
    stock_res["earnings_growth"] = fundamental.get("earnings_growth", np.nan)
    stock_res["profit_margins"] = fundamental.get("profit_margins", np.nan)
    stock_res["future_fundamental_score"] = future_fundamental_context.get("future_fundamental_score", np.nan)
    stock_res["future_fundamental_grade"] = future_fundamental_context.get("future_fundamental_grade", "n/a")
    stock_res["future_fundamental_direction"] = future_fundamental_context.get("future_fundamental_direction", "n/a")
    stock_res["future_fundamental_confidence"] = future_fundamental_context.get("future_fundamental_confidence", np.nan)
    stock_res["future_fundamental_phase"] = future_fundamental_context.get("future_phase", "Unknown")
    stock_res["future_fundamental_reason"] = future_fundamental_context.get("future_moat_reason", "n/a")
    stock_res["expected_revenue_growth_next_q"] = future_fundamental_context.get("expected_revenue_growth_next_q", np.nan)
    stock_res["expected_eps_growth_next_q"] = future_fundamental_context.get("expected_eps_growth_next_q", np.nan)
    stock_res["expected_margin_next_q"] = future_fundamental_context.get("expected_margin_next_q", np.nan)

    try:
        entry_plan = build_entry_plan(
            stock_res,
            entry_buffer_atr=entry_buffer_atr_local,
            stop_loss_atr=stop_loss_atr_local,
            target_1_atr=take_profit_1_atr_local,
            target_2_atr=take_profit_2_atr_local,
        )
    except Exception as exc:
        entry_plan = {
            "entry_valid": False,
            "entry_reason": f"build_entry_plan failed: {type(exc).__name__}",
        }
    stock_res["entry_plan"] = entry_plan
    stock_res.update(entry_plan)

    try:
        ifs_context = compute_institutional_forward_score(
            symbol=deep_ticker,
            price_df=stock_df,
            bench_df=bench_df,
            current_fundamental=fundamental,
            future_context=future_fundamental_context,
            technical_context=stock_res,
        )
    except Exception as exc:
        ifs_context = {
            "ifs_score": np.nan,
            "ifs_grade": "n/a",
            "ifs_breakdown": {},
            "ifs_detail": {},
            "error": f"compute_institutional_forward_score failed: {type(exc).__name__}: {exc}",
        }
    stock_res["ifs_score"] = ifs_context.get("ifs_score", np.nan)
    stock_res["ifs_grade"] = ifs_context.get("ifs_grade", "n/a")
    stock_res["ifs_breakdown"] = ifs_context.get("ifs_breakdown", {})
    stock_res["ifs_detail"] = ifs_context.get("ifs_detail", {})

    return {
        "symbol": deep_ticker,
        "stock_df": stock_df,
        "bench_df": bench_df,
        "macro_context": macro_context,
        "stock_res": stock_res,
        "fundamental": fundamental,
        "future_context": future_fundamental_context,
        "ifs_context": ifs_context,
        "entry_plan": entry_plan,
    }

# =========================================================
# Tabs
# =========================================================
tab1, tab2 = st.tabs(["📈 Market Structure", "🏦 Institutional Forward Score"])

with tab1:
    st.subheader("Market Structure Top 20")
    st.caption("Fokus pada trend, momentum, cycle, risk, dan setup teknikal yang paling kuat.")

    if run_global_scan:
        if not universe:
            st.error("Universe kosong. Isi tickers di sidebar terlebih dahulu.")
        else:
            st.write(f"⚙️ Memproses analisis struktural pada **{len(universe)}** emiten...")
            progress = st.progress(0)
            status = st.empty()
            results = []

            with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(process_symbol, sym): sym for sym in universe}
                done = 0
                total = len(futures)
                for fut in cf.as_completed(futures):
                    done += 1
                    progress.progress(done / total)
                    status.caption(f"Selesai mengurai: {done}/{total} -> {futures[fut]}")
                    results.append(fut.result())

            progress.empty()
            status.empty()

            st.session_state.global_scan_results = results
            valid_results = [r for r in results if r.get("valid")]
            valid_results = _recalibrate_global_scan_results(valid_results)
            st.session_state.global_valid_results = valid_results

            if not valid_results:
                st.session_state.global_watch_df_raw = pd.DataFrame()
                st.session_state.global_watch_df = pd.DataFrame()
                st.session_state.global_unicorn_df_raw = pd.DataFrame()
                st.session_state.global_unicorn_df = pd.DataFrame()
                st.session_state.global_eligible_results = []
                st.session_state.global_unicorn_results = []
                st.warning("Tidak ada kandidat valid dari universe ini.")
            else:
                watch_rows = []
                for r in valid_results:
                    watch_rows.append(
                        {
                            "Ticker": r["symbol"],
                            "Decision": r["decision"],
                            "DecisionRank": _decision_rank(r["decision"]),
                            "SniperRank": _sniper_rank(r),
                            "Score": round(r["score"], 2),
                            "ScoreRaw": round(r.get("score_raw", np.nan), 2) if pd.notna(r.get("score_raw", np.nan)) else np.nan,
                            "MarketStruct": round(r.get("market_structure_score", np.nan), 2) if pd.notna(r.get("market_structure_score", np.nan)) else np.nan,
                            "Trend": round(r.get("trend_score", np.nan), 1) if pd.notna(r.get("trend_score", np.nan)) else np.nan,
                            "Momentum": round(r.get("momentum_score", np.nan), 1) if pd.notna(r.get("momentum_score", np.nan)) else np.nan,
                            "Cycle": r.get("dominant_period", np.nan),
                            "CycleRel": round(r.get("cycle_reliability", np.nan), 1) if pd.notna(r.get("cycle_reliability", np.nan)) else np.nan,
                            "Risk": round(r.get("risk_score", np.nan), 1) if pd.notna(r.get("risk_score", np.nan)) else np.nan,
                            "SmartMoney": round(r.get("smart_money_score", np.nan), 2) if pd.notna(r.get("smart_money_score", np.nan)) else np.nan,
                            "Reversal": r["reversal_hits"],
                            "FVG": "🔥 YES" if r["fvg_present"] else "NO",
                            "FVG_Age": r.get("fvg_age_bars", np.nan),
                            "FVG_Status": r.get("fvg_status", "-"),
                            "Unicorn": "🦄 YES" if r.get("unicorn_setup", False) else "NO",
                            "UnicornValid": "YES" if r.get("unicorn_setup_valid", False) else "NO",
                            "UnicornStatus": r.get("unicorn_setup_status", "-"),
                            "UnicornState": r.get("unicorn_setup_state", "-"),
                            "Sniper": "🎯 YES" if r.get("unicorn_sniper", False) else "NO",
                            "SniperValid": "YES" if r.get("unicorn_sniper_valid", False) else "NO",
                            "SniperStatus": r.get("unicorn_sniper_status", "-"),
                            "SniperState": r.get("unicorn_sniper_state", "-"),
                            "OrderBlock": "🎯 YES" if r["ob_present"] else "NO",
                            "TrendState": "BULLISH" if r["trend_ok"] else "BEARISH",
                            "Phase": r.get("phase", "-"),
                            "PhaseConf": round(r.get("phase_confidence", np.nan), 0) if pd.notna(r.get("phase_confidence", np.nan)) else np.nan,
                            "MacroPhase": r.get("macro_phase", "-"),
                            "MarketRegime": r.get("market_regime", "-"),
                            "RegimeConf": round(r.get("market_regime_confidence", np.nan), 2) if pd.notna(r.get("market_regime_confidence", np.nan)) else np.nan,
                            "RegimeReason": r.get("market_regime_reason", "-"),
                            "MacroScore": round(r.get("macro_score", np.nan), 1) if pd.notna(r.get("macro_score", np.nan)) else np.nan,
                            "MacroGate": "ON" if r.get("macro_gate_ok", True) else "OFF",
                            "IFS": round(r.get("ifs_score", np.nan), 2) if pd.notna(r.get("ifs_score", np.nan)) else np.nan,
                            "IFSGrade": r.get("ifs_grade", "n/a"),
                            "Entry": round(r["entry_price"], 2) if pd.notna(r["entry_price"]) else np.nan,
                            "Stop": round(r["stop_price"], 2) if pd.notna(r["stop_price"]) else np.nan,
                            "TP1": round(r.get("target_1", np.nan), 2) if pd.notna(r.get("target_1", np.nan)) else np.nan,
                            "TP2": round(r.get("target_2", np.nan), 2) if pd.notna(r.get("target_2", np.nan)) else np.nan,
                            "RR1": round(r.get("risk_reward_1", np.nan), 2) if pd.notna(r.get("risk_reward_1", np.nan)) else np.nan,
                            "RR2": round(r.get("risk_reward_2", np.nan), 2) if pd.notna(r.get("risk_reward_2", np.nan)) else np.nan,
                            "Notes": r["notes"],
                        }
                    )

                watch_df_raw = pd.DataFrame(watch_rows)
                watch_df = _build_watch_df(watch_rows, ascending=(ranking_sort_mode == "Ascending"))

                unicorn_df_raw = watch_df_raw[watch_df_raw["SniperRank"] > 0].copy() if not watch_df_raw.empty else pd.DataFrame()
                unicorn_df = _build_watch_df(unicorn_df_raw.to_dict("records"), ascending=(ranking_sort_mode == "Ascending")) if not unicorn_df_raw.empty else pd.DataFrame()

                st.session_state.global_watch_df_raw = watch_df_raw
                st.session_state.global_watch_df = watch_df
                st.session_state.global_unicorn_df_raw = unicorn_df_raw
                st.session_state.global_unicorn_df = unicorn_df
                st.session_state.global_eligible_results = valid_results
                st.session_state.global_unicorn_results = [r for r in valid_results if str(r.get("unicorn_setup_status", "")).upper() == "ENTRY" or str(r.get("unicorn_sniper_status", "")).upper() == "ENTRY" or r.get("unicorn_setup_valid", False) or r.get("unicorn_sniper_valid", False)]

                watch_view = _display_watch_df(watch_df)
                unicorn_view = _display_watch_df(unicorn_df)

                st.caption(
                    f"Valid results: {len(watch_df)} | Unicorn/Sniper: {len(unicorn_df)} | Ranking order: {ranking_sort_mode}"
                )

                priority_df = _build_watch_df(watch_rows, ascending=False)
                top20_priority = priority_df.head(20).copy()
                st.subheader("🔥 Top 3 High-Conviction Setups")
                top3 = priority_df[priority_df["Decision"].isin(["BUY", "STRONG BUY"])].head(3)
                if not top3.empty:
                    cols = st.columns(len(top3))
                    for idx, row in enumerate(top3.itertuples()):
                        with cols[idx]:
                            st.metric(
                                label=f"🌟 {row.Ticker} ({row.Decision})",
                                value=f"Rp {row.Entry:,.0f}" if pd.notna(row.Entry) else f"Rp {row.Stop:,.0f}",
                                delta=f"IFS: {row.IFS}",
                            )
                            st.markdown(
                                f"**Market Struct:** `{row.MarketStruct}`  \n"
                                f"**Trend/Momentum:** `{row.Trend}` / `{row.Momentum}`  \n"
                                f"**Cycle:** `{row.Cycle}` bars | Rel `{row.CycleRel}`  \n"
                                f"**Risk:** `{row.Risk}`  \n"
                                f"**TP1/TP2:** `{row.TP1}` / `{row.TP2}`  \n"
                                f"**RR1/RR2:** `{row.RR1}` / `{row.RR2}`  \n"
                                f"**Smart Money:** `{row.SmartMoney}`  \n"
                                f"**Phase:** `{row.Phase}`"
                            )
                else:
                    st.info("Belum ada kandidat BUY/STRONG BUY pada universe saat ini.")

                st.markdown("---")
                st.subheader("🏆 Market Structure Ranking (Top 20)")
                st.dataframe(watch_view.head(20), width="stretch", hide_index=True)

                st.markdown("---")
                st.subheader("🦄 Unicorn / Sniper Candidates")
                if unicorn_view.empty:
                    st.info("Tidak ada kandidat Unicorn / Sniper pada universe ini.")
                else:
                    st.dataframe(unicorn_view.head(20), width="stretch", hide_index=True)
    else:
        if not st.session_state.global_watch_df_raw.empty:
            watch_df = _build_watch_df(st.session_state.global_watch_df_raw.to_dict("records"), ascending=(ranking_sort_mode == "Ascending"))
            watch_view = _display_watch_df(watch_df)
            unicorn_raw = st.session_state.global_unicorn_df_raw
            unicorn_df = _build_watch_df(unicorn_raw.to_dict("records"), ascending=(ranking_sort_mode == "Ascending")) if not unicorn_raw.empty else pd.DataFrame()
            unicorn_view = _display_watch_df(unicorn_df)

            st.caption(
                f"Valid results: {len(watch_df)} | Unicorn/Sniper: {len(unicorn_view)} | Ranking order: {ranking_sort_mode}"
            )
            st.subheader("🏆 Market Structure Ranking (Top 20)")
            st.dataframe(watch_view.head(20), width="stretch", hide_index=True)
            st.markdown("---")
            st.subheader("🦄 Unicorn / Sniper Candidates")
            if unicorn_view.empty:
                st.info("Tidak ada kandidat Unicorn / Sniper pada universe ini.")
            else:
                st.dataframe(unicorn_view.head(20), width="stretch", hide_index=True)
            st.info("Klik **Run global scan** di sidebar untuk memperbarui ranking.")
        else:
            st.info("Klik **Run global scan** di sidebar untuk mulai scan universe.")

with tab2:
    st.subheader("🏦 Institutional Forward Score")
    st.caption("Dibagi menjadi overview, factor breakdown, smart money, forward fundamental, entry plan, dan detail saham.")

    c1, c2, c3 = st.columns([1.2, 1.0, 1.0])
    with c1:
        ticker_input = st.text_input("Ticker saham", value="BMRI", key="deep_ticker_input")
    with c2:
        strategy_mode = st.selectbox(
            "Strategy mode",
            ["Conservative", "Balanced", "Aggressive"],
            index=1,
            key="deep_strategy_mode",
        )
    with c3:
        bandarmology_mode = st.selectbox(
            "Bandarmology",
            ["Big Akumulasi", "Small Akumulasi", "Netral", "Small Distribusi", "Big Distribusi"],
            index=2,
            key="deep_bandarmology_mode",
        )

    with st.expander("⚙️ Deep Dive Settings", expanded=True):
        d1, d2, d3, d4 = st.columns([1, 1, 1, 1])
        with d1:
            benchmark_symbol_local = st.text_input("Benchmark IHSG symbol", value="^JKSE", key="deep_benchmark_symbol")
        with d2:
            show_benchmark_local = st.checkbox("Tampilkan benchmark vs saham", value=True, key="deep_show_benchmark")
        with d3:
            entry_buffer_atr_local = st.slider("Entry buffer (x ATR)", 0.10, 1.00, 0.25, 0.05, key="deep_entry_buffer_atr")
        with d4:
            stop_loss_atr_local = st.slider("Stop Loss (x ATR)", 1.0, 5.0, 1.8, 0.1, key="deep_stop_loss_atr")

        d5, d6 = st.columns([1, 1])
        with d5:
            take_profit_1_atr_local = st.slider("Take Profit 1 (x ATR)", 1.0, 6.0, 2.2, 0.1, key="deep_take_profit_1_atr")
        with d6:
            take_profit_2_atr_local = st.slider("Take Profit 2 (x ATR)", 2.0, 8.0, 3.8, 0.1, key="deep_take_profit_2_atr")

        analyze_btn = st.button("Analyze ticker", type="primary", key="deep_analyze_btn")

    analysis_bundle = {}
    if analyze_btn:
        deep_ticker = normalize_ticker(ticker_input)
        analysis_bundle = run_deep_dive_analysis(
            ticker_input=ticker_input,
            strategy_mode=strategy_mode,
            bandarmology_mode=bandarmology_mode,
            benchmark_symbol_local=benchmark_symbol_local,
            show_benchmark_local=show_benchmark_local,
            entry_buffer_atr_local=entry_buffer_atr_local,
            stop_loss_atr_local=stop_loss_atr_local,
            take_profit_1_atr_local=take_profit_1_atr_local,
            take_profit_2_atr_local=take_profit_2_atr_local,
        )
        st.session_state.ifs_analysis = analysis_bundle
    else:
        analysis_bundle = st.session_state.get("ifs_analysis", {})

    stock_res = analysis_bundle.get("stock_res")
    ifs_context = analysis_bundle.get("ifs_context")
    fundamental = analysis_bundle.get("fundamental")
    future_context = analysis_bundle.get("future_context")
    deep_ticker = analysis_bundle.get("symbol", normalize_ticker(ticker_input))
    st.session_state["deep_selected_symbol"] = deep_ticker
    stock_df = analysis_bundle.get("stock_df", pd.DataFrame())
    bench_df = analysis_bundle.get("bench_df", pd.DataFrame())
    macro_context = analysis_bundle.get("macro_context")
    entry_plan = analysis_bundle.get("entry_plan", {})

    sub_overview, sub_factor, sub_smart, sub_forward, sub_entry, sub_news, sub_detail = st.tabs(
        ["Overview", "Factor Breakdown", "Smart Money", "Forward Fundamental", "Entry Plan", "News Catalyst", "Detail Saham"]
    )

    with sub_entry:
        st.subheader("Entry Plan")
        if ifs_context is not None and stock_res is not None:
            plan = stock_res.get("entry_plan", {})
            cols = st.columns(4)
            cols[0].metric("Signal", stock_res.get("decision", "n/a"), f'Confidence {ifs_context.get("ifs_detail", {}).get("future_confidence", np.nan):.0f}%')
            cols[1].metric("Entry", f'Rp {plan.get("entry_price_plan", np.nan):,.0f}' if pd.notna(plan.get("entry_price_plan", np.nan)) else "n/a")
            cols[2].metric("Stop Loss", f'Rp {plan.get("stop_loss_plan", np.nan):,.0f}' if pd.notna(plan.get("stop_loss_plan", np.nan)) else "n/a")
            cols[3].metric("Trigger", plan.get("entry_trigger", "n/a"), plan.get("plan_reason", "n/a"))

            entry_table = pd.DataFrame(
                [
                    {"Metric": "Decision", "Value": stock_res.get("decision", "n/a")},
                    {"Metric": "Entry Zone Low", "Value": f'Rp {plan.get("entry_zone_low", np.nan):,.0f}' if pd.notna(plan.get("entry_zone_low", np.nan)) else "n/a"},
                    {"Metric": "Entry Zone High", "Value": f'Rp {plan.get("entry_zone_high", np.nan):,.0f}' if pd.notna(plan.get("entry_zone_high", np.nan)) else "n/a"},
                    {"Metric": "Entry Trigger", "Value": plan.get("entry_trigger", "n/a")},
                    {"Metric": "Stop Loss", "Value": f'Rp {plan.get("stop_loss_plan", np.nan):,.0f}' if pd.notna(plan.get("stop_loss_plan", np.nan)) else "n/a"},
                    {"Metric": "Target 1", "Value": f'Rp {plan.get("target_1", np.nan):,.0f}' if pd.notna(plan.get("target_1", np.nan)) else "n/a"},
                    {"Metric": "Target 2", "Value": f'Rp {plan.get("target_2", np.nan):,.0f}' if pd.notna(plan.get("target_2", np.nan)) else "n/a"},
                    {"Metric": "Risk / Share", "Value": f'Rp {plan.get("risk_per_share", np.nan):,.0f}' if pd.notna(plan.get("risk_per_share", np.nan)) else "n/a"},
                    {"Metric": "RR1", "Value": f'{plan.get("risk_reward_1", np.nan):.2f}' if pd.notna(plan.get("risk_reward_1", np.nan)) else "n/a"},
                    {"Metric": "RR2", "Value": f'{plan.get("risk_reward_2", np.nan):.2f}' if pd.notna(plan.get("risk_reward_2", np.nan)) else "n/a"},
                    {"Metric": "Upside TP1", "Value": f'{plan.get("upside_to_t1_pct", np.nan):.2f}%' if pd.notna(plan.get("upside_to_t1_pct", np.nan)) else "n/a"},
                    {"Metric": "Upside TP2", "Value": f'{plan.get("upside_to_t2_pct", np.nan):.2f}%' if pd.notna(plan.get("upside_to_t2_pct", np.nan)) else "n/a"},
                    {"Metric": "Plan Reason", "Value": plan.get("plan_reason", "n/a")},
                ]
            )
            st.dataframe(entry_table, width="stretch", hide_index=True)
        else:
            st.info("Klik Analyze ticker untuk melihat entry plan otomatis.")


    with sub_news:
        st.subheader("News Catalyst")
        st.caption("Berita diklasifikasikan oleh modul Catalyst NLP: PASS / WATCH / REJECT berdasarkan relevansi struktural, bukan sekadar sentimen pasar.")

        news_default_symbol = st.session_state.get("deep_selected_symbol") or deep_ticker or normalize_ticker(ticker_input)
        if news_default_symbol:
            st.session_state["news_catalyst_symbol"] = news_default_symbol

        news_symbol = st.text_input(
            "Ticker untuk news scan",
            value=st.session_state.get("news_catalyst_symbol", news_default_symbol if news_default_symbol else "BBCA.JK"),
            key="news_catalyst_symbol",
        )

        news_source_mode = st.radio(
            "Sumber berita",
            ["Yahoo Finance", "Indonesia News", "Paste manual"],
            horizontal=True,
            key="news_catalyst_source_mode",
        )

        news_limit = st.slider("Maksimal item berita", 5, 25, 10, key="news_catalyst_limit")
        news_paste = st.text_area(
            "Paste berita manual (format: judul | sumber | ringkasan). Satu item per baris.",
            height=170,
            placeholder="Contoh:\nLaba BBCA tumbuh 12% | Reuters | Emiten membukukan pertumbuhan laba...\nBI tahan suku bunga | BI | Kebijakan moneter tetap akomodatif...",
            key="news_catalyst_manual_input",
        )

        fetch_col1, fetch_col2 = st.columns([1, 1])
        fetch_clicked = fetch_col1.button("Refresh news catalyst", type="primary", key="news_catalyst_fetch_btn")
        use_cached = fetch_col2.checkbox("Gunakan cache terakhir", value=True, key="news_catalyst_use_cache")

        cache = st.session_state.get("news_catalyst_cache", {})
        should_fetch = False
        if news_source_mode in ("Yahoo Finance", "Indonesia News"):
            cache_symbol = cache.get("symbol")
            cache_mode = cache.get("source_mode")
            if fetch_clicked:
                should_fetch = True
            elif use_cached and cache_symbol == news_symbol and cache_mode == news_source_mode and cache.get("raw_news"):
                should_fetch = False
            elif cache_symbol != news_symbol or cache_mode != news_source_mode or not cache.get("raw_news"):
                should_fetch = True

        if news_source_mode in ("Yahoo Finance", "Indonesia News"):
            if should_fetch:
                with st.spinner(f"Mengambil news untuk {news_symbol}..."):
                    if news_source_mode == "Indonesia News":
                        raw_news = fetch_indonesia_news_items(news_symbol, news_limit)
                    else:
                        raw_news = fetch_ticker_news_items(news_symbol, news_limit)
                    scored_news = filter_news_items(raw_news) if raw_news else []
                    cache = {
                        "symbol": news_symbol,
                        "source_mode": news_source_mode,
                        "raw_news": raw_news,
                        "scored_news": scored_news,
                    }
                    st.session_state.news_catalyst_cache = cache
            scored_news = cache.get("scored_news", []) if cache.get("symbol") == news_symbol and cache.get("source_mode") == news_source_mode else []
            raw_news = cache.get("raw_news", []) if cache.get("symbol") == news_symbol and cache.get("source_mode") == news_source_mode else []

            if not raw_news:
                if news_source_mode == "Indonesia News":
                    fallback_news = fetch_indonesia_news_items(news_symbol, news_limit)
                else:
                    fallback_news = fetch_ticker_news_search_items(news_symbol, news_limit)
                if fallback_news:
                    raw_news = fallback_news
                    scored_news = filter_news_items(raw_news)
                    cache = {
                        "symbol": news_symbol,
                        "source_mode": news_source_mode,
                        "raw_news": raw_news,
                        "scored_news": scored_news,
                        "fallback": "indonesia_google_news" if news_source_mode == "Indonesia News" else "yahoo_search",
                    }
                    st.session_state.news_catalyst_cache = cache

        else:
            raw_news = _parse_manual_news_lines(news_paste)
            scored_news = filter_news_items(raw_news) if raw_news else []
            if fetch_clicked:
                cache = {
                    "symbol": news_symbol,
                    "source_mode": "Paste manual",
                    "raw_news": raw_news,
                    "scored_news": scored_news,
                }
                st.session_state.news_catalyst_cache = cache

        if news_source_mode == "Paste manual" and fetch_clicked:
            raw_news = _parse_manual_news_lines(news_paste)
            scored_news = filter_news_items(raw_news) if raw_news else []
            cache = {
                "symbol": news_symbol,
                "source_mode": "Paste manual",
                "raw_news": raw_news,
                "scored_news": scored_news,
            }
            st.session_state.news_catalyst_cache = cache

        scored_news = scored_news if isinstance(scored_news, list) else []
        raw_news = raw_news if isinstance(raw_news, list) else []

        if not scored_news:
            st.info("Belum ada item news catalyst. Klik refresh untuk Indonesia News / Yahoo Finance, atau paste berita manual untuk dianalisis.")
        else:
            pass_count = sum(1 for item in scored_news if str(getattr(item, "decision", "")).upper() == "PASS")
            watch_count = sum(1 for item in scored_news if str(getattr(item, "decision", "")).upper() == "WATCH")
            reject_count = sum(1 for item in scored_news if str(getattr(item, "decision", "")).upper() == "REJECT")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Items", len(scored_news))
            m2.metric("PASS", pass_count)
            m3.metric("WATCH", watch_count)
            m4.metric("REJECT", reject_count)

            news_rows = []
            for idx, item in enumerate(scored_news):
                raw_item = raw_news[idx] if idx < len(raw_news) and isinstance(raw_news[idx], dict) else {}
                decision = str(getattr(item, "decision", "WATCH")).upper()
                news_rows.append(
                    {
                        "Decision": _news_decision_label(decision),
                        "Category": getattr(item, "category", "unknown"),
                        "Confidence": int(getattr(item, "confidence", 50)),
                        "Horizon": getattr(item, "impact_horizon", "days"),
                        "Materiality": getattr(item, "materiality", "medium"),
                        "Source Quality": getattr(item, "source_quality", "unknown"),
                        "Title": raw_item.get("title", ""),
                        "Source": raw_item.get("source", ""),
                        "Published": raw_item.get("published_at", ""),
                        "Link": raw_item.get("link", ""),
                        "Summary": raw_item.get("summary", ""),
                        "Reasons": ", ".join(getattr(item, "reasons", [])) if getattr(item, "reasons", None) else "",
                        "Red Flags": ", ".join(getattr(item, "red_flags", [])) if getattr(item, "red_flags", None) else "",
                        "Tags": ", ".join(getattr(item, "tags", [])) if getattr(item, "tags", None) else "",
                    }
                )

            news_df = pd.DataFrame(news_rows)
            if not news_df.empty:
                st.dataframe(
                    news_df[
                        [
                            "Decision",
                            "Confidence",
                            "Category",
                            "Horizon",
                            "Materiality",
                            "Source Quality",
                            "Published",
                            "Title",
                            "Source",
                            "Link",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("#### Detail per item")
                for i, row in news_df.iterrows():
                    title = row.get("Title", "")
                    badge = row.get("Decision", "WATCH")
                    confidence = row.get("Confidence", 50)
                    with st.expander(f"{badge} · {confidence}% · {title}", expanded=False):
                        st.write(f"**Category:** {row.get('Category', 'unknown')}")
                        st.write(f"**Horizon:** {row.get('Horizon', 'days')}")
                        st.write(f"**Materiality:** {row.get('Materiality', 'medium')}")
                        st.write(f"**Source quality:** {row.get('Source Quality', 'unknown')}")
                        if row.get("Published"):
                            st.write(f"**Published:** {row.get('Published')}")
                        if row.get("Source"):
                            st.write(f"**Source:** {row.get('Source')}")
                        if row.get("Link"):
                            st.write(f"**Link:** {row.get('Link')}")
                        if row.get("Summary"):
                            st.write(f"**Summary:** {row.get('Summary')}")
                        if row.get("Reasons"):
                            st.write(f"**Reasons:** {row.get('Reasons')}")
                        if row.get("Red Flags"):
                            st.write(f"**Red flags:** {row.get('Red Flags')}")
                        if row.get("Tags"):
                            st.write(f"**Tags:** {row.get('Tags')}")

    with sub_detail:
        if analyze_btn:
            deep_ticker = normalize_ticker(ticker_input)
            flow_val_local = map_flow_to_score(bandarmology_mode)

            stock_df = load_ticker_data(deep_ticker, months)
            bench_df = load_ticker_data(benchmark_symbol_local, months) if benchmark_symbol_local else pd.DataFrame()

            macro_context = None
            if show_benchmark_local and not bench_df.empty and len(bench_df) >= min_history_bars:
                macro_context = build_macro_liquidity_gate(bench_df.copy(), benchmark_symbol_local)

            if stock_df.empty or len(stock_df) < min_history_bars:
                st.error("Data ticker tidak cukup atau gagal diunduh.")
            else:
                future_fundamental_context = compute_future_fundamental_grade(deep_ticker, stock_df, macro_context)

                stock_res = score_stock_smc(
                    stock_df,
                    flow_used=True,
                    flow_val=flow_val_local,
                    min_avg_volume=min_avg_volume,
                    min_price=min_price,
                    max_price=max_price,
                    mode=strategy_mode,
                    min_history_bars=min_history_bars,
                    macro_context=macro_context,
                    future_fundamental_context=future_fundamental_context,
                )

                if not stock_res.get("valid", False):
                    st.warning(stock_res.get("reason", "Analisis teknikal tidak valid."))
                    st.stop()

                stock = stock_res.get("df", pd.DataFrame()).copy()
                stock_last = stock_res.get("last")
                fundamental = compute_fundamental_grade(deep_ticker)
                stock_res["peg_ratio"] = fundamental.get("peg_ratio", np.nan)
                stock_res["trailing_pe"] = fundamental.get("trailing_pe", np.nan)
                stock_res["forward_pe"] = fundamental.get("forward_pe", np.nan)
                stock_res["revenue_growth"] = fundamental.get("revenue_growth", np.nan)
                stock_res["earnings_growth"] = fundamental.get("earnings_growth", np.nan)
                stock_res["profit_margins"] = fundamental.get("profit_margins", np.nan)
                stock_res["future_fundamental_score"] = future_fundamental_context.get("future_fundamental_score", np.nan)
                stock_res["future_fundamental_grade"] = future_fundamental_context.get("future_fundamental_grade", "n/a")
                stock_res["future_fundamental_direction"] = future_fundamental_context.get("future_fundamental_direction", "n/a")
                stock_res["future_fundamental_confidence"] = future_fundamental_context.get("future_fundamental_confidence", np.nan)
                stock_res["future_fundamental_phase"] = future_fundamental_context.get("future_phase", "Unknown")
                stock_res["future_fundamental_reason"] = future_fundamental_context.get("future_moat_reason", "n/a")
                ifs_context = compute_institutional_forward_score(
                    symbol=deep_ticker,
                    price_df=stock_df,
                    bench_df=bench_df,
                    current_fundamental=fundamental,
                    future_context=future_fundamental_context,
                    technical_context=stock_res,
                )
                entry_plan = build_entry_plan(
                    stock_res,
                    entry_buffer_atr=entry_buffer_atr_local,
                    stop_loss_atr=stop_loss_atr_local,
                    target_1_atr=take_profit_1_atr_local,
                    target_2_atr=take_profit_2_atr_local,
                )
                stock_res["entry_plan"] = entry_plan
                stock_res.update(entry_plan)
                stock_res["ifs_score"] = ifs_context.get("ifs_score", np.nan)
                stock_res["ifs_grade"] = ifs_context.get("ifs_grade", "n/a")
                stock_res["ifs_breakdown"] = ifs_context.get("ifs_breakdown", {})
                stock_res["ifs_detail"] = ifs_context.get("ifs_detail", {})
                st.session_state.ifs_analysis = {
                    "symbol": deep_ticker,
                    "stock_df": stock_df,
                    "bench_df": bench_df,
                    "macro_context": macro_context,
                    "stock_res": stock_res,
                    "fundamental": fundamental,
                    "future_context": future_fundamental_context,
                    "ifs_context": ifs_context,
                    "entry_plan": entry_plan,
                    "strategy_mode": strategy_mode,
                    "bandarmology_mode": bandarmology_mode,
                    "ticker_input": ticker_input,
                }
                bench = pd.DataFrame()
                bench_cycle = None
                if macro_context is not None:
                    bench = bench_df.copy()
                    bench_cycle = macro_context.get("cycle_tuple")

                stock_status = "Near Bottom" if stock_res["time_to_bottom"] <= 4 else "Mid-Cycle Moving"
                bench_status = "n/a"
                if macro_context is not None:
                    bench_status = "Near Bottom" if macro_context.get("macro_time_to_bottom", 999) <= 4 else "Mid-Cycle Moving"
                stock_top = stock_res.get("time_to_top", np.nan)
                stock_phase_age = stock_res.get("phase_age_bars", np.nan)
                stock_phase_age_pct = stock_res.get("phase_age_pct", np.nan)
                macro_context = macro_context or build_macro_liquidity_gate(pd.DataFrame(), benchmark_symbol_local)
                bench_top = macro_context.get("macro_time_to_top", np.nan) if macro_context is not None else np.nan
                bench_phase_age = macro_context.get("macro_phase_age_bars", np.nan) if macro_context is not None else np.nan
                bench_phase_age_pct = macro_context.get("macro_phase_age_pct", np.nan) if macro_context is not None else np.nan

                st.markdown(
                    """
                    <div style="margin-top: 0.25rem;">
                        <h2 style="margin-bottom:0.25rem;">⏳ Trader Time Analysis Model</h2>
                        <div style="font-size:1.05rem; opacity:0.9;">
                            Mengukur frekuensi dominan dan estimasi waktu pembalikan tren berlandaskan struktur matematika siklus bursa.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                stock_period = stock_res["dominant_period"]
                stock_ttb = stock_res["time_to_bottom"]
                stock_cycle_info = stock_res.get("cycle_info", {})
                bench_period = bench_cycle[0] if bench_cycle is not None else None
                bench_ttb = bench_cycle[1] if bench_cycle is not None else None
                bench_cycle_info = bench_cycle[3] if bench_cycle is not None and len(bench_cycle) > 3 else {}

                stock_html = f"""
                <div style="background:linear-gradient(180deg, rgba(235,244,255,1) 0%, rgba(225,235,250,1) 100%); padding:22px; border-radius:18px; border:1px solid rgba(0,0,0,0.05); box-shadow:0 8px 24px rgba(0,0,0,0.04);">
                    <div style="font-size:1.15rem; font-weight:700; color:#173b6d; margin-bottom:18px;">Siklus Saham ({deep_ticker})</div>
                    <div style="font-size:1.02rem; color:#173b6d; line-height:2;">
                        <div>• <b>Periode Siklus Dominan:</b> {stock_period} Hari Bursa</div>
                        <div>• <b>Estimasi Sisa Waktu Menuju Bottom berikutnya:</b> {stock_ttb} Bar</div>
                        <div>• <b>Estimasi Menuju Top Berikutnya:</b> {stock_top} Bar</div>
                        <div>• <b>Phase Age:</b> {stock_phase_age} Bar ({stock_phase_age_pct:.0f}%)</div>
                        <div>• <b>Status Posisi Siklus:</b> {stock_status}</div>
                        <div>• <b>8-Phase Cycle:</b> {stock_res["phase"]} ({stock_res["phase_confidence"]:.0f}%)</div>
                        <div>• <b>FFT / Hilbert / Autocorr:</b> {stock_cycle_info.get("fft_period", "-")} / {stock_cycle_info.get("hilbert_period", "-")} / {stock_cycle_info.get("autocorr_period", "-")}</div>
                        <div>• <b>Weighted Composite:</b> {stock_cycle_info.get("weighted_period", stock_period)} bars</div>
                        <div>• <b>Cycle Reliability:</b> {stock_cycle_info.get("cycle_reliability", np.nan):.0f}%</div>
                        <div>• <b>Cycle Gate Reason:</b> {stock_cycle_info.get("cycle_gate_reason", "OK")}</div>
                        <div>• <b>Detrend Method:</b> {stock_cycle_info.get('detrend_method', 'HighPass+TailHilbert')}</div>
                        <div>• <b>Macro Gate:</b> {'ON' if stock_res.get('macro_gate_ok', True) else 'OFF'} ({stock_res.get('macro_phase', 'Unknown')})</div>
                        <div>• <b>Macro Gate Reason:</b> {stock_res.get('macro_gate_reason', 'OK')}</div>
                        <div>• <b>Trend Quality:</b> {'OK' if stock_res.get('trend_ok', False) else 'Weak'}</div>
                    </div>
                </div>
                """
                bench_html = f"""
                <div style="background:linear-gradient(180deg, rgba(255,248,230,1) 0%, rgba(248,238,210,1) 100%); padding:22px; border-radius:18px; border:1px solid rgba(0,0,0,0.05); box-shadow:0 8px 24px rgba(0,0,0,0.04);">
                    <div style="font-size:1.15rem; font-weight:700; color:#8a4b00; margin-bottom:18px;">Siklus Makro Komposit (IHSG)</div>
                    <div style="font-size:1.02rem; color:#8a4b00; line-height:2;">
                        <div>• <b>Periode Siklus Dominan:</b> {bench_period if bench_period is not None else '-'} Hari Bursa</div>
                        <div>• <b>Estimasi Sisa Waktu Menuju Bottom berikutnya:</b> {bench_ttb if bench_ttb is not None else '-'} Bar</div>
                        <div>• <b>Status Posisi Siklus Makro:</b> {bench_status}</div>
                        <div>• <b>FFT / Hilbert / Autocorr:</b> {bench_cycle_info.get("fft_period", "-")} / {bench_cycle_info.get("hilbert_period", "-")} / {bench_cycle_info.get("autocorr_period", "-")}</div>
                        <div>• <b>Weighted Composite:</b> {bench_cycle_info.get("weighted_period", bench_period if bench_period is not None else '-') } bars</div>
                        <div>• <b>Estimasi Menuju Top Berikutnya:</b> {bench_top} Bar</div>
                        <div>• <b>Phase Age:</b> {bench_phase_age} Bar ({bench_phase_age_pct:.0f}%)</div>
                        <div>• <b>Macro Score:</b> {macro_context.get('macro_score', np.nan):.0f}%</div>
                        <div>• <b>Macro Gate:</b> {'ON' if macro_context.get('macro_gate_ok', True) else 'OFF'}</div>
                        <div>• <b>Macro Gate Reason:</b> {macro_context.get('macro_gate_reason', 'OK')}</div>
                        <div>• <b>Detrend Method:</b> {bench_cycle_info.get('detrend_method', 'ZLEMA')}</div>
                        <div>• <b>Trend Lag:</b> {bench_cycle_info.get('trend_lag_bars', '-')} Bar</div>
                    </div>
                </div>
                """
                st.markdown(
                    f"""
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:18px; margin: 18px 0 8px 0;">
                        {stock_html}
                        {bench_html}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                ctop1, ctop2, ctop3, ctop4 = st.columns(4)
                ctop1.metric("Decision", stock_res["decision"])
                ctop2.metric("Score", f"{stock_res['score']:.2f}")
                ctop3.metric("Close", f"Rp {stock_res['close']:,.0f}")
                ctop4.metric("Phase", stock_res["phase"])

                ctop5, ctop6, ctop7, ctop8 = st.columns(4)
                ctop5.metric("Smart Money", f"{stock_res['smart_money_score']:.0f}")
                ctop6.metric("Fundamental", f"{fundamental.get('fundamental_score', np.nan):.0f}" if pd.notna(fundamental.get('fundamental_score', np.nan)) else "n/a")
                ctop7.metric("PEG", f"{fundamental.get('peg_ratio', np.nan):.2f}" if pd.notna(fundamental.get('peg_ratio', np.nan)) else "n/a")
                ctop8.metric("Grade", fundamental.get("fundamental_grade", "n/a"))

                cmid1, cmid2, cmid3, cmid4 = st.columns(4)
                cmid1.metric("Unicorn", stock_res.get("unicorn_setup_status", "n/a"))
                cmid2.metric("RSI14", f"{stock_res['rsi']:.2f}")
                cmid3.metric("ADX14", f"{stock_res['adx']:.2f}" if pd.notna(stock_res["adx"]) else "n/a")
                cmid4.metric("Phase Confidence", f"{stock_res['phase_confidence']:.0f}%")

                left, right = st.columns([1, 1])
                with left:
                    st.subheader("Time Analysis - Stock")
                    st.write(f"**Dominant cycle:** `{stock_res['dominant_period']} bars`")
                    st.write(f"**FFT / Hilbert / Autocorr:** `{stock_cycle_info.get('fft_period', '-')}` / `{stock_cycle_info.get('hilbert_period', '-')}` / `{stock_cycle_info.get('autocorr_period', '-')}`")
                    st.write(f"**Weighted composite:** `{stock_cycle_info.get('weighted_period', stock_res['dominant_period'])} bars`")
                    st.write(f"**Detrend method:** `{stock_cycle_info.get('detrend_method', 'ZLEMA')}` | lag `{stock_cycle_info.get('trend_lag_bars', '-')}` bars")
                    st.write(f"**Time to next bottom:** `{stock_res['time_to_bottom']} bars`")
                    st.write(f"**Time to next top:** `{stock_res.get('time_to_top', np.nan)} bars`")
                    st.write(f"**Phase age:** `{stock_res.get('phase_age_bars', np.nan)} bars` ({stock_res.get('phase_age_pct', np.nan):.0f}%)")
                    st.write(f"**Cycle status:** `{stock_status}`")
                    st.write(f"**8-Phase:** `{stock_res['phase']}`")
                    st.write(f"**Phase confidence:** `{stock_res['phase_confidence']:.0f}%`")
                    st.write(f"**Phase reason:** {stock_res['phase_reason']}")
                    st.write(f"**Reversal signals:** `{stock_res['reversal_hits']}`")
                    st.write(f"**OBV trend:** `{stock_res['obv_trend']}`")
                    st.write(f"**CMF20 / MFI14:** `{stock_res['cmf20']:.2f}` / `{stock_res['mfi14']:.2f}`")
                    st.write(f"**Stoch K/D:** `{stock_res['stoch_k']:.2f}` / `{stock_res['stoch_d']:.2f}`")
                    st.write(f"**PEG:** `{stock_res.get('peg_ratio', np.nan):.2f}`" if pd.notna(stock_res.get("peg_ratio", np.nan)) else "**PEG:** n/a")
                    st.write(f"**SMC:** FVG `{stock_res['fvg_present']}` | OB `{stock_res['ob_present']}` | Unicorn `{stock_res.get('unicorn_setup', False)}` | Sniper `{stock_res.get('unicorn_sniper', False)}`")
                    st.write(f"**Bandarmology input:** `{bandarmology_mode}`")

                with right:
                    st.subheader("Recommendation")
                    if stock_res["decision"] in {"BUY", "STRONG BUY"}:
                        st.success("Saham layak dibeli menurut filter saat ini.")
                        st.write(f"**Recommended entry:** `Rp {stock_res['entry_price']:,.0f}`")
                        st.write(f"**Recommended stoploss:** `Rp {stock_res['stop_price']:,.0f}`")
                        rr_risk = stock_res["entry_price"] - stock_res["stop_price"]
                        st.write(f"**Risk per share:** `Rp {rr_risk:,.0f}`")
                        tp_plan = stock_res.get("entry_plan", {})
                        tp1 = tp_plan.get("target_1", np.nan)
                        tp2 = tp_plan.get("target_2", np.nan)
                        if pd.notna(tp1) and pd.notna(tp2):
                            st.write(f"**Take profit T1/T2:** `Rp {tp1:,.0f}` / `Rp {tp2:,.0f}`")
                        elif pd.notna(tp1):
                            st.write(f"**Take profit target:** `Rp {tp1:,.0f}`")
                        else:
                            tp_price = stock_res["entry_price"] + take_profit_1_atr_local * float(stock_res["last"]["ATR14"])
                            st.write(f"**Take profit target:** `Rp {tp_price:,.0f}`")
                    else:
                        st.warning("Belum layak beli. Tunggu reversal / struktur membaik.")
                        st.write("Entry/stoploss tidak ditampilkan karena belum memenuhi kriteria beli.")

                st.markdown("---")
                fig = make_subplots(
                    rows=4,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.04,
                    row_heights=[0.45, 0.15, 0.20, 0.20],
                    subplot_titles=(
                        f"{deep_ticker} Price Action",
                        "Reversal / SMC / OBV Signals",
                        "Relative Strength vs Benchmark",
                        "Volume",
                    ),
                )

                fig.add_trace(
                    go.Candlestick(
                        x=stock.index,
                        open=stock["Open"],
                        high=stock["High"],
                        low=stock["Low"],
                        close=stock["Close"],
                        name="Price",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(go.Scatter(x=stock.index, y=stock["EMA20"], name="EMA20", mode="lines"), row=1, col=1)
                fig.add_trace(go.Scatter(x=stock.index, y=stock["EMA50"], name="EMA50", mode="lines"), row=1, col=1)
                fig.add_trace(go.Scatter(x=stock.index, y=stock["EMA200"], name="EMA200", mode="lines"), row=1, col=1)

                fvg_df = stock[stock["Bullish_FVG"]].tail(5)
                for idx, _ in fvg_df.iterrows():
                    loc = stock.index.get_loc(idx)
                    if loc >= 2:
                        fig.add_shape(
                            type="rect",
                            x0=idx,
                            x1=stock.index[-1],
                            y0=float(stock["High"].iloc[loc - 2]),
                            y1=float(stock["Low"].iloc[loc]),
                            fillcolor="rgba(0, 255, 0, 0.08)",
                            line=dict(width=0),
                            row=1,
                            col=1,
                        )

                ob_df = stock[stock["Bullish_OB"]].tail(5)
                for idx, _ in ob_df.iterrows():
                    loc = stock.index.get_loc(idx)
                    if loc >= 1:
                        fig.add_shape(
                            type="rect",
                            x0=stock.index[loc - 1],
                            x1=stock.index[-1],
                            y0=float(stock["Low"].iloc[loc - 1]),
                            y1=float(stock["High"].iloc[loc - 1]),
                            fillcolor="rgba(255, 165, 0, 0.10)",
                            line=dict(width=0),
                            row=1,
                            col=1,
                        )

                unicorn_df = stock[stock["Unicorn_Setup"]].tail(5)
                for idx, _ in unicorn_df.iterrows():
                    loc = stock.index.get_loc(idx)
                    if loc >= 2:
                        fig.add_shape(
                            type="rect",
                            x0=idx,
                            x1=stock.index[-1],
                            y0=float(stock["FVG_Bottom"].iloc[loc]),
                            y1=float(stock["FVG_Top"].iloc[loc]),
                            fillcolor="rgba(138, 43, 226, 0.10)",
                            line=dict(width=0),
                            row=1,
                            col=1,
                        )

                sig_names = [
                    "Bullish_Engulfing",
                    "Hammer",
                    "Inverted_Hammer",
                    "Morning_Star",
                    "EMA20_Reclaim",
                    "MACD_Bull_Cross",
                    "RSI_Bounce",
                    "Breakout_5D",
                ]
                for sig in sig_names:
                    y = stock["Low"] * (0.995 if sig in ["Hammer", "Inverted_Hammer"] else 1.005)
                    fig.add_trace(
                        go.Scatter(
                            x=stock.index,
                            y=np.where(stock[sig], y, np.nan),
                            mode="markers",
                            name=sig,
                        ),
                        row=2,
                        col=1,
                    )

                fig.add_trace(go.Scatter(x=stock.index, y=stock["OBV"], name="OBV", mode="lines"), row=2, col=1)

                if show_benchmark_local and not bench.empty:
                    rs_ratio = compute_relative_strength(stock["Close"], bench["Close"])
                    fig.add_trace(go.Scatter(x=rs_ratio.index, y=rs_ratio, name="Stock/Benchmark", mode="lines"), row=3, col=1)
                    fig.add_trace(go.Scatter(x=bench.index, y=bench["Close"], name=f"Benchmark {benchmark_symbol_local}", mode="lines"), row=3, col=1)
                else:
                    fig.add_trace(go.Scatter(x=stock.index, y=stock["RSI14"], name="RSI14", mode="lines"), row=3, col=1)

                fig.add_trace(go.Bar(x=stock.index, y=stock["Volume"], name="Daily Volume"), row=4, col=1)
                fig.add_trace(go.Scatter(x=stock.index, y=stock["VOL_SMA20"], name="Vol SMA20", mode="lines"), row=4, col=1)

                if np.isfinite(float(stock_last["Close"])):
                    fig.add_hline(y=float(stock_last["Close"]), line_width=1.2, line_dash="dash", annotation_text="Current", row=1, col=1)
                if stock_res["decision"] in {"BUY", "STRONG BUY"} and np.isfinite(stock_res["stop_price"]):
                    fig.add_hline(y=float(stock_res["stop_price"]), line_width=1.2, line_dash="dash", annotation_text="Stop", row=1, col=1)
                if stock_res["decision"] in {"BUY", "STRONG BUY"} and np.isfinite(stock_res["entry_price"]):
                    fig.add_hline(y=float(stock_res["entry_price"]), line_width=1.2, line_dash="dash", annotation_text="Entry", row=1, col=1)
                if stock_res["decision"] in {"BUY", "STRONG BUY"} and np.isfinite(stock_res.get("target_1", np.nan)):
                    fig.add_hline(y=float(stock_res["target_1"]), line_width=1.0, line_dash="dot", annotation_text="TP1", row=1, col=1)
                if stock_res["decision"] in {"BUY", "STRONG BUY"} and np.isfinite(stock_res.get("target_2", np.nan)):
                    fig.add_hline(y=float(stock_res["target_2"]), line_width=1.0, line_dash="dot", annotation_text="TP2", row=1, col=1)

                fig.update_layout(height=980, template="plotly_dark", xaxis_rangeslider_visible=False, showlegend=True)
                st.plotly_chart(fig, width="stretch")
                st.markdown("---")
                st.subheader("Entry Plan & Risk")
                plan_cols = st.columns(4)
                plan = stock_res.get("entry_plan", {})
                plan_cols[0].metric(
                    "Entry",
                    f"Rp {plan.get('entry_price_plan', np.nan):,.0f}" if pd.notna(plan.get("entry_price_plan", np.nan)) else "n/a",
                    f"Zone {plan.get('entry_zone_low', np.nan):,.0f} - {plan.get('entry_zone_high', np.nan):,.0f}" if pd.notna(plan.get("entry_zone_low", np.nan)) and pd.notna(plan.get("entry_zone_high", np.nan)) else "Zone n/a",
                )
                plan_cols[1].metric(
                    "Stop Loss",
                    f"Rp {plan.get('stop_loss_plan', np.nan):,.0f}" if pd.notna(plan.get("stop_loss_plan", np.nan)) else "n/a",
                    f"Risk / sh. Rp {plan.get('risk_per_share', np.nan):,.0f}" if pd.notna(plan.get("risk_per_share", np.nan)) else "Risk n/a",
                )
                plan_cols[2].metric(
                    "Target 1",
                    f"Rp {plan.get('target_1', np.nan):,.0f}" if pd.notna(plan.get("target_1", np.nan)) else "n/a",
                    f"RR {plan.get('risk_reward_1', np.nan):.2f}" if pd.notna(plan.get('risk_reward_1', np.nan)) else "RR n/a",
                )
                plan_cols[3].metric(
                    "Target 2",
                    f"Rp {plan.get('target_2', np.nan):,.0f}" if pd.notna(plan.get("target_2", np.nan)) else "n/a",
                    f"RR {plan.get('risk_reward_2', np.nan):.2f}" if pd.notna(plan.get('risk_reward_2', np.nan)) else "RR n/a",
                )
                plan_table = pd.DataFrame(
                    [
                        {"Metric": "Plan Reason", "Value": plan.get("plan_reason", "n/a")},
                        {"Metric": "Entry Zone Low", "Value": f"Rp {plan.get('entry_zone_low', np.nan):,.0f}" if pd.notna(plan.get("entry_zone_low", np.nan)) else "n/a"},
                        {"Metric": "Entry Zone High", "Value": f"Rp {plan.get('entry_zone_high', np.nan):,.0f}" if pd.notna(plan.get("entry_zone_high", np.nan)) else "n/a"},
                        {"Metric": "Entry Trigger", "Value": plan.get("entry_trigger", "n/a")},
                        {"Metric": "Entry Price", "Value": f"Rp {plan.get('entry_price_plan', np.nan):,.0f}" if pd.notna(plan.get("entry_price_plan", np.nan)) else "n/a"},
                        {"Metric": "Stop Loss", "Value": f"Rp {plan.get('stop_loss_plan', np.nan):,.0f}" if pd.notna(plan.get("stop_loss_plan", np.nan)) else "n/a"},
                        {"Metric": "Target 1", "Value": f"Rp {plan.get('target_1', np.nan):,.0f}" if pd.notna(plan.get("target_1", np.nan)) else "n/a"},
                        {"Metric": "Target 2", "Value": f"Rp {plan.get('target_2', np.nan):,.0f}" if pd.notna(plan.get("target_2", np.nan)) else "n/a"},
                        {"Metric": "RR 1", "Value": f"{plan.get('risk_reward_1', np.nan):.2f}" if pd.notna(plan.get("risk_reward_1", np.nan)) else "n/a"},
                        {"Metric": "RR 2", "Value": f"{plan.get('risk_reward_2', np.nan):.2f}" if pd.notna(plan.get("risk_reward_2", np.nan)) else "n/a"},
                        {"Metric": "Upside to TP1", "Value": f"{plan.get('upside_to_t1_pct', np.nan):.2f}%" if pd.notna(plan.get("upside_to_t1_pct", np.nan)) else "n/a"},
                        {"Metric": "Upside to TP2", "Value": f"{plan.get('upside_to_t2_pct', np.nan):.2f}%" if pd.notna(plan.get("upside_to_t2_pct", np.nan)) else "n/a"},
                    ]
                )
                st.dataframe(plan_table, width="stretch", hide_index=True)

                st.markdown("---")
                st.subheader("Detail indikator")
                detail_cols = st.columns(3)
                detail_cols[0].write(f"**Score:** `{stock_res['score']:.2f}`")
                detail_cols[0].write(f"**Core score:** `{stock_res['core_score']:.2f}`")
                detail_cols[0].write(f"**Smart money score:** `{stock_res['smart_money_score']:.2f}`")
                detail_cols[0].write(f"**Decision:** `{stock_res['decision']}`")
                detail_cols[0].write(f"**Dominant cycle:** `{stock_res['dominant_period']} bars`")
                detail_cols[0].write(f"**Time to top / bottom:** `{stock_res.get('time_to_top', np.nan)} / {stock_res['time_to_bottom']} bars`")
                detail_cols[1].write(f"**FVG:** `{stock_res['fvg_present']}`")
                detail_cols[1].write(f"**Order Block:** `{stock_res['ob_present']}`")
                detail_cols[1].write(f"**Reversal score:** `{stock_res['reversal_score']}`")
                detail_cols[1].write(f"**Phase:** `{stock_res['phase']}`")
                detail_cols[1].write(f"**Phase age:** `{stock_res.get('phase_age_bars', np.nan)} bars` ({stock_res.get('phase_age_pct', np.nan):.0f}%)")
                detail_cols[1].write(f"**Cycle reliability:** `{stock_res.get('cycle_reliability', np.nan):.0f}%`")
                detail_cols[1].write(f"**Cycle gate:** `{stock_res.get('cycle_gate_reason', 'OK')}`")
                detail_cols[1].write(f"**Macro gate:** `{stock_res.get('macro_gate_reason', 'OK')}`")
                detail_cols[1].write(f"**Macro score:** `{stock_res.get('macro_score', np.nan):.0f}`")
                detail_cols[2].write(f"**Entry:** `{stock_res['entry_price']:.2f}`" if pd.notna(stock_res["entry_price"]) else "**Entry:** n/a")
                detail_cols[2].write(f"**Stoploss:** `{stock_res['stop_price']:.2f}`" if pd.notna(stock_res["stop_price"]) else "**Stoploss:** n/a")
                detail_cols[2].write(f"**OBV trend:** `{stock_res['obv_trend']}`")
                detail_cols[2].write(f"**Phase confidence:** `{stock_res['phase_confidence']:.0f}%`")
                detail_cols[2].write(f"**PEG:** `{fundamental.get('peg_ratio', np.nan):.2f}`" if pd.notna(fundamental.get("peg_ratio", np.nan)) else "**PEG:** n/a")
                detail_cols[2].write(f"**Revenue QoQ:** `{format_growth_percent(fundamental.get('revenue_growth_qoq', np.nan), decimals=0)}`")
                detail_cols[2].write(f"**Revenue YoY:** `{format_growth_percent(fundamental.get('revenue_growth_yoy', np.nan), decimals=0)}`")
                detail_cols[2].write(f"**Revenue Annual YoY:** `{format_growth_percent(fundamental.get('revenue_growth_annual_yoy', np.nan), decimals=0)}`")
                detail_cols[2].write(f"**Revenue Y/Y Acceleration:** `{format_growth_percent(fundamental.get('revenue_yoy_acceleration', np.nan), decimals=0)}`")
                detail_cols[2].write(f"**Revenue Seasonal QoQ Divergence:** `{format_growth_percent(fundamental.get('revenue_seasonal_qoq_divergence', np.nan), decimals=0)}`")
                detail_cols[2].write(f"**Revenue growth period:** `{fundamental.get('revenue_growth_period', 'n/a')}`")
                detail_cols[2].write(f"**Revenue growth basis:** `{fundamental.get('revenue_growth_basis', 'n/a')}`")
                detail_cols[2].write(f"**Revenue growth source:** `{fundamental.get('revenue_growth_source', 'n/a')}`")
                detail_cols[2].write(f"**Earnings QoQ:** `{format_growth_percent(fundamental.get('earnings_growth_qoq', np.nan), decimals=0)}`")
                detail_cols[2].write(f"**Earnings YoY:** `{format_growth_percent(fundamental.get('earnings_growth_yoy', np.nan), decimals=0)}`")
                detail_cols[2].write(f"**Earnings Annual YoY:** `{format_growth_percent(fundamental.get('earnings_growth_annual_yoy', np.nan), decimals=0)}`")
                detail_cols[2].write(f"**Earnings Y/Y Acceleration:** `{format_growth_percent(fundamental.get('earnings_yoy_acceleration', np.nan), decimals=0)}`")
                detail_cols[2].write(f"**Earnings Seasonal QoQ Divergence:** `{format_growth_percent(fundamental.get('earnings_seasonal_qoq_divergence', np.nan), decimals=0)}`")
                detail_cols[2].write(f"**Earnings growth period:** `{fundamental.get('earnings_growth_period', 'n/a')}`")
                detail_cols[2].write(f"**Earnings growth basis:** `{fundamental.get('earnings_growth_basis', 'n/a')}`")
                detail_cols[2].write(f"**Earnings growth source:** `{fundamental.get('earnings_growth_source', 'n/a')}`")
                detail_cols[2].write(f"**Expected Revenue Next Q:** `{format_growth_percent(future_context.get('expected_revenue_growth_next_q', np.nan), decimals=0)}`")
                detail_cols[2].write(f"**Expected EPS Next Q:** `{format_growth_percent(future_context.get('expected_eps_growth_next_q', np.nan), decimals=0)}`")
                detail_cols[2].write(f"**Expected Margin Next Q:** `{format_growth_percent(future_context.get('expected_margin_next_q', np.nan), decimals=0)}`")
                detail_cols[2].write(f"**Current fundamental grade:** `{fundamental.get('fundamental_grade', 'n/a')}`")
                detail_cols[2].write(f"**Current fundamental score:** `{fundamental.get('fundamental_score', np.nan):.2f}`" if pd.notna(fundamental.get("fundamental_score", np.nan)) else "**Current fundamental score:** n/a")
                detail_cols[2].write("---")
                detail_cols[2].write(f"**Future fundamental grade:** `{stock_res.get('future_fundamental_grade', 'n/a')}`")
                detail_cols[2].write(f"**Future fundamental score:** `{stock_res.get('future_fundamental_score', np.nan):.2f}`" if pd.notna(stock_res.get("future_fundamental_score", np.nan)) else "**Future fundamental score:** n/a")
                detail_cols[2].write(f"**Future direction:** `{stock_res.get('future_fundamental_direction', 'n/a')}`")
                detail_cols[2].write(f"**Future confidence:** `{stock_res.get('future_fundamental_confidence', np.nan):.0f}%`" if pd.notna(stock_res.get("future_fundamental_confidence", np.nan)) else "**Future confidence:** n/a")
                detail_cols[2].write(f"**Future phase:** `{stock_res.get('future_fundamental_phase', 'Unknown')}`")
                detail_cols[2].write(f"**Future reason:** `{stock_res.get('future_fundamental_reason', 'n/a')}`")
                if pd.notna(fundamental.get("fundamental_score", np.nan)) and pd.notna(stock_res.get("future_fundamental_score", np.nan)):
                    divergence = float(stock_res.get("future_fundamental_score", np.nan)) - float(fundamental.get("fundamental_score", np.nan))
                    detail_cols[2].write(f"**Score delta:** `{divergence:+.2f}` pts")
                detail_cols[2].write(f"**Notes:** `{stock_res['notes']}`")
        else:
            st.info("Masukkan ticker lalu klik **Analyze ticker** untuk membuka deep dive.")

    analysis = st.session_state.get("ifs_analysis", {})
    ifs_context = analysis.get("ifs_context", {})
    stock_res = analysis.get("stock_res", {})
    fundamental = analysis.get("fundamental", {})
    future_context = analysis.get("future_context", {})
    stock_df = analysis.get("stock_df", pd.DataFrame())
    bench_df = analysis.get("bench_df", pd.DataFrame())
    selected_symbol = analysis.get("symbol", normalize_ticker(ticker_input))

    with sub_overview:
        st.subheader("Overview Ranking")
        valid_results = st.session_state.get("global_valid_results", [])
        if valid_results:
            rows = []
            for r in valid_results:
                ifs_score = _safe_float(r.get("ifs_score"), np.nan)
                rows.append(
                    {
                        "Rank": 0,
                        "Ticker": r.get("symbol", "-"),
                        "IFS": round(ifs_score, 2) if pd.notna(ifs_score) else np.nan,
                        "Grade": r.get("ifs_grade", "n/a"),
                        "Forward": round(_safe_float(r.get("ifs_breakdown", {}).get("Forward Fundamental"), np.nan), 1) if isinstance(r.get("ifs_breakdown", {}), dict) else np.nan,
                        "Accum": round(_safe_float(r.get("ifs_breakdown", {}).get("Accumulation"), np.nan), 1) if isinstance(r.get("ifs_breakdown", {}), dict) else np.nan,
                        "RS": round(_safe_float(r.get("ifs_breakdown", {}).get("Relative Strength"), np.nan), 1) if isinstance(r.get("ifs_breakdown", {}), dict) else np.nan,
                        "Qual": round(_safe_float(r.get("ifs_breakdown", {}).get("Quality"), np.nan), 1) if isinstance(r.get("ifs_breakdown", {}), dict) else np.nan,
                        "Catalyst": round(_safe_float(r.get("ifs_breakdown", {}).get("Catalyst"), np.nan), 1) if isinstance(r.get("ifs_breakdown", {}), dict) else np.nan,
                        "Decision": r.get("decision", "-"),
                        "MarketStruct": round(_safe_float(r.get("market_structure_score"), np.nan), 1) if pd.notna(r.get("market_structure_score", np.nan)) else np.nan,
                    }
                )
            ov_df = pd.DataFrame(rows).sort_values(["IFS", "MarketStruct"], ascending=[False, False], na_position="last").reset_index(drop=True)
            ov_df["Rank"] = np.arange(1, len(ov_df) + 1)
            st.dataframe(ov_df.head(20), width="stretch", hide_index=True)
        else:
            st.info("Jalankan global scan terlebih dahulu agar ranking IFS muncul di sini.")

    with sub_factor:
        st.subheader(f"Factor Breakdown — {selected_symbol}")
        if ifs_context:
            factor_df = pd.DataFrame(
                [
                    {"Factor": k, "Score": round(v, 2)}
                    for k, v in ifs_context.get("ifs_breakdown", {}).items()
                ]
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("IFS", f'{ifs_context.get("ifs_score", np.nan):.2f}', ifs_context.get("ifs_grade", "n/a"))
            c2.metric("Forward Direction", ifs_context.get("ifs_detail", {}).get("future_direction", "n/a"))
            c3.metric("Confidence", f'{ifs_context.get("ifs_detail", {}).get("future_confidence", np.nan):.0f}%')
            st.dataframe(factor_df, width="stretch", hide_index=True)
        else:
            st.info("Klik Analyze ticker untuk melihat breakdown faktor IFS.")

    with sub_smart:
        st.subheader("Smart Money")
        if ifs_context is not None and stock_res is not None:
            sm_cols = st.columns(4)
            sm_cols[0].metric("Smart Money Score", f'{ifs_context.get("ifs_detail", {}).get("smart_money_score", np.nan):.2f}')
            sm_cols[1].metric("Accumulation", f'{ifs_context.get("ifs_detail", {}).get("accumulation_score", np.nan):.2f}')
            sm_cols[2].metric("CMF20", f'{stock_res.get("cmf20", np.nan):.2f}' if pd.notna(stock_res.get("cmf20", np.nan)) else "n/a")
            sm_cols[3].metric("Unicorn", stock_res.get("unicorn_setup_status", "n/a"))
            smart_table = pd.DataFrame(
                [
                    {"Metric": "OBV Trend", "Value": stock_res.get("obv_trend", "n/a")},
                    {"Metric": "OBV Slope", "Value": f'{stock_res.get("obv_slope10", np.nan):.2f}' if pd.notna(stock_res.get("obv_slope10", np.nan)) else "n/a"},
                    {"Metric": "CMF20", "Value": f'{stock_res.get("cmf20", np.nan):.2f}' if pd.notna(stock_res.get("cmf20", np.nan)) else "n/a"},
                    {"Metric": "MFI14", "Value": f'{stock_res.get("mfi14", np.nan):.2f}' if pd.notna(stock_res.get("mfi14", np.nan)) else "n/a"},
                    {"Metric": "Smart Money Score", "Value": f'{ifs_context.get("ifs_detail", {}).get("smart_money_score", np.nan):.2f}'},
                    {"Metric": "FVG Age (bars)", "Value": f'{stock_res.get("fvg_age_bars", np.nan):.0f}' if pd.notna(stock_res.get("fvg_age_bars", np.nan)) else "n/a"},
                    {"Metric": "FVG Status", "Value": stock_res.get("fvg_status", "n/a")},
                    {"Metric": "Unicorn", "Value": "YES" if stock_res.get("unicorn_setup", False) else "NO"},
                    {"Metric": "Unicorn Valid", "Value": "YES" if stock_res.get("unicorn_setup_valid", False) else "NO"},
                    {"Metric": "Unicorn State", "Value": stock_res.get("unicorn_setup_state", "n/a")},
                    {"Metric": "Unicorn Status", "Value": stock_res.get("unicorn_setup_status", "n/a")},
                    {"Metric": "Sniper", "Value": "YES" if stock_res.get("unicorn_sniper", False) else "NO"},
                    {"Metric": "Sniper Valid", "Value": "YES" if stock_res.get("unicorn_sniper_valid", False) else "NO"},
                    {"Metric": "Sniper State", "Value": stock_res.get("unicorn_sniper_state", "n/a")},
                    {"Metric": "Sniper Status", "Value": stock_res.get("unicorn_sniper_status", "n/a")},
                    {"Metric": "Accumulation Score", "Value": f'{ifs_context.get("ifs_detail", {}).get("accumulation_score", np.nan):.2f}'},
                ]
            )
            st.dataframe(smart_table, width="stretch", hide_index=True)
        else:
            st.info("Belum ada hasil analisis untuk Smart Money.")

    with sub_forward:
        st.subheader("Forward Fundamental")
        if ifs_context and fundamental:
            current_score = _safe_float(fundamental.get("fundamental_score"), np.nan)
            future_score = _safe_float(future_context.get("future_fundamental_score"), np.nan)
            divergence = future_score - current_score if np.isfinite(current_score) and np.isfinite(future_score) else np.nan

            cols = st.columns(4)
            cols[0].metric("Current Facts", f'{current_score:.2f}' if pd.notna(current_score) else "n/a", fundamental.get("fundamental_grade", "n/a"))
            cols[1].metric("Model Forecast", f'{future_score:.2f}' if pd.notna(future_score) else "n/a", future_context.get("future_fundamental_grade", "n/a"))
            cols[2].metric("Score Delta", format_score_delta(divergence), "Forecast - Current")
            cols[3].metric("Model Confidence", f'{future_context.get("future_fundamental_confidence", np.nan):.0f}%' if pd.notna(future_context.get("future_fundamental_confidence", np.nan)) else "n/a")

            c_left, c_right = st.columns(2)

            with c_left:
                st.markdown("**Current Facts**")
                current_table = pd.DataFrame(
                    [
                        {"Metric": "Revenue QoQ", "Value": format_growth_percent(fundamental.get("revenue_growth_qoq", np.nan), decimals=0)},
                        {"Metric": "Revenue YoY", "Value": format_growth_percent(fundamental.get("revenue_growth_yoy", np.nan), decimals=0)},
                        {"Metric": "Revenue Annual YoY", "Value": format_growth_percent(fundamental.get("revenue_growth_annual_yoy", np.nan), decimals=0)},
                        {"Metric": "Revenue Y/Y Acceleration", "Value": format_growth_percent(fundamental.get("revenue_yoy_acceleration", np.nan), decimals=0)},
                        {"Metric": "Revenue Seasonal QoQ Div.", "Value": format_growth_percent(fundamental.get("revenue_seasonal_qoq_divergence", np.nan), decimals=0)},
                        {"Metric": "Earnings QoQ", "Value": format_growth_percent(fundamental.get("earnings_growth_qoq", np.nan), decimals=0)},
                        {"Metric": "Earnings YoY", "Value": format_growth_percent(fundamental.get("earnings_growth_yoy", np.nan), decimals=0)},
                        {"Metric": "Earnings Annual YoY", "Value": format_growth_percent(fundamental.get("earnings_growth_annual_yoy", np.nan), decimals=0)},
                        {"Metric": "Earnings Y/Y Acceleration", "Value": format_growth_percent(fundamental.get("earnings_yoy_acceleration", np.nan), decimals=0)},
                        {"Metric": "Earnings Seasonal QoQ Div.", "Value": format_growth_percent(fundamental.get("earnings_seasonal_qoq_divergence", np.nan), decimals=0)},
                        {"Metric": "Revenue Period", "Value": fundamental.get("revenue_growth_period", "n/a")},
                        {"Metric": "Revenue Basis", "Value": fundamental.get("revenue_growth_basis", "n/a")},
                        {"Metric": "Revenue Source", "Value": fundamental.get("revenue_growth_source", "n/a")},
                        {"Metric": "Earnings Period", "Value": fundamental.get("earnings_growth_period", "n/a")},
                        {"Metric": "Earnings Basis", "Value": fundamental.get("earnings_growth_basis", "n/a")},
                        {"Metric": "Earnings Source", "Value": fundamental.get("earnings_growth_source", "n/a")},
                        {"Metric": "PEG", "Value": f'{fundamental.get("peg_ratio", np.nan):.2f}' if pd.notna(fundamental.get("peg_ratio", np.nan)) else "n/a"},
                        {"Metric": "Current Fundamental Grade", "Value": fundamental.get("fundamental_grade", "n/a")},
                        {"Metric": "Fundamental Data Source", "Value": fundamental.get("fundamental_data_source", "n/a")},
                        {"Metric": "Data Quality Flag", "Value": fundamental.get("data_quality_flag", "n/a")},
                    ]
                )
                st.dataframe(current_table, width="stretch", hide_index=True)

            with c_right:
                st.markdown("**Model Forecast**")
                future_table = pd.DataFrame(
                    [
                        {"Metric": "Future Phase", "Value": future_context.get("future_phase", "Unknown")},
                        {"Metric": "Future Direction", "Value": future_context.get("future_fundamental_direction", "n/a")},
                        {"Metric": "Expected Revenue Next Q", "Value": format_growth_percent(future_context.get("expected_revenue_growth_next_q", np.nan), decimals=0)},
                        {"Metric": "Expected EPS Next Q", "Value": format_growth_percent(future_context.get("expected_eps_growth_next_q", np.nan), decimals=0)},
                        {"Metric": "Expected Margin Next Q", "Value": format_growth_percent(future_context.get("expected_margin_next_q", np.nan), decimals=0)},
                        {"Metric": "Future Reason", "Value": future_context.get("future_moat_reason", "n/a")},
                        {"Metric": "Future Macro Gate", "Value": future_context.get("future_macro_gate_reason", "OK")},
                        {"Metric": "Future Macro Adjusted", "Value": f'{future_context.get("future_macro_adjusted_score", np.nan):.2f}' if pd.notna(future_context.get("future_macro_adjusted_score", np.nan)) else "n/a"},
                    ]
                )
                st.dataframe(future_table, width="stretch", hide_index=True)

            st.markdown("**Explainability**")
            explain_table = pd.DataFrame(
                [
                    {"Component": "Forward Fundamental", "Value": f'{future_context.get("future_fundamental_score", np.nan):.2f}' if pd.notna(future_context.get("future_fundamental_score", np.nan)) else "n/a"},
                    {"Component": "Fundamental Momentum", "Value": f'{future_context.get("future_fundamental_momentum_score", np.nan):.2f}' if pd.notna(future_context.get("future_fundamental_momentum_score", np.nan)) else "n/a"},
                    {"Component": "Seasonal Anomaly", "Value": f'{future_context.get("future_seasonal_anomaly_score", np.nan):.2f}' if pd.notna(future_context.get("future_seasonal_anomaly_score", np.nan)) else "n/a"},
                    {"Component": "Inflection Score", "Value": f'{future_context.get("future_inflection_score", np.nan):.2f}' if pd.notna(future_context.get("future_inflection_score", np.nan)) else "n/a"},
                    {"Component": "Growth Proxy", "Value": f'{future_context.get("future_growth_proxy", np.nan):.2f}' if pd.notna(future_context.get("future_growth_proxy", np.nan)) else "n/a"},
                    {"Component": "Cash Flow Proxy", "Value": f'{future_context.get("future_cash_flow_proxy", np.nan):.2f}' if pd.notna(future_context.get("future_cash_flow_proxy", np.nan)) else "n/a"},
                    {"Component": "Balance Quality", "Value": f'{future_context.get("future_balance_quality", np.nan):.2f}' if pd.notna(future_context.get("future_balance_quality", np.nan)) else "n/a"},
                    {"Component": "Price Proxy", "Value": f'{future_context.get("future_price_proxy", np.nan):.2f}' if pd.notna(future_context.get("future_price_proxy", np.nan)) else "n/a"},
                    {"Component": "Cycle Support", "Value": f'{future_context.get("future_cycle_support", np.nan):.2f}' if pd.notna(future_context.get("future_cycle_support", np.nan)) else "n/a"},
                    {"Component": "Future Reliability", "Value": f'{future_context.get("future_reliability", np.nan):.2f}' if pd.notna(future_context.get("future_reliability", np.nan)) else "n/a"},
                ]
            )
            st.dataframe(explain_table, width="stretch", hide_index=True)
        else:
            st.info("Klik Analyze ticker untuk melihat forward fundamental.")