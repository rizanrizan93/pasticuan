import re
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - optional but available in requirements
    BeautifulSoup = None

try:
    from data_engine import _ticker_candidates as _data_engine_ticker_candidates
except Exception:
    _data_engine_ticker_candidates = None

try:
    from technical_analyst import _ensure_technical_columns, classify_8_phase, compute_cycle_features, _score_bucket
except Exception:
    def _ensure_technical_columns(df: pd.DataFrame) -> pd.DataFrame:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()

    def classify_8_phase(df: pd.DataFrame) -> dict:
        return {"phase": "Unknown", "phase_confidence": 0.0}

    def compute_cycle_features(close_series):
        return (20, 999, False, {})

    def _score_bucket(value: float, lo: float, hi: float, invert: bool = False) -> float:
        if value is None or pd.isna(value):
            return 50.0
        if hi == lo:
            return 50.0
        x = (float(value) - lo) / (hi - lo)
        x = float(np.clip(x, 0.0, 1.0))
        if invert:
            x = 1.0 - x
        return float(np.clip(x * 100.0, 0.0, 100.0))


def _ticker_candidates(symbol: str) -> list[str]:
    if callable(_data_engine_ticker_candidates):
        try:
            out = list(_data_engine_ticker_candidates(symbol))
            if out:
                return out
        except Exception:
            pass
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

try:
    from data_engine import load_yf_info as _data_engine_load_yf_info
except Exception:
    _data_engine_load_yf_info = None

@st.cache_data(ttl=21600, show_spinner=False)
def load_yf_info(symbol: str) -> dict:
    if callable(_data_engine_load_yf_info):
        try:
            out = _data_engine_load_yf_info(symbol)
            return out if isinstance(out, dict) else {}
        except Exception:
            return {}
    return {}


def _is_probably_indonesian_symbol(symbol: str) -> bool:
    base = str(symbol or "").strip().upper()
    if not base or base.startswith("^"):
        return False
    return base.endswith(".JK") or ":IDX" in base or ":JKT" in base or ":JAKARTA" in base


def _google_finance_candidates(symbol: str) -> list[str]:
    base = str(symbol or "").strip().upper()
    if not base or base == "NAN":
        return []
    core = base
    if ":" in core:
        core = core.split(":", 1)[0]
    if core.endswith(".JK"):
        core = core[:-3]

    candidates: list[str] = []

    def add(value: str) -> None:
        value = str(value or "").strip().upper()
        if value and value not in candidates:
            candidates.append(value)

    add(base)
    if core:
        add(f"{core}:IDX")
        add(f"{core}:JKT")
        add(f"{core}:JAKARTA")
        add(core)
    return candidates


def _parse_google_numeric(value: str):
    if value is None:
        return np.nan
    s = str(value).strip()
    if not s or s.upper() in {"N/A", "NA", "-", "—", "–"}:
        return np.nan
    s = s.replace(" ", " ")
    s = re.sub(r"^[A-Z]{1,3}\$", "", s)
    s = s.replace("Rp", "").replace("IDR", "").replace("USD", "").replace("US$", "")
    s = s.replace(",", "")
    s = s.replace(" ", "")
    m = re.search(r"([+-]?\d+(?:\.\d+)?)([KMBTkmbt%]?)", s)
    if not m:
        return np.nan
    try:
        num = float(m.group(1))
        suffix = m.group(2).upper()
        mult = {"": 1.0, "%": 0.01, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}.get(suffix, 1.0)
        return float(num * mult)
    except Exception:
        return np.nan


def _extract_label_value(flat_text: str, labels: list[str], max_window: int = 160):
    if not flat_text:
        return np.nan
    text = re.sub(r"\s+", " ", flat_text)
    lowered = text.lower()
    for label in labels:
        label = str(label or "").strip()
        if not label:
            continue
        idx = lowered.find(label.lower())
        if idx < 0:
            continue
        snippet = text[idx + len(label): idx + len(label) + max_window]
        snippet = snippet.lstrip(" :\t\n\r-–—|")
        if not snippet:
            continue
        candidates = re.findall(r"[A-Za-z]{0,3}\$?\d[\d.,]*\s*[KMBTkmbt%]?|\d+(?:\.\d+)?%|\d+(?:\.\d+)?", snippet)
        for candidate in candidates:
            parsed = _parse_google_numeric(candidate)
            if np.isfinite(parsed):
                return parsed
    return np.nan


def _extract_label_text(flat_text: str, labels: list[str], max_window: int = 160):
    if not flat_text:
        return ""
    text = re.sub(r"\s+", " ", flat_text)
    lowered = text.lower()
    for label in labels:
        label = str(label or "").strip()
        if not label:
            continue
        idx = lowered.find(label.lower())
        if idx < 0:
            continue
        snippet = text[idx + len(label): idx + len(label) + max_window]
        snippet = snippet.lstrip(" :\t\n\r-–—|")
        if snippet:
            return snippet.split(" ", 1)[0].strip()
    return ""


def _google_finance_url(symbol: str) -> str:
    return f"https://www.google.com/finance/quote/{symbol}?hl=en-US&gl=ID"


def _normalize_yahoo_value(value):
    """Normalize Yahoo Finance JSON/scalar wrappers into plain Python values."""
    if value is None:
        return np.nan
    if isinstance(value, dict):
        for key in ("raw", "fmt", "longFmt", "shortFmt"):
            if key in value and value[key] not in (None, ""):
                return value[key]
        for key in ("value", "amount"):
            if key in value and value[key] not in (None, ""):
                return value[key]
        if len(value) == 1:
            return _normalize_yahoo_value(next(iter(value.values())))
        return value
    if isinstance(value, list):
        if not value:
            return np.nan
        if len(value) == 1:
            return _normalize_yahoo_value(value[0])
        return [_normalize_yahoo_value(v) for v in value]
    return value


def _records_to_statement_frame(records) -> pd.DataFrame:
    """Convert Yahoo statement history records into a yfinance-like DataFrame."""
    if not records:
        return pd.DataFrame()

    rows = []
    columns = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        flat: dict[str, Any] = {}
        period = None
        for key, value in rec.items():
            norm = _normalize_yahoo_value(value)
            if key in {"endDate", "asOfDate", "startDate"}:
                period = norm if period is None else period
                continue
            if key in {"maxAge", "periodType", "currencyCode"}:
                continue
            flat[key] = norm
        if period is None:
            period = _normalize_yahoo_value(rec.get("endDate") or rec.get("asOfDate") or rec.get("startDate"))
        try:
            if isinstance(period, (int, float, np.integer, np.floating)) and np.isfinite(float(period)):
                period = pd.to_datetime(float(period), unit="s", errors="coerce")
            else:
                period = pd.to_datetime(period, errors="coerce")
        except Exception:
            period = pd.NaT
        if pd.isna(period):
            period = f"period_{len(columns)}"
        rows.append(flat)
        columns.append(period)

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows, index=columns).T
    try:
        parsed = pd.to_datetime(frame.columns, errors="coerce")
        if parsed.notna().sum() >= 2:
            ordered = frame.loc[:, parsed.notna()].copy()
            ordered.columns = parsed[parsed.notna()]
            ordered = ordered.sort_index(axis=1)
            return ordered
    except Exception:
        pass
    return frame


def _extract_yahoo_section_values(section: dict, keys: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not isinstance(section, dict):
        return out
    for key in keys:
        value = section.get(key)
        if value is None:
            continue
        norm = _normalize_yahoo_value(value)
        if norm is not None and not (isinstance(norm, float) and pd.isna(norm)):
            out[key] = norm
    return out


def _yahoo_quote_summary_bundle(symbol: str) -> dict:
    """Best-effort raw Yahoo Finance JSON fallback for financial statements."""
    base = str(symbol or "").strip().upper()
    if not base:
        return {}

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9,id;q=0.8",
    }

    modules = [
        "price",
        "summaryDetail",
        "defaultKeyStatistics",
        "financialData",
        "quoteType",
        "assetProfile",
        "incomeStatementHistory",
        "incomeStatementHistoryQuarterly",
        "balanceSheetHistory",
        "balanceSheetHistoryQuarterly",
        "cashflowStatementHistory",
        "cashflowStatementHistoryQuarterly",
    ]

    for candidate in _ticker_candidates(base):
        try:
            url = f"https://query2.finance.yahoo.com/v10/finance/quoteSummary/{candidate}"
            resp = requests.get(
                url,
                params={"modules": ",".join(modules), "formatted": "false", "lang": "en-US", "region": "US"},
                headers=headers,
                timeout=12,
            )
            if resp.status_code != 200 or not resp.text:
                continue
            payload = resp.json() if hasattr(resp, "json") else {}
            result = payload.get("quoteSummary", {}).get("result", [])
            if not result:
                continue
            root = result[0] if isinstance(result, list) and result else {}
            if not isinstance(root, dict) or not root:
                continue

            info: dict[str, Any] = {}
            info.update(_extract_yahoo_section_values(root.get("price", {}), [
                "marketCap", "trailingPE", "forwardPE", "beta", "currency", "financialCurrency",
                "regularMarketPrice", "currentPrice", "quoteType", "longName", "shortName",
                "exchange", "symbol", "sharesOutstanding", "trailingAnnualDividendYield",
            ]))
            info.update(_extract_yahoo_section_values(root.get("summaryDetail", {}), [
                "trailingPE", "forwardPE", "beta", "dividendYield", "marketCap", "trailingEps",
                "profitMargins", "52WeekChange", "fiveYearAvgDividendYield", "regularMarketPrice",
            ]))
            info.update(_extract_yahoo_section_values(root.get("defaultKeyStatistics", {}), [
                "marketCap", "pegRatio", "sharesOutstanding", "enterpriseValue", "trailingEps",
                "forwardPE", "trailingPE", "priceToSalesTrailing12Months", "priceToBook",
            ]))
            info.update(_extract_yahoo_section_values(root.get("financialData", {}), [
                "currentRatio", "debtToEquity", "returnOnEquity", "returnOnAssets",
                "operatingMargins", "grossMargins", "freeCashflow", "operatingCashflow",
                "profitMargins", "ebitdaMargins", "totalCash", "totalDebt", "targetMeanPrice",
            ]))
            info.update(_extract_yahoo_section_values(root.get("quoteType", {}), [
                "quoteType", "longName", "shortName", "exchange", "symbol",
            ]))
            info.update(_extract_yahoo_section_values(root.get("assetProfile", {}), [
                "industry", "sector", "longBusinessSummary", "fullTimeEmployees", "country",
            ]))

            def _records_for(module_name: str, key_name: str):
                module = root.get(module_name, {})
                if not isinstance(module, dict):
                    return []
                data = module.get(key_name, [])
                return data if isinstance(data, list) else []

            income_annual = _records_to_statement_frame(_records_for("incomeStatementHistory", "incomeStatementHistory"))
            income_quarterly = _records_to_statement_frame(_records_for("incomeStatementHistoryQuarterly", "incomeStatementHistoryQuarterly"))
            balance_annual = _records_to_statement_frame(_records_for("balanceSheetHistory", "balanceSheetStatements"))
            balance_quarterly = _records_to_statement_frame(_records_for("balanceSheetHistoryQuarterly", "balanceSheetStatements"))
            cash_annual = _records_to_statement_frame(_records_for("cashflowStatementHistory", "cashflowStatements"))
            cash_quarterly = _records_to_statement_frame(_records_for("cashflowStatementHistoryQuarterly", "cashflowStatements"))

            info["_resolved_symbol"] = candidate
            info["_info_source"] = "yahoo_quote_summary"
            return {
                "info": info,
                "income_annual": income_annual,
                "income_quarterly": income_quarterly,
                "balance_annual": balance_annual,
                "balance_quarterly": balance_quarterly,
                "cash_annual": cash_annual,
                "cash_quarterly": cash_quarterly,
                "_resolved_symbol": candidate,
                "_info_source": "yahoo_quote_summary",
            }
        except Exception:
            continue

    return {}


def _statement_frame_from_ticker_with_legacy(ticker, method_name: str, attempts: list[dict[str, Any]]) -> pd.DataFrame:
    fn = getattr(ticker, method_name, None)
    if not callable(fn):
        return pd.DataFrame()
    for kwargs in attempts:
        try:
            obj = fn(**kwargs)
        except TypeError:
            continue
        except Exception:
            continue
        if isinstance(obj, pd.DataFrame) and not obj.empty:
            frame = obj.copy()
            try:
                parsed = pd.to_datetime(frame.columns, errors="coerce")
                if parsed.notna().sum() >= 2:
                    ordered = frame.loc[:, parsed.notna()].copy()
                    ordered.columns = parsed[parsed.notna()]
                    ordered = ordered.sort_index(axis=1)
                    return ordered
            except Exception:
                pass
            return frame
    return pd.DataFrame()


@st.cache_data(ttl=21600, show_spinner=False)
def load_google_finance_info(symbol: str) -> dict:
    """Best-effort Google Finance metadata fallback for Indonesian equities.

    Google Finance does not expose full statement data, but it often carries a
    few useful valuation fields when Yahoo crumb / rate limits fail. This helper
    is intentionally conservative and returns an empty dict on any failure.
    """
    out: dict[str, Any] = {}
    base = str(symbol or "").strip().upper()
    if not base:
        return out

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9,id;q=0.8",
    }

    for candidate in _google_finance_candidates(base):
        try:
            url = _google_finance_url(candidate)
            resp = requests.get(url, headers=headers, timeout=12)
            if resp.status_code != 200 or not resp.text:
                continue

            soup = BeautifulSoup(resp.text, "html.parser") if BeautifulSoup is not None else None
            if soup is not None:
                flat_text = soup.get_text(" ", strip=True)
            else:
                flat_text = re.sub(r"<[^>]+>", " ", resp.text)

            if not flat_text:
                continue

            # Valuation / quote fields that are sometimes exposed on the page.
            market_cap = _extract_label_value(flat_text, ["Market cap", "Kapitalisasi pasar"])
            pe_ratio = _extract_label_value(flat_text, ["P/E ratio", "PE ratio", "P/E"])
            eps = _extract_label_value(flat_text, ["EPS", "Earnings per share"])
            beta = _extract_label_value(flat_text, ["Beta"])
            dividend_yield = _extract_label_value(flat_text, ["Dividend yield", "Imbal hasil dividen"])
            shares_outstanding = _extract_label_value(flat_text, ["Shares outstanding", "Saham beredar"])
            currency = _extract_label_text(flat_text, ["Currency in", "Mata uang"])

            if np.isfinite(market_cap):
                out["marketCap"] = market_cap
            if np.isfinite(pe_ratio):
                out["trailingPE"] = pe_ratio
                out["forwardPE"] = pe_ratio
            if np.isfinite(eps):
                out["trailingEps"] = eps
                out["epsTrailingTwelveMonths"] = eps
            if np.isfinite(beta):
                out["beta"] = beta
            if np.isfinite(dividend_yield):
                out["dividendYield"] = dividend_yield
            if np.isfinite(shares_outstanding):
                out["sharesOutstanding"] = shares_outstanding
            if currency:
                out["currency"] = currency.upper()
                out["financialCurrency"] = currency.upper()
                out["quoteCurrency"] = currency.upper()

            if out:
                out["_resolved_symbol"] = candidate
                out["_info_source"] = "google_finance"
                return out
        except Exception:
            continue

    return out


def _coerce_float(value, default=np.nan):
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default

def _pick_info_value(info: dict, *keys):
    if not isinstance(info, dict) or not info:
        return np.nan
    for key in keys:
        value = info.get(key)
        if value is None or value == "":
            continue
        try:
            return float(value)
        except Exception:
            continue
    return np.nan

def _statement_frame(ticker: yf.Ticker, attr_names: list[str]) -> pd.DataFrame:
    for attr in attr_names:
        try:
            obj = getattr(ticker, attr, None)
            if obj is None:
                continue
            if callable(obj):
                obj = obj()
            if isinstance(obj, pd.DataFrame) and not obj.empty:
                frame = obj.copy()
                # Yahoo financial statement columns are often newest-first.
                # Sort them chronologically so growth calculations always use
                # the last two periods in true time order.
                try:
                    parsed = pd.to_datetime(frame.columns, errors="coerce")
                    if parsed.notna().sum() >= 2:
                        ordered = frame.loc[:, parsed.notna()].copy()
                        ordered.columns = parsed[parsed.notna()]
                        ordered = ordered.sort_index(axis=1)
                        return ordered
                except Exception:
                    pass
                return frame
        except Exception:
            continue
    return pd.DataFrame()

def _statement_row_series(frame: pd.DataFrame, row_names: list[str]) -> pd.Series | None:
    if frame is None or frame.empty:
        return None

    wanted = [str(name).strip().lower() for name in row_names if str(name).strip()]
    if not wanted:
        return None

    for idx in frame.index:
        label = str(idx).strip().lower()
        if any(name in label for name in wanted):
            try:
                row = pd.to_numeric(frame.loc[idx], errors="coerce")
            except Exception:
                continue
            if isinstance(row, pd.Series):
                row = row.dropna()
                if not row.empty:
                    # Keep only values in chronological order if possible.
                    try:
                        parsed = pd.to_datetime(row.index, errors="coerce")
                        if parsed.notna().sum() >= 2:
                            ordered = row.loc[parsed.notna()].copy()
                            ordered.index = parsed[parsed.notna()]
                            ordered = ordered.sort_index()
                            return ordered
                    except Exception:
                        pass
                    return row
    return None

def _statement_scalar(frame: pd.DataFrame, row_names: list[str], position: int = 0) -> float:
    row = _statement_row_series(frame, row_names)
    if row is None or row.empty:
        return np.nan
    vals = row.dropna().to_list()
    if len(vals) <= position:
        return np.nan
    # position=0 means latest available value.
    return _coerce_float(vals[-1 - position])

def _statement_growth(frame: pd.DataFrame, row_names: list[str]) -> float:
    row = _statement_row_series(frame, row_names)
    if row is None or row.empty:
        return np.nan
    vals = [v for v in row.dropna().to_list() if np.isfinite(_coerce_float(v))]
    if len(vals) < 2:
        return np.nan
    latest = _coerce_float(vals[-1])
    prev = _coerce_float(vals[-2])
    if not np.isfinite(latest) or not np.isfinite(prev) or abs(prev) < 1e-12:
        return np.nan
    return (latest / prev) - 1.0

def _prepare_statement_series(series: pd.Series | None) -> pd.Series:
    """Normalize financial statement series into ascending timestamp order.

    The helper is intentionally strict: if the index cannot be interpreted as
    dates, the caller still gets a numeric series, but any time-based YoY logic
    will gracefully fall back to positional comparisons.
    """
    if series is None:
        return pd.Series(dtype=float)
    try:
        s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    except Exception:
        return pd.Series(dtype=float)
    if s.empty:
        return s

    try:
        parsed = pd.to_datetime(s.index, errors="coerce")
        if getattr(parsed, "notna", lambda: pd.Series(dtype=bool))().sum() >= 2:
            ordered = s.loc[parsed.notna()].copy()
            ordered.index = parsed[parsed.notna()]
            ordered = ordered[~ordered.index.duplicated(keep="last")].sort_index()
            return ordered.astype(float)
    except Exception:
        pass
    return s[~s.index.duplicated(keep="last")].astype(float)

def _growth_from_series(series: pd.Series | None, lag: int) -> float:
    """Compute growth using a lag, with a stricter YoY fallback for dated series.

    lag=1  -> previous observation (QoQ / annual if annual frame)
    lag=4  -> same quarter last year for quarterly statements when dates exist
    """
    s = _prepare_statement_series(series)
    lag = int(max(1, lag))
    if len(s) <= lag:
        return np.nan

    latest = float(s.iloc[-1])
    if not np.isfinite(latest):
        return np.nan

    # For quarterly YoY, try an exact same-quarter match first.
    if lag == 4 and isinstance(s.index, pd.DatetimeIndex):
        latest_idx = s.index[-1]
        target_year = latest_idx.year - 1
        target_quarter = latest_idx.quarter
        exact = s[(s.index.year == target_year) & (s.index.quarter == target_quarter)]
        if not exact.empty:
            prev = float(exact.iloc[-1])
            if np.isfinite(prev) and abs(prev) > 1e-12:
                return (latest - prev) / abs(prev)

    prev = float(s.iloc[-1 - lag])
    if not np.isfinite(prev) or abs(prev) < 1e-12:
        return np.nan
    return (latest - prev) / abs(prev)

def _quarterly_yoy_acceleration(series: pd.Series | None) -> float:
    """Measure whether YoY growth itself is accelerating quarter by quarter."""
    s = _prepare_statement_series(series)
    if len(s) < 5:
        return np.nan

    yoy_vals = []
    if isinstance(s.index, pd.DatetimeIndex):
        for i in range(1, len(s)):
            curr_idx = s.index[i]
            exact = s[(s.index.year == curr_idx.year - 1) & (s.index.quarter == curr_idx.quarter)]
            if not exact.empty:
                prev = float(exact.iloc[-1])
                curr = float(s.iloc[i])
                if np.isfinite(curr) and np.isfinite(prev) and abs(prev) > 1e-12:
                    yoy_vals.append((curr - prev) / abs(prev))
    if len(yoy_vals) < 2 and len(s) >= 5:
        # Positional fallback for incomplete date metadata.
        for i in range(4, len(s)):
            prev = float(s.iloc[i - 4])
            curr = float(s.iloc[i])
            if np.isfinite(curr) and np.isfinite(prev) and abs(prev) > 1e-12:
                yoy_vals.append((curr - prev) / abs(prev))

    if len(yoy_vals) < 2:
        return np.nan
    return float(yoy_vals[-1] - yoy_vals[-2])

def _seasonal_qoq_divergence(series: pd.Series | None) -> float:
    """Compare the current QoQ change against the historical seasonal pattern."""
    s = _prepare_statement_series(series)
    if len(s) < 5 or not isinstance(s.index, pd.DatetimeIndex):
        return np.nan

    current_qoq = _growth_from_series(s, 1)
    if not np.isfinite(current_qoq):
        return np.nan

    latest_q = int(s.index[-1].quarter)
    prev_q = int(s.index[-2].quarter)
    hist = []
    for i in range(1, len(s) - 1):
        if int(s.index[i].quarter) == latest_q and int(s.index[i - 1].quarter) == prev_q:
            prev = float(s.iloc[i - 1])
            curr = float(s.iloc[i])
            if abs(prev) > 1e-12:
                hist.append((curr - prev) / abs(prev))
    if len(hist) < 1:
        return np.nan
    baseline = float(np.nanmedian(hist))
    return float(current_qoq - baseline)

def _growth_bundle_from_frames(quarterly_frame: pd.DataFrame, annual_frame: pd.DataFrame, row_names: list[str]) -> dict:
    quarterly_series = _prepare_statement_series(_statement_row_series(quarterly_frame, row_names))
    annual_series = _prepare_statement_series(_statement_row_series(annual_frame, row_names))

    qoq = _growth_from_series(quarterly_series, 1)
    yoy = _growth_from_series(quarterly_series, 4)
    annual = _growth_from_series(annual_series, 1)
    acceleration = _quarterly_yoy_acceleration(quarterly_series)
    seasonal_divergence = _seasonal_qoq_divergence(quarterly_series)

    primary = np.nan
    basis = "n/a"
    source = "n/a"
    if np.isfinite(yoy):
        primary = yoy
        basis = "Quarterly YoY (same quarter last year)"
        source = "quarterly_income_stmt"
    elif np.isfinite(annual):
        primary = annual
        basis = "Annual YoY (annual statement)"
        source = "income_stmt"
    elif np.isfinite(qoq):
        primary = qoq
        basis = "QoQ (previous quarter)"
        source = "quarterly_income_stmt"

    quality = "missing"
    if np.isfinite(yoy):
        quality = "quarterly_yoy"
    elif np.isfinite(annual):
        quality = "annual_yoy"
    elif np.isfinite(qoq):
        quality = "qoq_only"

    return {
        "primary": primary,
        "qoq": qoq,
        "yoy": yoy,
        "annual": annual,
        "acceleration": acceleration,
        "seasonal_divergence": seasonal_divergence,
        "basis": basis,
        "source": source,
        "quality": quality,
        "quarterly_points": int(len(quarterly_series)),
        "annual_points": int(len(annual_series)),
    }

def _safe_float(v, default=np.nan):

    try:
        if v is None:
            return default
        if isinstance(v, str) and not v.strip():
            return default
        out = float(v)
        return out if np.isfinite(out) else default
    except Exception:
        return default

def format_growth_percent(v, decimals: int = 0) -> str:
    """Format a growth value that may come as 0.18 or 18.0 into 18%."""
    try:
        if v is None or pd.isna(v):
            return "n/a"
        x = float(v)
        if abs(x) <= 1.5:
            x *= 100.0
        return f"{x:.{decimals}f}%"
    except Exception:
        return "n/a"

@st.cache_data(ttl=21600, show_spinner=False)
def load_fundamental_snapshot(symbol: str) -> dict:
    out = {
        "market_cap": np.nan,
        "current_ratio": np.nan,
        "debt_to_equity": np.nan,
        "return_on_equity": np.nan,
        "return_on_assets": np.nan,
        "operating_margin": np.nan,
        "gross_margin": np.nan,
        "free_cashflow": np.nan,
        "operating_cashflow": np.nan,
        "peg_ratio": np.nan,
        "trailing_pe": np.nan,
        "forward_pe": np.nan,
        "revenue_growth": np.nan,
        "earnings_growth": np.nan,
        "profit_margins": np.nan,
        "revenue_growth_quarterly": np.nan,
        "revenue_growth_annual": np.nan,
        "earnings_growth_quarterly": np.nan,
        "earnings_growth_annual": np.nan,
        "revenue_growth_qoq": np.nan,
        "revenue_growth_yoy": np.nan,
        "revenue_growth_annual_yoy": np.nan,
        "earnings_growth_qoq": np.nan,
        "earnings_growth_yoy": np.nan,
        "earnings_growth_annual_yoy": np.nan,
        "revenue_yoy_acceleration": np.nan,
        "earnings_yoy_acceleration": np.nan,
        "revenue_seasonal_qoq_divergence": np.nan,
        "earnings_seasonal_qoq_divergence": np.nan,
        "revenue_growth_period": "n/a",
        "earnings_growth_period": "n/a",
        "revenue_growth_basis": "n/a",
        "earnings_growth_basis": "n/a",
        "revenue_growth_source": "n/a",
        "earnings_growth_source": "n/a",
        "data_quality_flag": "missing",
        "fundamental_data_source": "missing",
        "_resolved_symbol": "",
    }
    base = str(symbol).strip()
    if not base:
        return out

    info = {}
    google_info = {}
    resolved_symbol = base
    candidate_fn = globals().get("_ticker_candidates")
    if not callable(candidate_fn):
        def candidate_fn(symbol: str):
            base_local = str(symbol).strip().upper()
            if not base_local or base_local == "NAN":
                return []
            candidates_local: list[str] = []
            def add(candidate: str) -> None:
                candidate = str(candidate).strip().upper()
                if candidate and candidate not in candidates_local:
                    candidates_local.append(candidate)
            add(base_local)
            if not base_local.startswith("^"):
                if base_local.endswith(".JK"):
                    add(base_local[:-3])
                else:
                    add(f"{base_local}.JK")
            return candidates_local
    for candidate in candidate_fn(base):
        info = load_yf_info(candidate)
        if info:
            resolved_symbol = str(info.get("_resolved_symbol", candidate))
            break

    # Google Finance fallback for Indonesian equities when Yahoo is thin.
    if not info and _is_probably_indonesian_symbol(base):
        google_info = load_google_finance_info(base)
        if google_info:
            resolved_symbol = str(google_info.get("_resolved_symbol", resolved_symbol))
            info = dict(google_info)

    # If Yahoo is present, still opportunistically supplement with Google Finance.
    if _is_probably_indonesian_symbol(resolved_symbol):
        google_info = load_google_finance_info(resolved_symbol)
        if google_info:
            merged = dict(info)
            for key, value in google_info.items():
                if key.startswith("_"):
                    continue
                current_value = merged.get(key)
                if key not in merged or current_value in (None, "") or (isinstance(current_value, float) and pd.isna(current_value)) or pd.isna(current_value):
                    merged[key] = value
            if merged:
                merged.setdefault("_resolved_symbol", google_info.get("_resolved_symbol", resolved_symbol))
                merged.setdefault("_info_source", "mixed")
                info = merged

    out["_resolved_symbol"] = resolved_symbol

    yahoo_bundle = _yahoo_quote_summary_bundle(resolved_symbol if resolved_symbol else base)
    if isinstance(yahoo_bundle, dict) and yahoo_bundle:
        raw_info = yahoo_bundle.get("info", {}) if isinstance(yahoo_bundle.get("info", {}), dict) else {}
        if raw_info:
            if not info:
                info = dict(raw_info)
            else:
                for key, value in raw_info.items():
                    current_value = info.get(key)
                    if current_value in (None, "") or (isinstance(current_value, float) and pd.isna(current_value)) or pd.isna(current_value):
                        info[key] = value
        raw_income_annual = yahoo_bundle.get("income_annual", pd.DataFrame())
        raw_income_quarterly = yahoo_bundle.get("income_quarterly", pd.DataFrame())
        raw_balance_annual = yahoo_bundle.get("balance_annual", pd.DataFrame())
        raw_balance_quarterly = yahoo_bundle.get("balance_quarterly", pd.DataFrame())
        raw_cash_annual = yahoo_bundle.get("cash_annual", pd.DataFrame())
        raw_cash_quarterly = yahoo_bundle.get("cash_quarterly", pd.DataFrame())
    else:
        raw_income_annual = pd.DataFrame()
        raw_income_quarterly = pd.DataFrame()
        raw_balance_annual = pd.DataFrame()
        raw_balance_quarterly = pd.DataFrame()
        raw_cash_annual = pd.DataFrame()
        raw_cash_quarterly = pd.DataFrame()

    # --- Currency interceptor: bypass USD-denominated financials ---
    financial_currency = str(info.get("financialCurrency") or info.get("currency") or info.get("quoteCurrency") or "IDR").upper()
    if financial_currency == "USD" and not _is_probably_indonesian_symbol(resolved_symbol):
        out["data_quality_flag"] = "currency_mismatch_usd"
        out["fundamental_data_source"] = "bypassed"
        return out
    # ---------------------------------------------------------------

    # First pass: direct quote / info fields.
    out["market_cap"] = _pick_info_value(info, "marketCap", "market_cap", "marketcap")
    out["current_ratio"] = _pick_info_value(info, "currentRatio", "current_ratio")
    out["debt_to_equity"] = _pick_info_value(info, "debtToEquity", "debt_to_equity")
    out["return_on_equity"] = _pick_info_value(info, "returnOnEquity", "return_on_equity")
    out["return_on_assets"] = _pick_info_value(info, "returnOnAssets", "return_on_assets")
    out["operating_margin"] = _pick_info_value(info, "operatingMargins", "operating_margin")
    out["gross_margin"] = _pick_info_value(info, "grossMargins", "gross_margin")
    out["free_cashflow"] = _pick_info_value(info, "freeCashflow", "free_cashflow")
    out["operating_cashflow"] = _pick_info_value(info, "operatingCashflow", "operating_cashflow")
    out["peg_ratio"] = _pick_info_value(info, "pegRatio", "peg_ratio")
    out["trailing_pe"] = _pick_info_value(info, "trailingPE", "trailing_pe")
    out["forward_pe"] = _pick_info_value(info, "forwardPE", "forward_pe")
    out["profit_margins"] = _pick_info_value(info, "profitMargins", "profit_margin", "profit_margins")

    current_price = _pick_info_value(info, "currentPrice", "regularMarketPrice", "lastPrice", "last_price", "previousClose")
    shares_outstanding = _pick_info_value(info, "sharesOutstanding", "shares_outstanding")
    if not np.isfinite(out["market_cap"]) and np.isfinite(current_price) and np.isfinite(shares_outstanding):
        out["market_cap"] = current_price * shares_outstanding

    # Second pass: statement-derived values should be preferred for growth,
    # because Yahoo info fields can lag or mix quarterly/annual definitions.
    try:
        ticker = yf.Ticker(resolved_symbol)
    except Exception:
        ticker = None

    if ticker is not None:
        income_annual = _statement_frame(ticker, ["income_stmt", "financials"])
        income_quarterly = _statement_frame(ticker, ["quarterly_income_stmt", "quarterly_financials"])
        balance_annual = _statement_frame(ticker, ["balance_sheet"])
        balance_quarterly = _statement_frame(ticker, ["quarterly_balance_sheet"])
        cash_annual = _statement_frame(ticker, ["cashflow"])
        cash_quarterly = _statement_frame(ticker, ["quarterly_cashflow"])

        if income_annual.empty and callable(getattr(ticker, "get_income_stmt", None)):
            income_annual = _statement_frame_from_ticker_with_legacy(
                ticker,
                "get_income_stmt",
                [{"legacy": True}, {"legacy": True, "freq": "annual"}, {"freq": "yearly"}, {}],
            )
        if income_quarterly.empty and callable(getattr(ticker, "get_income_stmt", None)):
            income_quarterly = _statement_frame_from_ticker_with_legacy(
                ticker,
                "get_income_stmt",
                [{"legacy": True, "freq": "quarterly"}, {"freq": "quarterly"}, {"frequency": "quarterly"}],
            )
        if balance_annual.empty and callable(getattr(ticker, "get_balance_sheet", None)):
            balance_annual = _statement_frame_from_ticker_with_legacy(
                ticker,
                "get_balance_sheet",
                [{"legacy": True}, {"legacy": True, "freq": "annual"}, {"freq": "yearly"}, {}],
            )
        if balance_quarterly.empty and callable(getattr(ticker, "get_balance_sheet", None)):
            balance_quarterly = _statement_frame_from_ticker_with_legacy(
                ticker,
                "get_balance_sheet",
                [{"legacy": True, "freq": "quarterly"}, {"freq": "quarterly"}, {"frequency": "quarterly"}],
            )
        if cash_annual.empty and callable(getattr(ticker, "get_cashflow", None)):
            cash_annual = _statement_frame_from_ticker_with_legacy(
                ticker,
                "get_cashflow",
                [{"legacy": True}, {"legacy": True, "freq": "annual"}, {"freq": "yearly"}, {}],
            )
        if cash_quarterly.empty and callable(getattr(ticker, "get_cashflow", None)):
            cash_quarterly = _statement_frame_from_ticker_with_legacy(
                ticker,
                "get_cashflow",
                [{"legacy": True, "freq": "quarterly"}, {"freq": "quarterly"}, {"frequency": "quarterly"}],
            )

        if income_annual.empty and not raw_income_annual.empty:
            income_annual = raw_income_annual
        if income_quarterly.empty and not raw_income_quarterly.empty:
            income_quarterly = raw_income_quarterly
        if balance_annual.empty and not raw_balance_annual.empty:
            balance_annual = raw_balance_annual
        if balance_quarterly.empty and not raw_balance_quarterly.empty:
            balance_quarterly = raw_balance_quarterly
        if cash_annual.empty and not raw_cash_annual.empty:
            cash_annual = raw_cash_annual
        if cash_quarterly.empty and not raw_cash_quarterly.empty:
            cash_quarterly = raw_cash_quarterly

        income_frames = [income_quarterly, income_annual]
        balance_frames = [balance_annual, balance_quarterly]
        cash_frames = [cash_annual, cash_quarterly]

        def first_scalar(frames, row_names):
            for frame in frames:
                val = _statement_scalar(frame, row_names)
                if np.isfinite(val):
                    return val
            return np.nan

        def first_growth(frames, row_names):
            for frame in frames:
                val = _statement_growth(frame, row_names)
                if np.isfinite(val):
                    return val
            return np.nan

        revenue = first_scalar(income_frames, ["Total Revenue", "Operating Revenue", "Revenue"])
        revenue_bundle = _growth_bundle_from_frames(income_quarterly, income_annual, ["Total Revenue", "Operating Revenue", "Revenue"])
        net_income = first_scalar(income_frames, ["Net Income", "Net Income Common Stockholders", "Net Income Applicable To Common Shares"])
        earnings_bundle = _growth_bundle_from_frames(income_quarterly, income_annual, ["Net Income", "Net Income Common Stockholders", "Net Income Applicable To Common Shares"])
        operating_income = first_scalar(income_frames, ["Operating Income", "EBIT"])
        gross_profit = first_scalar(income_frames, ["Gross Profit"])
        total_assets = first_scalar(balance_frames, ["Total Assets"])
        current_assets = first_scalar(balance_frames, ["Current Assets"])
        current_liabilities = first_scalar(balance_frames, ["Current Liabilities"])
        total_equity = first_scalar(balance_frames, ["Total Stockholder Equity", "Stockholders Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"])
        total_debt = first_scalar(balance_frames, ["Total Debt", "Short Long Term Debt", "Long Term Debt", "Short Term Debt", "Long Term Debt And Capital Lease Obligation"])
        if not np.isfinite(total_debt):
            total_debt = first_scalar(balance_frames, ["Total Liabilities Net Minority Interest", "Total Liabilities"])
        operating_cashflow = first_scalar(cash_frames, ["Operating Cash Flow", "Total Cash From Operating Activities"])
        capex = first_scalar(cash_frames, ["Capital Expenditure", "Capital Expenditures"])

        # Revenue growth modes
        out["revenue_growth_qoq"] = revenue_bundle.get("qoq", np.nan)
        out["revenue_growth_yoy"] = revenue_bundle.get("yoy", np.nan)
        out["revenue_growth_annual_yoy"] = revenue_bundle.get("annual", np.nan)
        out["revenue_yoy_acceleration"] = revenue_bundle.get("acceleration", np.nan)
        out["revenue_seasonal_qoq_divergence"] = revenue_bundle.get("seasonal_divergence", np.nan)
        out["revenue_growth_quarterly"] = out["revenue_growth_qoq"]
        out["revenue_growth_annual"] = out["revenue_growth_annual_yoy"]
        out["revenue_growth_basis"] = revenue_bundle.get("basis", "n/a")

        # Earnings growth modes
        out["earnings_growth_qoq"] = earnings_bundle.get("qoq", np.nan)
        out["earnings_growth_yoy"] = earnings_bundle.get("yoy", np.nan)
        out["earnings_growth_annual_yoy"] = earnings_bundle.get("annual", np.nan)
        out["earnings_yoy_acceleration"] = earnings_bundle.get("acceleration", np.nan)
        out["earnings_seasonal_qoq_divergence"] = earnings_bundle.get("seasonal_divergence", np.nan)
        out["earnings_growth_quarterly"] = out["earnings_growth_qoq"]
        out["earnings_growth_annual"] = out["earnings_growth_annual_yoy"]
        out["earnings_growth_basis"] = earnings_bundle.get("basis", "n/a")

        # Backward-compatible primary fields: prefer YoY, then annual, then QoQ, then Yahoo info fallback.
        if np.isfinite(out["revenue_growth_yoy"]):
            out["revenue_growth"] = out["revenue_growth_yoy"]
            out["revenue_growth_period"] = "Quarterly YoY"
            out["revenue_growth_source"] = "quarterly_income_stmt"
        elif np.isfinite(out["revenue_growth_annual_yoy"]):
            out["revenue_growth"] = out["revenue_growth_annual_yoy"]
            out["revenue_growth_period"] = "Annual YoY"
            out["revenue_growth_source"] = "income_stmt"
        elif np.isfinite(out["revenue_growth_qoq"]):
            out["revenue_growth"] = out["revenue_growth_qoq"]
            out["revenue_growth_period"] = "QoQ"
            out["revenue_growth_source"] = "quarterly_income_stmt"
        elif not np.isfinite(out["revenue_growth"]):
            out["revenue_growth"] = _pick_info_value(info, "revenueGrowth", "revenue_growth")
            if np.isfinite(out["revenue_growth"]):
                out["revenue_growth_period"] = "Yahoo info"
                out["revenue_growth_source"] = "yahoo-info"

        if np.isfinite(out["earnings_growth_yoy"]):
            out["earnings_growth"] = out["earnings_growth_yoy"]
            out["earnings_growth_period"] = "Quarterly YoY"
            out["earnings_growth_source"] = "quarterly_income_stmt"
        elif np.isfinite(out["earnings_growth_annual_yoy"]):
            out["earnings_growth"] = out["earnings_growth_annual_yoy"]
            out["earnings_growth_period"] = "Annual YoY"
            out["earnings_growth_source"] = "income_stmt"
        elif np.isfinite(out["earnings_growth_qoq"]):
            out["earnings_growth"] = out["earnings_growth_qoq"]
            out["earnings_growth_period"] = "QoQ"
            out["earnings_growth_source"] = "quarterly_income_stmt"
        elif not np.isfinite(out["earnings_growth"]):
            out["earnings_growth"] = _pick_info_value(info, "earningsGrowth", "earningsQuarterlyGrowth", "earnings_growth")
            if np.isfinite(out["earnings_growth"]):
                out["earnings_growth_period"] = "Yahoo info"
                out["earnings_growth_source"] = "yahoo-info"

        if np.isfinite(revenue) and np.isfinite(net_income) and abs(revenue) > 1e-12:
            out["profit_margins"] = out["profit_margins"] if np.isfinite(out["profit_margins"]) else (net_income / revenue)

        if np.isfinite(current_assets) and np.isfinite(current_liabilities) and abs(current_liabilities) > 1e-12:
            out["current_ratio"] = out["current_ratio"] if np.isfinite(out["current_ratio"]) else (current_assets / current_liabilities)

        if np.isfinite(total_equity) and abs(total_equity) > 1e-12:
            if np.isfinite(total_debt):
                out["debt_to_equity"] = out["debt_to_equity"] if np.isfinite(out["debt_to_equity"]) else (total_debt / total_equity)
            if np.isfinite(net_income):
                out["return_on_equity"] = out["return_on_equity"] if np.isfinite(out["return_on_equity"]) else (net_income / total_equity)

        if np.isfinite(total_assets) and abs(total_assets) > 1e-12 and np.isfinite(net_income):
            out["return_on_assets"] = out["return_on_assets"] if np.isfinite(out["return_on_assets"]) else (net_income / total_assets)

        if np.isfinite(revenue) and abs(revenue) > 1e-12 and np.isfinite(operating_income):
            out["operating_margin"] = out["operating_margin"] if np.isfinite(out["operating_margin"]) else (operating_income / revenue)
        if np.isfinite(revenue) and abs(revenue) > 1e-12 and np.isfinite(gross_profit):
            out["gross_margin"] = out["gross_margin"] if np.isfinite(out["gross_margin"]) else (gross_profit / revenue)

        if np.isfinite(operating_cashflow):
            out["operating_cashflow"] = out["operating_cashflow"] if np.isfinite(out["operating_cashflow"]) else operating_cashflow
        if np.isfinite(operating_cashflow) and np.isfinite(capex):
            if capex <= 0:
                fcf = operating_cashflow + capex
            else:
                fcf = operating_cashflow - capex
            out["free_cashflow"] = out["free_cashflow"] if np.isfinite(out["free_cashflow"]) else fcf

        if any(np.isfinite(v) for v in [
            out["current_ratio"], out["debt_to_equity"], out["return_on_equity"], out["return_on_assets"],
            out["operating_margin"], out["gross_margin"], out["free_cashflow"], out["operating_cashflow"],
            out["revenue_growth"], out["earnings_growth"], out["profit_margins"]
        ]):
            out["fundamental_data_source"] = "mixed" if info else "statement-fallback"
            if np.isfinite(out.get("revenue_growth_yoy", np.nan)) or np.isfinite(out.get("earnings_growth_yoy", np.nan)):
                out["data_quality_flag"] = "quarterly_yoy"
            elif np.isfinite(out.get("revenue_growth_annual_yoy", np.nan)) or np.isfinite(out.get("earnings_growth_annual_yoy", np.nan)):
                out["data_quality_flag"] = "annual_yoy"
            elif np.isfinite(out.get("revenue_growth_qoq", np.nan)) or np.isfinite(out.get("earnings_growth_qoq", np.nan)):
                out["data_quality_flag"] = "qoq_only"
            else:
                out["data_quality_flag"] = "fallback" if info else "statement-fallback"

    if out["fundamental_data_source"] == "missing" and info:
        info_source = str(info.get("_info_source", "")).lower()
        if info_source == "google_finance":
            out["fundamental_data_source"] = "google-finance"
        elif info_source == "mixed":
            out["fundamental_data_source"] = "mixed+yahoo+google"
        elif info_source == "yahoo_quote_summary":
            out["fundamental_data_source"] = "yahoo-quote-summary"
        else:
            out["fundamental_data_source"] = "yahoo-info"
        if out["data_quality_flag"] == "missing":
            out["data_quality_flag"] = "fallback"

    return out

@st.cache_data(ttl=21600, show_spinner=False)
def compute_fundamental_grade(symbol: str) -> dict:
    try:
        snap = load_fundamental_snapshot(symbol).copy()
    except Exception:
        snap = {
            "market_cap": np.nan,
            "current_ratio": np.nan,
            "debt_to_equity": np.nan,
            "return_on_equity": np.nan,
            "return_on_assets": np.nan,
            "operating_margin": np.nan,
            "gross_margin": np.nan,
            "free_cashflow": np.nan,
            "operating_cashflow": np.nan,
            "peg_ratio": np.nan,
            "trailing_pe": np.nan,
            "forward_pe": np.nan,
            "revenue_growth": np.nan,
            "earnings_growth": np.nan,
            "profit_margins": np.nan,
            "data_quality_flag": "missing",
            "fundamental_data_source": "missing",
            "_resolved_symbol": "",
        }
    base = str(symbol).strip()
    if not base:
        snap.update(
            {
                "fundamental_score": np.nan,
                "growth_score": np.nan,
                "quality_score": np.nan,
                "health_score": np.nan,
                "valuation_score": np.nan,
                "fundamental_grade": "n/a",
            }
        )
        return snap

    # --- USD bypass: prevent distorted grading for non-IDR financial statements ---
    if snap.get("data_quality_flag") == "currency_mismatch_usd":
        snap.update(
            {
                "fundamental_score": np.nan,
                "growth_score": np.nan,
                "quality_score": np.nan,
                "health_score": np.nan,
                "valuation_score": np.nan,
                "fundamental_grade": "N/A (USD Bypass)",
                "growth_data_reliability": 0.0,
            }
        )
        return snap
    # ------------------------------------------------------------------------------

    def pct_like(v):
        if v is None or pd.isna(v):
            return np.nan
        v = float(v)
        return v if abs(v) <= 1.5 else v / 100.0

    def norm(v, lo, hi, invert=False):
        if v is None or pd.isna(v):
            return np.nan
        if hi == lo:
            return 0.5
        x = (float(v) - lo) / (hi - lo)
        x = float(np.clip(x, 0.0, 1.0))
        return 1.0 - x if invert else x

    rev_g = pct_like(snap.get("revenue_growth"))
    earn_g = pct_like(snap.get("earnings_growth"))
    profit_m = pct_like(snap.get("profit_margins"))
    roe = pct_like(snap.get("return_on_equity"))
    roa = pct_like(snap.get("return_on_assets"))
    op_m = pct_like(snap.get("operating_margin"))
    gross_m = pct_like(snap.get("gross_margin"))
    cr = _coerce_float(snap.get("current_ratio"), np.nan)
    dte = _coerce_float(snap.get("debt_to_equity"), np.nan)
    fcf = _coerce_float(snap.get("free_cashflow"), np.nan)
    ocf = _coerce_float(snap.get("operating_cashflow"), np.nan)

    peg = _coerce_float(snap.get("peg_ratio"), np.nan)
    trailing_pe = _coerce_float(snap.get("trailing_pe"), np.nan)
    forward_pe = _coerce_float(snap.get("forward_pe"), np.nan)
    if (pd.isna(peg) or not np.isfinite(float(peg))) and np.isfinite(forward_pe) and np.isfinite(earn_g):
        if float(earn_g) > 0:
            peg = float(forward_pe) / (float(earn_g) * 100.0 if abs(float(earn_g)) <= 1.5 else float(earn_g))
    snap["peg_ratio"] = peg

    quality_flag = str(snap.get("data_quality_flag", "missing"))
    growth_quality_factor = {
        "quarterly_yoy": 1.00,
        "annual_yoy": 0.85,
        "qoq_only": 0.60,
        "fallback": 0.35,
        "statement-fallback": 0.35,
        "missing": 0.00,
    }.get(quality_flag, 0.50)

    growth_score = 50.0
    if np.isfinite(rev_g):
        growth_score += norm(rev_g, 0.00, 0.30) * 25.0 * growth_quality_factor
    if np.isfinite(earn_g):
        growth_score += norm(earn_g, 0.00, 0.35) * 25.0 * growth_quality_factor
    growth_score = float(np.clip(growth_score, 0.0, 100.0))
    snap["growth_data_reliability"] = growth_quality_factor

    quality_score = 50.0
    if np.isfinite(roe):
        quality_score += norm(roe, 0.08, 0.25) * 25.0
    if np.isfinite(roa):
        quality_score += norm(roa, 0.03, 0.12) * 15.0
    if np.isfinite(profit_m):
        quality_score += norm(profit_m, 0.05, 0.25) * 10.0
    if np.isfinite(op_m):
        quality_score += norm(op_m, 0.05, 0.25) * 10.0
    if np.isfinite(gross_m):
        quality_score += norm(gross_m, 0.20, 0.55) * 5.0
    quality_score = float(np.clip(quality_score, 0.0, 100.0))

    health_score = 50.0
    if np.isfinite(cr):
        health_score += norm(cr, 1.0, 3.0) * 25.0
    if np.isfinite(dte):
        health_score += norm(dte, 150.0, 20.0, invert=True) * 25.0
    if np.isfinite(ocf):
        health_score += 5.0 if ocf > 0 else -5.0
    if np.isfinite(fcf):
        health_score += 5.0 if fcf > 0 else -5.0
    health_score = float(np.clip(health_score, 0.0, 100.0))

    valuation_score = 50.0
    if np.isfinite(peg):
        valuation_score += norm(peg, 0.8, 2.5, invert=True) * 35.0
    elif np.isfinite(trailing_pe) or np.isfinite(forward_pe):
        pe = forward_pe if np.isfinite(forward_pe) else trailing_pe
        valuation_score += norm(pe, 8.0, 25.0, invert=True) * 25.0
    valuation_score = float(np.clip(valuation_score, 0.0, 100.0))

    fundamental_score = float(
        np.clip(
            (growth_score * 0.35)
            + (quality_score * 0.30)
            + (health_score * 0.20)
            + (valuation_score * 0.15),
            0.0,
            100.0,
        )
    )

    if fundamental_score >= 80:
        grade = "A"
    elif fundamental_score >= 67:
        grade = "B"
    elif fundamental_score >= 55:
        grade = "C"
    elif fundamental_score >= 40:
        grade = "D"
    else:
        grade = "E"

    snap.update(
        {
            "fundamental_score": fundamental_score,
            "growth_score": growth_score,
            "quality_score": quality_score,
            "health_score": health_score,
            "valuation_score": valuation_score,
            "fundamental_grade": grade,
        }
    )
    return snap

def compute_future_fundamental_grade(
    symbol: str,
    price_df: pd.DataFrame | None = None,
    macro_context: dict | None = None,
) -> dict:
    """Free-data proxy for forward fundamental quality.

    The main output should reflect the company's forward quality and trajectory.
    Macro context is kept as a separate risk overlay so a strong company is not
    mechanically downgraded into E just because the market regime is weak.
    """
    base = str(symbol).strip()
    try:
        snap = compute_fundamental_grade(base) if base else {}
    except Exception:
        snap = {}
    current_score = _safe_float(snap.get("fundamental_score"), 50.0)
    current_grade = str(snap.get("fundamental_grade", "n/a"))

    d = _ensure_technical_columns(price_df.copy()) if price_df is not None and not price_df.empty else pd.DataFrame()
    last = d.iloc[-1] if not d.empty else None

    # Current-fundamental quality is the anchor.
    current_block = float(np.clip(current_score, 0.0, 100.0))

    # Technical leading proxies that often front-run improved fundamentals.
    price_proxy = 50.0
    if last is not None:
        close = _safe_float(last.get("Close"))
        ema20 = _safe_float(last.get("EMA20"))
        ema50 = _safe_float(last.get("EMA50"))
        ema200 = _safe_float(last.get("EMA200"))
        rsi_v = _safe_float(last.get("RSI14"), 50.0)
        adx_v = _safe_float(last.get("ADX14"), 0.0)
        cmf_v = _safe_float(last.get("CMF20"), 0.0)
        mfi_v = _safe_float(last.get("MFI14"), 50.0)
        obv_slope = _safe_float(last.get("OBV_SLOPE10"), 0.0)
        macd_hist = _safe_float(last.get("MACD_HIST"), 0.0)

        price_proxy = (
            float(close > ema20) * 14
            + float(ema20 > ema50) * 12
            + float(ema50 > ema200) * 10
            + float(rsi_v >= 50) * 8
            + float(adx_v >= 18) * 8
            + float(cmf_v > 0) * 12
            + float(mfi_v >= 50) * 6
            + float(obv_slope > 0) * 12
            + float(macd_hist > 0) * 10
        )
        price_proxy = float(np.clip(price_proxy, 0.0, 100.0))

    phase_info = classify_8_phase(d) if not d.empty and len(d) >= 60 else {"phase": "Unknown", "phase_confidence": 0.0}
    cycle_tuple = compute_cycle_features(d["Close"]) if not d.empty and len(d) >= 30 else (20, 999, False, {})
    dominant_period, time_to_bottom, cycle_ok, cycle_info = cycle_tuple if len(cycle_tuple) == 4 else (20, 999, False, {})
    time_to_top = _safe_float(cycle_info.get("time_to_next_top"), np.nan) if isinstance(cycle_info, dict) else np.nan
    cycle_reliability = _safe_float(cycle_info.get("cycle_reliability"), np.nan) if isinstance(cycle_info, dict) else np.nan

    # Macro regime is used as an overlay, not as the main company score.
    macro_multiplier = 1.0
    macro_gate_ok = True
    macro_gate_reason = "OK"
    macro_score = 50.0
    if isinstance(macro_context, dict) and macro_context:
        macro_multiplier = _safe_float(macro_context.get("macro_multiplier"), 1.0)
        macro_gate_ok = bool(macro_context.get("macro_gate_ok", True))
        macro_gate_reason = str(macro_context.get("macro_gate_reason", "OK"))
        macro_score = _safe_float(macro_context.get("macro_score"), 50.0)
    future_macro_score = float(macro_score)

    quality_score = current_block
    if quality_score < 35:
        quality_score = 35 + (quality_score * 0.4)

    revenue_qoq = _safe_float(snap.get("revenue_growth_qoq"), np.nan)
    revenue_yoy = _safe_float(snap.get("revenue_growth_yoy"), np.nan)
    revenue_yoy_prev = _safe_float(snap.get("revenue_growth_quarterly"), np.nan)
    revenue_yoy_acceleration = _safe_float(snap.get("revenue_yoy_acceleration"), np.nan)
    revenue_seasonal_qoq_divergence = _safe_float(snap.get("revenue_seasonal_qoq_divergence"), np.nan)

    earnings_qoq = _safe_float(snap.get("earnings_growth_qoq"), np.nan)
    earnings_yoy = _safe_float(snap.get("earnings_growth_yoy"), np.nan)
    earnings_yoy_prev = _safe_float(snap.get("earnings_growth_quarterly"), np.nan)
    earnings_yoy_acceleration = _safe_float(snap.get("earnings_yoy_acceleration"), np.nan)
    earnings_seasonal_qoq_divergence = _safe_float(snap.get("earnings_seasonal_qoq_divergence"), np.nan)

    inflection_score = 50.0
    if np.isfinite(revenue_yoy_acceleration):
        inflection_score += _score_bucket(revenue_yoy_acceleration, -0.20, 0.25) * 0.5
    if np.isfinite(earnings_yoy_acceleration):
        inflection_score += _score_bucket(earnings_yoy_acceleration, -0.20, 0.30) * 0.5
    if np.isfinite(revenue_yoy) and np.isfinite(revenue_yoy_prev):
        inflection_score += 8.0 if revenue_yoy > revenue_yoy_prev else -4.0
    if np.isfinite(earnings_yoy) and np.isfinite(earnings_yoy_prev):
        inflection_score += 8.0 if earnings_yoy > earnings_yoy_prev else -4.0
    if np.isfinite(revenue_qoq) and np.isfinite(revenue_seasonal_qoq_divergence):
        inflection_score += 4.0 if revenue_seasonal_qoq_divergence > 0 else -2.0
    if np.isfinite(earnings_qoq) and np.isfinite(earnings_seasonal_qoq_divergence):
        inflection_score += 4.0 if earnings_seasonal_qoq_divergence > 0 else -2.0
    inflection_score = float(np.clip(inflection_score, 0.0, 100.0))

    fundamental_momentum_score = 50.0
    if np.isfinite(revenue_yoy):
        fundamental_momentum_score += _score_bucket(revenue_yoy, -0.10, 0.35) * 0.30
    if np.isfinite(earnings_yoy):
        fundamental_momentum_score += _score_bucket(earnings_yoy, -0.20, 0.55) * 0.30
    if np.isfinite(revenue_yoy_acceleration):
        fundamental_momentum_score += _score_bucket(revenue_yoy_acceleration, -0.15, 0.25) * 0.20
    if np.isfinite(earnings_yoy_acceleration):
        fundamental_momentum_score += _score_bucket(earnings_yoy_acceleration, -0.20, 0.30) * 0.20
    fundamental_momentum_score = float(np.clip(fundamental_momentum_score, 0.0, 100.0))

    seasonal_anomaly_score = 50.0
    if np.isfinite(revenue_seasonal_qoq_divergence):
        seasonal_anomaly_score += _score_bucket(revenue_seasonal_qoq_divergence, -0.20, 0.20) * 0.50
    if np.isfinite(earnings_seasonal_qoq_divergence):
        seasonal_anomaly_score += _score_bucket(earnings_seasonal_qoq_divergence, -0.25, 0.25) * 0.50
    seasonal_anomaly_score = float(np.clip(seasonal_anomaly_score, 0.0, 100.0))

    growth_proxy = 50.0
    if not d.empty:
        try:
            rev_proxy = _score_bucket(d["Close"].pct_change(20).iloc[-1], -0.15, 0.25)
            mom_proxy = _score_bucket(d["Close"].pct_change(60).iloc[-1], -0.25, 0.40)
            accel_proxy = _score_bucket(d["Close"].pct_change(10).iloc[-1] - d["Close"].pct_change(30).iloc[-1], -0.20, 0.20)
            growth_proxy = float(np.clip((rev_proxy * 0.4) + (mom_proxy * 0.35) + (accel_proxy * 0.25), 0.0, 100.0))
        except Exception:
            growth_proxy = 50.0

    cash_flow_proxy = 50.0
    if not d.empty:
        cmf_v = _safe_float(last.get("CMF20"), 0.0) if last is not None else 0.0
        obv_slope = _safe_float(last.get("OBV_SLOPE10"), 0.0) if last is not None else 0.0
        mfi_v = _safe_float(last.get("MFI14"), 50.0) if last is not None else 50.0
        cash_flow_proxy = float(np.clip(
            (50.0
             + (cmf_v * 60.0)
             + (np.clip(obv_slope / (abs(obv_slope) + 1e-9), -1, 1) * 8.0)
             + ((mfi_v - 50.0) * 0.6)),
            0.0,
            100.0
        ))

    balance_quality = 50.0
    cr = _safe_float(snap.get("current_ratio"), np.nan)
    dte = _safe_float(snap.get("debt_to_equity"), np.nan)
    roe = _safe_float(snap.get("return_on_equity"), np.nan)
    roa = _safe_float(snap.get("return_on_assets"), np.nan)
    op_margin = _safe_float(snap.get("operating_margin"), np.nan)
    gross_margin = _safe_float(snap.get("gross_margin"), np.nan)

    if np.isfinite(cr):
        balance_quality += _score_bucket(cr, 0.9, 3.0)
    if np.isfinite(dte):
        balance_quality += _score_bucket(dte, 20.0, 150.0, invert=True) * 0.8
    if np.isfinite(roe):
        balance_quality += _score_bucket(roe, 0.06, 0.25) * 0.8
    if np.isfinite(roa):
        balance_quality += _score_bucket(roa, 0.02, 0.10) * 0.5
    if np.isfinite(op_margin):
        balance_quality += _score_bucket(op_margin, 0.05, 0.25) * 0.7
    if np.isfinite(gross_margin):
        balance_quality += _score_bucket(gross_margin, 0.20, 0.55) * 0.5
    balance_quality = float(np.clip(balance_quality / 2.0, 0.0, 100.0))

    cycle_support = 50.0
    if isinstance(phase_info, dict):
        phase = str(phase_info.get("phase", "Unknown"))
        if phase in {"Early Accumulation", "Accumulation", "Late Accumulation"}:
            cycle_support += 15.0
        elif phase in {"Early Markup", "Markup"}:
            cycle_support += 10.0
        elif phase in {"Distribution", "Markdown"}:
            cycle_support -= 15.0

    if np.isfinite(cycle_reliability):
        cycle_support += float(np.clip((cycle_reliability - 50.0) * 0.35, -15.0, 15.0))
    if np.isfinite(time_to_bottom):
        cycle_support += float(np.clip((8.0 - time_to_bottom) * 1.6, -12.0, 12.0))
    if np.isfinite(time_to_top):
        cycle_support += float(np.clip((time_to_top - 6.0) * 0.8, -8.0, 8.0))
    cycle_support = float(np.clip(cycle_support, 0.0, 100.0))

    future_core_score = (
        current_block * 0.18
        + growth_proxy * 0.10
        + fundamental_momentum_score * 0.22
        + seasonal_anomaly_score * 0.14
        + inflection_score * 0.12
        + cash_flow_proxy * 0.12
        + balance_quality * 0.08
        + price_proxy * 0.08
        + cycle_support * 0.06
    )
    future_score = float(np.clip(future_core_score, 0.0, 100.0))
    future_macro_adjusted_score = float(np.clip(future_score * float(np.clip(macro_multiplier, 0.5, 1.15)), 0.0, 100.0))


    # --- Explicit next-quarter forecasts (percentage / margin level) ---
    current_margin = _safe_float(snap.get("profit_margins"), np.nan)
    if not np.isfinite(current_margin):
        if np.isfinite(op_m):
            current_margin = float(op_m)
        elif np.isfinite(gross_m):
            current_margin = float(gross_m)

    revenue_driver = 0.0
    if np.isfinite(revenue_yoy_acceleration):
        revenue_driver += float(revenue_yoy_acceleration) * 0.40
    if np.isfinite(revenue_seasonal_qoq_divergence):
        revenue_driver += float(revenue_seasonal_qoq_divergence) * 0.20
    revenue_driver += ((float(growth_proxy) - 50.0) / 100.0) * 0.15
    revenue_driver += ((float(price_proxy) - 50.0) / 100.0) * 0.10
    revenue_driver += ((float(quality_score) - 50.0) / 100.0) * 0.05
    revenue_driver += ((float(future_macro_score) - 50.0) / 100.0) * 0.10

    earnings_driver = 0.0
    if np.isfinite(earnings_yoy_acceleration):
        earnings_driver += float(earnings_yoy_acceleration) * 0.45
    if np.isfinite(earnings_seasonal_qoq_divergence):
        earnings_driver += float(earnings_seasonal_qoq_divergence) * 0.20
    earnings_driver += ((float(growth_proxy) - 50.0) / 100.0) * 0.12
    earnings_driver += ((float(price_proxy) - 50.0) / 100.0) * 0.08
    earnings_driver += ((float(quality_score) - 50.0) / 100.0) * 0.10
    earnings_driver += ((float(future_macro_score) - 50.0) / 100.0) * 0.05

    if np.isfinite(revenue_yoy):
        expected_revenue_growth_next_q = float(np.clip(float(revenue_yoy) + revenue_driver, -0.40, 1.50))
    elif np.isfinite(revenue_qoq):
        expected_revenue_growth_next_q = float(np.clip(float(revenue_qoq) * 4.0 + revenue_driver, -0.40, 1.50))
    else:
        expected_revenue_growth_next_q = np.nan

    expected_margin_next_q = np.nan
    if np.isfinite(current_margin):
        margin_driver = 0.0
        if np.isfinite(revenue_yoy_acceleration):
            margin_driver += float(revenue_yoy_acceleration) * 0.18
        if np.isfinite(earnings_yoy_acceleration):
            margin_driver += float(earnings_yoy_acceleration) * 0.30
        if np.isfinite(cash_flow_proxy):
            margin_driver += ((float(cash_flow_proxy) - 50.0) / 100.0) * 0.05
        margin_driver += ((float(future_macro_score) - 50.0) / 100.0) * 0.03
        expected_margin_next_q = float(np.clip(current_margin + margin_driver, 0.0, 1.0))

    expected_eps_growth_next_q = np.nan
    if np.isfinite(expected_revenue_growth_next_q):
        margin_expansion = 0.0
        if np.isfinite(current_margin) and np.isfinite(expected_margin_next_q):
            margin_expansion = float(expected_margin_next_q - current_margin)

        dilution_penalty = 0.0
        if np.isfinite(dte):
            if dte > 100:
                dilution_penalty += 0.03
            elif dte > 50:
                dilution_penalty += 0.015
        if np.isfinite(growth_proxy):
            dilution_penalty += max(0.0, (50.0 - float(growth_proxy)) / 100.0) * 0.02

        expected_eps_growth_next_q = float(
            np.clip(
                float(expected_revenue_growth_next_q)
                + float(earnings_driver) * 0.50
                + margin_expansion * 0.75
                - dilution_penalty,
                -0.80,
                2.50,
            )
        )
    # ------------------------------------------------------------------
    if future_score >= 80:
        grade = "A"
    elif future_score >= 67:
        grade = "B"
    elif future_score >= 55:
        grade = "C"
    elif future_score >= 40:
        grade = "D"
    else:
        grade = "E"

    if future_score - current_block >= 8:
        direction = "Improving"
    elif current_block - future_score >= 8:
        direction = "Deteriorating"
    else:
        direction = "Flat"

    confidence_source_count = 1
    if np.isfinite(price_proxy):
        confidence_source_count += 1
    if np.isfinite(cycle_reliability):
        confidence_source_count += 1
    if np.isfinite(balance_quality):
        confidence_source_count += 1

    growth_quality_factor = _safe_float(snap.get("growth_data_reliability"), 0.5)
    growth_quality_factor = float(np.clip(growth_quality_factor, 0.0, 1.0))
    confidence = float(np.clip(40 + confidence_source_count * 10 + (growth_quality_factor * 10) + (5 if macro_gate_ok else -8), 0.0, 100.0))

    return {
        "current_fundamental_score": current_block,
        "current_fundamental_grade": current_grade,
        "future_fundamental_score": future_score,
        "future_fundamental_grade": grade,
        "future_fundamental_direction": direction,
        "future_fundamental_confidence": confidence,
        "future_growth_proxy": growth_proxy,
        "future_fundamental_momentum_score": fundamental_momentum_score,
        "future_seasonal_anomaly_score": seasonal_anomaly_score,
        "future_inflection_score": inflection_score,
        "future_cash_flow_proxy": cash_flow_proxy,
        "future_balance_quality": balance_quality,
        "future_price_proxy": price_proxy,
        "future_cycle_support": cycle_support,
        "future_macro_score": future_macro_score,
        "future_macro_adjusted_score": future_macro_adjusted_score,
        "future_macro_gate_ok": macro_gate_ok,
        "future_macro_gate_reason": macro_gate_reason,
        "expected_revenue_growth_next_q": expected_revenue_growth_next_q,
        "expected_eps_growth_next_q": expected_eps_growth_next_q,
        "expected_margin_next_q": expected_margin_next_q,
        "future_moat_reason": (
            f"{direction}"
            f" | revNQ={format_growth_percent(expected_revenue_growth_next_q, 0)}"
            f" | epsNQ={format_growth_percent(expected_eps_growth_next_q, 0)}"
            f" | marginNQ={format_growth_percent(expected_margin_next_q, 0)}"
            f" | cycle={phase_info.get('phase', 'Unknown') if isinstance(phase_info, dict) else 'Unknown'}"
            f" | inflection={inflection_score:.0f}"
        ),
        "future_reliability": cycle_reliability,
        "future_time_to_top": time_to_top,
        "future_time_to_bottom": time_to_bottom,
        "future_phase": phase_info.get("phase", "Unknown") if isinstance(phase_info, dict) else "Unknown",
    }



# =========================================================
# Safe public wrappers
# =========================================================

def _fundamental_snapshot_defaults(symbol: str = "") -> dict:
    return {
        "market_cap": np.nan,
        "current_ratio": np.nan,
        "debt_to_equity": np.nan,
        "return_on_equity": np.nan,
        "return_on_assets": np.nan,
        "operating_margin": np.nan,
        "gross_margin": np.nan,
        "free_cashflow": np.nan,
        "operating_cashflow": np.nan,
        "peg_ratio": np.nan,
        "trailing_pe": np.nan,
        "forward_pe": np.nan,
        "revenue_growth": np.nan,
        "earnings_growth": np.nan,
        "profit_margins": np.nan,
        "revenue_growth_quarterly": np.nan,
        "revenue_growth_annual": np.nan,
        "earnings_growth_quarterly": np.nan,
        "earnings_growth_annual": np.nan,
        "revenue_growth_qoq": np.nan,
        "revenue_growth_yoy": np.nan,
        "revenue_growth_annual_yoy": np.nan,
        "earnings_growth_qoq": np.nan,
        "earnings_growth_yoy": np.nan,
        "earnings_growth_annual_yoy": np.nan,
        "revenue_yoy_acceleration": np.nan,
        "earnings_yoy_acceleration": np.nan,
        "revenue_seasonal_qoq_divergence": np.nan,
        "earnings_seasonal_qoq_divergence": np.nan,
        "revenue_growth_period": "n/a",
        "earnings_growth_period": "n/a",
        "revenue_growth_basis": "n/a",
        "earnings_growth_basis": "n/a",
        "revenue_growth_source": "n/a",
        "earnings_growth_source": "n/a",
        "data_quality_flag": "missing",
        "fundamental_data_source": "missing",
        "_resolved_symbol": symbol,
    }

def _future_fundamental_defaults() -> dict:
    return {
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
        "future_macro_gate_ok": True,
        "future_macro_gate_reason": "n/a",
        "expected_revenue_growth_next_q": np.nan,
        "expected_eps_growth_next_q": np.nan,
        "expected_margin_next_q": np.nan,
        "future_moat_reason": "n/a",
        "future_reliability": np.nan,
        "future_time_to_top": np.nan,
        "future_time_to_bottom": np.nan,
        "future_phase": "Unknown",
    }

_orig_load_fundamental_snapshot = load_fundamental_snapshot

def load_fundamental_snapshot(symbol: str) -> dict:
    try:
        result = _orig_load_fundamental_snapshot(symbol)
        return result if isinstance(result, dict) else _fundamental_snapshot_defaults(symbol)
    except Exception:
        return _fundamental_snapshot_defaults(symbol)


_orig_compute_fundamental_grade = compute_fundamental_grade

def compute_fundamental_grade(symbol: str) -> dict:
    try:
        result = _orig_compute_fundamental_grade(symbol)
        return result if isinstance(result, dict) else _fundamental_snapshot_defaults(symbol)
    except Exception:
        snap = _fundamental_snapshot_defaults(symbol)
        snap.update(
            {
                "fundamental_score": np.nan,
                "growth_score": np.nan,
                "quality_score": np.nan,
                "health_score": np.nan,
                "valuation_score": np.nan,
                "fundamental_grade": "n/a",
            }
        )
        return snap


_orig_compute_future_fundamental_grade = compute_future_fundamental_grade

def compute_future_fundamental_grade(symbol: str, price_df: pd.DataFrame | None = None, macro_context: dict | None = None) -> dict:
    try:
        result = _orig_compute_future_fundamental_grade(symbol, price_df, macro_context)
        return result if isinstance(result, dict) else _future_fundamental_defaults()
    except Exception:
        return _future_fundamental_defaults()