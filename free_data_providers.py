"""Dependency-light public-data fallbacks used by IDX Super Scanner.

These functions deliberately return explicit provider metadata and never invent
values.  They are fallbacks for deployments where yfinance is absent or its
wrapper fails; Yahoo remains the upstream source for these routes.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterable
import math
import time

import numpy as np
import pandas as pd
import requests

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; IDX-Super-Scanner/7.14.0; +research)",
    "Accept": "application/json,text/plain,*/*",
}

_RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})


def _request_json(
    client: requests.Session,
    url: str,
    *,
    params: dict[str, Any],
    timeout: int,
    retry_count: int,
    retry_backoff: float,
) -> tuple[dict[str, Any], int]:
    """GET JSON with a bounded retry policy for transient provider failures."""
    attempts = max(1, int(retry_count))
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = client.get(
                url, params=params, headers=DEFAULT_HEADERS, timeout=timeout,
            )
            status = int(getattr(response, "status_code", 0) or 0)
            if status in _RETRYABLE_HTTP_STATUS and attempt < attempts:
                time.sleep(max(0.0, float(retry_backoff)) * (2 ** (attempt - 1)))
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Provider JSON root is not an object")
            return payload, attempt
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt >= attempts:
                raise
            time.sleep(max(0.0, float(retry_backoff)) * (2 ** (attempt - 1)))
    raise RuntimeError(f"Provider request failed: {last_error}")


def _request_text(
    client: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None,
    timeout: int,
    retry_count: int,
    retry_backoff: float,
) -> tuple[str, int]:
    """GET text with the same bounded retry contract as JSON providers."""
    attempts = max(1, int(retry_count))
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = client.get(
                url,
                params=params or {},
                headers={**DEFAULT_HEADERS, "Accept": "text/html,*/*"},
                timeout=timeout,
            )
            status = int(getattr(response, "status_code", 0) or 0)
            if status in _RETRYABLE_HTTP_STATUS and attempt < attempts:
                time.sleep(max(0.0, float(retry_backoff)) * (2 ** (attempt - 1)))
                continue
            response.raise_for_status()
            payload = str(getattr(response, "text", "") or "")
            if not payload.strip():
                raise ValueError("Provider text response is empty")
            return payload, attempt
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt >= attempts:
                raise
            time.sleep(max(0.0, float(retry_backoff)) * (2 ** (attempt - 1)))
    raise RuntimeError(f"Provider request failed: {last_error}")


def _series_type(value: Any) -> str:
    """Normalize Yahoo's scalar-or-list ``meta.type`` response contract."""
    if isinstance(value, (list, tuple)):
        return next((str(item) for item in value if str(item).strip()), "")
    return str(value or "")


def _raw(value: Any) -> Any:
    if isinstance(value, dict):
        if "raw" in value:
            return value.get("raw")
        if "fmt" in value:
            return value.get("fmt")
    return value


def _epoch(value: Any, default: int) -> int:
    if value is None:
        return default
    stamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(stamp):
        return default
    return int(stamp.timestamp())


def yahoo_chart_direct(
    ticker: str,
    *,
    period: str = "5y",
    start: Any | None = None,
    end: Any | None = None,
    timeout: int = 20,
    session: requests.Session | None = None,
    retry_count: int = 3,
    retry_backoff: float = 0.6,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch adjusted daily OHLCV from Yahoo's chart JSON endpoint.

    OHLC is scaled by Adj Close / raw Close per bar. Split/dividend dates are
    retained in ``DataFrame.attrs`` so quality checks can distinguish a known
    corporate action from an unexplained discontinuity.
    """
    client = session or requests.Session()
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params: dict[str, Any] = {
        "interval": "1d",
        "events": "div,splits,capitalGains",
        "includeAdjustedClose": "true",
    }
    if start is not None or end is not None:
        now = int(time.time())
        params["period1"] = _epoch(start, now - 6 * 365 * 86400)
        params["period2"] = _epoch(end, now + 2 * 86400)
    else:
        params["range"] = period
    payload, attempts = _request_json(
        client, url, params=params, timeout=timeout,
        retry_count=retry_count, retry_backoff=retry_backoff,
    )
    chart = payload.get("chart") or {}
    error = chart.get("error")
    if error:
        raise RuntimeError(f"Yahoo chart error: {error}")
    results = chart.get("result") or []
    if not results:
        return pd.DataFrame(), {"provider": "YAHOO_CHART_DIRECT", "status": "NO_DATA", "attempts": attempts}
    result = results[0]
    timestamps = result.get("timestamp") or []
    quote_list = ((result.get("indicators") or {}).get("quote") or [{}])
    quote = quote_list[0] if quote_list else {}
    adj_list = ((result.get("indicators") or {}).get("adjclose") or [{}])
    adj_values = (adj_list[0] if adj_list else {}).get("adjclose") or []
    columns = {
        "Open": quote.get("open") or [],
        "High": quote.get("high") or [],
        "Low": quote.get("low") or [],
        "Close": quote.get("close") or [],
        "Volume": quote.get("volume") or [],
    }
    n = min([len(timestamps), *(len(values) for values in columns.values())] or [0])
    if n <= 0:
        return pd.DataFrame(), {"provider": "YAHOO_CHART_DIRECT", "status": "NO_DATA", "attempts": attempts}
    index = pd.to_datetime(timestamps[:n], unit="s", utc=True).tz_convert("Asia/Jakarta").tz_localize(None).normalize()
    frame = pd.DataFrame({name: pd.to_numeric(pd.Series(values[:n]), errors="coerce").to_numpy() for name, values in columns.items()}, index=index)
    if len(adj_values) >= n:
        raw_close = pd.to_numeric(frame["Close"], errors="coerce")
        adjusted_close = pd.to_numeric(pd.Series(adj_values[:n], index=frame.index), errors="coerce")
        ratio = adjusted_close.div(raw_close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        ratio = ratio.where(ratio.gt(0)).fillna(1.0)
        for name in ("Open", "High", "Low", "Close"):
            frame[name] = pd.to_numeric(frame[name], errors="coerce") * ratio
        frame["Close"] = adjusted_close.where(adjusted_close.gt(0), frame["Close"])
    events = result.get("events") or {}
    split_dates: list[str] = []
    split_events: list[dict[str, Any]] = []
    dividend_dates: list[str] = []
    for item in (events.get("splits") or {}).values():
        stamp = pd.to_datetime(item.get("date"), unit="s", errors="coerce", utc=True)
        if pd.notna(stamp):
            date = stamp.tz_convert("Asia/Jakarta").date().isoformat()
            split_dates.append(date)
            try:
                numerator = float(item.get("numerator"))
                denominator = float(item.get("denominator"))
                ratio = numerator / denominator if denominator > 0 else np.nan
            except (TypeError, ValueError, ZeroDivisionError):
                numerator = denominator = ratio = np.nan
            if not np.isfinite(ratio) or ratio <= 0:
                ratio_text = str(item.get("splitRatio") or "")
                try:
                    left, right = ratio_text.replace(":", "/").split("/", 1)
                    numerator = float(left)
                    denominator = float(right)
                    ratio = numerator / denominator if denominator > 0 else np.nan
                except (TypeError, ValueError, ZeroDivisionError):
                    ratio = np.nan
            split_events.append({
                "date": date,
                "numerator": numerator if np.isfinite(numerator) else None,
                "denominator": denominator if np.isfinite(denominator) else None,
                "ratio": ratio if np.isfinite(ratio) and ratio > 0 else None,
                "source": "YAHOO_CHART_EVENT",
            })
    for item in (events.get("dividends") or {}).values():
        stamp = pd.to_datetime(item.get("date"), unit="s", errors="coerce", utc=True)
        if pd.notna(stamp):
            dividend_dates.append(stamp.tz_convert("Asia/Jakarta").date().isoformat())
    metadata = result.get("meta") or {}
    frame.attrs.update({
        "provider": "YAHOO_CHART_DIRECT",
        "adjusted_prices": True,
        "corporate_action_split_dates": sorted(set(split_dates)),
        "corporate_action_splits": sorted(
            split_events, key=lambda event: str(event.get("date") or ""),
        ),
        "corporate_action_dividend_dates": sorted(set(dividend_dates)),
        "currency": metadata.get("currency"),
        "exchange_timezone": metadata.get("exchangeTimezoneName"),
        "instrument_type": metadata.get("instrumentType"),
        "provider_checked_at": pd.Timestamp.now(tz="Asia/Jakarta").isoformat(),
    })
    return frame, {
        "provider": "YAHOO_CHART_DIRECT",
        "status": "OK",
        "rows": int(len(frame)),
        "attempts": attempts,
        "split_events": len(split_dates),
        "dividend_events": len(dividend_dates),
    }


_BI_JISDOR_URL = (
    "https://www.bi.go.id/en/statistik/informasi-kurs/jisdor/default.aspx"
)
_MONTH_TRANSLATIONS = {
    "januari": "January", "februari": "February", "maret": "March",
    "april": "April", "mei": "May", "juni": "June", "juli": "July",
    "agustus": "August", "september": "September", "oktober": "October",
    "november": "November", "desember": "December",
}


def _parse_jisdor_date(value: Any) -> pd.Timestamp:
    text = " ".join(str(value or "").replace("\xa0", " ").split())
    for local, english in _MONTH_TRANSLATIONS.items():
        text = text.replace(local.title(), english).replace(local, english)
    return pd.to_datetime(text, errors="coerce", dayfirst=True)


def _parse_jisdor_rate(value: Any) -> float:
    text = str(value or "").upper().replace("RP", "").replace(" ", "")
    text = "".join(character for character in text if character.isdigit() or character in ".,")
    if not text:
        return np.nan
    if "." in text and "," in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        tail = text.rsplit(",", 1)[-1]
        text = text.replace(",", ".") if len(tail) <= 2 else text.replace(",", "")
    elif "." in text:
        tail = text.rsplit(".", 1)[-1]
        if len(tail) == 3:
            text = text.replace(".", "")
    try:
        number = float(text)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) and 1_000.0 <= number <= 100_000.0 else np.nan


def bank_indonesia_jisdor_reference(
    *,
    as_of: Any | None = None,
    timeout: int = 15,
    session: requests.Session | None = None,
    retry_count: int = 2,
    retry_backoff: float = 0.6,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch the latest official USD/IDR JISDOR not later than ``as_of``.

    The official page is parsed conservatively.  A malformed page returns no
    rate rather than reusing a number with uncertain date or locale parsing.
    """
    client = session or requests.Session()
    html, attempts = _request_text(
        client,
        _BI_JISDOR_URL,
        params=None,
        timeout=timeout,
        retry_count=retry_count,
        retry_backoff=retry_backoff,
    )
    from bs4 import BeautifulSoup

    target = pd.to_datetime(as_of, errors="coerce")
    if pd.isna(target):
        target = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None)
    elif getattr(target, "tzinfo", None) is not None:
        target = target.tz_convert("Asia/Jakarta").tz_localize(None)
    target = pd.Timestamp(target).normalize()
    candidates: list[tuple[pd.Timestamp, float]] = []
    soup = BeautifulSoup(html, "html.parser")
    for row in soup.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
        if len(cells) < 2:
            continue
        date = _parse_jisdor_date(cells[0])
        rate = _parse_jisdor_rate(cells[1])
        if pd.notna(date) and np.isfinite(rate):
            candidates.append((pd.Timestamp(date).normalize(), float(rate)))
    if not candidates:
        # Accessible-mode pages can flatten the table. Restrict the fallback
        # to an explicit date followed by an Rp-formatted rate.
        import re
        text = soup.get_text(" ", strip=True)
        pattern = re.compile(
            r"(\d{1,2}\s+[A-Za-z]+\s+\d{4})\s+Rp\s*([0-9.,]+)",
            flags=re.IGNORECASE,
        )
        for date_text, rate_text in pattern.findall(text):
            date = _parse_jisdor_date(date_text)
            rate = _parse_jisdor_rate(rate_text)
            if pd.notna(date) and np.isfinite(rate):
                candidates.append((pd.Timestamp(date).normalize(), float(rate)))
    eligible = [(date, rate) for date, rate in candidates if date <= target]
    if not eligible:
        return pd.DataFrame(), {
            "provider": "BANK_INDONESIA_JISDOR",
            "status": "NO_DATA_NOT_LATER_THAN_ASOF",
            "attempts": attempts,
            "source_url": _BI_JISDOR_URL,
        }
    date, rate = max(eligible, key=lambda item: item[0])
    frame = pd.DataFrame([{
        "currency": "USD",
        "idr_per_unit": rate,
        "as_of": date,
        "source_family": "BANK_INDONESIA_JISDOR",
        "source_name": "Bank Indonesia JISDOR",
        "source_url": _BI_JISDOR_URL,
        "source_verified": True,
        "rate_type": "OFFICIAL_REFERENCE_RATE",
    }])
    return frame, {
        "provider": "BANK_INDONESIA_JISDOR",
        "status": "OK",
        "attempts": attempts,
        "as_of": date.date().isoformat(),
        "usd_idr": rate,
        "source_url": _BI_JISDOR_URL,
    }


def fetch_reference_fx_rates(
    *,
    as_of: Any | None = None,
    timeout: int = 15,
    session: requests.Session | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return official-first reference FX with an explicit proxy fallback."""
    reports: list[dict[str, Any]] = []
    try:
        frame, report = bank_indonesia_jisdor_reference(
            as_of=as_of, timeout=timeout, session=session,
        )
        reports.append(report)
        if not frame.empty:
            return frame, pd.DataFrame(reports)
    except Exception as exc:
        reports.append({
            "provider": "BANK_INDONESIA_JISDOR",
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {str(exc)[:160]}",
            "source_url": _BI_JISDOR_URL,
        })
    try:
        proxy, report = yahoo_chart_direct(
            "USDIDR=X", period="1mo", timeout=timeout, session=session,
        )
        target = pd.to_datetime(as_of, errors="coerce")
        if pd.isna(target):
            target = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None)
        elif getattr(target, "tzinfo", None) is not None:
            target = target.tz_convert("Asia/Jakarta").tz_localize(None)
        eligible = proxy.loc[pd.DatetimeIndex(proxy.index).normalize() <= pd.Timestamp(target).normalize()]
        if not eligible.empty:
            date = pd.Timestamp(eligible.index[-1]).normalize()
            rate = float(pd.to_numeric(eligible["Close"], errors="coerce").iloc[-1])
            if np.isfinite(rate) and rate > 0:
                reports.append({**report, "provider": "YAHOO_FX_PROXY", "as_of": date})
                return pd.DataFrame([{
                    "currency": "USD",
                    "idr_per_unit": rate,
                    "as_of": date,
                    "source_family": "YAHOO_FX_PROXY",
                    "source_name": "Yahoo USD/IDR price proxy",
                    "source_url": "https://query1.finance.yahoo.com/v8/finance/chart/USDIDR=X",
                    "source_verified": False,
                    "rate_type": "MARKET_PRICE_PROXY",
                }]), pd.DataFrame(reports)
    except Exception as exc:
        reports.append({
            "provider": "YAHOO_FX_PROXY",
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {str(exc)[:160]}",
        })
    return pd.DataFrame(), pd.DataFrame(reports)


TIMESERIES_MAP: dict[str, str] = {
    "revenue": "TotalRevenue",
    "gross_profit": "GrossProfit",
    "operating_income": "OperatingIncome",
    "ebit": "EBIT",
    "ebitda": "EBITDA",
    "net_income": "NetIncome",
    "operating_cash_flow": "OperatingCashFlow",
    "free_cash_flow": "FreeCashFlow",
    "capex": "CapitalExpenditure",
    "total_assets": "TotalAssets",
    "total_liabilities": "TotalLiabilitiesNetMinorityInterest",
    # Gross minority equity reconciles with TotalAssets and
    # TotalLiabilitiesNetMinorityInterest. StockholdersEquity remains a
    # fallback for issuers where the gross field is absent.
    "equity": "TotalEquityGrossMinorityInterest",
    "total_debt": "TotalDebt",
    "cash": "CashCashEquivalentsAndShortTermInvestments",
    "shares_outstanding": "OrdinarySharesNumber",
    "interest_expense": "InterestExpense",
}

TIMESERIES_ALIASES: dict[str, str] = {
    "StockholdersEquity": "equity",
}


def yahoo_fundamental_timeseries_direct(
    ticker: str,
    *,
    years_back: int = 6,
    timeout: int = 25,
    session: requests.Session | None = None,
    retry_count: int = 2,
    retry_backoff: float = 0.6,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch annual and quarterly fundamentals from Yahoo time-series JSON.

    Yahoo has served the same public route from both query1 and query2 hosts and
    response objects do not always expose ``meta.type`` consistently.  This
    implementation performs host failover and discovers the dynamic series key
    instead of converting a valid response into a false ``NO_DATA`` state.
    """
    client = session or requests.Session()
    all_types: list[str] = []
    for field in dict.fromkeys((*TIMESERIES_MAP.values(), *TIMESERIES_ALIASES)):
        all_types.extend((f"quarterly{field}", f"annual{field}"))
    quarterly_types = [value for value in all_types if value.startswith("quarterly")]
    annual_types = [value for value in all_types if value.startswith("annual")]
    now = pd.Timestamp.now(tz="UTC")
    base_params = {
        "symbol": ticker,
        "period1": int((now - timedelta(days=366 * max(2, years_back))).timestamp()),
        "period2": int((now + timedelta(days=2)).timestamp()),
        "merge": "false",
        "padTimeSeries": "false",
        "lang": "en-US",
        "region": "US",
        "corsDomain": "finance.yahoo.com",
    }
    routes = (
        ("QUERY1_ALL", "https://query1.finance.yahoo.com", all_types),
        ("QUERY2_ALL", "https://query2.finance.yahoo.com", all_types),
        ("QUERY1_QUARTERLY", "https://query1.finance.yahoo.com", quarterly_types),
        ("QUERY1_ANNUAL", "https://query1.finance.yahoo.com", annual_types),
    )

    payload_results: list[dict[str, Any]] = []
    route_errors: list[str] = []
    attempts_total = 0
    route_success: list[str] = []
    for route_name, host, requested_types in routes:
        # Split routes are only needed when both combined-host calls produced no
        # usable result.  Once any result exists, still allow the paired split
        # route to complete annual+quarterly coverage, but skip redundant hosts.
        if payload_results and route_name == "QUERY2_ALL":
            continue
        url = f"{host}/ws/fundamentals-timeseries/v1/finance/timeseries/{ticker}"
        params = {**base_params, "type": ",".join(requested_types)}
        headers = {
            **DEFAULT_HEADERS,
            "Origin": "https://finance.yahoo.com",
            "Referer": f"https://finance.yahoo.com/quote/{ticker}/financials",
        }
        try:
            # _request_json uses DEFAULT_HEADERS internally.  A dedicated
            # Session header keeps Origin/Referer without changing other users.
            client.headers.update(headers)
            payload, attempts = _request_json(
                client, url, params=params, timeout=timeout,
                retry_count=retry_count if route_name.endswith("ALL") else 1,
                retry_backoff=retry_backoff,
            )
            attempts_total += attempts
            root = payload.get("timeseries") or payload.get("finance") or {}
            if root.get("error"):
                raise RuntimeError(f"Yahoo timeseries error: {root.get('error')}")
            result = root.get("result") or []
            if isinstance(result, list) and result:
                payload_results.extend(item for item in result if isinstance(item, dict))
                route_success.append(route_name)
                if route_name in {"QUERY1_ALL", "QUERY2_ALL"}:
                    break
            else:
                route_errors.append(f"{route_name}:empty result")
        except Exception as exc:
            route_errors.append(f"{route_name}:{type(exc).__name__}:{str(exc)[:120]}")

    by_period: dict[tuple[str, str], dict[str, Any]] = {}
    reverse = {value: key for key, value in TIMESERIES_MAP.items()}
    reverse.update(TIMESERIES_ALIASES)
    currency_by_period: dict[tuple[str, str], str] = {}

    for series in payload_results:
        meta = series.get("meta") or {}
        declared = _series_type(meta.get("type") or series.get("type"))
        dynamic_keys = [
            key for key, value in series.items()
            if isinstance(key, str)
            and key.startswith(("quarterly", "annual"))
            and isinstance(value, list)
        ]
        series_keys = list(dict.fromkeys(([declared] if declared else []) + dynamic_keys))
        for series_type in series_keys:
            prefix = "quarterly" if series_type.startswith("quarterly") else "annual" if series_type.startswith("annual") else ""
            if not prefix:
                continue
            base = series_type[len(prefix):]
            canonical = reverse.get(base)
            if not canonical:
                continue
            points = series.get(series_type) or []
            if not isinstance(points, list):
                continue
            for point in points:
                if not isinstance(point, dict):
                    continue
                date_text = point.get("asOfDate") or point.get("date")
                date = pd.to_datetime(date_text, errors="coerce")
                if pd.isna(date):
                    continue
                key = (date.date().isoformat(), "Q" if prefix == "quarterly" else "FY")
                row = by_period.setdefault(key, {
                    "ticker": ticker,
                    "period_end": date.normalize(),
                    "period_type": key[1],
                    "statement_basis": "STANDALONE_QUARTER" if key[1] == "Q" else "ANNUAL",
                    "source_family": "YAHOO",
                    "source_name": "Yahoo Fundamentals Timeseries Direct",
                    "source_url": f"https://finance.yahoo.com/quote/{ticker}/financials",
                    "currency": "",
                    "source_verified": False,
                    "validation_flags": "",
                })
                reported = point.get("reportedValue") if isinstance(point.get("reportedValue"), dict) else point
                value = _raw(reported)
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    number = np.nan
                if np.isfinite(number):
                    priority_key = f"_source_priority_{canonical}"
                    priority = 2 if TIMESERIES_MAP.get(canonical) == base else 1
                    if priority >= int(row.get(priority_key, 0) or 0):
                        row[canonical] = number
                        row[priority_key] = priority
                currency = str(reported.get("currencyCode") or "") if isinstance(reported, dict) else ""
                if currency:
                    currency_by_period[key] = currency

    for key, row in by_period.items():
        row["currency"] = currency_by_period.get(key, row.get("currency", ""))
        ocf = row.get("operating_cash_flow")
        fcf = row.get("free_cash_flow")
        capex = row.get("capex")
        try:
            ocf_number = float(ocf)
        except (TypeError, ValueError):
            ocf_number = np.nan
        if not np.isfinite(ocf_number):
            try:
                fcf_number = float(fcf)
                capex_number = float(capex)
            except (TypeError, ValueError):
                fcf_number = capex_number = np.nan
            if np.isfinite(fcf_number) and np.isfinite(capex_number):
                row["operating_cash_flow"] = fcf_number - capex_number if capex_number < 0 else fcf_number + capex_number
                flags = [flag for flag in str(row.get("validation_flags") or "").split("|") if flag]
                flags.append("OCF_RECONSTRUCTED_FROM_REPORTED_FCF_AND_CAPEX")
                row["validation_flags"] = "|".join(dict.fromkeys(flags))
        row.pop("free_cash_flow", None)
        for hidden in [name for name in row if name.startswith("_source_priority_")]:
            row.pop(hidden, None)

    frame = pd.DataFrame(list(by_period.values()))
    if not frame.empty:
        frame = frame.sort_values(["period_end", "period_type"]).reset_index(drop=True)
    return frame, {
        "ticker": ticker,
        "provider": "YAHOO_TIMESERIES_DIRECT",
        "status": "OK" if not frame.empty else "NO_DATA",
        "rows": int(len(frame)),
        "attempts": attempts_total,
        "routes": "|".join(route_success),
        "error": " | ".join(route_errors)[:500],
        "error_code": "" if not frame.empty else "NO_SYMBOL_DATA",
    }


def yahoo_quote_summary_direct(
    ticker: str,
    *,
    timeout: int = 20,
    session: requests.Session | None = None,
    retry_count: int = 2,
    retry_backoff: float = 0.6,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fetch a bounded snapshot used by the legacy fundamental scorer."""
    client = session or requests.Session()
    modules = "financialData,defaultKeyStatistics,summaryDetail,price,assetProfile,calendarEvents"
    url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker}"
    payload, attempts = _request_json(
        client, url, params={"modules": modules}, timeout=timeout,
        retry_count=retry_count, retry_backoff=retry_backoff,
    )
    root = payload.get("quoteSummary") or {}
    if root.get("error"):
        raise RuntimeError(f"Yahoo quoteSummary error: {root.get('error')}")
    results = root.get("result") or []
    if not results:
        return {}, {"provider": "YAHOO_QUOTE_SUMMARY_DIRECT", "status": "NO_DATA", "attempts": attempts}
    result = results[0]
    merged: dict[str, Any] = {}
    for module in result.values():
        if isinstance(module, dict):
            for key, value in module.items():
                merged[key] = _raw(value)
    return merged, {"provider": "YAHOO_QUOTE_SUMMARY_DIRECT", "status": "OK", "fields": len(merged), "attempts": attempts}
