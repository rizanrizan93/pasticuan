from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Iterable

import numpy as np
import pandas as pd


def _num(value: Any) -> float:
    try:
        result = float(value)
        return result if np.isfinite(result) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _linear_score(value: float, bad: float, good: float, higher_is_better: bool = True) -> float | None:
    if not np.isfinite(value):
        return None
    if good == bad:
        return 50.0
    ratio = (value - bad) / (good - bad)
    if not higher_is_better:
        ratio = 1 - ratio
    return float(np.clip(100 * ratio, 0, 100))


def score_fundamentals(info: dict[str, Any]) -> dict[str, Any]:
    sector_text = str(info.get("sector") or "")
    industry_text = str(info.get("industry") or "")
    is_financial = "financial" in sector_text.lower() or "bank" in industry_text.lower()
    debt_equity_raw = _num(info.get("debtToEquity"))
    debt_equity = debt_equity_raw / 100 if np.isfinite(debt_equity_raw) else np.nan
    total_cash = _num(info.get("totalCash"))
    total_debt = _num(info.get("totalDebt"))
    market_cap = _num(info.get("marketCap"))
    fcf = _num(info.get("freeCashflow"))
    ocf = _num(info.get("operatingCashflow"))
    cash_to_debt = total_cash / total_debt if np.isfinite(total_cash) and total_debt > 0 else (
        5.0 if np.isfinite(total_cash) and total_debt == 0 else np.nan
    )
    fcf_yield = fcf / market_cap if np.isfinite(fcf) and market_cap > 0 else np.nan
    metrics = {
        "revenue_growth": _num(info.get("revenueGrowth")),
        "earnings_growth": _num(info.get("earningsGrowth")),
        "gross_margin": _num(info.get("grossMargins")),
        "operating_margin": _num(info.get("operatingMargins")),
        "net_margin": _num(info.get("profitMargins")),
        "roe": _num(info.get("returnOnEquity")),
        "roa": _num(info.get("returnOnAssets")),
        "debt_equity": debt_equity,
        "current_ratio": _num(info.get("currentRatio")),
        "cash_to_debt": cash_to_debt,
        "operating_cash_flow": ocf,
        "free_cash_flow": fcf,
        "fcf_yield": fcf_yield,
        "trailing_pe": _num(info.get("trailingPE")),
        "forward_pe": _num(info.get("forwardPE")),
        "price_to_book": _num(info.get("priceToBook")),
        "peg_ratio": _num(info.get("pegRatio")),
        "market_cap": market_cap,
        "sector": sector_text or industry_text,
        "company_name": info.get("shortName") or info.get("longName") or "",
        "fundamental_model": "FINANCIAL" if is_financial else "GENERAL",
    }
    weighted: list[tuple[float, float]] = []
    applicable_weight = 0.0

    def add(score: float | None, weight: float) -> None:
        nonlocal applicable_weight
        applicable_weight += weight
        if score is not None and np.isfinite(score):
            weighted.append((float(score), weight))

    add(_linear_score(metrics["revenue_growth"], -0.05, 0.20), 14)
    add(_linear_score(metrics["earnings_growth"], -0.10, 0.25), 14)
    add(_linear_score(metrics["roe"], 0.05, 0.22), 10)
    add(_linear_score(metrics["roa"], 0.01, 0.10), 7)
    add(_linear_score(metrics["gross_margin"], 0.10, 0.45), 6)
    add(_linear_score(metrics["operating_margin"], 0.02, 0.20), 7)
    add(_linear_score(metrics["net_margin"], 0.01, 0.15), 6)
    if not is_financial:
        add(_linear_score(metrics["debt_equity"], 2.0, 0.3, higher_is_better=True), 8)
        add(_linear_score(metrics["current_ratio"], 0.8, 2.0), 5)
        add(_linear_score(metrics["cash_to_debt"], 0.1, 1.2), 5)
        add(100.0 if np.isfinite(ocf) and ocf > 0 else 0.0 if np.isfinite(ocf) else None, 6)
        add(100.0 if np.isfinite(fcf) and fcf > 0 else 0.0 if np.isfinite(fcf) else None, 6)
        add(_linear_score(metrics["fcf_yield"], 0.0, 0.08), 3)
    peg = metrics["peg_ratio"]
    peg_score = None
    if np.isfinite(peg):
        peg_score = 100.0 if 0 < peg <= 1.5 else 65.0 if peg <= 2.5 else 20.0 if peg > 0 else 0.0
    add(peg_score, 3)
    score = sum(value * weight for value, weight in weighted) / sum(weight for _, weight in weighted) if weighted else np.nan
    coverage = sum(weight for _, weight in weighted) / applicable_weight if applicable_weight else 0.0
    red_flags: list[str] = []
    if np.isfinite(metrics["revenue_growth"]) and metrics["revenue_growth"] < 0:
        red_flags.append("Revenue menyusut")
    if np.isfinite(metrics["earnings_growth"]) and metrics["earnings_growth"] < 0:
        red_flags.append("Laba menyusut")
    if not is_financial:
        if np.isfinite(ocf) and ocf <= 0:
            red_flags.append("OCF negatif")
        if np.isfinite(fcf) and fcf <= 0:
            red_flags.append("FCF negatif")
        if np.isfinite(debt_equity) and debt_equity > 2:
            red_flags.append("DER tinggi")
    if np.isfinite(metrics["net_margin"]) and metrics["net_margin"] <= 0:
        red_flags.append("Margin bersih negatif")
    metrics.update(
        {
            "fundamental_score": round(float(score), 1) if np.isfinite(score) else np.nan,
            "fundamental_coverage": round(100 * coverage, 1),
            "fundamental_reliability": "HIGH" if coverage >= 0.70 else "MEDIUM" if coverage >= 0.45 else "LOW",
            "fundamental_red_flags": " • ".join(red_flags),
        }
    )
    return metrics


def fetch_one_fundamental(ticker: str) -> dict[str, Any]:
    import yfinance as yf

    try:
        info = yf.Ticker(ticker).get_info()
        row = score_fundamentals(info or {})
        row.update({"ticker": ticker, "fundamental_error": ""})
        return row
    except Exception as exc:
        return {
            "ticker": ticker,
            "fundamental_score": np.nan,
            "fundamental_coverage": 0.0,
            "fundamental_reliability": "NONE",
            "fundamental_red_flags": "",
            "fundamental_error": f"{type(exc).__name__}: {str(exc)[:100]}",
        }


def fetch_fundamentals(tickers: Iterable[str], max_workers: int = 4) -> pd.DataFrame:
    names = list(dict.fromkeys(tickers))
    rows: list[dict[str, Any]] = []
    if not names:
        return pd.DataFrame()
    with ThreadPoolExecutor(max_workers=min(max_workers, len(names))) as pool:
        futures = {pool.submit(fetch_one_fundamental, ticker): ticker for ticker in names}
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows)


def attach_fundamentals(signals: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals.copy()
    if fundamentals.empty:
        result = signals.copy()
        result["fundamental_score"] = np.nan
        result["fundamental_coverage"] = 0.0
        result["fundamental_reliability"] = "NONE"
        result["fundamental_red_flags"] = ""
        result["fundamental_error"] = "Fundamental tidak diambil/tersedia"
        result["composite_score"] = result["quality_score"]
        return result
    result = signals.merge(fundamentals, on="ticker", how="left")
    usable = result["fundamental_coverage"].fillna(0) >= 45
    result["composite_score"] = result["quality_score"]
    result.loc[usable, "composite_score"] = (
        0.78 * result.loc[usable, "quality_score"] + 0.22 * result.loc[usable, "fundamental_score"]
    ).round(1)
    return result
