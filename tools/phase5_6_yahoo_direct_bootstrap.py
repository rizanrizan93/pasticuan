from __future__ import annotations

"""One-off/free-source bootstrap for Phase 5.6 shared fundamental facts.

Uses Yahoo Finance through yfinance, not ZAPI. Facts are persisted into the
scanner-neutral Shared Evidence Hub and are explicitly non-official.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import math
import time
from typing import Any

import pandas as pd
import yfinance as yf

from shared_evidence_hub import HubConfig, SupabaseEvidenceBackend
from shared_fundamental_runtime import SharedFundamentalRuntime, bare_ticker, jk_ticker


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=160)
    p.add_argument("--sleep", type=float, default=0.75)
    return p.parse_args()


def _finite(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return x if math.isfinite(x) else None


def _pct_ratio(value: Any) -> float | None:
    x = _finite(value)
    return x * 100.0 if x is not None else None


def _frame_value(frame: pd.DataFrame, names: tuple[str, ...], col: Any = None) -> float | None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    column = col if col is not None else frame.columns[0]
    for name in names:
        if name in frame.index:
            return _finite(frame.at[name, column])
    return None


def _latest_period(*frames: pd.DataFrame) -> str | None:
    dates: list[pd.Timestamp] = []
    for frame in frames:
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            for col in frame.columns:
                stamp = pd.to_datetime(col, errors="coerce")
                if pd.notna(stamp):
                    dates.append(pd.Timestamp(stamp))
    return max(dates).date().isoformat() if dates else None


def _yoy_growth(frame: pd.DataFrame, names: tuple[str, ...]) -> float | None:
    if not isinstance(frame, pd.DataFrame) or frame.empty or len(frame.columns) < 2:
        return None
    series = None
    for name in names:
        if name in frame.index:
            series = pd.to_numeric(frame.loc[name], errors="coerce").dropna()
            break
    if series is None or len(series) < 2:
        return None
    latest_col = pd.to_datetime(series.index[0], errors="coerce")
    latest = _finite(series.iloc[0])
    if pd.isna(latest_col) or latest is None:
        return None
    candidates: list[tuple[int, float]] = []
    for col, raw in series.iloc[1:].items():
        stamp = pd.to_datetime(col, errors="coerce")
        value = _finite(raw)
        if pd.isna(stamp) or value is None:
            continue
        days = abs((pd.Timestamp(latest_col) - pd.Timestamp(stamp)).days)
        if 300 <= days <= 430:
            candidates.append((abs(days - 365), value))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    prior = candidates[0][1]
    if prior == 0:
        return None
    return 100.0 * (latest / prior - 1.0)


def _collect(symbol: str) -> tuple[dict[str, float], dict[str, str], str | None, str]:
    ticker = yf.Ticker(symbol)
    try:
        info = ticker.info or {}
    except Exception:
        info = {}
    try:
        income = ticker.quarterly_financials
    except Exception:
        income = pd.DataFrame()
    try:
        balance = ticker.quarterly_balance_sheet
    except Exception:
        balance = pd.DataFrame()
    try:
        cashflow = ticker.quarterly_cashflow
    except Exception:
        cashflow = pd.DataFrame()

    period = _latest_period(income, balance, cashflow)
    latest_income_col = income.columns[0] if isinstance(income, pd.DataFrame) and not income.empty else None
    latest_balance_col = balance.columns[0] if isinstance(balance, pd.DataFrame) and not balance.empty else None
    latest_cash_col = cashflow.columns[0] if isinstance(cashflow, pd.DataFrame) and not cashflow.empty else None

    revenue = _frame_value(income, ("Total Revenue", "Operating Revenue"), latest_income_col) or _finite(info.get("totalRevenue"))
    net_income = _frame_value(income, ("Net Income", "Net Income Common Stockholders"), latest_income_col) or _finite(info.get("netIncomeToCommon"))
    operating_income = _frame_value(income, ("Operating Income",), latest_income_col)
    ocf = _frame_value(cashflow, ("Operating Cash Flow", "Total Cash From Operating Activities"), latest_cash_col) or _finite(info.get("operatingCashflow"))
    fcf = _frame_value(cashflow, ("Free Cash Flow",), latest_cash_col) or _finite(info.get("freeCashflow"))
    cash = _frame_value(balance, ("Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents", "Cash"), latest_balance_col) or _finite(info.get("totalCash"))
    debt = _frame_value(balance, ("Total Debt",), latest_balance_col) or _finite(info.get("totalDebt"))
    equity = _frame_value(balance, ("Stockholders Equity", "Total Equity Gross Minority Interest"), latest_balance_col)
    assets = _frame_value(balance, ("Total Assets",), latest_balance_col)
    liabilities = _frame_value(balance, ("Total Liabilities Net Minority Interest", "Total Liab"), latest_balance_col)
    current_assets = _frame_value(balance, ("Current Assets", "Total Current Assets"), latest_balance_col)
    current_liabilities = _frame_value(balance, ("Current Liabilities", "Total Current Liabilities"), latest_balance_col)

    metrics: dict[str, float] = {}
    units: dict[str, str] = {}

    def put(name: str, value: Any, unit: str) -> None:
        x = _finite(value)
        if x is not None:
            metrics[name] = x
            units[name] = unit

    rev_growth = _pct_ratio(info.get("revenueGrowth"))
    if rev_growth is None:
        rev_growth = _yoy_growth(income, ("Total Revenue", "Operating Revenue"))
    earn_growth = _pct_ratio(info.get("earningsGrowth"))
    if earn_growth is None:
        earn_growth = _yoy_growth(income, ("Net Income", "Net Income Common Stockholders"))

    put("revenue_growth_pct", rev_growth, "PERCENT")
    put("earnings_growth_pct", earn_growth, "PERCENT")
    put("roe_pct", _pct_ratio(info.get("returnOnEquity")), "PERCENT")
    put("roa_pct", _pct_ratio(info.get("returnOnAssets")), "PERCENT")
    net_margin = _pct_ratio(info.get("profitMargins"))
    if net_margin is None and revenue not in (None, 0) and net_income is not None:
        net_margin = 100.0 * net_income / revenue
    op_margin = _pct_ratio(info.get("operatingMargins"))
    if op_margin is None and revenue not in (None, 0) and operating_income is not None:
        op_margin = 100.0 * operating_income / revenue
    put("net_margin_pct", net_margin, "PERCENT")
    put("operating_margin_pct", op_margin, "PERCENT")

    de = _finite(info.get("debtToEquity"))
    if de is not None:
        de /= 100.0
    elif debt is not None and equity not in (None, 0):
        de = debt / equity
    put("debt_equity", de, "RATIO")
    cr = _finite(info.get("currentRatio"))
    if cr is None and current_assets is not None and current_liabilities not in (None, 0):
        cr = current_assets / current_liabilities
    put("current_ratio", cr, "RATIO")

    for name, value in (
        ("revenue", revenue), ("net_income", net_income), ("operating_income", operating_income),
        ("operating_cash_flow", ocf), ("free_cash_flow", fcf), ("cash", cash),
        ("total_debt", debt), ("equity", equity), ("total_assets", assets),
        ("total_liabilities", liabilities), ("current_assets", current_assets),
        ("current_liabilities", current_liabilities), ("market_cap", info.get("marketCap")),
    ):
        put(name, value, "CURRENCY_NATIVE")

    if cash is not None and debt not in (None, 0):
        put("cash_to_debt_ratio", cash / debt, "RATIO")
    if ocf is not None and net_income not in (None, 0):
        put("ocf_conversion_ratio", ocf / net_income, "RATIO")
    if liabilities is not None and equity not in (None, 0):
        put("total_liabilities_to_equity", liabilities / equity, "RATIO")
    if debt is not None and cash is not None and equity not in (None, 0):
        put("net_debt_to_equity", (debt - cash) / equity, "RATIO")

    state = "OK" if metrics else "EMPTY"
    return metrics, units, period, state


def main() -> int:
    args = _args()
    config = HubConfig.from_environment(client_id="PASTICUAN")
    backend = SupabaseEvidenceBackend(config)
    runtime = SharedFundamentalRuntime("PASTICUAN", config=config, backend=backend)
    source_rows = backend.read_rows(
        "latest_fundamental_snapshots", {},
        select=("ticker,fundamental_source_families,revenue_growth,earnings_growth,roe,net_margin,operating_cash_flow,debt_equity"),
        limit=5000,
    )
    gaps: list[str] = []
    for row in source_rows:
        keys = ("revenue_growth", "earnings_growth", "roe", "net_margin", "operating_cash_flow", "debt_equity")
        present = sum(row.get(k) not in (None, "") for k in keys)
        if not str(row.get("fundamental_source_families") or "").strip() or present < 4:
            code = bare_ticker(row.get("ticker"))
            if code:
                gaps.append(code)
    gaps = list(dict.fromkeys(gaps))[: max(0, args.limit)]

    summary = Counter()
    failures: list[dict[str, str]] = []
    rows_written = 0
    for idx, code in enumerate(gaps, 1):
        try:
            metrics, units, period, state = _collect(jk_ticker(code))
            if metrics:
                written, meta = runtime.publish_metrics(
                    code, metrics,
                    provider="YAHOO_DIRECT_YFINANCE_BOOTSTRAP",
                    source_families="YAHOO_DIRECT_YFINANCE",
                    observed_at=datetime.now(timezone.utc).isoformat(),
                    period_end=period,
                    units=units,
                )
                rows_written += len(written)
                summary["refreshed"] += 1
                summary["rows"] += len(written)
            else:
                summary["empty"] += 1
                failures.append({"ticker": code, "state": state})
        except Exception as exc:
            summary["failed"] += 1
            if len(failures) < 30:
                failures.append({"ticker": code, "state": f"{type(exc).__name__}: {str(exc)[:160]}"})
        print(f"BOOTSTRAP_PROGRESS {idx}/{len(gaps)} {code} refreshed={summary['refreshed']} rows={rows_written}", flush=True)
        if idx < len(gaps):
            time.sleep(max(0.0, float(args.sleep)))

    print({
        "candidate_gaps": len(gaps),
        "refreshed": summary["refreshed"],
        "empty": summary["empty"],
        "failed": summary["failed"],
        "rows": rows_written,
        "failure_samples": failures,
        "provider": "YAHOO_DIRECT_YFINANCE_BOOTSTRAP",
        "official_verified": False,
        "policy": "FACTS_ONLY_NO_SCORING_OR_GATE_CHANGE",
    }, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
