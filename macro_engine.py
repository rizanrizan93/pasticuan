from __future__ import annotations

"""Macro-first regime and issuer transmission engine.

The engine intentionally separates three concepts:
1. global/IDX market regime, which is shared by the entire universe;
2. sector transmission, which varies by business sector;
3. issuer macro alignment, which varies by ticker through direct exposure data
   when available and sector-relative confirmation otherwise.

No macro factor is allowed to create a buy signal on its own.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from free_data_providers import yahoo_chart_direct


MACRO_ENGINE_VERSION = "9.5.0-macro-breadth-guard"

MACRO_SYMBOLS: dict[str, str] = {
    "USDIDR": "USDIDR=X",
    "DXY": "DX-Y.NYB",
    "US10Y": "^TNX",
    "OIL": "CL=F",
    "GOLD": "GC=F",
}


@dataclass(frozen=True)
class MacroRegimeResult:
    snapshot: pd.DataFrame
    factors: dict[str, float]
    sector_map: pd.DataFrame
    issuer_map: pd.DataFrame
    source_report: pd.DataFrame


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _clip(value: Any, lower: float = 0.0, upper: float = 100.0) -> float:
    number = _finite(value, np.nan)
    if not np.isfinite(number):
        return np.nan
    return float(np.clip(number, lower, upper))


def _ticker(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    return text if text.endswith(".JK") else f"{text}.JK"


def _normalise_sector(value: Any) -> str:
    text = str(value or "").strip().upper().replace("&", "AND")
    aliases = {
        "BASIC MATERIALS": "BASIC MATERIALS",
        "BASIC INDUSTRY": "BASIC MATERIALS",
        "MATERIALS": "BASIC MATERIALS",
        "ENERGY": "ENERGY",
        "FINANCIALS": "FINANCIALS",
        "FINANCE": "FINANCIALS",
        "BANKING": "FINANCIALS",
        "CONSUMER CYCLICAL": "CONSUMER CYCLICALS",
        "CONSUMER CYCLICALS": "CONSUMER CYCLICALS",
        "CYCLICAL": "CONSUMER CYCLICALS",
        "CONSUMER NON-CYCLICAL": "CONSUMER NON-CYCLICALS",
        "CONSUMER NON-CYCLICALS": "CONSUMER NON-CYCLICALS",
        "NON-CYCLICAL": "CONSUMER NON-CYCLICALS",
        "INDUSTRIALS": "INDUSTRIALS",
        "INDUSTRIAL": "INDUSTRIALS",
        "INFRASTRUCTURE": "INFRASTRUCTURE",
        "PROPERTIES AND REAL ESTATE": "PROPERTY",
        "PROPERTY": "PROPERTY",
        "REAL ESTATE": "PROPERTY",
        "TECHNOLOGY": "TECHNOLOGY",
        "HEALTHCARE": "HEALTHCARE",
        "TRANSPORTATION AND LOGISTIC": "TRANSPORTATION",
        "TRANSPORTATION AND LOGISTICS": "TRANSPORTATION",
        "TRANSPORTATION": "TRANSPORTATION",
    }
    if text in aliases:
        return aliases[text]
    for key, mapped in aliases.items():
        if key in text:
            return mapped
    return text or "UNKNOWN"


def _normalise_ohlcv(frame: pd.DataFrame | None) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(col[0]) for col in out.columns]
    close_col = next((c for c in ("Close", "close", "Adj Close", "adjclose") if c in out.columns), None)
    if close_col is None:
        return pd.DataFrame()
    out = out.rename(columns={close_col: "Close"})
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    return out.dropna(subset=["Close"]).sort_index()


def _series_change(frame: pd.DataFrame | None, periods: int = 20) -> float:
    clean = _normalise_ohlcv(frame)
    if len(clean) <= periods:
        return np.nan
    first = _finite(clean["Close"].iloc[-periods - 1], np.nan)
    last = _finite(clean["Close"].iloc[-1], np.nan)
    if not np.isfinite(first) or first == 0 or not np.isfinite(last):
        return np.nan
    return last / first - 1.0


def _benchmark_features(benchmark: pd.DataFrame | None) -> dict[str, float]:
    clean = _normalise_ohlcv(benchmark)
    if clean.empty:
        return {
            "ihsg_return_20d": np.nan,
            "ihsg_return_60d": np.nan,
            "ihsg_trend_score": np.nan,
            "ihsg_volatility_20d": np.nan,
        }
    close = clean["Close"]
    ret20 = _series_change(clean, 20)
    ret60 = _series_change(clean, 60)
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
    last = close.iloc[-1]
    trend_votes = [last > ema20, ema20 > ema50, ema50 > ema200]
    trend_score = 100.0 * sum(bool(v) for v in trend_votes) / len(trend_votes)
    volatility = close.pct_change().tail(20).std(ddof=0) * np.sqrt(252)
    return {
        "ihsg_return_20d": ret20,
        "ihsg_return_60d": ret60,
        "ihsg_trend_score": trend_score,
        "ihsg_volatility_20d": volatility,
    }


def _prepared_breadth(prepared: Mapping[str, pd.DataFrame]) -> dict[str, float]:
    rows: list[tuple[bool, bool, bool]] = []
    for frame in prepared.values():
        clean = _normalise_ohlcv(frame)
        if len(clean) < 60:
            continue
        close = clean["Close"]
        last = _finite(close.iloc[-1], np.nan)
        ema50 = _finite(close.ewm(span=50, adjust=False).mean().iloc[-1], np.nan)
        ema200 = _finite(close.ewm(span=200, adjust=False).mean().iloc[-1], np.nan)
        ret20 = _series_change(clean, 20)
        if not np.isfinite(last):
            continue
        rows.append((last > ema50 if np.isfinite(ema50) else False,
                     last > ema200 if np.isfinite(ema200) else False,
                     ret20 > 0 if np.isfinite(ret20) else False))
    if not rows:
        return {"breadth_above_ema50_pct": np.nan, "breadth_above_ema200_pct": np.nan,
                "breadth_positive_20d_pct": np.nan, "breadth_sample": 0.0}
    array = np.asarray(rows, dtype=float)
    return {
        "breadth_above_ema50_pct": float(array[:, 0].mean() * 100.0),
        "breadth_above_ema200_pct": float(array[:, 1].mean() * 100.0),
        "breadth_positive_20d_pct": float(array[:, 2].mean() * 100.0),
        "breadth_sample": float(len(rows)),
    }


def fetch_macro_series(*, period: str = "6mo", timeout: int = 12) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    series: dict[str, pd.DataFrame] = {}
    reports: list[dict[str, Any]] = []
    for factor, symbol in MACRO_SYMBOLS.items():
        try:
            frame, report = yahoo_chart_direct(symbol, period=period, timeout=timeout)
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                series[factor] = frame
            reports.append({
                "factor": factor,
                "symbol": symbol,
                "status": report.get("status", "OK" if factor in series else "EMPTY"),
                "source": report.get("provider", "YAHOO_CHART_DIRECT"),
                "error": report.get("error", ""),
            })
        except Exception as exc:
            reports.append({
                "factor": factor,
                "symbol": symbol,
                "status": "FAIL_SOFT",
                "source": "YAHOO_CHART_DIRECT",
                "error": f"{type(exc).__name__}: {str(exc)[:160]}",
            })
    return series, pd.DataFrame(reports)



def _mean_available(values: Sequence[float]) -> float:
    observed = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(observed)) if observed else np.nan

def _factor_score(change: float, *, positive_is_good: bool, scale: float) -> float:
    if not np.isfinite(change):
        return np.nan
    signed = change if positive_is_good else -change
    return float(np.clip(50.0 + 50.0 * signed / max(scale, 1e-9), 0.0, 100.0))


def _build_factor_state(
    benchmark: pd.DataFrame | None,
    prepared: Mapping[str, pd.DataFrame],
    macro_series: Mapping[str, pd.DataFrame] | None,
    breadth_features: Mapping[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    breadth = dict(breadth_features or _prepared_breadth(prepared))
    features = {**_benchmark_features(benchmark), **breadth}
    series = dict(macro_series or {})
    usd_change = _series_change(series.get("USDIDR"), 20)
    dxy_change = _series_change(series.get("DXY"), 20)
    yield_change = _series_change(series.get("US10Y"), 20)
    oil_change = _series_change(series.get("OIL"), 20)
    gold_change = _series_change(series.get("GOLD"), 20)

    factor_scores = {
        "risk_appetite": _mean_available([
            _factor_score(features.get("ihsg_return_20d", np.nan), positive_is_good=True, scale=0.08),
            _clip(features.get("ihsg_trend_score", np.nan)),
            _clip(features.get("breadth_above_ema50_pct", np.nan)),
            _clip(features.get("breadth_positive_20d_pct", np.nan)),
        ]),
        "currency_stability": _mean_available([
            _factor_score(usd_change, positive_is_good=False, scale=0.05),
            _factor_score(dxy_change, positive_is_good=False, scale=0.05),
        ]),
        "rate_support": _factor_score(yield_change, positive_is_good=False, scale=0.10),
        "commodity_energy": _factor_score(oil_change, positive_is_good=True, scale=0.15),
        "defensive_demand": _factor_score(gold_change, positive_is_good=True, scale=0.12),
    }
    observed = [v for v in factor_scores.values() if np.isfinite(v)]
    coverage = 100.0 * len(observed) / len(factor_scores)
    regime_score = float(np.nanmean(observed)) if observed else np.nan
    if not np.isfinite(regime_score):
        regime = "DATA_PENDING"
    elif regime_score >= 67:
        regime = "RISK_ON_EXPANSION"
    elif regime_score >= 56:
        regime = "SELECTIVE_RISK_ON"
    elif regime_score >= 44:
        regime = "NEUTRAL_SELECTIVE"
    elif regime_score >= 33:
        regime = "SELECTIVE_RISK_OFF"
    else:
        regime = "RISK_OFF_CONTRACTION"

    raw = {
        **features,
        "usd_idr_change_20d": usd_change,
        "dxy_change_20d": dxy_change,
        "us10y_change_20d": yield_change,
        "oil_change_20d": oil_change,
        "gold_change_20d": gold_change,
        "macro_regime_score": regime_score,
        "macro_data_coverage_pct": coverage,
        "macro_regime": regime,
    }
    return factor_scores, raw


_SECTOR_WEIGHTS: dict[str, dict[str, float]] = {
    "ENERGY": {"risk_appetite": 0.20, "currency_stability": 0.10, "rate_support": 0.05,
               "commodity_energy": 0.60, "defensive_demand": 0.05},
    "BASIC MATERIALS": {"risk_appetite": 0.35, "currency_stability": 0.20, "rate_support": 0.05,
                        "commodity_energy": 0.30, "defensive_demand": 0.10},
    "FINANCIALS": {"risk_appetite": 0.45, "currency_stability": 0.20, "rate_support": 0.25,
                   "commodity_energy": 0.05, "defensive_demand": 0.05},
    "PROPERTY": {"risk_appetite": 0.40, "currency_stability": 0.10, "rate_support": 0.40,
                 "commodity_energy": 0.05, "defensive_demand": 0.05},
    "TECHNOLOGY": {"risk_appetite": 0.45, "currency_stability": 0.10, "rate_support": 0.35,
                   "commodity_energy": 0.00, "defensive_demand": 0.10},
    "CONSUMER CYCLICALS": {"risk_appetite": 0.50, "currency_stability": 0.25, "rate_support": 0.15,
                           "commodity_energy": 0.00, "defensive_demand": 0.10},
    "CONSUMER NON-CYCLICALS": {"risk_appetite": 0.20, "currency_stability": 0.35, "rate_support": 0.10,
                               "commodity_energy": 0.00, "defensive_demand": 0.35},
    "INDUSTRIALS": {"risk_appetite": 0.45, "currency_stability": 0.20, "rate_support": 0.15,
                    "commodity_energy": 0.10, "defensive_demand": 0.10},
    "INFRASTRUCTURE": {"risk_appetite": 0.35, "currency_stability": 0.15, "rate_support": 0.30,
                       "commodity_energy": 0.10, "defensive_demand": 0.10},
    "TRANSPORTATION": {"risk_appetite": 0.45, "currency_stability": 0.20, "rate_support": 0.10,
                       "commodity_energy": -0.20, "defensive_demand": 0.05},
    "HEALTHCARE": {"risk_appetite": 0.20, "currency_stability": 0.30, "rate_support": 0.10,
                   "commodity_energy": 0.00, "defensive_demand": 0.40},
    "UNKNOWN": {"risk_appetite": 0.40, "currency_stability": 0.20, "rate_support": 0.20,
                "commodity_energy": 0.10, "defensive_demand": 0.10},
}


def build_sector_map(factors: Mapping[str, float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sector, weights in _SECTOR_WEIGHTS.items():
        numerator = 0.0
        denominator = 0.0
        for factor, weight in weights.items():
            value = _finite(factors.get(factor), np.nan)
            if not np.isfinite(value):
                continue
            # Negative transmission means a rising factor is adverse.
            transformed = value if weight >= 0 else 100.0 - value
            numerator += abs(weight) * transformed
            denominator += abs(weight)
        score = numerator / denominator if denominator else np.nan
        coverage = 100.0 * denominator / sum(abs(v) for v in weights.values()) if weights else 0.0
        rows.append({
            "sector": sector,
            "macro_sector_score": round(score, 1) if np.isfinite(score) else np.nan,
            "macro_sector_coverage_pct": round(coverage, 1),
            "macro_sector_state": "SCORED" if np.isfinite(score) and coverage >= 50 else "PARTIAL" if coverage > 0 else "MISSING",
            "macro_engine_version": MACRO_ENGINE_VERSION,
        })
    return pd.DataFrame(rows).sort_values("macro_sector_score", ascending=False, na_position="last").reset_index(drop=True)


def _latest_by_ticker(frame: pd.DataFrame | None) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "ticker" not in frame.columns:
        return pd.DataFrame(columns=["ticker"])
    out = frame.copy()
    out["ticker"] = out["ticker"].map(_ticker)
    return out[out["ticker"].ne("")].drop_duplicates("ticker", keep="last")


def build_issuer_map(
    fundamentals: pd.DataFrame | None,
    sector_map: pd.DataFrame,
) -> pd.DataFrame:
    base = _latest_by_ticker(fundamentals)
    if base.empty:
        return pd.DataFrame(columns=[
            "ticker", "sector", "macro_sector_score", "issuer_macro_alignment_score",
            "issuer_macro_alignment_coverage_pct", "issuer_macro_alignment_state",
        ])
    sector_scores = sector_map.set_index("sector")["macro_sector_score"].to_dict() if not sector_map.empty else {}
    rows: list[dict[str, Any]] = []
    for _, row in base.iterrows():
        sector = _normalise_sector(row.get("sector"))
        sector_score = _finite(sector_scores.get(sector, sector_scores.get("UNKNOWN")), np.nan)
        components: list[tuple[float, float, str]] = []
        if np.isfinite(sector_score):
            components.append((sector_score, 0.60, "SECTOR_TRANSMISSION"))

        sector_relative = np.nan
        for name in ("growth_sector_relative_score", "sector_relative_quality_score", "sector_relative_strength_score"):
            value = _finite(row.get(name), np.nan)
            if np.isfinite(value):
                sector_relative = _clip(value)
                break
        if np.isfinite(sector_relative):
            components.append((sector_relative, 0.20, "SECTOR_RELATIVE_CONFIRMATION"))

        direct_adjustments: list[float] = []
        export_pct = _finite(row.get("export_revenue_pct"), np.nan)
        import_pct = _finite(row.get("import_cost_exposure_pct"), np.nan)
        usd_debt_pct = _finite(row.get("usd_debt_pct"), np.nan)
        commodity_pct = _finite(row.get("commodity_revenue_pct"), np.nan)
        for value, positive in ((export_pct, True), (commodity_pct, True), (import_pct, False), (usd_debt_pct, False)):
            if not np.isfinite(value):
                continue
            fraction = value / 100.0 if abs(value) > 1.5 else value
            direct_adjustments.append(np.clip(50.0 + (25.0 if positive else -25.0) * fraction, 0.0, 100.0))
        if direct_adjustments:
            components.append((float(np.mean(direct_adjustments)), 0.20, "DIRECT_ISSUER_EXPOSURE"))

        if components:
            total_weight = sum(weight for _, weight, _ in components)
            score = sum(value * weight for value, weight, _ in components) / total_weight
            coverage = 100.0 * total_weight
            basis = " | ".join(label for _, _, label in components)
        else:
            score, coverage, basis = np.nan, 0.0, "NO_MACRO_TRANSMISSION_EVIDENCE"
        rows.append({
            "ticker": row["ticker"],
            "sector": sector,
            "macro_sector_score": round(sector_score, 1) if np.isfinite(sector_score) else np.nan,
            "issuer_macro_alignment_score": round(score, 1) if np.isfinite(score) else np.nan,
            "issuer_macro_alignment_coverage_pct": round(coverage, 1),
            "issuer_macro_alignment_state": "SCORED" if coverage >= 60 else "PARTIAL" if coverage > 0 else "MISSING",
            "issuer_macro_alignment_basis": basis,
            "macro_engine_version": MACRO_ENGINE_VERSION,
        })
    return pd.DataFrame(rows)


def build_macro_regime(
    *,
    benchmark: pd.DataFrame | None,
    prepared: Mapping[str, pd.DataFrame],
    fundamentals: pd.DataFrame | None = None,
    macro_series: Mapping[str, pd.DataFrame] | None = None,
    source_report: pd.DataFrame | None = None,
    breadth_features: Mapping[str, float] | None = None,
) -> MacroRegimeResult:
    factors, raw = _build_factor_state(benchmark, prepared, macro_series, breadth_features=breadth_features)
    sector_map = build_sector_map(factors)
    issuer_map = build_issuer_map(fundamentals, sector_map)
    snapshot = pd.DataFrame([{
        **raw,
        **{f"factor_{name}_score": round(value, 1) if np.isfinite(value) else np.nan for name, value in factors.items()},
        "macro_engine_version": MACRO_ENGINE_VERSION,
    }])
    return MacroRegimeResult(
        snapshot=snapshot,
        factors=dict(factors),
        sector_map=sector_map,
        issuer_map=issuer_map,
        source_report=source_report.copy() if isinstance(source_report, pd.DataFrame) else pd.DataFrame(),
    )


__all__ = [
    "MACRO_ENGINE_VERSION",
    "MACRO_SYMBOLS",
    "MacroRegimeResult",
    "fetch_macro_series",
    "build_sector_map",
    "build_issuer_map",
    "build_macro_regime",
]
