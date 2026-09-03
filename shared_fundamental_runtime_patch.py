from __future__ import annotations

"""PASTICUAN runtime integration for Phase 5.6 shared fundamental facts."""

from datetime import datetime, timezone
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from shared_fundamental_runtime import SharedFundamentalRuntime, jk_ticker


PATCH_VERSION = "1.0.0-phase5.6-shared-fundamental"
_INSTALLED = False


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _ratio_from_pct(value: Any) -> float:
    number = _finite(value)
    return number / 100.0 if number is not None else np.nan


def _statement_age_days(period: Any) -> float:
    parsed = pd.to_datetime(period, errors="coerce")
    if pd.isna(parsed):
        return np.nan
    now = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None).normalize()
    stamp = pd.Timestamp(parsed)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_localize(None)
    return max(0.0, float((now - stamp.normalize()).days))


def _shared_frame(tickers: Iterable[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    runtime = SharedFundamentalRuntime("PASTICUAN")
    bundle, meta = runtime.read_bundle(tickers)
    rows: list[dict[str, Any]] = []
    if bundle:
        from scanner import score_fundamentals

        for bare, item in bundle.items():
            metrics = dict(item.get("proxy_metrics") or {})
            if not metrics:
                continue
            official = dict(item.get("official_metrics") or {})
            source_families = " • ".join(item.get("source_families") or ["SHARED_EVIDENCE_HUB"])
            debt_ratio = _finite(metrics.get("debt_equity"))
            if debt_ratio is None:
                debt_ratio = _finite(metrics.get("interest_bearing_debt_to_equity"))
            info = {
                "revenueGrowth": _ratio_from_pct(metrics.get("revenue_growth_pct") or metrics.get("revenue_growth_yoy_pct")),
                "earningsGrowth": _ratio_from_pct(metrics.get("earnings_growth_pct") or metrics.get("earnings_growth_yoy_pct")),
                "grossMargins": _ratio_from_pct(metrics.get("gross_margin_pct")),
                "operatingMargins": _ratio_from_pct(metrics.get("operating_margin_pct")),
                "profitMargins": _ratio_from_pct(metrics.get("net_margin_pct")),
                "returnOnEquity": _ratio_from_pct(metrics.get("roe_pct")),
                "returnOnAssets": _ratio_from_pct(metrics.get("roa_pct")),
                "debtToEquity": debt_ratio * 100.0 if debt_ratio is not None else np.nan,
                "currentRatio": metrics.get("current_ratio"),
                "totalCash": metrics.get("cash"),
                "totalDebt": metrics.get("total_debt"),
                "operatingCashflow": metrics.get("operating_cash_flow"),
                "freeCashflow": metrics.get("free_cash_flow", metrics.get("free_cash_flow_proxy")),
                "marketCap": metrics.get("market_cap"),
                "totalRevenue": metrics.get("revenue"),
                "ebitda": metrics.get("ebitda"),
            }
            scored = score_fundamentals(info)
            period = item.get("proxy_period_end") or item.get("official_period_end")
            observed = item.get("proxy_observed_at") or item.get("official_observed_at") or datetime.now(timezone.utc).isoformat()
            scored.update({
                "ticker": jk_ticker(bare),
                "period_end": period,
                "statement_date": period,
                "statement_age_days": _statement_age_days(period),
                "fundamental_source_count": max(1, len(item.get("source_families") or [])),
                "fundamental_source_families": source_families,
                "fundamental_official_verified": bool(official),
                "fundamental_provider": "SHARED_EVIDENCE_HUB",
                "fundamental_fetched_at": observed,
                "fundamental_route_state": "SHARED_HUB_FACTUAL_FALLBACK",
                "fundamental_error": "",
                "shared_fundamental_runtime_version": PATCH_VERSION,
                "shared_official_coverage_pct": item.get("official_coverage_pct", 0.0),
            })
            # Preserve canonical fields even when score_fundamentals does not use them.
            scored["revenue_growth"] = _ratio_from_pct(metrics.get("revenue_growth_pct") or metrics.get("revenue_growth_yoy_pct"))
            scored["earnings_growth"] = _ratio_from_pct(metrics.get("earnings_growth_pct") or metrics.get("earnings_growth_yoy_pct"))
            scored["roe"] = _ratio_from_pct(metrics.get("roe_pct"))
            scored["roa"] = _ratio_from_pct(metrics.get("roa_pct"))
            scored["net_margin"] = _ratio_from_pct(metrics.get("net_margin_pct"))
            scored["operating_margin"] = _ratio_from_pct(metrics.get("operating_margin_pct"))
            if debt_ratio is not None:
                scored["debt_equity"] = debt_ratio
            for source_key, target_key in (
                ("operating_cash_flow", "operating_cash_flow"),
                ("free_cash_flow", "free_cash_flow"),
                ("market_cap", "market_cap"),
                ("current_ratio", "current_ratio"),
            ):
                if _finite(metrics.get(source_key)) is not None:
                    scored[target_key] = metrics[source_key]
            rows.append(scored)
    audit = pd.DataFrame([{
        "provider": "SHARED_EVIDENCE_HUB",
        "status": str(meta.get("state") or "UNKNOWN"),
        "items": len(rows),
        "detail": f"shared_rows={meta.get('rows', 0)}; shared_tickers={meta.get('tickers', 0)}",
    }])
    return pd.DataFrame(rows), audit


def _publish_frame(frame: pd.DataFrame | None) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "ticker" not in frame.columns:
        return
    runtime = SharedFundamentalRuntime("PASTICUAN")
    if not runtime.ready:
        return
    units = {
        "revenue_growth_pct": "PERCENT", "earnings_growth_pct": "PERCENT",
        "roe_pct": "PERCENT", "roa_pct": "PERCENT", "net_margin_pct": "PERCENT",
        "operating_margin_pct": "PERCENT", "debt_equity": "RATIO",
        "current_ratio": "RATIO", "operating_cash_flow": "CURRENCY_NATIVE",
        "free_cash_flow": "CURRENCY_NATIVE", "market_cap": "CURRENCY_NATIVE",
    }
    for _, row in frame.drop_duplicates("ticker", keep="last").iterrows():
        families = str(row.get("fundamental_source_families") or row.get("fundamental_provider") or "").strip()
        if not families or "SHARED_EVIDENCE_HUB" in families.upper():
            continue
        metrics: dict[str, Any] = {}
        for source, target, scale in (
            ("revenue_growth", "revenue_growth_pct", 100.0),
            ("earnings_growth", "earnings_growth_pct", 100.0),
            ("roe", "roe_pct", 100.0), ("roa", "roa_pct", 100.0),
            ("net_margin", "net_margin_pct", 100.0), ("operating_margin", "operating_margin_pct", 100.0),
            ("debt_equity", "debt_equity", 1.0), ("current_ratio", "current_ratio", 1.0),
            ("operating_cash_flow", "operating_cash_flow", 1.0), ("free_cash_flow", "free_cash_flow", 1.0),
            ("market_cap", "market_cap", 1.0),
        ):
            value = _finite(row.get(source))
            if value is not None:
                metrics[target] = value * scale
        if not metrics:
            continue
        try:
            runtime.publish_metrics(
                str(row.get("ticker")), metrics,
                provider="PASTICUAN_NORMALIZED_RUNTIME",
                source_families=families,
                observed_at=row.get("fundamental_fetched_at") or datetime.now(timezone.utc).isoformat(),
                period_end=row.get("period_end") or row.get("statement_date"),
                units=units,
            )
        except Exception:
            pass


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    import resumable_app_engine as engine

    if not hasattr(engine, "_phase56_original_load_fundamentals"):
        engine._phase56_original_load_fundamentals = engine._load_fundamentals
    if not hasattr(engine, "_phase56_original_refresh_missing_daily_evidence"):
        engine._phase56_original_refresh_missing_daily_evidence = engine._refresh_missing_daily_evidence

    original_load = engine._phase56_original_load_fundamentals
    original_refresh = engine._phase56_original_refresh_missing_daily_evidence

    def load_shared_first(bridge: Any, tickers: Sequence[str], cfg: Any):
        base, history, report = original_load(bridge, tickers, cfg)
        try:
            shared, shared_audit = _shared_frame(tickers)
            if not shared.empty:
                merged = engine._coalesce_primary_evidence(base, shared)
                merged = engine.enrich_fundamentals_with_history(merged, history)
                merged = engine.normalize_fundamental_classification(engine._mark_history_eligible(merged))
            else:
                merged = base
            report = pd.concat([report, shared_audit], ignore_index=True, sort=False) if isinstance(report, pd.DataFrame) and not report.empty else shared_audit
            return merged, history, report
        except Exception as exc:
            extra = pd.DataFrame([{"provider":"SHARED_EVIDENCE_HUB","status":"FAIL_SOFT","error":f"{type(exc).__name__}: {str(exc)[:180]}"}])
            report = pd.concat([report, extra], ignore_index=True, sort=False) if isinstance(report, pd.DataFrame) and not report.empty else extra
            return base, history, report

    def refresh_shared_first(bridge: Any, tickers: Sequence[str], fundamentals: pd.DataFrame, history: pd.DataFrame,
                             market: pd.DataFrame, news: pd.DataFrame, cfg: Any, config: Mapping[str, Any]):
        prepared = fundamentals
        try:
            shared, _ = _shared_frame(tickers)
            if not shared.empty:
                prepared = engine._coalesce_primary_evidence(fundamentals, shared)
                prepared = engine.enrich_fundamentals_with_history(prepared, history)
        except Exception:
            prepared = fundamentals
        result = original_refresh(bridge, tickers, prepared, history, market, news, cfg, config)
        try:
            _publish_frame(result[0])
        except Exception:
            pass
        return result

    engine._load_fundamentals = load_shared_first
    engine._refresh_missing_daily_evidence = refresh_shared_first
    _INSTALLED = True


__all__ = ["PATCH_VERSION", "install"]
