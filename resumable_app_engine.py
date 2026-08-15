from __future__ import annotations

"""Chunk processors and finalizers for the durable v9.4 scan jobs."""

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import os
import re
import time

import numpy as np
import pandas as pd

from database_first import _coerce_bool_series, build_database_coverage, readiness_summary
from evidence_enrichment import enrich_fundamental_evidence
from ihsg_direction import IHSGDirectionConfig, analyze_ihsg_direction
from macro_engine import build_macro_regime, fetch_macro_series
from issuer_classification import normalize_fundamental_classification
from official_evidence_guard import canonicalize_official_fundamental_evidence
from narrative_engine import build_narrative_intelligence
from resumable_scan import ItemOutcome, json_safe
from scanner import (
    ScanConfig,
    ScanEngine,
    analyze_portfolio_positions,
    apply_execution_snapshot_gate,
    apply_fundamental_gate,
    apply_independent_price_gate,
    apply_market_status_gate,
    apply_news_gate,
    apply_universe_integrity_gate,
    attach_fundamentals,
    attach_ohlcv_source_lineage,
    attach_position_sizing,
    build_independent_price_validation,
    combine_fundamental_history,
    clean_ohlcv,
    download_benchmark,
    download_ohlcv,
    enrich_fundamentals_with_valuation,
    enrich_fundamentals_with_history,
    fetch_automatic_independent_prices,
    fetch_execution_snapshots,
    fetch_idx_fundamental_history,
    IDX_DAILY_FINAL_HOUR,
    IDX_DAILY_FINAL_MINUTE,
    fetch_resilient_fundamentals,
    fetch_resilient_market_status,
    fetch_resilient_news_review,
    fetch_yahoo_fundamental_history,
    finalize_execution_integrity,
    read_cached_fundamental_history,
    read_cached_fundamentals,
    read_cached_market_status,
    read_cached_news_review,
    select_yahoo_fundamental_tickers,
    seed_daily_ohlcv_cache,
)
from scanner_database import ScannerDatabaseBridge
from idx_trading_calendar import idx_session_lag, is_idx_session, previous_idx_session
from simple_focus import build_simple_focus, build_silent_profiles
from two_stage_pipeline import ShortlistConfig, build_enrichment_shortlist
from fundamental_calibration import maintenance_refresh_priority, reporting_refresh_profile
from release_contract import SCANNER_RELEASE_VERSION

ENGINE_VERSION = SCANNER_RELEASE_VERSION

LEAN_FINAL_PERSISTENCE_TABLES = (
    "fundamental_cache", "fundamental_snapshots",
    "multibagger_snapshots", "technical_snapshots",
    "ihsg_direction_snapshots", "narrative_snapshots", "scan_runs",
)
FULL_FINAL_PERSISTENCE_TABLES = (
    "fundamental_cache", "fundamental_snapshots",
    "multibagger_snapshots", "technical_snapshots",
    "ihsg_direction_snapshots", "provider_health",
    "scan_runs", "scan_checkpoints",
    "narrative_events", "narrative_event_outcomes",
    "narrative_snapshots", "selector_snapshots",
    "selector_outcomes", "selector_model_evaluations",
)


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _int_config(config: Mapping[str, Any], key: str, default: int) -> int:
    """Read an integer setting without turning an explicit zero into default.

    ``int(config.get(key) or default)`` made 0 impossible, so a supposedly
    disabled provider lane silently reverted to its default request budget.
    """
    raw = config.get(key, default)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        raw = default
    try:
        return int(raw)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _ticker(value: Any) -> str:
    text = str(value or "").upper().strip()
    return text if text.endswith(".JK") else f"{text}.JK" if text else ""


def _merge_primary(primary: pd.DataFrame | None, fallback: pd.DataFrame | None) -> pd.DataFrame:
    frames = [frame.copy() for frame in (primary, fallback) if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not frames:
        return pd.DataFrame()
    # Pandas 2.3+ warns when concat infers dtypes from all-NA columns. Drop
    # those columns per frame, concatenate once, then restore the stable union.
    ordered_columns: list[str] = []
    compact: list[pd.DataFrame] = []
    for frame in frames:
        for column in frame.columns:
            if column not in ordered_columns:
                ordered_columns.append(column)
        keep = [column for column in frame.columns if not frame[column].isna().all()]
        compact.append(frame.loc[:, keep].copy())
    combined = pd.concat(compact, ignore_index=True, sort=False).reindex(columns=ordered_columns)
    if "ticker" not in combined.columns:
        return combined
    combined["ticker"] = combined["ticker"].map(_ticker)
    return combined.loc[combined["ticker"].ne("")].drop_duplicates("ticker", keep="first").reset_index(drop=True)


def _coalesce_primary_evidence(primary: pd.DataFrame | None, fallback: pd.DataFrame | None) -> pd.DataFrame:
    """Ticker-wise evidence merge where primary non-missing values win.

    The persistent database is the production source of truth.  Local/session
    cache is only a field-level fallback, never allowed to overwrite a value
    that is already present in the primary row.  The same helper is used when
    a fresh live snapshot is merged over stored evidence: live values win while
    older fields remain available only where the live provider is silent.
    """
    frames = []
    for frame in (primary, fallback):
        if isinstance(frame, pd.DataFrame) and not frame.empty and "ticker" in frame.columns:
            local = frame.copy()
            local["ticker"] = local["ticker"].map(_ticker)
            local = local.loc[local["ticker"].ne("")].drop_duplicates("ticker", keep="last")
            frames.append(local)
        else:
            frames.append(pd.DataFrame())
    first, second = frames
    if first.empty:
        return second.reset_index(drop=True)
    if second.empty:
        return first.reset_index(drop=True)
    # Empty strings from provider payloads mean missing evidence, not an
    # authoritative value that should erase a populated fallback field.
    for local in (first, second):
        object_cols = local.select_dtypes(include=["object", "string"]).columns
        for column in object_cols:
            if column == "ticker":
                continue
            blank_mask = local[column].map(lambda value: isinstance(value, str) and not value.strip())
            local[column] = local[column].mask(blank_mask, pd.NA)

    # ``normalize_fundamental_classification`` represents unavailable sector
    # evidence with the literal value UNKNOWN. That sentinel must not mask an
    # explicit IDX-IC sector supplied by the uploaded universe. Hotfix 6 kept
    # the upload columns, but the generic ``combine_first`` below still kept
    # UNKNOWN plus its 0%-confidence classification bundle.
    if "sector" in first.columns and "sector" in second.columns:
        first_indexed = first.set_index("ticker")
        second_indexed = second.set_index("ticker")
        common = first_indexed.index.intersection(second_indexed.index)
        missing_tokens = {"", "UNKNOWN", "MISSING", "UNCLASSIFIED", "NAN", "NONE", "NULL"}
        primary_sector = first_indexed.loc[common, "sector"].astype("string").str.strip().str.upper()
        fallback_sector = second_indexed.loc[common, "sector"].astype("string").str.strip().str.upper()
        replace_index = common[
            primary_sector.fillna("").isin(missing_tokens)
            & ~fallback_sector.fillna("").isin(missing_tokens)
        ]
        classification_columns = (
            "sector", "sector_raw", "sector_source", "sector_confidence_pct",
            "sector_classification_version", "idx_sector", "sector_idx_ic",
        )
        for column in classification_columns:
            if column in second_indexed.columns:
                if column not in first_indexed.columns:
                    first_indexed[column] = pd.NA
                first_indexed.loc[replace_index, column] = second_indexed.loc[replace_index, column]
        first = first_indexed.reset_index()
    merged = first.set_index("ticker").combine_first(second.set_index("ticker"))
    return merged.reset_index()


def _history_counts(history: pd.DataFrame | None) -> dict[str, int]:
    if history is None or history.empty or "ticker" not in history.columns:
        return {}
    local = history.copy()
    local["ticker"] = local["ticker"].map(_ticker)
    period_col = next((column for column in ("period_end", "statement_date", "date", "as_of") if column in local.columns), None)
    if period_col is None:
        return local.groupby("ticker").size().astype(int).to_dict()
    local[period_col] = pd.to_datetime(local[period_col], errors="coerce")
    return local.dropna(subset=[period_col]).groupby("ticker")[period_col].nunique().astype(int).to_dict()


def _mark_history_eligible(fundamentals: pd.DataFrame | None) -> pd.DataFrame:
    """Apply production freshness/history gates to normalized fundamentals.

    v9.5 allowed an old cached ``fundamental_score_eligible=True`` flag to
    survive for up to ~550 days.  That made a stale snapshot look production
    ready.  v9.6 recomputes eligibility from the current evidence every run.
    """
    if fundamentals is None or fundamentals.empty:
        return pd.DataFrame()
    out = normalize_fundamental_classification(fundamentals)
    score = pd.to_numeric(out.get("fundamental_score", pd.Series(np.nan, index=out.index)), errors="coerce")
    coverage = pd.to_numeric(out.get("fundamental_coverage", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    source_count = pd.to_numeric(out.get("fundamental_source_count", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    age = pd.to_numeric(out.get("statement_age_days", pd.Series(np.nan, index=out.index)), errors="coerce")
    quarters = pd.to_numeric(out.get("fundamental_history_quarters", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    years = pd.to_numeric(out.get("fundamental_history_years", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    period_ok = quarters.ge(2.0) | years.ge(2.0)
    freshness_ok = age.notna() & age.le(300.0)
    derived = score.notna() & coverage.ge(45.0) & source_count.ge(1.0) & period_ok & freshness_ok
    out["fundamental_score_eligible"] = derived.astype(bool)
    out["fundamental_production_ready"] = derived.astype(bool)
    out["fundamental_freshness_state"] = np.select(
        [age.notna() & age.le(210.0), age.notna() & age.le(300.0), age.notna()],
        ["CURRENT", "ACCEPTABLE_STALE", "STALE"],
        default="MISSING_DATE",
    )
    out["fundamental_refresh_required"] = (~derived | age.isna() | age.gt(210.0) | out["sector"].eq("UNKNOWN")).astype(bool)
    if "fundamental_route_state" not in out.columns:
        out["fundamental_route_state"] = ""
    out.loc[derived, "fundamental_route_state"] = out.loc[derived, "fundamental_route_state"].replace("", "HISTORY_DERIVED_VERIFIED")
    out.loc[~freshness_ok & score.notna(), "fundamental_route_state"] = "STALE_REQUIRES_REFRESH"
    return out


def _load_fundamentals(bridge: ScannerDatabaseBridge, tickers: Sequence[str], cfg: ScanConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        db_snapshot, audit_snapshot = bridge.read_fundamental_cache(tickers)
    except Exception as exc:
        db_snapshot, audit_snapshot = pd.DataFrame(), pd.DataFrame([{'provider':'DATABASE_FUNDAMENTAL','status':'READ_FAIL_SOFT','error':f'{type(exc).__name__}: {str(exc)[:180]}'}])
    try:
        db_history, audit_history = bridge.read_fundamental_history_cache(tickers)
    except Exception as exc:
        db_history, audit_history = pd.DataFrame(), pd.DataFrame([{'provider':'DATABASE_FUNDAMENTAL_HISTORY','status':'READ_FAIL_SOFT','error':f'{type(exc).__name__}: {str(exc)[:180]}'}])
    local_snapshot = read_cached_fundamentals(tickers, cfg)
    if not local_snapshot.empty and "fundamental_score_eligible" in local_snapshot.columns:
        # Retain cached rows as evidence; v9.6 recomputes production eligibility
        # after history enrichment instead of trusting an old boolean flag.
        local_snapshot = local_snapshot.copy()
    local_history = read_cached_fundamental_history(tickers)
    snapshot = _coalesce_primary_evidence(db_snapshot, local_snapshot)
    history = combine_fundamental_history(db_history, local_history)
    snapshot = normalize_fundamental_classification(
        _mark_history_eligible(
            canonicalize_official_fundamental_evidence(
                enrich_fundamentals_with_history(snapshot, history)
            )
        )
    )
    audits = [frame for frame in (audit_snapshot, audit_history) if isinstance(frame, pd.DataFrame) and not frame.empty]
    return snapshot, history, pd.concat(audits, ignore_index=True, sort=False) if audits else pd.DataFrame()


def _load_aux(bridge: ScannerDatabaseBridge, tickers: Sequence[str], cfg: ScanConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        db_market, _ = bridge.read_market_status_cache(tickers, max_age_days=max(1, int(getattr(cfg, "market_status_cache_days", 3))))
    except Exception:
        db_market = pd.DataFrame()
    try:
        db_news, _ = bridge.read_news_review_cache(tickers, max_age_days=max(1, int(getattr(cfg, "news_cache_days", 7))))
    except Exception:
        db_news = pd.DataFrame()
    market = _merge_primary(read_cached_market_status(tickers, cfg), db_market)
    news = _merge_primary(read_cached_news_review(tickers, lookback_days=45, config=cfg), db_news)
    return market, news


def _cfg(config: Mapping[str, Any]) -> ScanConfig:
    return ScanConfig(
        account_size_idr=float(config.get("account_size_idr", 5_000_000) or 5_000_000),
        cash_on_hand_idr=float(config.get("cash_on_hand_idr", 1_000_000) or 1_000_000),
        risk_per_trade_pct=float(config.get("risk_per_trade_pct", 0.0075) or 0.0075),
        ai_enabled=False,
        ai_max_weight=0.0,
        selector_max_ai_weight=0.0,
        time_cycle_enabled=False,
        time_cycle_core_max_weight=0.0,
        time_cycle_multibagger_max_weight=0.0,
        eoff_enabled=False,
        eoff_ephemeris_enabled=False,
        sharia_only=bool(config.get("sharia_only", False)),
        provider_retry_count=1,
        fundamental_provider_batch_size=max(4, int(config.get("provider_batch_size", 12) or 12)),
        idx_fundamental_years_back=2,
    )



def _merge_ohlcv(base: pd.DataFrame | None, update: pd.DataFrame | None, *, max_bars: int = 900) -> pd.DataFrame:
    left = clean_ohlcv(base, strict=True) if isinstance(base, pd.DataFrame) else pd.DataFrame()
    right = clean_ohlcv(update, strict=True) if isinstance(update, pd.DataFrame) else pd.DataFrame()
    if left.empty:
        return right.tail(max_bars).copy()
    if right.empty:
        return left.tail(max_bars).copy()
    merged = pd.concat([left, right], axis=0, sort=False)
    merged.index = pd.to_datetime(merged.index, errors="coerce")
    merged = merged.loc[~pd.DatetimeIndex(merged.index).isna()].copy()
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return clean_ohlcv(merged, strict=True).tail(max_bars).copy()


def _expected_completed_session(now: Any | None = None) -> pd.Timestamp:
    stamp = pd.Timestamp(now or datetime.now(timezone.utc))
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    local = stamp.tz_convert("Asia/Jakarta")
    today = local.tz_localize(None).normalize()
    include_today = bool(
        is_idx_session(today)
        and (local.hour, local.minute) >= (IDX_DAILY_FINAL_HOUR, IDX_DAILY_FINAL_MINUTE)
    )
    return previous_idx_session(today, include_date=include_today)


def _report_maps(report: Any) -> tuple[dict[str, str], dict[str, str]]:
    tiers = {str(k).upper().strip(): str(v or "") for k, v in dict(getattr(report, "source_tiers", {}) or {}).items()}
    errors = {str(k).upper().strip(): str(v or "") for k, v in dict(getattr(report, "failed", {}) or {}).items()}
    return tiers, errors


def _database_first_ohlcv(
    bridge: ScannerDatabaseBridge,
    tickers: Sequence[str],
    *,
    period: str,
    itick_api_token: str,
    min_bars: int = 120,
    max_stale_sessions: int = 5,
    force_refresh: bool = False,
) -> tuple[dict[str, pd.DataFrame], Any, pd.DataFrame]:
    names = list(dict.fromkeys(str(value).upper().strip() for value in tickers if str(value).strip()))
    expected = _expected_completed_session()
    try:
        cached, cache_audit = bridge.read_ohlcv_cache(names, min_bars=min_bars)
    except Exception as exc:
        cached, cache_audit = {}, pd.DataFrame([{
            "provider": "SUPABASE_OHLCV", "status": "READ_FAIL_SOFT",
            "error": f"{type(exc).__name__}: {str(exc)[:240]}",
        }])
    cache_meta = {}
    if isinstance(cache_audit, pd.DataFrame) and not cache_audit.empty and "ticker" in cache_audit.columns:
        cache_meta = cache_audit.drop_duplicates("ticker", keep="last").set_index("ticker").to_dict("index")

    refresh_targets: list[str] = []
    initial_lag: dict[str, int] = {}
    for ticker in names:
        frame = clean_ohlcv(cached.get(ticker), strict=True) if ticker in cached else pd.DataFrame()
        lag = idx_session_lag(frame.index[-1], expected) if not frame.empty else 9999
        initial_lag[ticker] = int(lag)
        if not frame.empty:
            try:
                seed_daily_ohlcv_cache(ticker, frame, str(cache_meta.get(ticker, {}).get("source_family") or "DATABASE"))
            except Exception:
                pass
        if force_refresh or len(frame) < min_bars or lag > 0:
            refresh_targets.append(ticker)

    live_histories: dict[str, pd.DataFrame] = {}
    report: Any = None
    if refresh_targets:
        try:
            live_histories, report = download_ohlcv(
                refresh_targets,
                period=period,
                itick_api_token=itick_api_token,
            )
        except Exception as exc:
            report = type("DownloadFailure", (), {
                "source_tiers": {},
                "failed": {ticker: f"{type(exc).__name__}: {str(exc)[:300]}" for ticker in refresh_targets},
                "warnings": {},
            })()
    live_tiers, live_errors = _report_maps(report)

    histories: dict[str, pd.DataFrame] = {}
    source_tiers: dict[str, str] = {}
    refresh_states: dict[str, str] = {}
    audit_rows: list[dict[str, Any]] = []
    persist_histories: dict[str, pd.DataFrame] = {}
    persist_errors: dict[str, str] = {}
    # Legacy JSONB rows are converted lazily to compact OHLCV. The conversion
    # budget is intentionally bounded so first deployment cannot turn into a
    # one-time 400-row migration stall.
    legacy_compact_candidates = [
        ticker for ticker in names
        if str(cache_meta.get(ticker, {}).get("payload_format") or "").upper() == "LEGACY_JSON"
        and ticker in cached
    ][:max(0, min(80, int(os.environ.get("SCANNER_COMPACT_OHLCV_MIGRATION_CAP", "400") or 400)))]
    for ticker in names:
        merged = _merge_ohlcv(cached.get(ticker), live_histories.get(ticker))
        bars = len(merged)
        lag = idx_session_lag(merged.index[-1], expected) if not merged.empty else 9999
        live_ok = ticker in live_histories and isinstance(live_histories.get(ticker), pd.DataFrame) and not live_histories[ticker].empty
        if bars >= min_bars:
            histories[ticker] = merged
        if live_ok:
            tier = live_tiers.get(ticker, "LIVE_PUBLIC")
        elif bars >= min_bars and lag == 0:
            tier = f"DATABASE_CURRENT_{cache_meta.get(ticker, {}).get('source_family') or 'PUBLIC'}"
        elif bars >= min_bars and lag <= max_stale_sessions:
            tier = f"DATABASE_STALE_USABLE_{cache_meta.get(ticker, {}).get('source_family') or 'PUBLIC'}"
        elif bars >= min_bars:
            tier = f"DATABASE_EXPIRED_STALE_{cache_meta.get(ticker, {}).get('source_family') or 'PUBLIC'}"
        else:
            tier = "UNAVAILABLE"
        source_tiers[ticker] = tier
        if bars < min_bars:
            state = "MISSING_OR_INSUFFICIENT"
        elif lag <= 0:
            state = "CURRENT"
        elif lag <= max_stale_sessions:
            state = "STALE_USABLE"
        else:
            state = "EXPIRED_STALE"
        refresh_states[ticker] = state
        error = live_errors.get(ticker, "")
        if ticker in refresh_targets or ticker in legacy_compact_candidates:
            persist_histories[ticker] = merged
            if error:
                persist_errors[ticker] = error
        audit_rows.append({
            "ticker": ticker,
            "provider": "DATABASE_FIRST_OHLCV",
            "status": state,
            "bars": bars,
            "last_bar_date": json_safe(merged.index[-1]) if not merged.empty else None,
            "expected_session": expected.date().isoformat(),
            "session_lag": int(lag),
            "refresh_attempted": ticker in refresh_targets,
            "live_refresh_success": live_ok,
            "source_tier": tier,
            "error": error,
        })

    write_audit = pd.DataFrame()
    if persist_histories and hasattr(bridge, "write_ohlcv_cache"):
        try:
            write_audit = bridge.write_ohlcv_cache(
                persist_histories,
                source_tiers=source_tiers,
                refresh_states=refresh_states,
                errors=persist_errors,
                max_bars=900,
            )
        except Exception as exc:
            write_audit = pd.DataFrame([{
                "provider": "SUPABASE_OHLCV", "status": "WRITE_FAIL_SOFT",
                "error": f"{type(exc).__name__}: {str(exc)[:240]}",
            }])
    # Acquisition, cache-read and persistence are different facts. Older code
    # concatenated them as competing ``status`` rows; downstream
    # ``drop_duplicates(..., keep='last')`` then selected WRITE_FAILED as the
    # market-data state and erased source tier/session lag.
    combined_audit = pd.DataFrame(audit_rows)
    if not combined_audit.empty:
        combined_audit["acquisition_status"] = combined_audit["status"]
        combined_audit["audit_scope"] = "TICKER"
    global_rows: list[dict[str, Any]] = []
    for extra, prefix in ((cache_audit, "database_cache_"), (write_audit, "database_write_")):
        if not isinstance(extra, pd.DataFrame) or extra.empty:
            continue
        local = extra.copy()
        if "ticker" in local.columns:
            ticker_mask = local["ticker"].notna() & local["ticker"].astype(str).str.strip().ne("")
            ticker_rows = local.loc[ticker_mask].copy()
            if not ticker_rows.empty:
                ticker_rows["ticker"] = ticker_rows["ticker"].map(_ticker)
                ticker_rows = ticker_rows.drop_duplicates("ticker", keep="last")
                ticker_rows = ticker_rows.rename(columns={
                    column: f"{prefix}{column}" for column in ticker_rows.columns if column != "ticker"
                })
                combined_audit = combined_audit.merge(ticker_rows, on="ticker", how="left")
            local = local.loc[~ticker_mask].copy()
        if not local.empty:
            for source in local.to_dict("records"):
                global_rows.append({
                    "ticker": pd.NA,
                    "audit_scope": "GLOBAL",
                    **{f"{prefix}{key}": value for key, value in source.items() if key != "ticker"},
                })
    if global_rows:
        combined_audit = pd.concat([combined_audit, pd.DataFrame(global_rows)], ignore_index=True, sort=False)
    return histories, report, combined_audit


def _database_first_benchmark(
    bridge: ScannerDatabaseBridge,
    *,
    period: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbol = "^JKSE"
    expected = _expected_completed_session()
    try:
        cached, audit = bridge.read_ohlcv_cache([symbol], min_bars=120)
    except Exception as exc:
        cached, audit = {}, pd.DataFrame([{
            "ticker": symbol, "provider": "SUPABASE_OHLCV", "status": "READ_FAIL_SOFT",
            "error": f"{type(exc).__name__}: {str(exc)[:240]}",
        }])
    frame = clean_ohlcv(cached.get(symbol), strict=True) if symbol in cached else pd.DataFrame()
    lag = idx_session_lag(frame.index[-1], expected) if not frame.empty else 9999
    live = pd.DataFrame()
    error = ""
    if len(frame) < 120 or lag > 0:
        if not frame.empty:
            try:
                seed_daily_ohlcv_cache(symbol, frame, "DATABASE")
            except Exception:
                pass
        try:
            live = download_benchmark(period=period)
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
    merged = _merge_ohlcv(frame, live)
    final_lag = idx_session_lag(merged.index[-1], expected) if not merged.empty else 9999
    state = "CURRENT" if final_lag <= 0 else "STALE_USABLE" if len(merged) >= 120 else "MISSING_OR_INSUFFICIENT"
    if (not live.empty or error) and hasattr(bridge, "write_ohlcv_cache"):
        try:
            bridge.write_ohlcv_cache(
                {symbol: merged},
                source_tiers={symbol: "YAHOO_BENCHMARK" if not live.empty else "DATABASE_STALE_USABLE"},
                refresh_states={symbol: state},
                errors={symbol: error} if error else {},
                max_bars=900,
            )
        except Exception:
            pass
    row = pd.DataFrame([{
        "ticker": symbol, "provider": "DATABASE_FIRST_BENCHMARK", "status": state,
        "bars": len(merged), "last_bar_date": json_safe(merged.index[-1]) if not merged.empty else None,
        "expected_session": expected.date().isoformat(), "session_lag": int(final_lag), "error": error,
    }])
    frames = [x for x in (audit, row) if isinstance(x, pd.DataFrame) and not x.empty]
    return merged, pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()



_JOB_BENCHMARK_CACHE: dict[tuple[str, str, str], tuple[pd.DataFrame, pd.DataFrame]] = {}
# Durable jobs are normally processed by one server worker. Evidence does not
# change between adjacent technical chunks, so load it once per job instead of
# issuing 6+ Supabase reads for every 20 tickers. A host recycle simply rebuilds
# this cache from the durable database.
_JOB_EVIDENCE_CACHE: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
_JOB_FORWARD_CACHE: dict[str, pd.DataFrame] = {}


def _subset_tickers(frame: pd.DataFrame | None, tickers: Sequence[str]) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty or "ticker" not in frame.columns:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    names = set(_ticker(value) for value in tickers if _ticker(value))
    if not names:
        return frame.iloc[0:0].copy()
    normalized = frame["ticker"].map(_ticker)
    return frame.loc[normalized.isin(names)].copy().reset_index(drop=True)


def _universe_metadata(
    records: Sequence[Mapping[str, Any]] | None,
    universe: Sequence[str],
) -> pd.DataFrame:
    """Return conservative metadata-only issuer evidence from the upload."""
    names = set(_ticker(value) for value in universe if _ticker(value))
    rows: list[dict[str, Any]] = []
    for raw in records or []:
        if not isinstance(raw, Mapping):
            continue
        normalized = {
            re.sub(r"[^a-z0-9]+", "_", str(key or "").strip().lower()).strip("_"): value
            for key, value in raw.items()
        }

        def first(*keys: str) -> Any:
            for key in keys:
                value = normalized.get(key)
                if value is None:
                    continue
                try:
                    if bool(pd.isna(value)):
                        continue
                except (TypeError, ValueError):
                    pass
                if str(value).strip():
                    return value
            return ""

        ticker = _ticker(first("ticker", "yahoo_ticker", "symbol", "kode"))
        if not ticker or ticker not in names:
            continue
        sector = first("idx_sector", "sector_idx_ic", "idx_ic_sector", "sector")
        rows.append({
            "ticker": ticker,
            "idx_sector": sector,
            "sector_idx_ic": sector,
            "universe_rank": first("rank_universe", "universe_rank"),
            "universe_role": first("universe_role", "role"),
            "universe_priority": first("priority", "universe_priority"),
            "universe_active_scan": first("active_scan", "scan_active"),
            "universe_metadata_source": "UPLOADED_UNIVERSE",
        })
    if not rows:
        return pd.DataFrame()
    return normalize_fundamental_classification(
        pd.DataFrame(rows).drop_duplicates("ticker", keep="first")
    )


def _job_evidence(
    job_id: str,
    bridge: ScannerDatabaseBridge,
    universe: Sequence[str],
    cfg: ScanConfig,
    *,
    include_narrative_history: bool = True,
    universe_records: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    key = str(job_id or "")
    cached = _JOB_EVIDENCE_CACHE.get(key)
    if cached is not None:
        return tuple(frame.copy() for frame in cached)  # type: ignore[return-value]
    # These read families are independent. Parallel loading cuts warm-job startup
    # latency without increasing the number of database calls.
    with ThreadPoolExecutor(max_workers=4) as pool:
        fundamental_future = pool.submit(_load_fundamentals, bridge, universe, cfg)
        aux_future = pool.submit(_load_aux, bridge, universe, cfg)
        def _safe_events():
            try:
                return bridge.read_narrative_events(universe, limit=max(1000, len(universe) * 20))
            except Exception:
                return pd.DataFrame()
        def _safe_outcomes():
            try:
                return bridge.read_narrative_event_outcomes(universe, limit=max(1000, len(universe) * 20))
            except Exception:
                return pd.DataFrame()
        event_future = pool.submit(_safe_events) if include_narrative_history else None
        outcome_future = pool.submit(_safe_outcomes) if include_narrative_history else None
        fundamentals, history, report = fundamental_future.result()
        market, news = aux_future.result()
        events = event_future.result() if event_future is not None else pd.DataFrame()
        outcomes = outcome_future.result() if outcome_future is not None else pd.DataFrame()
    metadata = _universe_metadata(universe_records, universe)
    if not metadata.empty:
        fundamentals = _mark_history_eligible(
            _coalesce_primary_evidence(fundamentals, metadata)
        )
    fundamentals = enrich_fundamental_evidence(fundamentals)
    value = (fundamentals, history, report, market, news, events, outcomes)
    _JOB_EVIDENCE_CACHE[key] = tuple(frame.copy() for frame in value)  # type: ignore[assignment]
    while len(_JOB_EVIDENCE_CACHE) > 4:
        _JOB_EVIDENCE_CACHE.pop(next(iter(_JOB_EVIDENCE_CACHE)))
    return tuple(frame.copy() for frame in value)  # type: ignore[return-value]


def _job_forward_quality(
    job_id: str,
    bridge: ScannerDatabaseBridge,
    universe: Sequence[str],
) -> pd.DataFrame:
    """Load durable project/management evidence once per resumable job."""
    key = str(job_id or "")
    cached = _JOB_FORWARD_CACHE.get(key)
    if cached is not None:
        return cached.copy()
    frame = pd.DataFrame()
    if hasattr(bridge, "read_forward_quality_cache"):
        try:
            value, _ = bridge.read_forward_quality_cache(universe)
            frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame()
        except Exception:
            frame = pd.DataFrame()
    _JOB_FORWARD_CACHE[key] = frame.copy()
    while len(_JOB_FORWARD_CACHE) > 4:
        _JOB_FORWARD_CACHE.pop(next(iter(_JOB_FORWARD_CACHE)))
    return frame.copy()


def _job_benchmark(
    job_id: str,
    bridge: ScannerDatabaseBridge,
    *,
    period: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reuse the same benchmark snapshot for every chunk of one durable job.

    The benchmark cannot legitimately change between adjacent 20-ticker chunks
    of the same EOD job.  v9.7.0 still re-read/revalidated it for every chunk.
    """
    session_key = _expected_completed_session().date().isoformat()
    key = (str(job_id), str(period), session_key)
    cached = _JOB_BENCHMARK_CACHE.get(key)
    if cached is not None:
        return cached[0].copy(), cached[1].copy()
    benchmark, report = _database_first_benchmark(bridge, period=period)
    _JOB_BENCHMARK_CACHE[key] = (benchmark.copy(), report.copy())
    # Bound process memory when many historical jobs are viewed in one host.
    while len(_JOB_BENCHMARK_CACHE) > 8:
        _JOB_BENCHMARK_CACHE.pop(next(iter(_JOB_BENCHMARK_CACHE)))
    return benchmark, report


def _refresh_missing_daily_evidence(
    bridge: ScannerDatabaseBridge,
    tickers: Sequence[str],
    fundamentals: pd.DataFrame,
    history: pd.DataFrame,
    market: pd.DataFrame,
    news: pd.DataFrame,
    cfg: ScanConfig,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    names = list(dict.fromkeys(_ticker(value) for value in tickers if _ticker(value)))
    reports: list[pd.DataFrame] = []
    if not hasattr(bridge, "persist_scan_result"):
        return fundamentals, history, market, news, pd.DataFrame()

    fundamentals = normalize_fundamental_classification(_mark_history_eligible(fundamentals)) if not fundamentals.empty else fundamentals
    counts = _history_counts(history)
    fundamental_map = (
        fundamentals.drop_duplicates("ticker", keep="last").set_index("ticker").to_dict("index")
        if isinstance(fundamentals, pd.DataFrame) and not fundamentals.empty and "ticker" in fundamentals.columns
        else {}
    )

    def refresh_priority(ticker: str) -> tuple[int, float, str]:
        row = fundamental_map.get(ticker, {})
        eligible = bool(row.get("fundamental_score_eligible", False))
        sector_missing = str(row.get("sector") or "UNKNOWN").upper() == "UNKNOWN"
        age = _finite(row.get("statement_age_days"), np.nan)
        history_missing = counts.get(ticker, 0) < 2
        calendar_due = bool(reporting_refresh_profile(row).get("fundamental_refresh_due", False))
        # Lower tuple sorts first. A quarterly reporting window that has opened
        # is a real refresh condition even when the old statement is <210 days
        # old. Hotfix 3 promoted those names in the shortlist but this inner
        # filter still discarded them, making the calibration ineffective.
        if calendar_due:
            bucket = 0
        elif sector_missing:
            bucket = 1
        elif not eligible:
            bucket = 2
        elif history_missing:
            bucket = 3
        elif not np.isfinite(age) or age > 210:
            bucket = 4
        else:
            bucket = 9
        age_rank = -age if np.isfinite(age) else -9999.0
        return bucket, age_rank, ticker

    refresh_candidates = [ticker for ticker in names if refresh_priority(ticker)[0] < 9]
    refresh_candidates.sort(key=refresh_priority)
    fundamental_limit = max(0, _int_config(config, "daily_fundamental_refresh_limit", 6))
    targets = refresh_candidates[:fundamental_limit]
    if targets:
        # One-button database maintenance is official-first.  The normal scan
        # repairs missing/stale statement history itself instead of requiring a
        # separate backfill mode. Yahoo is used only when IDX history remains
        # insufficient, preserving both source quality and warm-scan speed.
        official_limit = max(0, _int_config(config, "daily_official_fundamental_refresh_limit", fundamental_limit))
        official_targets = targets[:official_limit]
        idx_history = pd.DataFrame()
        idx_report = pd.DataFrame()
        if official_targets:
            try:
                idx_history, idx_report = fetch_idx_fundamental_history(
                    official_targets,
                    max_tickers=len(official_targets),
                    years_back=max(1, int(getattr(cfg, "idx_fundamental_years_back", 2))),
                    timeout=4,
                    max_workers=min(8, max(1, len(official_targets))),
                )
            except Exception as exc:
                idx_report = pd.DataFrame([{
                    "provider": "IDX_OFFICIAL", "status": "FAIL_SOFT",
                    "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                }])
        history_after_idx = combine_fundamental_history(history, idx_history)
        yahoo_targets = select_yahoo_fundamental_tickers(
            targets,
            history_after_idx,
            max_tickers=len(targets),
            crosscheck_top_n=0,
            min_official_periods=2,
        )
        yahoo_history, yahoo_report = fetch_yahoo_fundamental_history(
            yahoo_targets,
            max_workers=min(6, max(1, len(yahoo_targets))),
            max_tickers=len(yahoo_targets),
            enable_yfinance_fallback=False,
        ) if yahoo_targets else (pd.DataFrame(), pd.DataFrame())
        history = combine_fundamental_history(history_after_idx, yahoo_history)

        snapshot_limit = max(0, _int_config(config, "daily_snapshot_refresh_limit", 4))
        # Snapshot fetch repairs sector/company identity and valuation fields;
        # prioritize UNKNOWN sectors, then the remaining stale targets.
        sector_targets = [t for t in targets if str(fundamental_map.get(t, {}).get("sector") or "UNKNOWN").upper() == "UNKNOWN"]
        snapshot_targets = list(dict.fromkeys(sector_targets + targets))[:snapshot_limit]
        live_snapshot = fetch_resilient_fundamentals(snapshot_targets, cfg) if snapshot_targets else pd.DataFrame()
        fundamentals = normalize_fundamental_classification(
            _mark_history_eligible(
                canonicalize_official_fundamental_evidence(
                    enrich_fundamentals_with_history(_coalesce_primary_evidence(live_snapshot, fundamentals), history)
                )
            )
        )
        for report in (idx_report, yahoo_report):
            if isinstance(report, pd.DataFrame) and not report.empty:
                reports.append(report)

    market_present = set(market.get("ticker", pd.Series(dtype=str)).map(_ticker)) if not market.empty else set()
    market_limit = max(0, _int_config(config, "daily_market_refresh_limit", 6))
    market_targets = [ticker for ticker in names if ticker not in market_present][:market_limit]
    if market_targets:
        market_live = fetch_resilient_market_status(market_targets, cfg)
        market = _merge_primary(market_live, market)

    news_present = set(news.get("ticker", pd.Series(dtype=str)).map(_ticker)) if not news.empty else set()
    news_limit = max(0, _int_config(config, "daily_news_refresh_limit", 5))
    news_targets = [ticker for ticker in names if ticker not in news_present][:news_limit]
    if news_targets:
        news_live = fetch_resilient_news_review(news_targets, lookback_days=7, config=cfg)
        news = _merge_primary(news_live, news)

    if targets or market_targets or news_targets:
        try:
            sync = bridge.persist_scan_result({
                "mode": "daily_delta_refresh",
                "scanner_version": ENGINE_VERSION,
                "fundamentals": fundamentals,
                "fundamental_history": history,
                "fundamental_history_report": pd.concat(reports, ignore_index=True, sort=False) if reports else pd.DataFrame(),
                "market_status": market,
                "news_review": news,
                "focus_screens": {},
                "prepared": {},
                "ticker_count": len(names),
            }, tables=(
                "fundamental_snapshots", "fundamental_cache",
                "fundamental_history_cache", "refresh_state",
                "source_events", "provider_health",
            ))
            if isinstance(sync, pd.DataFrame) and not sync.empty:
                reports.append(sync.assign(source_family="DATABASE_DELTA_SYNC"))
        except Exception as exc:
            reports.append(pd.DataFrame([{
                "provider": "DATABASE_DELTA_SYNC", "status": "FAIL_SOFT",
                "error": f"{type(exc).__name__}: {str(exc)[:240]}",
            }]))
    return fundamentals, history, market, news, pd.concat(reports, ignore_index=True, sort=False) if reports else pd.DataFrame()






def _breadth_row(frame: pd.DataFrame | None) -> dict[str, Any]:
    if frame is None or frame.empty or "Close" not in frame.columns:
        return {"valid": False}
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if len(close) < 60:
        return {"valid": False}
    last = float(close.iloc[-1])
    ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
    ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1]) if len(close) >= 200 else np.nan
    ret20 = float(close.iloc[-1] / close.iloc[-21] - 1.0) if len(close) >= 21 and close.iloc[-21] else np.nan
    return {
        "valid": True,
        "above_ema50": bool(last > ema50),
        "above_ema200": bool(last > ema200) if np.isfinite(ema200) else False,
        "positive_20d": bool(ret20 > 0) if np.isfinite(ret20) else False,
        "last_price": last,
        "last_date": json_safe(frame.index[-1]),
    }


def process_daily_scan_chunk(
    job: Mapping[str, Any],
    items: pd.DataFrame,
    worker_id: str,
    *,
    runtime: Mapping[str, str] | None = None,
    bridge_override: ScannerDatabaseBridge | None = None,
) -> Mapping[str, ItemOutcome]:
    runtime = dict(runtime or {})
    config = dict(job.get("config_payload") or {})
    cfg = _cfg(config)
    tickers = [_ticker(value) for value in items.get("ticker", pd.Series(dtype=str)).tolist()]
    tickers = [value for value in tickers if value]
    period = str(config.get("period", "5y") or "5y")
    bridge = bridge_override or ScannerDatabaseBridge()
    # Stage 1 is intentionally cache-first and provider-light. Deep fundamental,
    # official IDX/XBRL, market-status and news repair is performed ONCE in the
    # finalizer for a ranked evidence shortlist. Job-level evidence is loaded once
    # for the whole universe and sliced in memory for adjacent technical chunks.
    universe = [_ticker(value) for value in (job.get("universe_payload") or [])]
    full_fundamentals, full_history, full_report, full_market, full_news, full_events, full_outcomes = _job_evidence(
        str(job.get("job_id") or ""), bridge, universe or tickers, cfg,
        include_narrative_history=not bool(config.get("lean_skip_narrative_history", False)),
        universe_records=config.get("universe_records") if isinstance(config.get("universe_records"), list) else None,
    )
    full_forward = _job_forward_quality(
        str(job.get("job_id") or ""), bridge, universe or tickers,
    )
    fundamentals = _subset_tickers(full_fundamentals, tickers)
    chunk_history = _subset_tickers(full_history, tickers)
    chunk_fundamental_report = _subset_tickers(full_report, tickers) if "ticker" in full_report.columns else full_report.copy()
    market = _subset_tickers(full_market, tickers)
    news = _subset_tickers(full_news, tickers)
    existing_events = _subset_tickers(full_events, tickers)
    existing_outcomes = _subset_tickers(full_outcomes, tickers)
    project_management = _subset_tickers(full_forward, tickers)

    histories, download_report, ohlcv_report = _database_first_ohlcv(
        bridge, tickers, period=period,
        itick_api_token=str(runtime.get("itick_api_token", "") or ""),
        min_bars=260, max_stale_sessions=5,
    )
    benchmark, benchmark_report = _job_benchmark(str(job.get("job_id") or ""), bridge, period=period)
    bounded = {ticker: frame.tail(800).copy() for ticker, frame in histories.items() if isinstance(frame, pd.DataFrame) and not frame.empty}
    core = ScanEngine(cfg).scan(bounded, benchmark.tail(800).copy() if isinstance(benchmark, pd.DataFrame) else benchmark)
    fundamentals = enrich_fundamentals_with_valuation(
        enrich_fundamental_evidence(fundamentals),
        core.get("prepared", {}),
    )
    base = _merge_primary(core.get("signals", pd.DataFrame()), core.get("universe", pd.DataFrame()))
    lineage_tiers: dict[str, str] = {}
    if isinstance(ohlcv_report, pd.DataFrame) and not ohlcv_report.empty and {"ticker", "source_tier"}.issubset(ohlcv_report.columns):
        lineage_tiers = {
            str(row.get("ticker", "")).upper().strip(): str(row.get("source_tier") or "")
            for row in ohlcv_report.dropna(subset=["ticker"]).drop_duplicates("ticker", keep="last").to_dict("records")
            if str(row.get("ticker", "")).strip()
        }
    live_tiers = dict(getattr(download_report, "source_tiers", {}) or {})
    lineage_tiers.update({str(key).upper().strip(): str(value or "") for key, value in live_tiers.items()})
    if not base.empty:
        base = apply_universe_integrity_gate(base, tickers, core.get("prepared", {}).keys(), cfg)
        base = attach_ohlcv_source_lineage(base, lineage_tiers)
        base = apply_news_gate(
            apply_market_status_gate(
                apply_fundamental_gate(attach_fundamentals(base, fundamentals), cfg),
                market,
                cfg,
            ),
            news,
            cfg,
        )
    silent_profiles, silent_frame = build_silent_profiles(core.get("prepared", {}))
    narrative = build_narrative_intelligence(
        prepared=core.get("prepared", {}),
        fundamentals=fundamentals,
        news_review=news,
        project_management=project_management,
        market_status=market,
        existing_events=existing_events,
        existing_outcomes=existing_outcomes,
        benchmark=benchmark,
        silent_profiles=silent_profiles,
        scan_config=cfg,
    )
    signal_map = base.set_index("ticker").to_dict("index") if not base.empty and "ticker" in base.columns else {}
    silent_map = silent_frame.set_index("ticker").to_dict("index") if not silent_frame.empty else {}
    profiles = narrative.get("profiles", pd.DataFrame())
    profile_map = profiles.set_index("ticker").to_dict("index") if isinstance(profiles, pd.DataFrame) and not profiles.empty else {}
    events = narrative.get("events", pd.DataFrame())
    outcomes = narrative.get("outcomes", pd.DataFrame())
    ohlcv_meta: dict[str, dict[str, Any]] = {}
    if isinstance(ohlcv_report, pd.DataFrame) and not ohlcv_report.empty and "ticker" in ohlcv_report.columns:
        ohlcv_meta = ohlcv_report.drop_duplicates("ticker", keep="last").set_index("ticker").to_dict("index")
    item_lookup: dict[str, dict[str, Any]] = {}
    for row in items.to_dict("records"):
        normalized = _ticker(row.get("ticker"))
        if normalized:
            item_lookup[normalized] = dict(row)
    result: dict[str, ItemOutcome] = {}
    for ticker in tickers:
        item_row = item_lookup.get(ticker, {})
        key = str(item_row.get("item_key", ""))
        if not key:
            continue
        frame = core.get("prepared", {}).get(ticker)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            attempts = int(item_row.get("attempt_count", 0) or 0)
            max_attempts = max(1, int(item_row.get("max_attempts", 2) or 2))
            partial_payload = {
                "ticker": ticker,
                "completion_state": "TECHNICAL_UNAVAILABLE",
                "technical_ready": False,
                "acquisition_error": str(ohlcv_meta.get(ticker, {}).get("error") or "OHLCV_NOT_PREPARED"),
                "ohlcv_state": str(ohlcv_meta.get(ticker, {}).get("status") or "MISSING"),
                "ohlcv_bars": int(ohlcv_meta.get(ticker, {}).get("bars") or 0),
                "ohlcv_last_bar_date": ohlcv_meta.get(ticker, {}).get("last_bar_date"),
                "ohlcv_session_lag": ohlcv_meta.get(ticker, {}).get("session_lag"),
                "ohlcv_source_tier": str(ohlcv_meta.get(ticker, {}).get("source_tier") or "UNAVAILABLE"),
                "ohlcv_database_cache_status": ohlcv_meta.get(ticker, {}).get("database_cache_status"),
                "ohlcv_database_write_status": ohlcv_meta.get(ticker, {}).get("database_write_status"),
                "ohlcv_database_rows_written": ohlcv_meta.get(ticker, {}).get("database_write_rows_written"),
                "ohlcv_database_write_error": ohlcv_meta.get(ticker, {}).get("database_write_error"),
                "worker_id": worker_id,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            if attempts >= max_attempts:
                # Provider exhaustion is a data-availability state, not a system
                # failure. Checkpoint it as COMPLETE so the rest of the universe
                # can rank, while keeping the ticker explicitly unrankable.
                result[key] = ItemOutcome(True, payload=partial_payload)
            else:
                result[key] = ItemOutcome(
                    False,
                    payload=partial_payload,
                    error="OHLCV_NOT_PREPARED",
                    retry_delay_seconds=30,
                )
            continue
        ticker_events = events.loc[events["ticker"].astype(str).eq(ticker)].to_dict("records") if isinstance(events, pd.DataFrame) and not events.empty and "ticker" in events else []
        ticker_outcomes = outcomes.loc[outcomes["ticker"].astype(str).eq(ticker)].to_dict("records") if isinstance(outcomes, pd.DataFrame) and not outcomes.empty and "ticker" in outcomes else []
        payload = {
            "ticker": ticker,
            "completion_state": "TECHNICAL_READY",
            "technical_ready": True,
            "signal": json_safe({"ticker": ticker, **signal_map.get(ticker, {"status": "PENDING_DATA"})}),
            "silent_profile": json_safe({"ticker": ticker, **silent_map.get(ticker, {})}),
            "narrative_profile": json_safe({"ticker": ticker, **profile_map.get(ticker, {})}),
            "narrative_events": json_safe(ticker_events),
            "narrative_outcomes": json_safe(ticker_outcomes),
            "fundamental_snapshot": json_safe({"ticker": ticker, **(
                fundamentals.loc[fundamentals["ticker"].map(_ticker).eq(ticker)].drop_duplicates("ticker", keep="last").iloc[-1].to_dict()
                if isinstance(fundamentals, pd.DataFrame) and not fundamentals.empty and "ticker" in fundamentals.columns
                and fundamentals["ticker"].map(_ticker).eq(ticker).any()
                else {}
            )}),
            "breadth": json_safe(_breadth_row(frame)),
            "ohlcv_state": str(ohlcv_meta.get(ticker, {}).get("status") or "READY"),
            "ohlcv_bars": int(len(frame)),
            "ohlcv_last_bar_date": json_safe(frame.index[-1]),
            "ohlcv_session_lag": ohlcv_meta.get(ticker, {}).get("session_lag"),
            "ohlcv_source_tier": str(ohlcv_meta.get(ticker, {}).get("source_tier") or "DATABASE_OR_PUBLIC"),
            "ohlcv_database_cache_status": ohlcv_meta.get(ticker, {}).get("database_cache_status"),
            "ohlcv_database_write_status": ohlcv_meta.get(ticker, {}).get("database_write_status"),
            "ohlcv_database_rows_written": ohlcv_meta.get(ticker, {}).get("database_write_rows_written"),
            "ohlcv_database_write_error": ohlcv_meta.get(ticker, {}).get("database_write_error"),
            "worker_id": worker_id,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        result[key] = ItemOutcome(True, payload=payload)
    return result


def _unpack_job_items(
    items: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, float], list[str], dict[str, Any]]:
    signal_rows: list[dict[str, Any]] = []
    silent_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    fundamental_rows: list[dict[str, Any]] = []
    technical_tickers: list[str] = []
    processed_tickers: list[str] = []
    unavailable_tickers: list[str] = []
    breadth_rows: list[dict[str, Any]] = []
    ohlcv_state_counts: dict[str, int] = {}
    ohlcv_ready_tickers: list[str] = []
    for row in items.to_dict("records"):
        if str(row.get("status", "")).upper() != "COMPLETE":
            continue
        payload = row.get("result_payload") or {}
        if not isinstance(payload, Mapping):
            continue
        ticker = _ticker(payload.get("ticker") or row.get("ticker"))
        if not ticker:
            continue
        processed_tickers.append(ticker)
        ohlcv_state = str(payload.get("ohlcv_state") or "UNKNOWN").upper()
        ohlcv_state_counts[ohlcv_state] = ohlcv_state_counts.get(ohlcv_state, 0) + 1
        if int(payload.get("ohlcv_bars") or 0) >= 260:
            ohlcv_ready_tickers.append(ticker)
        signal = payload.get("signal")
        completion_state = str(payload.get("completion_state", "")).upper()
        technical_ready = bool(payload.get("technical_ready", isinstance(signal, Mapping)))
        if completion_state == "TECHNICAL_UNAVAILABLE":
            technical_ready = False
        if not technical_ready:
            unavailable_tickers.append(ticker)
            continue
        technical_tickers.append(ticker)
        if isinstance(signal, Mapping):
            # set_index(...).to_dict("index") removes the index key. Restore
            # ticker defensively so final execution gates and ranking merges
            # always receive the required primary key.
            signal_row = dict(signal)
            signal_row["ticker"] = ticker
            signal_rows.append(signal_row)
        silent = payload.get("silent_profile")
        if isinstance(silent, Mapping):
            silent_rows.append(dict(silent))
        profile = payload.get("narrative_profile")
        if isinstance(profile, Mapping):
            profile_rows.append(dict(profile))
        if isinstance(payload.get("narrative_events"), list):
            event_rows.extend([dict(v) for v in payload["narrative_events"] if isinstance(v, Mapping)])
        if isinstance(payload.get("narrative_outcomes"), list):
            outcome_rows.extend([dict(v) for v in payload["narrative_outcomes"] if isinstance(v, Mapping)])
        fundamental = payload.get("fundamental_snapshot")
        if isinstance(fundamental, Mapping):
            fundamental_row = dict(fundamental)
            fundamental_row["ticker"] = ticker
            fundamental_rows.append(fundamental_row)
        breadth = payload.get("breadth")
        if isinstance(breadth, Mapping) and breadth.get("valid"):
            breadth_rows.append(dict(breadth))
    count = len(breadth_rows)
    breadth_features = {
        "breadth_above_ema50_pct": 100.0 * sum(bool(row.get("above_ema50")) for row in breadth_rows) / count if count else np.nan,
        "breadth_above_ema200_pct": 100.0 * sum(bool(row.get("above_ema200")) for row in breadth_rows) / count if count else np.nan,
        "breadth_positive_20d_pct": 100.0 * sum(bool(row.get("positive_20d")) for row in breadth_rows) / count if count else np.nan,
        "breadth_sample": float(count),
    }
    narrative = {
        "profiles": pd.DataFrame(profile_rows).drop_duplicates("ticker", keep="last") if profile_rows else pd.DataFrame(),
        "events": pd.DataFrame(event_rows),
        "outcomes": pd.DataFrame(outcome_rows),
        "audit": pd.DataFrame([{
            "state": "RESUMED_FROM_DURABLE_CHUNKS",
            "processed_ticker_count": len(set(processed_tickers)),
            "technical_ready_count": len(set(technical_tickers)),
            "technical_unavailable_count": len(set(unavailable_tickers)),
        }]),
        "fundamentals": pd.DataFrame(fundamental_rows).drop_duplicates("ticker", keep="last") if fundamental_rows else pd.DataFrame(),
    }
    item_audit = {
        "processed_tickers": list(dict.fromkeys(processed_tickers)),
        "technical_tickers": list(dict.fromkeys(technical_tickers)),
        "technical_unavailable_tickers": list(dict.fromkeys(unavailable_tickers)),
        "ohlcv_ready_tickers": list(dict.fromkeys(ohlcv_ready_tickers)),
        "ohlcv_state_counts": dict(ohlcv_state_counts),
    }
    return (
        pd.DataFrame(signal_rows),
        pd.DataFrame(silent_rows),
        narrative,
        breadth_features,
        list(dict.fromkeys(technical_tickers)),
        item_audit,
    )


def _ohlcv_item_audit(items: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if items is None or items.empty:
        return pd.DataFrame()
    for row in items.to_dict("records"):
        payload = row.get("result_payload") or {}
        if not isinstance(payload, Mapping):
            payload = {}
        ticker = _ticker(payload.get("ticker") or row.get("ticker"))
        rows.append({
            "ticker": ticker,
            "item_status": str(row.get("status") or ""),
            "completion_state": str(payload.get("completion_state") or ""),
            "technical_ready": bool(payload.get("technical_ready", False)),
            "ohlcv_state": str(payload.get("ohlcv_state") or "UNKNOWN"),
            "ohlcv_bars": int(payload.get("ohlcv_bars") or 0),
            "ohlcv_last_bar_date": payload.get("ohlcv_last_bar_date"),
            "ohlcv_session_lag": payload.get("ohlcv_session_lag"),
            "ohlcv_source_tier": payload.get("ohlcv_source_tier"),
            "acquisition_error": payload.get("acquisition_error"),
            "database_cache_status": payload.get("ohlcv_database_cache_status"),
            "database_write_status": payload.get("ohlcv_database_write_status"),
            "database_rows_written": payload.get("ohlcv_database_rows_written"),
            "database_write_error": payload.get("ohlcv_database_write_error"),
            "attempt_count": int(row.get("attempt_count") or 0),
        })
    return pd.DataFrame(rows)


def finalize_daily_scan_job(
    job: Mapping[str, Any],
    bridge: ScannerDatabaseBridge,
    worker_id: str,
    *,
    runtime: Mapping[str, str] | None = None,
    items_override: pd.DataFrame | None = None,
    durable_updates: bool = True,
    persist_artifacts: bool = True,
    return_result: bool = False,
) -> Mapping[str, Any]:
    runtime = dict(runtime or {})
    config = dict(job.get("config_payload") or {})
    cfg = _cfg(config)
    job_id = str(job.get("job_id"))
    finalizer_started = time.perf_counter()
    stage_timings: dict[str, float] = {}
    stage_mark = finalizer_started

    def _mark_stage(name: str) -> None:
        nonlocal stage_mark
        now = time.perf_counter()
        stage_timings[name] = round(now - stage_mark, 3)
        stage_mark = now

    def _update_job(*args: Any, **kwargs: Any) -> Any:
        if not durable_updates:
            return {}
        return bridge.update_scan_job(*args, **kwargs)

    def _persist_job_artifacts(*args: Any, **kwargs: Any) -> Any:
        if not (durable_updates and persist_artifacts):
            return pd.DataFrame()
        return bridge.persist_scan_job_artifacts_batch(*args, **kwargs)

    items = items_override.copy() if isinstance(items_override, pd.DataFrame) else bridge.read_scan_job_items(job_id, phase="TECHNICAL", include_payload=True, limit=5000)
    signals, silent_frame, narrative, breadth_features, completed_tickers, item_audit = _unpack_job_items(items)
    # Legacy/minimal checkpoints may omit fields that execution gates expect.
    # Restore a safe non-executable signal contract instead of crashing the finalizer.
    if not signals.empty:
        if "ticker" not in signals.columns:
            signals["ticker"] = pd.Series(dtype=str)
        signals["ticker"] = signals["ticker"].map(_ticker)
        if "status" not in signals.columns:
            signals["status"] = "WATCHLIST"
        else:
            signals["status"] = signals["status"].astype("string").fillna("WATCHLIST").astype(str)
        if "decision_state" not in signals.columns:
            signals["decision_state"] = signals["status"]
        if "setup_status" not in signals.columns:
            signals["setup_status"] = signals["status"]
    universe = [_ticker(value) for value in (job.get("universe_payload") or [])]
    fundamentals, history, fundamental_report, market, news, cached_events, cached_outcomes = _job_evidence(
        job_id, bridge, universe, cfg,
        include_narrative_history=not bool(config.get("lean_skip_narrative_history", False)),
        universe_records=config.get("universe_records") if isinstance(config.get("universe_records"), list) else None,
    )
    chunk_fundamentals = narrative.pop("fundamentals", pd.DataFrame())
    if isinstance(chunk_fundamentals, pd.DataFrame) and not chunk_fundamentals.empty:
        fundamentals = _coalesce_primary_evidence(chunk_fundamentals, fundamentals)
    fundamentals = enrich_fundamental_evidence(fundamentals)
    project_management = _job_forward_quality(job_id, bridge, universe)
    # Canonical evidence maps are initialized immediately after the evidence load.
    # Hotfix 3 referenced ``fundamental_map`` while building the refresh-priority
    # lane before the map was assigned later in the function, which produced an
    # UnboundLocalError on the exact cold/partial-cache path used by production.
    # Keep all state needed by ranking/enrichment initialized before any branch.
    fundamental_map = (
        fundamentals.drop_duplicates("ticker", keep="last").set_index("ticker").to_dict("index")
        if isinstance(fundamentals, pd.DataFrame) and not fundamentals.empty and "ticker" in fundamentals.columns
        else {}
    )
    history_counts = _history_counts(history)
    # Chunk payloads hold the narrative produced against the same cached evidence;
    # cached_events/outcomes remain available for a resumed job but are not needed
    # to rebuild every chunk during finalization.
    _mark_stage("LOAD_DURABLE_EVIDENCE")
    period = str(config.get("period", "5y") or "5y")
    benchmark, benchmark_report = _job_benchmark(str(job.get("job_id") or ""), bridge, period=period)
    if bool(config.get('macro_external_enabled', True)):
        try:
            macro_series, macro_report = fetch_macro_series(
                period="6mo", timeout=max(3, int(config.get('macro_timeout_seconds', 5) or 5))
            )
        except Exception as exc:
            # External macro factors are context, never a reason to lose the
            # entire 400-name ranking. IHSG + universe breadth remains usable.
            macro_series = {}
            macro_report = pd.DataFrame([{
                'factor':'EXTERNAL_MACRO','status':'FAIL_SOFT','source':'IHSG_BREADTH_FALLBACK',
                'error':f'{type(exc).__name__}: {str(exc)[:240]}',
            }])
    else:
        macro_series, macro_report = {}, pd.DataFrame([{'factor':'EXTERNAL_MACRO','status':'SKIPPED_LEAN_MODE','source':'IHSG_BREADTH_ONLY'}])
    breadth_sample = int(_finite(breadth_features.get("breadth_sample"), 0.0))
    breadth_coverage_pct = 100.0 * breadth_sample / len(universe) if universe else 0.0
    breadth_features = {**breadth_features, "breadth_coverage_pct": breadth_coverage_pct}
    macro = build_macro_regime(
        benchmark=benchmark,
        prepared={},
        fundamentals=fundamentals,
        macro_series=macro_series,
        source_report=macro_report,
        breadth_features=breadth_features,
    )
    if not macro.snapshot.empty:
        raw_regime = str(macro.snapshot.iloc[0].get("macro_regime", "DATA_PENDING"))
        macro.snapshot["macro_regime_raw"] = raw_regime
        macro.snapshot["macro_breadth_coverage_pct"] = round(breadth_coverage_pct, 1)
        macro.snapshot["macro_breadth_sample"] = breadth_sample
        if breadth_coverage_pct < 50.0:
            macro.snapshot["macro_regime"] = f"PROVISIONAL_{raw_regime}"
            macro.snapshot["macro_regime_state"] = "INSUFFICIENT_BREADTH_SAMPLE"
            if not macro.issuer_map.empty and "issuer_macro_alignment_coverage_pct" in macro.issuer_map.columns:
                macro.issuer_map["issuer_macro_alignment_coverage_pct"] = (
                    pd.to_numeric(macro.issuer_map["issuer_macro_alignment_coverage_pct"], errors="coerce").fillna(0.0)
                    * max(0.25, breadth_coverage_pct / 100.0)
                ).clip(upper=100.0).round(1)
                macro.issuer_map["issuer_macro_alignment_state"] = "PROVISIONAL_BREADTH"
        else:
            macro.snapshot["macro_regime_state"] = "VALIDATED_BREADTH"
    _mark_stage("MACRO_AND_BENCHMARK")

    # Publish a durable provisional ranking before execution verification. This
    # makes rankings visible while the job is FINALIZING and keeps useful output
    # available even when an external price provider is slow or unavailable.
    provisional_focus = build_simple_focus(
        {},
        fundamentals=fundamentals,
        signals=signals,
        news_review=news,
        market_status=market,
        benchmark=benchmark,
        macro_result=macro,
        config=cfg,
        universe_tickers=completed_tickers,
        precomputed_silent_frame=silent_frame,
        precomputed_narrative=narrative,
    )
    provisional_technical_coverage = 100.0 * len(completed_tickers) / len(universe) if universe else 0.0
    provisional_audit = pd.DataFrame([{
        "requested_tickers": len(universe),
        "processed_tickers": len(item_audit.get("processed_tickers", [])),
        "processed_ticker_list": list(item_audit.get("processed_tickers", [])),
        "technical_ready_tickers": len(completed_tickers),
        "completed_tickers": len(completed_tickers),
        "completed_ticker_list": list(completed_tickers),
        "technical_unavailable_tickers": len(item_audit.get("technical_unavailable_tickers", [])),
        "technical_unavailable_ticker_list": list(item_audit.get("technical_unavailable_tickers", [])),
        "ohlcv_ready_tickers": len(item_audit.get("ohlcv_ready_tickers", [])),
        "ohlcv_ready_ticker_list": list(item_audit.get("ohlcv_ready_tickers", [])),
        "technical_coverage_pct": round(provisional_technical_coverage, 1),
        "breadth_coverage_pct": round(breadth_coverage_pct, 1),
        "ranking_state": "VALID" if provisional_technical_coverage >= 70.0 else "PARTIAL_UNIVERSE",
        "next_leader_final_score_valid": int(len(provisional_focus.get("next_leaders", pd.DataFrame()))),
        "swing_final_score_valid": int(len(provisional_focus.get("swing_ready", pd.DataFrame()))),
        "ranking_basis": "PROVISIONAL_PRE_EXECUTION_VERIFICATION",
        "worker_id": worker_id,
        "published_at": datetime.now(timezone.utc).isoformat(),
    }])
    provisional_artifacts = {
        "PROVISIONAL_NEXT_LEADERS": json_safe(provisional_focus.get("next_leaders", pd.DataFrame())),
        "PROVISIONAL_SWING_READY": json_safe(provisional_focus.get("swing_ready", pd.DataFrame())),
        "PROVISIONAL_JOB_AUDIT": json_safe(provisional_audit),
    }
    try:
        _persist_job_artifacts(job_id, provisional_artifacts, model_version=ENGINE_VERSION)
    except Exception:
        # A failed batch is already isolated/retried by the database bridge.
        # Replaying every artifact individually on a systemic outage multiplied
        # FINALIZING time without improving durability.
        pass
    try:
        _update_job(
            job_id,
            status="FINALIZING",
            phase="RANKING_READY",
            result_summary={
                "ranking_state": "PROVISIONAL_READY",
                "processed_tickers": len(item_audit.get("processed_tickers", [])),
                "technical_ready_tickers": len(completed_tickers),
                "technical_unavailable_tickers": len(item_audit.get("technical_unavailable_tickers", [])),
            },
        )
    except Exception:
        pass
    _mark_stage("PROVISIONAL_RANKING")

    # Stage 2: ranked evidence enrichment. Only a bounded set is allowed to call
    # expensive official/fundamental/news providers. This preserves one-button
    # database maintenance without turning every 20-ticker chunk into a deep scan.
    portfolio_records = config.get("portfolio_records") or []
    portfolio = pd.DataFrame(portfolio_records) if isinstance(portfolio_records, list) else pd.DataFrame()
    portfolio_tickers = portfolio.get("ticker", pd.Series(dtype=str)).map(_ticker).dropna().tolist() if not portfolio.empty else []

    def _ranked_tickers(frame: pd.DataFrame | None) -> list[str]:
        if not isinstance(frame, pd.DataFrame) or frame.empty or "ticker" not in frame.columns:
            return []
        return list(dict.fromkeys(frame["ticker"].map(_ticker).dropna().tolist()))

    requested_evidence_cap = max(0, _int_config(config, "evidence_refresh_cap", 16))
    evidence_cap = min(20, requested_evidence_cap, len(completed_tickers)) if completed_tickers else 0
    maintenance_reserve = min(2, max(0, evidence_cap // 4)) if evidence_cap else 0
    requested_decision_cap = max(0, _int_config(config, "decision_evidence_cap", 12))
    decision_refresh_cap = min(max(0, evidence_cap - maintenance_reserve), requested_decision_cap) if evidence_cap else 0
    fallback_evidence, _ = build_enrichment_shortlist(
        completed_tickers,
        preliminary_focus={},
        signals=signals,
        portfolio_tickers=portfolio_tickers,
        config=ShortlistConfig(
            max_tickers=max(1, decision_refresh_cap),
            multibagger_quota=max(1, decision_refresh_cap // 2),
            core_quota=max(1, decision_refresh_cap // 3),
            technical_rescue_quota=max(1, decision_refresh_cap),
        ),
    ) if decision_refresh_cap else ([], pd.DataFrame())
    # A top-ranked name whose last statement has crossed the next quarterly
    # refresh window is promoted ahead of generic technical rescue.  This fixes
    # cases where a stale-but-<210-day Q1 proxy could keep a high rank even
    # after the Q2/H1 reporting window had opened.
    provisional_ranked = list(dict.fromkeys(
        _ranked_tickers(provisional_focus.get("next_leaders", pd.DataFrame()))
        + _ranked_tickers(provisional_focus.get("swing_ready", pd.DataFrame()))
    ))
    refresh_due_ranked = [
        ticker for ticker in provisional_ranked
        if bool(reporting_refresh_profile(fundamental_map.get(ticker, {})).get("fundamental_refresh_due", False))
    ]
    priority_evidence = list(dict.fromkeys(
        portfolio_tickers
        + refresh_due_ranked
        + provisional_ranked
        + list(fallback_evidence)
    ))[:decision_refresh_cap]

    # Reserve a small maintenance lane so database coverage keeps expanding even
    # when the same leaders remain at the top for many sessions. Once a ticker is
    # current it naturally drops behind the next missing/stale issuer.
    def _maintenance_priority(ticker: str) -> tuple[int, float, str]:
        row = fundamental_map.get(ticker, {})
        bucket, age_priority = maintenance_refresh_priority(
            row, history_count=history_counts.get(ticker, 0),
        )
        return bucket, age_priority, ticker

    maintenance_slots = max(0, evidence_cap - len(priority_evidence))
    maintenance_candidates = [
        ticker for ticker in completed_tickers
        if ticker not in set(priority_evidence) and _maintenance_priority(ticker)[0] < 9
    ]
    maintenance_candidates.sort(key=_maintenance_priority)
    maintenance_evidence = maintenance_candidates[:maintenance_slots]
    evidence_targets = list(dict.fromkeys(priority_evidence + maintenance_evidence))[:evidence_cap]
    if evidence_targets:
        try:
            _update_job(
                job_id, status="FINALIZING", phase="EVIDENCE_REFRESH",
                result_summary={
                    "ranking_state": "PROVISIONAL_READY",
                    "evidence_refresh_tickers": len(evidence_targets),
                },
            )
        except Exception:
            pass
        enrichment_config = dict(config)
        enrichment_config.update({
            "daily_fundamental_refresh_limit": min(len(evidence_targets), max(0, _int_config(config, "evidence_fundamental_cap", 12))),
            "daily_official_fundamental_refresh_limit": min(len(evidence_targets), max(0, _int_config(config, "evidence_official_cap", 8))),
            "daily_snapshot_refresh_limit": min(len(evidence_targets), max(0, _int_config(config, "evidence_snapshot_cap", 12))),
            "daily_market_refresh_limit": min(len(evidence_targets), max(0, _int_config(config, "evidence_market_cap", 6))),
            "daily_news_refresh_limit": min(len(evidence_targets), max(0, _int_config(config, "evidence_news_cap", 12))),
        })
        try:
            fundamentals, history, market, news, enrichment_report = _refresh_missing_daily_evidence(
                bridge, evidence_targets, fundamentals, history, market, news, cfg, enrichment_config,
            )
            fundamentals = enrich_fundamental_evidence(fundamentals)
        except Exception as exc:
            # Enrichment is bounded improvement, not a prerequisite for the
            # research ranking. Preserve cached evidence and surface the error.
            enrichment_report = pd.DataFrame([{
                "provider": "BOUNDED_EVIDENCE_REFRESH",
                "status": "FAIL_SOFT",
                "error": f"{type(exc).__name__}: {str(exc)[:240]}",
            }])
        if isinstance(enrichment_report, pd.DataFrame) and not enrichment_report.empty:
            if isinstance(fundamental_report, pd.DataFrame) and not fundamental_report.empty:
                fundamental_report = pd.concat([fundamental_report, enrichment_report], ignore_index=True, sort=False)
            else:
                fundamental_report = enrichment_report.copy()
        # Recompute macro alignment after refreshed issuer fundamentals.
        macro = build_macro_regime(
            benchmark=benchmark, prepared={}, fundamentals=fundamentals,
            macro_series=macro_series, source_report=macro_report,
            breadth_features=breadth_features,
        )
        if not macro.snapshot.empty:
            raw_regime = str(macro.snapshot.iloc[0].get("macro_regime", "DATA_PENDING"))
            macro.snapshot["macro_regime_raw"] = raw_regime
            macro.snapshot["macro_breadth_coverage_pct"] = round(breadth_coverage_pct, 1)
            macro.snapshot["macro_breadth_sample"] = breadth_sample
            if breadth_coverage_pct < 50.0:
                macro.snapshot["macro_regime"] = f"PROVISIONAL_{raw_regime}"
                macro.snapshot["macro_regime_state"] = "INSUFFICIENT_BREADTH_SAMPLE"
                if not macro.issuer_map.empty and "issuer_macro_alignment_coverage_pct" in macro.issuer_map.columns:
                    macro.issuer_map["issuer_macro_alignment_coverage_pct"] = (
                        pd.to_numeric(macro.issuer_map["issuer_macro_alignment_coverage_pct"], errors="coerce").fillna(0.0)
                        * max(0.25, breadth_coverage_pct / 100.0)
                    ).clip(upper=100.0).round(1)
                    macro.issuer_map["issuer_macro_alignment_state"] = "PROVISIONAL_BREADTH"
            else:
                macro.snapshot["macro_regime_state"] = "VALIDATED_BREADTH"
    _mark_stage("EVIDENCE_REFRESH")

    # Chunk payloads already contain technical scores. The shortlist helper has
    # a technical-rescue lane and deterministic universe fill, so finalisation
    # does not reload OHLCV for all completed tickers.
    preliminary: dict[str, pd.DataFrame] = {}
    # Execution verification is an expensive second-source operation.  Verify
    # the actual provisional leaders/swing candidates first instead of spending
    # up to 40 provider checks on a generic technical rescue list.
    requested_verify_cap = max(0, _int_config(config, "execution_verification_cap", 10))
    verification_cap = (
        min(
            12,
            requested_verify_cap,
            len(completed_tickers),
            max(1, int(cfg.max_automatic_price_candidates)),
        )
        if completed_tickers and requested_verify_cap > 0
        else 0
    )

    priority_verification = list(dict.fromkeys(
        portfolio_tickers
        + _ranked_tickers(provisional_focus.get("next_leaders", pd.DataFrame()))
        + _ranked_tickers(provisional_focus.get("swing_ready", pd.DataFrame()))
    ))
    fallback_verification, shortlist = build_enrichment_shortlist(
        completed_tickers,
        preliminary_focus=preliminary,
        signals=signals,
        portfolio_tickers=portfolio_tickers,
        config=ShortlistConfig(
            max_tickers=max(1, verification_cap),
            multibagger_quota=max(1, verification_cap // 2),
            core_quota=max(1, verification_cap // 3),
            technical_rescue_quota=max(1, verification_cap),
        ),
    ) if verification_cap else ([], pd.DataFrame())
    verification = list(dict.fromkeys(priority_verification + list(fallback_verification)))[:verification_cap]
    try:
        _update_job(
            job_id,
            status="FINALIZING",
            phase="EXECUTION_VERIFY",
            result_summary={
                "ranking_state": "PROVISIONAL_READY",
                "verification_tickers": len(verification),
            },
        )
    except Exception:
        pass
    verify_histories, download_report, verification_ohlcv_report = _database_first_ohlcv(
        bridge, verification, period=period,
        itick_api_token=str(runtime.get("itick_api_token", "") or ""),
        min_bars=260, max_stale_sessions=0, force_refresh=False,
    ) if verification else ({}, None, pd.DataFrame())
    primary_reference = {
        ticker: (frame.index[-1], float(pd.to_numeric(frame["Close"], errors="coerce").dropna().iloc[-1]))
        for ticker, frame in verify_histories.items()
        if isinstance(frame, pd.DataFrame) and not frame.empty and "Close" in frame
    }
    primary_source_tiers = dict(getattr(download_report, "source_tiers", {}) or {}) if download_report is not None else {}
    if isinstance(verification_ohlcv_report, pd.DataFrame) and not verification_ohlcv_report.empty and {"ticker", "source_tier"}.issubset(verification_ohlcv_report.columns):
        tier_rows = verification_ohlcv_report.dropna(subset=["ticker"]).drop_duplicates("ticker", keep="last")
        primary_source_tiers.update({
            _ticker(row.get("ticker")): str(row.get("source_tier") or "")
            for row in tier_rows.to_dict("records")
            if _ticker(row.get("ticker"))
        })
    if verification:
        # Yahoo execution snapshots and the independent-provider chain do not
        # depend on each other. Run both bounded I/O branches concurrently, but
        # make each branch independently fail-soft. Verification can block an
        # order; it must never erase a valid research ranking.
        def _safe_snapshots() -> pd.DataFrame:
            try:
                value = fetch_execution_snapshots(verification)
                return value if isinstance(value, pd.DataFrame) else pd.DataFrame()
            except Exception:
                return pd.DataFrame()

        def _safe_independent() -> tuple[pd.DataFrame, pd.DataFrame]:
            try:
                value, report = fetch_automatic_independent_prices(
                    verification,
                    twelve_data_api_key=str(runtime.get("twelve_data_api_key", "") or ""),
                    itick_api_token=str(runtime.get("itick_api_token", "") or ""),
                    primary_reference=primary_reference,
                    primary_source_tiers=primary_source_tiers,
                    config=cfg,
                )
                return (
                    value if isinstance(value, pd.DataFrame) else pd.DataFrame(),
                    report if isinstance(report, pd.DataFrame) else pd.DataFrame(),
                )
            except Exception as exc:
                return pd.DataFrame(), pd.DataFrame([{
                    "provider": "INDEPENDENT_PRICE", "status": "FAIL_SOFT",
                    "error": f"{type(exc).__name__}: {str(exc)[:240]}",
                }])

        with ThreadPoolExecutor(max_workers=2) as pool:
            snapshot_future = pool.submit(_safe_snapshots)
            independent_future = pool.submit(_safe_independent)
            snapshots = snapshot_future.result()
            independent, independent_report = independent_future.result()
    else:
        snapshots = pd.DataFrame()
        independent, independent_report = pd.DataFrame(), pd.DataFrame()
    final_signals = apply_execution_snapshot_gate(signals, snapshots, cfg) if not signals.empty else signals
    price_validation = build_independent_price_validation(
        verify_histories,
        independent,
        config=cfg,
        primary_source_tiers=primary_source_tiers,
    ) if verification else pd.DataFrame()
    if not final_signals.empty:
        final_signals = finalize_execution_integrity(
            attach_position_sizing(apply_independent_price_gate(final_signals, price_validation, cfg), cfg), cfg
        )
    _mark_stage("EXECUTION_VERIFY")
    focus = build_simple_focus(
        {},
        fundamentals=fundamentals,
        signals=final_signals,
        news_review=news,
        market_status=market,
        benchmark=benchmark,
        macro_result=macro,
        config=cfg,
        universe_tickers=completed_tickers,
        precomputed_silent_frame=silent_frame,
        precomputed_narrative=narrative,
    )
    portfolio_histories = dict(verify_histories)
    missing_portfolio = [ticker for ticker in portfolio_tickers if ticker not in portfolio_histories]
    if missing_portfolio:
        extra, _, _ = _database_first_ohlcv(
            bridge, missing_portfolio, period=period,
            itick_api_token=str(runtime.get("itick_api_token", "") or ""),
            min_bars=260, max_stale_sessions=5,
        )
        portfolio_histories.update(extra)
    portfolio_analysis, portfolio_summary = analyze_portfolio_positions(
        portfolio,
        portfolio_histories,
        fundamentals=fundamentals,
        signals=final_signals,
        account_equity_idr=float(config.get("account_size_idr", 5_000_000) or 5_000_000),
        cash_on_hand_idr=float(config.get("cash_on_hand_idr", 1_000_000) or 1_000_000),
        config=cfg,
    )
    coverage = build_database_coverage(
        universe,
        fundamentals=fundamentals,
        fundamental_history=history,
        market_status=market,
        news_review=news,
        fundamental_report=fundamental_report,
    )
    coverage_summary = readiness_summary(coverage)
    _mark_stage("FINAL_RANKING_AND_PORTFOLIO")
    prepared_placeholder = {ticker: pd.DataFrame() for ticker in completed_tickers}
    result = {
        "mode": "resumable_chunked_daily_scan",
        "scanner_version": ENGINE_VERSION,
        "scan_started_at": job.get("started_at"),
        "scan_finished_at": datetime.now(timezone.utc).isoformat(),
        "ticker_count": len(universe),
        "prepared": prepared_placeholder,
        "signals": final_signals,
        "fundamentals": fundamentals,
        "fundamental_history": history,
        "fundamental_history_report": fundamental_report,
        "market_status": market,
        "news_review": news,
        "execution_snapshots": snapshots,
        "independent_price_data": independent,
        "independent_provider_report": independent_report,
        "price_validation": price_validation,
        "focus_screens": focus,
        "portfolio": portfolio,
        "portfolio_analysis": portfolio_analysis,
        "portfolio_summary": portfolio_summary,
        "macro_snapshot": macro.snapshot,
        "macro_sector_map": macro.sector_map,
        "macro_issuer_map": macro.issuer_map,
        "macro_source_report": pd.concat(
            [frame for frame in (macro.source_report, benchmark_report, verification_ohlcv_report) if isinstance(frame, pd.DataFrame) and not frame.empty],
            ignore_index=True, sort=False,
        ) if any(isinstance(frame, pd.DataFrame) and not frame.empty for frame in (macro.source_report, benchmark_report, verification_ohlcv_report)) else pd.DataFrame(),
        "ihsg_direction": analyze_ihsg_direction(benchmark, {}, config=IHSGDirectionConfig()),
        "database_coverage_after": coverage,
        "database_summary_after": coverage_summary,
        "two_stage_shortlist": shortlist,
        "narrative_events": narrative.get("events", pd.DataFrame()),
        "narrative_event_outcomes": narrative.get("outcomes", pd.DataFrame()),
        "narrative_profiles": narrative.get("profiles", pd.DataFrame()),
        "project_management_review": project_management,
    }
    try:
        _update_job(
            job_id,
            status="FINALIZING",
            phase="DATABASE_SYNC",
            result_summary={
                "ranking_state": "FINAL_COMPUTED",
                "verification_tickers": len(verification),
            },
        )
    except Exception:
        pass

    if hasattr(bridge, "persist_scan_result"):
        try:
            # Bounded provider deltas may already be durable, but finalization
            # must also persist valuation and derived-evidence fields computed
            # from the complete chunk set.
            final_tables = (
                LEAN_FINAL_PERSISTENCE_TABLES
                if bool(config.get("lean_persistence", False))
                else FULL_FINAL_PERSISTENCE_TABLES
            )
            result["database_sync_report"] = bridge.persist_scan_result(result, tables=final_tables)
        except Exception as exc:
            result["database_sync_report"] = pd.DataFrame([{
                "provider": "DATABASE_FINAL_SYNC", "state": "FAIL_SOFT",
                "error": f"{type(exc).__name__}: {str(exc)[:240]}",
            }])
    else:
        result["database_sync_report"] = pd.DataFrame([{
            "provider": "DATABASE_FINAL_SYNC", "state": "NOT_AVAILABLE",
        }])
    _mark_stage("DATABASE_SYNC")

    ohlcv_audit = _ohlcv_item_audit(items)
    technical_coverage_pct = 100.0 * len(completed_tickers) / len(universe) if universe else 0.0
    ranking_state = "VALID" if technical_coverage_pct >= 70.0 else "PARTIAL_UNIVERSE"
    artifacts = {
        "FINAL_NEXT_LEADERS": focus.get("next_leaders", pd.DataFrame()),
        "FINAL_SWING_READY": focus.get("swing_ready", pd.DataFrame()),
        "FINAL_NEXT_LEADERS_ALL": focus.get("next_leaders_all", pd.DataFrame()),
        "FINAL_SWING_READY_ALL": focus.get("swing_ready_all", pd.DataFrame()),
        "FINAL_EVIDENCE_DETAIL": focus.get("production_evidence_detail", pd.DataFrame()),
        "FINAL_SCORING_CONTRACT": focus.get("production_scoring_audit", pd.DataFrame()),
        "FINAL_MACRO_SNAPSHOT": macro.snapshot,
        "FINAL_MACRO_SECTOR_MAP": macro.sector_map,
        "FINAL_PORTFOLIO": portfolio_analysis,
        "FINAL_COVERAGE": coverage,
        "FINAL_OHLCV_AUDIT": ohlcv_audit,
        "FINAL_DATABASE_SYNC_REPORT": result.get("database_sync_report", pd.DataFrame()),
        "FINAL_FUNDAMENTAL_PROVIDER_AUDIT": fundamental_report,
        "FINAL_STAGE_TIMINGS": pd.DataFrame([{
            **stage_timings,
            "elapsed_before_artifact_publish_sec": round(time.perf_counter() - finalizer_started, 3),
            "evidence_refresh_tickers": len(evidence_targets),
            "execution_verification_tickers": len(verification),
        }]),
        "FINAL_JOB_AUDIT": pd.DataFrame([{
            "requested_tickers": len(universe),
            "processed_tickers": len(item_audit.get("processed_tickers", [])),
            "processed_ticker_list": list(item_audit.get("processed_tickers", [])),
            "completed_tickers": len(completed_tickers),
            "completed_ticker_list": list(completed_tickers),
            "technical_unavailable_tickers": len(item_audit.get("technical_unavailable_tickers", [])),
            "technical_unavailable_ticker_list": list(item_audit.get("technical_unavailable_tickers", [])),
            "ohlcv_ready_tickers": len(item_audit.get("ohlcv_ready_tickers", [])),
            "ohlcv_ready_ticker_list": list(item_audit.get("ohlcv_ready_tickers", [])),
            "ohlcv_state_counts": dict(item_audit.get("ohlcv_state_counts", {})),
            "technical_coverage_pct": round(technical_coverage_pct, 1),
            "breadth_coverage_pct": round(breadth_coverage_pct, 1),
            "ranking_state": ranking_state,
            "next_leader_final_score_valid": int(len(focus.get("next_leaders", pd.DataFrame()))),
            "swing_final_score_valid": int(len(focus.get("swing_ready", pd.DataFrame()))),
            "failed_tickers": max(0, len(universe) - len(item_audit.get("processed_tickers", []))),
            "fundamental_ready": int(
                _coerce_bool_series(
                    coverage.get("fundamental_snapshot_ready", pd.Series(False, index=coverage.index)),
                    default=False,
                ).sum()
            ) if not coverage.empty else 0,
            "verification_tickers": len(verification),
            "worker_id": worker_id,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }]),
    }
    try:
        _update_job(
            job_id,
            status="FINALIZING",
            phase="ARTIFACT_PUBLISH",
            result_summary={
                "ranking_state": ranking_state,
                "verification_tickers": len(verification),
            },
        )
    except Exception:
        pass
    safe_artifacts = {artifact_type: json_safe(payload) for artifact_type, payload in artifacts.items()}
    try:
        _persist_job_artifacts(job_id, safe_artifacts, model_version=ENGINE_VERSION)
    except Exception:
        # Do not turn one database outage into N sequential artifact retries.
        # The ranking is already computed and the durable job can be retried.
        pass
    _mark_stage("ARTIFACT_PUBLISH")
    stage_timings["FINALIZER_TOTAL"] = round(time.perf_counter() - finalizer_started, 3)
    result["scan_coverage_summary"] = artifacts["FINAL_JOB_AUDIT"].copy()
    result["ohlcv_database_audit"] = ohlcv_audit.copy()
    result["stage_timings"] = pd.DataFrame([{**stage_timings}])
    result["ranking_state"] = "FINAL"
    result["ranking_quality_state"] = ranking_state
    result["scan_id"] = job_id
    _JOB_EVIDENCE_CACHE.pop(job_id, None)
    _JOB_FORWARD_CACHE.pop(job_id, None)
    summary = {
        "requested_tickers": len(universe),
        "processed_tickers": len(item_audit.get("processed_tickers", [])),
        "completed_tickers": len(completed_tickers),
        "technical_unavailable_tickers": len(item_audit.get("technical_unavailable_tickers", [])),
        "ohlcv_ready_tickers": len(item_audit.get("ohlcv_ready_tickers", [])),
        "technical_coverage_pct": round(technical_coverage_pct, 1),
        "breadth_coverage_pct": round(breadth_coverage_pct, 1),
        "ranking_state": ranking_state,
        "failed_tickers": max(0, len(universe) - len(item_audit.get("processed_tickers", []))),
        "next_leaders": int(len(focus.get("next_leaders", pd.DataFrame()))),
        "swing_ready": int(len(focus.get("swing_ready", pd.DataFrame()))),
        "verification_tickers": len(verification),
        "evidence_refresh_tickers": len(evidence_targets),
        "stage_timings_sec": stage_timings,
        "finalized_at": datetime.now(timezone.utc).isoformat(),
        "worker_id": worker_id,
    }
    if return_result:
        return {**summary, "result": result}
    return summary


__all__ = [
    "ENGINE_VERSION",
    "LEAN_FINAL_PERSISTENCE_TABLES", "FULL_FINAL_PERSISTENCE_TABLES",
    "process_daily_scan_chunk",
    "finalize_daily_scan_job",
]
