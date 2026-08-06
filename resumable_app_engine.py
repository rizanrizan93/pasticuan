from __future__ import annotations

"""Chunk processors and finalizers for the durable v9.4 scan jobs."""

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence
import time

import numpy as np
import pandas as pd

from database_first import _coerce_bool_series, build_database_coverage, readiness_summary
from ihsg_direction import IHSGDirectionConfig, analyze_ihsg_direction
from macro_engine import build_macro_regime, fetch_macro_series
from narrative_engine import build_narrative_intelligence
from resumable_scan import ItemOutcome, frame_from_records, json_safe
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
    enrich_fundamentals_with_history,
    fetch_automatic_independent_prices,
    fetch_execution_snapshots,
    fetch_idx_fundamental_history,
    fetch_resilient_fundamentals,
    fetch_resilient_market_status,
    fetch_resilient_news_review,
    fetch_twelve_data_fundamental_history,
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
from two_stage_pipeline import ShortlistConfig, build_enrichment_shortlist, build_lightweight_preliminary_focus

ENGINE_VERSION = "9.5.0"


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return number if np.isfinite(number) else float(default)


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
    if fundamentals is None or fundamentals.empty:
        return pd.DataFrame()
    out = fundamentals.copy()
    score = pd.to_numeric(out.get("fundamental_score", pd.Series(np.nan, index=out.index)), errors="coerce")
    coverage = pd.to_numeric(out.get("fundamental_coverage", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    source_count = pd.to_numeric(out.get("fundamental_source_count", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    age = pd.to_numeric(out.get("statement_age_days", pd.Series(np.nan, index=out.index)), errors="coerce")
    existing = _coerce_bool_series(out.get("fundamental_score_eligible", pd.Series(False, index=out.index)), default=False)
    derived = score.notna() & coverage.ge(45.0) & source_count.ge(1.0) & (age.isna() | age.le(550.0))
    out["fundamental_score_eligible"] = (existing | derived).astype(bool)
    if "fundamental_route_state" not in out.columns:
        out["fundamental_route_state"] = ""
    out.loc[derived & ~existing, "fundamental_route_state"] = "HISTORY_DERIVED_VERIFIED"
    return out


def _load_fundamentals(bridge: ScannerDatabaseBridge, tickers: Sequence[str], cfg: ScanConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    db_snapshot, audit_snapshot = bridge.read_fundamental_cache(tickers)
    db_history, audit_history = bridge.read_fundamental_history_cache(tickers)
    local_snapshot = read_cached_fundamentals(tickers, cfg)
    if not local_snapshot.empty and "fundamental_score_eligible" in local_snapshot.columns:
        local_snapshot = local_snapshot.loc[_coerce_bool_series(local_snapshot["fundamental_score_eligible"], default=False)].copy()
    local_history = read_cached_fundamental_history(tickers)
    snapshot = _merge_primary(local_snapshot, db_snapshot)
    history = combine_fundamental_history(db_history, local_history)
    snapshot = _mark_history_eligible(enrich_fundamentals_with_history(snapshot, history))
    audits = [frame for frame in (audit_snapshot, audit_history) if isinstance(frame, pd.DataFrame) and not frame.empty]
    return snapshot, history, pd.concat(audits, ignore_index=True, sort=False) if audits else pd.DataFrame()


def _load_aux(bridge: ScannerDatabaseBridge, tickers: Sequence[str], cfg: ScanConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    db_market, _ = bridge.read_market_status_cache(tickers, max_age_days=max(1, int(getattr(cfg, "market_status_cache_days", 3))))
    db_news, _ = bridge.read_news_review_cache(tickers, max_age_days=max(1, int(getattr(cfg, "news_cache_days", 7))))
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
    include_today = bool(is_idx_session(today) and (local.hour > 16 or (local.hour == 16 and local.minute >= 30)))
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
        if ticker in refresh_targets:
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
    if refresh_targets and hasattr(bridge, "write_ohlcv_cache"):
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
    frames = [frame for frame in (cache_audit, pd.DataFrame(audit_rows), write_audit) if isinstance(frame, pd.DataFrame) and not frame.empty]
    return histories, report, pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


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
    # Unit/offline repositories may intentionally expose only checkpoint reads.
    # Live delta acquisition requires a durable writer; otherwise retain the
    # already loaded evidence without attempting external providers.
    if not hasattr(bridge, "persist_scan_result"):
        return fundamentals, history, market, news, pd.DataFrame()
    counts = _history_counts(history)
    eligible = set()
    if not fundamentals.empty and "fundamental_score_eligible" in fundamentals.columns:
        eligible = set(fundamentals.loc[
            _coerce_bool_series(fundamentals["fundamental_score_eligible"], default=False), "ticker"
        ].map(_ticker))
    fundamental_missing = [ticker for ticker in names if ticker not in eligible or counts.get(ticker, 0) < 2]
    fundamental_limit = max(0, int(config.get("daily_fundamental_refresh_limit", 4) or 4))
    targets = fundamental_missing[:fundamental_limit]
    if targets:
        live_history, history_report = fetch_yahoo_fundamental_history(
            targets,
            max_workers=min(6, max(1, len(targets))),
            max_tickers=len(targets),
            enable_yfinance_fallback=False,
        )
        history = combine_fundamental_history(history, live_history)
        live_snapshot = fetch_resilient_fundamentals(targets[:max(1, min(2, len(targets)))], cfg)
        fundamentals = _mark_history_eligible(
            enrich_fundamentals_with_history(_merge_primary(live_snapshot, fundamentals), history)
        )
        if isinstance(history_report, pd.DataFrame) and not history_report.empty:
            reports.append(history_report)

    market_present = set(market.get("ticker", pd.Series(dtype=str)).map(_ticker)) if not market.empty else set()
    market_targets = [ticker for ticker in names if ticker not in market_present]
    if market_targets:
        market_live = fetch_resilient_market_status(market_targets, cfg)
        market = _merge_primary(market_live, market)

    news_present = set(news.get("ticker", pd.Series(dtype=str)).map(_ticker)) if not news.empty else set()
    news_limit = max(0, int(config.get("daily_news_refresh_limit", 5) or 5))
    news_targets = [ticker for ticker in names if ticker not in news_present][:news_limit]
    if news_targets:
        news_live = fetch_resilient_news_review(news_targets, lookback_days=7, config=cfg)
        news = _merge_primary(news_live, news)

    if (targets or market_targets or news_targets) and hasattr(bridge, "persist_scan_result"):
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
            })
            if isinstance(sync, pd.DataFrame) and not sync.empty:
                reports.append(sync.assign(source_family="DATABASE_DELTA_SYNC"))
        except Exception as exc:
            reports.append(pd.DataFrame([{
                "provider": "DATABASE_DELTA_SYNC", "status": "FAIL_SOFT",
                "error": f"{type(exc).__name__}: {str(exc)[:240]}",
            }]))
    return fundamentals, history, market, news, pd.concat(reports, ignore_index=True, sort=False) if reports else pd.DataFrame()


def process_backfill_chunk(
    job: Mapping[str, Any],
    items: pd.DataFrame,
    worker_id: str,
    *,
    runtime: Mapping[str, str] | None = None,
) -> Mapping[str, ItemOutcome]:
    runtime = dict(runtime or {})
    config = dict(job.get("config_payload") or {})
    cfg = _cfg(config)
    tickers = [_ticker(value) for value in items.get("ticker", pd.Series(dtype=str)).tolist()]
    tickers = [value for value in tickers if value]
    bridge = ScannerDatabaseBridge()
    baseline, baseline_history, _ = _load_fundamentals(bridge, tickers, cfg)
    cached_market, _ = _load_aux(bridge, tickers, cfg)

    idx_history = pd.DataFrame()
    idx_report = pd.DataFrame()
    if bool(config.get("official_idx_refresh", False)):
        idx_history, idx_report = fetch_idx_fundamental_history(
            tickers,
            max_tickers=len(tickers),
            years_back=2,
            timeout=4,
            max_workers=min(8, max(1, len(tickers))),
        )
    history_after_idx = combine_fundamental_history(baseline_history, idx_history)
    yahoo_targets = select_yahoo_fundamental_tickers(
        tickers,
        history_after_idx,
        max_tickers=len(tickers),
        crosscheck_top_n=0,
        min_official_periods=2,
    )
    yahoo_history, yahoo_report = fetch_yahoo_fundamental_history(
        yahoo_targets,
        max_workers=min(8, max(1, len(yahoo_targets))),
        max_tickers=len(yahoo_targets),
        enable_yfinance_fallback=False,
    ) if yahoo_targets else (pd.DataFrame(), pd.DataFrame())
    history = combine_fundamental_history(history_after_idx, yahoo_history)
    counts = _history_counts(history)
    twelve_candidates = [ticker for ticker in tickers if counts.get(ticker, 0) < 2]
    twelve_cap = min(int(config.get("twelve_fallback_limit", 6) or 6), len(twelve_candidates))
    twelve_targets = twelve_candidates[:max(0, twelve_cap)]
    twelve_key = str(runtime.get("twelve_data_api_key", "") or "").strip()
    if twelve_targets and twelve_key:
        twelve_history, twelve_report = fetch_twelve_data_fundamental_history(
            twelve_targets,
            api_key=twelve_key,
            max_tickers=len(twelve_targets),
            timeout=5,
            max_workers=min(8, len(twelve_targets)),
        )
        history = combine_fundamental_history(history, twelve_history)
    else:
        twelve_report = pd.DataFrame()

    fundamentals = _mark_history_eligible(enrich_fundamentals_with_history(baseline, history))
    ready_snapshot = set()
    if not fundamentals.empty and "fundamental_score_eligible" in fundamentals.columns:
        ready_snapshot = set(fundamentals.loc[_coerce_bool_series(fundamentals["fundamental_score_eligible"], default=False), "ticker"].map(_ticker))
    fallback_limit = max(0, int(config.get("snapshot_fallback_limit", 4) or 4))
    fallback = [ticker for ticker in tickers if ticker not in ready_snapshot][:fallback_limit]
    if fallback:
        live = fetch_resilient_fundamentals(fallback, cfg)
        fundamentals = _mark_history_eligible(enrich_fundamentals_with_history(_merge_primary(live, fundamentals), history))

    market_live = fetch_resilient_market_status(tickers, cfg)
    market = _merge_primary(market_live, cached_market)
    ohlcv_histories, ohlcv_download_report, ohlcv_report = _database_first_ohlcv(
        bridge, tickers, period=str(config.get("period", "5y") or "5y"),
        itick_api_token=str(runtime.get("itick_api_token", "") or ""),
        min_bars=260, max_stale_sessions=5,
    )
    provider_report = pd.concat(
        [frame for frame in (idx_report, yahoo_report, twelve_report, ohlcv_report) if isinstance(frame, pd.DataFrame) and not frame.empty],
        ignore_index=True,
        sort=False,
    ) if any(isinstance(frame, pd.DataFrame) and not frame.empty for frame in (idx_report, yahoo_report, twelve_report, ohlcv_report)) else pd.DataFrame()

    persist = {
        "mode": "resumable_backfill_chunk",
        "scanner_version": ENGINE_VERSION,
        "fundamentals": fundamentals,
        "fundamental_history": history,
        "fundamental_history_report": provider_report,
        "market_status": market,
        "news_review": pd.DataFrame(),
        "focus_screens": {},
        "prepared": {},
        "ohlcv_database_report": ohlcv_report,
        "ticker_count": len(tickers),
    }
    sync = bridge.persist_scan_result(persist)
    counts = _history_counts(history)
    snapshot_ready = set()
    if not fundamentals.empty and "fundamental_score_eligible" in fundamentals.columns:
        snapshot_ready = set(fundamentals.loc[_coerce_bool_series(fundamentals["fundamental_score_eligible"], default=False), "ticker"].map(_ticker))
    market_ready = set(market.get("ticker", pd.Series(dtype=str)).map(_ticker)) if not market.empty else set()
    # A durable job item represents whether the acquisition/checkpoint step ran,
    # not whether every issuer happened to expose two usable reporting periods.
    # Missing/partial public evidence is a data state, not an infrastructure error.
    hard_sync_failure = False
    sync_state = "OK"
    if sync.empty:
        sync_state = "EMPTY_SYNC_REPORT"
    elif "state" in sync.columns:
        states = sync["state"].fillna("").astype(str).str.upper()
        if states.isin({"CONFIG_INCOMPLETE", "CONFIG_UNSAFE_KEY", "DISABLED_NO_DATABASE"}).any():
            hard_sync_failure = True
            sync_state = "DATABASE_CONFIG_ERROR"
        else:
            critical_tables = {"fundamental_cache", "fundamental_history_cache", "fundamental_snapshots", "source_events"}
            local = sync.copy()
            if "table" in local.columns:
                local = local.loc[local["table"].astype(str).isin(critical_tables)]
            attempted = pd.to_numeric(local.get("rows_attempted", pd.Series(0, index=local.index)), errors="coerce").fillna(0).sum()
            written = pd.to_numeric(local.get("rows_written", pd.Series(0, index=local.index)), errors="coerce").fillna(0).sum()
            failed_attempt = local.get("state", pd.Series("", index=local.index)).astype(str).str.upper().isin({"DATABASE_FAIL_SOFT", "PARTIAL_WRITE"})
            if attempted > 0 and written <= 0 and bool(failed_attempt.any()):
                hard_sync_failure = True
                sync_state = "CORE_DATABASE_WRITE_FAILED"
            elif bool(failed_attempt.any()):
                sync_state = "PARTIAL_DATABASE_WRITE"

    snapshot_present = set(fundamentals.get("ticker", pd.Series(dtype=str)).map(_ticker)) if not fundamentals.empty else set()
    history_present = {ticker for ticker, period_count in counts.items() if int(period_count) > 0}
    ohlcv_ready = {ticker for ticker, frame in ohlcv_histories.items() if isinstance(frame, pd.DataFrame) and len(frame) >= 260}
    ohlcv_meta: dict[str, dict[str, Any]] = {}
    if isinstance(ohlcv_report, pd.DataFrame) and not ohlcv_report.empty and "ticker" in ohlcv_report.columns:
        ohlcv_meta = ohlcv_report.drop_duplicates("ticker", keep="last").set_index("ticker").to_dict("index")

    by_ticker: dict[str, ItemOutcome] = {}
    item_lookup = {_ticker(row.get("ticker")): str(row.get("item_key")) for row in items.to_dict("records")}
    for ticker in tickers:
        key = item_lookup.get(ticker)
        if not key:
            continue
        periods = int(counts.get(ticker, 0))
        eligible_snapshot = ticker in snapshot_ready
        stored_snapshot = ticker in snapshot_present
        core_ready = eligible_snapshot and periods >= 2 and ticker in ohlcv_ready
        any_evidence = stored_snapshot or periods > 0 or ticker in market_ready or ticker in ohlcv_ready
        if core_ready:
            completion_state = "CORE_READY"
        elif eligible_snapshot or stored_snapshot:
            completion_state = "PARTIAL_SNAPSHOT"
        elif periods > 0:
            completion_state = "PARTIAL_HISTORY"
        elif ticker in market_ready:
            completion_state = "MARKET_STATUS_ONLY"
        else:
            completion_state = "NO_PUBLIC_EVIDENCE"
        payload = {
            "ticker": ticker,
            "completion_state": completion_state,
            "core_ready": core_ready,
            "snapshot_present": stored_snapshot,
            "snapshot_ready": eligible_snapshot,
            "history_periods": periods,
            "market_status_ready": ticker in market_ready,
            "ohlcv_ready": ticker in ohlcv_ready,
            "ohlcv_bars": int(len(ohlcv_histories.get(ticker, pd.DataFrame()))),
            "ohlcv_state": str(ohlcv_meta.get(ticker, {}).get("status") or "MISSING"),
            "ohlcv_last_bar_date": ohlcv_meta.get(ticker, {}).get("last_bar_date"),
            "any_evidence": any_evidence,
            "refresh_recommended": not core_ready,
            "database_sync_state": sync_state,
            "worker_id": worker_id,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        if hard_sync_failure:
            by_ticker[key] = ItemOutcome(
                False,
                payload=payload,
                error=f"DATABASE_WRITE_FAILED: {sync_state}",
                retry_delay_seconds=30,
            )
        else:
            # COMPLETE means this ticker was processed and checkpointed. Coverage
            # quality remains explicit in the payload and final readiness artifact.
            by_ticker[key] = ItemOutcome(True, payload=payload)
    return by_ticker


def finalize_backfill_job(job: Mapping[str, Any], bridge: ScannerDatabaseBridge, worker_id: str) -> Mapping[str, Any]:
    tickers = [_ticker(value) for value in (job.get("universe_payload") or [])]
    cfg = _cfg(dict(job.get("config_payload") or {}))
    fundamentals, history, report = _load_fundamentals(bridge, tickers, cfg)
    market, news = _load_aux(bridge, tickers, cfg)
    coverage = build_database_coverage(
        tickers,
        fundamentals=fundamentals,
        fundamental_history=history,
        market_status=market,
        news_review=news,
        fundamental_report=report,
    )
    summary = readiness_summary(coverage)
    try:
        items = bridge.read_scan_job_items(
            str(job.get("job_id")), phase=str(job.get("phase") or "BACKFILL_CORE"),
            include_payload=True, limit=5000,
        )
    except Exception:
        items = pd.DataFrame()
    states: list[str] = []
    if not items.empty and "result_payload" in items.columns:
        states = [
            str(value.get("completion_state", "UNKNOWN")) if isinstance(value, Mapping) else "UNKNOWN"
            for value in items["result_payload"].tolist()
        ]
    execution_summary = pd.DataFrame([{
        "processed_items": int((items.get("status", pd.Series(dtype=str)).astype(str).str.upper() == "COMPLETE").sum()) if not items.empty else 0,
        "core_ready_items": int(sum(state == "CORE_READY" for state in states)),
        "partial_items": int(sum(state in {"PARTIAL_SNAPSHOT", "PARTIAL_HISTORY", "MARKET_STATUS_ONLY"} for state in states)),
        "no_public_evidence_items": int(sum(state == "NO_PUBLIC_EVIDENCE" for state in states)),
        "system_failed_items": int((items.get("status", pd.Series(dtype=str)).astype(str).str.upper() == "FAILED").sum()) if not items.empty else 0,
    }])
    bridge.persist_scan_job_artifact(str(job.get("job_id")), "BACKFILL_COVERAGE", json_safe(coverage), model_version=ENGINE_VERSION)
    bridge.persist_scan_job_artifact(str(job.get("job_id")), "BACKFILL_SUMMARY", json_safe(summary), model_version=ENGINE_VERSION)
    bridge.persist_scan_job_artifact(str(job.get("job_id")), "BACKFILL_EXECUTION_SUMMARY", json_safe(execution_summary), model_version=ENGINE_VERSION)
    row = summary.iloc[0].to_dict() if not summary.empty else {}
    execution_row = execution_summary.iloc[0].to_dict()
    return {**json_safe(row), **json_safe(execution_row), "worker_id": worker_id, "finalized_at": datetime.now(timezone.utc).isoformat()}


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
) -> Mapping[str, ItemOutcome]:
    runtime = dict(runtime or {})
    config = dict(job.get("config_payload") or {})
    cfg = _cfg(config)
    tickers = [_ticker(value) for value in items.get("ticker", pd.Series(dtype=str)).tolist()]
    tickers = [value for value in tickers if value]
    period = str(config.get("period", "5y") or "5y")
    bridge = ScannerDatabaseBridge()
    fundamentals, chunk_history, chunk_fundamental_report = _load_fundamentals(bridge, tickers, cfg)
    market, news = _load_aux(bridge, tickers, cfg)
    fundamentals, chunk_history, market, news, delta_report = _refresh_missing_daily_evidence(
        bridge, tickers, fundamentals, chunk_history, market, news, cfg, config,
    )
    existing_events = bridge.read_narrative_events(tickers, limit=max(200, len(tickers) * 20))
    existing_outcomes = bridge.read_narrative_event_outcomes(tickers, limit=max(200, len(tickers) * 20))

    histories, download_report, ohlcv_report = _database_first_ohlcv(
        bridge, tickers, period=period,
        itick_api_token=str(runtime.get("itick_api_token", "") or ""),
        min_bars=260, max_stale_sessions=5,
    )
    benchmark, benchmark_report = _database_first_benchmark(bridge, period=period)
    bounded = {ticker: frame.tail(750).copy() for ticker, frame in histories.items() if isinstance(frame, pd.DataFrame) and not frame.empty}
    core = ScanEngine(cfg).scan(bounded, benchmark.tail(750).copy() if isinstance(benchmark, pd.DataFrame) else benchmark)
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
            "breadth": json_safe(_breadth_row(frame)),
            "ohlcv_state": str(ohlcv_meta.get(ticker, {}).get("status") or "READY"),
            "ohlcv_bars": int(len(frame)),
            "ohlcv_last_bar_date": json_safe(frame.index[-1]),
            "ohlcv_session_lag": ohlcv_meta.get(ticker, {}).get("session_lag"),
            "ohlcv_source_tier": str(ohlcv_meta.get(ticker, {}).get("source_tier") or "DATABASE_OR_PUBLIC"),
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
            "attempt_count": int(row.get("attempt_count") or 0),
        })
    return pd.DataFrame(rows)


def finalize_daily_scan_job(
    job: Mapping[str, Any],
    bridge: ScannerDatabaseBridge,
    worker_id: str,
    *,
    runtime: Mapping[str, str] | None = None,
) -> Mapping[str, Any]:
    runtime = dict(runtime or {})
    config = dict(job.get("config_payload") or {})
    cfg = _cfg(config)
    job_id = str(job.get("job_id"))
    items = bridge.read_scan_job_items(job_id, phase="TECHNICAL", include_payload=True, limit=5000)
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
    fundamentals, history, fundamental_report = _load_fundamentals(bridge, universe, cfg)
    market, news = _load_aux(bridge, universe, cfg)
    period = str(config.get("period", "5y") or "5y")
    benchmark, benchmark_report = _database_first_benchmark(bridge, period=period)
    macro_series, macro_report = fetch_macro_series(period="6mo", timeout=8)
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
    bridge.persist_scan_job_artifact(
        job_id,
        "PROVISIONAL_NEXT_LEADERS",
        json_safe(provisional_focus.get("next_leaders", pd.DataFrame())),
        model_version=ENGINE_VERSION,
    )
    bridge.persist_scan_job_artifact(
        job_id,
        "PROVISIONAL_SWING_READY",
        json_safe(provisional_focus.get("swing_ready", pd.DataFrame())),
        model_version=ENGINE_VERSION,
    )
    bridge.persist_scan_job_artifact(
        job_id,
        "PROVISIONAL_JOB_AUDIT",
        json_safe(provisional_audit),
        model_version=ENGINE_VERSION,
    )
    try:
        bridge.update_scan_job(
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
    # Chunk payloads already contain technical scores. The shortlist helper has
    # a technical-rescue lane and deterministic universe fill, so finalisation
    # does not reload OHLCV for all completed tickers.
    preliminary: dict[str, pd.DataFrame] = {}
    verification_cap = min(40, len(completed_tickers), max(1, int(cfg.max_automatic_price_candidates)))
    portfolio_records = config.get("portfolio_records") or []
    portfolio = pd.DataFrame(portfolio_records) if isinstance(portfolio_records, list) else pd.DataFrame()
    portfolio_tickers = portfolio.get("ticker", pd.Series(dtype=str)).map(_ticker).dropna().tolist() if not portfolio.empty else []
    verification, shortlist = build_enrichment_shortlist(
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
    )
    verification = verification[:verification_cap]
    try:
        bridge.update_scan_job(
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
        min_bars=260, max_stale_sessions=2, force_refresh=True,
    ) if verification else ({}, None, pd.DataFrame())
    snapshots = fetch_execution_snapshots(verification) if verification else pd.DataFrame()
    final_signals = apply_execution_snapshot_gate(signals, snapshots, cfg) if not signals.empty else signals
    independent, independent_report = fetch_automatic_independent_prices(
        verification,
        twelve_data_api_key=str(runtime.get("twelve_data_api_key", "") or ""),
        itick_api_token=str(runtime.get("itick_api_token", "") or ""),
        primary_reference={ticker: (frame.index[-1], float(pd.to_numeric(frame["Close"], errors="coerce").dropna().iloc[-1])) for ticker, frame in verify_histories.items() if isinstance(frame, pd.DataFrame) and not frame.empty and "Close" in frame},
        primary_source_tiers=getattr(download_report, "source_tiers", {}) if download_report is not None else {},
        config=cfg,
    ) if verification else (pd.DataFrame(), pd.DataFrame())
    price_validation = build_independent_price_validation(
        verify_histories,
        independent,
        config=cfg,
        primary_source_tiers=getattr(download_report, "source_tiers", {}) if download_report is not None else {},
    ) if verification else pd.DataFrame()
    if not final_signals.empty:
        final_signals = finalize_execution_integrity(
            attach_position_sizing(apply_independent_price_gate(final_signals, price_validation, cfg), cfg), cfg
        )
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
    }
    if hasattr(bridge, "persist_scan_result"):
        try:
            result["database_sync_report"] = bridge.persist_scan_result(result)
        except Exception as exc:
            result["database_sync_report"] = pd.DataFrame([{
                "provider": "DATABASE_FINAL_SYNC", "state": "FAIL_SOFT",
                "error": f"{type(exc).__name__}: {str(exc)[:240]}",
            }])
    else:
        result["database_sync_report"] = pd.DataFrame([{
            "provider": "DATABASE_FINAL_SYNC", "state": "NOT_AVAILABLE",
        }])

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
    for artifact_type, payload in artifacts.items():
        bridge.persist_scan_job_artifact(job_id, artifact_type, json_safe(payload), model_version=ENGINE_VERSION)
    return {
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
        "finalized_at": datetime.now(timezone.utc).isoformat(),
        "worker_id": worker_id,
    }


__all__ = [
    "ENGINE_VERSION",
    "process_backfill_chunk",
    "finalize_backfill_job",
    "process_daily_scan_chunk",
    "finalize_daily_scan_job",
]
