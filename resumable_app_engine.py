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
)
from scanner_database import ScannerDatabaseBridge
from simple_focus import build_simple_focus, build_silent_profiles
from two_stage_pipeline import ShortlistConfig, build_enrichment_shortlist, build_lightweight_preliminary_focus

ENGINE_VERSION = "9.4.0"


def _ticker(value: Any) -> str:
    text = str(value or "").upper().strip()
    return text if text.endswith(".JK") else f"{text}.JK" if text else ""


def _merge_primary(primary: pd.DataFrame | None, fallback: pd.DataFrame | None) -> pd.DataFrame:
    frames = [frame.copy() for frame in (primary, fallback) if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
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
    provider_report = pd.concat(
        [frame for frame in (idx_report, yahoo_report, twelve_report) if isinstance(frame, pd.DataFrame) and not frame.empty],
        ignore_index=True,
        sort=False,
    ) if any(isinstance(frame, pd.DataFrame) and not frame.empty for frame in (idx_report, yahoo_report, twelve_report)) else pd.DataFrame()

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
        "ticker_count": len(tickers),
    }
    sync = bridge.persist_scan_result(persist)
    counts = _history_counts(history)
    snapshot_ready = set()
    if not fundamentals.empty and "fundamental_score_eligible" in fundamentals.columns:
        snapshot_ready = set(fundamentals.loc[_coerce_bool_series(fundamentals["fundamental_score_eligible"], default=False), "ticker"].map(_ticker))
    market_ready = set(market.get("ticker", pd.Series(dtype=str)).map(_ticker)) if not market.empty else set()
    sync_failed = False
    if not sync.empty and "state" in sync.columns:
        sync_failed = sync["state"].astype(str).str.upper().isin({"DATABASE_FAIL_SOFT", "CONFIG_INCOMPLETE", "CONFIG_UNSAFE_KEY"}).any()

    by_ticker: dict[str, ItemOutcome] = {}
    item_lookup = {str(row.get("ticker")): str(row.get("item_key")) for row in items.to_dict("records")}
    for ticker in tickers:
        key = item_lookup.get(ticker)
        if not key:
            continue
        core_ready = ticker in snapshot_ready and int(counts.get(ticker, 0)) >= 2
        payload = {
            "ticker": ticker,
            "snapshot_ready": ticker in snapshot_ready,
            "history_periods": int(counts.get(ticker, 0)),
            "market_status_ready": ticker in market_ready,
            "worker_id": worker_id,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        if core_ready and not sync_failed:
            by_ticker[key] = ItemOutcome(True, payload=payload)
        else:
            reason = "DATABASE_WRITE_FAILED" if sync_failed else f"CORE_UNRESOLVED snapshot={ticker in snapshot_ready} periods={counts.get(ticker, 0)}"
            by_ticker[key] = ItemOutcome(False, payload=payload, error=reason, retry_delay_seconds=20)
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
    bridge.persist_scan_job_artifact(str(job.get("job_id")), "BACKFILL_COVERAGE", json_safe(coverage), model_version=ENGINE_VERSION)
    bridge.persist_scan_job_artifact(str(job.get("job_id")), "BACKFILL_SUMMARY", json_safe(summary), model_version=ENGINE_VERSION)
    row = summary.iloc[0].to_dict() if not summary.empty else {}
    return {**json_safe(row), "worker_id": worker_id, "finalized_at": datetime.now(timezone.utc).isoformat()}


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
    fundamentals, _, _ = _load_fundamentals(bridge, tickers, cfg)
    market, news = _load_aux(bridge, tickers, cfg)
    existing_events = bridge.read_narrative_events(tickers, limit=max(200, len(tickers) * 20))
    existing_outcomes = bridge.read_narrative_event_outcomes(tickers, limit=max(200, len(tickers) * 20))

    histories, download_report = download_ohlcv(tickers, period=period, itick_api_token=str(runtime.get("itick_api_token", "") or ""))
    benchmark = download_benchmark(period=period)
    bounded = {ticker: frame.tail(750).copy() for ticker, frame in histories.items() if isinstance(frame, pd.DataFrame) and not frame.empty}
    core = ScanEngine(cfg).scan(bounded, benchmark.tail(750).copy() if isinstance(benchmark, pd.DataFrame) else benchmark)
    base = _merge_primary(core.get("signals", pd.DataFrame()), core.get("universe", pd.DataFrame()))
    if not base.empty:
        base = apply_universe_integrity_gate(base, tickers, core.get("prepared", {}).keys(), cfg)
        base = attach_ohlcv_source_lineage(base, getattr(download_report, "source_tiers", {}) or {})
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
    item_lookup = {str(row.get("ticker")): str(row.get("item_key")) for row in items.to_dict("records")}
    result: dict[str, ItemOutcome] = {}
    for ticker in tickers:
        key = item_lookup.get(ticker)
        if not key:
            continue
        frame = core.get("prepared", {}).get(ticker)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            result[key] = ItemOutcome(False, payload={"ticker": ticker}, error="OHLCV_NOT_PREPARED", retry_delay_seconds=30)
            continue
        ticker_events = events.loc[events["ticker"].astype(str).eq(ticker)].to_dict("records") if isinstance(events, pd.DataFrame) and not events.empty and "ticker" in events else []
        ticker_outcomes = outcomes.loc[outcomes["ticker"].astype(str).eq(ticker)].to_dict("records") if isinstance(outcomes, pd.DataFrame) and not outcomes.empty and "ticker" in outcomes else []
        payload = {
            "ticker": ticker,
            "signal": json_safe(signal_map.get(ticker, {"ticker": ticker, "status": "PENDING_DATA"})),
            "silent_profile": json_safe({"ticker": ticker, **silent_map.get(ticker, {})}),
            "narrative_profile": json_safe({"ticker": ticker, **profile_map.get(ticker, {})}),
            "narrative_events": json_safe(ticker_events),
            "narrative_outcomes": json_safe(ticker_outcomes),
            "breadth": json_safe(_breadth_row(frame)),
            "worker_id": worker_id,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        result[key] = ItemOutcome(True, payload=payload)
    return result


def _unpack_job_items(items: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, float], list[str]]:
    signal_rows: list[dict[str, Any]] = []
    silent_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    tickers: list[str] = []
    breadth_rows: list[dict[str, Any]] = []
    for row in items.to_dict("records"):
        if str(row.get("status", "")).upper() != "COMPLETE":
            continue
        payload = row.get("result_payload") or {}
        if not isinstance(payload, Mapping):
            continue
        ticker = _ticker(payload.get("ticker") or row.get("ticker"))
        if not ticker:
            continue
        tickers.append(ticker)
        signal = payload.get("signal")
        if isinstance(signal, Mapping): signal_rows.append(dict(signal))
        silent = payload.get("silent_profile")
        if isinstance(silent, Mapping): silent_rows.append(dict(silent))
        profile = payload.get("narrative_profile")
        if isinstance(profile, Mapping): profile_rows.append(dict(profile))
        if isinstance(payload.get("narrative_events"), list): event_rows.extend([dict(v) for v in payload["narrative_events"] if isinstance(v, Mapping)])
        if isinstance(payload.get("narrative_outcomes"), list): outcome_rows.extend([dict(v) for v in payload["narrative_outcomes"] if isinstance(v, Mapping)])
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
        "audit": pd.DataFrame([{"state": "RESUMED_FROM_DURABLE_CHUNKS", "ticker_count": len(set(tickers))}]),
    }
    return pd.DataFrame(signal_rows), pd.DataFrame(silent_rows), narrative, breadth_features, list(dict.fromkeys(tickers))


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
    signals, silent_frame, narrative, breadth_features, completed_tickers = _unpack_job_items(items)
    universe = [_ticker(value) for value in (job.get("universe_payload") or [])]
    fundamentals, history, fundamental_report = _load_fundamentals(bridge, universe, cfg)
    market, news = _load_aux(bridge, universe, cfg)
    period = str(config.get("period", "5y") or "5y")
    benchmark = download_benchmark(period=period)
    macro_series, macro_report = fetch_macro_series(period="6mo", timeout=8)
    macro = build_macro_regime(
        benchmark=benchmark,
        prepared={},
        fundamentals=fundamentals,
        macro_series=macro_series,
        source_report=macro_report,
        breadth_features=breadth_features,
    )
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
    verify_histories, download_report = download_ohlcv(verification, period=period, itick_api_token=str(runtime.get("itick_api_token", "") or "")) if verification else ({}, None)
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
        extra, _ = download_ohlcv(missing_portfolio, period=period, itick_api_token=str(runtime.get("itick_api_token", "") or ""))
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
        "macro_source_report": macro.source_report,
        "ihsg_direction": analyze_ihsg_direction(benchmark, {}, config=IHSGDirectionConfig()),
        "database_coverage_after": coverage,
        "database_summary_after": coverage_summary,
        "two_stage_shortlist": shortlist,
        "narrative_events": narrative.get("events", pd.DataFrame()),
        "narrative_event_outcomes": narrative.get("outcomes", pd.DataFrame()),
        "narrative_profiles": narrative.get("profiles", pd.DataFrame()),
    }
    result["database_sync_report"] = bridge.persist_scan_result(result)

    artifacts = {
        "FINAL_NEXT_LEADERS": focus.get("next_leaders", pd.DataFrame()),
        "FINAL_SWING_READY": focus.get("swing_ready", pd.DataFrame()),
        "FINAL_MACRO_SNAPSHOT": macro.snapshot,
        "FINAL_MACRO_SECTOR_MAP": macro.sector_map,
        "FINAL_PORTFOLIO": portfolio_analysis,
        "FINAL_COVERAGE": coverage,
        "FINAL_JOB_AUDIT": pd.DataFrame([{
            "requested_tickers": len(universe),
            "completed_tickers": len(completed_tickers),
            "completed_ticker_list": list(completed_tickers),
            "failed_tickers": max(0, len(universe) - len(completed_tickers)),
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
        "completed_tickers": len(completed_tickers),
        "failed_tickers": max(0, len(universe) - len(completed_tickers)),
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
