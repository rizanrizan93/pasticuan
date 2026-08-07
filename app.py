from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd
import streamlit as st


APP_VERSION = "9.6.0-quality-integrity"
APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

st.set_page_config(
    page_title="IDX Scanner v9 Macro-First",
    page_icon="📊",
    layout="wide",
)

REQUIRED_FILES = (
    "scanner.py",
    "scanner_database.py",
    "narrative_engine.py",
    "macro_engine.py",
    "simple_focus.py",
    "two_stage_pipeline.py",
    "free_data_providers.py",
    "ihsg_direction.py",
    "idx_trading_calendar.py",
    "incremental_store.py",
    "research_maintenance.py",
    "selector_engine.py",
    "database_first.py",
    "resumable_scan.py",
    "resumable_app_engine.py",
)
missing = [name for name in REQUIRED_FILES if not (APP_ROOT / name).is_file()]
if missing:
    st.error("Deployment v9 tidak lengkap.")
    st.code("\n".join(missing), language="text")
    st.stop()

from scanner import (  # noqa: E402
    ScanConfig,
    ScanEngine,
    analyze_portfolio_positions,
    apply_execution_snapshot_gate,
    apply_fundamental_gate,
    apply_independent_price_gate,
    apply_market_status_gate,
    apply_news_gate,
    apply_universe_integrity_gate,
    apply_validation_gate,
    attach_backtest_stats,
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
    fetch_yahoo_fundamental_history,
    fetch_twelve_data_fundamental_history,
    finalize_execution_integrity,
    parse_portfolio_csv,
    parse_ticker_csv,
    read_cached_fundamental_history,
    read_cached_fundamentals,
    read_cached_market_status,
    read_cached_news_review,
    run_adaptive_walkforward_validation,
    select_yahoo_fundamental_tickers,
)
from scanner_database import ScannerDatabaseBridge  # noqa: E402
from narrative_engine import build_narrative_intelligence  # noqa: E402
from ihsg_direction import IHSGDirectionConfig, analyze_ihsg_direction  # noqa: E402
from macro_engine import (  # noqa: E402
    MACRO_ENGINE_VERSION,
    build_macro_regime,
    fetch_macro_series,
)
from simple_focus import SIMPLE_FOCUS_VERSION, build_simple_focus  # noqa: E402
from two_stage_pipeline import (  # noqa: E402
    ShortlistConfig,
    build_enrichment_shortlist,
    build_lightweight_preliminary_focus,
    build_two_stage_coverage_audit,
)
from database_first import (  # noqa: E402
    DATABASE_FIRST_VERSION,
    DatabaseReadinessPolicy,
    build_database_coverage,
    _coerce_bool_series,
    estimate_remaining_passes,
    readiness_summary,
    select_database_refresh_queue,
    select_evidence_refresh_queues,
    run_parallel_backfill_jobs,
)
from resumable_scan import (  # noqa: E402
    ACTIVE_JOB_STATES,
    RESUMABLE_SCAN_VERSION,
    frame_from_records,
    run_durable_job_loop,
    start_worker,
    worker_status,
)
from resumable_app_engine import (  # noqa: E402
    finalize_backfill_job,
    finalize_daily_scan_job,
    process_backfill_chunk,
    process_daily_scan_chunk,
)


st.markdown(
    """
    <style>
      .block-container {padding-top: 1rem; padding-bottom: 2.5rem;}
      [data-testid="stMetricValue"] {font-size: 1.35rem;}
      .v9-note {border:1px solid #334155; border-radius:10px; padding:12px 14px; background:#0f172a;}
      .small-muted {font-size:.84rem; color:#94a3b8;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _finite(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _merge_prefer_primary(primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    frames = [frame.copy() for frame in (primary, fallback) if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0].reset_index(drop=True)
    # Exclude all-NA columns before concat so pandas does not perform deprecated
    # dtype inference. Re-add columns that are all-NA across every input after the
    # merge. Primary order still encodes precedence.
    ordered_columns = list(dict.fromkeys(column for frame in frames for column in frame.columns))
    useful_frames = []
    for frame in frames:
        keep = [column for column in frame.columns if not frame[column].isna().all()]
        useful_frames.append(frame.loc[:, keep].copy())
    combined = pd.concat(useful_frames, ignore_index=True, sort=False)
    # Reindex adds any all-NA-only columns in one operation without fragmenting
    # an already wide scanner DataFrame.
    combined = combined.reindex(columns=ordered_columns)
    if "ticker" not in combined.columns:
        return combined.reset_index(drop=True)
    combined["ticker"] = combined["ticker"].astype(str).str.upper().str.strip()
    return combined.drop_duplicates("ticker", keep="first").reset_index(drop=True)


def _bounded_histories(histories: Mapping[str, pd.DataFrame], *, max_bars: int) -> dict[str, pd.DataFrame]:
    limit = max(260, int(max_bars))
    return {
        str(ticker): frame.tail(limit).copy()
        for ticker, frame in histories.items()
        if isinstance(frame, pd.DataFrame) and not frame.empty
    }


def _primary_reference(histories: Mapping[str, pd.DataFrame]) -> dict[str, tuple[object, float]]:
    output: dict[str, tuple[object, float]] = {}
    for ticker, frame in histories.items():
        if not isinstance(frame, pd.DataFrame) or frame.empty or "Close" not in frame:
            continue
        close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
        if close.empty:
            continue
        output[str(ticker).upper()] = (frame.index[-1], float(close.iloc[-1]))
    return output


def _safe_display(frame: pd.DataFrame | None, columns: Iterable[str] | None = None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if columns is not None:
        selected = [column for column in columns if column in out.columns]
        out = out.loc[:, selected]
    for column in out.select_dtypes(include=["object"]).columns:
        out[column] = out[column].map(lambda value: "" if value is None else str(value))
    return out.reset_index(drop=True)


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_market_data(tickers: tuple[str, ...], period: str, itick_token: str):
    histories, report = download_ohlcv(tickers, period=period, itick_api_token=itick_token)
    benchmark = download_benchmark(period=period)
    return histories, report, benchmark


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_macro_data():
    return fetch_macro_series(period="6mo", timeout=12)


def _database_first_aux_evidence(tickers: list[str], cfg: ScanConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge persistent Supabase evidence with fresh local fallback caches."""
    bridge = ScannerDatabaseBridge()
    try:
        db_market, market_audit = bridge.read_market_status_cache(
            tickers, max_age_days=max(1, int(getattr(cfg, "market_status_cache_days", 3))),
        )
    except Exception as exc:
        db_market, market_audit = pd.DataFrame(), pd.DataFrame([{
            "provider": "SUPABASE_DATABASE_FIRST", "scope": "MARKET_STATUS",
            "status": "READ_FAIL_SOFT", "error": f"{type(exc).__name__}: {str(exc)[:160]}",
        }])
    try:
        db_news, news_audit = bridge.read_news_review_cache(
            tickers, max_age_days=max(1, int(getattr(cfg, "news_cache_days", 7))),
        )
    except Exception as exc:
        db_news, news_audit = pd.DataFrame(), pd.DataFrame([{
            "provider": "SUPABASE_DATABASE_FIRST", "scope": "NEWS_REVIEW",
            "status": "READ_FAIL_SOFT", "error": f"{type(exc).__name__}: {str(exc)[:160]}",
        }])
    local_market = read_cached_market_status(tickers, cfg)
    local_news = read_cached_news_review(tickers, lookback_days=45, config=cfg)
    market = _merge_prefer_primary(local_market, db_market)
    news = _merge_prefer_primary(local_news, db_news)
    audits = [frame for frame in (market_audit, news_audit) if isinstance(frame, pd.DataFrame) and not frame.empty]
    return market, news, pd.concat(audits, ignore_index=True, sort=False) if audits else pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def _cached_baseline_fundamentals(tickers: tuple[str, ...]):
    cfg = ScanConfig()
    bridge = ScannerDatabaseBridge()
    try:
        db_snapshot, db_snapshot_audit = bridge.read_fundamental_cache(list(tickers))
    except Exception as exc:
        db_snapshot, db_snapshot_audit = pd.DataFrame(), pd.DataFrame([{
            "provider": "SUPABASE_DATABASE_FIRST", "status": "READ_FAIL_SOFT",
            "error": f"{type(exc).__name__}: {str(exc)[:160]}",
        }])
    try:
        db_history, db_history_audit = bridge.read_fundamental_history_cache(list(tickers))
    except Exception as exc:
        db_history, db_history_audit = pd.DataFrame(), pd.DataFrame([{
            "provider": "SUPABASE_DATABASE_FIRST", "status": "READ_FAIL_SOFT",
            "error": f"{type(exc).__name__}: {str(exc)[:160]}",
        }])
    local_snapshot = read_cached_fundamentals(tickers, cfg)
    if not local_snapshot.empty and "fundamental_score_eligible" in local_snapshot:
        local_snapshot = local_snapshot.loc[_coerce_bool_series(local_snapshot["fundamental_score_eligible"], default=False)].copy()
    local_history = read_cached_fundamental_history(tickers)
    snapshot = _merge_prefer_primary(local_snapshot, db_snapshot)
    history = combine_fundamental_history(db_history, local_history)
    enriched = enrich_fundamentals_with_history(snapshot, history)
    cache_audit = pd.DataFrame([{
        "provider": "LOCAL_CACHE_ONLY", "status": "BASELINE_CACHE_READ",
        "requested_tickers": len(tickers),
        "snapshot_tickers": int(enriched["ticker"].nunique()) if not enriched.empty and "ticker" in enriched else 0,
        "history_tickers": int(history["ticker"].nunique()) if not history.empty and "ticker" in history else 0,
    }])
    reports = [frame for frame in (db_snapshot_audit, db_history_audit, cache_audit)
               if isinstance(frame, pd.DataFrame) and not frame.empty]
    return enriched, history, pd.concat(reports, ignore_index=True, sort=False) if reports else pd.DataFrame()



@st.cache_data(ttl=900, show_spinner=False)
def _cached_narrative_memory(tickers: tuple[str, ...]):
    bridge = ScannerDatabaseBridge()
    try:
        events = bridge.read_narrative_events(list(tickers))
    except Exception:
        events = pd.DataFrame()
    try:
        outcomes = bridge.read_narrative_event_outcomes(list(tickers))
    except Exception:
        outcomes = pd.DataFrame()
    return events, outcomes

def _enrich_shortlist_fundamentals(
    *,
    all_tickers: tuple[str, ...],
    shortlist: tuple[str, ...],
    official_refresh: tuple[str, ...],
    baseline_snapshot: pd.DataFrame,
    baseline_history: pd.DataFrame,
    baseline_report: pd.DataFrame,
    cfg: ScanConfig,
    yahoo_limit: int = 24,
):
    live_snapshot = fetch_resilient_fundamentals(shortlist, cfg)
    if not live_snapshot.empty and "fundamental_score_eligible" in live_snapshot:
        usable_live = live_snapshot.loc[_coerce_bool_series(live_snapshot["fundamental_score_eligible"], default=False)].copy()
    else:
        usable_live = live_snapshot
    snapshot = _merge_prefer_primary(usable_live, baseline_snapshot)
    idx_history, idx_report = fetch_idx_fundamental_history(
        official_refresh,
        max_tickers=len(official_refresh),
        years_back=max(1, int(getattr(cfg, "idx_fundamental_years_back", 3))),
    )
    history_before_yahoo = combine_fundamental_history(baseline_history, idx_history)
    yahoo_targets = select_yahoo_fundamental_tickers(
        shortlist,
        history_before_yahoo,
        max_tickers=max(0, min(int(yahoo_limit), len(shortlist))),
        crosscheck_top_n=min(8, len(shortlist)),
    )
    yahoo_history, yahoo_report = fetch_yahoo_fundamental_history(yahoo_targets, max_tickers=len(yahoo_targets))
    history = combine_fundamental_history(baseline_history, idx_history, yahoo_history)
    enriched = enrich_fundamentals_with_history(snapshot, history)
    snapshot_set = set(enriched.get("ticker", pd.Series(dtype=str)).dropna().astype(str)) if not enriched.empty else set()
    history_set = set(history.get("ticker", pd.Series(dtype=str)).dropna().astype(str)) if not history.empty else set()
    refresh_state_rows = []
    for ticker in shortlist:
        refresh_state_rows.extend([
            {
                "ticker": ticker, "provider": "DATABASE_DELTA_REFRESH", "scope": "FUNDAMENTAL_SNAPSHOT",
                "status": "CURRENT" if ticker in snapshot_set else "REFRESH_UNRESOLVED",
                "database_read_state": "LIVE_OR_CACHE_CURRENT" if ticker in snapshot_set else "REFRESH_UNRESOLVED",
                "refresh_required": ticker not in snapshot_set, "rows": int(ticker in snapshot_set),
            },
            {
                "ticker": ticker, "provider": "DATABASE_DELTA_REFRESH", "scope": "FUNDAMENTAL_HISTORY",
                "status": "CURRENT" if ticker in history_set else "REFRESH_UNRESOLVED",
                "database_read_state": "LIVE_OR_CACHE_CURRENT" if ticker in history_set else "REFRESH_UNRESOLVED",
                "refresh_required": ticker not in history_set, "rows": int(ticker in history_set),
            },
        ])
    reports = [frame for frame in (
        baseline_report, idx_report, yahoo_report,
        pd.DataFrame(refresh_state_rows),
        pd.DataFrame([{
            "provider": "DATABASE_FIRST_PLANNER", "status": "DELTA_REFRESH",
            "universe_tickers": len(all_tickers), "refresh_tickers": len(shortlist),
            "official_refresh_tickers": len(official_refresh), "yahoo_history_tickers": len(yahoo_targets),
        }]),
    ) if isinstance(frame, pd.DataFrame) and not frame.empty]
    return enriched, history, pd.concat(reports, ignore_index=True, sort=False) if reports else pd.DataFrame()


def _simple_contract_audit(
    tickers: list[str],
    prepared: Mapping[str, pd.DataFrame],
    fundamentals: pd.DataFrame,
    focus: Mapping[str, pd.DataFrame],
    macro_issuer: pd.DataFrame,
) -> pd.DataFrame:
    fund_set = set(fundamentals.get("ticker", pd.Series(dtype=str)).astype(str)) if not fundamentals.empty else set()
    leader = focus.get("next_leaders", pd.DataFrame())
    swing = focus.get("swing_ready", pd.DataFrame())
    leader_map = leader.set_index("ticker").to_dict("index") if not leader.empty else {}
    swing_map = swing.set_index("ticker").to_dict("index") if not swing.empty else {}
    macro_set = set(macro_issuer.get("ticker", pd.Series(dtype=str)).astype(str)) if not macro_issuer.empty else set()
    rows = []
    for ticker in tickers:
        key = str(ticker).upper()
        rows.append({
            "ticker": key,
            "ohlcv_ready": key in prepared,
            "fundamental_ready": key in fund_set,
            "macro_issuer_ready": key in macro_set,
            "next_leader_score": leader_map.get(key, {}).get("v9_next_leader_score", np.nan),
            "next_leader_status": leader_map.get(key, {}).get("status", "DATA_PENDING"),
            "swing_score": swing_map.get(key, {}).get("v9_swing_score", np.nan),
            "swing_status": swing_map.get(key, {}).get("status", "DATA_PENDING"),
        })
    return pd.DataFrame(rows)



def _history_period_counts(history: pd.DataFrame | None) -> dict[str, int]:
    if history is None or history.empty or "ticker" not in history.columns:
        return {}
    local = history.copy()
    local["ticker"] = local["ticker"].astype(str).str.upper().str.strip()
    period_column = next((column for column in ("period_end", "statement_date", "date", "as_of") if column in local.columns), None)
    if period_column is None:
        return local.groupby("ticker").size().astype(int).to_dict()
    local[period_column] = pd.to_datetime(local[period_column], errors="coerce")
    return local.dropna(subset=[period_column]).groupby("ticker")[period_column].nunique().astype(int).to_dict()


def _mark_history_derived_eligibility(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """Allow verified statement history to serve as the production snapshot.

    A separate quoteSummary call is not required when normalized statements already
    produce a finite business score and adequate coverage. This removes duplicate
    provider work during database backfill without inventing data.
    """
    if fundamentals is None or fundamentals.empty or "ticker" not in fundamentals.columns:
        return pd.DataFrame() if fundamentals is None else fundamentals.copy()
    out = fundamentals.copy()
    score = pd.to_numeric(out.get("fundamental_score", pd.Series(np.nan, index=out.index)), errors="coerce")
    coverage = pd.to_numeric(out.get("fundamental_coverage", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    age = pd.to_numeric(out.get("statement_age_days", pd.Series(np.nan, index=out.index)), errors="coerce")
    source_count = pd.to_numeric(out.get("fundamental_source_count", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
    existing = _coerce_bool_series(
        out.get("fundamental_score_eligible", pd.Series(False, index=out.index)),
        default=False,
    )
    history_ready = (
        score.notna()
        & coverage.ge(45.0)
        & source_count.ge(1.0)
        & (age.isna() | age.le(550.0))
    )
    eligible = existing | history_ready
    out["fundamental_score_eligible"] = eligible.astype(bool)
    derived = history_ready & ~existing
    if derived.any():
        provider = out.get("fundamental_provider", pd.Series("", index=out.index)).fillna("").astype(str)
        source_family = out.get("fundamental_source_families", pd.Series("", index=out.index)).fillna("").astype(str)
        out.loc[derived, "fundamental_provider"] = provider.loc[derived].where(
            provider.loc[derived].str.strip().ne(""),
            source_family.loc[derived].where(source_family.loc[derived].str.strip().ne(""), "STATEMENT_HISTORY"),
        )
        out.loc[derived, "fundamental_route_state"] = "HISTORY_DERIVED_VERIFIED"
        now = pd.Timestamp.now(tz="Asia/Jakarta").isoformat()
        if "fundamental_fetched_at" not in out.columns:
            out["fundamental_fetched_at"] = pd.NA
        out.loc[derived & out["fundamental_fetched_at"].isna(), "fundamental_fetched_at"] = now
    return out


def _provider_status_summary(*reports: pd.DataFrame) -> pd.DataFrame:
    frames = [frame.copy() for frame in reports if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "provider" not in combined.columns:
        combined["provider"] = "UNKNOWN"
    if "status" not in combined.columns:
        combined["status"] = "UNKNOWN"
    combined["provider"] = combined["provider"].fillna("UNKNOWN").astype(str)
    combined["status"] = combined["status"].fillna("UNKNOWN").astype(str).str.upper()
    grouped = combined.groupby(["provider", "status"], dropna=False).size().rename("rows").reset_index()
    success_statuses = {"OK", "PARTIAL", "CURRENT", "CACHE_FALLBACK", "VERIFIED", "COMPLETE"}
    grouped["success_class"] = grouped["status"].isin(success_statuses).map({True: "SUCCESS", False: "OTHER"})
    return grouped.sort_values(["provider", "success_class", "rows"], ascending=[True, True, False], kind="stable").reset_index(drop=True)


def _run_database_backfill(
    *,
    all_tickers: list[str],
    portfolio_tickers: list[str],
    batch_size: int,
    yahoo_limit: int,
    market_batch_size: int,
    news_batch_size: int,
    twelve_api_key: str,
    cfg: ScanConfig,
    official_idx_refresh: bool = False,
    include_news: bool = False,
    snapshot_fallback_limit: int = 8,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, object]:
    """Fast core-first database fill.

    Fundamental history is acquired once and used to derive the latest production
    snapshot. News and narrative enrichment are optional and no longer block the
    first stock scan. Only delta rows are persisted, reducing provider and database
    round trips on a 150-400 ticker universe.
    """
    started = time.perf_counter()

    def progress(value: int, label: str) -> None:
        if progress_callback is not None:
            try:
                progress_callback(int(value), str(label))
            except Exception:
                pass

    progress(5, "Membaca cache dan database persisten…")
    baseline_fundamentals, baseline_history, baseline_report = _cached_baseline_fundamentals(tuple(all_tickers))
    cached_market_status, cached_news_review, auxiliary_database_audit = _database_first_aux_evidence(all_tickers, cfg)
    baseline_fundamentals = _mark_history_derived_eligibility(
        enrich_fundamentals_with_history(baseline_fundamentals, baseline_history)
    )

    coverage_before = build_database_coverage(
        all_tickers,
        fundamentals=baseline_fundamentals,
        fundamental_history=baseline_history,
        market_status=cached_market_status,
        news_review=cached_news_review,
        fundamental_report=baseline_report,
    )
    summary_before = readiness_summary(coverage_before)
    progress(15, "Menyusun antrean core MISSING/STALE…")
    queues, refresh_audit = select_evidence_refresh_queues(
        all_tickers,
        coverage_before,
        snapshot_batch_size=max(0, int(batch_size)),
        history_batch_size=max(0, int(batch_size)),
        market_batch_size=max(0, int(market_batch_size)),
        news_batch_size=max(0, int(news_batch_size)) if include_news else 0,
        portfolio_tickers=portfolio_tickers,
    )
    # Snapshot and history are one core workload. History is authoritative and the
    # snapshot is derived from it; no duplicate full-batch snapshot provider pass.
    core_targets = list(dict.fromkeys(
        queues.get("FUNDAMENTAL_HISTORY", []) + queues.get("FUNDAMENTAL_SNAPSHOT", [])
    ))[:max(0, int(batch_size))]
    snapshot_targets = list(core_targets)
    history_targets = list(core_targets)
    market_targets = queues.get("MARKET_STATUS", [])
    news_targets = queues.get("NEWS_REVIEW", []) if include_news else []
    queues["FUNDAMENTAL_SNAPSHOT"] = snapshot_targets
    queues["FUNDAMENTAL_HISTORY"] = history_targets
    queues["NEWS_REVIEW"] = news_targets
    if isinstance(refresh_audit, pd.DataFrame) and not refresh_audit.empty:
        selected_map = {
            "FUNDAMENTAL_SNAPSHOT": set(snapshot_targets),
            "FUNDAMENTAL_HISTORY": set(history_targets),
            "MARKET_STATUS": set(market_targets),
            "NEWS_REVIEW": set(news_targets),
        }
        refresh_audit = refresh_audit.copy()
        refresh_audit["selected_for_refresh"] = refresh_audit.apply(
            lambda row: str(row.get("ticker", "")) in selected_map.get(str(row.get("queue", "")), set()), axis=1
        )
    all_refresh_targets = list(dict.fromkeys(core_targets + market_targets + news_targets))

    if not all_refresh_targets:
        return {
            "state": "DATABASE_ALREADY_CURRENT",
            "coverage_before": coverage_before,
            "coverage_after": coverage_before,
            "summary_before": summary_before,
            "summary_after": summary_before,
            "refresh_queue": refresh_audit,
            "queue_summary": pd.DataFrame(),
            "provider_summary": pd.DataFrame(),
            "fundamentals": baseline_fundamentals,
            "fundamental_history": baseline_history,
            "market_status": cached_market_status,
            "news_review": cached_news_review,
            "fundamental_report": pd.concat(
                [frame for frame in (baseline_report, auxiliary_database_audit) if isinstance(frame, pd.DataFrame) and not frame.empty],
                ignore_index=True, sort=False,
            ) if any(isinstance(frame, pd.DataFrame) and not frame.empty for frame in (baseline_report, auxiliary_database_audit)) else pd.DataFrame(),
            "database_sync_report": pd.DataFrame(),
            "elapsed_seconds": round(time.perf_counter() - started, 2),
            "remaining_passes": 0.0,
            "stage_timing": pd.DataFrame(),
            "narrative_engine_audit": pd.DataFrame(),
        }

    def core_job():
        if official_idx_refresh and history_targets:
            idx_history, idx_report = fetch_idx_fundamental_history(
                history_targets,
                max_tickers=len(history_targets),
                years_back=min(2, max(1, int(getattr(cfg, "idx_fundamental_years_back", 2)))),
                timeout=4,
                max_workers=8,
            )
        else:
            idx_history = pd.DataFrame()
            idx_report = pd.DataFrame([{
                "provider": "IDX_OFFICIAL_XBRL",
                "status": "SKIPPED_FAST_CORE",
                "rows": 0,
                "error": "Verifikasi IDX live dimatikan pada fast core; cache official tetap digunakan",
                "error_code": "OPTIONAL_OFFICIAL_REFRESH_DISABLED",
            }])
        history_after_idx = combine_fundamental_history(baseline_history, idx_history)
        yahoo_targets = select_yahoo_fundamental_tickers(
            history_targets,
            history_after_idx,
            max_tickers=max(0, min(int(yahoo_limit), len(history_targets))),
            crosscheck_top_n=0,
            min_official_periods=2,
        )
        yahoo_history, yahoo_report = fetch_yahoo_fundamental_history(
            yahoo_targets,
            max_workers=8,
            max_tickers=len(yahoo_targets),
            enable_yfinance_fallback=False,
        ) if yahoo_targets else (pd.DataFrame(), pd.DataFrame())
        history_after_yahoo = combine_fundamental_history(history_after_idx, yahoo_history)
        period_counts = _history_period_counts(history_after_yahoo)
        twelve_candidates = [ticker for ticker in history_targets if int(period_counts.get(ticker, 0)) < 2]
        twelve_cap = max(0, min(int(yahoo_limit), 8, len(twelve_candidates)))
        twelve_targets = twelve_candidates[:twelve_cap]
        if twelve_targets and str(twelve_api_key or "").strip():
            twelve_history, twelve_report = fetch_twelve_data_fundamental_history(
                twelve_targets,
                api_key=twelve_api_key,
                max_tickers=len(twelve_targets),
                timeout=5,
                max_workers=8,
            )
        else:
            twelve_history = pd.DataFrame()
            twelve_report = pd.DataFrame([{
                "provider": "TWELVE_DATA",
                "status": "DISABLED" if not str(twelve_api_key or "").strip() else "NOT_REQUIRED",
                "rows": 0,
                "error": "API key tidak dikonfigurasi" if not str(twelve_api_key or "").strip() else "Tidak ada target fallback",
                "error_code": "PROVIDER_DISABLED" if not str(twelve_api_key or "").strip() else "",
            }])
        history = combine_fundamental_history(history_after_yahoo, twelve_history)
        fundamentals = _mark_history_derived_eligibility(
            enrich_fundamentals_with_history(baseline_fundamentals, history)
        )
        ready = set()
        if not fundamentals.empty and "fundamental_score_eligible" in fundamentals.columns:
            ready = set(
                fundamentals.loc[_coerce_bool_series(fundamentals["fundamental_score_eligible"], default=False), "ticker"]
                .dropna().astype(str).str.upper().str.strip()
            )
        fallback_targets = [ticker for ticker in core_targets if ticker not in ready][:max(0, int(snapshot_fallback_limit))]
        if fallback_targets:
            live_snapshot = fetch_resilient_fundamentals(fallback_targets, cfg)
            if not live_snapshot.empty and "fundamental_score_eligible" in live_snapshot.columns:
                live_snapshot = live_snapshot.loc[
                    _coerce_bool_series(live_snapshot["fundamental_score_eligible"], default=False)
                ].copy()
            snapshot = _merge_prefer_primary(live_snapshot, baseline_fundamentals)
            fundamentals = _mark_history_derived_eligibility(
                enrich_fundamentals_with_history(snapshot, history)
            )
        else:
            fallback_targets = []
        return {
            "fundamentals": fundamentals,
            "history": history,
            "idx_report": idx_report,
            "yahoo_report": yahoo_report,
            "twelve_report": twelve_report,
            "yahoo_targets": list(yahoo_targets),
            "twelve_targets": list(twelve_targets),
            "snapshot_fallback_targets": list(fallback_targets),
        }

    def market_job():
        live = fetch_resilient_market_status(market_targets, cfg) if market_targets else pd.DataFrame()
        return _merge_prefer_primary(live, cached_market_status)

    jobs: dict[str, Callable[[], object]] = {"CORE_FUNDAMENTALS": core_job}
    if market_targets:
        jobs["MARKET_STATUS"] = market_job
    if include_news and news_targets:
        jobs["NEWS_REVIEW"] = lambda: _merge_prefer_primary(
            fetch_resilient_news_review(news_targets, lookback_days=30, config=cfg),
            cached_news_review,
        )
    progress(25, "Mengisi core fundamental dan market status secara paralel…")
    parallel_results, stage_timing = run_parallel_backfill_jobs(jobs, max_workers=min(3, len(jobs)))
    progress(70, "Menggabungkan history dan menurunkan snapshot terbaru…")

    core_result = parallel_results.get("CORE_FUNDAMENTALS", {})
    if not isinstance(core_result, Mapping):
        core_result = {}
    fundamentals = core_result.get("fundamentals", baseline_fundamentals)
    history = core_result.get("history", baseline_history)
    idx_report = core_result.get("idx_report", pd.DataFrame())
    yahoo_report = core_result.get("yahoo_report", pd.DataFrame())
    twelve_report = core_result.get("twelve_report", pd.DataFrame())
    yahoo_targets = list(core_result.get("yahoo_targets", []))
    twelve_targets = list(core_result.get("twelve_targets", []))
    snapshot_fallback_targets = list(core_result.get("snapshot_fallback_targets", []))
    market_result = parallel_results.get("MARKET_STATUS", cached_market_status)
    market_status = market_result if isinstance(market_result, pd.DataFrame) else cached_market_status
    news_result = parallel_results.get("NEWS_REVIEW", cached_news_review)
    news_review = news_result if isinstance(news_result, pd.DataFrame) else cached_news_review

    if not fundamentals.empty and "fundamental_score_eligible" in fundamentals.columns:
        snapshot_ready_set = set(
            fundamentals.loc[_coerce_bool_series(fundamentals["fundamental_score_eligible"], default=False), "ticker"]
            .dropna().astype(str).str.upper().str.strip()
        )
    else:
        snapshot_ready_set = set()
    history_counts = _history_period_counts(history)
    history_ready_set = {ticker for ticker, count in history_counts.items() if int(count) >= 2}
    market_ready_set = set()
    if not market_status.empty and "ticker" in market_status.columns:
        eligible = _coerce_bool_series(
            market_status.get("market_status_score_eligible", pd.Series(True, index=market_status.index)),
            default=False,
        )
        market_ready_set = set(market_status.loc[eligible, "ticker"].dropna().astype(str).str.upper().str.strip())
    news_ready_set = set()
    if not news_review.empty and "ticker" in news_review.columns:
        eligible = _coerce_bool_series(
            news_review.get("news_score_eligible", pd.Series(True, index=news_review.index)),
            default=False,
        )
        news_ready_set = set(news_review.loc[eligible, "ticker"].dropna().astype(str).str.upper().str.strip())

    queue_ready_sets = {
        "FUNDAMENTAL_SNAPSHOT": snapshot_ready_set,
        "FUNDAMENTAL_HISTORY": history_ready_set,
        "MARKET_STATUS": market_ready_set,
        "NEWS_REVIEW": news_ready_set,
    }
    refresh_state_rows: list[dict[str, object]] = []
    for queue_name, targets in queues.items():
        ready_set = queue_ready_sets.get(queue_name, set())
        for ticker in targets:
            ready = ticker in ready_set
            refresh_state_rows.append({
                "ticker": ticker,
                "provider": "FAST_CORE_BACKFILL",
                "scope": queue_name,
                "status": "CURRENT" if ready else "REFRESH_UNRESOLVED",
                "database_read_state": "LIVE_OR_CACHE_CURRENT" if ready else "REFRESH_UNRESOLVED",
                "refresh_required": not ready,
                "rows": int(ready),
            })
    queue_summary = pd.DataFrame([{
        "queue": queue_name,
        "selected": len(targets),
        "ready_after": sum(ticker in queue_ready_sets.get(queue_name, set()) for ticker in targets),
        "unresolved_after": sum(ticker not in queue_ready_sets.get(queue_name, set()) for ticker in targets),
    } for queue_name, targets in queues.items()])

    delta_report_frames = [frame for frame in (
        idx_report, yahoo_report, twelve_report, pd.DataFrame(refresh_state_rows),
        pd.DataFrame([{
            "provider": "FAST_CORE_BACKFILL",
            "status": "REFRESH_BATCH_COMPLETED",
            "requested_tickers": len(all_tickers),
            "core_targets": len(core_targets),
            "market_targets": len(market_targets),
            "news_targets": len(news_targets),
            "yahoo_history_tickers": len(yahoo_targets),
            "twelve_history_tickers": len(twelve_targets),
            "snapshot_fallback_tickers": len(snapshot_fallback_targets),
            "scope": "DATABASE_BACKFILL",
        }]),
    ) if isinstance(frame, pd.DataFrame) and not frame.empty]
    delta_report = pd.concat(delta_report_frames, ignore_index=True, sort=False) if delta_report_frames else pd.DataFrame()
    report_frames = [frame for frame in (baseline_report, auxiliary_database_audit, delta_report) if isinstance(frame, pd.DataFrame) and not frame.empty]
    fundamental_report = pd.concat(report_frames, ignore_index=True, sort=False) if report_frames else pd.DataFrame()

    core_set = set(core_targets)
    market_set = set(market_targets)
    news_set = set(news_targets)
    delta_fundamentals = fundamentals.loc[fundamentals["ticker"].astype(str).isin(core_set)].copy() if core_set and not fundamentals.empty and "ticker" in fundamentals else pd.DataFrame()
    delta_history = history.loc[history["ticker"].astype(str).isin(core_set)].copy() if core_set and not history.empty and "ticker" in history else pd.DataFrame()
    delta_market = market_status.loc[market_status["ticker"].astype(str).isin(market_set)].copy() if market_set and not market_status.empty and "ticker" in market_status else pd.DataFrame()
    delta_news = news_review.loc[news_review["ticker"].astype(str).isin(news_set)].copy() if news_set and not news_review.empty and "ticker" in news_review else pd.DataFrame()

    bridge = ScannerDatabaseBridge()
    persist_result = {
        "mode": "fast_core_database_backfill",
        "scanner_version": APP_VERSION,
        "fundamentals": delta_fundamentals,
        "fundamental_history": delta_history,
        "fundamental_history_report": delta_report,
        "database_read_report": pd.DataFrame(),
        "market_status": delta_market,
        "news_review": delta_news,
        "narrative_events": pd.DataFrame(),
        "narrative_event_outcomes": pd.DataFrame(),
        "narrative_profiles": pd.DataFrame(),
        "focus_screens": {},
    }
    progress(80, "Menyimpan hanya delta batch ke database…")
    persist_started = time.perf_counter()
    try:
        database_sync_report = bridge.persist_scan_result(persist_result)
    except Exception as exc:
        database_sync_report = pd.DataFrame([{
            "state": "DATABASE_FAIL_SOFT",
            "detail": f"{type(exc).__name__}: {str(exc)[:300]}",
        }])
    persistence_row = pd.DataFrame([{
        "stage": "DATABASE_DELTA_WRITE",
        "elapsed_seconds": round(time.perf_counter() - persist_started, 3),
        "state": "OK" if not database_sync_report.empty else "NO_REPORT",
        "error": "",
    }])
    stage_timing = pd.concat([stage_timing, persistence_row], ignore_index=True, sort=False)

    coverage_after = build_database_coverage(
        all_tickers,
        fundamentals=fundamentals,
        fundamental_history=history,
        market_status=market_status,
        news_review=news_review,
        fundamental_report=fundamental_report,
    )
    summary_after = readiness_summary(coverage_after)
    before = summary_before.iloc[0] if not summary_before.empty else {}
    after = summary_after.iloc[0] if not summary_after.empty else {}
    gains = {
        "snapshot_ready_gain": int(after.get("fundamental_snapshot_ready", 0)) - int(before.get("fundamental_snapshot_ready", 0)),
        "history_ready_gain": int(after.get("fundamental_history_ready", 0)) - int(before.get("fundamental_history_ready", 0)),
        "market_ready_gain": int(after.get("market_status_ready", 0)) - int(before.get("market_status_ready", 0)),
        "news_ready_gain": int(after.get("news_review_ready", 0)) - int(before.get("news_review_ready", 0)),
        "database_ready_gain": int(after.get("database_ready_tickers", 0)) - int(before.get("database_ready_tickers", 0)),
    }
    positive_gain = max([0] + [int(value) for value in gains.values()])
    critical_tables = set()
    if core_targets:
        critical_tables.update({"fundamental_cache", "fundamental_history_cache"})
    if market_targets or news_targets:
        critical_tables.add("source_events")
    persistence_failed = False
    if isinstance(database_sync_report, pd.DataFrame) and not database_sync_report.empty and critical_tables:
        sync_local = database_sync_report.copy()
        sync_local["table"] = sync_local.get("table", pd.Series("", index=sync_local.index)).fillna("").astype(str)
        sync_local["state"] = sync_local.get("state", pd.Series("", index=sync_local.index)).fillna("").astype(str).str.upper()
        critical_rows = sync_local.loc[sync_local["table"].isin(critical_tables)]
        persistence_failed = critical_rows.empty or critical_rows["state"].isin({
            "DATABASE_FAIL_SOFT", "PARTIAL_WRITE", "CONFIG_INCOMPLETE",
            "CONFIG_UNSAFE_KEY", "DISABLED_NO_DATABASE",
        }).any()
    model_state = str(after.get("database_state", "BACKFILL_REQUIRED"))
    if persistence_failed:
        state = "DATABASE_WRITE_FAILED"
    elif model_state == "READY_FOR_DAILY_SCAN":
        state = model_state
    elif all_refresh_targets and positive_gain <= 0:
        state = "BACKFILL_STALLED"
    else:
        state = "BACKFILL_PROGRESS"
    remaining = estimate_remaining_passes(coverage_after, max(1, int(batch_size)), net_ready_gain=positive_gain)
    try:
        _cached_baseline_fundamentals.clear()
    except Exception:
        pass
    progress(100, "Fast core backfill selesai")
    return {
        "state": state,
        "coverage_before": coverage_before,
        "coverage_after": coverage_after,
        "summary_before": summary_before,
        "summary_after": summary_after,
        "refresh_queue": refresh_audit,
        "queue_summary": queue_summary,
        "provider_summary": _provider_status_summary(delta_report, database_sync_report),
        "refreshed_tickers": pd.DataFrame({"ticker": all_refresh_targets}),
        "fundamentals": fundamentals,
        "fundamental_history": history,
        "market_status": market_status,
        "news_review": news_review,
        "fundamental_report": fundamental_report,
        "narrative_engine_audit": pd.DataFrame([{
            "state": "DEFERRED_TO_DAILY_SCAN",
            "detail": "Narrative/news enrichment tidak menahan core database backfill",
        }]),
        "database_sync_report": database_sync_report,
        "batch_gains": pd.DataFrame([gains]),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "stage_timing": stage_timing,
        "remaining_passes": remaining,
    }

def _render_database_backfill(result: Mapping[str, object]) -> None:
    summary_before = result.get("summary_before", pd.DataFrame())
    summary_after = result.get("summary_after", pd.DataFrame())
    before = summary_before.iloc[0] if isinstance(summary_before, pd.DataFrame) and not summary_before.empty else {}
    after = summary_after.iloc[0] if isinstance(summary_after, pd.DataFrame) and not summary_after.empty else {}

    first = st.columns(4)
    first[0].metric(
        "Snapshot ready",
        f"{_finite(after.get('fundamental_snapshot_pct')):.1f}%",
        f"{_finite(after.get('fundamental_snapshot_pct')) - _finite(before.get('fundamental_snapshot_pct')):+.1f} pp",
    )
    first[0].caption(f"Tersimpan: {int(_finite(after.get('fundamental_snapshot_stored'), 0))}/{int(_finite(after.get('requested_tickers'), 0))}")
    first[1].metric(
        "History ≥2 periode",
        f"{_finite(after.get('fundamental_history_pct')):.1f}%",
        f"{_finite(after.get('fundamental_history_pct')) - _finite(before.get('fundamental_history_pct')):+.1f} pp",
    )
    first[1].caption(f"Punya history: {int(_finite(after.get('fundamental_history_stored'), 0))}/{int(_finite(after.get('requested_tickers'), 0))}")
    first[2].metric(
        "Market status ready",
        f"{_finite(after.get('market_status_pct')):.1f}%",
        f"{_finite(after.get('market_status_pct')) - _finite(before.get('market_status_pct')):+.1f} pp",
    )
    first[3].metric(
        "News review ready",
        f"{_finite(after.get('news_review_pct')):.1f}%",
        f"{_finite(after.get('news_review_pct')) - _finite(before.get('news_review_pct')):+.1f} pp",
    )

    remaining_value = result.get("remaining_passes")
    try:
        remaining_number = float(remaining_value)
    except (TypeError, ValueError):
        remaining_number = np.nan
    remaining_label = "—" if not np.isfinite(remaining_number) else str(int(remaining_number))
    second = st.columns(3)
    second[0].metric("Database ready", f"{_finite(after.get('database_ready_pct')):.1f}%")
    second[1].metric("Estimasi proses tersisa", remaining_label)
    second[2].metric("Waktu backfill", f"{_finite(result.get('elapsed_seconds')):.1f} dtk")

    state = str(result.get("state", "BACKFILL_REQUIRED"))
    if state in {"READY_FOR_DAILY_SCAN", "DATABASE_ALREADY_CURRENT"}:
        st.success("Database telah memenuhi ambang Daily Scan.")
    elif state == "DATABASE_WRITE_FAILED":
        st.error("Evidence live berhasil diproses, tetapi write ke database persisten gagal atau parsial. Periksa Database Sync Audit sebelum melanjutkan.")
    elif state == "BACKFILL_STALLED":
        st.error(
            "Backfill berhenti tanpa tambahan evidence siap pakai. Jangan mengulang batch yang sama sebelum melihat provider audit; "
            "scanner kini menampilkan sumber yang gagal dan tidak lagi memberi estimasi proses palsu."
        )
    else:
        st.warning("Database belum memenuhi ambang Daily Scan, tetapi batch menghasilkan kemajuan. Lanjutkan backfill.")

    gains = result.get("batch_gains", pd.DataFrame())
    if isinstance(gains, pd.DataFrame) and not gains.empty:
        st.subheader("Kemajuan batch nyata")
        st.dataframe(_safe_display(gains), width="stretch", hide_index=True)
    st.dataframe(_safe_display(summary_after), width="stretch", hide_index=True)
    timing = result.get("stage_timing", pd.DataFrame())
    if isinstance(timing, pd.DataFrame) and not timing.empty:
        with st.expander("Waktu per tahap", expanded=True):
            st.dataframe(_safe_display(timing), width="stretch", hide_index=True)
    with st.expander("Ringkasan antrean", expanded=True):
        st.dataframe(_safe_display(result.get("queue_summary", pd.DataFrame())), width="stretch", hide_index=True)
    with st.expander("Refresh queue"):
        st.dataframe(_safe_display(result.get("refresh_queue", pd.DataFrame())), width="stretch", hide_index=True)
    with st.expander("Coverage per ticker"):
        st.dataframe(_safe_display(result.get("coverage_after", pd.DataFrame())), width="stretch", hide_index=True)
    with st.expander("Provider dan database audit", expanded=state in {"BACKFILL_STALLED", "DATABASE_WRITE_FAILED"}):
        st.dataframe(_safe_display(result.get("provider_summary", pd.DataFrame())), width="stretch", hide_index=True)
        st.dataframe(_safe_display(result.get("fundamental_report", pd.DataFrame())), width="stretch", hide_index=True)
        st.dataframe(_safe_display(result.get("narrative_engine_audit", pd.DataFrame())), width="stretch", hide_index=True)
        st.dataframe(_safe_display(result.get("database_sync_report", pd.DataFrame())), width="stretch", hide_index=True)



def _runtime_tokens(itick_token: str = "", twelve_token: str = "") -> dict[str, str]:
    def secret(name: str) -> str:
        try:
            return str(st.secrets.get(name, "") or "").strip()
        except Exception:
            return ""

    # Tokens are read at runtime only and are never persisted in scan_jobs.
    return {
        "itick_api_token": str(itick_token or os.getenv("ITICK_API_TOKEN", "") or secret("ITICK_API_TOKEN")).strip(),
        "twelve_data_api_key": str(twelve_token or os.getenv("TWELVE_DATA_API_KEY", "") or secret("TWELVE_DATA_API_KEY")).strip(),
    }


def _start_resumable_worker(job: Mapping[str, object], runtime: Mapping[str, str]) -> object:
    job_id = str(job.get("job_id", ""))
    job_type = str(job.get("job_type", "")).upper()
    runtime_copy = dict(runtime)

    def runner(worker_id: str) -> None:
        if job_type == "DATABASE_BACKFILL":
            process = lambda current_job, items, wid: process_backfill_chunk(
                current_job, items, wid, runtime=runtime_copy,
            )
            finalize = lambda current_job, bridge, wid: finalize_backfill_job(current_job, bridge, wid)
        else:
            process = lambda current_job, items, wid: process_daily_scan_chunk(
                current_job, items, wid, runtime=runtime_copy,
            )
            finalize = lambda current_job, bridge, wid: finalize_daily_scan_job(
                current_job, bridge, wid, runtime=runtime_copy,
            )
        run_durable_job_loop(
            bridge_factory=ScannerDatabaseBridge,
            job_id=job_id,
            worker_id=worker_id,
            process_chunk=process,
            finalize_job=finalize,
        )

    return start_worker(job_id, runner)


def _artifact_frame(artifacts: pd.DataFrame, artifact_type: str) -> pd.DataFrame:
    if artifacts is None or artifacts.empty or "artifact_type" not in artifacts.columns:
        return pd.DataFrame()
    subset = artifacts.loc[artifacts["artifact_type"].astype(str).eq(artifact_type)]
    if subset.empty:
        return pd.DataFrame()
    payload = subset.sort_values("chunk_number").iloc[-1].get("payload")
    return frame_from_records(payload)


def _job_result_from_artifacts(job: Mapping[str, object], artifacts: pd.DataFrame) -> dict[str, object]:
    artifact_types = set(artifacts.get("artifact_type", pd.Series(dtype=str)).astype(str).tolist()) if isinstance(artifacts, pd.DataFrame) else set()
    final_available = "FINAL_NEXT_LEADERS" in artifact_types or "FINAL_SWING_READY" in artifact_types
    leaders = _artifact_frame(artifacts, "FINAL_NEXT_LEADERS")
    swings = _artifact_frame(artifacts, "FINAL_SWING_READY")
    if not final_available:
        leaders = _artifact_frame(artifacts, "PROVISIONAL_NEXT_LEADERS")
        swings = _artifact_frame(artifacts, "PROVISIONAL_SWING_READY")
    leaders_all = _artifact_frame(artifacts, "FINAL_NEXT_LEADERS_ALL")
    swings_all = _artifact_frame(artifacts, "FINAL_SWING_READY_ALL")
    evidence_detail = _artifact_frame(artifacts, "FINAL_EVIDENCE_DETAIL")
    scoring_contract = _artifact_frame(artifacts, "FINAL_SCORING_CONTRACT")
    macro_snapshot = _artifact_frame(artifacts, "FINAL_MACRO_SNAPSHOT")
    macro_sector = _artifact_frame(artifacts, "FINAL_MACRO_SECTOR_MAP")
    portfolio = _artifact_frame(artifacts, "FINAL_PORTFOLIO")
    coverage = _artifact_frame(artifacts, "FINAL_COVERAGE")
    ohlcv_audit = _artifact_frame(artifacts, "FINAL_OHLCV_AUDIT")
    audit = _artifact_frame(artifacts, "FINAL_JOB_AUDIT")
    if audit.empty:
        audit = _artifact_frame(artifacts, "PROVISIONAL_JOB_AUDIT")
    completed = int(job.get("completed_items", 0) or 0)
    universe = list(job.get("universe_payload") or [])
    completed_tickers: list[str] = []
    if not audit.empty:
        row = audit.iloc[0]
        raw_completed = row.get("ohlcv_ready_ticker_list") or row.get("completed_ticker_list")
        if isinstance(raw_completed, list):
            completed_tickers = [str(value) for value in raw_completed if str(value).strip()]
    if not completed_tickers:
        completed_tickers = [str(value) for value in universe[:completed]]
    elapsed = 0.0
    try:
        started = pd.Timestamp(job.get("started_at"))
        finished = pd.Timestamp(job.get("finished_at"))
        elapsed = max(0.0, float((finished - started).total_seconds()))
    except Exception:
        pass
    return {
        "mode": "resumable_chunked_daily_scan",
        "scanner_version": APP_VERSION,
        "prepared": {str(ticker): pd.DataFrame() for ticker in completed_tickers},
        "focus_screens": {
            "next_leaders": leaders,
            "swing_ready": swings,
            "next_leaders_all": leaders_all,
            "swing_ready_all": swings_all,
            "multibagger": leaders,
            "core_swing": swings,
            "production_evidence_detail": evidence_detail,
            "production_scoring_audit": scoring_contract,
        },
        "macro_snapshot": macro_snapshot,
        "macro_sector_map": macro_sector,
        "portfolio_analysis": portfolio,
        "database_coverage_after": coverage,
        "scan_coverage_summary": audit,
        "ohlcv_database_audit": ohlcv_audit,
        "scan_elapsed_seconds": elapsed,
        "job_id": job.get("job_id"),
        "job_status": job.get("status"),
        "ranking_state": "FINAL" if final_available else "PROVISIONAL",
        "ranking_quality_state": (
            str(audit.iloc[0].get("ranking_state") or "UNKNOWN").upper()
            if isinstance(audit, pd.DataFrame) and not audit.empty else "UNKNOWN"
        ),
    }


def _render_resumable_job(job: Mapping[str, object], bridge: ScannerDatabaseBridge) -> None:
    total = max(0, int(job.get("total_items", 0) or 0))
    completed = max(0, int(job.get("completed_items", 0) or 0))
    failed = max(0, int(job.get("failed_items", 0) or 0))
    retries = max(0, int(job.get("retry_items", 0) or 0))
    done = min(total, completed + failed)
    item_progress_pct = 100.0 * done / total if total else 0.0
    status = str(job.get("status", "UNKNOWN")).upper()
    # Item processing may be 100% while ranking/execution finalisation is still
    # running. Reserve the last 5% for durable ranking artifacts.
    progress_pct = min(item_progress_pct, 95.0) if status == "FINALIZING" else item_progress_pct

    terminal_items = pd.DataFrame()
    partial_count = 0
    if status in {"FINALIZING", "COMPLETE", "COMPLETE_WITH_FAILURES", "FAILED", "CANCELLED"}:
        try:
            terminal_items = bridge.read_scan_job_items(
                str(job.get("job_id")), include_payload=True, limit=5000,
            )
            if not terminal_items.empty and "result_payload" in terminal_items.columns:
                completion_states = terminal_items["result_payload"].map(
                    lambda value: str(value.get("completion_state", ""))
                    if isinstance(value, Mapping) else ""
                )
                partial_count = int(completion_states.isin({
                    "PARTIAL_SNAPSHOT", "PARTIAL_HISTORY", "MARKET_STATUS_ONLY", "NO_PUBLIC_EVIDENCE",
                    "TECHNICAL_UNAVAILABLE",
                }).sum())
        except Exception:
            terminal_items = pd.DataFrame()

    cols = st.columns(6)
    cols[0].metric("Status job", status)
    cols[1].metric("Selesai diproses", f"{completed}/{total}")
    cols[2].metric("Data parsial", partial_count)
    cols[3].metric("Gagal sistem", failed)
    cols[4].metric("Menunggu retry", retries)
    cols[5].metric("Progress", f"{progress_pct:.1f}%")
    st.progress(min(100, int(progress_pct)), text=f"{job.get('phase', '')} • chunk {job.get('chunk_size', 0)} ticker")
    if status == "FINALIZING":
        st.info(
            f"Pemrosesan ticker terminal {done}/{total}. Ranking sedang dibentuk dan dipersistenkan; "
            "angka 100% ticker tidak lagi ditampilkan sebagai 100% job sebelum artifact ranking siap."
        )
    worker = worker_status(str(job.get("job_id", "")))
    if worker is not None and worker.alive:
        st.caption(f"Worker server aktif: {worker.worker_id}. Browser boleh terputus; checkpoint tersimpan di database. Jika host tidur/recycle, job pause lalu session berikutnya melanjutkan setelah lease kedaluwarsa.")
    else:
        st.caption("Tidak ada worker lokal aktif. Job tetap tersimpan dan dapat diklaim kembali oleh session berikutnya.")
    if job.get("last_error"):
        st.warning(str(job.get("last_error")))
    with st.expander("Status ticker / retry", expanded=status in {"PAUSED", "COMPLETE_WITH_FAILURES", "FAILED"}):
        try:
            items = terminal_items if not terminal_items.empty else bridge.read_scan_job_items(
                str(job.get("job_id")), include_payload=True, limit=5000,
            )
            if not items.empty and "result_payload" in items.columns:
                local = items.copy()
                local["completion_state"] = local["result_payload"].map(
                    lambda value: value.get("completion_state") if isinstance(value, Mapping) else None
                )
                local["snapshot_ready"] = local["result_payload"].map(
                    lambda value: value.get("snapshot_ready") if isinstance(value, Mapping) else None
                )
                local["history_periods"] = local["result_payload"].map(
                    lambda value: value.get("history_periods") if isinstance(value, Mapping) else None
                )
                local["database_sync_state"] = local["result_payload"].map(
                    lambda value: value.get("database_sync_state") if isinstance(value, Mapping) else None
                )
                local = local.drop(columns=["result_payload"], errors="ignore")
                items = local
        except Exception as exc:
            items = pd.DataFrame([{"status": "READ_FAIL", "last_error": str(exc)}])
        st.dataframe(_safe_display(items), width="stretch", hide_index=True)


st.title("IDX Super Scanner v9 — Macro-First")
st.caption(
    f"{APP_VERSION} • database-first {DATABASE_FIRST_VERSION} • macro {MACRO_ENGINE_VERSION} • decision {SIMPLE_FOCUS_VERSION}"
)
st.markdown(
    """
    <div class="v9-note">
      <b>Resumable chunked scan</b><br>
      Setiap ticker memiliki status, retry counter, lease, dan payload hasil di database. Browser boleh reconnect dan session baru membaca checkpoint yang sama.
      Job melanjutkan chunk berikutnya tanpa mengulang ticker COMPLETE; ticker gagal dicoba terbatas lalu dilewati. Output tetap <b>The Next Leader</b> dan <b>Swing Ready</b>.
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Input")
    ticker_file = st.file_uploader("Universe ticker CSV", type=["csv"])
    portfolio_file = st.file_uploader("Portfolio CSV (opsional)", type=["csv"])
    operation_mode = st.radio("Mode kerja", ["Isi Database", "Scan Harian"], horizontal=True)
    account_size = float(st.number_input("Nilai akun (Rp)", min_value=0, value=5_000_000, step=500_000))
    cash_on_hand = float(st.number_input("Cash tersedia (Rp)", min_value=0, value=1_000_000, step=100_000))
    with st.expander("Pengaturan resumable", expanded=True):
        if operation_mode == "Isi Database":
            job_chunk_size = st.slider("Ticker per chunk", 5, 40, 15, 5)
        else:
            job_chunk_size = st.slider("Ticker per chunk", 5, 40, 20, 5)
        max_attempts = st.slider("Maksimum percobaan per ticker", 1, 3, 2, 1)
        st.caption("Setiap chunk disimpan sebelum chunk berikutnya. Ticker gagal tidak menghapus hasil ticker lain.")
    with st.expander("Pengaturan lanjutan"):
        period = st.selectbox("Riwayat OHLCV", ["3y", "5y", "10y"], index=1)
        risk_per_trade_pct = st.slider("Risiko per transaksi", 0.25, 2.00, 0.75, 0.25) / 100.0
        official_idx_refresh = st.checkbox("Verifikasi IDX live saat backfill (lebih lambat)", value=False, disabled=operation_mode != "Isi Database")
        snapshot_fallback_limit = st.slider("Snapshot fallback maksimum per chunk", 0, 8, 4, 2, disabled=operation_mode != "Isi Database")
        twelve_fallback_limit = st.slider("Twelve Data fallback per chunk", 0, 8, 4, 2, disabled=operation_mode != "Isi Database")
        run_oos = st.checkbox("Chronological OOS", value=False, disabled=True, help="OOS penuh ditunda; resumable v9.4 memprioritaskan scan selesai dan durable.")
        allow_partial_database = st.checkbox("Izinkan scan saat database belum siap", value=True, disabled=operation_mode == "Isi Database")
        itick_token = st.text_input("iTick API token", value=os.getenv("ITICK_API_TOKEN", ""), type="password")
        twelve_token = st.text_input("Twelve Data API key", value=os.getenv("TWELVE_DATA_API_KEY", ""), type="password")
    button_label = "Mulai / Lanjutkan Isi Database" if operation_mode == "Isi Database" else "Mulai / Lanjutkan Scan Harian"
    run_scan = st.button(button_label, type="primary", width="stretch")
    refresh_status = st.button("Segarkan status job", width="stretch")


# -----------------------------------------------------------------------------
# Durable job control. The browser only starts or observes a server/database job.
# -----------------------------------------------------------------------------
bridge = ScannerDatabaseBridge()
job_type = "DATABASE_BACKFILL" if operation_mode == "Isi Database" else "DAILY_SCAN"
runtime = _runtime_tokens(itick_token, twelve_token)
active_job: dict[str, object] = {}
job_error = ""
try:
    requested_job_id = str(st.session_state.get("v94_job_id", "") or "")
    if requested_job_id:
        candidate_job = bridge.read_scan_job(requested_job_id)
        if str(candidate_job.get("job_type", "")).upper() == job_type:
            active_job = candidate_job
    if not active_job:
        active_job = bridge.read_latest_scan_job(
            job_type=job_type,
            statuses=["PENDING", "RUNNING", "PAUSED", "FINALIZING", "COMPLETE", "COMPLETE_WITH_FAILURES"],
        )
except Exception as exc:
    job_error = f"{type(exc).__name__}: {str(exc)[:500]}"

if run_scan:
    portfolio = pd.DataFrame()
    if portfolio_file is not None:
        try:
            portfolio = parse_portfolio_csv(portfolio_file)
        except Exception as exc:
            st.error(f"Portfolio CSV tidak valid: {exc}")
            st.stop()

    if ticker_file is not None:
        try:
            tickers = parse_ticker_csv(ticker_file, max_tickers=400, strict_limit=True)
        except Exception as exc:
            st.error(f"CSV ticker tidak valid: {exc}")
            st.stop()
        portfolio_tickers = portfolio["ticker"].dropna().astype(str).drop_duplicates().tolist() if not portfolio.empty and "ticker" in portfolio else []
        all_tickers = list(dict.fromkeys(portfolio_tickers + tickers))[:400]
        safe_config = {
            "period": period,
            "account_size_idr": account_size,
            "cash_on_hand_idr": cash_on_hand,
            "risk_per_trade_pct": risk_per_trade_pct,
            "official_idx_refresh": bool(official_idx_refresh),
            "snapshot_fallback_limit": int(snapshot_fallback_limit),
            "twelve_fallback_limit": int(twelve_fallback_limit),
            "allow_partial_database": bool(allow_partial_database),
            "portfolio_records": portfolio.to_dict("records") if not portfolio.empty else [],
            "provider_batch_size": int(job_chunk_size),
        }
        try:
            active_job = bridge.create_or_resume_scan_job(
                job_type=job_type,
                tickers=all_tickers,
                config_payload=safe_config,
                phase="BACKFILL_CORE" if job_type == "DATABASE_BACKFILL" else "TECHNICAL",
                chunk_size=int(job_chunk_size),
                max_attempts=int(max_attempts),
                model_version=APP_VERSION,
            )
            st.session_state["v94_job_id"] = str(active_job.get("job_id", ""))
        except Exception as exc:
            st.error(
                "Job resumable tidak dapat dibuat. Jalankan database/migration_v12_resumable_scan_jobs.sql, "
                "lalu database/permissions_hotfix_v9_4_1.sql untuk grant service_role. "
                f"Pastikan Supabase memakai secret/service-role key. Detail: {type(exc).__name__}: {str(exc)[:500]}"
            )
            st.stop()
    elif not active_job or str(active_job.get("job_type", "")).upper() != job_type:
        st.error("Upload CSV untuk membuat job baru. Untuk melanjutkan job lama, pilih mode yang sama lalu tekan tombol lagi.")
        st.stop()

    _start_resumable_worker(active_job, runtime)
    active_job = bridge.read_scan_job(str(active_job.get("job_id")))

if job_error and not active_job:
    st.error(
        "Repository job resumable belum dapat dibaca. Bila tabel sudah dibuat tetapi muncul HTTP 403/42501, "
        "jalankan database/permissions_hotfix_v9_4_1.sql. "
        f"Detail: {job_error}"
    )

if active_job and str(active_job.get("job_type", "")).upper() == job_type:
    status = str(active_job.get("status", "")).upper()
    pause_resume_key = f"v94_auto_resumed_{active_job.get('job_id', '')}"
    auto_resume_paused = status == "PAUSED" and not bool(st.session_state.get(pause_resume_key, False))
    if auto_resume_paused:
        st.session_state[pause_resume_key] = True
    should_start_worker = status in {"PENDING", "RUNNING", "FINALIZING"} or status == "PAUSED" and (run_scan or auto_resume_paused)
    if should_start_worker:
        _start_resumable_worker(active_job, runtime)
        try:
            active_job = bridge.read_scan_job(str(active_job.get("job_id")))
        except Exception:
            pass
    _render_resumable_job(active_job, bridge)

    status = str(active_job.get("status", "")).upper()
    if status in {"COMPLETE_WITH_FAILURES", "FAILED"}:
        try:
            failed_rows = bridge.read_scan_job_items(
                str(active_job.get("job_id")), include_payload=False, limit=5000,
            )
            failed_rows = failed_rows.loc[
                failed_rows.get("status", pd.Series(dtype=str)).astype(str).str.upper().eq("FAILED")
            ] if not failed_rows.empty else pd.DataFrame()
        except Exception:
            failed_rows = pd.DataFrame()
        if not failed_rows.empty:
            st.error(
                "Job lama selesai dengan ticker gagal. v9.6.0 mengulang ticker tersebut di job yang sama, "
                "sehingga checkpoint ticker COMPLETE dan basis ranking sebelumnya tetap digabungkan."
            )
            if st.button("Ulangi hanya ticker gagal dalam job yang sama", type="primary", key=f"retry_failed_{active_job.get('job_id')}"):
                requeued = bridge.requeue_failed_scan_job_items(str(active_job.get("job_id")))
                st.session_state.pop("v9_scan_result", None)
                refreshed_job = bridge.read_scan_job(str(active_job.get("job_id")))
                _start_resumable_worker(refreshed_job, runtime)
                st.success(f"{requeued} ticker dikembalikan ke antrean tanpa menghapus hasil ticker yang sudah COMPLETE.")
                st.rerun()

    if status in {"COMPLETE", "COMPLETE_WITH_FAILURES"}:
        artifacts = bridge.read_scan_job_artifacts(str(active_job.get("job_id")))
        if job_type == "DATABASE_BACKFILL":
            summary = _artifact_frame(artifacts, "BACKFILL_SUMMARY")
            coverage = _artifact_frame(artifacts, "BACKFILL_COVERAGE")
            if not summary.empty:
                row = summary.iloc[0]
                cols = st.columns(4)
                cols[0].metric("Snapshot ready", f"{_finite(row.get('fundamental_snapshot_pct')):.1f}%")
                cols[1].metric("History ≥2", f"{_finite(row.get('fundamental_history_pct')):.1f}%")
                cols[2].metric("Market status", f"{_finite(row.get('market_status_pct')):.1f}%")
                cols[3].metric("Database state", str(row.get("database_state", "UNKNOWN")))
            with st.expander("Coverage final", expanded=False):
                st.dataframe(_safe_display(coverage), width="stretch", hide_index=True)
            st.info("Isi Database telah selesai secara durable. Pindah ke mode Scan Harian untuk membuat job scan saham.")
            st.stop()
        st.session_state["v9_scan_result"] = _job_result_from_artifacts(active_job, artifacts)
    elif status in {"PENDING", "RUNNING", "FINALIZING"}:
        provisional_loaded = False
        if status == "FINALIZING" and job_type == "DAILY_SCAN":
            try:
                artifacts = bridge.read_scan_job_artifacts(str(active_job.get("job_id")))
                provisional = _job_result_from_artifacts(active_job, artifacts)
                artifact_types = set(artifacts.get("artifact_type", pd.Series(dtype=str)).astype(str).tolist()) if isinstance(artifacts, pd.DataFrame) else set()
                has_ranking_artifact = "PROVISIONAL_NEXT_LEADERS" in artifact_types or "PROVISIONAL_SWING_READY" in artifact_types
                if has_ranking_artifact:
                    st.session_state["v9_scan_result"] = provisional
                    provisional_loaded = True
                    st.warning(
                        "Ranking sementara sudah tersedia dari ticker technical-ready. Finalizer masih memverifikasi harga/entry; "
                        "tekan Muat ulang status untuk mengambil ranking final setelah status COMPLETE."
                    )
                    if st.button("Muat ulang status / ranking final", key=f"refresh_finalizing_{active_job.get('job_id')}"):
                        st.rerun()
            except Exception as exc:
                st.caption(f"Artifact ranking sementara belum dapat dibaca: {type(exc).__name__}")
        if not provisional_loaded:
            st.info("Job berjalan di server. Ponsel/browser boleh ditutup; buka kembali aplikasi untuk membaca checkpoint terbaru.")
            time.sleep(2.0)
            st.rerun()
    elif status == "PAUSED":
        st.warning("Job dipause secara aman. Session baru mencoba auto-resume satu kali; tekan Mulai / Lanjutkan bila provider/database sudah pulih.")
        st.stop()
    elif status in {"FAILED", "CANCELLED"}:
        st.error("Job berhenti terminal. Audit ticker di atas menunjukkan item yang gagal.")
        st.stop()


result = st.session_state.get("v9_scan_result")
if not result:
    st.info("Upload universe ticker lalu jalankan scanner. Maksimum 400 ticker.")
    st.stop()

focus = result.get("focus_screens", {})
leaders = focus.get("next_leaders", pd.DataFrame()).copy()
swings = focus.get("swing_ready", pd.DataFrame()).copy()
macro_snapshot = result.get("macro_snapshot", pd.DataFrame())
elapsed = _finite(result.get("scan_elapsed_seconds"), 0.0)
ranking_state = str(result.get("ranking_state", "FINAL")).upper()
coverage_summary = result.get("scan_coverage_summary", pd.DataFrame())
if ranking_state == "PROVISIONAL":
    basis = ""
    if isinstance(coverage_summary, pd.DataFrame) and not coverage_summary.empty:
        row = coverage_summary.iloc[0]
        requested = int(_finite(row.get("requested_tickers"), 0))
        ready = int(_finite(row.get("technical_ready_tickers", row.get("completed_tickers")), 0))
        unavailable = int(_finite(row.get("technical_unavailable_tickers"), 0))
        basis = f" Basis saat ini {ready}/{requested} ticker technical-ready; {unavailable} belum memiliki OHLCV valid."
    st.warning(
        "Ranking yang tampil masih PROVISIONAL karena job sedang FINALIZING. "
        "Entry/SL/TP dan execution readiness dapat berubah setelah verifikasi harga selesai." + basis
    )

coverage_row = coverage_summary.iloc[0] if isinstance(coverage_summary, pd.DataFrame) and not coverage_summary.empty else pd.Series(dtype=object)
requested_metric = int(_finite(coverage_row.get("requested_tickers"), len(result.get("prepared", {}))))
ohlcv_ready_metric = int(_finite(coverage_row.get("ohlcv_ready_tickers"), len(result.get("prepared", {}))))
leader_valid_metric = int(_finite(coverage_row.get("next_leader_final_score_valid"), len(leaders)))
swing_valid_metric = int(_finite(coverage_row.get("swing_final_score_valid"), len(swings)))
metrics = st.columns(6)
metrics[0].metric("Universe", requested_metric)
metrics[1].metric("OHLCV ready", f"{ohlcv_ready_metric}/{requested_metric}" if requested_metric else ohlcv_ready_metric)
metrics[2].metric("Leader scored", leader_valid_metric)
metrics[3].metric("Swing scored", swing_valid_metric)
metrics[4].metric("Macro regime", str(macro_snapshot.iloc[0].get("macro_regime", "DATA_PENDING")) if not macro_snapshot.empty else "DATA_PENDING")
metrics[5].metric("Waktu", f"{elapsed:.1f} dtk")
ranking_quality_state = str(result.get("ranking_quality_state", coverage_row.get("ranking_state", "UNKNOWN"))).upper()
if ranking_quality_state not in {"VALID", "UNKNOWN"}:
    st.error(
        f"Ranking quality: {ranking_quality_state}. Hasil hanya mencakup {ohlcv_ready_metric}/{requested_metric} ticker OHLCV-ready dan belum layak menjadi dasar order penuh."
    )
if isinstance(coverage_summary, pd.DataFrame) and not coverage_summary.empty:
    requested_count = int(_finite(coverage_row.get("requested_tickers"), 0))
    fundamental_count = int(_finite(coverage_row.get("fundamental_ready"), 0))
    if requested_count > 0 and fundamental_count < requested_count:
        st.warning(
            f"Evidence fundamental lengkap tersedia untuk {fundamental_count}/{requested_count} ticker. "
            "Ticker lain tetap dipindai secara teknikal tetapi berstatus DATA_PENDING sampai cache/backfill terisi."
        )

if not leaders.empty:
    leaders["Final Score"] = pd.to_numeric(leaders.get("final_score", leaders.get("v9_next_leader_score")), errors="coerce")
if not swings.empty:
    swings["Final Score"] = pd.to_numeric(swings.get("final_score", swings.get("v9_swing_score")), errors="coerce")

market_tab, leader_tab, swing_tab, portfolio_tab = st.tabs([
    "Market Map", "The Next Leader", "Swing Ready", "Portfolio & Audit",
])

with market_tab:
    if not macro_snapshot.empty:
        row = macro_snapshot.iloc[0]
        cols = st.columns(4)
        cols[0].metric("Macro score", f"{_finite(row.get('macro_regime_score')):.1f}")
        cols[1].metric("Coverage", f"{_finite(row.get('macro_data_coverage_pct')):.0f}%")
        cols[2].metric("Breadth > EMA50", f"{_finite(row.get('breadth_above_ema50_pct')):.1f}%")
        cols[3].metric("IHSG 20D", f"{100.0 * _finite(row.get('ihsg_return_20d')):.2f}%")
        factor_cols = [column for column in macro_snapshot.columns if column.startswith("factor_")]
        factor_view = pd.DataFrame({
            "factor": [column.replace("factor_", "").replace("_score", "") for column in factor_cols],
            "score": [row.get(column) for column in factor_cols],
        })
        st.subheader("Macro factors")
        st.dataframe(_safe_display(factor_view), width="stretch", hide_index=True)
    st.subheader("Sector opportunity map")
    st.dataframe(_safe_display(result.get("macro_sector_map", pd.DataFrame())), width="stretch", hide_index=True)
    ihsg = result.get("ihsg_direction") or {}
    if isinstance(ihsg, Mapping):
        st.caption(
            f"IHSG model: {ihsg.get('consensus_direction', 'NO_EDGE')} • "
            f"confidence {_finite(ihsg.get('consensus_confidence')):.1f}% • "
            f"risk multiplier {_finite(ihsg.get('risk_budget_multiplier'), 0.5):.2f}x"
        )
    with st.expander("Macro source audit"):
        st.dataframe(_safe_display(result.get("macro_source_report", pd.DataFrame())), width="stretch", hide_index=True)

with leader_tab:
    columns = [
        "rank", "ticker", "sector", "candidate_type", "status",
        "Final Score", "score_coverage_pct",
        "fundamental_freshness_state", "sector_source", "sector_confidence_pct",
        "business_quality_score", "future_fundamental_score", "valuation_mos_score",
        "management_capital_score", "issuer_macro_alignment_score",
        "narrative_flow_score", "technical_readiness_score",
        "silent_accumulation_score", "retail_adoption_stage",
        "entry_low", "entry_high", "trigger", "stop_loss", "tp1", "tp2", "rr1",
        "recommended_allocation_idr", "recommended_lots", "selected_reason", "primary_risk",
    ]
    view = _safe_display(leaders.head(50), columns)
    if view.empty:
        st.warning("Belum ada kandidat The Next Leader.")
    else:
        st.dataframe(view, width="stretch", hide_index=True)
        st.download_button("Download The Next Leader CSV", view.to_csv(index=False).encode("utf-8-sig"), "v9_the_next_leader.csv", "text/csv")

with swing_tab:
    columns = [
        "rank", "ticker", "sector", "status", "Final Score", "score_coverage_pct",
        "production_gate_reason", "technical_execution_score", "issuer_macro_alignment_score", "narrative_flow_score",
        "business_quality_score", "risk_data_score", "next_leader_score", "strategy",
        "entry_low", "entry_high", "trigger_price", "stop_loss", "tp1", "tp2", "rr1", "rr2",
        "stockbit_order_lots", "next_action", "selected_reason", "primary_risk",
    ]
    view = _safe_display(swings.head(50), columns)
    if view.empty:
        st.warning("Belum ada setup Swing Ready.")
    else:
        st.dataframe(view, width="stretch", hide_index=True)
        st.download_button("Download Swing Ready CSV", view.to_csv(index=False).encode("utf-8-sig"), "v9_swing_ready.csv", "text/csv")

with portfolio_tab:
    portfolio_analysis = result.get("portfolio_analysis", pd.DataFrame())
    portfolio_summary = result.get("portfolio_summary", {})
    if isinstance(portfolio_summary, Mapping) and portfolio_summary:
        cols = st.columns(4)
        cols[0].metric("Nilai pasar", f"Rp {_finite(portfolio_summary.get('market_value_idr')):,.0f}")
        cols[1].metric("Unrealized P/L", f"Rp {_finite(portfolio_summary.get('unrealized_pnl_idr')):,.0f}")
        cols[2].metric("Open risk", f"Rp {_finite(portfolio_summary.get('open_risk_idr')):,.0f}")
        cols[3].metric("Estimasi equity", f"Rp {_finite(portfolio_summary.get('estimated_equity_idr')):,.0f}")
    if isinstance(portfolio_analysis, pd.DataFrame) and not portfolio_analysis.empty:
        st.dataframe(_safe_display(portfolio_analysis), width="stretch", hide_index=True)
    else:
        st.info("Portfolio CSV belum diunggah.")

    with st.expander("Ticker belum masuk ranking"):
        leader_all = focus.get("next_leaders_all", pd.DataFrame())
        swing_all = focus.get("swing_ready_all", pd.DataFrame())
        pending_frames = []
        if isinstance(leader_all, pd.DataFrame) and not leader_all.empty:
            leader_mask = leader_all["rank_eligible"].fillna(False).astype(bool) if "rank_eligible" in leader_all.columns else pd.Series(False, index=leader_all.index)
            local = leader_all.loc[~leader_mask].copy()
            if not local.empty:
                local["model"] = "THE_NEXT_LEADER"
                pending_frames.append(local)
        if isinstance(swing_all, pd.DataFrame) and not swing_all.empty:
            swing_mask = swing_all["rank_eligible"].fillna(False).astype(bool) if "rank_eligible" in swing_all.columns else pd.Series(False, index=swing_all.index)
            local = swing_all.loc[~swing_mask].copy()
            if not local.empty:
                local["model"] = "SWING_READY"
                pending_frames.append(local)
        pending = pd.concat(pending_frames, ignore_index=True, sort=False) if pending_frames else pd.DataFrame()
        st.dataframe(_safe_display(pending), width="stretch", hide_index=True)
    with st.expander("Scoring contract"):
        st.dataframe(_safe_display(focus.get("production_scoring_audit", pd.DataFrame())), width="stretch", hide_index=True)
    with st.expander("Evidence per ticker"):
        evidence = _safe_display(focus.get("production_evidence_detail", pd.DataFrame()))
        st.dataframe(evidence, width="stretch", hide_index=True)
        if not evidence.empty:
            st.download_button("Download Evidence CSV", evidence.to_csv(index=False).encode("utf-8-sig"), "v9_evidence.csv", "text/csv")
    with st.expander("Data coverage dan pipeline"):
        st.dataframe(_safe_display(result.get("scan_coverage_summary", pd.DataFrame())), width="stretch", hide_index=True)
        st.dataframe(_safe_display(result.get("scanner_data_contract_audit", pd.DataFrame())), width="stretch", hide_index=True)
        st.dataframe(_safe_display(result.get("stage_timings", pd.DataFrame())), width="stretch", hide_index=True)
        st.dataframe(_safe_display(result.get("two_stage_coverage_audit", pd.DataFrame())), width="stretch", hide_index=True)
    with st.expander("OHLCV database dan provider audit"):
        ohlcv_audit = _safe_display(result.get("ohlcv_database_audit", pd.DataFrame()))
        st.dataframe(ohlcv_audit, width="stretch", hide_index=True)
        if not ohlcv_audit.empty:
            st.download_button("Download OHLCV Audit CSV", ohlcv_audit.to_csv(index=False).encode("utf-8-sig"), "v9_ohlcv_audit.csv", "text/csv")
    with st.expander("Fundamental dan database audit"):
        st.dataframe(_safe_display(result.get("fundamental_history_report", pd.DataFrame())), width="stretch", hide_index=True)
        st.dataframe(_safe_display(result.get("database_sync_report", pd.DataFrame())), width="stretch", hide_index=True)
