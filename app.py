from __future__ import annotations

import hashlib
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Mapping

# yfinance 0.2.66 uses a generic pandas Timedelta internally. NumPy 2.5 emits
# a deprecation warning for that third-party implementation. The scanner pins
# both dependencies and captures provider failures in DownloadReport, so only
# this exact external warning and yfinance console logger are silenced.
warnings.filterwarnings(
    "ignore",
    message=r"The 'generic' unit for NumPy timedelta is deprecated.*",
    category=DeprecationWarning,
    module=r"yfinance(?:\..*)?",
)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="IDX Wealth & IHSG Direction Scanner v7.16.3", page_icon="🛡️", layout="wide")

# Streamlit runs the selected entrypoint from its deployment workspace. Keep
# the application directory explicit on sys.path and validate that the whole
# core module was uploaded, so a partial GitHub upload produces a useful UI
# message instead of a ModuleNotFoundError traceback.
APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

REQUIRED_SCANNER_FILES = (
    "scanner.py", "scanner_focus.py", "scanner_database.py", "research_maintenance.py",
    "ai_engine.py", "selector_engine.py", "time_cycle.py", "eoff_reconstruction.py", "dashboard_v660.py",
    "ihsg_direction.py", "narrative_engine.py", "incremental_store.py",
)
missing_source_files = [name for name in REQUIRED_SCANNER_FILES if not (APP_ROOT / name).is_file()]
if missing_source_files:
    st.error("Deployment tidak lengkap: modul scanner belum lengkap di root repository.")
    st.code(
        "Root repository harus berisi:\n"
        "app.py\nscanner.py\nscanner_focus.py\nscanner_database.py\nresearch_maintenance.py\n"
        "ihsg_direction.py\nai_engine.py\nselector_engine.py\ntime_cycle.py\neoff_reconstruction.py\n"
        "narrative_engine.py\nincremental_store.py\ndashboard_v660.py\nrequirements.txt",
        language="text",
    )
    st.write("File yang belum ditemukan:", ", ".join(missing_source_files))
    st.info("Ekstrak ZIP revisi, lalu upload seluruh isinya—bukan hanya app.py dan requirements.txt—ke branch yang dideploy.")
    st.stop()

from scanner import (
    ScanConfig,
    ScanEngine,
    apply_fundamental_gate,
    apply_market_status_gate,
    apply_news_gate,
    apply_validation_gate,
    apply_execution_snapshot_gate,
    apply_independent_price_gate,
    apply_universe_integrity_gate,
    attach_backtest_stats,
    attach_broker_summary,
    parse_broker_summary_csv,
    attach_fundamentals,
    attach_position_sizing,
    enforce_portfolio_execution_budget,
    apply_analyst_fusion_gate,
    enforce_analyst_portfolio_budget,
    finalize_execution_integrity,
    download_benchmark,
    download_ohlcv,
    fetch_resilient_market_status,
    fetch_resilient_news_review,
    fetch_resilient_fundamentals,
    select_yahoo_fundamental_tickers,
    fetch_yahoo_fundamental_history,
    fetch_idx_fundamental_history,
    fetch_twelve_data_fundamental_history,
    parse_fundamental_history_csv,
    combine_fundamental_history,
    enrich_fundamentals_with_history,
    fetch_execution_snapshots,
    fetch_automatic_independent_prices,
    make_signal_chart,
    parse_ticker_csv,
    parse_portfolio_csv,
    analyze_portfolio_positions,
    run_walkforward_validation,
    run_adaptive_walkforward_validation,
    select_walkforward_universe,
    build_independent_price_validation,
    build_source_quorum_audit,
    idx_daily_bar_is_final,
    idx_regular_decision_window,
    normalize_idx_ticker,
    safe_number,
    safe_text,
)
from scanner_focus import (
    parse_project_management_csv, collect_automatic_forward_quality,
    merge_project_management_reviews, build_focus_screens,
    build_multibagger_diagnostic_views,
    build_scanner_data_contract_audit,
)
from free_data_providers import fetch_reference_fx_rates

from time_cycle import (
    TimeCycleConfig, analyze_time_cycle, enrich_core_signals_with_time_cycle,
    make_time_cycle_chart, TIME_CYCLE_VERSION,
    EOFF_VERSION,
)

from ai_engine import (
    LocalAIConfig, load_memory, update_outcome_memory, resolved_memory_events,
    memory_summary, parse_memory_csv, validation_events_to_memory, save_memory, AI_VERSION,
)
from dashboard_v660 import (
    DASHBOARD_VERSION, build_top20_ranking, render_top20_dashboard,
    render_time_cycle_main_tab, resolve_trade_plan, streamlit_safe_frame,
)

from scanner_database import ScannerDatabaseBridge, DATABASE_BRIDGE_VERSION, DATABASE_SCHEMA_VERSION
from research_maintenance import (
    select_round_robin_backfill, update_research_outcomes, research_outcome_summary,
    model_registry_frame, MODEL_VERSIONS,
)
from ihsg_direction import (
    IHSGDirectionConfig,
    IHSG_DIRECTION_VERSION,
    IHSG_TICKER,
    analyze_ihsg_direction,
    make_ihsg_direction_chart,
    update_ihsg_outcomes,
)
from selector_engine import SelectorConfig, update_selector_outcomes
from narrative_engine import (
    NARRATIVE_ENGINE_VERSION,
    parse_narrative_event_csv,
)


st.markdown(
    """
    <style>
      .block-container {padding-top: 1.35rem; padding-bottom: 2.5rem;}
      [data-testid="stMetricValue"] {font-size: 1.55rem;}
      .scanner-note {border:1px solid #2a3345; border-radius:12px; padding:12px 14px; background:#101723;}
      .small-muted {color:#9aa7b8; font-size:.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=1_800, show_spinner=False)
def cached_market_data(
    tickers: tuple[str, ...],
    period: str,
    itick_enabled: bool = False,
    _itick_api_token: str = "",
):
    histories, report = download_ohlcv(
        tickers, period=period,
        itick_api_token=_itick_api_token if itick_enabled else "",
    )
    benchmark = download_benchmark(period=period)
    report.benchmark_ok = not benchmark.empty
    return histories, report, benchmark


@st.cache_data(ttl=1_800, show_spinner=False)
def cached_standalone_ihsg_direction(period: str = "5y"):
    benchmark = download_benchmark(period=period)
    forecast = analyze_ihsg_direction(
        benchmark,
        {},
        config=IHSGDirectionConfig(),
        eod_final=bool(
            benchmark is not None
            and not benchmark.empty
            and str(getattr(benchmark, "attrs", {}).get("bar_state", "")).upper() == "FINAL_EOD"
        ),
    )
    return benchmark, forecast


@st.cache_data(ttl=1_800, show_spinner=False)
def cached_portfolio_market_data(
    tickers: tuple[str, ...],
    period: str,
    itick_enabled: bool = False,
    _itick_api_token: str = "",
):
    """Portfolio-only path: download holdings without requiring a universe CSV."""
    return download_ohlcv(
        tickers, period=period,
        itick_api_token=_itick_api_token if itick_enabled else "",
    )


@st.cache_data(ttl=21_600, show_spinner=False)
def cached_fundamentals(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_resilient_fundamentals(tickers)


@st.cache_data(ttl=21_600, show_spinner=False)
def cached_reference_fx() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Official-first JISDOR; the provider returns explicit fallback state."""
    return fetch_reference_fx_rates()


@st.cache_data(ttl=21_600, show_spinner=False)
def cached_yahoo_fundamental_history(
    tickers: tuple[str, ...], max_tickers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return fetch_yahoo_fundamental_history(tickers, max_workers=4, max_tickers=max_tickers)


@st.cache_data(ttl=86_400, show_spinner=False)
def cached_idx_fundamental_history(
    tickers: tuple[str, ...], max_tickers: int, years_back: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return fetch_idx_fundamental_history(
        tickers, max_tickers=max_tickers, years_back=years_back,
    )


@st.cache_data(ttl=21_600, show_spinner=False)
def cached_twelve_fundamental_history(
    tickers: tuple[str, ...], enabled: bool, max_tickers: int,
    _twelve_data_api_key: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Even when the live API is disabled, the provider function may recover
    # previously validated statements from the durable local cache.
    return fetch_twelve_data_fundamental_history(
        tickers, api_key=_twelve_data_api_key if enabled else "", max_tickers=max_tickers,
    )


@st.cache_data(ttl=900, show_spinner=False)
def cached_automatic_market_status(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_resilient_market_status(tickers)


@st.cache_data(ttl=900, show_spinner=False)
def cached_automatic_news(tickers: tuple[str, ...], lookback_days: int) -> pd.DataFrame:
    return fetch_resilient_news_review(tickers, lookback_days=lookback_days)


@st.cache_data(ttl=300, show_spinner=False)
def cached_execution_snapshots(tickers: tuple[str, ...]) -> pd.DataFrame:
    return fetch_execution_snapshots(tickers)


def configured_twelve_data_key() -> str:
    """Read an optional deployment secret without requiring per-scan input."""
    key = os.environ.get("TWELVE_DATA_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("TWELVE_DATA_API_KEY", "")).strip()
    except Exception:
        return ""


def configured_itick_token() -> str:
    """Read the optional no-cost iTick fallback token from deployment secrets."""
    token = os.environ.get("ITICK_API_TOKEN", "").strip()
    if token:
        return token
    try:
        return str(st.secrets.get("ITICK_API_TOKEN", "")).strip()
    except Exception:
        return ""


@st.cache_data(ttl=900, show_spinner=False)
def cached_automatic_independent_prices(
    tickers: tuple[str, ...],
    reference_date: str,
    primary_reference: tuple[tuple[str, str, float], ...],
    primary_source_tiers: tuple[tuple[str, str], ...],
    config: ScanConfig,
    _twelve_data_api_key: str = "",
    _itick_api_token: str = "",
):
    return fetch_automatic_independent_prices(
        tickers,
        reference_date=reference_date,
        twelve_data_api_key=_twelve_data_api_key,
        itick_api_token=_itick_api_token,
        primary_reference={ticker: (date, close) for ticker, date, close in primary_reference},
        primary_source_tiers=dict(primary_source_tiers),
        config=config,
    )


def upload_fingerprint(*files: object) -> str:
    digest = hashlib.sha256()
    for uploaded_file in files:
        if uploaded_file is None:
            digest.update(b"<none>")
        else:
            digest.update(getattr(uploaded_file, "name", "upload").encode("utf-8", errors="ignore"))
            digest.update(uploaded_file.getvalue())
        digest.update(b"\x00")
    return digest.hexdigest()


class ScanStageProfiler:
    """Small monotonic stage profiler shown in the in-app audit tab."""

    def __init__(self) -> None:
        self._scan_started = time.perf_counter()
        self._stage_started = self._scan_started
        self._rows: list[dict[str, object]] = []

    def mark(
        self,
        stage: str,
        *,
        workload: int | None = None,
        workload_unit: str = "",
        stage_type: str = "CPU",
    ) -> None:
        now = time.perf_counter()
        self._rows.append({
            "stage": stage,
            "seconds": round(now - self._stage_started, 3),
            "stage_type": stage_type,
            "workload": workload,
            "workload_unit": workload_unit,
        })
        self._stage_started = now

    def frame(self) -> pd.DataFrame:
        out = pd.DataFrame(self._rows)
        if out.empty:
            return out
        out["share_pct"] = (
            100.0 * out["seconds"] / max(float(out["seconds"].sum()), 1e-9)
        ).round(1)
        out["total_scan_seconds"] = round(
            time.perf_counter() - self._scan_started,
            3,
        )
        return out.sort_values("seconds", ascending=False, kind="stable").reset_index(drop=True)


def _fundamental_row_usable(row: Mapping[str, object] | dict[str, object]) -> bool:
    coverage = pd.to_numeric(pd.Series([row.get("fundamental_coverage")]), errors="coerce").iloc[0]
    error = str(row.get("fundamental_error") or "").strip()
    return bool(pd.notna(coverage) and float(coverage) >= 45.0 and not error)


def database_first_fundamentals(
    tickers: tuple[str, ...],
    config: ScanConfig | None = None,
    priority_tickers: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read complete fundamental rows from Supabase before provider refresh.

    CURRENT rows generate zero external fundamental calls. STALE rows remain a
    fail-soft fallback while only stale/missing symbols are sent to providers.
    """
    names = tuple(dict.fromkeys(tickers))
    if not names:
        return pd.DataFrame(), pd.DataFrame()
    cfg = config or ScanConfig()
    bridge = ScannerDatabaseBridge()
    database_rows, read_audit = bridge.read_fundamental_cache(names)
    due_names = tuple(bridge.refresh_tickers(read_audit, names))
    event_names = tuple(bridge.read_pending_event_tickers(names)) if bool(getattr(cfg, "event_aware_refresh_enabled", True)) else ()
    scheduler_names = tuple(dict.fromkeys(due_names + event_names))
    default_priorities = tuple(dict.fromkeys(priority_tickers or due_names[: min(24, len(due_names))]))
    completion_limit = int(getattr(cfg, "full_completion_max_tickers", 400))
    completion_mode = len(names) <= max(1, completion_limit)
    if completion_mode:
        refresh_names = list(dict.fromkeys(scheduler_names or names))
        selected = set(refresh_names)
        scheduler_audit = pd.DataFrame({
            "ticker": list(names),
            "selected_for_refresh": [ticker in selected for ticker in names],
            "scheduler_state": "FULL_COMPLETION_UP_TO_400_UNIVERSE",
        })
    else:
        refresh_names, scheduler_audit = select_round_robin_backfill(
            scheduler_names,
            read_audit[read_audit.get("ticker", pd.Series(dtype=str)).isin(scheduler_names)].copy()
            if read_audit is not None and not read_audit.empty else read_audit,
            priority_tickers=default_priorities,
            event_tickers=event_names,
            cohort_count=int(getattr(cfg, "fundamental_backfill_cohorts", 7)),
            max_count=int(getattr(cfg, "fundamental_backfill_max_per_scan", 400)),
        )
    if scheduler_audit is not None and not scheduler_audit.empty:
        scheduler_audit = scheduler_audit.copy()
        scheduler_audit["provider"] = "ROUND_ROBIN_BACKFILL"
        scheduler_audit["scope"] = "FUNDAMENTAL_SNAPSHOT"
        scheduler_audit["status"] = np.where(scheduler_audit["selected_for_refresh"], "SELECTED", "DEFERRED")
        scheduler_audit["rows"] = 0
        scheduler_audit["error"] = ""
        scheduler_audit["source_family"] = "SCHEDULER"
    provider_rows = cached_fundamentals(tuple(refresh_names)) if refresh_names else pd.DataFrame()

    database_map = {
        str(row["ticker"]): row.to_dict()
        for _, row in database_rows.dropna(subset=["ticker"]).drop_duplicates("ticker", keep="last").iterrows()
    } if database_rows is not None and not database_rows.empty and "ticker" in database_rows else {}
    provider_map = {
        str(row["ticker"]): row.to_dict()
        for _, row in provider_rows.dropna(subset=["ticker"]).drop_duplicates("ticker", keep="last").iterrows()
    } if provider_rows is not None and not provider_rows.empty and "ticker" in provider_rows else {}

    resolved: list[dict[str, object]] = []
    provider_audit: list[dict[str, object]] = []
    now_iso = pd.Timestamp.now(tz="Asia/Jakarta").isoformat()
    for ticker in names:
        live = provider_map.get(ticker)
        cached = database_map.get(ticker)
        if live and _fundamental_row_usable(live):
            row = dict(live)
            row["database_source_state"] = "PROVIDER_REFRESHED"
            row["database_source_checked_at"] = row.get("fundamental_fetched_at") or now_iso
            row.setdefault("fundamental_fetched_at", now_iso)
            provider_audit.append({
                "ticker": ticker, "provider": "FUNDAMENTAL_PROVIDER_REFRESH", "scope": "FUNDAMENTAL_SNAPSHOT",
                "status": "PROVIDER_REFRESHED", "database_read_state": "PROVIDER_REFRESHED",
                "age_days": 0.0, "refresh_required": False, "rows": 1, "asof": row["database_source_checked_at"],
                "error": "", "source_family": str(row.get("fundamental_provider") or "PROVIDER"),
            })
        elif cached:
            row = dict(cached)
            state = str(row.get("database_source_state") or "DATABASE_STALE_USABLE")
            if live and not _fundamental_row_usable(live):
                row["fundamental_provider_refresh_error"] = str(live.get("fundamental_error") or live.get("fundamental_error_code") or "Provider refresh gagal")
                row["database_source_state"] = "DATABASE_STALE_FALLBACK"
                provider_audit.append({
                    "ticker": ticker, "provider": "FUNDAMENTAL_PROVIDER_REFRESH", "scope": "FUNDAMENTAL_SNAPSHOT",
                    "status": "CACHE_FALLBACK", "database_read_state": "DATABASE_STALE_FALLBACK",
                    "age_days": np.nan, "refresh_required": True, "rows": 1,
                    "asof": row.get("database_source_checked_at", ""),
                    "error": row["fundamental_provider_refresh_error"], "source_family": "DATABASE",
                })
            else:
                row["database_source_state"] = state
        elif live:
            row = dict(live)
            row["database_source_state"] = "PROVIDER_UNRESOLVED"
            row["database_source_checked_at"] = now_iso
        else:
            row = {
                "ticker": ticker, "fundamental_score": np.nan, "fundamental_coverage": 0.0,
                "fundamental_reliability": "NONE", "fundamental_error": "Database dan provider tidak memiliki data",
                "database_source_state": "DATA_NOT_SCORED", "database_source_checked_at": now_iso,
            }
        row["ticker"] = ticker
        resolved.append(row)

    audit_frames = [frame for frame in (read_audit, scheduler_audit, pd.DataFrame(provider_audit)) if frame is not None and not frame.empty]
    return pd.DataFrame(resolved), pd.concat(audit_frames, ignore_index=True, sort=False) if audit_frames else pd.DataFrame()


def enrich_fundamental_shortlist(
    fundamentals: pd.DataFrame,
    tickers: tuple[str, ...],
    uploaded_history: pd.DataFrame,
    config: ScanConfig,
    twelve_enabled: bool,
    twelve_api_key: str,
    priority_tickers: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Database-first bounded statement history with official refresh.

    Supabase history is read in bulk. Only missing/stale tickers enter the IDX
    official-first provider queue; stale database history remains available as
    a fallback if all free providers fail.
    """
    if not tickers:
        combined = combine_fundamental_history(uploaded_history)
        return enrich_fundamentals_with_history(fundamentals, combined), combined, pd.DataFrame()
    if fundamentals is None or fundamentals.empty:
        fundamentals = pd.DataFrame({"ticker": list(dict.fromkeys(tickers))})

    bridge = ScannerDatabaseBridge()
    database_history, database_audit = bridge.read_fundamental_history_cache(tickers)
    due_names = tuple(bridge.refresh_tickers(database_audit, tickers))
    event_names = tuple(bridge.read_pending_event_tickers(tickers)) if bool(getattr(config, "event_aware_refresh_enabled", True)) else ()
    scheduler_names = tuple(dict.fromkeys(due_names + event_names))
    due_set = set(due_names)
    priority_limit = min(
        int(getattr(
            config, "fundamental_history_priority_top_n", 16,
        )),
        int(getattr(
            config, "fundamental_backfill_max_per_scan",
            config.fundamental_history_top_n,
        )),
    )
    ordered_priorities = tuple(
        ticker for ticker in dict.fromkeys(priority_tickers)
        if ticker in due_set
    )[:max(1, priority_limit)]
    if not ordered_priorities:
        ordered_priorities = due_names[:min(8, len(due_names))]
    completion_mode = len(tickers) <= int(getattr(config, "full_completion_max_tickers", 400))
    if completion_mode:
        refresh_names_list = list(dict.fromkeys(scheduler_names or tickers))
        scheduler_audit = pd.DataFrame({
            'ticker': list(dict.fromkeys(tickers)),
            'selected_for_refresh': [ticker in set(refresh_names_list) for ticker in dict.fromkeys(tickers)],
            'scheduler_state': 'FULL_COMPLETION_UP_TO_400_UNIVERSE',
        })
    else:
        refresh_names_list, scheduler_audit = select_round_robin_backfill(
            scheduler_names,
            database_audit[database_audit.get("ticker", pd.Series(dtype=str)).isin(scheduler_names)].copy()
            if database_audit is not None and not database_audit.empty else database_audit,
            priority_tickers=ordered_priorities,
            event_tickers=event_names,
            cohort_count=int(getattr(config, "fundamental_backfill_cohorts", 7)),
            max_count=int(getattr(config, "fundamental_backfill_max_per_scan", config.fundamental_history_top_n)),
        )
    refresh_names = tuple(refresh_names_list)
    if scheduler_audit is not None and not scheduler_audit.empty:
        scheduler_audit = scheduler_audit.copy()
        scheduler_audit["provider"] = "ROUND_ROBIN_BACKFILL"
        scheduler_audit["scope"] = "FUNDAMENTAL_HISTORY"
        scheduler_audit["status"] = np.where(scheduler_audit["selected_for_refresh"], "SELECTED", "DEFERRED")
        scheduler_audit["rows"] = 0
        scheduler_audit["error"] = ""
        scheduler_audit["source_family"] = "SCHEDULER"

    idx_history, idx_report = cached_idx_fundamental_history(
        refresh_names,
        max_tickers=min(int(getattr(config, "official_fundamental_refresh_max_per_scan", config.idx_fundamental_top_n)), len(refresh_names)),
        years_back=int(config.idx_fundamental_years_back),
    ) if refresh_names else (pd.DataFrame(), pd.DataFrame())

    official_plus_database = combine_fundamental_history(database_history, idx_history)
    yahoo_names = select_yahoo_fundamental_tickers(
        refresh_names, official_plus_database,
        max_tickers=len(refresh_names) if completion_mode else min(int(config.fundamental_history_top_n), len(refresh_names)),
        crosscheck_top_n=int(getattr(config, "fundamental_crosscheck_top_n", 8)),
        min_official_periods=4,
    ) if refresh_names else []
    yahoo_history, yahoo_report = cached_yahoo_fundamental_history(
        tuple(yahoo_names), max_tickers=len(yahoo_names) if completion_mode else min(int(config.fundamental_history_top_n), len(yahoo_names)),
    ) if yahoo_names else (pd.DataFrame(), pd.DataFrame())
    twelve_history, twelve_report = cached_twelve_fundamental_history(
        refresh_names,
        enabled=bool(twelve_enabled and twelve_api_key),
        max_tickers=min(int(config.twelve_fundamental_top_n), len(refresh_names)),
        _twelve_data_api_key=twelve_api_key,
    ) if refresh_names else (pd.DataFrame(), pd.DataFrame())

    provider_checked_at = pd.Timestamp.now(tz="Asia/Jakarta").isoformat()
    for original in (idx_history, yahoo_history, twelve_history):
        if original is not None and not original.empty:
            original["database_source_checked_at"] = provider_checked_at
            original["database_source_state"] = "PROVIDER_REFRESHED"

    combined = combine_fundamental_history(
        uploaded_history, database_history, yahoo_history, idx_history, twelve_history,
    )
    enriched = enrich_fundamentals_with_history(fundamentals, combined)
    reports = [
        frame for frame in (database_audit, scheduler_audit, yahoo_report, idx_report, twelve_report)
        if frame is not None and not frame.empty
    ]
    if uploaded_history is not None and not uploaded_history.empty:
        reports.append(pd.DataFrame([{
            "ticker": "ALL_UPLOAD", "provider": "IDX_REFERENCE_UPLOAD", "scope": "FUNDAMENTAL_HISTORY",
            "status": "OK", "rows": len(uploaded_history), "error": "", "source_family": "UPLOAD",
        }]))
    report = pd.concat(reports, ignore_index=True, sort=False) if reports else pd.DataFrame()
    return enriched, combined, report


def database_first_forward_quality(
    fundamentals: pd.DataFrame,
    tickers: tuple[str, ...],
    config: ScanConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read project/management evidence first and refresh only due symbols."""
    names = tuple(dict.fromkeys(tickers))
    if not names:
        return pd.DataFrame(), pd.DataFrame()
    bridge = ScannerDatabaseBridge()
    database_rows, database_audit = bridge.read_forward_quality_cache(names)
    due_names = tuple(bridge.refresh_tickers(database_audit, names))
    event_names = tuple(bridge.read_pending_event_tickers(names)) if bool(getattr(config, "event_aware_refresh_enabled", True)) else ()
    scheduler_names = tuple(dict.fromkeys(due_names + event_names))
    refresh_names_list, scheduler_audit = select_round_robin_backfill(
        scheduler_names,
        database_audit[database_audit.get("ticker", pd.Series(dtype=str)).isin(scheduler_names)].copy()
        if database_audit is not None and not database_audit.empty else database_audit,
        priority_tickers=due_names[: min(4, len(due_names))],
        event_tickers=event_names,
        cohort_count=int(getattr(config, "forward_backfill_cohorts", 7)),
        max_count=int(getattr(config, "automatic_forward_quality_top_n", 12)),
    )
    refresh_names = tuple(refresh_names_list)
    if scheduler_audit is not None and not scheduler_audit.empty:
        scheduler_audit = scheduler_audit.copy()
        scheduler_audit["provider"] = "ROUND_ROBIN_BACKFILL"
        scheduler_audit["scope"] = "FORWARD_QUALITY"
        scheduler_audit["status"] = np.where(scheduler_audit["selected_for_refresh"], "SELECTED", "DEFERRED")
        scheduler_audit["rows"] = 0
        scheduler_audit["error"] = ""
        scheduler_audit["source_family"] = "SCHEDULER"
    current_rows = pd.DataFrame()
    provider_report = pd.DataFrame()
    if refresh_names:
        refresh_fundamentals = fundamentals[
            fundamentals.get("ticker", pd.Series(dtype=str)).isin(refresh_names)
        ].copy() if fundamentals is not None and not fundamentals.empty else pd.DataFrame({"ticker": refresh_names})
        try:
            current_rows, provider_report = collect_automatic_forward_quality(
                refresh_fundamentals, list(refresh_names), config, force_refresh=False,
            )
        except Exception as exc:
            provider_report = pd.DataFrame([{
                "ticker": "SYSTEM", "provider": "FORWARD_QUALITY_PROVIDER", "scope": "FORWARD_QUALITY",
                "status": "FORWARD_ENRICHMENT_FAIL_SOFT", "rows": 0,
                "error": f"{type(exc).__name__}: {str(exc)[:240]}", "source_family": "PROVIDER",
            }])
    if provider_report is not None and not provider_report.empty:
        provider_report = provider_report.copy()
        if "provider" not in provider_report.columns:
            provider_report["provider"] = "FORWARD_QUALITY_PROVIDER"
        if "scope" not in provider_report.columns:
            provider_report["scope"] = provider_report.get("ticker", pd.Series("ALL", index=provider_report.index))
        if "status" not in provider_report.columns:
            provider_report["status"] = provider_report.get("state", pd.Series("UNKNOWN", index=provider_report.index))
        if "error" not in provider_report.columns:
            provider_report["error"] = provider_report.get("errors", pd.Series("", index=provider_report.index))
        if "source_family" not in provider_report.columns:
            provider_report["source_family"] = "FORWARD_PROVIDER"
    combined = merge_project_management_reviews(database_rows, current_rows)
    reports = [frame for frame in (database_audit, scheduler_audit, provider_report) if frame is not None and not frame.empty]
    return combined, pd.concat(reports, ignore_index=True, sort=False) if reports else pd.DataFrame()

def rupiah(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"Rp{float(value):,.0f}".replace(",", ".")


def prepare_display(signals: pd.DataFrame) -> pd.DataFrame:
    out = signals.copy()
    if {"historical_events", "bayes_probability", "entry_fill_rate_5d"}.issubset(out.columns):
        reliable = pd.to_numeric(out["historical_events"], errors="coerce").fillna(0) >= 30
        out["probability_estimate"] = pd.to_numeric(out["bayes_probability"], errors="coerce").where(reliable)
        out["entry_fill_estimate"] = pd.to_numeric(out["entry_fill_rate_5d"], errors="coerce").where(reliable)
    else:
        out["probability_estimate"] = np.nan
        out["entry_fill_estimate"] = np.nan
    # Streamlit NumberColumn uses printf-style formatting and does not
    # multiply fractional ratios automatically. Convert only display copies.
    for ratio_column in (
        "stop_pct", "quote_spread_pct", "broksum_net_ratio",
        "independent_price_divergence_pct",
    ):
        if ratio_column in out:
            out[ratio_column] = pd.to_numeric(out[ratio_column], errors="coerce") * 100.0
    columns = [
        "ticker",
        "status",
        "execution_policy",
        "setup_state",
        "account_order_state",
        "manual_execution_candidate",
        "signal_risk_grade",
        "signal_risk_warnings",
        "signal_execution_blockers",
        "setup",
        "grade",
        "quality_score",
        "composite_score",
        "analyst_fusion_score",
        "analyst_order_mode",
        "execution_mode",
        "autopilot_verified",
        "autopilot_score",
        "autopilot_blockers",
        "autopilot_primary_setup",
        "confluence_setup_count",
        "confluence_setups",
        "strict_execution_ready",
        "requires_stockbit_price_check",
        "analyst_candidate_reason",
        "analyst_decision_basis",
        "analyst_hard_blockers",
        "strict_primary_execution_blocker",
        "strict_execution_gate_failures",
        "execution_integrity_score",
        "execution_confidence_score",
        "projected_completeness_with_independent_price",
        "projected_confidence_with_independent_price",
        "execution_readiness_pct",
        "primary_execution_blocker",
        "execution_gate_failures",
        "data_completeness_score",
        "data_completeness_tier",
        "data_missing_layers",
        "technical_data_coverage",
        "risk_data_coverage",
        "fundamental_data_coverage",
        "validation_data_coverage",
        "market_status_data_coverage",
        "news_data_coverage",
        "quote_data_coverage",
        "universe_data_coverage",
        "evidence_state",
        "validation_gate_score",
        "validation_tier",
        "probability_estimate",
        "entry_fill_estimate",
        "historical_events",
        "median_fill_bars",
        "median_time_to_tp1_bars",
        "last_price",
        "entry_type",
        "entry_low",
        "entry_high",
        "entry",
        "stop_loss",
        "tp1",
        "tp2",
        "tp1_basis",
        "tp2_basis",
        "rr1",
        "rr2",
        "stop_pct",
        "distance_atr",
        "volume_ratio",
        "adtv20_idr",
        "fundamental_score",
        "fundamental_score_10",
        "fundamental_data_grade",
        "fundamental_source_count",
        "fundamental_history_source_count",
        "fundamental_snapshot_source_count",
        "fundamental_source_families",
        "fundamental_official_verified",
        "fundamental_consensus_score",
        "fundamental_conflicts",
        "fundamental_coverage",
        "statement_age_days",
        "silent_accumulation_score",
        "up_down_value_ratio20",
        "quote_last_price",
        "quote_spread_pct",
        "quote_market_state",
        "independent_price_state",
        "independent_source",
        "independent_source_family",
        "independent_asof",
        "independent_last_price",
        "independent_price_divergence_pct",
        "independent_overlap_bars",
        "independent_return_correlation",
        "suggested_lots",
        "sizing_is_informational",
        "proposed_order_instruction",
        "order_instruction",
        "stockbit_trigger_price",
        "stockbit_limit_price",
        "execution_timing",
        "stockbit_order_price",
        "stockbit_order_lots",
        "execution_rank",
        "capital_required_idr",
        "max_loss_idr",
        "max_loss_pct_account",
        "broksum_signal",
        "broksum_net_ratio",
        "verified_catalyst_count",
        "catalyst_summary",
        "market_status_coverage",
        "market_status_confidence",
        "news_review_status",
        "news_confidence",
        "fundamental_confidence",
        "quote_confidence",
        "universe_confidence",
        "automation_decision",
        "critical_blockers",
        "evidence_warnings",
        "portfolio_blockers",
        "market_regime",
        "action",
        "valid_until",
        "blockers",
        "reason",
    ]
    return streamlit_safe_frame(
        out[[c for c in columns if c in out.columns]]
    )


def sort_signals(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals
    out = signals.copy()
    if "composite_score" not in out:
        out["composite_score"] = out["quality_score"]
    out["status_rank"] = out["status"].map({
        "EXECUTION_READY": 0,
        "READY_FOR_STOCKBIT_VERIFY": 1,
        "SIGNAL_READY": 2,
        "ENTRY_PLAN_READY": 3,
        "READY_NOT_SELECTED": 4,
        "READY_FOR_PRICE_VERIFY": 5,
        "PENDING_CLOSE": 6,
        "PENDING_DATA": 7,
        "WATCHLIST_ENTRY": 8,
        "BLOCKED_CONTEXT": 9,
        "REJECT": 10,
    }).fillna(99)
    return out.sort_values(
        ["status_rank", "composite_score", "quality_score", "rr2", "adtv20_idr"],
        ascending=[True, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)


def execution_funnel_summary(signals: pd.DataFrame) -> pd.DataFrame:
    if signals is None or signals.empty:
        return pd.DataFrame(columns=["Tahap", "Ticker unik"])
    ticker = signals.get("ticker", pd.Series(signals.index.astype(str), index=signals.index)).astype(str)
    actionable = signals.get("analyst_order_mode", pd.Series("WATCH_ONLY", index=signals.index)).fillna("WATCH_ONLY").ne("WATCH_ONLY")
    no_hard = signals.get("analyst_hard_blockers", pd.Series("", index=signals.index)).fillna("").astype(str).str.strip().eq("")
    final_bar = ~signals.get("pending_close", pd.Series(False, index=signals.index)).fillna(False).astype(bool)
    valid_signal = signals.get("setup_valid_signal", pd.Series(False, index=signals.index)).fillna(False).astype(bool)
    def unique_count(mask: pd.Series) -> int:
        return int(ticker[mask].nunique())
    return pd.DataFrame([
        {"Tahap": "Setup terdeteksi", "Ticker unik": int(ticker.nunique())},
        {"Tahap": "Order mode actionable", "Ticker unik": unique_count(actionable)},
        {"Tahap": "Tidak ada invalidasi struktural/data", "Ticker unik": unique_count(actionable & no_hard)},
        {"Tahap": "Candle EOD final", "Ticker unik": unique_count(actionable & no_hard & final_bar)},
        {"Tahap": "Setup valid Signal-First", "Ticker unik": unique_count(valid_signal)},
        {"Tahap": "SIGNAL_READY", "Ticker unik": unique_count(signals["status"].eq("SIGNAL_READY"))},
        {"Tahap": "ENTRY_PLAN_READY", "Ticker unik": unique_count(signals["status"].eq("ENTRY_PLAN_READY"))},
        {"Tahap": "Harga independen satu sesi", "Ticker unik": unique_count(signals.get("independent_price_verified", pd.Series(False, index=signals.index)).fillna(False).astype(bool))},
        {"Tahap": "READY_FOR_STOCKBIT_VERIFY", "Ticker unik": unique_count(signals["status"].eq("READY_FOR_STOCKBIT_VERIFY"))},
        {"Tahap": "CORE_PLAN_VERIFIED", "Ticker unik": unique_count(signals.get("autopilot_verified", pd.Series(False, index=signals.index)).fillna(False).astype(bool))},
        {"Tahap": "EXECUTION_READY", "Ticker unik": unique_count(signals["status"].eq("EXECUTION_READY"))},
    ])


def result_table(df: pd.DataFrame) -> None:
    st.dataframe(
        prepare_display(df),
        width="stretch",
        hide_index=True,
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", pinned=True),
            "status": st.column_config.TextColumn("Status", pinned=True),
            "execution_policy": st.column_config.TextColumn("Policy"),
            "setup_state": st.column_config.TextColumn("Setup state"),
            "manual_execution_candidate": st.column_config.CheckboxColumn("Verify Stockbit"),
            "signal_risk_grade": st.column_config.TextColumn("Risk grade"),
            "signal_risk_warnings": st.column_config.TextColumn("Risk warnings"),
            "setup": "Setup",
            "quality_score": st.column_config.NumberColumn("Technical", format="%.1f"),
            "composite_score": st.column_config.NumberColumn("Composite", format="%.1f"),
            "analyst_fusion_score": st.column_config.NumberColumn("Signal score", format="%.1f"),
            "analyst_order_mode": st.column_config.TextColumn("Analyst order mode"),
            "execution_mode": st.column_config.TextColumn("Execution mode"),
            "autopilot_verified": st.column_config.CheckboxColumn("Autopilot"),
            "autopilot_score": st.column_config.NumberColumn("Autopilot score", format="%.1f%%"),
            "autopilot_primary_setup": st.column_config.CheckboxColumn("Primary order"),
            "confluence_setup_count": st.column_config.NumberColumn("Setup count", format="%d"),
            "strict_execution_ready": st.column_config.CheckboxColumn("Strict ready"),
            "requires_stockbit_price_check": st.column_config.CheckboxColumn("Check Stockbit"),
            "execution_integrity_score": st.column_config.NumberColumn("Execution confidence", format="%.1f%%"),
            "execution_confidence_score": st.column_config.NumberColumn("Confidence", format="%.1f%%"),
            "data_completeness_score": st.column_config.NumberColumn("Data completeness", format="%.1f%%"),
            "data_completeness_tier": st.column_config.TextColumn("Completeness tier"),
            "daily_session_current": st.column_config.CheckboxColumn("Current IDX session"),
            "data_lag_sessions": st.column_config.NumberColumn("Lag sessions", format="%d"),
            "expected_last_completed_session": st.column_config.TextColumn("Expected EOD session"),
            "ohlcv_source_tier": st.column_config.TextColumn("OHLCV route"),
            "technical_data_coverage": st.column_config.NumberColumn("Tech data", format="%.0f%%"),
            "risk_data_coverage": st.column_config.NumberColumn("Risk data", format="%.0f%%"),
            "fundamental_data_coverage": st.column_config.NumberColumn("Fund. data", format="%.0f%%"),
            "validation_data_coverage": st.column_config.NumberColumn("OOS data", format="%.0f%%"),
            "market_status_data_coverage": st.column_config.NumberColumn("IDX status data", format="%.0f%%"),
            "news_data_coverage": st.column_config.NumberColumn("News data", format="%.0f%%"),
            "quote_data_coverage": st.column_config.NumberColumn("Quote data", format="%.0f%%"),
            "universe_data_coverage": st.column_config.NumberColumn("Universe data", format="%.0f%%"),
            "validation_gate_score": st.column_config.NumberColumn("OOS quality gate", format="%.0f%%"),
            "probability_estimate": st.column_config.NumberColumn("Setup OOS P(TP1<SL)*", format="%.1f%%"),
            "entry_fill_estimate": st.column_config.NumberColumn("P(fill≤5d)*", format="%.1f%%"),
            "historical_events": st.column_config.NumberColumn("Sample OOS", format="%d"),
            "median_fill_bars": st.column_config.NumberColumn("Median fill", format="%.1f bar"),
            "median_time_to_tp1_bars": st.column_config.NumberColumn("Median TP1", format="%.1f bar"),
            "last_price": st.column_config.NumberColumn("Last", format="Rp %.0f"),
            "entry_low": st.column_config.NumberColumn("Zone low", format="Rp %.0f"),
            "entry_high": st.column_config.NumberColumn("Zone high", format="Rp %.0f"),
            "entry": st.column_config.NumberColumn("Entry / trigger", format="Rp %.0f"),
            "reclaim_trigger_price": st.column_config.NumberColumn("Reclaim trigger", format="Rp %.0f"),
            "retest_reference_price": st.column_config.NumberColumn("Retest reference", format="Rp %.0f"),
            "trigger_basis": st.column_config.TextColumn("Trigger basis"),
            "trigger_instruction": st.column_config.TextColumn("Trigger instruction"),
            "trigger_valid_until": st.column_config.DatetimeColumn("Trigger valid until", format="DD MMM YYYY"),
            "stop_loss": st.column_config.NumberColumn("SL", format="Rp %.0f"),
            "tp1": st.column_config.NumberColumn("TP1", format="Rp %.0f"),
            "tp2": st.column_config.NumberColumn("TP2", format="Rp %.0f"),
            "tp1_basis": st.column_config.TextColumn("TP1 basis"),
            "tp2_basis": st.column_config.TextColumn("TP2 basis"),
            "target_model": st.column_config.TextColumn("Target model"),
            "target_structure": st.column_config.TextColumn("Target structure"),
            "rr1": st.column_config.NumberColumn("RR1", format="%.2f"),
            "rr2": st.column_config.NumberColumn("RR2", format="%.2f"),
            "stop_pct": st.column_config.NumberColumn("Risk", format="%.1f%%"),
            "distance_atr": st.column_config.NumberColumn("Dist. ATR", format="%.2f"),
            "volume_ratio": st.column_config.NumberColumn("Vol x", format="%.2f"),
            "adtv20_idr": st.column_config.NumberColumn("ADTV20", format="Rp %.0f"),
            "fundamental_score": st.column_config.NumberColumn("Fund.", format="%.1f"),
            "fundamental_score_10": st.column_config.NumberColumn("Fund. /10", format="%.2f"),
            "fundamental_data_grade": st.column_config.TextColumn("Fund. data grade"),
            "fundamental_source_count": st.column_config.NumberColumn("History sources", format="%d"),
            "fundamental_history_source_count": st.column_config.NumberColumn("History sources (explicit)", format="%d"),
            "fundamental_snapshot_source_count": st.column_config.NumberColumn("Snapshot source", format="%d"),
            "fundamental_official_verified": st.column_config.CheckboxColumn("IDX/XBRL verified"),
            "fundamental_consensus_score": st.column_config.NumberColumn("Fund. consensus", format="%.1f"),
            "fundamental_coverage": st.column_config.NumberColumn("Fund. coverage", format="%.0f%%"),
            "statement_age_days": st.column_config.NumberColumn("FS age", format="%.0f d"),
            "silent_accumulation_score": st.column_config.NumberColumn("Accumulation", format="%.0f"),
            "up_down_value_ratio20": st.column_config.NumberColumn("Up/Down value", format="%.2f"),
            "quote_last_price": st.column_config.NumberColumn("Quote", format="Rp %.0f"),
            "independent_last_price": st.column_config.NumberColumn("Independent", format="Rp %.0f"),
            "independent_price_divergence_pct": st.column_config.NumberColumn("Price diff", format="%.2f%%"),
            "independent_return_correlation": st.column_config.NumberColumn("Return corr.", format="%.3f"),
            "quote_spread_pct": st.column_config.NumberColumn("Spread", format="%.2f%%"),
            "suggested_lots": st.column_config.NumberColumn("Lot", format="%d"),
            "stockbit_order_price": st.column_config.NumberColumn("Order price", format="Rp %.0f"),
            "stockbit_order_lots": st.column_config.NumberColumn("Order lot", format="%d"),
            "execution_rank": st.column_config.NumberColumn("Rank", format="%.0f"),
            "capital_required_idr": st.column_config.NumberColumn("Modal order", format="Rp %.0f"),
            "max_loss_idr": st.column_config.NumberColumn("Max loss est.", format="Rp %.0f"),
            "max_loss_pct_account": st.column_config.NumberColumn("Risk akun", format="%.2f%%"),
            "broksum_net_ratio": st.column_config.NumberColumn("Broksum net", format="%.1f%%"),
            "valid_until": st.column_config.DatetimeColumn("Valid until", format="DD MMM YYYY"),
        },
    )



def focus_table(df: pd.DataFrame | None, height: int = 460) -> None:
    """Render focused Core Swing/Multibagger evidence without legacy columns."""
    if df is None or df.empty:
        st.info("Belum ada kandidat yang memenuhi filter fokus.")
        return
    display = prepare_display(df.copy())
    st.dataframe(display, hide_index=True, height=height, width="stretch")


def render_focus_download(label: str, df: pd.DataFrame | None, filename: str) -> None:
    st.download_button(
        label,
        b"" if df is None or df.empty else df.to_csv(index=False).encode("utf-8"),
        filename,
        "text/csv",
        width="stretch",
        disabled=df is None or df.empty,
    )


def render_ihsg_direction_panel(
    forecast: Mapping[str, object] | None,
    benchmark: pd.DataFrame | None,
) -> None:
    if not isinstance(forecast, Mapping):
        st.info("Belum ada hasil analisis IHSG.")
        return
    st.subheader("IHSG Direction Lab — probabilitas, regime, dan risk budget")
    st.caption(
        "Mesin menilai horizon 1/5/20 hari bursa dengan historical analogue, breadth universe, "
        "dan chronological walk-forward. `ABSTAIN` berarti bukti belum cukup. Arah IHSG tidak "
        "menaikkan Final Score saham dan tidak membuat order otomatis."
    )
    regime = str(forecast.get("regime", "UNKNOWN"))
    consensus = str(forecast.get("consensus_direction", "NO_EDGE"))
    data_state = str(forecast.get("data_state", "UNKNOWN"))
    risk_budget = float(forecast.get("risk_budget_pct", 0.0) or 0.0)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Regime", regime)
    c2.metric("Consensus", consensus)
    c3.metric("Risk budget cap", f"{risk_budget:.0f}%")
    c4.metric("Confidence", f"{float(forecast.get('consensus_confidence', 0.0) or 0.0):.1f}%")
    c5.metric("Breadth > EMA50", f"{float(forecast.get('breadth_ema50_pct', np.nan)):.1f}%" if pd.notna(forecast.get("breadth_ema50_pct")) else "N/A")
    c6.metric("Data", data_state)

    if data_state != "READY":
        st.warning(
            f"Forecast fail-closed: {data_state}. Risk budget dibatasi dan arah produksi tidak diterbitkan."
        )
    elif bool(forecast.get("crash_risk")):
        st.error(
            "Crash-risk guard aktif: drawdown dan volatilitas berada pada zona defensif. "
            "Prioritas utama adalah preservasi modal."
        )
    elif consensus == "NO_EDGE":
        st.warning(
            "Model belum memiliki edge OOS yang cukup untuk arah produksi. Raw probabilities tetap "
            "ditampilkan sebagai riset, tetapi keputusan resmi adalah ABSTAIN."
        )
    else:
        st.info(
            f"{forecast.get('regime_reason', '')} · Kebijakan risiko: {forecast.get('risk_action', '')}."
        )
    if bool(forecast.get("risk_cap_applied")):
        st.success(
            "Account-Guarded: risk cap telah diterapkan pada risk-per-trade, portfolio heat, "
            f"dan budget Multibagger. Effective risk/trade "
            f"{100.0 * float(forecast.get('effective_risk_per_trade_pct', 0.0) or 0.0):.3f}%."
        )
    else:
        st.caption(
            "Signal-First: IHSG risk cap bersifat informasional; sizing dan keputusan akhir tetap dikelola pengguna."
        )

    horizons = forecast.get("horizons")
    if isinstance(horizons, pd.DataFrame) and not horizons.empty:
        horizons = horizons.copy()
        probability_columns = {
            "UP": "prob_up_pct",
            "SIDEWAYS": "prob_sideways_pct",
            "DOWN": "prob_down_pct",
        }
        if all(column in horizons for column in probability_columns.values()):
            probability_view = horizons[
                list(probability_columns.values())
            ].apply(pd.to_numeric, errors="coerce")
            label_by_column = {
                column: label for label, column in probability_columns.items()
            }
            valid_probability_rows = probability_view.notna().any(axis=1)
            probability_leader = pd.Series(
                "N/A", index=probability_view.index, dtype="string",
            )
            if valid_probability_rows.any():
                probability_leader.loc[valid_probability_rows] = (
                    probability_view.loc[valid_probability_rows]
                    .idxmax(axis=1)
                    .map(label_by_column)
                    .astype("string")
                )
            horizons["probability_leader_research"] = probability_leader

            probability_values = probability_view.to_numpy(dtype=float)
            finite_values = np.where(
                np.isfinite(probability_values), probability_values, -np.inf,
            )
            sorted_probability = np.sort(finite_values, axis=1)
            probability_counts = probability_view.notna().sum(axis=1).to_numpy()
            horizons["probability_edge_pct"] = np.where(
                probability_counts >= 2,
                sorted_probability[:, -1] - sorted_probability[:, -2],
                np.nan,
            )
        display_columns = [
            "horizon", "prediction_state", "probability_leader_research",
            "probability_edge_pct", "raw_direction", "prob_up_pct",
            "prob_sideways_pct", "prob_down_pct", "confidence_pct",
            "expected_return_pct", "return_p25_pct", "return_p75_pct",
            "analogue_count", "validation_state", "directional_accuracy_pct",
            "directional_accuracy_ci_low_pct", "brier_skill_pct", "abstain_reason",
        ]
        display = streamlit_safe_frame(horizons.loc[
            :, [column for column in display_columns if column in horizons.columns]
        ])
        st.dataframe(
            display,
            hide_index=True,
            width="stretch",
            column_config={
                "prob_up_pct": st.column_config.NumberColumn("P(UP)", format="%.1f%%"),
                "prob_sideways_pct": st.column_config.NumberColumn("P(SIDE)", format="%.1f%%"),
                "prob_down_pct": st.column_config.NumberColumn("P(DOWN)", format="%.1f%%"),
                "probability_leader_research": st.column_config.TextColumn("Raw leader (research)"),
                "probability_edge_pct": st.column_config.NumberColumn("Raw edge", format="%.1f%%"),
                "confidence_pct": st.column_config.NumberColumn("Confidence", format="%.1f%%"),
                "expected_return_pct": st.column_config.NumberColumn("Analog mean", format="%.2f%%"),
                "return_p25_pct": st.column_config.NumberColumn("Analog P25", format="%.2f%%"),
                "return_p75_pct": st.column_config.NumberColumn("Analog P75", format="%.2f%%"),
                "directional_accuracy_pct": st.column_config.NumberColumn("OOS accuracy", format="%.1f%%"),
                "directional_accuracy_ci_low_pct": st.column_config.NumberColumn("OOS CI low", format="%.1f%%"),
                "brier_skill_pct": st.column_config.NumberColumn("Brier skill", format="%.1f%%"),
            },
        )
        st.download_button(
            "Download IHSG direction audit",
            horizons.to_csv(index=False).encode("utf-8"),
            "ihsg_direction_audit_v7_3.csv",
            "text/csv",
            key="download_ihsg_direction_v720",
        )
    if benchmark is not None and not benchmark.empty:
        try:
            chart = make_ihsg_direction_chart(benchmark, forecast)
            if chart is not None:
                st.plotly_chart(chart, width="stretch", key="ihsg_direction_chart_v720")
        except Exception as exc:
            st.caption(f"Chart IHSG tidak dapat dirender: {exc}")
    with st.expander("Batas model dan arti probabilitas"):
        for item in list(forecast.get("limitations", []) or []):
            st.write("•", item)
        st.write(
            "Kriteria produksi: completed EOD, data current, probability terpisah, confidence minimum, "
            "dan walk-forward OOS harus berstatus `OOS_POSITIVE`. Jika salah satu gagal, keluaran menjadi `ABSTAIN`."
        )


def build_tradingview_bridge(
    core_signals: pd.DataFrame,
    focus_screens: dict[str, pd.DataFrame] | None,
) -> pd.DataFrame:
    """Normalize scanner outputs into levels that can be copied to TradingView.

    Pine Script cannot read a local Streamlit dataframe directly. This bridge
    exports a stable schema so the selected row can be pasted into the
    indicator's MANUAL SCANNER LEVELS inputs without reinterpreting prices.
    """
    columns = [
        "source", "ticker", "tv_symbol", "setup", "status", "timeframe",
        "entry_low", "entry_high", "entry", "order_trigger", "confirmation_level",
        "reclaim_trigger_price", "retest_reference_price", "trigger_basis",
        "trigger_instruction", "trigger_valid_until",
        "stop_loss", "tp1", "tp2", "rr1", "rr2", "execution_plan_mode",
        "execution_plan_state", "execution_plan_valid", "valid_until", "market_regime", "quality_score",
        "data_completeness_score", "execution_confidence_score", "scanner_note",
    ]
    rows: list[dict[str, object]] = []

    def clean_ticker(value: object) -> str:
        ticker = str(value or "").strip().upper()
        return ticker[:-3] if ticker.endswith(".JK") else ticker

    def add_rows(
        frame: pd.DataFrame | None,
        *,
        source: str,
        setup_default: str,
        status_column: str,
        timeframe: str,
        entry_column: str = "entry",
        stop_column: str = "stop_loss",
        tp1_column: str = "tp1",
        tp2_column: str = "tp2",
        setup_column: str = "setup",
        entry_low_column: str = "entry_low",
        entry_high_column: str = "entry_high",
        note_column: str = "blockers",
    ) -> None:
        if frame is None or frame.empty or "ticker" not in frame:
            return
        for _, row in frame.iterrows():
            ticker = clean_ticker(row.get("ticker"))
            if not ticker:
                continue
            plan_row = row.copy()
            if entry_column != "entry":
                plan_row["entry"] = row.get(entry_column)
            if entry_low_column != "entry_low":
                plan_row["entry_low"] = row.get(entry_low_column)
            if entry_high_column != "entry_high":
                plan_row["entry_high"] = row.get(entry_high_column)
            if stop_column != "stop_loss":
                plan_row["stop_loss"] = row.get(stop_column)
            if tp1_column != "tp1":
                plan_row["tp1"] = row.get(tp1_column)
            if tp2_column != "tp2":
                plan_row["tp2"] = row.get(tp2_column)
            plan = resolve_trade_plan(plan_row)
            plan_valid = bool(plan.get("execution_plan_valid", False))
            entry = plan.get("execution_price", np.nan) if plan_valid else np.nan
            entry_low = plan.get("entry_low", np.nan) if plan_valid else np.nan
            entry_high = plan.get("entry_high", np.nan) if plan_valid else np.nan
            setup_value = str(row.get(setup_column) or setup_default)
            source_status = str(row.get(status_column) or "NOT_EVALUATED")
            status_value = source_status if plan_valid else "WAIT_FOR_VALID_PLAN"
            note_parts = [
                str(row.get(note_column) or "").strip(),
                str(row.get("reason") or "").strip(),
                str(row.get("action") or "").strip(),
            ]
            rows.append({
                "source": source,
                "ticker": f"{ticker}.JK",
                "tv_symbol": f"IDX:{ticker}",
                "setup": setup_value,
                "status": status_value,
                "timeframe": timeframe,
                "entry_low": entry_low,
                "entry_high": entry_high,
                "entry": entry,
                "order_trigger": plan.get("trigger", np.nan) if plan_valid else np.nan,
                "confirmation_level": plan.get("confirmation_level", np.nan) if plan_valid else np.nan,
                "reclaim_trigger_price": row.get("reclaim_trigger_price", plan.get("trigger", np.nan)) if plan_valid else np.nan,
                "retest_reference_price": row.get("retest_reference_price", np.nan),
                "trigger_basis": row.get("trigger_basis", ""),
                "trigger_instruction": row.get("trigger_instruction", ""),
                "trigger_valid_until": row.get("trigger_valid_until", row.get("valid_until", pd.NaT)),
                "stop_loss": plan.get("stop_loss", np.nan) if plan_valid else np.nan,
                "tp1": plan.get("tp1", np.nan) if plan_valid else np.nan,
                "tp2": plan.get("tp2", np.nan) if plan_valid else np.nan,
                "rr1": plan.get("rr1", np.nan) if plan_valid else np.nan,
                "rr2": plan.get("rr2", np.nan) if plan_valid else np.nan,
                "execution_plan_mode": plan.get("execution_plan_mode", "NO_VALID_PLAN"),
                "execution_plan_state": plan.get("execution_plan_state", "NO_VALID_PLAN"),
                "execution_plan_valid": plan_valid,
                "valid_until": row.get("valid_until", pd.NaT),
                "market_regime": row.get("market_regime", ""),
                "quality_score": row.get("quality_score", row.get("sniper_score", np.nan)),
                "data_completeness_score": row.get("data_completeness_score", np.nan),
                "execution_confidence_score": row.get("execution_confidence_score", np.nan),
                "scanner_note": " | ".join(dict.fromkeys(
                    part for part in (
                        *note_parts,
                        "Trade plan invalid; actionable levels intentionally hidden."
                        if not plan_valid else "",
                    ) if part
                )),
            })

    add_rows(
        core_signals,
        source="CORE",
        setup_default="CORE",
        status_column="status",
        timeframe="1D",
    )
    focus = focus_screens or {}
    add_rows(
        focus.get("multibagger"), source="MULTIBAGGER", setup_default="MULTIBAGGER",
        status_column="multibagger_status", timeframe="1W / 1D", setup_column="active_setup",
        note_column="red_flags",
    )
    bridge = pd.DataFrame(rows, columns=columns)
    if bridge.empty:
        return bridge
    numeric = ["entry_low", "entry_high", "entry", "stop_loss", "tp1", "tp2", "rr1", "rr2"]
    for column in numeric:
        bridge[column] = pd.to_numeric(bridge[column], errors="coerce")
    bridge = bridge.drop_duplicates(["source", "ticker", "setup", "status"], keep="first")
    status_rank = {
        "EXECUTION_READY": 0,
        "READY_FOR_STOCKBIT_VERIFY": 1,
        "MULTIBAGGER_A_CANDIDATE": 1,
        "SIGNAL_READY": 2,
        "READY_FOR_PRICE_VERIFY": 2,
        "ENTRY_PLAN_READY": 3,
        "MULTIBAGGER_B_CANDIDATE": 3,
        "MULTIBAGGER_WATCHLIST": 4,
    }
    bridge["_rank"] = bridge["status"].map(status_rank).fillna(9)
    return bridge.sort_values(["_rank", "source", "ticker"]).drop(columns="_rank").reset_index(drop=True)


def render_portfolio_panel(result: dict) -> None:
    portfolio_analysis: pd.DataFrame = result.get("portfolio_analysis", pd.DataFrame())
    portfolio_summary: dict = result.get("portfolio_summary", {})
    if portfolio_analysis.empty:
        st.info("Upload portfolio CSV untuk memperoleh keputusan portfolio.")
        return

    st.subheader("Portfolio decision engine")
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.metric("Nilai posisi", rupiah(portfolio_summary.get("market_value_idr", np.nan)))
    p2.metric("Unrealized P/L", rupiah(portfolio_summary.get("unrealized_pnl_idr", np.nan)))
    p3.metric("P/L %", f"{portfolio_summary.get('unrealized_pnl_pct', 0):.1%}")
    p4.metric("Open risk", rupiah(portfolio_summary.get("open_risk_idr", np.nan)))
    p5.metric("Cash", rupiah(portfolio_summary.get("cash_on_hand_idr", np.nan)))
    p6.metric("Equity basis", rupiah(portfolio_summary.get("estimated_equity_idr", np.nan)))

    inferred = float(portfolio_summary.get("inferred_equity_idr", 0.0) or 0.0)
    equity = float(portfolio_summary.get("estimated_equity_idr", 0.0) or 0.0)
    source = str(portfolio_summary.get("equity_source", ""))
    if source == "ACCOUNT_EQUITY_INPUT" and inferred > 0 and equity > inferred * 1.50:
        st.warning(
            "Equity akun yang diinput jauh lebih besar daripada nilai posisi + cash. "
            "Bobot posisi menggunakan equity input. Pastikan angka Equity akun dan Cash on hand sesuai Stockbit terbaru."
        )

    portfolio_columns = [
        "ticker", "position_action", "action_reason", "lots", "avg_price", "last_price",
        "unrealized_pnl_idr", "unrealized_pnl_pct", "position_weight_pct",
        "open_risk_idr", "open_risk_pct_equity_pct", "existing_stop_loss",
        "structural_stop_loss", "suggested_stop_loss", "suggested_tp1", "suggested_tp2",
        "tp1_basis", "tp2_basis", "scanner_setup", "scanner_status", "portfolio_add_setup",
        "avg_down_lots", "avg_down_price", "new_average_after_avg", "trend_up",
        "long_term_structure_intact", "flow_positive", "stop_breached",
        "confirmed_structure_breakdown", "fundamental_distress",
    ]
    portfolio_view = portfolio_analysis[[c for c in portfolio_columns if c in portfolio_analysis.columns]].copy()
    if "unrealized_pnl_pct" in portfolio_view:
        portfolio_view["unrealized_pnl_pct"] = pd.to_numeric(
            portfolio_view["unrealized_pnl_pct"], errors="coerce"
        ) * 100.0
    st.dataframe(
        portfolio_view, width="stretch", hide_index=True,
        column_config={
            "avg_price": st.column_config.NumberColumn("Average", format="Rp %.0f"),
            "last_price": st.column_config.NumberColumn("Last", format="Rp %.0f"),
            "unrealized_pnl_idr": st.column_config.NumberColumn("Unrealized P/L", format="Rp %.0f"),
            "unrealized_pnl_pct": st.column_config.NumberColumn("P/L %", format="%.1f%%"),
            "position_weight_pct": st.column_config.NumberColumn("Weight", format="%.1f%%"),
            "open_risk_idr": st.column_config.NumberColumn("Open risk", format="Rp %.0f"),
            "open_risk_pct_equity_pct": st.column_config.NumberColumn("Risk equity", format="%.2f%%"),
            "existing_stop_loss": st.column_config.NumberColumn("Existing SL", format="Rp %.0f"),
            "structural_stop_loss": st.column_config.NumberColumn("Structural SL", format="Rp %.0f"),
            "suggested_stop_loss": st.column_config.NumberColumn("Suggested SL", format="Rp %.0f"),
            "suggested_tp1": st.column_config.NumberColumn("TP1", format="Rp %.0f"),
            "suggested_tp2": st.column_config.NumberColumn("TP2", format="Rp %.0f"),
            "avg_down_price": st.column_config.NumberColumn("Avg-down price", format="Rp %.0f"),
            "new_average_after_avg": st.column_config.NumberColumn("New average", format="Rp %.0f"),
        },
    )
    st.caption(
        "Tidak adanya setup entry baru bersifat netral untuk posisi lama. CUT_LOSS hanya muncul saat stop tersentuh, "
        "breakdown multi-faktor terkonfirmasi, atau distress berat disertai kerusakan struktur."
    )
    source_quorum = result.get("source_quorum_audit", pd.DataFrame())
    if not source_quorum.empty:
        with st.expander("Audit source quorum portfolio"):
            st.dataframe(source_quorum, hide_index=True, width="stretch")
    st.download_button(
        "Download portfolio action plan",
        portfolio_analysis.to_csv(index=False).encode("utf-8"),
        "stockbit_portfolio_action_plan.csv",
        "text/csv",
        width="stretch",
    )


def run_single_ticker_deep_dive(
    ticker: str,
    lookback: str,
    config: ScanConfig,
    *,
    twelve_data_api_key: str = "",
    itick_api_token: str = "",
) -> dict:
    """Run a bounded full daily review for one ticker without a universe CSV."""
    audit_warnings: list[str] = []
    normalized = normalize_idx_ticker(ticker)
    histories, download_report = download_ohlcv(
        (normalized,), period=lookback, itick_api_token=itick_api_token,
    )
    history = histories.get(normalized)
    if history is None or history.empty:
        reason = getattr(download_report, "failed", {}).get(normalized, "OHLCV tidak tersedia")
        raise ValueError(reason)
    benchmark = download_benchmark(period=lookback)
    mini = ScanEngine(config).scan(histories, benchmark)
    signals = mini.get("signals", pd.DataFrame()).copy()
    if not signals.empty:
        signals["technical_setup_ready"] = signals.get("status", pd.Series(index=signals.index, dtype=str)).eq("EXECUTION_READY")
        source_tiers = getattr(download_report, "source_tiers", {}) or {}
        signals["ohlcv_source_tier"] = signals["ticker"].map(source_tiers).fillna("UNAVAILABLE")
        signals = apply_universe_integrity_gate(signals, [normalized], mini.get("prepared", {}).keys(), config)

    fundamentals, database_fundamental_audit = database_first_fundamentals((normalized,), config, (normalized,))
    fundamental_history = pd.DataFrame()
    fundamental_report = pd.DataFrame()
    if fundamentals is not None and not fundamentals.empty:
        try:
            fundamentals, fundamental_history, fundamental_report = enrich_fundamental_shortlist(
                fundamentals,
                (normalized,),
                pd.DataFrame(),
                config,
                bool(twelve_data_api_key),
                twelve_data_api_key,
            )
        except Exception as exc:
            audit_warnings.append(f"Fundamental history enrichment gagal: {exc}")

    automatic_pm = pd.DataFrame()
    automatic_forward_report = pd.DataFrame()
    if fundamentals is not None and not fundamentals.empty and bool(getattr(config, "automatic_forward_quality_enabled", True)):
        try:
            automatic_pm, automatic_forward_report = database_first_forward_quality(
                fundamentals, (normalized,), config,
            )
        except Exception as exc:
            automatic_pm, automatic_forward_report = pd.DataFrame(), pd.DataFrame()
            audit_warnings.append(f"Forward project/management collection gagal: {exc}")

    signals = attach_fundamentals(signals, fundamentals)
    signals = apply_fundamental_gate(signals, config)
    market_status = cached_automatic_market_status((normalized,))
    news_review = cached_automatic_news((normalized,), int(getattr(config, "min_news_lookback_days", 30)))
    signals = apply_market_status_gate(signals, market_status, config)
    signals = apply_news_gate(signals, news_review, config)

    snapshots = cached_execution_snapshots((normalized,))
    signals = apply_execution_snapshot_gate(signals, snapshots, config)
    source_tiers = getattr(download_report, "source_tiers", {}) or {}
    independent_data = pd.DataFrame()
    independent_report = pd.DataFrame()
    try:
        last_date = pd.Timestamp(history.index[-1]).strftime("%Y-%m-%d")
        last_close = float(pd.to_numeric(history["Close"], errors="coerce").dropna().iloc[-1])
        independent_data, independent_report = fetch_automatic_independent_prices(
            (normalized,),
            reference_date=last_date,
            twelve_data_api_key=twelve_data_api_key,
            itick_api_token=itick_api_token,
            primary_reference={normalized: (last_date, last_close)},
            primary_source_tiers={normalized: str(source_tiers.get(normalized, "UNKNOWN"))},
            config=config,
        )
    except Exception as exc:
        independent_data, independent_report = pd.DataFrame(), pd.DataFrame()
        audit_warnings.append(f"Harga independen otomatis gagal: {exc}")
    price_validation = build_independent_price_validation(
        histories, independent_data, config, primary_source_tiers=source_tiers,
    )
    signals = apply_independent_price_gate(signals, price_validation, config)
    signals = attach_position_sizing(signals, config)
    signals = apply_analyst_fusion_gate(signals, config)
    signals = finalize_execution_integrity(signals, config)
    signals = sort_signals(signals)

    tc_config = TimeCycleConfig(
        min_bars=int(getattr(config, "time_cycle_min_history_bars", 260)),
        lunar_enabled=bool(getattr(config, "time_cycle_lunar_enabled", True)),
        eoff_enabled=bool(getattr(config, "eoff_enabled", True)),
        eoff_ephemeris_enabled=bool(getattr(config, "eoff_ephemeris_enabled", True)),
        eoff_min_fib_cluster=int(getattr(config, "eoff_min_fib_cluster", 4)),
        eoff_min_unique_anchors=int(getattr(config, "eoff_min_unique_anchors", 3)),
        eoff_max_dominant_anchor_share=float(getattr(config, "eoff_max_dominant_anchor_share", 0.55)),
        eoff_aspect_orb_deg=float(getattr(config, "eoff_aspect_orb_deg", 3.0)),
        eoff_require_astro_fib_confluence=bool(getattr(config, "eoff_require_astro_fib_confluence", True)),
        idx_trading_holidays=tuple(getattr(config, "idx_trading_holidays", ()) or ()),
        idx_official_open_dates=tuple(getattr(config, "idx_official_open_dates", ()) or ()),
        idx_official_closed_dates=tuple(getattr(config, "idx_official_closed_dates", ()) or ()),
        require_official_idx_calendar=bool(getattr(config, "require_official_idx_calendar", False)),
    )
    signals = enrich_core_signals_with_time_cycle(
        signals,
        mini.get("prepared", {}),
        enabled=bool(getattr(config, "time_cycle_enabled", True)),
        max_weight=float(getattr(config, "time_cycle_core_max_weight", 0.04)),
        min_confidence=float(getattr(config, "time_cycle_min_confidence", 55.0)),
        config=tc_config,
    )
    time_cycle = analyze_time_cycle(history, config=tc_config)

    try:
        narrative_bridge = ScannerDatabaseBridge()
        persisted_narrative_events = narrative_bridge.read_narrative_events(
            (normalized,),
        )
        persisted_narrative_outcomes = (
            narrative_bridge.read_narrative_event_outcomes((normalized,))
        )
    except Exception:
        persisted_narrative_events = pd.DataFrame()
        persisted_narrative_outcomes = pd.DataFrame()
    try:
        reference_fx, reference_fx_report = cached_reference_fx()
    except Exception as exc:
        reference_fx, reference_fx_report = pd.DataFrame(), pd.DataFrame([{
            'provider': 'REFERENCE_FX', 'status': 'ERROR',
            'error': f'{type(exc).__name__}: {str(exc)[:160]}',
        }])
        audit_warnings.append(f"Reference FX gagal: {exc}")
    try:
        focus = build_focus_screens(
            mini.get("prepared", {}),
            fundamentals=fundamentals,
            core_signals=signals,
            project_management=automatic_pm,
            news_review=news_review,
            market_status=market_status,
            narrative_events=persisted_narrative_events,
            narrative_outcomes=persisted_narrative_outcomes,
            benchmark=benchmark,
            config=config,
            validation_events=pd.DataFrame(),
            ai_memory=pd.DataFrame(),
            reference_fx=reference_fx,
        )
        focus['reference_fx'] = reference_fx
        focus['reference_fx_report'] = reference_fx_report
    except Exception as exc:
        focus = {"multibagger": pd.DataFrame(), "core_swing": pd.DataFrame(), "profit_order_builder": pd.DataFrame()}
        audit_warnings.append(f"Focus ranking gagal: {exc}")

    detail_result = {
        "mode": "single_ticker",
        "ticker": normalized,
        "signals": signals,
        "prepared": mini.get("prepared", {}),
        "all_histories": histories,
        "focus_screens": focus,
        "fundamentals": fundamentals,
    }
    ranking = build_top20_ranking(detail_result, limit=5)
    if not ranking.empty:
        row = ranking.iloc[0].to_dict()
        summary = {
            "decision": row.get("decision"),
            "score": row.get("combined_score"),
            "best_buy_date": row.get("best_buy_date"),
            "entry_low": row.get("entry_low"),
            "entry_high": row.get("entry_high"),
            "entry_zone_state": row.get("entry_zone_state"),
            "execution_price": row.get("execution_price"),
            "trigger": row.get("trigger"),
            "confirmation_level": row.get("confirmation_level"),
            "stop_loss": row.get("stop_loss"),
            "tp1": row.get("tp1"),
            "tp2": row.get("tp2"),
            "time_cycle_score": row.get("time_cycle_score"),
            "reason": row.get("reason"),
        }
    else:
        summary = {
            "decision": time_cycle.get("quick_buy_action", "WAIT"),
            "score": time_cycle.get("best_buy_score", 0.0),
            "best_buy_date": time_cycle.get("best_buy_date", ""),
            "entry_low": time_cycle.get("best_buy_entry_low"),
            "entry_high": time_cycle.get("best_buy_entry_high"),
            "entry_zone_state": "RANGE",
            "execution_price": time_cycle.get("best_buy_trigger"),
            "trigger": time_cycle.get("best_buy_trigger"),
            "confirmation_level": time_cycle.get("best_buy_trigger"),
            "stop_loss": time_cycle.get("best_buy_stop_loss"),
            "tp1": time_cycle.get("best_buy_tp1"),
            "tp2": time_cycle.get("best_buy_tp2"),
            "time_cycle_score": time_cycle.get("time_cycle_score"),
            "reason": time_cycle.get("best_buy_reason"),
        }
    multibagger = focus.get("multibagger", pd.DataFrame())
    if isinstance(multibagger, pd.DataFrame) and not multibagger.empty:
        mb_row = multibagger.iloc[0]
        summary["multibagger_score"] = _finite_or_default(mb_row.get("multibagger_score"), np.nan)
        summary["multibagger_quality_score"] = _finite_or_default(mb_row.get("multibagger_quality_score", mb_row.get("multibagger_score")), np.nan)
        summary["execution_readiness_score"] = _finite_or_default(mb_row.get("execution_readiness_score"), np.nan)
        summary["multibagger_candidate_type"] = mb_row.get("multibagger_candidate_type", "")
        summary["research_recommendation_status"] = mb_row.get("research_recommendation_status", "WAIT")
        summary["capital_conviction_score"] = _finite_or_default(mb_row.get("capital_conviction_score"), np.nan)
        summary["multibagger_scoring_state"] = mb_row.get("multibagger_scoring_state", "SCORED")
        summary["multibagger_score_reason"] = mb_row.get("multibagger_score_reason", "")
    else:
        summary["multibagger_score"] = np.nan
        summary["multibagger_quality_score"] = np.nan
        summary["execution_readiness_score"] = np.nan
        summary["multibagger_candidate_type"] = ""
        summary["research_recommendation_status"] = "WAIT"
        summary["capital_conviction_score"] = np.nan
        summary["multibagger_scoring_state"] = "DATA_NOT_SCORED"
        summary["multibagger_score_reason"] = "Baris Multibagger tidak terbentuk."
    return {
        "ticker": normalized,
        "summary": summary,
        "signals": signals,
        "multibagger": multibagger,
        "time_cycle": time_cycle,
        "history": history,
        "fundamentals": fundamentals,
        "fundamental_history": fundamental_history,
        "fundamental_report": fundamental_report,
        "database_fundamental_audit": database_fundamental_audit,
        "forward_report": automatic_forward_report,
        "independent_report": independent_report,
        "audit_warnings": audit_warnings,
    }


def _finite_or_default(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


st.title("IDX Wealth Scanner — Multibagger, Core Swing & Arah IHSG v7.16.3")
st.caption(
    "Quality-first scanner dengan IHSG regime/risk overlay. Model boleh ABSTAIN; tanpa outcome "
    "teresolusi dan skill OOS, prediksi tidak boleh menjadi sinyal produksi atau menaikkan ranking."
)

with st.sidebar:
    twelve_data_api_key = configured_twelve_data_key()
    itick_api_token = configured_itick_token()
    st.header("Execution policy")
    execution_policy = st.radio(
        "Mode keputusan",
        ["SIGNAL_FIRST", "ACCOUNT_GUARDED"],
        index=0,
        horizontal=True,
        help=(
            "SIGNAL_FIRST menerbitkan SIGNAL_READY tanpa menyebutnya siap eksekusi; risiko tetap terlihat. "
            "ACCOUNT_GUARDED menambahkan verifikasi harga, sizing, cash, slot posisi, dan portfolio heat sebelum ORDER_READY."
        ),
    )
    if execution_policy == "SIGNAL_FIRST":
        st.info("Signal-First aktif: SIGNAL_READY adalah radar, bukan instruksi beli. READY_FOR_STOCKBIT_VERIFY berarti semua gate non-akun lolos dan harga/spread/gap masih wajib dicek di Stockbit.")
    else:
        st.warning("Account-Guarded aktif: ORDER_READY dapat diblokir oleh cash, sizing, regime, likuiditas, RR, dan portfolio heat.")
    real_money_mode = True
    with st.expander("Akun dan risiko — informasional / Account-Guarded", expanded=execution_policy == "ACCOUNT_GUARDED"):
        account_size = st.number_input("Equity akun (Rp)", 1_000_000, 10_000_000_000, 5_000_000, 500_000)
        cash_on_hand = st.number_input("Cash on hand (Rp)", 0, 10_000_000_000, 5_000_000, 100_000)
        risk_per_trade = st.slider("Risiko maksimum per transaksi", 0.25, 1.00, 0.50, 0.05) / 100
        max_positions = st.number_input("Maksimum posisi bersamaan", 1, 10, 3, 1)
        max_position_pct = st.slider("Maksimum modal per saham", 10, 50, 35, 5) / 100
        current_positions_manual = st.number_input("Posisi yang sedang terbuka", 0, 20, 0, 1)
        current_invested_manual = float(st.number_input("Modal sedang terpakai (Rp)", 0, 10_000_000_000, 0, 100_000))
        current_open_risk_manual = float(st.number_input("Open risk saat ini (Rp)", 0, 1_000_000_000, 0, 10_000))
        portfolio_equity_mode = st.selectbox(
            "Dasar bobot portfolio",
            ["Gunakan Equity akun", "Estimasi nilai posisi + cash"],
            index=0,
            help="Jika portfolio CSV diunggah, nilai posisi aktual menggantikan input manual.",
        )
    period = st.selectbox("Riwayat OHLCV", ["5y", "3y", "2y"], index=0)
    min_adtv_b = st.number_input("Minimum ADTV20 (Rp miliar)", 0.1, 100.0, 2.0, 0.5)
    min_score = st.slider("Minimum quality score", 50, 90, 72)
    execution_score = st.slider("Execution-ready technical score", min_score, 95, max(82, min_score))
    min_execution_confidence = st.slider("Minimum total execution confidence", 75, 95, 82)
    min_data_completeness = st.slider(
        "Minimum data completeness", 70, 95, 80,
        help="BUY_LIMIT tidak akan diterbitkan bila evidence coverage di bawah batas ini.",
    )
    max_stop_pct = st.slider("Maksimum jarak SL", 3.0, 10.0, 7.0, 0.5) / 100
    min_rr2 = st.slider("Minimum RR ke TP2", 2.0, 4.0, 2.7, 0.1)
    validate = st.checkbox(
        "Chronological OOS validation",
        value=True,
        help="Dipakai pada scanner universe; tidak diwajibkan untuk portfolio-only review.",
    )
    validation_max_tickers = st.slider(
        "Maksimum ticker untuk OOS per scan",
        10,
        400,
        400,
        10,
        disabled=not validate,
        help=(
            "Sampel deterministik berstrata likuiditas—bukan berdasarkan score. Default 400 "
            "memvalidasi seluruh universe; hasil yang belum "
            "memenuhi minimum event tetap fail-closed dan bobot AI menjadi nol."
        ),
    )
    use_fundamentals = st.checkbox(
        "Ambil fundamental kandidat dan portfolio",
        value=True,
        help="Real-money BUY_LIMIT memerlukan fundamental coverage minimal 45%; bila dimatikan, kandidat tetap menjadi watchlist/PENDING_DATA.",
    )
    multibagger_full_universe = st.checkbox(
        "Multibagger: fundamental seluruh universe",
        value=True,
        disabled=not use_fundamentals,
        help=(
            "Mode riset penuh. Untuk universe sampai 400 ticker, snapshot dan statement fallback "
            "diupayakan selesai pada scan yang sama; IDX official dipakai sebagai cross-check bounded. "
            "Semua hasil sukses disimpan sehingga retry hanya mengenai ticker yang belum lengkap."
        ),
    )
    fundamental_n = st.slider("Jumlah kandidat fundamental", 10, 400, 120, disabled=not use_fundamentals or multibagger_full_universe)
    fundamental_history_n = st.slider(
        "Histori laporan untuk shortlist Multibagger", 10, 400, 400, 10,
        disabled=not use_fundamentals,
        help="Dalam mode full-universe sampai 400 ticker, Yahoo/direct fallback mengisi semua issuer; IDX official tetap menjadi cross-check bounded.",
    )
    idx_fundamental_n = st.slider(
        "IDX/XBRL otomatis (maksimum emiten)", 5, 120, 80, 5,
        disabled=not use_fundamentals,
        help=(
            "Tanpa upload dan tanpa API key. Scanner mencari serta memparse filing XBRL resmi untuk shortlist. "
            "Endpoint halaman publik IDX dapat berubah, sehingga kegagalan akan fail-soft dan terlihat di audit provider."
        ),
    )
    enable_twelve_fundamentals = st.checkbox(
        "Twelve Data: laporan fundamental kedua",
        value=bool(twelve_data_api_key),
        disabled=(not use_fundamentals) or (not bool(twelve_data_api_key)),
        help="Memerlukan TWELVE_DATA_API_KEY dan paket yang mencakup endpoint income statement, balance sheet, serta cash flow.",
    )
    with st.expander("Compounding Multibagger — Capital Allocation", expanded=True):
        multibagger_base_capital = float(st.number_input("Modal pokok bucket Multibagger (Rp)", 0, 10_000_000_000, 0, 100_000))
        realized_profit_to_compound = float(st.number_input("Profit trading terealisasi untuk dipindahkan (Rp)", 0, 10_000_000_000, 0, 50_000))
        profit_allocation_pct = st.slider("Porsi profit ke bucket Multibagger", 0, 100, 100, 5) / 100
        multibagger_max_holdings = st.slider("Maksimum saham Multibagger inti", 2, 10, 5, 1)
        multibagger_min_conviction = st.slider("Minimum capital conviction", 60, 90, 72, 1)
        effective_multibagger_budget = multibagger_base_capital + realized_profit_to_compound * profit_allocation_pct
        st.caption(
            f"Budget efektif: {rupiah(effective_multibagger_budget)}. Dana terbesar diarahkan ke conviction tertinggi; "
            "bagian kandidat yang belum masuk zona entry tetap menjadi cash reserve."
        )
    with st.expander("AI Lokal Gratis — Hybrid Learning", expanded=True):
        ai_enabled = st.checkbox(
            "Aktifkan AI lokal", value=True,
            help="Tidak memakai API berbayar. Model belajar dari chronological walk-forward dan outcome memory scanner.",
        )
        ai_mode = st.selectbox(
            "Mode AI", ["HYBRID_GUARDED", "SHADOW_ONLY", "RULE_ONLY"], index=0,
            help=(
                "HYBRID_GUARDED mengoreksi ranking maksimal sesuai confidence data. "
                "SHADOW_ONLY hanya menampilkan prediksi AI tanpa mengubah urutan. RULE_ONLY mematikan AI."
            ),
        )
        ai_max_weight_pct = st.slider(
            "Bobot maksimum AI dalam conviction", 5, 35, 35, 5,
            disabled=not ai_enabled or ai_mode == "RULE_ONLY",
            help="Bobot aktual menjadi nol tanpa outcome, tanpa dukungan strategi, jika coverage fitur rendah, atau bila model tidak mengalahkan baseline OOS.",
        )
        ai_min_training_events = st.slider(
            "Minimum event untuk model statistik", 20, 100, 30, 5,
            disabled=not ai_enabled or ai_mode == "RULE_ONLY",
        )
        ai_memory_uploaded = st.file_uploader(
            "Import AI outcome memory (opsional)", type=["csv"], key="ai_memory_upload",
            help=(
                "Streamlit gratis dapat menghapus disk saat sleep/redeploy. Export memory dari Order Builder dan import kembali "
                "agar pembelajaran lintas sesi tidak hilang."
            ),
        )
        st.caption(
            "AI = validated regularized logistic untuk ranking; similarity/KNN + Bayesian prior sebagai diagnostik/shadow; chronological evaluation + drift/coverage guard. "
            "No evidence = no influence; rule engine tetap memegang invalidasi struktur, harga, dan data."
        )
    with st.expander("Time-Cycle Intelligence — Swing/Core & Multibagger", expanded=True):
        time_cycle_enabled = st.checkbox(
            "Aktifkan objective Astronacci-style time cycle", value=True,
            help=(
                "Menerapkan swing timing, Fibonacci time, autocorrelation, spectral cycle, dan moon phase tervalidasi. "
                "Hanya memengaruhi Core Swing dan Multibagger."
            ),
        )
        time_cycle_core_weight_pct = st.slider(
            "Bobot maksimum time-cycle/EOFF pada ranking core", 0, 5, 4, 1, disabled=not time_cycle_enabled,
        )
        time_cycle_multibagger_weight_pct = 0
        multibagger_time_cycle_full_refresh_enabled = st.checkbox(
            "Hitung full time-cycle/EOFF untuk Multibagger (shadow, lambat)",
            value=False,
            disabled=not time_cycle_enabled,
            help=(
                "Default mati karena bobot produksi Multibagger 0% dan evaluasi "
                "400 ticker sangat mahal. Aktifkan hanya untuk riset shadow; "
                "hasilnya tidak membuka allocation gate."
            ),
        )
        st.caption(
            "Bobot time-cycle pada Multibagger quality/capital conviction dikunci 0%. "
            "Core Swing tetap memakai overlay bounded; full refresh Multibagger "
            "bersifat opt-in dan research-only."
        )
        time_cycle_min_confidence = st.slider(
            "Minimum confidence cycle agar berpengaruh", 45, 75, 55, 5, disabled=not time_cycle_enabled,
        )
        time_cycle_lunar_enabled = st.checkbox(
            "Gunakan Full/New Moon sebagai time marker eksperimental", value=True, disabled=not time_cycle_enabled,
            help="Moon phase tidak menentukan arah. Bobotnya hanya aktif bila kedekatan pivot historis mengalahkan baseline.",
        )
        st.markdown("**Clean-room Eye-of-Future reconstruction**")
        eoff_enabled = st.checkbox(
            "Aktifkan EOFF reconstruction lengkap", value=True, disabled=not time_cycle_enabled,
            help=(
                "Menggabungkan ≥4 multi-anchor Fibonacci time projections, objective cycle, price, pattern, dan momentum. Ephemeris tetap tersedia sebagai shadow diagnostic."
            ),
        )
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            eoff_min_fib_cluster = st.slider(
                "Minimum Fibonacci projection cluster", 3, 8, 4, 1,
                disabled=not time_cycle_enabled or not eoff_enabled,
            )
        with ec2:
            eoff_aspect_orb_deg = st.slider(
                "Planetary aspect orb (derajat)", 1.0, 6.0, 3.0, 0.5,
                disabled=not time_cycle_enabled or not eoff_enabled,
            )
        with ec3:
            eoff_require_astro_fib_confluence = st.checkbox(
                "Wajib astro + Fibonacci cluster (legacy research)", value=False,
                disabled=not time_cycle_enabled or not eoff_enabled,
                help="Default OFF. Astro tetap dicatat sebagai shadow diagnostic dengan bobot produksi 0%; arah berasal dari Fibonacci, cycle, price, pattern, momentum, dan forward validation.",
            )
        eoff_ephemeris_enabled = st.checkbox(
            "Aktifkan ephemeris geosentris offline", value=True,
            disabled=not time_cycle_enabled or not eoff_enabled,
            help="Menggunakan PyEphem lokal; tidak membutuhkan API atau file ephemeris eksternal.",
        )
        st.caption(
            "Implementasi clean-room ini tidak mengklaim formula proprietary asli. Mulai v7.1.0, astro tidak memiliki bobot produksi. EOFF memengaruhi timing hanya setelah cluster waktu, arah harga, dan chronological forward evidence lolos."
        )
    st.subheader("Data otomatis")
    st.caption(
        "OHLCV gratis: cache current → Yahoo → IDX Stock Summary EOD → iTick free bila token tersedia. "
        "Harga kedua memakai cache sama-session → IDX → Google Finance → iTick → Twelve Data opsional. "
        "Fundamental shortlist: IDX/XBRL official-first + Yahoo fallback/cross-check; upload hanya fallback."
    )
    if itick_api_token:
        st.success("Fallback OHLCV iTick free terkonfigurasi; rate guard internal maksimum 4 call/menit.")
    else:
        st.caption("iTick belum dikonfigurasi. Scanner tetap berjalan gratis dengan cache, Yahoo, dan IDX resmi.")
    if twelve_data_api_key:
        st.success("Twelve Data terkonfigurasi untuk harga independen dan, bila paket mendukung, laporan fundamental kedua.")
    else:
        st.caption("Twelve Data tidak dikonfigurasi; IDX dan Google tetap berjalan tanpa API key.")
    st.divider()
    st.caption("Semua setup tetap ditampilkan. Core plan verified membawa lot/template, tetapi submit Stockbit tetap manual setelah revalidasi broker.")

sample_csv = b"ticker\nADRO\nANTM\nBRMS\nMDKA\nTAPG\n"
portfolio_sample_csv = (
    b"ticker,lots,avg_price,stop_loss,take_profit,notes\n"
    b"ADRO,10,2150,,,Core position\n"
    b"ANTM,5,1860,,,Trading position\n"
)
fundamental_history_sample_csv = (
    b"ticker,period_end,period_type,statement_basis,source_url,currency,unit_multiplier,shares_multiplier,revenue,net_income,operating_cash_flow,capex,total_assets,total_liabilities,equity,total_debt,cash,shares_outstanding,operating_income,ebit,ebitda,interest_expense,car,npl_gross,ldr\n"
    b"ANTM,2025-03-31,Q1,YTD_CUMULATIVE,https://www.idx.co.id/id/perusahaan-tercatat/laporan-keuangan-dan-tahunan,IDR,1000000,1,1000,100,120,20,5000,2000,3000,800,400,24000000000,150,150,180,20,,,\n"
    b"ANTM,2025-06-30,Q2,YTD_CUMULATIVE,https://www.idx.co.id/id/perusahaan-tercatat/laporan-keuangan-dan-tahunan,IDR,1000000,1,2200,230,260,45,5300,2100,3200,780,470,24000000000,330,330,390,42,,,\n"
)
project_management_sample_csv = (
    b"ticker,as_of,source_url,source_verified,project_name,project_stage,project_completion_pct,project_capex_idr,funding_secured_pct,offtake_secured_pct,expected_revenue_idr,expected_ebitda_idr,expected_cod,ownership_pct,project_delay_months,cost_overrun_pct,strategic_project,project_risk,ceo_name,ceo_tenure_years,board_avg_tenure_years,management_revenue_cagr,management_roic_change_pct,capital_allocation_score,governance_score,board_turnover_3y,insider_ownership_pct,audit_clean,related_party_risk,legal_governance_flags,management_source_url,management_verified\n"
    b"ANTM,2026-07-01,https://www.idx.co.id/,TRUE,Smelter Expansion,CONSTRUCTION,65,2500000000000,100,70,1800000000000,350000000000,2027-06-30,100,0,3,TRUE,MEDIUM,Example CEO,4,5,0.12,0.03,82,80,1,0.10,TRUE,LOW,,https://www.idx.co.id/,TRUE\n"
)
narrative_event_sample_csv = (
    b"ticker,event_date,detected_at,event_type,headline,summary,source_url,source_family,official_verified,official_domain,materiality_score,impact_direction,financial_bridge_score,event_status,resolved_at,resolution_source_url,supersedes_event_id,entity_match_state\n"
    b"ANTM,2026-07-20,2026-07-20T18:00:00+07:00,CAPACITY_OR_EXPANSION,Commissioning smelter memasuki tahap akhir,Target operasi dan kapasitas harus diverifikasi pada keterbukaan resmi,https://www.idx.co.id/id/perusahaan-tercatat/keterbukaan-informasi/,IDX_DISCLOSURE,TRUE,idx.co.id,85,POSITIVE,82,ACTIVE,,,,ENTITY_VERIFIED\n"
)
left, right = st.columns([3, 1])
with left:
    uploaded = st.file_uploader(
        "Upload CSV universe ticker IDX",
        type=["csv", "txt"],
        help="Diperlukan hanya untuk menjalankan scanner universe.",
    )
with right:
    st.write("")
    st.write("")
    st.download_button("Unduh contoh universe", sample_csv, "sample_tickers.csv", "text/csv", width="stretch")

p1, p2 = st.columns([3, 1])
with p1:
    portfolio_uploaded = st.file_uploader(
        "Upload snapshot portfolio Stockbit CSV",
        type=["csv", "txt"],
        help="Kolom wajib: ticker, lots, avg_price. Dapat dianalisis tanpa upload universe ticker.",
        key="portfolio_upload",
    )
with p2:
    st.write("")
    st.write("")
    st.download_button(
        "Unduh template portfolio", portfolio_sample_csv, "stockbit_portfolio_template.csv", "text/csv", width="stretch"
    )

fundamental_history_uploaded = None
with st.expander("Fallback manual laporan fundamental — biasanya tidak diperlukan"):
    st.caption(
        "Default v7.16.3 mengambil IDX/XBRL official-first, lalu Yahoo timeseries direct bounded sebagai fallback/cross-check. Gunakan CSV hanya bila endpoint IDX sedang berubah/terblokir, "
        "emiten tidak menyediakan XBRL yang dapat diparse, atau Anda ingin menambah hasil rekonsiliasi sendiri."
    )
    f1, f2 = st.columns([3, 1])
    with f1:
        fundamental_history_uploaded = st.file_uploader(
            "Upload histori laporan IDX/XBRL (fallback opsional)",
            type=["csv", "txt"],
            key="fundamental_history_upload",
            help=(
                "Gunakan minimal 8 kuartal. source_url harus HTTPS pada idx.co.id agar provenance dikenali. "
                "Pilih YTD_CUMULATIVE untuk laporan interim kumulatif; scanner mengubahnya menjadi kuartal mandiri."
            ),
        )
    with f2:
        st.write("")
        st.write("")
        st.download_button(
            "Template fallback", fundamental_history_sample_csv,
            "idx_fundamental_history_template.csv", "text/csv", width="stretch",
        )
    st.caption(
        "Upload tidak otomatis dianggap benar hanya karena diberi label IDX: domain sumber, identitas akuntansi, "
        "staleness, dan perbedaan antar-provider tetap diperiksa."
    )

with st.expander("Forward Intelligence Otomatis — Project, Management & Future Fundamental", expanded=False):
    st.caption(
        "Default: scanner mencari dokumen IDX/OJK dan investor-relations emiten, mengekstrak proyek serta manajemen, "
        "memeriksa quorum sumber, lalu menghitung skenario dampak ke revenue, EBITDA, laba, FCF, dan utang. "
        "Upload manual hanya override opsional bila Anda memiliki bukti yang lebih lengkap."
    )
    automatic_forward_quality_enabled = st.checkbox(
        "Aktifkan pencarian project & management otomatis", value=True,
        help="Hanya kandidat fundamental teratas yang dicari. Hasil disimpan cache agar scan berikutnya lebih cepat."
    )
    af1, af2, af3 = st.columns(3)
    with af1:
        automatic_forward_quality_top_n = st.slider("Emiten forward review", 5, 80, 40, 5)
    with af2:
        automatic_forward_quality_cache_days = st.slider("Cache review (hari)", 3, 30, 14, 1)
    with af3:
        automatic_forward_quality_max_documents = st.slider("Dokumen/emiten", 2, 8, 5, 1)
    pm1, pm2 = st.columns([3, 1])
    with pm1:
        project_management_uploaded = st.file_uploader(
            "Project & Management Review CSV (opsional)",
            type=["csv", "txt"],
            key="project_management_upload",
            help=(
                "Satu ticker boleh memiliki beberapa baris proyek. Field utama: project_stage, completion, funding, offtake, delay, cost overrun, "
                "CEO/board tenure, revenue CAGR/ROIC di bawah manajemen, capital allocation, governance, dan source verification."
            ),
        )
    with pm2:
        st.write("")
        st.write("")
        st.download_button(
            "Template Project/Management", project_management_sample_csv,
            "project_management_review_template.csv", "text/csv", width="stretch",
        )

narrative_event_uploaded = None
with st.expander("Narrative Event Database — point-in-time evidence", expanded=False):
    st.caption(
        "Scanner otomatis membentuk event dari berita, keterbukaan, laporan, dan "
        "project review. CSV ini opsional untuk menambah event resmi yang belum "
        "tertangkap. detected_at wajib mencerminkan kapan informasi pertama kali "
        "tersedia bagi riset—bukan tanggal yang diisi mundur setelah harga bergerak."
    )
    ne1, ne2 = st.columns([3, 1])
    with ne1:
        narrative_event_uploaded = st.file_uploader(
            "Narrative Event CSV (opsional)",
            type=["csv", "txt"],
            key="narrative_event_upload",
            help=(
                "Kolom wajib: ticker, event_date, headline, source_url. "
                "Tambahkan detected_at, event_type, official_verified, "
                "official_domain, impact_direction, materiality_score, "
                "financial_bridge_score, event_status, dan resolution_source_url. "
                "Status RESOLVED/SUPERSEDED/REVERSED tanpa sumber resolusi HTTPS "
                "akan tetap dianggap ACTIVE."
            ),
        )
    with ne2:
        st.write("")
        st.write("")
        st.download_button(
            "Template Narrative Event",
            narrative_event_sample_csv,
            "narrative_event_template.csv",
            "text/csv",
            width="stretch",
        )

broksum_sample_csv = (
    b"ticker,date,broker_code,buy_value,sell_value\n"
    b"ANTM,2026-07-15,YP,15000000000,5000000000\n"
    b"ANTM,2026-07-15,CC,7000000000,9000000000\n"
)
with st.expander("Konfirmasi opsional: Broker Summary"):
    st.caption(
        "Tidak wajib. Broker Summary aktual dapat memperkuat validasi akumulasi Core Swing dan Multibagger, "
        "tetapi upload manual diklasifikasikan observed/unverified dan tidak membuka direct-rank atau allocation. "
        "Broker code juga bukan identitas beneficial owner."
    )
    broksum_uploaded = st.file_uploader(
        "Broker Summary CSV (opsional)", type=["csv", "txt"], key="broksum_upload",
        help="Kolom: ticker, date, broker_code, dan buy_value/sell_value atau buy_volume/sell_volume.",
    )
    st.download_button("Template Broker Summary", broksum_sample_csv, "broker_summary_template.csv", "text/csv")

now_jkt_ui = pd.Timestamp.now(tz="Asia/Jakarta")
if idx_regular_decision_window(now_jkt_ui):
    st.warning(
        "Jam reguler IDX masih berada dalam window candle berjalan. Kandidat teknikal ditampilkan sebagai "
        "PENDING_CLOSE dan perlu di-refresh setelah 16:20 WIB."
    )
elif now_jkt_ui.weekday() < 5 and not idx_daily_bar_is_final(now_jkt_ui):
    st.success(
        "Mode PRE-MARKET: scanner memakai completed EOD hari bursa sebelumnya. Kandidat yang lulus seluruh gate "
        "adalah PLAN_VERIFIED; harga pembukaan dan spread Stockbit tetap harus diperiksa setelah pasar mulai."
    )

st.info(
    "Begitu CSV universe diunggah, scan pertama dimulai otomatis. Tombol refresh dipakai untuk mengambil snapshot "
    "terbaru. Analisis portfolio saja tetap tersedia tanpa universe ticker."
)

b1, b2 = st.columns(2)
manual_refresh = b1.button("Scan ulang / refresh data", type="primary", width="stretch", disabled=uploaded is None)
run_portfolio = b2.button(
    "Analisis portfolio saja", type="secondary", width="stretch", disabled=portfolio_uploaded is None
)
scan_signature = upload_fingerprint(
    uploaded, portfolio_uploaded, fundamental_history_uploaded, project_management_uploaded,
    narrative_event_uploaded, broksum_uploaded, ai_memory_uploaded,
) if uploaded is not None else ""
new_upload = bool(
    uploaded is not None
    and st.session_state.get("_last_auto_scan_signature") != scan_signature
)
run_scan = bool(uploaded is not None and (new_upload or manual_refresh) and not run_portfolio)
if new_upload and not run_portfolio:
    st.caption("CSV baru terdeteksi—scanner dimulai otomatis.")

cfg = ScanConfig().replace(
    min_adtv_idr=float(min_adtv_b) * 1_000_000_000,
    min_score=float(min_score),
    execution_score=float(execution_score),
    min_execution_confidence=float(min_execution_confidence),
    min_data_completeness=float(min_data_completeness),
    validation_max_tickers=int(validation_max_tickers),
    max_stop_pct=float(max_stop_pct),
    min_rr2=float(min_rr2),
    fundamental_top_n=int(fundamental_n),
    fundamental_history_top_n=int(fundamental_history_n),
    idx_fundamental_top_n=min(int(idx_fundamental_n), int(fundamental_history_n)),
    idx_fundamental_years_back=3,
    twelve_fundamental_top_n=min(20, int(fundamental_history_n)),
    real_money_mode=bool(real_money_mode),
    require_fundamentals=bool(use_fundamentals),
    require_market_status=False,
    require_news_review=False,
    require_validation=False,
    require_independent_price_verification=True,
    execution_policy=str(execution_policy),
    autopilot_enabled=bool(execution_policy == "ACCOUNT_GUARDED"),
    allow_autopilot_risk_off=False,
    account_size_idr=float(account_size),
    cash_on_hand_idr=float(cash_on_hand),
    risk_per_trade_pct=float(risk_per_trade),
    max_positions=int(max_positions),
    max_position_pct=float(max_position_pct),
    multibagger_profit_allocation_pct=float(profit_allocation_pct),
    multibagger_capital_budget_idr=float(effective_multibagger_budget),
    multibagger_max_holdings=int(multibagger_max_holdings),
    multibagger_min_capital_conviction=float(multibagger_min_conviction),
    automatic_forward_quality_enabled=bool(automatic_forward_quality_enabled),
    automatic_forward_quality_top_n=int(automatic_forward_quality_top_n),
    automatic_forward_quality_cache_days=int(automatic_forward_quality_cache_days),
    automatic_forward_quality_max_documents=int(automatic_forward_quality_max_documents),
    ai_enabled=bool(ai_enabled),
    ai_mode=str(ai_mode),
    ai_max_weight=float(ai_max_weight_pct) / 100.0,
    ai_min_training_events=int(ai_min_training_events),
    time_cycle_enabled=bool(time_cycle_enabled),
    time_cycle_core_max_weight=float(time_cycle_core_weight_pct) / 100.0,
    time_cycle_multibagger_max_weight=float(time_cycle_multibagger_weight_pct) / 100.0,
    multibagger_time_cycle_full_refresh_enabled=bool(
        multibagger_time_cycle_full_refresh_enabled
    ),
    time_cycle_min_confidence=float(time_cycle_min_confidence),
    time_cycle_lunar_enabled=bool(time_cycle_lunar_enabled),
    eoff_enabled=bool(eoff_enabled),
    eoff_ephemeris_enabled=bool(eoff_ephemeris_enabled),
    eoff_min_fib_cluster=int(eoff_min_fib_cluster),
    eoff_aspect_orb_deg=float(eoff_aspect_orb_deg),
    eoff_require_astro_fib_confluence=bool(eoff_require_astro_fib_confluence),
)
portfolio_equity_input = float(account_size) if portfolio_equity_mode == "Gunakan Equity akun" else None

uploaded_project_management = pd.DataFrame()
if project_management_uploaded is not None:
    try:
        uploaded_project_management = parse_project_management_csv(project_management_uploaded)
        st.success(
            f"Project/management review terbaca: {len(uploaded_project_management)} baris, "
            f"{uploaded_project_management['ticker'].nunique()} emiten."
        )
    except Exception as exc:
        st.error(f"Project/management CSV tidak dapat dibaca: {exc}")
        st.stop()

uploaded_narrative_events = pd.DataFrame()
if narrative_event_uploaded is not None:
    try:
        uploaded_narrative_events = parse_narrative_event_csv(
            narrative_event_uploaded,
        )
        st.success(
            f"Narrative event terbaca: {len(uploaded_narrative_events)} event, "
            f"{uploaded_narrative_events['ticker'].nunique()} emiten."
        )
    except Exception as exc:
        st.error(f"Narrative Event CSV tidak dapat dibaca: {exc}")
        st.stop()

uploaded_fundamental_history = pd.DataFrame()
if fundamental_history_uploaded is not None:
    try:
        uploaded_fundamental_history = parse_fundamental_history_csv(fundamental_history_uploaded)
        official_rows = int(uploaded_fundamental_history["source_family"].eq("IDX_OFFICIAL_REFERENCE").sum())
        st.success(
            f"Histori fundamental terbaca: {len(uploaded_fundamental_history)} baris, "
            f"{uploaded_fundamental_history['ticker'].nunique()} emiten, {official_rows} baris merujuk domain IDX."
        )
    except Exception as exc:
        st.error(f"Histori fundamental tidak dapat dibaca: {exc}")
        st.stop()

if run_portfolio and portfolio_uploaded is not None:
    try:
        portfolio = parse_portfolio_csv(portfolio_uploaded)
    except Exception as exc:
        st.error(f"Portfolio CSV tidak dapat dibaca: {exc}")
        st.stop()
    if portfolio.empty:
        st.error("Portfolio CSV tidak memiliki posisi yang valid.")
        st.stop()

    portfolio_tickers = tuple(portfolio["ticker"].drop_duplicates().tolist())
    progress = st.progress(0, text=f"Mengunduh OHLCV {len(portfolio_tickers)} posisi portfolio…")
    histories, report = cached_portfolio_market_data(tuple(portfolio_tickers), period, bool(itick_api_token), itick_api_token)
    progress.progress(55, text="Menghitung struktur, flow, stop, target, dan bobot posisi…")
    fundamentals, database_fundamental_audit = database_first_fundamentals(portfolio_tickers, cfg, portfolio_tickers) if use_fundamentals else (pd.DataFrame(), pd.DataFrame())
    fundamental_history = pd.DataFrame()
    fundamental_history_report = pd.DataFrame()
    if use_fundamentals and not fundamentals.empty:
        progress.progress(72, text="Mengambil IDX/XBRL official-first + Yahoo fallback/cross-check dan memeriksa kualitas laba…")
        fundamentals, fundamental_history, fundamental_history_report = enrich_fundamental_shortlist(
            fundamentals,
            portfolio_tickers,
            uploaded_fundamental_history,
            cfg,
            enable_twelve_fundamentals,
            twelve_data_api_key,
        )
    portfolio_analysis, portfolio_summary = analyze_portfolio_positions(
        portfolio,
        histories,
        fundamentals=fundamentals,
        signals=pd.DataFrame(),
        account_equity_idr=portfolio_equity_input,
        cash_on_hand_idr=float(cash_on_hand),
        config=cfg,
    )
    source_quorum_audit = build_source_quorum_audit(
        portfolio_tickers,
        source_tiers=getattr(report, "source_tiers", {}) or {},
        fundamental_history=fundamental_history,
        config=cfg,
    )
    st.session_state["scan_result"] = {
        "mode": "portfolio",
        "portfolio": portfolio,
        "portfolio_analysis": portfolio_analysis,
        "portfolio_summary": portfolio_summary,
        "fundamentals": fundamentals,
        "fundamental_history": fundamental_history,
        "fundamental_history_report": fundamental_history_report,
        "database_read_report": database_fundamental_audit,
        "project_management_review": combined_project_management if "combined_project_management" in locals() else uploaded_project_management,
        "automatic_forward_report": automatic_forward_report if "automatic_forward_report" in locals() else pd.DataFrame(),
        "source_quorum_audit": source_quorum_audit,
        "all_histories": histories,
        "download_report": report,
    }
    progress.progress(100, text="Portfolio review selesai")
    progress.empty()

if run_scan and uploaded is not None:
    try:
        scan_tickers = parse_ticker_csv(uploaded)
    except Exception as exc:
        st.error(f"CSV universe tidak dapat dibaca: {exc}")
        st.stop()
    if not scan_tickers:
        st.error("Tidak menemukan ticker yang valid di CSV universe.")
        st.stop()

    portfolio = pd.DataFrame()
    if portfolio_uploaded is not None:
        try:
            portfolio = parse_portfolio_csv(portfolio_uploaded)
        except Exception as exc:
            st.error(f"Portfolio CSV tidak dapat dibaca: {exc}")
            st.stop()
    portfolio_tickers = portfolio["ticker"].drop_duplicates().tolist() if not portfolio.empty else []
    all_tickers = list(dict.fromkeys(scan_tickers + portfolio_tickers))

    market_status = pd.DataFrame()
    news_review = pd.DataFrame()
    broksum = pd.DataFrame()
    if broksum_uploaded is not None:
        try:
            broksum = parse_broker_summary_csv(
                broksum_uploaded,
                source_type='USER_UPLOAD',
                source_verified=False,
            )
            st.info(
                "Broker Summary terbaca sebagai observed flow research; "
                "provenance direct tidak diklaim dari upload manual."
            )
        except Exception as exc:
            st.error(f"Broker Summary CSV tidak dapat dibaca: {exc}")
            st.stop()
    scan_profiler = ScanStageProfiler()
    progress = st.progress(0, text=f"Mengunduh OHLCV {len(all_tickers)} ticker dan IHSG…")
    histories, report, benchmark = cached_market_data(tuple(all_tickers), period, bool(itick_api_token), itick_api_token)
    scan_profiler.mark(
        "OHLCV + benchmark",
        workload=len(all_tickers),
        workload_unit="ticker",
        stage_type="NETWORK/CACHE",
    )
    progress.progress(30, text="Menghitung indikator, struktur pasar, core setup, dan posisi portfolio…")
    result = ScanEngine(cfg).scan(histories, benchmark)
    ihsg_direction = analyze_ihsg_direction(
        benchmark,
        result.get("prepared", {}),
        config=IHSGDirectionConfig(),
        now=now_jkt_ui,
        eod_final=bool(
            benchmark is not None
            and not benchmark.empty
            and str(getattr(benchmark, "attrs", {}).get("bar_state", "")).upper() == "FINAL_EOD"
        ),
    )
    ihsg_risk_cap = float(np.clip(
        _finite_or_default(ihsg_direction.get("risk_budget_multiplier"), 0.50),
        0.20,
        1.00,
    ))
    ihsg_direction["risk_cap_applied"] = bool(execution_policy == "ACCOUNT_GUARDED")
    ihsg_direction["risk_cap_policy"] = (
        "ACCOUNT_GUARDED_POSITION_AND_CAPITAL_CAP"
        if execution_policy == "ACCOUNT_GUARDED"
        else "INFORMATION_ONLY_SIGNAL_FIRST"
    )
    ihsg_direction["base_risk_per_trade_pct"] = float(cfg.risk_per_trade_pct)
    ihsg_direction["effective_risk_per_trade_pct"] = (
        float(cfg.risk_per_trade_pct) * ihsg_risk_cap
        if execution_policy == "ACCOUNT_GUARDED"
        else float(cfg.risk_per_trade_pct)
    )
    ihsg_direction["base_multibagger_budget_idr"] = float(cfg.multibagger_capital_budget_idr)
    ihsg_direction["effective_multibagger_budget_idr"] = (
        float(cfg.multibagger_capital_budget_idr) * ihsg_risk_cap
        if execution_policy == "ACCOUNT_GUARDED"
        else float(cfg.multibagger_capital_budget_idr)
    )
    if execution_policy == "ACCOUNT_GUARDED":
        cfg = cfg.replace(
            risk_per_trade_pct=float(cfg.risk_per_trade_pct) * ihsg_risk_cap,
            max_portfolio_risk_pct=float(cfg.max_portfolio_risk_pct) * ihsg_risk_cap,
            multibagger_capital_budget_idr=float(cfg.multibagger_capital_budget_idr) * ihsg_risk_cap,
        )
    scan_profiler.mark(
        "Core technical + IHSG",
        workload=len(result.get("prepared", {})),
        workload_unit="ticker",
        stage_type="CPU",
    )
    signals = result["signals"]
    if not signals.empty:
        signals["technical_setup_ready"] = signals["status"].eq("EXECUTION_READY")
    source_tier_map = getattr(report, "source_tiers", {}) or {}
    if not signals.empty and "ticker" in signals:
        signals["ohlcv_source_tier"] = signals["ticker"].map(source_tier_map).fillna("UNAVAILABLE")
    signals = apply_universe_integrity_gate(signals, scan_tickers, result["prepared"].keys(), cfg)

    stats = pd.DataFrame()
    trades = pd.DataFrame()
    validation_universe_audit = pd.DataFrame()
    if validate and result["prepared"]:
        progress.progress(
            45,
            text=(
                "Menjalankan adaptive chronological OOS validation; cohort diperluas "
                "sampai target evidence atau batas ticker tercapai…"
            ),
        )
        stats, trades, validation_universe_audit = run_adaptive_walkforward_validation(
            result["prepared"], cfg, initial_tickers=min(80, int(cfg.validation_max_tickers)),
        )
    scan_profiler.mark(
        "Chronological OOS",
        workload=(
            int(validation_universe_audit["selected"].fillna(False).astype(bool).sum())
            if not validation_universe_audit.empty
            else 0
        ),
        workload_unit="ticker",
        stage_type="CPU",
    )
    signals = attach_backtest_stats(signals, stats)
    signals = apply_validation_gate(signals, cfg)

    fundamentals = pd.DataFrame()
    fundamental_history = pd.DataFrame()
    fundamental_history_report = pd.DataFrame()
    database_fundamental_audit = pd.DataFrame()
    if use_fundamentals and (not signals.empty or portfolio_tickers):
        progress.progress(63, text="Mengambil fundamental kandidat dan posisi portfolio…")
        ranked_names = (
            signals.sort_values(["status_rank", "quality_score"], ascending=[True, False])["ticker"].drop_duplicates().tolist()
            if not signals.empty else []
        )
        execution_names = (
            signals.loc[signals["status"].eq("EXECUTION_READY"), "ticker"].drop_duplicates().tolist()
            if not signals.empty else []
        )
        completion_scope = bool(multibagger_full_universe or len(scan_tickers) <= int(getattr(cfg, "full_completion_max_tickers", 400)))
        if completion_scope:
            # Small/medium scans are cheap enough to complete every issuer.
            # This prevents low-ranked names from being permanently starved of
            # snapshot/history evidence and producing misleading NOT_SCORED rows.
            top_names = list(dict.fromkeys(portfolio_tickers + scan_tickers))
        else:
            top_names = list(dict.fromkeys(portfolio_tickers + execution_names + ranked_names[:fundamental_n]))
        technical_priority_names: list[str] = []
        if not signals.empty and "ticker" in signals:
            priority = signals.copy()
            silent_raw = pd.to_numeric(
                priority.get(
                    "silent_accumulation_score",
                    pd.Series(np.nan, index=priority.index),
                ),
                errors="coerce",
            )
            silent_confidence = pd.to_numeric(
                priority.get(
                    "silent_accumulation_confidence",
                    pd.Series(0.0, index=priority.index),
                ),
                errors="coerce",
            ).fillna(0.0).clip(0.0, 100.0)
            priority["_effective_silent"] = (
                50.0
                + (silent_raw.fillna(50.0) - 50.0)
                * silent_confidence / 100.0
            )
            priority["_quality"] = pd.to_numeric(
                priority.get(
                    "quality_score",
                    pd.Series(np.nan, index=priority.index),
                ),
                errors="coerce",
            )
            priority["_relative_strength"] = pd.to_numeric(
                priority.get(
                    "relative_strength60",
                    pd.Series(np.nan, index=priority.index),
                ),
                errors="coerce",
            )
            priority["_adtv"] = pd.to_numeric(
                priority.get(
                    "adtv20_idr",
                    pd.Series(np.nan, index=priority.index),
                ),
                errors="coerce",
            )
            technical_priority = priority.groupby(
                "ticker", as_index=False,
            ).agg(
                effective_silent=("_effective_silent", "max"),
                technical_quality=("_quality", "max"),
                relative_strength=("_relative_strength", "max"),
                adtv20_idr=("_adtv", "max"),
            )
            for source, destination in (
                ("effective_silent", "_silent_rank"),
                ("technical_quality", "_quality_rank"),
                ("relative_strength", "_rs_rank"),
                ("adtv20_idr", "_liquidity_rank"),
            ):
                technical_priority[destination] = (
                    technical_priority[source]
                    .rank(pct=True, method="average")
                    .fillna(0.0)
                )
            technical_priority["_refresh_priority"] = (
                0.35 * technical_priority["_silent_rank"]
                + 0.30 * technical_priority["_quality_rank"]
                + 0.20 * technical_priority["_rs_rank"]
                + 0.15 * technical_priority["_liquidity_rank"]
            )
            technical_priority_names = (
                technical_priority.sort_values(
                    ["_refresh_priority", "ticker"],
                    ascending=[False, True],
                    kind="stable",
                )["ticker"]
                .drop_duplicates()
                .head(int(cfg.fundamental_backfill_max_per_scan))
                .tolist()
            )
        fundamental_priority_names = tuple(dict.fromkeys(
            list(portfolio_tickers)
            + execution_names
            + technical_priority_names
            + ranked_names[:12]
        ))
        fundamentals, database_fundamental_audit = database_first_fundamentals(
            tuple(top_names), cfg, fundamental_priority_names,
        )
        if not fundamentals.empty:
            progress.progress(68, text="Mengambil IDX/XBRL official-first + Yahoo fallback/cross-check, kualitas laba, dilusi, dan consensus…")
            ranking = fundamentals.copy()
            numeric_column = lambda name, default: pd.to_numeric(
                ranking[name] if name in ranking else pd.Series(default, index=ranking.index),
                errors="coerce",
            ).fillna(default)
            snapshot_score = numeric_column("fundamental_score", 0.0)
            snapshot_coverage = numeric_column("fundamental_coverage", 0.0)
            revenue_growth = numeric_column("revenue_growth", -0.25).clip(-0.25, 0.50)
            earnings_growth = numeric_column("earnings_growth", -0.25).clip(-0.25, 0.75)
            if (
                isinstance(signals, pd.DataFrame)
                and not signals.empty
                and "ticker" in signals
                and "silent_accumulation_score" in signals
            ):
                silent_priority = (
                    signals.assign(
                        _silent_priority=pd.to_numeric(
                            signals["silent_accumulation_score"],
                            errors="coerce",
                        )
                    )
                    .groupby("ticker")["_silent_priority"]
                    .max()
                    .to_dict()
                )
            else:
                silent_priority = {}
            ranking["_silent_priority"] = (
                ranking["ticker"].map(silent_priority).fillna(50.0)
            )
            # Backfill priority is not a Multibagger score. It only decides
            # which bounded provider calls receive statement-history evidence
            # first: snapshot quality/growth leads, while Silent Accumulation
            # prevents technically strong candidates from waiting indefinitely.
            ranking["_history_priority"] = (
                0.60 * snapshot_score
                + 0.10 * snapshot_coverage
                + 0.20 * ranking["_silent_priority"]
                + 10.0 * revenue_growth
                + 8.0 * earnings_growth
            )
            multibagger_ranked = ranking.sort_values("_history_priority", ascending=False)["ticker"].drop_duplicates().tolist()
            history_priority_names = tuple(dict.fromkeys(
                list(portfolio_tickers)
                + multibagger_ranked[:24]
                + execution_names
                + ranked_names[:12]
            ))
            uploaded_names = (
                uploaded_fundamental_history["ticker"].drop_duplicates().tolist()
                if not uploaded_fundamental_history.empty else []
            )
            if completion_scope:
                # Read persisted statement history for the whole small/medium
                # universe; the enrichment scheduler also switches to full
                # completion mode at this size.
                # let the bounded round-robin scheduler refresh only one cohort.
                # Previously this scope was cut to the same top 40 on every
                # scan, starving the remaining issuers indefinitely.
                history_names = tuple(dict.fromkeys(
                    list(history_priority_names) + uploaded_names
                    + multibagger_ranked + list(scan_tickers)
                ))
            else:
                history_names = tuple(dict.fromkeys(
                    list(history_priority_names) + uploaded_names
                    + multibagger_ranked[:int(fundamental_history_n)]
                ))
            fundamentals, fundamental_history, fundamental_history_report = enrich_fundamental_shortlist(
                fundamentals,
                history_names,
                uploaded_fundamental_history,
                cfg,
                enable_twelve_fundamentals,
                twelve_data_api_key,
                priority_tickers=history_priority_names,
            )
    scan_profiler.mark(
        "Fundamental snapshot + history",
        workload=len(fundamentals),
        workload_unit="ticker",
        stage_type="NETWORK/CACHE",
    )
    automatic_project_management = pd.DataFrame()
    automatic_forward_report = pd.DataFrame()
    if use_fundamentals and automatic_forward_quality_enabled and not fundamentals.empty:
        progress.progress(72, text=f"Mencari project, management, dan forward impact untuk kandidat teratas…")
        # Forward-quality discovery is optional enrichment. A malformed or
        # incomplete provider snapshot must never terminate the entire scan.
        try:
            automatic_project_management, automatic_forward_report = database_first_forward_quality(
                fundamentals, tuple(all_tickers), cfg,
            )
        except Exception as exc:
            automatic_project_management = pd.DataFrame()
            automatic_forward_report = pd.DataFrame([{
                "ticker": "SYSTEM",
                "state": "FORWARD_ENRICHMENT_FAIL_SOFT",
                "documents": 0,
                "rows": 0,
                "errors": f"{type(exc).__name__}: {str(exc)[:240]}",
            }])
    combined_project_management = merge_project_management_reviews(
        automatic_project_management, uploaded_project_management,
    )
    scan_profiler.mark(
        "Forward intelligence",
        workload=len(automatic_project_management),
        workload_unit="evidence rows",
        stage_type="NETWORK/CACHE",
    )
    signals = attach_fundamentals(signals, fundamentals)
    signals = apply_fundamental_gate(signals, cfg)

    progress.progress(76, text="Menyelesaikan status IDX dan berita…")
    # Resolve context for every visible actionable candidate, not only rows that
    # were already EXECUTION_READY before context was fetched. Limiting context
    # to pre-ready rows created a circular dependency and left watchlist rows
    # permanently at 37.5–50% completeness.
    visible_context_names = (
        signals.loc[~signals["status"].eq("REJECT")]
        .sort_values(["status_rank", "quality_score"], ascending=[True, False])
        ["ticker"].drop_duplicates().head(60).tolist()
        if not signals.empty else []
    )
    # Keep the request budget bounded while reserving source coverage for the
    # highest-priority Multibagger research queue.
    narrative_priority_names = (
        multibagger_ranked[:24]
        if 'multibagger_ranked' in locals() else []
    )
    # Refresh every ticker progressively instead of permanently starving names
    # outside the first 60. Priority names are always included; remaining slots
    # rotate deterministically by ISO week and persist through the database cache.
    narrative_limit = int(getattr(cfg, "narrative_refresh_max_per_scan", 80))
    priority_context = list(dict.fromkeys(
        list(portfolio_tickers) + narrative_priority_names + visible_context_names
    ))
    rotation_seed = pd.Timestamp.now(tz="Asia/Jakarta").strftime("%G-W%V")
    rotation_pool = sorted(
        [ticker for ticker in all_tickers if ticker not in set(priority_context)],
        key=lambda ticker: hashlib.sha256(f"{rotation_seed}|{ticker}".encode("utf-8")).hexdigest(),
    )
    narrative_completion_limit = int(getattr(cfg, "narrative_full_completion_max_tickers", 400))
    context_cap = max(narrative_limit, len(all_tickers)) if len(all_tickers) <= narrative_completion_limit else narrative_limit
    context_names = list(dict.fromkeys(priority_context + rotation_pool))[:context_cap]
    if context_names:
        market_status = cached_automatic_market_status(tuple(context_names))
        news_review = cached_automatic_news(tuple(context_names), cfg.min_news_lookback_days)
    signals = apply_market_status_gate(signals, market_status, cfg)
    signals = apply_news_gate(signals, news_review, cfg)
    signals = attach_broker_summary(signals, broksum)

    quote_candidates = (
        signals.loc[~signals["status"].eq("REJECT")]
        .sort_values(["status_rank", "quality_score"], ascending=[True, False])
        ["ticker"].drop_duplicates().head(40).tolist()
        if not signals.empty else []
    )
    snapshots = cached_execution_snapshots(tuple(quote_candidates)) if quote_candidates else pd.DataFrame()
    signals = apply_execution_snapshot_gate(signals, snapshots, cfg)

    independent_price_data = pd.DataFrame()
    automatic_price_report = pd.DataFrame()
    if quote_candidates:
        reference_dates = [
            pd.Timestamp(histories[ticker].index[-1])
            for ticker in quote_candidates
            if ticker in histories and histories[ticker] is not None and not histories[ticker].empty
        ]
        reference_date = max(reference_dates).strftime("%Y-%m-%d") if reference_dates else now_jkt_ui.strftime("%Y-%m-%d")
        automatic_names = tuple(quote_candidates[: int(cfg.max_automatic_price_candidates)])
        primary_reference = tuple(
            (
                ticker,
                pd.Timestamp(histories[ticker].index[-1]).strftime("%Y-%m-%d"),
                float(pd.to_numeric(histories[ticker]["Close"], errors="coerce").dropna().iloc[-1]),
            )
            for ticker in automatic_names
            if ticker in histories
            and histories[ticker] is not None
            and not histories[ticker].empty
            and pd.to_numeric(histories[ticker]["Close"], errors="coerce").dropna().size > 0
        )
        progress.progress(
            84,
            text=f"Memvalidasi harga otomatis untuk {len(automatic_names)} kandidat (IDX → Google → fallback)…",
        )
        primary_source_tiers = tuple(
            (ticker, str(source_tier_map.get(ticker, "UNKNOWN")))
            for ticker in automatic_names
        )
        independent_price_data, automatic_price_report = cached_automatic_independent_prices(
            automatic_names,
            reference_date,
            primary_reference,
            primary_source_tiers,
            cfg,
            twelve_data_api_key,
            itick_api_token,
        )
    price_validation = build_independent_price_validation(
        histories, independent_price_data, cfg,
        primary_source_tiers=source_tier_map,
    )
    signals = apply_independent_price_gate(signals, price_validation, cfg)
    scan_profiler.mark(
        "Context + quote + price cross-check",
        workload=len(context_names) + len(quote_candidates),
        workload_unit="requests/candidates",
        stage_type="NETWORK/CACHE",
    )
    signals = attach_position_sizing(signals, cfg)
    signals = apply_analyst_fusion_gate(signals, cfg)

    portfolio_analysis, portfolio_summary = analyze_portfolio_positions(
        portfolio, histories, fundamentals=fundamentals, signals=signals,
        account_equity_idr=portfolio_equity_input, cash_on_hand_idr=float(cash_on_hand), config=cfg,
    )
    if not portfolio.empty:
        current_positions = int(portfolio_summary.get("positions", len(portfolio)))
        current_invested = float(portfolio_summary.get("market_value_idr", 0.0))
        current_open_risk = float(portfolio_summary.get("open_risk_idr", 0.0))
    else:
        current_positions = int(current_positions_manual)
        current_invested = float(current_invested_manual)
        current_open_risk = float(current_open_risk_manual)

    progress.progress(90, text="Meranking seluruh setup valid tanpa account-risk gate…")
    signals = enforce_analyst_portfolio_budget(
        signals, cfg, current_positions=current_positions, current_open_risk_idr=current_open_risk,
        current_invested_idr=current_invested, cash_on_hand_idr=float(cash_on_hand),
    )
    signals = finalize_execution_integrity(signals, cfg)
    signals = sort_signals(signals)

    # Load the durable IDX session calendar in one bounded database request.
    # An empty table deliberately falls back to weekday/holiday validation and
    # is surfaced as unverified rather than silently presented as official.
    idx_trading_calendar = pd.DataFrame()
    try:
        calendar_start = pd.Timestamp.now(tz="Asia/Jakarta").normalize() - pd.DateOffset(days=45)
        calendar_end = calendar_start + pd.DateOffset(days=240)
        idx_trading_calendar = ScannerDatabaseBridge().read_idx_trading_calendar(calendar_start, calendar_end)
        if idx_trading_calendar is not None and not idx_trading_calendar.empty:
            calendar_local = idx_trading_calendar.copy()
            calendar_local["trade_date"] = pd.to_datetime(calendar_local.get("trade_date"), errors="coerce")
            open_mask = calendar_local.get("is_open", pd.Series(False, index=calendar_local.index)).fillna(False).astype(bool)
            open_dates = tuple(calendar_local.loc[open_mask, "trade_date"].dropna().dt.date.astype(str))
            closed_dates = tuple(calendar_local.loc[~open_mask, "trade_date"].dropna().dt.date.astype(str))
            cfg = cfg.replace(
                idx_official_open_dates=open_dates,
                idx_official_closed_dates=closed_dates,
            )
    except Exception:
        idx_trading_calendar = pd.DataFrame()

    signals = enrich_core_signals_with_time_cycle(
        signals, result["prepared"], enabled=bool(cfg.time_cycle_enabled),
        max_weight=float(cfg.time_cycle_core_max_weight),
        min_confidence=float(cfg.time_cycle_min_confidence),
        config=TimeCycleConfig(
            min_bars=int(cfg.time_cycle_min_history_bars),
            lunar_enabled=bool(cfg.time_cycle_lunar_enabled),
            eoff_enabled=bool(cfg.eoff_enabled),
            eoff_ephemeris_enabled=bool(cfg.eoff_ephemeris_enabled),
            eoff_min_fib_cluster=int(cfg.eoff_min_fib_cluster),
            eoff_min_unique_anchors=int(cfg.eoff_min_unique_anchors),
            eoff_max_dominant_anchor_share=float(cfg.eoff_max_dominant_anchor_share),
            eoff_aspect_orb_deg=float(cfg.eoff_aspect_orb_deg),
            eoff_require_astro_fib_confluence=bool(cfg.eoff_require_astro_fib_confluence),
            idx_trading_holidays=tuple(cfg.idx_trading_holidays or ()),
            idx_official_open_dates=tuple(cfg.idx_official_open_dates or ()),
            idx_official_closed_dates=tuple(cfg.idx_official_closed_dates or ()),
            require_official_idx_calendar=bool(cfg.require_official_idx_calendar),
        ),
    )
    scan_profiler.mark(
        "Portfolio gate + time-cycle",
        workload=int(signals["ticker"].nunique()) if not signals.empty else 0,
        workload_unit="ticker",
        stage_type="CPU",
    )
    portfolio_analysis, portfolio_summary = analyze_portfolio_positions(
        portfolio, histories, fundamentals=fundamentals, signals=signals,
        account_equity_idr=portfolio_equity_input, cash_on_hand_idr=float(cash_on_hand), config=cfg,
    )

    progress.progress(96, text="Membangun Multibagger dan Core Swing ranking…")
    ai_cfg = LocalAIConfig(
        enabled=bool(cfg.ai_enabled), mode=str(cfg.ai_mode), max_weight=float(cfg.ai_max_weight),
        min_training_events=int(cfg.ai_min_training_events), min_strategy_events=int(cfg.ai_min_strategy_events),
        knn_k=int(cfg.ai_knn_k), memory_entry_window_bars=int(cfg.ai_memory_entry_window_bars),
        memory_horizon_bars=int(cfg.ai_memory_horizon_bars),
        min_expectancy_r=float(cfg.execution_ai_min_expectancy_r),
        max_oos_drawdown_pct=float(cfg.execution_ai_max_drawdown_pct),
        min_profit_factor=float(cfg.execution_ai_min_profit_factor),
    )
    try:
        imported_memory = load_memory(ai_memory_uploaded)
    except Exception as exc:
        st.warning(f"AI memory tidak dapat diimpor; memakai cache lokal: {exc}")
        imported_memory = load_memory(None)
    try:
        database_ai_memory = ScannerDatabaseBridge().read_ai_execution_outcomes(all_tickers)
    except Exception:
        database_ai_memory = pd.DataFrame()
    memory_parts = [
        frame for frame in (database_ai_memory, imported_memory)
        if isinstance(frame, pd.DataFrame) and not frame.empty
    ]
    merged_ai_memory = (
        pd.concat(memory_parts, ignore_index=True, sort=False)
        if memory_parts else pd.DataFrame()
    )
    if not merged_ai_memory.empty and "signal_id" in merged_ai_memory:
        merged_ai_memory = merged_ai_memory.drop_duplicates("signal_id", keep="last")
    ai_memory = update_outcome_memory(
        pd.DataFrame(), result["prepared"], merged_ai_memory, ai_cfg,
    )
    try:
        narrative_bridge = ScannerDatabaseBridge()
        database_narrative_events = narrative_bridge.read_narrative_events(
            all_tickers,
        )
        database_narrative_outcomes = (
            narrative_bridge.read_narrative_event_outcomes(all_tickers)
        )
    except Exception:
        database_narrative_events = pd.DataFrame()
        database_narrative_outcomes = pd.DataFrame()
    narrative_event_parts = [
        frame for frame in (
            database_narrative_events, uploaded_narrative_events,
        )
        if isinstance(frame, pd.DataFrame) and not frame.empty
    ]
    combined_narrative_events = (
        pd.concat(narrative_event_parts, ignore_index=True, sort=False)
        if narrative_event_parts else pd.DataFrame()
    )
    try:
        reference_fx, reference_fx_report = cached_reference_fx()
    except Exception as exc:
        reference_fx, reference_fx_report = pd.DataFrame(), pd.DataFrame([{
            'provider': 'REFERENCE_FX', 'status': 'ERROR',
            'error': f'{type(exc).__name__}: {str(exc)[:160]}',
        }])
        st.warning(f"Reference FX tidak tersedia; valuasi emiten non-IDR akan fail-closed: {exc}")
    focus_screens = build_focus_screens(
        result["prepared"],
        fundamentals=fundamentals,
        core_signals=signals,
        project_management=combined_project_management,
        news_review=news_review,
        market_status=market_status,
        narrative_events=combined_narrative_events,
        narrative_outcomes=database_narrative_outcomes,
        benchmark=benchmark,
        config=cfg,
        validation_events=trades,
        ai_memory=resolved_memory_events(ai_memory),
        reference_fx=reference_fx,
    )
    focus_screens['reference_fx'] = reference_fx
    focus_screens['reference_fx_report'] = reference_fx_report
    core_cache_report = result.get(
        'core_incremental_cache_report', pd.DataFrame(),
    )
    focus_cache_report = focus_screens.get(
        'incremental_cache_report', pd.DataFrame(),
    )
    cache_parts = []
    for cache_layer, frame in (
        ('CORE', core_cache_report), ('FOCUS', focus_cache_report),
    ):
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            part = frame.copy()
            part.insert(0, 'cache_layer', cache_layer)
            cache_parts.append(part)
    incremental_cache_report = (
        pd.concat(cache_parts, ignore_index=True, sort=False)
        if cache_parts else pd.DataFrame()
    )
    if not incremental_cache_report.empty:
        cache_states = incremental_cache_report.get(
            'cache_state', pd.Series(dtype=str),
        ).fillna('').astype(str)
        cache_hits = int(cache_states.eq('HIT').sum())
        cache_rebuilds = int(cache_states.str.endswith('REBUILT').sum())
        cache_load_ms = pd.to_numeric(
            incremental_cache_report.get('load_ms', pd.Series(dtype=float)),
            errors='coerce',
        ).fillna(0.0).sum()
        st.caption(
            f"Incremental cache v7.16.3: {cache_hits} stage hit, "
            f"{cache_rebuilds} stage rebuilt, load {cache_load_ms:.0f} ms."
        )
        with st.expander('Audit incremental cache', expanded=False):
            st.dataframe(streamlit_safe_frame(incremental_cache_report), width="stretch", hide_index=True)
    profit_order_builder = focus_screens.get("profit_order_builder", pd.DataFrame())
    ai_memory = update_outcome_memory(profit_order_builder, result["prepared"], ai_memory, ai_cfg)
    chronological_memory = validation_events_to_memory(trades)
    if isinstance(chronological_memory, pd.DataFrame) and not chronological_memory.empty:
        ai_memory = pd.concat([ai_memory, chronological_memory], ignore_index=True, sort=False)
        if "signal_id" in ai_memory:
            ai_memory = ai_memory.drop_duplicates("signal_id", keep="last").reset_index(drop=True)
        save_memory(ai_memory, int(getattr(ai_cfg, "memory_max_rows", 10000)))
    focus_screens["ai_outcome_memory"] = ai_memory
    focus_screens["ai_memory_summary"] = memory_summary(ai_memory)

    # Durable research memory is separate from the AI trade-memory layer. It
    # records EOFF and Silent Accumulation predictions by semantic model
    # version and resolves them only after enough forward bars exist.
    research_ranking_frames = [
        frame for frame in (
            focus_screens.get("multibagger", pd.DataFrame()),
            focus_screens.get("core_swing", pd.DataFrame()),
        ) if frame is not None and not frame.empty
    ]
    research_ranking = pd.concat(research_ranking_frames, ignore_index=True, sort=False) if research_ranking_frames else pd.DataFrame()
    try:
        existing_research_outcomes = ScannerDatabaseBridge().read_research_outcomes(
            list(all_tickers) + [IHSG_TICKER]
        )
    except Exception:
        existing_research_outcomes = pd.DataFrame()
    research_outcomes = update_research_outcomes(
        existing_research_outcomes,
        research_ranking,
        result.get("prepared", {}),
    )
    research_outcomes = update_ihsg_outcomes(
        research_outcomes,
        ihsg_direction,
        benchmark,
    )
    try:
        existing_selector_outcomes = ScannerDatabaseBridge().read_selector_outcomes(
            list(all_tickers)
        )
    except Exception:
        existing_selector_outcomes = pd.DataFrame()
    selector_outcomes = update_selector_outcomes(
        existing_selector_outcomes,
        focus_screens.get("stock_selector", pd.DataFrame()),
        result.get("prepared", {}),
        SelectorConfig(
            roundtrip_cost_pct=float(cfg.selector_roundtrip_cost_pct),
        ),
    )
    focus_screens["selector_outcomes"] = selector_outcomes
    focus_screens["research_outcomes"] = research_outcomes
    focus_screens["research_outcome_summary"] = research_outcome_summary(research_outcomes)
    focus_screens["model_registry"] = model_registry_frame()
    scanner_data_contract_audit = build_scanner_data_contract_audit(
        scan_tickers,
        histories=histories,
        prepared=result.get("prepared", {}),
        fundamentals=fundamentals,
        fundamental_history=fundamental_history,
        selector=focus_screens.get("stock_selector", pd.DataFrame()),
        multibagger=focus_screens.get("multibagger", pd.DataFrame()),
        core_signals=signals,
        order_builder=focus_screens.get(
            "profit_order_builder", pd.DataFrame(),
        ),
        order_builder_coverage=focus_screens.get(
            "profit_data_coverage_audit", pd.DataFrame(),
        ),
    )
    focus_screens["scanner_data_contract_audit"] = (
        scanner_data_contract_audit
    )
    scan_profiler.mark(
        "Focus ranking + outcome memory",
        workload=len(profit_order_builder),
        workload_unit="ranked rows",
        stage_type="CPU",
    )

    database_read_frames: list[pd.DataFrame] = []
    for frame in (database_fundamental_audit, fundamental_history_report, automatic_forward_report):
        if frame is None or frame.empty:
            continue
        local = frame.copy()
        if "provider" in local.columns:
            # Scheduler/provider rows intentionally carry database_read_state
            # for context.  They are not database reads and previously doubled
            # the denominator (400 x 3 became 2,440 rows after refresh audit).
            local = local[
                local["provider"].fillna("").astype(str)
                .eq("SUPABASE_DATABASE_FIRST")
            ]
        elif "database_read_state" in local.columns:
            local = local[local["database_read_state"].fillna("").astype(str).str.len().gt(0)]
        else:
            local = pd.DataFrame()
        if not local.empty:
            database_read_frames.append(local)
    database_read_report = pd.concat(database_read_frames, ignore_index=True, sort=False) if database_read_frames else pd.DataFrame()

    source_quorum_audit = build_source_quorum_audit(
        all_tickers,
        source_tiers=source_tier_map,
        price_validation=price_validation,
        fundamental_history=fundamental_history,
        market_status=market_status,
        news_review=news_review,
        validation_stats=stats,
        broker_summary=broksum,
        config=cfg,
    )

    result.update({
        "mode": "scanner", "signals": signals, "validation_stats": stats,
        "validation_trades": trades, "ai_outcome_memory": ai_memory,
        "validation_universe_audit": validation_universe_audit,
        "ai_model_audit": focus_screens.get("ai_model_audit", pd.DataFrame()),
        "selector_model_audit": focus_screens.get("selector_model_audit", pd.DataFrame()),
        "selector_outcomes": selector_outcomes,
        "fundamentals": fundamentals,
        "fundamental_history": fundamental_history,
        "fundamental_history_report": fundamental_history_report,
        "database_read_report": database_read_report,
        "source_quorum_audit": source_quorum_audit,
        "scanner_data_contract_audit": scanner_data_contract_audit,
        "market_status": market_status, "news_review": news_review,
        "broker_summary": broksum, "execution_snapshots": snapshots,
        "independent_price_data": independent_price_data,
        "price_validation": price_validation,
        "independent_provider_report": automatic_price_report,
        "twelve_data_report": (
            automatic_price_report.loc[automatic_price_report.get("provider", pd.Series(dtype=str)).eq("TWELVE_DATA")].copy()
            if not automatic_price_report.empty else pd.DataFrame()
        ),
        "download_report": report, "portfolio": portfolio,
        "portfolio_analysis": portfolio_analysis, "portfolio_summary": portfolio_summary,
        "all_histories": histories,
        "focus_screens": focus_screens,
        "realized_profit_to_compound_idr": realized_profit_to_compound,
        "base_compounding_budget_idr": effective_multibagger_budget,
        "compounding_budget_idr": float(cfg.multibagger_capital_budget_idr),
        "ihsg_adjusted_compounding_budget_idr": float(cfg.multibagger_capital_budget_idr),
        "project_management_review": combined_project_management,
        "automatic_forward_report": automatic_forward_report,
        "research_outcomes": research_outcomes,
        "research_outcome_summary": focus_screens.get("research_outcome_summary", pd.DataFrame()),
        "narrative_events": focus_screens.get("narrative_events", pd.DataFrame()),
        "narrative_event_outcomes": focus_screens.get("narrative_event_outcomes", pd.DataFrame()),
        "narrative_profiles": focus_screens.get("narrative_profiles", pd.DataFrame()),
        "model_registry": focus_screens.get("model_registry", pd.DataFrame()),
        "idx_trading_calendar": idx_trading_calendar,
        "benchmark": benchmark,
        "ihsg_direction": ihsg_direction,
        "scanner_version": "7.16.3-database-write-readback-observability",
    })
    # Database is optional and fail-soft. Until the weekend setup is completed,
    # this returns DISABLED_NO_DATABASE and performs no network request.
    try:
        result["database_sync_report"] = ScannerDatabaseBridge().persist_scan_result(result)
    except Exception as exc:
        result["database_sync_report"] = pd.DataFrame([{
            "bridge_version": DATABASE_BRIDGE_VERSION,
            "schema_version": DATABASE_SCHEMA_VERSION,
            "database_mode": "UNKNOWN",
            "state": "DATABASE_FAIL_SOFT",
            "table": "",
            "rows_attempted": 0,
            "rows_written": 0,
            "detail": f"{type(exc).__name__}: {str(exc)[:240]}",
        }])
    scan_profiler.mark(
        "Database sync",
        workload=len(signals),
        workload_unit="signal rows",
        stage_type="NETWORK/CACHE",
    )
    result["scan_performance_profile"] = scan_profiler.frame()
    st.session_state["scan_result"] = result
    st.session_state["_last_auto_scan_signature"] = scan_signature
    progress.progress(100, text="Scan dan portfolio review selesai")
    progress.empty()

if "scan_result" not in st.session_state:
    st.markdown(
        """
        <div class="scanner-note">
          <b>Alur otomatis v7.16.3 — Currency/Split-Aware Valuation, Complete Universe, Provenance-Gated Evidence, Statistical Selector, Core Swing & Arah IHSG</b><br>
          1) Tab Arah IHSG dapat dijalankan tanpa CSV dan akan ABSTAIN bila edge belum terbukti.<br>
          2) Upload universe ticker untuk menambahkan breadth serta membangun Top 20 ticker unik.<br>
          3) Portfolio Stockbit bersifat opsional; unggah hanya bila ingin cash/heat/posisi dihitung dari snapshot nyata.
        </div>
        """,
        unsafe_allow_html=True,
    )
    result = ScanEngine(cfg).scan({}, None)
    result.update({
        "mode": "scanner",
        "signals": pd.DataFrame(columns=["ticker", "status", "setup", "quality_score"]),
        "universe": pd.DataFrame(),
        "prepared": {},
        "focus_screens": {},
        "all_histories": {},
        "benchmark": pd.DataFrame(),
        "ihsg_direction": None,
        "validation_stats": pd.DataFrame(),
        "validation_trades": pd.DataFrame(),
        "validation_universe_audit": pd.DataFrame(),
        "scan_performance_profile": pd.DataFrame(),
        "download_report": None,
    })
else:
    result = st.session_state["scan_result"]
if result.get("mode") == "portfolio":
    st.success("Portfolio-only review selesai. Scanner universe dan backtest tidak dijalankan.")
    render_portfolio_panel(result)
    report = result.get("download_report")
    if report is not None and getattr(report, "failed", None):
        with st.expander(f"Ticker portfolio gagal diunduh ({len(report.failed)})"):
            st.dataframe(pd.DataFrame(report.failed.items(), columns=["ticker", "error"]), hide_index=True)
    st.stop()

signals: pd.DataFrame = result["signals"]
universe: pd.DataFrame = result["universe"]
context = result["market_context"]
report = result.get("download_report")

if context.regime == "RISK_ON":
    st.success(f"Regime: {context.regime} — {context.reason}")
elif context.regime == "RISK_OFF":
    st.error(f"Regime: {context.regime} — {context.reason}")
else:
    st.warning(f"Regime: {context.regime} — {context.reason}")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Ticker valid", len(result["prepared"]))
m2.metric("Setup rows", len(signals))
m3.metric("Trigger confirmed", int(signals.loc[signals.get("analyst_order_mode", pd.Series('', index=signals.index)).eq("TRIGGER_CONFIRMED"), "ticker"].nunique()) if not signals.empty else 0)
m4.metric("Entry plans", int(signals.loc[signals["status"].eq("ENTRY_PLAN_READY"), "ticker"].nunique()) if not signals.empty else 0)
m5.metric("Core plan verified", int(signals.loc[signals.get("autopilot_verified", pd.Series(False, index=signals.index)).fillna(False).astype(bool), "ticker"].nunique()) if not signals.empty else 0)
m6, m7, m8, m9, m10 = st.columns(5)
m6.metric("Account gate", "ON" if execution_policy == "ACCOUNT_GUARDED" else "USER MANAGED")
m7.metric("Menunggu close", int(signals.loc[signals["status"].eq("PENDING_CLOSE"), "ticker"].nunique()) if not signals.empty else 0)
m8.metric("Pending data", int(signals.loc[signals["status"].eq("PENDING_DATA"), "ticker"].nunique()) if not signals.empty else 0)
m9.metric("Watchlist", int(signals.loc[signals["status"].eq("WATCHLIST_ENTRY"), "ticker"].nunique()) if not signals.empty else 0)
m10.metric("Breadth > EMA50", f"{context.breadth_ema50:.0f}%" if context.breadth_ema50 is not None else "N/A")

main_ihsg, main_best, main_existing, main_time_cycle = st.tabs(
    [
        "1 · Arah IHSG",
        "2 · Saham Terbaik",
        "3 · Dashboard Lengkap",
        "4 · Time-Cycle",
    ]
)

with main_ihsg:
    ihsg_forecast = result.get("ihsg_direction")
    ihsg_benchmark = result.get("benchmark", pd.DataFrame())
    standalone = st.session_state.get("standalone_ihsg_direction_v720")
    if not isinstance(ihsg_forecast, Mapping) and isinstance(standalone, tuple) and len(standalone) == 2:
        ihsg_benchmark, ihsg_forecast = standalone
    if not isinstance(ihsg_forecast, Mapping):
        st.info(
            "Arah IHSG dapat dianalisis tanpa universe CSV. Tanpa universe, breadth saham berstatus "
            "missing dan model memakai fitur indeks saja."
        )
        if st.button(
            "Analisis arah IHSG sekarang",
            type="primary",
            key="run_standalone_ihsg_v720",
        ):
            with st.spinner("Mengambil completed EOD IHSG dan menjalankan walk-forward…"):
                ihsg_benchmark, ihsg_forecast = cached_standalone_ihsg_direction("5y")
            st.session_state["standalone_ihsg_direction_v720"] = (
                ihsg_benchmark,
                ihsg_forecast,
            )
    if isinstance(ihsg_forecast, Mapping):
        render_ihsg_direction_panel(ihsg_forecast, ihsg_benchmark)

with main_best:
    render_top20_dashboard(
        result,
        lambda ticker, lookback: run_single_ticker_deep_dive(
            ticker,
            lookback,
            cfg,
            twelve_data_api_key=twelve_data_api_key,
            itick_api_token=itick_api_token,
        ),
    )

with main_existing:
    tab_multibagger, tab_orders, tab_daily, tab_setups, tab_portfolio, tab_chart, tab_bridge, tab_validation, tab_audit, tab_method = st.tabs(
        [
            "Multibagger", "Core Swing Ranking", "Daily Focus", "Core Setups",
            "Portfolio Stockbit", "Chart", "TradingView / Stockbit",
            "Validation", "Audit Universe", "Metodologi",
        ]
    )

    focus_screens: dict[str, pd.DataFrame] = result.get("focus_screens", {})

    with tab_orders:
        core_radar = focus_screens.get("core_swing", pd.DataFrame())
        selector_audit_view = focus_screens.get("selector_model_audit", pd.DataFrame())
        st.subheader("Cross-Sectional Core Swing Radar")
        st.caption(
            "Urutan ini memilih saham lebih dahulu berdasarkan excess return 5D/20D/60D terhadap IHSG, "
            "trend, relative strength, Silent Accumulation, dan Emir Public-Framework Score. Framework Emir hanya menaikkan ranking bila narrative lifecycle dan flow sama-sama terkonfirmasi; fase crowded/distribution diblokir. Setup dicari sesudah seleksi; karena itu "
            "NO_SETUP berarti tunggu trigger, bukan kandidat otomatis dibuang."
        )
        if core_radar.empty:
            st.info("Selector belum memiliki histori/cross-section yang cukup.")
        else:
            radar_columns = [
                column for column in (
                    "swing_selection_rank", "swing_production_rank", "swing_evidence_class",
                    "swing_rank_eligible", "swing_score_comparability_pct",
                    "ticker", "swing_selection_score",
                    "absolute_swing_score", "absolute_selector_score",
                    "relative_selector_overlay_score", "relative_overlay_weight_pct",
                    "cross_sectional_peer_count", "selector_universe_state",
                    "score_inflation_guard_active",
                    "technical_selection_score", "swing_momentum_quality_score",
                    "momentum_continuity_score", "momentum_consistency_score",
                    "high_52w_proximity_score",
                    "effective_silent_accumulation_score",
                    "stock_universe_familiarity_score",
                    "stock_universe_familiarity_coverage_pct",
                    "stock_universe_familiarity_state",
                    "smart_money_behavior_score",
                    "smart_money_behavior_coverage_pct",
                    "smart_money_behavior_state",
                    "smart_money_flow_evidence_mode",
                    "distribution_severity_score", "distribution_penalty_points",
                    "distribution_evidence_state", "broker_summary_score",
                    "narrative_lifecycle_score",
                    "narrative_lifecycle_state",
                    "flow_preceded_narrative",
                    "emir_method_score",
                    "emir_method_coverage_pct",
                    "emir_method_score_state",
                    "emir_method_state",
                    "emir_method_production_eligible",
                    "emir_method_reliability_pct",
                    "emir_position_cap_pct",
                    "emir_selection_reason",
                    "emir_risk_flags",
                    "core_priority_score_pre_overlay",
                    "core_narrative_contribution_points",
                    "core_emir_contribution_points",
                    "core_priority_score_formula",
                    "silent_accumulation_score",
                    "silent_accumulation_confidence",
                    "narrative_flow_effective_score",
                    "narrative_flow_research_score",
                    "narrative_flow_score_state",
                    "narrative_flow_convergence_state",
                    "narrative_evidence_coverage_pct",
                    "narrative_evidence_mode",
                    "structured_financial_evidence_state",
                    "structured_financial_evidence_coverage_pct",
                    "structured_financial_source_count",
                    "structured_financial_latest_period",
                    "evidence_acquisition_status",
                    "evidence_acquisition_complete",
                    "evidence_acquisition_missing",
                    "narrative_score_state",
                    "operating_narrative_proxy_score",
                    "operating_narrative_proxy_coverage_pct",
                    "issuer_alignment_effective_score",
                    "issuer_alignment_score_state",
                    "issuer_alignment_evidence_basis",
                    "issuer_alignment_state",
                    "retail_adoption_stage",
                    "narrative_conversion_rate_5d_pct",
                    "narrative_conversion_resolved_5d",
                    "narrative_conversion_rate_20d_pct",
                    "narrative_conversion_resolved_20d",
                    "narrative_crowding_risk_score",
                    "narrative_hard_block",
                    "relative_strength_score", "sector_relative_strength_score",
                    "sector_peer_count", "trend_score", "liquidity_bucket",
                    "selector_rank_eligible", "selector_data_state",
                    "technical_feature_coverage_pct",
                    "selector_missing_feature_count",
                    "selector_missing_features",
                    "estimated_market_impact_cost_pct", "estimated_total_cost_pct",
                    "selector_expected_excess_return_5d_pct",
                    "selector_expected_excess_return_20d_pct",
                    "selector_expected_excess_return_60d_pct",
                    "selector_outperform_probability_5d_pct",
                    "selector_outperform_probability_20d_pct",
                    "selector_outperform_probability_60d_pct",
                    "selector_model_state", "active_setup", "setup_status",
                    "selected_reason", "not_entry_reason", "trigger_waiting",
                    "retest_reference_price", "reclaim_trigger_price",
                    "trigger_basis", "trigger_instruction", "trigger_valid_until",
                    "invalidation_reason", "primary_risk", "entry", "stop_loss",
                    "tp1", "tp2", "rr1", "rr2",
                ) if column in core_radar.columns
            ]
            st.dataframe(
                streamlit_safe_frame(core_radar[radar_columns]),
                hide_index=True,
                width="stretch",
                height=430,
            )
        if not selector_audit_view.empty:
            with st.expander("Walk-forward selector: rule vs independen vs AI vs relative strength", expanded=False):
                st.dataframe(selector_audit_view, hide_index=True, width="stretch")
                st.caption(
                    "AI selector hanya dipromosikan bila lower confidence bound SciPy untuk expectancy net dan "
                    "Brier skill tetap positif, paired advantage mengalahkan baseline terkuat setelah koreksi "
                    "tiga horizon, CSCV PBO berada di bawah guard, dan maximum drawdown tetap terkendali. "
                    "Label return sudah dikurangi biaya eksplisit serta allowance market impact per bucket likuiditas."
                )
        st.divider()
        profit_orders = focus_screens.get("profit_order_builder", pd.DataFrame())
        st.subheader("Profit Order Builder — kandidat yang sudah punya setup")
        st.caption(
            "Urutan khusus setup Core Swing harian. Rule conviction digabung dengan AI lokal yang belajar dari chronological walk-forward "
            "dan outcome memory. Time-cycle/EOFF clean-room ikut memengaruhi daily core/swing hanya setelah confluence dan evidence lolos. "
            "Bobot AI otomatis diturunkan saat sampel lemah, kalibrasi buruk, atau terjadi feature drift; bukan jaminan profit."
        )
        if profit_orders.empty:
            st.info("Belum ada setup profit-engine yang melewati minimum conviction.")
        else:
            p1, p2, p3, p4, p5, p6 = st.columns(6)
            p1.metric("Kandidat terurut", len(profit_orders))
            top_rule = float(profit_orders.iloc[0].get('profit_conviction_score', 0.0))
            top_hybrid = float(profit_orders.iloc[0].get('hybrid_conviction_score', top_rule))
            p2.metric("Hybrid conviction", f"{top_hybrid:.1f}", delta=f"Rule {top_rule:.1f}")
            p3.metric("P(fill × TP1-before-SL)", f"{float(profit_orders.iloc[0].get('ai_trade_success_probability_pct', np.nan)):.1f}%" if pd.notna(profit_orders.iloc[0].get('ai_trade_success_probability_pct')) else "N/A")
            p4.metric("Expected R net", f"{float(profit_orders.iloc[0].get('ai_expected_r', np.nan)):.2f}R" if pd.notna(profit_orders.iloc[0].get('ai_expected_r')) else "N/A")
            p5.metric("Bobot AI aktual", f"{float(profit_orders.iloc[0].get('ai_effective_weight_pct', 0.0)):.1f}%")
            p6.metric("Prioritas #1", str(profit_orders.iloc[0].get('ticker', '-')))
            if not bool(profit_orders.iloc[0].get('ai_can_influence_ranking', False)):
                st.info("AI masih shadow-by-evidence untuk kandidat teratas; hybrid conviction tetap sama dengan rule score. Lihat AI gate reasons pada tabel.")
            st.dataframe(
                streamlit_safe_frame(profit_orders),
                hide_index=True,
                width="stretch",
                height=520,
            )
            st.download_button(
                "Download Profit Conviction Order Builder",
                profit_orders.to_csv(index=False).encode("utf-8"),
                "profit_conviction_order_builder.csv", "text/csv", width="stretch",
            )
            ai_memory_view = focus_screens.get("ai_outcome_memory", pd.DataFrame())
            ai_audit_view = focus_screens.get("ai_model_audit", pd.DataFrame())
            ai_summary_view = focus_screens.get("ai_memory_summary", pd.DataFrame())
            a1, a2 = st.columns(2)
            with a1:
                if not ai_memory_view.empty:
                    st.download_button(
                        "Export AI outcome memory", ai_memory_view.to_csv(index=False).encode("utf-8"),
                        "idx_scanner_ai_outcome_memory.csv", "text/csv", width="stretch",
                    )
            with a2:
                st.caption(f"AI runtime: {AI_VERSION}")
            with st.expander("Audit AI lokal & pembelajaran", expanded=False):
                if not ai_summary_view.empty:
                    st.dataframe(streamlit_safe_frame(ai_summary_view), hide_index=True, width="stretch")
                if not ai_audit_view.empty:
                    st.dataframe(streamlit_safe_frame(ai_audit_view), hide_index=True, width="stretch")
                st.info(
                    "Outcome utama disimpan persisten ke database setelah migration v6. "
                    "CSV tetap tersedia sebagai backup/fail-soft."
                )
        strategy_audit = focus_screens.get("profit_strategy_audit", pd.DataFrame())
        if not strategy_audit.empty:
            with st.expander("Mengapa strategi tertentu tidak muncul di ranking?", expanded=False):
                st.dataframe(strategy_audit, hide_index=True, width="stretch")
                if len(profit_orders) and profit_orders["strategy"].nunique() == 1:
                    only_strategy = str(profit_orders.iloc[0].get("strategy", "-"))
                    st.info(
                        f"Ranking saat ini hanya berisi {only_strategy} karena strategi lain tidak memiliki status eligible, "
                        "berada di bawah minimum conviction, kalah deduplikasi ticker, atau berada di luar batas Top-N. "
                        "Ini bukan pembatasan permanen terhadap strategi lain."
                    )
        st.divider()
        if signals.empty:
            st.warning("Tidak ada setup core yang valid. Audit Universe menjelaskan alasan per ticker.")
        else:
            execution = signals[signals.get("autopilot_verified", pd.Series(False, index=signals.index)).fillna(False).astype(bool)].sort_values(
                ["analyst_fusion_score", "quality_score"], ascending=False
            )
            manual_verify = signals[signals["status"].eq("READY_FOR_STOCKBIT_VERIFY")].sort_values(
                ["analyst_fusion_score", "quality_score"], ascending=False
            )
            signal_ready = signals[signals["status"].eq("SIGNAL_READY")].sort_values(
                ["analyst_fusion_score", "quality_score"], ascending=False
            )
            entry_plans = signals[signals["status"].eq("ENTRY_PLAN_READY")].sort_values(
                ["analyst_fusion_score", "quality_score"], ascending=False
            )
            confluence_alternates = signals[signals["status"].eq("READY_NOT_SELECTED")].sort_values(
                ["analyst_fusion_score", "quality_score"], ascending=False
            )
            pending_close = signals[signals["status"].eq("PENDING_CLOSE")]
            price_verify = signals[signals["status"].eq("READY_FOR_PRICE_VERIFY")]
            st.subheader("Order Builder — CORE PLAN VERIFIED")
            st.caption("Tabel ini boleh dijadikan template Stockbit, tetapi submit tetap manual setelah pemeriksaan harga, gap pembukaan, dan spread. BUY_LIT memakai trigger terpisah dari limit.")
            if execution.empty:
                if execution_policy == "SIGNAL_FIRST":
                    st.info("SIGNAL_FIRST memang tidak menerbitkan EXECUTION_READY. Gunakan tabel READY_FOR_STOCKBIT_VERIFY di bawah untuk kandidat manual yang telah lolos seluruh gate non-akun.")
                else:
                    st.info("Tidak ada order yang lolos seluruh Account Guard saat ini. Tidak melakukan transaksi adalah hasil yang valid.")
            else:
                result_table(execution)
            if not manual_verify.empty:
                st.subheader("READY_FOR_STOCKBIT_VERIFY — kandidat manual prioritas")
                st.warning("Seluruh gate otomatis non-akun telah lolos. Sebelum order: cocokkan last price, bid/offer, spread, gap, dan batas ARA/ARB di Stockbit; tentukan lot secara manual.")
                result_table(manual_verify)
            if not signal_ready.empty:
                st.subheader("SIGNAL_READY — radar, belum boleh dibeli")
                st.info("Setup dan trigger teknikal terdeteksi, tetapi masih ada gate data/risiko/konteks yang belum lolos. Lihat Risk warnings dan Signal blockers.")
                result_table(signal_ready)
            if not entry_plans.empty:
                st.subheader("ENTRY_PLAN_READY — tunggu harga/konfirmasi")
                st.info("Zona, SL, dan target sudah valid, tetapi order belum boleh dipasang otomatis sebelum harga masuk zona atau confirmation muncul.")
                result_table(entry_plans)
            if not confluence_alternates.empty:
                st.subheader("Confluence alternate — setup kedua pada ticker yang sama")
                st.info("Setup tetap ditampilkan, tetapi tidak membuat order kedua agar risiko saham yang sama tidak terhitung dua kali.")
                result_table(confluence_alternates)
            if not pending_close.empty:
                st.subheader("PENDING_CLOSE")
                st.info("Setup teknikal sudah terbentuk dari completed EOD terakhir. Refresh setelah 16:20 WIB; belum boleh disalin sebagai order.")
                result_table(pending_close)
            if not price_verify.empty:
                st.subheader("READY_FOR_PRICE_VERIFY")
                st.warning(
                    "Harga otomatis kedua belum tervalidasi pada mode Account-Guarded. "
                    "Last price dan bid/offer wajib dicocokkan sebelum kandidat dapat dinaikkan."
                )
                result_table(price_verify)
            st.subheader("Execution funnel")
            st.dataframe(execution_funnel_summary(signals), hide_index=True, width="stretch")
            blocker_series = signals.get("primary_execution_blocker", pd.Series(dtype=str)).replace("NONE", np.nan).dropna()
            if not blocker_series.empty:
                top_blockers = blocker_series.value_counts().rename_axis("Penyebab utama").reset_index(name="Jumlah")
                st.dataframe(top_blockers, hide_index=True, width="stretch")
            technical_details = (
                signals.get("blockers", pd.Series(dtype=str)).dropna().astype(str)
                .str.split(" • ").explode().str.strip().dropna()
            )
            technical_details = technical_details[technical_details.ne("")]
            if not technical_details.empty:
                st.caption("Rincian blocker detector/tradeability paling sering")
                st.dataframe(
                    technical_details.value_counts().head(12).rename_axis("Blocker").reset_index(name="Jumlah"),
                    hide_index=True, width="stretch",
                )
            c1, c2, c3 = st.columns(3)
            c1.download_button(
                "Download semua hasil CSV",
                signals.drop(columns=["status_rank"], errors="ignore").to_csv(index=False).encode("utf-8"),
                "idx_super_scanner_results.csv", "text/csv", width="stretch",
            )
            c2.download_button(
                "Download ready-for-Stockbit CSV",
                manual_verify.drop(columns=["status_rank"], errors="ignore").to_csv(index=False).encode("utf-8"),
                "idx_ready_for_stockbit_verify.csv", "text/csv", width="stretch",
                disabled=manual_verify.empty,
            )
            c3.download_button(
                "Download execution-ready CSV",
                execution.drop(columns=["status_rank"], errors="ignore").to_csv(index=False).encode("utf-8"),
                "idx_execution_ready.csv", "text/csv", width="stretch",
                disabled=execution.empty,
            )

    with tab_daily:
        daily_board = focus_screens.get("daily_opportunities", pd.DataFrame())
        st.subheader("Daily Opportunity Board")
        st.caption("SETUP_READY/SIGNAL_READY berarti struktur valid. ORDER_READY hanya tersedia pada mode ACCOUNT_GUARDED; pada SIGNAL_FIRST ukuran order dikelola manual.")
        if daily_board.empty:
            st.info("Universe belum menghasilkan kandidat yang dapat diranking.")
        else:
            order_count = int(daily_board.get("order_ready", pd.Series(False, index=daily_board.index)).fillna(False).astype(bool).sum())
            d1, d2, d3 = st.columns(3)
            d1.metric("Kandidat harian", len(daily_board))
            d2.metric("ORDER_READY", order_count)
            d3.metric("Cash compounding", rupiah(result.get("compounding_budget_idr", 0.0)))
            st.dataframe(daily_board, hide_index=True, width="stretch")
            st.download_button("Download Daily Focus", daily_board.to_csv(index=False).encode("utf-8"), "daily_opportunity_board.csv", "text/csv", width="stretch")

    with tab_setups:
        st.caption("Setiap detector dipisahkan agar shortlist, blocker, dan level order tidak tercampur.")
        setup_specs = [
            ("Pullback Continuation", "PULLBACK_CONTINUATION", "pullback_continuation.csv"),
            ("Breakout Retest", "BREAKOUT_RETEST", "breakout_retest.csv"),
            ("Reversal Accumulation", "REVERSAL_ACCUMULATION", "reversal_accumulation.csv"),
            ("Unicorn / ICT", "UNICORN_SNIPER_ICT", "unicorn_ict.csv"),
        ]
        setup_tabs = st.tabs([item[0] for item in setup_specs])
        for setup_tab, (label, setup_code, filename) in zip(setup_tabs, setup_specs):
            with setup_tab:
                subset = signals[signals["setup"].eq(setup_code)].copy() if not signals.empty else pd.DataFrame()
                signal_count = int(subset["status"].eq("SIGNAL_READY").sum()) if not subset.empty else 0
                verify_count = int(subset["status"].eq("READY_FOR_STOCKBIT_VERIFY").sum()) if not subset.empty else 0
                ready_count = int(subset.get("autopilot_verified", pd.Series(False, index=subset.index)).fillna(False).astype(bool).sum()) if not subset.empty else 0
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Kandidat", len(subset))
                c2.metric("Signal ready", signal_count)
                c3.metric("Verify Stockbit", verify_count)
                c4.metric("Execution verified", ready_count)
                if subset.empty:
                    st.info(f"Tidak ada kandidat {label}.")
                else:
                    result_table(subset)
                    st.download_button(
                        f"Download {label}",
                        subset.drop(columns=["status_rank"], errors="ignore").to_csv(index=False).encode("utf-8"),
                        filename, "text/csv", width="stretch",
                    )


    with tab_multibagger:
        multibagger = focus_screens.get("multibagger", pd.DataFrame())
        multibagger_decision_summary = focus_screens.get(
            "multibagger_decision_summary", pd.DataFrame(),
        )
        growth_compounder = focus_screens.get(
            "multibagger_growth_compounder", pd.DataFrame(),
        )
        turnaround = focus_screens.get(
            "multibagger_turnaround", pd.DataFrame(),
        )
        st.subheader("Multibagger Research Radar")
        requested_multibagger_count = (
            len(getattr(report, "requested", []) or [])
            if report is not None
            else len(multibagger)
        )
        multibagger_diagnostics = build_multibagger_diagnostic_views(
            multibagger,
            expected_ticker_count=requested_multibagger_count,
            limit=30,
        )
        coverage_view = multibagger_diagnostics["coverage"]
        if not coverage_view.empty:
            coverage_counts = coverage_view.set_index("stage")["count"].to_dict()
            q1, q2, q3, q4, q5 = st.columns(5)
            q1.metric(
                "OHLCV siap",
                int(coverage_counts.get("OHLCV siap dinilai", 0)),
                f"dari {int(coverage_counts.get('Universe diminta', 0))}",
            )
            q2.metric(
                "Snapshot fundamental",
                int(coverage_counts.get("Snapshot fundamental tersedia", 0)),
            )
            q3.metric(
                "Histori laporan",
                int(coverage_counts.get("Histori laporan tersedia", 0)),
            )
            q4.metric(
                "Lolos riset",
                int(coverage_counts.get("Lolos gate riset", 0)),
            )
            q5.metric(
                "Menunggu data",
                int(coverage_counts.get("Menunggu evidence fundamental", 0)),
            )
            if int(coverage_counts.get("OHLCV siap dinilai", 0)) < int(
                coverage_counts.get("Universe diminta", 0)
            ):
                failed_names = list(
                    (getattr(report, "failed", {}) or {}).keys()
                ) if report is not None else []
                detail = (
                    " Ticker: " + ", ".join(failed_names[:12]) + "."
                    if failed_names else ""
                )
                st.warning(
                    "Sebagian ticker CSV tidak mencapai tahap OHLCV/prepared."
                    + detail
                    + " Periksa Audit coverage untuk alasan provider/minimum bar."
                )
            if int(coverage_counts.get("Histori laporan tersedia", 0)) == 0:
                st.warning(
                    "Radar qualified kosong karena histori laporan belum berhasil "
                    "dinormalisasi/terverifikasi. Ini bukan kesimpulan bahwa seluruh "
                    "universe tidak memiliki potensi Multibagger."
                )
        growth_tab, turnaround_tab = st.tabs([
            "Growth Compounder", "Turnaround / Cyclical",
        ])
        radar_columns = [
            "ticker", "multibagger_production_rank", "multibagger_evidence_class",
            "multibagger_rank_eligible", "multibagger_score_comparability_pct",
            "multibagger_lane", "multibagger_candidate_type",
            "growth_compounder_selection_score", "turnaround_selection_score",
            "growth_compounder_score", "turnaround_recovery_score",
            "turnaround_research_state", "turnaround_recovery_signals",
            "effective_silent_accumulation_score",
            "stock_universe_familiarity_score",
            "stock_universe_familiarity_coverage_pct",
            "stock_universe_familiarity_state",
            "smart_money_behavior_score",
            "smart_money_behavior_coverage_pct",
            "smart_money_behavior_state",
            "smart_money_flow_evidence_mode",
            "distribution_severity_score", "distribution_penalty_points",
            "distribution_evidence_state", "broker_summary_score",
            "narrative_lifecycle_score",
            "narrative_lifecycle_state",
            "flow_preceded_narrative",
            "emir_method_score",
            "emir_method_coverage_pct",
            "emir_method_score_state",
            "emir_method_state",
            "emir_method_production_eligible",
            "emir_method_reliability_pct",
            "emir_position_cap_pct",
            "emir_selection_reason",
            "emir_risk_flags",
            "growth_narrative_contribution_points",
            "growth_emir_contribution_points",
            "turnaround_narrative_contribution_points",
            "turnaround_emir_contribution_points",
            "multibagger_final_score_formula",
            "narrative_flow_effective_score",
            "narrative_flow_research_score",
            "narrative_flow_score_state",
            "narrative_flow_convergence_state",
            "narrative_evidence_coverage_pct",
            "narrative_evidence_mode",
            "structured_financial_evidence_state",
            "structured_financial_evidence_coverage_pct",
            "structured_financial_source_count",
            "structured_financial_latest_period",
            "evidence_acquisition_missing",
            "narrative_score_state",
            "operating_narrative_proxy_score",
            "operating_narrative_proxy_coverage_pct",
            "issuer_alignment_effective_score",
            "issuer_alignment_score_state",
            "issuer_alignment_evidence_basis",
            "issuer_alignment_state", "retail_adoption_stage",
            "narrative_conversion_rate_20d_pct",
            "narrative_conversion_resolved_20d",
            "narrative_crowding_risk_score", "narrative_hard_block",
            "multibagger_scoring_state",
            "multibagger_metric_coverage_pct",
            "multibagger_metric_data_gate",
            "fundamental_complete_for_multibagger",
            "fundamental_acquisition_state",
            "fundamental_missing_core_fields",
            "fundamental_statement_family_coverage_pct",
            "fundamental_history_period_coverage_pct",
            "fundamental_income_statement_coverage_pct",
            "fundamental_balance_sheet_coverage_pct",
            "fundamental_cashflow_statement_coverage_pct",
            "growth_pillar_coverage_pct",
            "profitability_pillar_coverage_pct",
            "cashflow_pillar_coverage_pct",
            "safety_pillar_coverage_pct",
            "runway_pillar_coverage_pct",
            "valuation_pillar_coverage_pct",
            "silent_accumulation_state", "overall_research_confidence",
            "research_eligible", "portfolio_allocation_eligible",
            "technical_entry_state", "selected_reason", "trigger_waiting",
            "invalidation_reason", "primary_risk",
        ]
        # Keep the main radar decision-dense. The complete evidence matrix is
        # still available below as a separate audit view/download.
        radar_columns = [
            "ticker", "multibagger_production_rank",
            "multibagger_proxy_research_rank", "multibagger_evidence_class",
            "multibagger_rank_eligible", "multibagger_proxy_rank_eligible",
            "multibagger_lane", "multibagger_status",
            "multibagger_selection_score",
            "effective_silent_accumulation_score",
            "execution_readiness_score", "research_eligible",
            "multibagger_score_comparability_pct",
            "multibagger_proxy_comparability_pct",
            "multibagger_direct_event_verified",
            "multibagger_direct_flow_verified",
            "multibagger_direct_alignment_verified",
            "fundamental_score", "fundamental_coverage",
            "narrative_effective_score", "issuer_alignment_effective_score",
            "broker_summary_score", "emir_method_score",
            "emir_method_production_eligible", "technical_entry_state",
            "selected_reason", "trigger_waiting", "primary_risk",
        ]
        provisional_columns = [
            "research_queue_rank", "ticker", "candidate_state",
            "growth_provisional_priority_score",
            "turnaround_provisional_priority_score",
            "growth_compounder_selection_score",
            "turnaround_selection_score", "growth_compounder_score",
            "turnaround_recovery_score",
            "effective_silent_accumulation_score",
            "narrative_flow_effective_score",
            "narrative_flow_research_score",
            "narrative_flow_score_state",
            "narrative_flow_convergence_state",
            "issuer_alignment_effective_score",
            "issuer_alignment_score_state",
            "issuer_alignment_evidence_basis",
            "retail_adoption_stage",
            "selector_trend_score", "selector_relative_strength_score",
            "fundamental_score", "fundamental_coverage",
            "multibagger_scoring_state",
            "multibagger_metric_coverage_pct",
            "multibagger_metric_data_gate",
            "fundamental_data_grade", "fundamental_history_source_count",
            "overall_research_confidence", "near_miss_state",
            "next_required_evidence", "selected_reason",
            "not_entry_reason", "trigger_waiting", "invalidation_reason",
            "primary_risk", "near_miss_reason", "capital_state",
        ]
        with growth_tab:
            if isinstance(growth_compounder, pd.DataFrame) and not growth_compounder.empty:
                st.dataframe(
                    streamlit_safe_frame(
                        growth_compounder.loc[:, [
                            column for column in radar_columns
                            if column in growth_compounder.columns
                        ]].head(15)
                    ),
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.info(
                    "Belum ada Growth Compounder yang lolos gate riset. "
                    "Berikut kandidat provisional terdekat; statusnya bukan "
                    "rekomendasi beli dan alokasi tetap 0."
                )
                growth_queue = multibagger_diagnostics[
                    "growth_research_queue"
                ]
                if not growth_queue.empty:
                    st.dataframe(
                        streamlit_safe_frame(
                            growth_queue.loc[:, [
                                column for column in provisional_columns
                                if column in growth_queue.columns
                            ]].head(15)
                        ),
                        hide_index=True,
                        width="stretch",
                    )
                else:
                    pending_queue = multibagger_diagnostics["data_pending"]
                    if not pending_queue.empty:
                        st.warning(
                            "Belum ada near-miss yang dapat diskor. Berikut "
                            "prioritas backfill data; baris ini belum merupakan "
                            "kandidat Multibagger."
                        )
                        st.dataframe(
                            streamlit_safe_frame(
                                pending_queue.loc[:, [
                                    column for column in [
                                        "research_queue_rank", "ticker",
                                        "data_refresh_priority_score",
                                        "effective_silent_accumulation_score",
                                        "selector_trend_score",
                                        "selector_relative_strength_score",
                                        "adtv20_idr",
                                        "next_required_evidence",
                                        "near_miss_reason", "capital_state",
                                    ]
                                    if column in pending_queue.columns
                                ]].head(15)
                            ),
                            hide_index=True,
                            width="stretch",
                        )
                    else:
                        st.caption(
                            "Belum ada near-miss maupun baris data-pending; "
                            "periksa kegagalan OHLCV/provider pada Audit coverage."
                        )
        with turnaround_tab:
            if isinstance(turnaround, pd.DataFrame) and not turnaround.empty:
                st.dataframe(
                    streamlit_safe_frame(
                        turnaround.loc[:, [
                            column for column in radar_columns
                            if column in turnaround.columns
                        ]].head(15)
                    ),
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "Turnaround/Cyclical adalah radar riset. Alokasi modal tetap "
                    "nol sampai kandidat juga lolos gate Multibagger A/B."
                )
            else:
                st.info(
                    "Belum ada Turnaround/Cyclical dengan minimal dua sinyal "
                    "recovery dan tanpa risiko kritis. Berikut kandidat "
                    "provisional; alokasi tetap 0."
                )
                turnaround_queue = multibagger_diagnostics[
                    "turnaround_research_queue"
                ]
                if not turnaround_queue.empty:
                    st.dataframe(
                        streamlit_safe_frame(
                            turnaround_queue.loc[:, [
                                column for column in provisional_columns
                                if column in turnaround_queue.columns
                            ]].head(15)
                        ),
                        hide_index=True,
                        width="stretch",
                    )
                else:
                    pending_queue = multibagger_diagnostics["data_pending"]
                    if not pending_queue.empty:
                        st.warning(
                            "Belum ada recovery near-miss yang dapat diskor. "
                            "Berikut prioritas backfill data; baris ini belum "
                            "merupakan kandidat Turnaround/Cyclical."
                        )
                        st.dataframe(
                            streamlit_safe_frame(
                                pending_queue.loc[:, [
                                    column for column in [
                                        "research_queue_rank", "ticker",
                                        "data_refresh_priority_score",
                                        "effective_silent_accumulation_score",
                                        "selector_trend_score",
                                        "selector_relative_strength_score",
                                        "adtv20_idr",
                                        "next_required_evidence",
                                        "near_miss_reason", "capital_state",
                                    ]
                                    if column in pending_queue.columns
                                ]].head(15)
                            ),
                            hide_index=True,
                            width="stretch",
                        )
                    else:
                        st.caption(
                            "Belum ada near-miss maupun baris data-pending; "
                            "periksa kegagalan OHLCV/provider pada Audit coverage."
                        )
        with st.expander(
            "Narrative Intelligence — Event, Alignment, Conversion & Flow",
            expanded=False,
        ):
            narrative_profiles_view = focus_screens.get(
                "narrative_profiles", pd.DataFrame(),
            )
            narrative_events_view = focus_screens.get(
                "narrative_events", pd.DataFrame(),
            )
            narrative_outcomes_view = focus_screens.get(
                "narrative_event_outcomes", pd.DataFrame(),
            )
            narrative_audit_view = focus_screens.get(
                "narrative_engine_audit", pd.DataFrame(),
            )
            st.caption(
                "Retail Adoption adalah proxy attention berbasis event, volume, "
                "return, dan kedekatan high—bukan identitas investor ritel. "
                "Narrative Conversion 5D/20D/60D memakai directional excess return "
                "net terhadap IHSG dari sesi selesai pertama setelah detected_at. "
                "Metrik conversion tetap shadow sebelum sampel minimum. Scanner sekarang "
                "mengisi evidence melalui jalur official-first, issuer IR, public provider lineage, "
                "dan dated structured financial evidence. Operating facts dapat menjadi skor produksi "
                "bila sumber, periode laporan, freshness, dan coverage lolos; nilai kosong hanya muncul "
                "setelah seluruh jalur acquisition gagal atau data memang tidak dapat diaudit. "
                "Silent Accumulation tetap menjadi ranking key terpisah agar tidak dihitung berulang. Emir Public-Framework Layer menguji stock-universe familiarity, urutan flow→story, smart-money proxy/direct broker evidence, crowding, distribusi, dan position cap. Klaim performa pihak publik tidak dimasukkan sebagai parameter model."
            )
            if not narrative_audit_view.empty:
                st.dataframe(
                    streamlit_safe_frame(narrative_audit_view),
                    hide_index=True,
                    width="stretch",
                )
            if narrative_profiles_view.empty:
                st.info("Narrative profile belum terbentuk.")
            else:
                profile_columns = [
                    column for column in (
                        "ticker", "narrative_state",
                        "stock_universe_familiarity_score",
                        "stock_universe_familiarity_coverage_pct",
                        "stock_universe_familiarity_state",
                        "smart_money_behavior_score",
                        "smart_money_behavior_coverage_pct",
                        "smart_money_behavior_state",
                        "smart_money_flow_evidence_mode",
                        "broker_summary_score",
                        "narrative_lifecycle_score",
                        "narrative_lifecycle_state",
                        "flow_preceded_narrative",
                        "emir_method_score",
                        "emir_method_coverage_pct",
                        "emir_method_score_state",
                        "emir_method_state",
                        "emir_method_production_eligible",
                        "emir_method_reliability_pct",
                        "emir_position_cap_pct",
                        "emir_selection_reason",
                        "emir_risk_flags",
                        "narrative_effective_score",
                        "narrative_evidence_coverage_pct",
                        "narrative_evidence_mode",
                        "structured_financial_evidence_state",
                        "structured_financial_evidence_coverage_pct",
                        "structured_financial_source_count",
                        "structured_financial_latest_period",
                        "evidence_acquisition_missing",
                        "narrative_score_state",
                        "operating_narrative_proxy_score",
                        "operating_narrative_proxy_coverage_pct",
                        "operating_narrative_proxy_state",
                        "operating_narrative_proxy_basis",
                        "issuer_alignment_effective_score",
                        "issuer_alignment_coverage_pct",
                        "issuer_alignment_score_state",
                        "issuer_alignment_evidence_basis",
                        "issuer_alignment_state",
                        "retail_adoption_stage",
                        "retail_adoption_proxy_score",
                        "retail_adoption_proxy_coverage_pct",
                        "narrative_flow_effective_score",
                        "narrative_flow_research_score",
                        "narrative_flow_score_state",
                        "narrative_flow_convergence_state",
                        "narrative_flow_convergence_coverage_pct",
                        "narrative_conversion_rate_5d_pct",
                        "narrative_conversion_expectancy_5d_pct",
                        "narrative_conversion_resolved_5d",
                        "narrative_conversion_state_5d",
                        "narrative_conversion_rate_20d_pct",
                        "narrative_conversion_expectancy_20d_pct",
                        "narrative_conversion_resolved_20d",
                        "narrative_conversion_state_20d",
                        "narrative_conversion_rate_60d_pct",
                        "narrative_conversion_resolved_60d",
                        "narrative_crowding_risk_score",
                        "narrative_missing_source_event_count",
                        "narrative_inactive_lifecycle_event_count",
                        "narrative_entity_unverified_event_count",
                        "narrative_production_policy",
                        "narrative_hard_block",
                        "latest_narrative_event",
                        "narrative_primary_reason",
                        "narrative_primary_risk",
                    ) if column in narrative_profiles_view.columns
                ]
                sorted_profiles = narrative_profiles_view.sort_values(
                    [
                        "narrative_hard_block",
                        "narrative_flow_effective_score",
                    ],
                    ascending=[True, False],
                    kind="stable",
                    na_position="last",
                )
                st.dataframe(
                    streamlit_safe_frame(
                        sorted_profiles[profile_columns].head(60)
                    ),
                    hide_index=True,
                    width="stretch",
                    height=430,
                )
            if not narrative_events_view.empty:
                st.markdown("**Event ledger point-in-time**")
                event_columns = [
                    column for column in (
                        "detected_at", "ticker", "event_type",
                        "impact_direction", "headline",
                        "source_family", "source_url",
                        "source_hostname", "source_state", "source_present",
                        "official_verified", "registered_official_domain",
                        "entity_match_state", "event_status",
                        "lifecycle_evidence_state", "resolution_source_url",
                        "source_quality_score",
                        "materiality_score", "novelty_score",
                        "financial_bridge_score",
                        "narrative_decay_weight",
                        "event_evidence_state",
                    ) if column in narrative_events_view.columns
                ]
                st.dataframe(
                    streamlit_safe_frame(
                        narrative_events_view[event_columns].head(100)
                    ),
                    hide_index=True,
                    width="stretch",
                    height=360,
                )
                st.download_button(
                    "Download Narrative Event Ledger",
                    narrative_events_view.to_csv(index=False).encode("utf-8"),
                    "narrative_event_ledger_v763.csv",
                    "text/csv",
                    width="stretch",
                )
            if not narrative_outcomes_view.empty:
                resolved_count = int(pd.to_numeric(
                    narrative_outcomes_view.get(
                        "directional_excess_return_20d_pct",
                        pd.Series(
                            np.nan,
                            index=narrative_outcomes_view.index,
                        ),
                    ),
                    errors="coerce",
                ).notna().sum())
                st.caption(
                    f"Outcome tersimpan: {len(narrative_outcomes_view)}; "
                    f"20D resolved: {resolved_count}. Engine {NARRATIVE_ENGINE_VERSION}."
                )
        st.markdown("#### Coverage, Near-Miss & Antrean Data")
        st.caption(
            "Near-miss bukan rekomendasi beli. Tabel ini sengaja mempertahankan "
            "calon yang belum lolos agar kegagalan data tidak disalahartikan "
            "sebagai kualitas bisnis nol; alokasi modal tetap dipaksa 0."
        )
        diag_coverage, diag_scores, diag_growth, diag_turnaround, diag_pending, diag_gates = st.tabs(
            [
                "Coverage",
                "Audit Nilai Sama",
                "Growth Near-Miss",
                "Turnaround Near-Miss",
                "Data Pending",
                "Gate Blockers",
            ]
        )
        with diag_coverage:
            if coverage_view.empty:
                st.info("Belum ada baris Multibagger untuk diaudit.")
            else:
                st.dataframe(
                    streamlit_safe_frame(coverage_view),
                    hide_index=True,
                    width="stretch",
                )
        with diag_scores:
            score_dispersion_view = multibagger_diagnostics.get(
                "score_dispersion", pd.DataFrame()
            )
            st.caption(
                "Audit ini membedakan nilai yang benar-benar konstan dari nilai yang "
                "belum dapat diskor. Coverage boleh sama antar-emiten karena mengukur "
                "ketersediaan field, bukan kualitas bisnis. Narrative/alignment/conversion "
                "tanpa evidence ditampilkan kosong, bukan neutral 50."
            )
            if score_dispersion_view.empty:
                st.info("Belum ada metrik lintas-emiten untuk diaudit.")
            else:
                st.dataframe(
                    streamlit_safe_frame(score_dispersion_view),
                    hide_index=True,
                    width="stretch",
                )
        with diag_growth:
            growth_near_miss = multibagger_diagnostics["growth_near_miss"]
            if growth_near_miss.empty:
                st.info("Belum ada Growth near-miss yang dapat diskor.")
            else:
                st.dataframe(
                    streamlit_safe_frame(growth_near_miss),
                    hide_index=True,
                    width="stretch",
                )
        with diag_turnaround:
            turnaround_near_miss = multibagger_diagnostics["turnaround_near_miss"]
            if turnaround_near_miss.empty:
                st.info("Belum ada Turnaround near-miss yang dapat diskor.")
            else:
                st.dataframe(
                    streamlit_safe_frame(turnaround_near_miss),
                    hide_index=True,
                    width="stretch",
                )
        with diag_pending:
            data_pending_view = multibagger_diagnostics["data_pending"]
            if data_pending_view.empty:
                st.success("Tidak ada ticker yang menunggu snapshot fundamental.")
            else:
                st.dataframe(
                    streamlit_safe_frame(data_pending_view),
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "data_refresh_priority_score hanya mengurutkan antrean backfill "
                    "berdasarkan teknikal, relative strength, Silent Accumulation, "
                    "dan likuiditas; bukan skor Multibagger."
                )
        with diag_gates:
            gate_failure_view = multibagger_diagnostics["gate_failures"]
            if gate_failure_view.empty:
                st.success("Tidak ada gate riset yang gagal.")
            else:
                st.dataframe(
                    streamlit_safe_frame(gate_failure_view),
                    hide_index=True,
                    width="stretch",
                )
        st.subheader("Multibagger Capital Allocation")
        budget = float(result.get("compounding_budget_idr", 0.0) or 0.0)
        accumulate_now = int(multibagger.get("compounding_state", pd.Series(dtype=str)).isin(["ACCUMULATE_NOW", "STARTER_NOW"]).sum()) if not multibagger.empty else 0
        deployed = float(pd.to_numeric(multibagger.get("estimated_order_value_idr", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not multibagger.empty else 0.0
        reserve = max(0.0, budget - deployed)
        top_destination = "CASH_RESERVE"
        top_amount = 0.0
        capital_view = multibagger.copy() if isinstance(multibagger, pd.DataFrame) else pd.DataFrame()
        if not capital_view.empty:
            capital_view["_capital_selected"] = capital_view.get(
                "portfolio_allocation_eligible",
                capital_view.get(
                    "allocation_eligible",
                    pd.Series(False, index=capital_view.index),
                ),
            ).fillna(False).astype(bool)
            capital_view = capital_view.sort_values(
                [
                    "_capital_selected", "capital_priority_rank",
                    "capital_priority_score", "multibagger_selection_score",
                ],
                ascending=[False, True, False, False],
                kind="stable",
                na_position="last",
            ).drop(columns="_capital_selected")
        if not multibagger.empty:
            selected_capital = capital_view[
                pd.to_numeric(
                    capital_view.get(
                        "strategic_target_amount_idr",
                        pd.Series(0.0, index=capital_view.index),
                    ),
                    errors="coerce",
                ).fillna(0.0).gt(0.0)
            ]
            if not selected_capital.empty:
                top_destination = str(
                    selected_capital.iloc[0].get("ticker", "CASH_RESERVE")
                )
                top_amount = float(
                    selected_capital.iloc[0].get(
                        "strategic_target_amount_idr", 0.0,
                    ) or 0.0
                )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Budget Multibagger", rupiah(budget))
        c2.metric("Deploy sekarang", rupiah(deployed))
        c3.metric("Cash reserve", rupiah(reserve))
        c4.metric("Seleksi #1", top_destination, rupiah(top_amount))
        st.caption(
            "Multibagger Quality v7.16.3 memisahkan Growth Compounder dari Turnaround/Cyclical serta direct evidence dari proxy coverage, menjaga seluruh ticker tetap terlihat walau history pendek, dan menghitung valuation secara currency/split-aware dengan lineage. Growth menilai growth persistence, profitability termasuk gross profitability, cash conversion/accrual quality, balance-sheet safety, serta reinvestment runway yang berbobot langsung 15%. "
            "Turnaround memakai infleksi point-in-time, akselerasi pendapatan/laba, pemulihan margin dan cash conversion, safety, serta katalis; governance/accounting risk tetap hard blocker. "
            "Execution Readiness dihitung terpisah dari momentum, Silent Accumulation v4, Core Swing, dan objective EOFF timing; timing tidak dapat mengubah bisnis lemah menjadi kandidat berkualitas. "
            "Broker summary hanya menjadi proxy kecil bila data asli bertimestamp tersedia. Project/management memprioritaskan IDX/OJK dan IR emiten; single-source atau modelled impact diberi confidence lebih rendah. "
            "Deploy now hanya aktif saat zona entry valid, dan cap per saham mencegah all-in tunggal."
        )
        if not multibagger.empty:
            allocation_columns = list(dict.fromkeys(
                column for column in [
                    "multibagger_selection_rank", "ticker", "multibagger_selection_score",
                    "multibagger_lane", "growth_compounder_selection_score",
                    "turnaround_selection_score", "growth_compounder_score",
                    "turnaround_recovery_score", "turnaround_research_state",
                    "turnaround_recovery_signals", "turnaround_gate_reasons",
                    "research_eligible", "research_eligibility_reason",
                    "portfolio_allocation_eligible",
                    "multibagger_status", "multibagger_quality_score",
                    "growth_persistence_pillar", "profitability_pillar",
                    "cash_conversion_pillar", "balance_sheet_safety_pillar",
                    "reinvestment_runway_pillar", "quality_pillar_coverage_pct",
                    "fundamental_inflection_score", "fundamental_inflection_coverage_pct",
                    "revenue_growth_acceleration", "earnings_growth_acceleration",
                    "gross_margin_change_yoy", "cash_conversion_change_yoy",
                    "quality_pillars_strong", "quality_pillar_gate",
                    "sector", "sector_peer_count", "sector_relative_quality_score",
                    "sector_relative_strength_score", "sector_relative_state",
                    "silent_accumulation_score", "effective_silent_accumulation_score",
                    "stock_universe_familiarity_score",
                    "stock_universe_familiarity_coverage_pct",
                    "stock_universe_familiarity_state",
                    "smart_money_behavior_score",
                    "smart_money_behavior_coverage_pct",
                    "smart_money_behavior_state",
                    "smart_money_flow_evidence_mode",
                    "broker_summary_score",
                    "narrative_lifecycle_score",
                    "narrative_lifecycle_state",
                    "flow_preceded_narrative",
                    "emir_method_score",
                    "emir_method_coverage_pct",
                    "emir_method_score_state",
                    "emir_method_state",
                    "emir_method_production_eligible",
                    "emir_method_reliability_pct",
                    "emir_position_cap_pct",
                    "emir_selection_reason",
                    "emir_risk_flags",
                    "growth_narrative_contribution_points",
                    "growth_emir_contribution_points",
                    "turnaround_narrative_contribution_points",
                    "turnaround_emir_contribution_points",
                    "multibagger_final_score_formula",
                    "narrative_effective_score",
                    "narrative_evidence_coverage_pct",
                    "narrative_evidence_mode",
                    "structured_financial_evidence_state",
                    "structured_financial_evidence_coverage_pct",
                    "structured_financial_source_count",
                    "structured_financial_latest_period",
                    "evidence_acquisition_status",
                    "evidence_acquisition_complete",
                    "evidence_acquisition_missing",
                    "narrative_score_state",
                    "operating_narrative_proxy_score",
                    "operating_narrative_proxy_coverage_pct",
                    "issuer_alignment_effective_score",
                    "issuer_alignment_score_state",
                    "issuer_alignment_evidence_basis",
                    "issuer_alignment_state",
                    "retail_adoption_stage",
                    "narrative_flow_effective_score",
                    "narrative_flow_research_score",
                    "narrative_flow_score_state",
                    "narrative_flow_convergence_state",
                    "narrative_conversion_rate_20d_pct",
                    "narrative_conversion_resolved_20d",
                    "narrative_crowding_risk_score",
                    "narrative_hard_block",
                    "selector_trend_score",
                    "selector_relative_strength_score", "multibagger_timing_selector_score",
                    "selector_expected_excess_return_20d_pct", "selector_expected_excess_return_60d_pct",
                    "selector_model_state", "selected_reason", "not_entry_reason",
                    "trigger_waiting", "invalidation_reason", "primary_risk",
                    "capital_priority_rank", "capital_tier", "capital_conviction_score",
                    "multibagger_quality_score", "confidence_adjusted_multibagger_score", "execution_readiness_score",
                    "overall_research_confidence", "overall_research_confidence_grade",
                    "data_confidence_score", "fundamental_confidence_score", "future_fundamental_confidence_score",
                    "technical_confidence_score", "eoff_confidence_score",
                    "top_positive_drivers", "top_negative_drivers", "scoring_reason_codes",
                    "research_recommendation_status", "multibagger_candidate_type", "multibagger_status", "compounding_state",
                    "economic_earnings_score", "economic_earnings_confidence", "economic_earnings_state",
                    "ocf_ebitda_conversion", "minority_leakage_pct",
                    "silent_accumulation_version", "silent_accumulation_score", "silent_accumulation_state", "silent_accumulation_confidence",
                    "silent_accumulation_base_score_v2", "silent_accumulation_v3_adjustment", "silent_accumulation_v4_adjustment",
                    "accumulation_persistence_score", "accumulation_positive_windows_pct", "accumulation_longest_run",
                    "accumulation_regime", "accumulation_weight_profile",
                    "absorption_confirmed_days20", "failed_absorption_days20", "effort_result_absorption20",
                    "effort_result_distribution20", "persistent_bid_score", "supply_pressure_ratio20",
                    "distribution_days20", "up_down_value_ratio20", "pullback_volume_ratio20", "broker_flow_signal",
                    "project_pipeline_score", "project_stage", "project_stage_probability_pct", "project_success_probability_pct",
                    "management_quality_score", "forward_quality_coverage", "future_fundamental_impact_score",
                    "future_impact_confidence", "future_revenue_uplift_base_pct", "future_ebitda_uplift_base_pct",
                    "future_net_profit_uplift_base_pct", "future_fcf_pressure_idr", "future_net_debt_change_pct",
                    "multibagger_time_cycle_score", "time_cycle_capital_weight_pct", "quick_buy_action",
                    "best_buy_date", "best_buy_window_start", "best_buy_window_end", "best_buy_score",
                    "best_buy_confidence", "best_buy_entry_low", "best_buy_entry_high", "best_buy_trigger",
                    "best_buy_stop_loss", "best_buy_tp1", "best_buy_tp2", "eoff_strength_label",
                    "eoff_reconstruction_score", "eoff_signal_active", "eoff_direction_bias",
                    "eoff_fib_cluster_count", "eoff_reversal_date", "eoff_astro_score", "eoff_adaptive_total_weight_pct", "eoff_validation_path",
                    "project_source_families", "project_source_quorum_verified", "project_source_urls",
                    "management_source_urls", "ceo_name", "project_names",
                    "strategic_target_weight_pct", "deploy_now_weight_pct", "strategic_target_amount_idr",
                    "recommended_allocation_idr", "recommended_lots", "allocation_action", "allocation_reason", "red_flags",
                ] if column in multibagger.columns
            ))
            st.markdown("#### Capital destination ranking")
            st.dataframe(
                streamlit_safe_frame(capital_view.loc[:, allocation_columns].head(10)),
                hide_index=True,
                width="stretch",
            )
        forward_report = result.get("automatic_forward_report", pd.DataFrame())
        if isinstance(forward_report, pd.DataFrame) and not forward_report.empty:
            with st.expander("Audit pencarian forward intelligence otomatis", expanded=False):
                st.dataframe(streamlit_safe_frame(forward_report), hide_index=True, width="stretch")
                st.caption("AUTO_VERIFIED berarti minimal dua keluarga sumber resmi berbeda. AUTO_SINGLE_SOURCE tetap dipakai dengan confidence lebih rendah.")
        st.markdown("#### Ringkasan keputusan Multibagger")
        focus_table(multibagger_decision_summary, height=500)
        render_focus_download(
            "Download Ringkasan Keputusan",
            multibagger_decision_summary,
            "multibagger_decision_summary.csv",
        )
        with st.expander("Full Multibagger evidence — audit teknis", expanded=False):
            focus_table(multibagger, height=560)
            render_focus_download(
                "Download Full Evidence Audit", multibagger,
                "multibagger_full_evidence_audit.csv",
            )



    with tab_portfolio:
        render_portfolio_panel(result)

    with tab_chart:
        @st.fragment
        def render_signal_chart_fragment() -> None:
            if signals.empty:
                st.info("Belum ada setup untuk digambar.")
                return
            labels = [f"{row.ticker} · {row.setup}" for row in signals.itertuples()]
            key = "chart_setup_selector_v460"
            prior = st.session_state.get(key)
            if prior not in labels:
                st.session_state[key] = labels[0]
            selected = st.selectbox("Pilih setup", labels, key=key)
            selected_pos = labels.index(selected)
            signal = signals.iloc[selected_pos].to_dict()
            frame = result["prepared"][signal["ticker"]]
            st.plotly_chart(
                make_signal_chart(frame, signal),
                width="stretch",
                key=f"signal_chart_{signal['ticker']}_{signal['setup']}",
            )
            st.write("Evidence:", signal.get("evidence", "—"))
            if signal.get("signal_risk_warnings"):
                st.warning("Risk disclosure: " + str(signal["signal_risk_warnings"]))
            if signal.get("blockers"):
                st.caption("Detector notes: " + str(signal["blockers"]))
        render_signal_chart_fragment()


    with tab_bridge:
        st.subheader("TradingView confirmation bridge")
        st.caption(
            "Pilih kandidat dari tabel, buka simbol pada TradingView, lalu salin entry zone, order entry, SL, TP1, dan TP2 "
            "ke mode MANUAL SCANNER LEVELS. Pine Script tidak membaca dataframe Streamlit atau Stockbit secara langsung."
        )
        bridge = build_tradingview_bridge(signals, focus_screens)
        if bridge.empty:
            st.info("Belum ada level scanner yang dapat diekspor.")
        else:
            st.dataframe(
                bridge,
                hide_index=True,
                width="stretch",
                column_config={
                    "ticker": st.column_config.TextColumn("Ticker", pinned=True),
                    "tv_symbol": st.column_config.TextColumn("TradingView"),
                    "entry_low": st.column_config.NumberColumn("Entry low", format="Rp %.0f"),
                    "entry_high": st.column_config.NumberColumn("Entry high", format="Rp %.0f"),
                    "entry": st.column_config.NumberColumn("Order entry", format="Rp %.0f"),
                    "reclaim_trigger_price": st.column_config.NumberColumn("Reclaim trigger", format="Rp %.0f"),
                    "retest_reference_price": st.column_config.NumberColumn("Retest reference", format="Rp %.0f"),
                    "trigger_basis": st.column_config.TextColumn("Trigger basis"),
                    "trigger_instruction": st.column_config.TextColumn("Trigger instruction"),
                    "trigger_valid_until": st.column_config.DatetimeColumn("Trigger valid until", format="DD MMM YYYY"),
                    "stop_loss": st.column_config.NumberColumn("SL", format="Rp %.0f"),
                    "tp1": st.column_config.NumberColumn("TP1", format="Rp %.0f"),
                    "tp2": st.column_config.NumberColumn("TP2", format="Rp %.0f"),
                    "rr1": st.column_config.NumberColumn("RR1", format="%.2f"),
                    "rr2": st.column_config.NumberColumn("RR2", format="%.2f"),
                },
            )
            st.download_button(
                "Download TradingView bridge CSV",
                bridge.to_csv(index=False).encode("utf-8"),
                "tradingview_scanner_bridge.csv",
                "text/csv",
                width="stretch",
            )
        pine_file = APP_ROOT / "IDX_Scanner_Confirmation_v1.pine"
        stockbit_file = APP_ROOT / "STOCKBIT_SCREENER_PRESETS.md"
        c1, c2 = st.columns(2)
        if pine_file.is_file():
            c1.download_button(
                "Download Pine Script indicator",
                pine_file.read_bytes(),
                pine_file.name,
                "text/plain",
                width="stretch",
            )
        else:
            c1.warning("File Pine Script tidak ditemukan dalam deployment.")
        if stockbit_file.is_file():
            c2.download_button(
                "Download preset screener Stockbit",
                stockbit_file.read_bytes(),
                stockbit_file.name,
                "text/markdown",
                width="stretch",
            )
        else:
            c2.warning("Panduan screener Stockbit tidak ditemukan dalam deployment.")
        st.markdown(
            """
            **Urutan eksekusi:** Stockbit menyaring universe → scanner menilai setup, data, dan risiko → TradingView mengonfirmasi struktur/bar penutup → order tetap dimasukkan manual sebagai limit order di Stockbit.

            **Peringatan:** directional score pada indikator adalah skor konfluensi berbasis aturan, bukan probabilitas statistik dan bukan prediksi yang dijamin.
            """
        )

    with tab_validation:
        stats: pd.DataFrame = result.get("validation_stats", pd.DataFrame())
        trades: pd.DataFrame = result.get("validation_trades", pd.DataFrame())
        validation_universe_audit: pd.DataFrame = result.get(
            "validation_universe_audit",
            pd.DataFrame(),
        )
        if not validation_universe_audit.empty:
            selected_count = int(
                validation_universe_audit["selected"].fillna(False).astype(bool).sum()
            )
            eligible_count = int(len(validation_universe_audit))
            st.caption(
                f"Cohort OOS: {selected_count}/{eligible_count} ticker eligible. "
                "Pemilihan deterministik berstrata likuiditas dan tidak memakai Final Score."
            )
            expansion_audit = validation_universe_audit.attrs.get("expansion_audit", pd.DataFrame())
            if isinstance(expansion_audit, pd.DataFrame) and not expansion_audit.empty:
                last_step = expansion_audit.iloc[-1]
                st.caption(
                    "Adaptive OOS expansion: "
                    f"{int(last_step.get('target_tickers', selected_count))} ticker, "
                    f"{int(last_step.get('oos_events', len(trades)))} genuine events; "
                    f"target_met={bool(last_step.get('evidence_target_met', False))}."
                )
            with st.expander("Audit cohort validasi", expanded=False):
                st.dataframe(
                    validation_universe_audit,
                    width="stretch",
                    hide_index=True,
                )
                if isinstance(expansion_audit, pd.DataFrame) and not expansion_audit.empty:
                    st.markdown("**Adaptive expansion audit**")
                    st.dataframe(expansion_audit, width="stretch", hide_index=True)
        if stats.empty:
            st.info(
                "Belum ada event OOS yang lolos live-plan parity. Ini bukan bukti edge nol; "
                "model tetap fail-closed dan bobot AI tidak dinaikkan."
            )
        else:
            st.subheader("Chronological out-of-sample validation")
            st.caption("Statistik ini merupakan chronological holdout pada level setup, bukan probabilitas khusus ticker. Detector dan structural levels dibangun ulang pada setiap tanggal kandidat; entry harus tersentuh dalam 5 bar, gap yang merusak RR dibatalkan, biaya/slippage masuk, dan bar ambigu dihitung konservatif sebagai SL lebih dulu.")
            st.dataframe(stats, width="stretch", hide_index=True)
            st.download_button(
                "Download seluruh trade historis",
                trades.to_csv(index=False).encode("utf-8"),
                "walkforward_trades.csv",
                "text/csv",
            )

    with tab_audit:
        st.subheader("Audit coverage dan sumber data")
        data_contract_audit = result.get(
            "scanner_data_contract_audit",
            pd.DataFrame(),
        )
        if (
            isinstance(data_contract_audit, pd.DataFrame)
            and not data_contract_audit.empty
        ):
            with st.expander(
                "Data contract—mengapa tabel dapat kosong",
                expanded=True,
            ):
                st.dataframe(
                    streamlit_safe_frame(data_contract_audit),
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "NO_CANDIDATE_VALID hanya berarti nol kandidat setelah "
                    "upstream evidence cukup. NOT_EVALUABLE_DATA_INSUFFICIENT "
                    "berarti scanner belum berhak menyimpulkan tidak ada kandidat."
                )
        swing_coverage_audit = focus_screens.get(
            "profit_data_coverage_audit",
            pd.DataFrame(),
        )
        if (
            isinstance(swing_coverage_audit, pd.DataFrame)
            and not swing_coverage_audit.empty
        ):
            with st.expander(
                "Detail ticker Swing yang gagal coverage gate",
                expanded=False,
            ):
                st.dataframe(
                    streamlit_safe_frame(swing_coverage_audit),
                    hide_index=True,
                    width="stretch",
                )
        requested_count = len(getattr(report, "requested", []) or []) if report is not None else 0
        downloaded_count = len(getattr(report, "downloaded", []) or []) if report is not None else 0
        download_coverage = 100.0 * downloaded_count / requested_count if requested_count else 0.0
        fund_audit = result.get("fundamentals", pd.DataFrame())
        fundamental_history_audit = result.get("fundamental_history", pd.DataFrame())
        fundamental_history_report = result.get("fundamental_history_report", pd.DataFrame())
        status_audit = result.get("market_status", pd.DataFrame())
        news_audit = result.get("news_review", pd.DataFrame())
        quote_audit = result.get("execution_snapshots", pd.DataFrame())
        independent_audit = result.get("price_validation", pd.DataFrame())
        provider_audit = result.get("independent_provider_report", pd.DataFrame())
        source_quorum_audit = result.get("source_quorum_audit", pd.DataFrame())
        scan_performance_profile = result.get("scan_performance_profile", pd.DataFrame())
        database_sync_report = result.get("database_sync_report", pd.DataFrame())
        database_read_report = result.get("database_read_report", pd.DataFrame())
        fund_series = fund_audit.get("fundamental_coverage", pd.Series(index=fund_audit.index, dtype=float))
        fund_ok = int(pd.to_numeric(fund_series, errors="coerce").ge(45).sum()) if not fund_audit.empty else 0
        status_ok = int(status_audit.get("market_status_verified", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not status_audit.empty else 0
        news_ok = int(news_audit.get("provider_query_ok", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not news_audit.empty else 0
        quote_ok = int(quote_audit.get("quote_verified", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not quote_audit.empty else 0
        independent_ok = int(independent_audit.get("independent_price_verified", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not independent_audit.empty else 0
        provider_ok = int(provider_audit.get("status", pd.Series(dtype=str)).eq("OK").sum()) if not provider_audit.empty else 0
        a1, a2, a3, a4, a5, a6, a7 = st.columns(7)
        a1.metric("OHLCV coverage", f"{download_coverage:.1f}%")
        a2.metric("Fundamental ≥45%", fund_ok)
        a3.metric("Status IDX verified", status_ok)
        a4.metric("News query OK", news_ok)
        a5.metric("Quote verified", quote_ok)
        a6.metric("Harga independen", independent_ok)
        a7.metric("Auto provider OK", provider_ok)
        if isinstance(scan_performance_profile, pd.DataFrame) and not scan_performance_profile.empty:
            with st.expander("Profil waktu scan—lihat bottleneck aktual", expanded=True):
                slowest = scan_performance_profile.iloc[0]
                st.caption(
                    f"Tahap terlama: {slowest.get('stage')} "
                    f"({float(slowest.get('seconds', 0.0)):.1f} detik; "
                    f"{float(slowest.get('share_pct', 0.0)):.1f}% dari pipeline terukur)."
                )
                st.dataframe(
                    scan_performance_profile,
                    hide_index=True,
                    width="stretch",
                )
        grade_series = fund_audit.get("fundamental_data_grade", pd.Series(index=fund_audit.index, dtype=str)).astype(str)
        score10_series = pd.to_numeric(fund_audit.get("fundamental_score_10", pd.Series(index=fund_audit.index, dtype=float)), errors="coerce")
        sources_series = pd.to_numeric(fund_audit.get("fundamental_source_count", pd.Series(index=fund_audit.index, dtype=float)), errors="coerce")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Fundamental data grade A", int(grade_series.eq("A").sum()))
        f2.metric("Fundamental data grade B", int(grade_series.eq("B").sum()))
        f3.metric("Fundamental score ≥8/10", int(score10_series.ge(8.0).sum()))
        f4.metric("Multi-source statement history", int(sources_series.ge(2).sum()))
        source_tiers = getattr(report, "source_tiers", {}) or {} if report is not None else {}
        live_yahoo_ohlcv = sum(str(tier).startswith("LIVE_YAHOO") for tier in source_tiers.values())
        live_idx_ohlcv = sum(str(tier).startswith("LIVE_IDX") for tier in source_tiers.values())
        live_itick_ohlcv = sum(str(tier).startswith("LIVE_ITICK") for tier in source_tiers.values())
        fresh_cache_ohlcv = sum(str(tier).startswith("CACHE_FRESH_VERIFIED") for tier in source_tiers.values())
        stale_cache_ohlcv = sum(str(tier) == "CACHE_FALLBACK" for tier in source_tiers.values())
        unavailable_ohlcv = sum(str(tier) == "UNAVAILABLE" for tier in source_tiers.values())
        st.caption(
            f"Tier OHLCV: {live_yahoo_ohlcv} Yahoo · {live_idx_ohlcv} IDX patch · "
            f"{live_itick_ohlcv} iTick free · {fresh_cache_ohlcv} cache current · "
            f"{stale_cache_ohlcv} cache stale · {unavailable_ohlcv} unavailable."
        )
        o1, o2, o3, o4, o5 = st.columns(5)
        o1.metric("OHLCV zero-call cache", int(getattr(report, "cache_hits", 0) or 0))
        o2.metric("Incremental refresh", int(getattr(report, "incremental_refreshes", 0) or 0))
        o3.metric("Full refresh", int(getattr(report, "full_refreshes", 0) or 0))
        o4.metric("Provider calls", int(getattr(report, "provider_calls", 0) or 0))
        o5.metric("Downloaded bars", int(getattr(report, "downloaded_bars", 0) or 0))
        st.subheader("Source quorum per lapisan")
        st.caption(
            "Fallback tidak sama dengan verifikasi simultan. TWO_SOURCE hanya diberikan bila dua keluarga sumber benar-benar "
            "hadir; data regulator dan data akun memakai kebijakan authority-first. Kandidat tetap ditampilkan saat quorum parsial."
        )
        if not source_quorum_audit.empty:
            st.dataframe(source_quorum_audit, hide_index=True, width="stretch")
        st.subheader("Database-first readiness")
        st.caption(
            "Bridge v11 membaca cache fundamental, histori laporan, forward quality, coverage-aware lane Growth/Turnaround, kalender IDX, selector outcomes, execution-AI outcomes, arah IHSG, narrative safety lineage, serta evidence comparability dan production-rank contract dari Supabase. "
            "Refresh dipicu oleh miss/stale, perubahan semantic version, event material, atau cohort round-robin. "
            "Semua kegagalan tetap fail-soft."
        )
        db_col1, db_col2 = st.columns([1, 3])
        with db_col1:
            if st.button("Test koneksi database", key="database_health_check_v7163"):
                st.session_state["database_health_report_v7163"] = ScannerDatabaseBridge().health_check()
        with db_col2:
            health_report = st.session_state.get("database_health_report_v7163")
            if isinstance(health_report, dict) and health_report:
                st.dataframe(pd.DataFrame([health_report]), hide_index=True, width="stretch")
                if str(health_report.get("state")) == "MIGRATION_REQUIRED_V3":
                    st.error("Jalankan database/migration_v3_database_first.sql terlebih dahulu.")
                elif str(health_report.get("state")) == "MIGRATION_REQUIRED_V4":
                    st.error("Jalankan database/migration_v4_research_memory.sql di Supabase SQL Editor, lalu reboot aplikasi.")
                elif str(health_report.get("state")) == "MIGRATION_REQUIRED_V5":
                    st.error("Jalankan database/migration_v5_ihsg_direction.sql di Supabase SQL Editor, lalu reboot aplikasi.")
                elif str(health_report.get("state")) == "MIGRATION_REQUIRED_V6":
                    st.error("Jalankan database/migration_v6_selector_ai_outcomes.sql di Supabase SQL Editor, lalu reboot aplikasi.")
                elif str(health_report.get("state")) == "MIGRATION_REQUIRED_V7":
                    st.error("Jalankan database/migration_v7_multibagger_lanes.sql di Supabase SQL Editor, lalu reboot aplikasi.")
                elif str(health_report.get("state")) == "MIGRATION_REQUIRED_V8":
                    st.error("Jalankan database/migration_v8_data_contract.sql di Supabase SQL Editor, lalu reboot aplikasi.")
                elif str(health_report.get("state")) == "MIGRATION_REQUIRED_V9":
                    st.error(
                        "Jalankan database/migration_v9_narrative_intelligence.sql "
                        "di Supabase SQL Editor, lalu reboot aplikasi."
                    )
                elif str(health_report.get("state")) == "MIGRATION_REQUIRED_V10":
                    st.error(
                        "Jalankan database/migration_v10_narrative_safety.sql "
                        "di Supabase SQL Editor, lalu reboot aplikasi."
                    )
                elif str(health_report.get("state")) == "MIGRATION_REQUIRED_V11":
                    st.error(
                        "Jalankan database/migration_v11_400_universe_evidence.sql "
                        "di Supabase SQL Editor, lalu jalankan verify v11 dan reboot aplikasi."
                    )
        if not database_read_report.empty:
            read_state = database_read_report.get("database_read_state", pd.Series(dtype=str)).astype(str)
            current_hits = int(read_state.eq("DATABASE_CURRENT").sum())
            stale_hits = int(read_state.isin(["DATABASE_STALE_USABLE", "DATABASE_STALE_FALLBACK"]).sum())
            misses = int(read_state.isin(["DATABASE_MISS", "DATABASE_EXPIRED", "DATABASE_MODEL_STALE", "DATABASE_EVENT_DUE", "DATABASE_CHECK_DUE", "MIGRATION_REQUIRED_V3", "MIGRATION_REQUIRED_V4", "MIGRATION_REQUIRED_V5", "MIGRATION_REQUIRED_V6", "MIGRATION_REQUIRED_V7", "MIGRATION_REQUIRED_V8", "MIGRATION_REQUIRED_V9", "MIGRATION_REQUIRED_V10", "MIGRATION_REQUIRED_V11"]).sum())
            total_reads = int(len(database_read_report))
            hit_rate = 100.0 * current_hits / total_reads if total_reads else 0.0
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Database current hits", current_hits)
            r2.metric("Database stale fallback", stale_hits)
            r3.metric("Refresh/miss", misses)
            r4.metric("Current hit rate", f"{hit_rate:.1f}%")
            st.dataframe(database_read_report, hide_index=True, width="stretch")

        research_outcomes_view = result.get("research_outcomes", pd.DataFrame())
        research_summary_view = result.get("research_outcome_summary", pd.DataFrame())
        model_registry_view = result.get("model_registry", pd.DataFrame())
        if isinstance(research_summary_view, pd.DataFrame) and not research_summary_view.empty:
            st.subheader("IHSG, EOFF & Silent Accumulation outcome memory")
            st.caption(
                "Outcome hanya diselesaikan setelah forward bars tersedia dan selalu dipisahkan berdasarkan signal family, "
                "liquidity bucket, dan semantic model version. Data ini belum otomatis menaikkan bobot produksi."
            )
            st.dataframe(research_summary_view, hide_index=True, width="stretch")
            with st.expander("Detail research outcomes", expanded=False):
                st.dataframe(research_outcomes_view, hide_index=True, width="stretch")
                st.download_button(
                    "Download research outcome memory",
                    research_outcomes_view.to_csv(index=False).encode("utf-8"),
                    "research_outcomes_v7_3.csv", "text/csv", key="download_research_outcomes_v730",
                )
        if isinstance(model_registry_view, pd.DataFrame) and not model_registry_view.empty:
            with st.expander("Semantic model registry", expanded=False):
                st.dataframe(model_registry_view, hide_index=True, width="stretch")

        st.subheader("Database upload proof")
        st.caption(
            "Laporan write membuktikan respons HTTP saat upsert. Tombol readback di bawah melakukan GET ulang berdasarkan exact snapshot/outcome key, "
            "sehingga row yang benar-benar tersedia di Supabase dapat dibedakan dari write yang hanya tampak berhasil."
        )
        verify_col1, verify_col2 = st.columns([1, 3])
        with verify_col1:
            if st.button("Verifikasi upload scan terakhir", key="database_readback_verify_v7163"):
                try:
                    st.session_state["database_readback_report_v7163"] = (
                        ScannerDatabaseBridge().verify_persisted_scan(result)
                    )
                except Exception as exc:
                    st.session_state["database_readback_report_v7163"] = pd.DataFrame([{
                        "bridge_version": DATABASE_BRIDGE_VERSION,
                        "schema_version": DATABASE_SCHEMA_VERSION,
                        "database_mode": "UNKNOWN",
                        "state": "READBACK_FAIL_SOFT",
                        "table": "__SUMMARY__",
                        "scan_id": str(result.get("scan_id", "")),
                        "rows_attempted": 0,
                        "rows_written": 0,
                        "rows_verified": 0,
                        "verification_pct": 0.0,
                        "detail": f"{type(exc).__name__}: {str(exc)[:500]}",
                    }])
        with verify_col2:
            readback_report = st.session_state.get("database_readback_report_v7163", pd.DataFrame())
            if isinstance(readback_report, pd.DataFrame) and not readback_report.empty:
                summary_row = readback_report.loc[
                    readback_report.get("table", pd.Series(dtype=str)).astype(str).eq("__SUMMARY__")
                ]
                if not summary_row.empty:
                    summary_state = str(summary_row.iloc[0].get("state", ""))
                    summary_pct = float(pd.to_numeric(summary_row.iloc[0].get("verification_pct"), errors="coerce") or 0.0)
                    if summary_state == "VERIFIED_ALL_CRITICAL_TABLES":
                        st.success(f"Upload database terverifikasi {summary_pct:.1f}% untuk seluruh tabel kritis.")
                    else:
                        st.warning(f"Readback database belum lengkap: {summary_state} ({summary_pct:.1f}%).")
                st.dataframe(readback_report, hide_index=True, width="stretch")
                st.download_button(
                    "Download database readback audit",
                    readback_report.to_csv(index=False).encode("utf-8"),
                    "database_readback_audit_v7_16_3.csv",
                    "text/csv",
                    key="download_database_readback_v7163",
                )

        if not database_sync_report.empty:
            scan_id_value = str(result.get("scan_id", "") or database_sync_report.get("scan_id", pd.Series(dtype=str)).dropna().astype(str).head(1).squeeze())
            if scan_id_value:
                st.caption(f"Scan ID database: `{scan_id_value}`")
            database_states = set(database_sync_report.get("state", pd.Series(dtype=str)).astype(str))
            if database_states.intersection({"DATABASE_FAIL_SOFT", "PARTIAL_WRITE"}):
                st.warning(
                    "Sinkronisasi database belum penuh. Periksa kolom detail; v7.16.3 menampilkan body error PostgREST "
                    "dan mengisolasi record yang tidak valid tanpa menghentikan scan."
                )
            elif database_states and database_states.issubset({"OK", "NO_ROWS"}):
                st.success("Sinkronisasi database selesai tanpa kegagalan tabel.")
            st.dataframe(database_sync_report, hide_index=True, width="stretch")
        st.dataframe(
            pd.DataFrame(
                [
                    ["Verified local cache", "Menghindari unduh ulang full-history ketika EOD sudah current", "Automatic cache-first"],
                    ["Yahoo Finance via yfinance", "Primary OHLCV harian, IHSG, fundamental", "Automatic bounded batch"],
                    ["Yahoo statement history", "Quarterly/annual revenue, earnings, cash flow, balance sheet, dilution", "Automatic bounded Multibagger shortlist"],
                    ["IDX official XBRL/iXBRL", "First-party quarterly/annual statement history", "Automatic, no key; bounded + cached + fail-soft"],
                    ["Manual IDX/XBRL CSV", "Fallback bila filing otomatis tidak dapat diparse", "Optional; provenance/accounting checks"],
                    ["Twelve Data fundamentals", "Third-source income statement, balance sheet, and cash flow consensus", "Optional eligible API plan; bounded shortlist"],
                    ["IDX Stock Summary API", "Menambal bar EOD terakhir saat Yahoo gagal dan cache historis tersedia", "Automatic official fallback"],
                    ["iTick free tier", "Fallback OHLCV harian untuk ticker gagal", "Optional token; ≤4 call/min internal"],
                    ["Official IDX public pages", "Daftar saham, FCA/pemantauan, suspensi, aksi korporasi, disclosure", "Automated + cache"],
                    ["Google News RSS", "Berita luas per ticker", "Automated"],
                    ["IDX Stock Summary API", "Cross-check harga EOD resmi kandidat", "Automatic primary verification"],
                    ["Google Finance public quote", "Fallback harga independen kandidat", "Automatic bounded shortlist"],
                    ["Persistent evidence cache", "Mempertahankan bukti harga provider yang masih sama-session", "Automatic; tidak mengubah identitas sumber"],
                    ["iTick free tier", "Fallback harga independen untuk shortlist yang belum terpecahkan", "Optional token; bounded"],
                    ["Twelve Data XIDX EOD", "Fallback recent close dan return path", "Automatic bila deployment secret tersedia"],
                    ["OHLCV flow model", "Proxy akumulasi/distribusi tanpa broker-summary upload", "Automatic; bukan beneficial-owner data"],
                ],
                columns=["Provider family", "Dipakai untuk", "Mode"],
            ),
            hide_index=True,
            width="stretch",
        )
        if independent_ok:
            st.success(f"{independent_ok} ticker mempunyai cross-validation harga dari keluarga sumber independen.")
        else:
            st.warning(
                "Belum ada harga independen yang terverifikasi. Setup tetap terlihat sebagai SIGNAL_READY, "
                "tetapi tidak menjadi kandidat manual atau tiket Stockbit."
            )
        if not independent_audit.empty:
            st.dataframe(independent_audit, hide_index=True, width="stretch")
        if not provider_audit.empty:
            with st.expander("Audit provider harga otomatis", expanded=bool(provider_audit["status"].ne("OK").any())):
                st.dataframe(provider_audit, hide_index=True, width="stretch")
        if not fundamental_history_report.empty:
            with st.expander(
                "Audit provider fundamental historis",
                expanded=bool(fundamental_history_report.get("status", pd.Series(dtype=str)).ne("OK").any()),
            ):
                st.dataframe(fundamental_history_report, hide_index=True, width="stretch")
        if not fundamental_history_audit.empty:
            st.caption(
                f"Statement history: {len(fundamental_history_audit)} baris · "
                f"{fundamental_history_audit['ticker'].nunique()} emiten · "
                f"{fundamental_history_audit['source_family'].nunique()} keluarga sumber."
            )
        st.subheader("Execution gate diagnostics")
        st.dataframe(execution_funnel_summary(signals), hide_index=True, width="stretch")
        st.subheader("Satu baris untuk setiap ticker")
        st.dataframe(universe, width="stretch", hide_index=True)
        if report is not None and report.failed:
            with st.expander(f"Ticker gagal diunduh ({len(report.failed)})"):
                st.dataframe(pd.DataFrame(report.failed.items(), columns=["ticker", "error"]), hide_index=True)
        if report is not None and report.warnings:
            with st.expander(f"Peringatan kualitas OHLCV ({len(report.warnings)})"):
                st.dataframe(pd.DataFrame(report.warnings.items(), columns=["ticker", "warning"]), hide_index=True)

    with tab_method:
        st.markdown(
            """
            ### Hirarki sinyal v7.16.3 — Currency/Split-Aware Valuation, Direct-vs-Proxy Multibagger, IHSG & Core Swing

            1. **Arah IHSG:** historical analogue + breadth + chronological OOS menerbitkan UP/SIDEWAYS/DOWN atau `ABSTAIN` untuk horizon 1/5/20 hari bursa.
            2. **Risk overlay:** regime IHSG hanya membatasi risk budget; tidak pernah menaikkan Final Score atau mengubah saham buruk menjadi layak.
            3. **Cross-sectional selector:** setiap saham dibandingkan terhadap IHSG untuk excess return 5D/20D/60D sebelum setup dicari.
            4. **Core Swing:** teknikal, relative strength, dan Silent Accumulation menentukan radar; Pullback, Breakout Retest, Reversal, dan Unicorn/ICT kemudian menentukan entry, trigger, stop, TP, RR, dan expiry.
            5. **Multibagger:** kualitas bisnis, growth, profitability, OCF/FCF, solvabilitas, valuasi, proyek, manajemen, ownership, dan future fundamental menjadi fondasi utama.
            6. **Execution AI:** fill, TP1-before-SL, MFE, MAE, expectancy setelah biaya, Brier skill, dan drawdown diuji OOS sebelum AI boleh memengaruhi ranking.
            7. **Validitas data:** candle EOD harus final, OHLCV tidak boleh unavailable atau terlalu lama, dan suspensi/FCA tetap memblokir.
            8. **Kandidat tetap lengkap:** satu ticker boleh muncul pada beberapa setup Core Swing; Top-20 memakai satu slot per ticker agar ranking tidak didominasi duplikasi.
            9. **Dua kebijakan eksekusi:** `SIGNAL_FIRST` untuk radar dan `ACCOUNT_GUARDED` untuk sizing, cash, slot posisi, serta portfolio heat.
            10. **Verifikasi harga:** IDX EOD → Google Finance → iTick opsional → Twelve Data opsional, dengan cache bukti provider yang masih sama-session.
            11. **Submit manual:** semua order tetap direvalidasi dan dikirim manual di Stockbit.

            `SIGNAL_READY` bukan instruksi beli. `READY_FOR_STOCKBIT_VERIFY` adalah kandidat manual, sedangkan `EXECUTION_READY` selalu mensyaratkan `autopilot_verified = True`; seluruh submit tetap dilakukan manual di Stockbit.
            """
        )


with main_time_cycle:
    render_time_cycle_main_tab(
        cfg,
        lambda tickers, lookback: download_ohlcv(
            tickers, period=lookback, itick_api_token=itick_api_token,
        ),
    )
