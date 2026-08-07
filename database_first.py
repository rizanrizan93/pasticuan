from __future__ import annotations

import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

DATABASE_FIRST_VERSION = "9.6.0-quality-integrity"

def run_parallel_backfill_jobs(
    jobs: Mapping[str, Callable[[], object]],
    max_workers: int = 4,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Run independent evidence queues concurrently and report stage timing."""
    if not jobs:
        return {}, pd.DataFrame(columns=["stage", "elapsed_seconds", "state", "error"])
    results: dict[str, object] = {}
    rows: list[dict[str, object]] = []

    def timed(name: str, job: Callable[[], object]):
        started = time.perf_counter()
        try:
            value = job()
            return name, value, round(time.perf_counter() - started, 3), "OK", ""
        except Exception as exc:
            return name, exc, round(time.perf_counter() - started, 3), "FAILED", f"{type(exc).__name__}: {str(exc)[:240]}"

    workers = min(max(1, int(max_workers)), len(jobs))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(timed, name, job) for name, job in jobs.items()]
        for future in as_completed(futures):
            name, value, elapsed, state, error = future.result()
            results[name] = value
            rows.append({"stage": name, "elapsed_seconds": elapsed, "state": state, "error": error})
    return results, pd.DataFrame(rows).sort_values("stage").reset_index(drop=True)



@dataclass(frozen=True)
class DatabaseReadinessPolicy:
    """Minimum evidence coverage before the full-universe daily scan is allowed."""

    fundamental_snapshot_min_pct: float = 80.0
    fundamental_history_min_pct: float = 70.0
    market_status_min_pct: float = 80.0
    # News/narrative is a delta enrichment layer and no longer blocks the first scan.
    news_review_min_pct: float = 0.0


def _names(tickers: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(t).upper().strip() for t in tickers if str(t).strip()))


def _coerce_bool_series(
    values: pd.Series | Sequence[object],
    *,
    default: bool,
    index: pd.Index | None = None,
) -> pd.Series:
    """Return strict bools without pandas silent-downcast or string truthiness."""
    if isinstance(values, pd.Series):
        series = values.copy()
    else:
        series = pd.Series(values, index=index)

    true_tokens = {"1", "true", "t", "yes", "y", "on", "ok", "verified", "complete"}
    false_tokens = {"0", "false", "f", "no", "n", "off", "", "none", "null", "nan"}

    def convert(value: object) -> bool:
        if value is None or value is pd.NA:
            return bool(default)
        try:
            if bool(pd.isna(value)):
                return bool(default)
        except (TypeError, ValueError):
            pass
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return bool(value) if np.isfinite(value) else bool(default)
        token = str(value).strip().lower()
        if token in true_tokens:
            return True
        if token in false_tokens:
            return False
        return bool(default)

    return series.map(convert).astype(bool)


def _normalised_ticker_series(frame: pd.DataFrame | None) -> pd.Series:
    if frame is None or frame.empty or "ticker" not in frame.columns:
        return pd.Series(dtype=str)
    return frame["ticker"].dropna().astype(str).str.upper().str.strip()


def _ticker_set(frame: pd.DataFrame | None, *, eligible_column: str | None = None) -> set[str]:
    if frame is None or frame.empty or "ticker" not in frame.columns:
        return set()
    local = frame.copy()
    # Only use stored rows as ready when the producer did not expose an
    # eligibility contract.  If the column exists and every row is false, zero
    # rows are ready; silently falling back to all stored rows is a coverage bug.
    if eligible_column and eligible_column in local.columns:
        local = local.loc[_coerce_bool_series(local[eligible_column], default=False)]
    return set(_normalised_ticker_series(local))


def _history_stats(history: pd.DataFrame | None) -> tuple[set[str], set[str], dict[str, int]]:
    if history is None or history.empty or "ticker" not in history.columns:
        return set(), set(), {}
    local = history.copy()
    local["ticker"] = local["ticker"].astype(str).str.upper().str.strip()
    stored = set(local["ticker"].dropna())
    period_column = next((c for c in ("period_end", "statement_date", "date", "as_of") if c in local.columns), None)
    if period_column is None:
        counts = local.groupby("ticker").size().astype(int)
    else:
        local[period_column] = pd.to_datetime(local[period_column], errors="coerce")
        counts = local.dropna(subset=[period_column]).groupby("ticker")[period_column].nunique().astype(int)
    period_counts = {str(key): int(value) for key, value in counts.items()}
    ready = set(counts[counts >= 2].index.astype(str))
    return stored, ready, period_counts


def _audit_refresh_map(report: pd.DataFrame | None, scope: str) -> dict[str, bool]:
    if report is None or report.empty or "ticker" not in report.columns:
        return {}
    local = report.copy()
    if "scope" in local.columns:
        local = local.loc[local["scope"].astype(str).eq(scope)]
    if local.empty:
        return {}
    refresh = _coerce_bool_series(
        local.get("refresh_required", pd.Series(True, index=local.index)),
        default=True,
    )
    return dict(zip(local["ticker"].astype(str).str.upper().str.strip(), refresh))


def build_database_coverage(
    tickers: Sequence[str],
    *,
    fundamentals: pd.DataFrame | None = None,
    fundamental_history: pd.DataFrame | None = None,
    market_status: pd.DataFrame | None = None,
    news_review: pd.DataFrame | None = None,
    fundamental_report: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build stored, score-ready and refresh states for every requested ticker."""
    names = _names(tickers)

    snapshot_stored = _ticker_set(fundamentals)
    snapshot_ready = _ticker_set(
        fundamentals,
        eligible_column="fundamental_score_eligible" if isinstance(fundamentals, pd.DataFrame) and "fundamental_score_eligible" in fundamentals.columns else None,
    )
    history_stored, history_ready, history_periods = _history_stats(fundamental_history)

    market_stored = _ticker_set(market_status)
    market_ready = _ticker_set(
        market_status,
        eligible_column="market_status_score_eligible" if isinstance(market_status, pd.DataFrame) and "market_status_score_eligible" in market_status.columns else None,
    )
    news_stored = _ticker_set(news_review)
    news_ready = _ticker_set(
        news_review,
        eligible_column="news_score_eligible" if isinstance(news_review, pd.DataFrame) and "news_score_eligible" in news_review.columns else None,
    )

    snapshot_refresh = _audit_refresh_map(fundamental_report, "FUNDAMENTAL_SNAPSHOT")
    history_refresh = _audit_refresh_map(fundamental_report, "FUNDAMENTAL_HISTORY")

    rows: list[dict[str, object]] = []
    for ticker in names:
        snap_stored = ticker in snapshot_stored
        snap_ready = ticker in snapshot_ready
        hist_stored = ticker in history_stored
        hist_ready = ticker in history_ready
        market_has_row = ticker in market_stored
        market_is_ready = ticker in market_ready
        news_has_row = ticker in news_stored
        news_is_ready = ticker in news_ready

        stale_snapshot = bool(snapshot_refresh.get(ticker, not snap_ready))
        stale_history = bool(history_refresh.get(ticker, not hist_ready))
        ready_count = int(snap_ready) + int(hist_ready) + int(market_is_ready) + int(news_is_ready)
        stored_count = int(snap_stored) + int(hist_stored) + int(market_has_row) + int(news_has_row)
        missing_count = 4 - stored_count
        unusable_count = int(snap_stored and not snap_ready) + int(hist_stored and not hist_ready) + int(market_has_row and not market_is_ready) + int(news_has_row and not news_is_ready)
        stale_count = int(stale_snapshot) + int(stale_history)

        rows.append({
            "ticker": ticker,
            "fundamental_snapshot_stored": snap_stored,
            "fundamental_snapshot_ready": snap_ready,
            "fundamental_history_stored": hist_stored,
            "fundamental_history_periods": int(history_periods.get(ticker, 0)),
            "fundamental_history_ready": hist_ready,
            "market_status_stored": market_has_row,
            "market_status_ready": market_is_ready,
            "news_review_stored": news_has_row,
            "news_review_ready": news_is_ready,
            "fundamental_snapshot_refresh_required": stale_snapshot,
            "fundamental_history_refresh_required": stale_history,
            "evidence_stored_count": stored_count,
            "evidence_ready_count": ready_count,
            "evidence_missing_count": missing_count,
            "evidence_unusable_count": unusable_count,
            "evidence_stale_count": stale_count,
            "database_ticker_ready": bool(snap_ready and hist_ready),
        })
    return pd.DataFrame(rows)


def readiness_summary(
    coverage: pd.DataFrame,
    *,
    policy: DatabaseReadinessPolicy | None = None,
) -> pd.DataFrame:
    policy = policy or DatabaseReadinessPolicy()
    total = int(len(coverage))
    if total <= 0:
        return pd.DataFrame([{
            "requested_tickers": 0,
            "fundamental_snapshot_stored": 0,
            "fundamental_snapshot_stored_pct": 0.0,
            "fundamental_snapshot_ready": 0,
            "fundamental_snapshot_pct": 0.0,
            "fundamental_history_stored": 0,
            "fundamental_history_stored_pct": 0.0,
            "fundamental_history_ready": 0,
            "fundamental_history_pct": 0.0,
            "market_status_ready": 0,
            "market_status_pct": 0.0,
            "news_review_ready": 0,
            "news_review_pct": 0.0,
            "database_ready_tickers": 0,
            "database_ready_pct": 0.0,
            "database_state": "EMPTY_UNIVERSE",
        }])

    def count(column: str) -> int:
        return int(_coerce_bool_series(
            coverage.get(column, pd.Series(False, index=coverage.index)),
            default=False,
        ).sum())

    def pct(column: str) -> float:
        return 100.0 * count(column) / total

    snapshot_stored_count = count("fundamental_snapshot_stored")
    snapshot_ready_count = count("fundamental_snapshot_ready")
    history_stored_count = count("fundamental_history_stored")
    history_ready_count = count("fundamental_history_ready")
    market_stored_count = count("market_status_stored")
    market_ready_count = count("market_status_ready")
    news_stored_count = count("news_review_stored")
    news_ready_count = count("news_review_ready")
    ready_count = count("database_ticker_ready")

    snapshot_pct = 100.0 * snapshot_ready_count / total
    history_pct = 100.0 * history_ready_count / total
    market_pct = 100.0 * market_ready_count / total
    news_pct = 100.0 * news_ready_count / total
    ready_pct = 100.0 * ready_count / total

    state = "READY_FOR_DAILY_SCAN" if (
        snapshot_pct >= policy.fundamental_snapshot_min_pct
        and history_pct >= policy.fundamental_history_min_pct
        and market_pct >= policy.market_status_min_pct
        and news_pct >= policy.news_review_min_pct
    ) else "BACKFILL_REQUIRED"

    return pd.DataFrame([{
        "requested_tickers": total,
        "fundamental_snapshot_stored": snapshot_stored_count,
        "fundamental_snapshot_stored_pct": round(100.0 * snapshot_stored_count / total, 2),
        "fundamental_snapshot_ready": snapshot_ready_count,
        "fundamental_snapshot_pct": round(snapshot_pct, 2),
        "fundamental_snapshot_unusable": max(0, snapshot_stored_count - snapshot_ready_count),
        "fundamental_history_stored": history_stored_count,
        "fundamental_history_stored_pct": round(100.0 * history_stored_count / total, 2),
        "fundamental_history_ready": history_ready_count,
        "fundamental_history_pct": round(history_pct, 2),
        "fundamental_history_under_2_periods": max(0, history_stored_count - history_ready_count),
        "market_status_stored": market_stored_count,
        "market_status_ready": market_ready_count,
        "market_status_pct": round(market_pct, 2),
        "news_review_stored": news_stored_count,
        "news_review_ready": news_ready_count,
        "news_review_pct": round(news_pct, 2),
        "database_ready_tickers": ready_count,
        "database_ready_pct": round(ready_pct, 2),
        "database_state": state,
    }])


def _stable_bucket(ticker: str, cohorts: int) -> int:
    digest = sha256(ticker.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(1, int(cohorts))


def _prepare_queue_base(
    tickers: Sequence[str],
    coverage: pd.DataFrame,
    portfolio_tickers: Sequence[str],
    cohorts: int,
    active_cohort: int | None,
) -> pd.DataFrame:
    names = _names(tickers)
    portfolio = set(_names(portfolio_tickers))
    local = coverage.copy() if coverage is not None and not coverage.empty else build_database_coverage(names)
    local["ticker"] = local["ticker"].astype(str).str.upper().str.strip()
    local = local.set_index("ticker").reindex(names).reset_index()
    bool_columns = (
        "fundamental_snapshot_stored", "fundamental_snapshot_ready",
        "fundamental_history_stored", "fundamental_history_ready",
        "market_status_stored", "market_status_ready",
        "news_review_stored", "news_review_ready",
        "fundamental_snapshot_refresh_required", "fundamental_history_refresh_required",
    )
    for column in bool_columns:
        if column not in local.columns:
            local[column] = False
        local[column] = _coerce_bool_series(local[column], default=False)
    for column in ("evidence_missing_count", "evidence_unusable_count", "evidence_stale_count"):
        if column not in local.columns:
            local[column] = 0
        local[column] = pd.to_numeric(local[column], errors="coerce").fillna(0).astype(int)
    if active_cohort is None:
        active_cohort = int(pd.Timestamp.utcnow().dayofyear) % max(1, int(cohorts))
    local["portfolio_priority"] = local["ticker"].isin(portfolio)
    local["ticker_cohort"] = local["ticker"].map(lambda t: _stable_bucket(t, cohorts))
    local["active_cohort"] = int(active_cohort)
    local["cohort_due"] = local["ticker_cohort"].eq(int(active_cohort))
    return local


def _select_one_queue(
    local: pd.DataFrame,
    *,
    queue_name: str,
    batch_size: int,
    missing_column: str,
    ready_column: str,
    stale_column: str | None = None,
) -> tuple[list[str], pd.DataFrame]:
    queue = local.copy()
    queue["queue"] = queue_name
    queue["missing"] = ~queue[missing_column]
    queue["unusable"] = queue[missing_column] & ~queue[ready_column]
    queue["stale"] = queue[stale_column] if stale_column and stale_column in queue.columns else False
    queue["refresh_reason"] = np.select(
        [queue["missing"], queue["unusable"], queue["stale"], queue["cohort_due"]],
        [f"MISSING_{queue_name}", f"UNUSABLE_{queue_name}", f"STALE_{queue_name}", f"ROUND_ROBIN_{queue_name}"],
        default="CURRENT",
    )
    priority_map = {
        f"MISSING_{queue_name}": 0,
        f"UNUSABLE_{queue_name}": 1,
        f"STALE_{queue_name}": 2,
        f"ROUND_ROBIN_{queue_name}": 3,
        "CURRENT": 9,
    }
    queue["refresh_priority"] = queue["refresh_reason"].map(priority_map).fillna(9).astype(int)
    queue.loc[queue["portfolio_priority"] & queue["refresh_priority"].lt(9), "refresh_priority"] -= 1
    candidates = queue.loc[queue["refresh_reason"].ne("CURRENT")].sort_values(
        ["refresh_priority", "portfolio_priority", "evidence_missing_count", "evidence_unusable_count", "evidence_stale_count", "ticker"],
        ascending=[True, False, False, False, False, True],
        kind="stable",
    )
    selected = candidates.head(max(0, int(batch_size)))["ticker"].tolist()
    queue["selected_for_refresh"] = queue["ticker"].isin(selected)
    rank_map = {ticker: rank + 1 for rank, ticker in enumerate(selected)}
    queue["queue_rank"] = queue["ticker"].map(rank_map)
    columns = [
        "queue", "ticker", "selected_for_refresh", "queue_rank", "refresh_reason",
        "refresh_priority", "portfolio_priority", "ticker_cohort", "active_cohort",
        missing_column, ready_column, "evidence_missing_count", "evidence_unusable_count",
        "evidence_stale_count",
    ]
    return selected, queue[columns].sort_values(
        ["selected_for_refresh", "queue_rank", "refresh_priority", "ticker"],
        ascending=[False, True, True, True], kind="stable",
    ).reset_index(drop=True)


def select_evidence_refresh_queues(
    tickers: Sequence[str],
    coverage: pd.DataFrame,
    *,
    snapshot_batch_size: int,
    history_batch_size: int,
    market_batch_size: int,
    news_batch_size: int,
    portfolio_tickers: Sequence[str] = (),
    cohorts: int = 7,
    active_cohort: int | None = None,
) -> tuple[dict[str, list[str]], pd.DataFrame]:
    """Select independent queues so one failed provider cannot starve the others."""
    local = _prepare_queue_base(tickers, coverage, portfolio_tickers, cohorts, active_cohort)
    specs = (
        ("FUNDAMENTAL_SNAPSHOT", snapshot_batch_size, "fundamental_snapshot_stored", "fundamental_snapshot_ready", "fundamental_snapshot_refresh_required"),
        ("FUNDAMENTAL_HISTORY", history_batch_size, "fundamental_history_stored", "fundamental_history_ready", "fundamental_history_refresh_required"),
        ("MARKET_STATUS", market_batch_size, "market_status_stored", "market_status_ready", None),
        ("NEWS_REVIEW", news_batch_size, "news_review_stored", "news_review_ready", None),
    )
    queues: dict[str, list[str]] = {}
    audits: list[pd.DataFrame] = []
    for queue_name, batch_size, stored_column, ready_column, stale_column in specs:
        selected, audit = _select_one_queue(
            local,
            queue_name=queue_name,
            batch_size=batch_size,
            missing_column=stored_column,
            ready_column=ready_column,
            stale_column=stale_column,
        )
        queues[queue_name] = selected
        audits.append(audit)
    return queues, pd.concat(audits, ignore_index=True, sort=False) if audits else pd.DataFrame()


def select_database_refresh_queue(
    tickers: Sequence[str],
    coverage: pd.DataFrame,
    *,
    batch_size: int,
    portfolio_tickers: Sequence[str] = (),
    cohorts: int = 7,
    active_cohort: int | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Backward-compatible union queue used by older callers/tests."""
    queues, audit = select_evidence_refresh_queues(
        tickers,
        coverage,
        snapshot_batch_size=batch_size,
        history_batch_size=batch_size,
        market_batch_size=batch_size,
        news_batch_size=batch_size,
        portfolio_tickers=portfolio_tickers,
        cohorts=cohorts,
        active_cohort=active_cohort,
    )
    priority_order = ("FUNDAMENTAL_SNAPSHOT", "FUNDAMENTAL_HISTORY", "MARKET_STATUS", "NEWS_REVIEW")
    union: list[str] = []
    for queue_name in priority_order:
        union.extend(queues.get(queue_name, []))
    selected = list(dict.fromkeys(union))[:max(0, int(batch_size))]
    compat = audit.loc[audit["queue"].eq("FUNDAMENTAL_SNAPSHOT")].copy()
    compat["selected_for_refresh"] = compat["ticker"].isin(selected)
    rank_map = {ticker: rank + 1 for rank, ticker in enumerate(selected)}
    compat["queue_rank"] = compat["ticker"].map(rank_map)
    return selected, compat.sort_values(
        ["selected_for_refresh", "queue_rank", "refresh_priority", "ticker"],
        ascending=[False, True, True, True], kind="stable",
    ).reset_index(drop=True)


def estimate_remaining_passes(
    coverage: pd.DataFrame,
    batch_size: int,
    *,
    net_ready_gain: int | None = None,
) -> float:
    """Estimate passes only when the latest batch demonstrated positive progress."""
    if coverage is None or coverage.empty:
        return 0.0
    pending = int((~_coerce_bool_series(
        coverage.get("database_ticker_ready", pd.Series(False, index=coverage.index)),
        default=False,
    )).sum())
    if pending <= 0:
        return 0.0
    if net_ready_gain is not None:
        if int(net_ready_gain) <= 0:
            return float("nan")
        return float(np.ceil(pending / max(1, int(net_ready_gain))))
    return float(np.ceil(pending / max(1, int(batch_size))))
