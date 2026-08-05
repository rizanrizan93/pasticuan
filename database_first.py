from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

DATABASE_FIRST_VERSION = "9.1.1"


@dataclass(frozen=True)
class DatabaseReadinessPolicy:
    fundamental_snapshot_min_pct: float = 90.0
    fundamental_history_min_pct: float = 80.0
    market_status_min_pct: float = 0.0
    news_review_min_pct: float = 0.0


def _names(tickers: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(t).upper().strip() for t in tickers if str(t).strip()))


def _coerce_bool_series(
    values: pd.Series | Sequence[object],
    *,
    default: bool,
    index: pd.Index | None = None,
) -> pd.Series:
    """Return a strict boolean Series without pandas silent-downcast warnings.

    Unlike ``astype(bool)``, string values such as ``"False"`` are not treated
    as truthy. Unknown tokens fall back to ``default``.
    """
    if isinstance(values, pd.Series):
        series = values.copy()
    else:
        series = pd.Series(values, index=index)

    true_tokens = {"1", "true", "t", "yes", "y", "on"}
    false_tokens = {"0", "false", "f", "no", "n", "off", ""}

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


def _ticker_set(frame: pd.DataFrame | None, *, eligible_column: str | None = None) -> set[str]:
    if frame is None or frame.empty or "ticker" not in frame.columns:
        return set()
    local = frame.copy()
    if eligible_column and eligible_column in local.columns:
        local = local.loc[_coerce_bool_series(local[eligible_column], default=False)]
    return set(local["ticker"].dropna().astype(str).str.upper().str.strip())


def _history_ready_set(history: pd.DataFrame | None) -> set[str]:
    if history is None or history.empty or "ticker" not in history.columns:
        return set()
    local = history.copy()
    local["ticker"] = local["ticker"].astype(str).str.upper().str.strip()
    period_column = next((c for c in ("period_end", "statement_date", "date", "as_of") if c in local.columns), None)
    if period_column is None:
        counts = local.groupby("ticker").size()
    else:
        local[period_column] = pd.to_datetime(local[period_column], errors="coerce")
        counts = local.dropna(subset=[period_column]).groupby("ticker")[period_column].nunique()
    return set(counts[counts >= 2].index.astype(str))


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
    names = _names(tickers)
    snapshot_ready = _ticker_set(fundamentals, eligible_column="fundamental_score_eligible")
    if not snapshot_ready:
        snapshot_ready = _ticker_set(fundamentals)
    history_ready = _history_ready_set(fundamental_history)
    market_ready = _ticker_set(market_status, eligible_column="market_status_score_eligible")
    news_ready = _ticker_set(news_review, eligible_column="news_score_eligible")
    snapshot_refresh = _audit_refresh_map(fundamental_report, "FUNDAMENTAL_SNAPSHOT")
    history_refresh = _audit_refresh_map(fundamental_report, "FUNDAMENTAL_HISTORY")

    rows: list[dict[str, object]] = []
    for ticker in names:
        snap = ticker in snapshot_ready
        hist = ticker in history_ready
        market = ticker in market_ready
        news = ticker in news_ready
        stale_snapshot = bool(snapshot_refresh.get(ticker, not snap))
        stale_history = bool(history_refresh.get(ticker, not hist))
        missing_count = int(not snap) + int(not hist) + int(not market) + int(not news)
        stale_count = int(stale_snapshot) + int(stale_history)
        evidence_count = int(snap) + int(hist) + int(market) + int(news)
        rows.append({
            "ticker": ticker,
            "fundamental_snapshot_ready": snap,
            "fundamental_history_ready": hist,
            "market_status_ready": market,
            "news_review_ready": news,
            "fundamental_snapshot_refresh_required": stale_snapshot,
            "fundamental_history_refresh_required": stale_history,
            "evidence_ready_count": evidence_count,
            "evidence_missing_count": missing_count,
            "evidence_stale_count": stale_count,
            "database_ticker_ready": bool(snap and hist),
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
            "fundamental_snapshot_pct": 0.0,
            "fundamental_history_pct": 0.0,
            "market_status_pct": 0.0,
            "news_review_pct": 0.0,
            "database_ready_tickers": 0,
            "database_ready_pct": 0.0,
            "database_state": "EMPTY_UNIVERSE",
        }])

    def pct(column: str) -> float:
        return 100.0 * float(_coerce_bool_series(
            coverage.get(column, pd.Series(False, index=coverage.index)),
            default=False,
        ).sum()) / total

    snapshot_pct = pct("fundamental_snapshot_ready")
    history_pct = pct("fundamental_history_ready")
    market_pct = pct("market_status_ready")
    news_pct = pct("news_review_ready")
    ready_count = int(_coerce_bool_series(
        coverage.get("database_ticker_ready", pd.Series(False, index=coverage.index)),
        default=False,
    ).sum())
    ready_pct = 100.0 * ready_count / total
    state = "READY_FOR_DAILY_SCAN" if (
        snapshot_pct >= policy.fundamental_snapshot_min_pct
        and history_pct >= policy.fundamental_history_min_pct
        and market_pct >= policy.market_status_min_pct
        and news_pct >= policy.news_review_min_pct
    ) else "BACKFILL_REQUIRED"
    return pd.DataFrame([{
        "requested_tickers": total,
        "fundamental_snapshot_ready": int(coverage["fundamental_snapshot_ready"].sum()),
        "fundamental_snapshot_pct": round(snapshot_pct, 2),
        "fundamental_history_ready": int(coverage["fundamental_history_ready"].sum()),
        "fundamental_history_pct": round(history_pct, 2),
        "market_status_ready": int(coverage["market_status_ready"].sum()),
        "market_status_pct": round(market_pct, 2),
        "news_review_ready": int(coverage["news_review_ready"].sum()),
        "news_review_pct": round(news_pct, 2),
        "database_ready_tickers": ready_count,
        "database_ready_pct": round(ready_pct, 2),
        "database_state": state,
    }])


def _stable_bucket(ticker: str, cohorts: int) -> int:
    digest = sha256(ticker.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(1, int(cohorts))


def select_database_refresh_queue(
    tickers: Sequence[str],
    coverage: pd.DataFrame,
    *,
    batch_size: int,
    portfolio_tickers: Sequence[str] = (),
    cohorts: int = 7,
    active_cohort: int | None = None,
) -> tuple[list[str], pd.DataFrame]:
    names = _names(tickers)
    portfolio = set(_names(portfolio_tickers))
    if coverage is None or coverage.empty:
        coverage = build_database_coverage(names)
    local = coverage.copy()
    local["ticker"] = local["ticker"].astype(str).str.upper().str.strip()
    local = local.set_index("ticker").reindex(names).reset_index()
    for column in (
        "fundamental_snapshot_ready", "fundamental_history_ready",
        "market_status_ready", "news_review_ready",
        "fundamental_snapshot_refresh_required", "fundamental_history_refresh_required",
    ):
        if column not in local.columns:
            local[column] = False
        local[column] = _coerce_bool_series(local[column], default=False)
    if active_cohort is None:
        active_cohort = int(pd.Timestamp.utcnow().dayofyear) % max(1, int(cohorts))
    local["portfolio_priority"] = local["ticker"].isin(portfolio)
    local["missing_snapshot"] = ~local["fundamental_snapshot_ready"]
    local["missing_history"] = ~local["fundamental_history_ready"]
    local["missing_market"] = ~local["market_status_ready"]
    local["missing_news"] = ~local["news_review_ready"]
    local["ticker_cohort"] = local["ticker"].map(lambda t: _stable_bucket(t, cohorts))
    local["active_cohort"] = int(active_cohort)
    local["cohort_due"] = local["ticker_cohort"].eq(int(active_cohort))

    def reason(row: Mapping[str, object]) -> str:
        if bool(row.get("missing_snapshot")):
            return "MISSING_FUNDAMENTAL_SNAPSHOT"
        if bool(row.get("missing_history")):
            return "MISSING_FUNDAMENTAL_HISTORY"
        if bool(row.get("missing_market")):
            return "MISSING_MARKET_STATUS"
        if bool(row.get("missing_news")):
            return "MISSING_NEWS_REVIEW"
        if bool(row.get("fundamental_snapshot_refresh_required")):
            return "STALE_FUNDAMENTAL_SNAPSHOT"
        if bool(row.get("fundamental_history_refresh_required")):
            return "STALE_FUNDAMENTAL_HISTORY"
        if bool(row.get("cohort_due")):
            return "ROUND_ROBIN_DUE"
        return "CURRENT"

    local["refresh_reason"] = local.apply(lambda row: reason(row.to_dict()), axis=1)
    priority_map = {
        "MISSING_FUNDAMENTAL_SNAPSHOT": 0,
        "MISSING_FUNDAMENTAL_HISTORY": 1,
        "MISSING_MARKET_STATUS": 2,
        "MISSING_NEWS_REVIEW": 3,
        "STALE_FUNDAMENTAL_SNAPSHOT": 4,
        "STALE_FUNDAMENTAL_HISTORY": 5,
        "ROUND_ROBIN_DUE": 6,
        "CURRENT": 9,
    }
    local["refresh_priority"] = local["refresh_reason"].map(priority_map).fillna(9).astype(int)
    local.loc[local["portfolio_priority"] & local["refresh_priority"].lt(9), "refresh_priority"] -= 1
    candidates = local.loc[local["refresh_reason"].ne("CURRENT")].copy()
    candidates = candidates.sort_values(
        ["refresh_priority", "portfolio_priority", "evidence_missing_count", "evidence_stale_count", "ticker"],
        ascending=[True, False, False, False, True],
        kind="stable",
    )
    selected = candidates.head(max(0, int(batch_size)))["ticker"].tolist()
    local["selected_for_refresh"] = local["ticker"].isin(selected)
    local["queue_rank"] = np.nan
    if selected:
        rank_map = {ticker: idx + 1 for idx, ticker in enumerate(selected)}
        local.loc[local["selected_for_refresh"], "queue_rank"] = local.loc[local["selected_for_refresh"], "ticker"].map(rank_map)
    audit_columns = [
        "ticker", "selected_for_refresh", "queue_rank", "refresh_reason", "refresh_priority",
        "portfolio_priority", "ticker_cohort", "active_cohort", "fundamental_snapshot_ready",
        "fundamental_history_ready", "market_status_ready", "news_review_ready",
        "evidence_missing_count", "evidence_stale_count",
    ]
    return selected, local[audit_columns].sort_values(
        ["selected_for_refresh", "queue_rank", "refresh_priority", "ticker"],
        ascending=[False, True, True, True], kind="stable",
    ).reset_index(drop=True)


def estimate_remaining_passes(coverage: pd.DataFrame, batch_size: int) -> int:
    if coverage is None or coverage.empty:
        return 0
    pending = int((~_coerce_bool_series(
        coverage.get("database_ticker_ready", pd.Series(False, index=coverage.index)),
        default=False,
    )).sum())
    return int(np.ceil(pending / max(1, int(batch_size))))
