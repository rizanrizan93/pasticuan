from __future__ import annotations

"""Research maintenance, calibration and outcome memory for v8 Slim.

The module is deliberately dependency-light and deterministic. It provides:
- semantic model-version lineage;
- event-aware refresh decisions;
- round-robin fundamental-history backfill;
- liquidity-bucket calibration for Silent Accumulation;
- IDX trading-day validation with an official-calendar hook;
- durable Silent Accumulation and selector outcome memory.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from typing import Any, Iterable, Mapping, Sequence
import json
import math
import os
import re

import numpy as np
import pandas as pd

from release_contract import SCANNER_RELEASE_VERSION

RESEARCH_MAINTENANCE_VERSION = "9.6.0-quality-integrity"
RESEARCH_OUTCOME_SCHEMA_VERSION = "research_outcomes_v1.2"
SCANNER_VERSION = SCANNER_RELEASE_VERSION

MODEL_RELEASED_AT = "2026-08-15T00:00:00+00:00"

MODEL_VERSIONS: dict[str, str] = {
    "scanner": "9.6.0",
    "ranking": "9.6.0",
    "production_scoring": "9.6.0",
    "fundamental": "7.6.0",
    "fundamental_parser": "1.3.0",
    "eoff": "0.0.0-removed-from-production",
    "time_cycle": "0.0.0-removed-from-production",
    "silent_accumulation": "5.0.0",
    "ihsg_direction": "1.0.0",
    "stock_selector": "8.0.0",
    "two_stage_pipeline": "9.6.0",
    "execution_ai": "0.0.0-removed-from-production",
    "narrative_engine": "2.0.0-pure-event-separation",
    "database_schema": "11.0.0",
    "research_outcome": "1.2.0",
}


def _finite(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return str(value).strip()


def parse_semver(value: Any) -> tuple[int, int, int]:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", _text(value))
    if not match:
        return (0, 0, 0)
    return tuple(int(part) for part in match.groups())


def semantic_refresh_reason(stored_version: Any, current_version: Any, *, refresh_on_minor: bool = True) -> str:
    stored = parse_semver(stored_version)
    current = parse_semver(current_version)
    if stored == (0, 0, 0):
        return "MODEL_VERSION_MISSING"
    if stored[0] != current[0]:
        return "MODEL_MAJOR_CHANGED"
    if refresh_on_minor and stored[1] != current[1]:
        return "MODEL_MINOR_CHANGED"
    return ""


def model_registry_frame(as_of: Any | None = None) -> pd.DataFrame:
    stamp = pd.to_datetime(as_of, errors="coerce", utc=True)
    if pd.isna(stamp):
        stamp = pd.Timestamp.now(tz="UTC")
    rows = []
    for component, version in MODEL_VERSIONS.items():
        rows.append({
            "component": component,
            "semantic_version": version,
            "is_active": True,
            "released_at": MODEL_RELEASED_AT,
            "config_hash": sha256(f"{component}|{version}".encode("utf-8")).hexdigest(),
            "metadata": {
                "maintenance_version": RESEARCH_MAINTENANCE_VERSION,
                "registry_generated_at": stamp.isoformat(),
            },
        })
    return pd.DataFrame(rows)


LIQUIDITY_BUCKETS = (
    ("VERY_ILLIQUID", 0.0, 2_000_000_000.0),
    ("ILLIQUID", 2_000_000_000.0, 10_000_000_000.0),
    ("MEDIUM", 10_000_000_000.0, 50_000_000_000.0),
    ("LIQUID", 50_000_000_000.0, 250_000_000_000.0),
    ("VERY_LIQUID", 250_000_000_000.0, math.inf),
)


def liquidity_bucket(adtv20_idr: Any) -> str:
    value = max(0.0, _finite(adtv20_idr, 0.0))
    for label, lower, upper in LIQUIDITY_BUCKETS:
        if lower <= value < upper:
            return label
    return "VERY_ILLIQUID"


@dataclass(frozen=True)
class SilentAccumulationCalibration:
    bucket: str
    raw_score: float
    calibrated_score: float
    adjustment: float
    confidence_multiplier: float
    minimum_confirmation_score: float
    policy: str


def calibrate_silent_accumulation(
    raw_score: Any,
    adtv20_idr: Any,
    *,
    data_coverage: Any = 100.0,
    distribution_days: Any = 0,
    failed_absorption_days: Any = 0,
) -> SilentAccumulationCalibration:
    """Calibrate accumulation evidence across liquidity regimes.

    Illiquid names require stronger evidence and are shrunk toward neutral,
    reducing microstructure false positives. Highly liquid names receive only
    a small adjustment because price-volume features are more stable.
    """
    raw = max(0.0, min(100.0, _finite(raw_score, 0.0)))
    coverage = max(0.0, min(100.0, _finite(data_coverage, 0.0)))
    distribution = max(0.0, _finite(distribution_days, 0.0))
    failed_absorption = max(0.0, _finite(failed_absorption_days, 0.0))
    bucket = liquidity_bucket(adtv20_idr)
    params = {
        "VERY_ILLIQUID": (0.58, -7.0, 0.72, 78.0),
        "ILLIQUID": (0.70, -4.0, 0.80, 74.0),
        "MEDIUM": (0.82, -1.5, 0.90, 70.0),
        "LIQUID": (0.92, 0.0, 0.96, 68.0),
        "VERY_LIQUID": (0.96, 0.5, 1.00, 67.0),
    }
    shrink, base_adjustment, confidence_multiplier, minimum = params[bucket]
    calibrated = 50.0 + shrink * (raw - 50.0) + base_adjustment
    calibrated -= min(12.0, 1.4 * distribution + 2.2 * failed_absorption)
    if coverage < 70.0:
        calibrated -= min(8.0, (70.0 - coverage) * 0.12)
        confidence_multiplier *= max(0.60, coverage / 70.0)
    calibrated = max(0.0, min(100.0, calibrated))
    return SilentAccumulationCalibration(
        bucket=bucket,
        raw_score=round(raw, 2),
        calibrated_score=round(calibrated, 2),
        adjustment=round(calibrated - raw, 2),
        confidence_multiplier=round(max(0.0, min(1.0, confidence_multiplier)), 4),
        minimum_confirmation_score=minimum,
        policy="LIQUIDITY_BUCKET_SHRINKAGE_AND_MICROSTRUCTURE_PENALTY",
    )


def stable_cohort(ticker: Any, cohort_count: int = 7) -> int:
    count = max(1, int(cohort_count))
    digest = sha256(_text(ticker).upper().encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % count


def active_cohort(as_of: Any | None = None, cohort_count: int = 7) -> int:
    count = max(1, int(cohort_count))
    stamp = pd.to_datetime(as_of, errors="coerce")
    if pd.isna(stamp):
        stamp = pd.Timestamp.now(tz="Asia/Jakarta").tz_localize(None)
    return int(pd.Timestamp(stamp).date().toordinal() % count)


def select_round_robin_backfill(
    tickers: Sequence[str],
    audit: pd.DataFrame | None,
    *,
    priority_tickers: Iterable[str] = (),
    event_tickers: Iterable[str] = (),
    as_of: Any | None = None,
    cohort_count: int = 7,
    max_count: int = 40,
) -> tuple[list[str], pd.DataFrame]:
    """Select a bounded refresh queue without starving urgent symbols.

    Priority and event-triggered symbols always lead. Missing/expired rows are
    filled by deterministic round-robin cohort, then stale rows from the same
    cohort. This prevents every scan from hammering all free providers.
    """
    names = list(dict.fromkeys(_text(ticker) for ticker in tickers if _text(ticker)))
    priorities = {_text(ticker) for ticker in priority_tickers if _text(ticker)}
    events = {_text(ticker) for ticker in event_tickers if _text(ticker)}
    state_map: dict[str, str] = {}
    if isinstance(audit, pd.DataFrame) and not audit.empty and "ticker" in audit:
        for _, row in audit.iterrows():
            state_map[_text(row.get("ticker"))] = _text(row.get("database_read_state") or row.get("status")).upper()
    cohort = active_cohort(as_of, cohort_count)
    rows: list[dict[str, Any]] = []
    candidates: list[tuple[int, int, str]] = []
    for order, ticker in enumerate(names):
        state = state_map.get(ticker, "DATABASE_MISS")
        ticker_cohort = stable_cohort(ticker, cohort_count)
        reason = "NOT_DUE"
        priority = 99
        due = False
        if ticker in events:
            due, priority, reason = True, 0, "EVENT_TRIGGERED"
        elif ticker in priorities:
            due, priority, reason = True, 1, "PRIORITY_TICKER"
        elif state in {"DATABASE_EXPIRED", "DATABASE_MODEL_STALE", "DATABASE_EVENT_DUE", "DATABASE_PAYLOAD_INVALID"}:
            due, priority, reason = True, 2, state
        elif state in {"DATABASE_MISS", "MISSING_TIMESTAMP", "MIGRATION_REQUIRED_V4"} and ticker_cohort == cohort:
            due, priority, reason = True, 3, "ROUND_ROBIN_MISSING"
        elif state in {"DATABASE_STALE_USABLE", "DATABASE_STALE_FALLBACK"} and ticker_cohort == cohort:
            due, priority, reason = True, 4, "ROUND_ROBIN_STALE"
        elif state == "DATABASE_CURRENT":
            reason = "DATABASE_CURRENT"
        candidates.append((priority, order, ticker)) if due else None
        rows.append({
            "ticker": ticker,
            "database_read_state": state,
            "ticker_cohort": ticker_cohort,
            "active_cohort": cohort,
            "refresh_due": due,
            "refresh_reason": reason,
        })
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    limit = max(1, int(max_count))
    selected = [ticker for _, _, ticker in candidates[:limit]]
    selected_set = set(selected)
    audit_rows = pd.DataFrame(rows)
    if not audit_rows.empty:
        audit_rows["selected_for_refresh"] = audit_rows["ticker"].isin(selected_set)
        audit_rows["scheduler_version"] = RESEARCH_MAINTENANCE_VERSION
    return selected, audit_rows


def _holiday_set(values: Iterable[Any] = ()) -> set[date]:
    result: set[date] = set()
    for value in values:
        stamp = pd.to_datetime(value, errors="coerce")
        if pd.notna(stamp):
            result.add(pd.Timestamp(stamp).date())
    env_values = os.getenv("IDX_TRADING_HOLIDAYS", "")
    for token in env_values.split(","):
        stamp = pd.to_datetime(token.strip(), errors="coerce")
        if pd.notna(stamp):
            result.add(pd.Timestamp(stamp).date())
    return result


def adjust_to_idx_trading_day(
    value: Any,
    *,
    holidays: Iterable[Any] = (),
    official_open_dates: Iterable[Any] = (),
    official_closed_dates: Iterable[Any] = (),
    direction: str = "next",
    max_steps: int = 15,
) -> dict[str, Any]:
    stamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(stamp):
        return {
            "raw_date": "", "trading_date": "", "calendar_state": "INVALID_DATE",
            "adjustment_days": 0, "is_trading_day": False, "calendar_verified": False,
        }
    raw = pd.Timestamp(stamp).normalize()
    open_dates = _holiday_set(official_open_dates)
    closed_dates = _holiday_set(official_closed_dates) | _holiday_set(holidays)
    has_official = bool(open_dates or set(_holiday_set(official_closed_dates)))
    step = 1 if str(direction).lower() != "previous" else -1
    current = raw
    for moved in range(max(0, int(max_steps)) + 1):
        day = current.date()
        weekday_open = current.weekday() < 5
        official_known = day in open_dates or day in closed_dates
        if has_official and official_known:
            is_open = day in open_dates and day not in closed_dates
            state = "OFFICIAL_IDX_OPEN" if is_open else "OFFICIAL_IDX_CLOSED"
            verified = True
        else:
            is_open = weekday_open and day not in closed_dates
            if has_official:
                state = "PARTIAL_CALENDAR_FALLBACK_UNVERIFIED" if is_open else "WEEKEND_OR_CONFIGURED_HOLIDAY"
            else:
                state = "WEEKDAY_FALLBACK_UNVERIFIED" if is_open else "WEEKEND_OR_CONFIGURED_HOLIDAY"
            verified = False
        if is_open:
            return {
                "raw_date": raw.date().isoformat(),
                "trading_date": current.date().isoformat(),
                "calendar_state": state,
                "adjustment_days": int((current - raw).days),
                "is_trading_day": True,
                "calendar_verified": verified,
            }
        current = current + timedelta(days=step)
    return {
        "raw_date": raw.date().isoformat(), "trading_date": "", "calendar_state": "NO_OPEN_DATE_FOUND",
        "adjustment_days": 0, "is_trading_day": False, "calendar_verified": bool(has_official),
    }


def _normalise_history(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()
    if out.index.tz is not None:
        out.index = out.index.tz_convert("Asia/Jakarta").tz_localize(None)
    for column in ("Open", "High", "Low", "Close", "Volume"):
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.dropna(subset=["Close"])


def _outcome_id(ticker: str, signal_family: str, signal_date: str, anchor_id: str, model_version: str) -> str:
    return sha256(f"{ticker}|{signal_family}|{signal_date}|{anchor_id}|{model_version}".encode("utf-8")).hexdigest()


def _prediction_rows(ranking: pd.DataFrame | None, as_of: Any | None = None) -> list[dict[str, Any]]:
    if ranking is None or ranking.empty or "ticker" not in ranking:
        return []
    timestamp = pd.to_datetime(as_of, errors="coerce", utc=True)
    if pd.isna(timestamp):
        timestamp = pd.Timestamp.now(tz="UTC")
    signal_date = timestamp.tz_convert("Asia/Jakarta").date().isoformat()
    rows: list[dict[str, Any]] = []
    for _, row in ranking.drop_duplicates("ticker", keep="first").iterrows():
        ticker = _text(row.get("ticker"))
        if not ticker:
            continue
        adtv = _finite(row.get("adtv20_idr"), 0.0)
        bucket = liquidity_bucket(adtv)
        silent_score = _finite(row.get("silent_accumulation_score"), np.nan)
        silent_state = _text(row.get("silent_accumulation_state")).upper()
        minimum_confirmation = _finite(row.get("silent_accumulation_liquidity_min_confirmation"), 68.0)
        silent_signal_valid = silent_state in {"SILENT_ACCUMULATION_CONFIRMED", "EARLY_ACCUMULATION"}
        if np.isfinite(silent_score) and silent_score >= minimum_confirmation and silent_signal_valid:
            family = "SILENT_ACCUMULATION"
            anchor = _text(row.get("silent_accumulation_state")) or f"SCORE_{silent_score:.1f}"
            model = _text(row.get("silent_accumulation_version")) or MODEL_VERSIONS["silent_accumulation"]
            rows.append({
                "outcome_id": _outcome_id(ticker, family, signal_date, anchor, model),
                "ticker": ticker, "signal_family": family, "signal_timestamp": timestamp.isoformat(),
                "signal_date": signal_date, "anchor_id": anchor, "liquidity_bucket": bucket,
                "predicted_state": _text(row.get("silent_accumulation_state")),
                "predicted_direction": "BULLISH", "signal_score": silent_score,
                "signal_confidence": _finite(row.get("silent_accumulation_confidence"), np.nan),
                "prediction_window_start": signal_date, "prediction_window_end": "",
                "entry_reference": _finite(row.get("last_price"), np.nan),
                "horizon_bars": 20, "outcome_status": "OPEN", "model_version": model,
                "payload": {
                    "accumulation_regime": _text(row.get("accumulation_regime")),
                    "final_score": _finite(row.get("final_score", row.get("multibagger_quality_score")), np.nan),
                },
            })
        eoff_active = bool(row.get("eoff_signal_active")) or _text(row.get("eoff_state")).startswith("ACTIVE")
        best_buy = _text(row.get("best_buy_date"))
        validation = _text(row.get("eoff_public_validation_state") or row.get("time_cycle_state")).upper()
        if eoff_active and best_buy and validation in {"VALIDATED", "WALK_FORWARD_VALIDATED", "PUBLIC_VALIDATED"}:
            family = "EOFF"
            anchor = _text(row.get("eoff_unique_anchor_signature")) or f"{_text(row.get('eoff_fib_unique_anchor_count'))}|{best_buy}"
            model = _text(row.get("eoff_version")) or MODEL_VERSIONS["eoff"]
            rows.append({
                "outcome_id": _outcome_id(ticker, family, signal_date, anchor, model),
                "ticker": ticker, "signal_family": family, "signal_timestamp": timestamp.isoformat(),
                "signal_date": signal_date, "anchor_id": anchor, "liquidity_bucket": bucket,
                "predicted_state": _text(row.get("eoff_state")),
                "predicted_direction": _text(row.get("eoff_direction_bias")) or "NEUTRAL",
                "signal_score": _finite(row.get("eoff_reconstruction_score"), np.nan),
                "signal_confidence": _finite(row.get("best_buy_confidence", row.get("time_cycle_confidence")), np.nan),
                "prediction_window_start": _text(row.get("best_buy_window_start")) or best_buy,
                "prediction_window_end": _text(row.get("best_buy_window_end")) or best_buy,
                "entry_reference": _finite(row.get("last_price"), np.nan),
                "horizon_bars": 20, "outcome_status": "OPEN", "model_version": model,
                "payload": {
                    "best_buy_date": best_buy,
                    "calendar_state": _text(row.get("best_buy_calendar_state")),
                    "unique_anchor_count": _finite(row.get("eoff_fib_unique_anchor_count"), np.nan),
                },
            })
        category = _text(row.get("category")).upper()
        multibagger_status = _text(row.get("multibagger_status")).upper()
        multibagger_lane = _text(row.get("multibagger_lane")).upper()
        multibagger_score = _finite(
            (
                row.get(
                    "turnaround_recovery_score",
                    row.get("multibagger_quality_score"),
                )
                if multibagger_lane == "TURNAROUND_CYCLICAL"
                else row.get(
                    "multibagger_quality_score",
                    row.get("multibagger_score", row.get("final_score")),
                )
            ),
            np.nan,
        )
        explicit_research = row.get("research_eligible")
        if explicit_research is None or not _text(explicit_research):
            research_gate = multibagger_status in {
                "MULTIBAGGER_A_CANDIDATE",
                "MULTIBAGGER_B_CANDIDATE",
            }
        else:
            research_gate = (
                explicit_research is True
                or _text(explicit_research).upper() in {
                    "TRUE", "YES", "Y", "1",
                }
            )
        multibagger_eligible = bool(
            (
                "MULTIBAGGER" in category
                or multibagger_lane in {
                    "GROWTH_COMPOUNDER", "TURNAROUND_CYCLICAL",
                }
            )
            and research_gate
            and np.isfinite(multibagger_score)
        )
        if multibagger_eligible:
            # One point-in-time cohort per calendar quarter avoids registering
            # a nearly identical 12–36 month thesis on every dashboard refresh.
            local_stamp = timestamp.tz_convert("Asia/Jakarta")
            cohort_date = pd.Timestamp(local_stamp.tz_localize(None)).to_period("Q").start_time.date().isoformat()
            for label, horizon_bars, calendar_days in (
                ("12M", 252, 365),
                ("24M", 504, 730),
                ("36M", 756, 1095),
            ):
                family = f"MULTIBAGGER_{label}"
                anchor = (
                    f"{_text(row.get('fundamental_history_latest_period')) or cohort_date}"
                    f"|{multibagger_lane or 'UNCLASSIFIED'}"
                    f"|{_text(row.get('turnaround_research_state')) or multibagger_status or 'CANDIDATE'}"
                    f"|{label}"
                )
                model = MODEL_VERSIONS["ranking"]
                rows.append({
                    "outcome_id": _outcome_id(
                        ticker, family, cohort_date, anchor, model,
                    ),
                    "ticker": ticker,
                    "signal_family": family,
                    "signal_timestamp": timestamp.isoformat(),
                    "signal_date": signal_date,
                    "anchor_id": anchor,
                    "liquidity_bucket": bucket,
                    "predicted_state": (
                        _text(row.get("turnaround_research_state"))
                        if multibagger_lane == "TURNAROUND_CYCLICAL"
                        else multibagger_status
                    ) or "MULTIBAGGER_CANDIDATE",
                    "predicted_direction": "OUTPERFORM_IHSG",
                    "signal_score": multibagger_score,
                    "signal_confidence": _finite(
                        row.get("overall_research_confidence"), np.nan,
                    ),
                    "prediction_window_start": signal_date,
                    "prediction_window_end": (
                        pd.Timestamp(signal_date) + timedelta(days=calendar_days)
                    ).date().isoformat(),
                    "entry_reference": _finite(row.get("last_price"), np.nan),
                    "horizon_bars": horizon_bars,
                    "outcome_status": "OPEN",
                    "model_version": model,
                    "payload": {
                        "outcome_schema": RESEARCH_OUTCOME_SCHEMA_VERSION,
                        "multibagger_quality_score": multibagger_score,
                        "multibagger_lane": multibagger_lane,
                        "growth_compounder_score": _finite(
                            row.get("growth_compounder_score"), np.nan,
                        ),
                        "turnaround_recovery_score": _finite(
                            row.get("turnaround_recovery_score"), np.nan,
                        ),
                        "turnaround_research_state": _text(
                            row.get("turnaround_research_state"),
                        ),
                        "portfolio_allocation_eligible": bool(
                            row.get("portfolio_allocation_eligible", False),
                        ),
                        "silent_accumulation_score": silent_score,
                        "sector": _text(row.get("sector")),
                        "fundamental_inflection_score": _finite(
                            row.get("fundamental_inflection_score"), np.nan,
                        ),
                        "corporate_action_policy": "ADJUSTED_OHLCV_WHEN_PROVIDER_AVAILABLE",
                        "delisting_policy": "KEEP_OPEN_OR_MARK_UNAVAILABLE; NEVER_IMPUTE_A_WIN",
                    },
                })
    return rows


def _resolve_one(record: Mapping[str, Any], history: pd.DataFrame) -> dict[str, Any]:
    local = dict(record)
    frame = _normalise_history(history)
    signal_date = pd.to_datetime(local.get("signal_date") or local.get("signal_timestamp"), errors="coerce")
    if frame.empty or pd.isna(signal_date):
        return local
    signal_date = pd.Timestamp(signal_date)
    if signal_date.tzinfo is not None:
        signal_date = signal_date.tz_convert("Asia/Jakarta").tz_localize(None)
    future = frame.loc[frame.index.normalize() > pd.Timestamp(signal_date).normalize()].head(max(20, int(_finite(local.get("horizon_bars"), 20))))
    if len(future) < 5:
        return local
    entry = _finite(local.get("entry_reference"), np.nan)
    if not np.isfinite(entry) or entry <= 0:
        entry = _finite(frame.loc[frame.index.normalize() <= pd.Timestamp(signal_date).normalize(), "Close"].tail(1).squeeze(), np.nan)
    if not np.isfinite(entry) or entry <= 0:
        return local
    closes = pd.to_numeric(future["Close"], errors="coerce")
    highs = pd.to_numeric(future.get("High", closes), errors="coerce")
    lows = pd.to_numeric(future.get("Low", closes), errors="coerce")
    for horizon in (1, 5, 10, 20):
        if len(closes) >= horizon and np.isfinite(closes.iloc[horizon - 1]):
            local[f"forward_return_{horizon}d"] = float(closes.iloc[horizon - 1] / entry - 1.0)
    local["maximum_favourable_excursion"] = float(highs.max() / entry - 1.0) if highs.notna().any() else np.nan
    local["maximum_adverse_excursion"] = float(lows.min() / entry - 1.0) if lows.notna().any() else np.nan
    local["actual_low_date"] = pd.Timestamp(lows.idxmin()).date().isoformat() if lows.notna().any() else ""
    local["actual_high_date"] = pd.Timestamp(highs.idxmax()).date().isoformat() if highs.notna().any() else ""
    family = _text(local.get("signal_family")).upper()
    direction = _text(local.get("predicted_direction")).upper()
    r10 = _finite(local.get("forward_return_10d"), np.nan)
    r20 = _finite(local.get("forward_return_20d"), np.nan)
    mfe = _finite(local.get("maximum_favourable_excursion"), np.nan)
    mae = _finite(local.get("maximum_adverse_excursion"), np.nan)
    if family.startswith("MULTIBAGGER_"):
        horizon = max(1, int(_finite(local.get("horizon_bars"), 252)))
        payload = dict(local.get("payload", {}) or {})
        if len(closes) >= horizon:
            stock_return = float(closes.iloc[horizon - 1] / entry - 1.0)
            benchmark = pd.to_numeric(
                future.get("BENCH_CLOSE", pd.Series(np.nan, index=future.index)),
                errors="coerce",
            )
            benchmark_start = _finite(
                frame.loc[
                    frame.index.normalize() <= pd.Timestamp(signal_date).normalize(),
                    "BENCH_CLOSE",
                ].tail(1).squeeze()
                if "BENCH_CLOSE" in frame else np.nan,
                np.nan,
            )
            benchmark_return = (
                float(benchmark.iloc[horizon - 1] / benchmark_start - 1.0)
                if len(benchmark) >= horizon
                and np.isfinite(benchmark.iloc[horizon - 1])
                and benchmark_start > 0
                else np.nan
            )
            net_excess = (
                stock_return - benchmark_return - 0.0065
                if np.isfinite(benchmark_return) else np.nan
            )
            payload.update({
                "forward_return_horizon": stock_return,
                "benchmark_return_horizon": benchmark_return,
                "net_excess_return_horizon": net_excess,
                "outperformed_ihsg_after_cost": (
                    bool(net_excess > 0.0) if np.isfinite(net_excess) else None
                ),
                "multiple_2x_hit": bool(stock_return >= 1.0),
                "multiple_3x_hit": bool(stock_return >= 2.0),
            })
            local["payload"] = payload
            hit = bool(net_excess > 0.0) if np.isfinite(net_excess) else bool(stock_return > 0.0)
        else:
            hit = False
    elif family == "EOFF":
        if direction == "BEARISH":
            hit = bool(np.isfinite(r10) and r10 < 0 and np.isfinite(mae) and abs(mae) >= max(0.02, mfe if np.isfinite(mfe) else 0.0))
        else:
            hit = bool(np.isfinite(r10) and r10 > 0 and np.isfinite(mfe) and mfe >= max(0.02, abs(mae) if np.isfinite(mae) else 0.0))
    else:
        hit = bool(np.isfinite(r20) and r20 > 0 and np.isfinite(mfe) and mfe >= max(0.03, abs(mae) if np.isfinite(mae) else 0.0))
    if len(future) >= int(_finite(local.get("horizon_bars"), 20)):
        local["hit"] = hit
        local["outcome_status"] = "RESOLVED"
        local["resolved_at"] = pd.Timestamp(future.index[min(len(future), int(_finite(local.get("horizon_bars"), 20))) - 1]).tz_localize("UTC").isoformat() if pd.Timestamp(future.index[0]).tzinfo is None else pd.Timestamp(future.index[min(len(future), int(_finite(local.get("horizon_bars"), 20))) - 1]).tz_convert("UTC").isoformat()
    return local


def update_research_outcomes(
    existing: pd.DataFrame | None,
    ranking: pd.DataFrame | None,
    prepared: Mapping[str, pd.DataFrame] | None,
    *,
    as_of: Any | None = None,
) -> pd.DataFrame:
    records: dict[str, dict[str, Any]] = {}
    if isinstance(existing, pd.DataFrame) and not existing.empty:
        for row in existing.to_dict("records"):
            outcome_id = _text(row.get("outcome_id"))
            if outcome_id:
                records[outcome_id] = dict(row)
    for row in _prediction_rows(ranking, as_of=as_of):
        records.setdefault(row["outcome_id"], row)
    histories = prepared or {}
    for outcome_id, row in list(records.items()):
        if _text(row.get("outcome_status")).upper() == "RESOLVED":
            continue
        ticker = _text(row.get("ticker"))
        history = histories.get(ticker)
        if history is not None:
            records[outcome_id] = _resolve_one(row, history)
    columns = [
        "outcome_id", "ticker", "signal_family", "signal_timestamp", "signal_date", "anchor_id",
        "liquidity_bucket", "predicted_state", "predicted_direction", "signal_score",
        "signal_confidence", "prediction_window_start", "prediction_window_end", "entry_reference",
        "horizon_bars", "outcome_status", "resolved_at", "actual_low_date", "actual_high_date",
        "forward_return_1d", "forward_return_5d", "forward_return_10d", "forward_return_20d",
        "maximum_favourable_excursion", "maximum_adverse_excursion", "hit", "model_version", "payload",
    ]
    frame = pd.DataFrame(list(records.values()))
    if frame.empty:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in frame:
            frame[column] = np.nan
    return frame[columns].sort_values(["signal_timestamp", "ticker", "signal_family"], ascending=[False, True, True], kind="stable").reset_index(drop=True)


def research_outcome_summary(outcomes: pd.DataFrame | None) -> pd.DataFrame:
    if outcomes is None or outcomes.empty:
        return pd.DataFrame(columns=[
            "signal_family", "liquidity_bucket", "events", "resolved",
            "hit_rate_pct", "median_horizon_return_pct", "horizon_return_column",
            "median_return_20d_pct", "median_mfe_pct", "median_mae_pct",
        ])
    rows = []
    grouped = outcomes.groupby(["signal_family", "liquidity_bucket"], dropna=False)
    for (family, bucket), group in grouped:
        resolved = group[group.get("outcome_status", pd.Series(index=group.index, dtype=str)).astype(str).str.upper().eq("RESOLVED")]
        hits = pd.to_numeric(resolved.get("hit"), errors="coerce").dropna()
        horizon_match = re.search(r"_(1|5|10|20)D$", _text(family).upper())
        horizon_column = (
            f"forward_return_{horizon_match.group(1)}d"
            if horizon_match else "forward_return_20d"
        )
        if _text(family).upper().startswith("MULTIBAGGER_"):
            horizon_returns = pd.Series([
                _finite(
                    (payload if isinstance(payload, Mapping) else {}).get(
                        "net_excess_return_horizon",
                        (payload if isinstance(payload, Mapping) else {}).get(
                            "forward_return_horizon",
                        ),
                    ),
                    np.nan,
                )
                for payload in resolved.get(
                    "payload", pd.Series({}, index=resolved.index),
                )
            ], index=resolved.index, dtype=float)
            horizon_column = "payload.net_excess_return_horizon"
        else:
            horizon_returns = pd.to_numeric(
                resolved.get(horizon_column, pd.Series(index=resolved.index, dtype=float)),
                errors="coerce",
            )
        rows.append({
            "signal_family": family,
            "liquidity_bucket": bucket,
            "events": len(group),
            "resolved": len(resolved),
            "hit_rate_pct": round(100.0 * float(hits.mean()), 1) if len(hits) else np.nan,
            "median_horizon_return_pct": round(
                100.0 * float(horizon_returns.median()), 2
            ) if horizon_returns.notna().any() else np.nan,
            "horizon_return_column": horizon_column,
            "median_return_20d_pct": round(100.0 * float(pd.to_numeric(resolved.get("forward_return_20d"), errors="coerce").median()), 2) if len(resolved) else np.nan,
            "median_mfe_pct": round(100.0 * float(pd.to_numeric(resolved.get("maximum_favourable_excursion"), errors="coerce").median()), 2) if len(resolved) else np.nan,
            "median_mae_pct": round(100.0 * float(pd.to_numeric(resolved.get("maximum_adverse_excursion"), errors="coerce").median()), 2) if len(resolved) else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["signal_family", "liquidity_bucket"], kind="stable").reset_index(drop=True)


__all__ = [
    "RESEARCH_MAINTENANCE_VERSION", "RESEARCH_OUTCOME_SCHEMA_VERSION", "SCANNER_VERSION", "MODEL_VERSIONS",
    "parse_semver", "semantic_refresh_reason", "model_registry_frame", "liquidity_bucket",
    "SilentAccumulationCalibration", "calibrate_silent_accumulation", "stable_cohort", "active_cohort",
    "select_round_robin_backfill", "adjust_to_idx_trading_day", "update_research_outcomes",
    "research_outcome_summary",
]
