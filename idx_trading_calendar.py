"""IDX trading-session calendar and OHLCV session sanitation.

Official exchange closures are authoritative for known calendar years.  For
older/provider-specific history, a conservative cross-universe consensus rule
can identify market-wide synthetic zero-volume holiday rows without deleting a
legitimate zero-volume observation from a single illiquid security.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time, timedelta
from enum import Enum
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

CALENDAR_VERSION = "2026.08.30-official-2025-2026-session-contract"
CALENDAR_SOURCE = "IDX_OFFICIAL_EXCHANGE_HOLIDAY_ANNOUNCEMENT"
JAKARTA_TIMEZONE = "Asia/Jakarta"
SESSION_COMPLETION_TIME = time(16, 20)


class CalendarState(str, Enum):
    TRADING_SESSION = "TRADING_SESSION"
    CLOSED = "CLOSED"
    UNKNOWN = "UNKNOWN"


class CalendarCoverageError(ValueError):
    """Raised when a decision asks the static calendar to invent coverage."""


# Static exchange calendars are intentionally local: normal scans and tests do
# not depend on a holiday API.  Unknown years remain UNKNOWN/fail closed.
_OFFICIAL_CLOSED_2025 = {
    "2025-01-01", "2025-01-27", "2025-01-28", "2025-01-29",
    "2025-03-28", "2025-03-31",
    "2025-04-01", "2025-04-02", "2025-04-03", "2025-04-04", "2025-04-07", "2025-04-18",
    "2025-05-01", "2025-05-12", "2025-05-13", "2025-05-29", "2025-05-30",
    "2025-06-06", "2025-06-09", "2025-06-27",
    "2025-08-18", "2025-09-05", "2025-12-25", "2025-12-26", "2025-12-31",
}

# Official 2026 IDX exchange closures (Peng-00171/BEI.POP/09-2025).
_OFFICIAL_CLOSED_2026 = {
    "2026-01-01", "2026-01-16",
    "2026-02-16", "2026-02-17",
    "2026-03-18", "2026-03-19", "2026-03-20", "2026-03-23", "2026-03-24",
    "2026-04-03",
    "2026-05-01", "2026-05-14", "2026-05-15", "2026-05-27", "2026-05-28",
    "2026-06-01", "2026-06-16",
    "2026-08-17", "2026-08-25",
    "2026-12-24", "2026-12-25", "2026-12-31",
}

_OFFICIAL_CLOSED_BY_YEAR = {2025: _OFFICIAL_CLOSED_2025, 2026: _OFFICIAL_CLOSED_2026}
SUPPORTED_CALENDAR_YEARS = frozenset(_OFFICIAL_CLOSED_BY_YEAR)
OFFICIAL_CLOSED_DATES = frozenset(
    pd.Timestamp(value).normalize()
    for values in _OFFICIAL_CLOSED_BY_YEAR.values()
    for value in values
)


def _jakarta_timestamp(value: Any) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid calendar value: {value!r}") from None
    if pd.isna(stamp):
        raise ValueError(f"invalid calendar value: {value!r}")
    if stamp.tzinfo is None:
        return stamp.tz_localize(JAKARTA_TIMEZONE)
    return stamp.tz_convert(JAKARTA_TIMEZONE)


def calendar_state(
    value: Any,
    *,
    extra_open_dates: Iterable[Any] | None = None,
    extra_closed_dates: Iterable[Any] | None = None,
) -> CalendarState:
    try:
        day = _jakarta_timestamp(value).tz_localize(None).normalize()
    except (TypeError, ValueError):
        return CalendarState.UNKNOWN
    if day.year not in SUPPORTED_CALENDAR_YEARS:
        return CalendarState.UNKNOWN
    opens = _normalise_dates(extra_open_dates)
    extra_closes = _normalise_dates(extra_closed_dates)
    if day in opens:
        return CalendarState.TRADING_SESSION
    if day.weekday() >= 5 or day in OFFICIAL_CLOSED_DATES or day in extra_closes:
        return CalendarState.CLOSED
    return CalendarState.TRADING_SESSION


def _normalise_dates(values: Iterable[Any] | None) -> set[pd.Timestamp]:
    result: set[pd.Timestamp] = set()
    for value in values or ():
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.notna(parsed):
            stamp = pd.Timestamp(parsed)
            if stamp.tzinfo is not None:
                stamp = stamp.tz_convert("Asia/Jakarta").tz_localize(None)
            result.add(stamp.normalize())
    return result


def is_idx_session(
    value: Any,
    *,
    extra_open_dates: Iterable[Any] | None = None,
    extra_closed_dates: Iterable[Any] | None = None,
) -> bool:
    return calendar_state(
        value,
        extra_open_dates=extra_open_dates,
        extra_closed_dates=extra_closed_dates,
    ) is CalendarState.TRADING_SESSION


def previous_idx_session(
    value: Any,
    *,
    include_date: bool = True,
    extra_open_dates: Iterable[Any] | None = None,
    extra_closed_dates: Iterable[Any] | None = None,
) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("Asia/Jakarta").tz_localize(None)
    day = parsed.normalize() if include_date else parsed.normalize() - timedelta(days=1)
    for _ in range(740):
        state = calendar_state(
            day,
            extra_open_dates=extra_open_dates,
            extra_closed_dates=extra_closed_dates,
        )
        if state is CalendarState.TRADING_SESSION:
            return day
        if state is CalendarState.UNKNOWN:
            raise CalendarCoverageError(f"IDX calendar has no coverage for {day.date().isoformat()}")
        day -= timedelta(days=1)
    raise RuntimeError("Tidak dapat menemukan sesi IDX sebelumnya dalam 740 hari")


def n_idx_sessions_ago(value: Any, sessions: int) -> pd.Timestamp:
    if int(sessions) < 0:
        raise ValueError("sessions must be >= 0")
    current = previous_idx_session(value, include_date=True)
    for _ in range(int(sessions)):
        current = previous_idx_session(current, include_date=False)
    return current


def latest_expected_completed_session(
    value: Any = None,
    *,
    completion_time: time = SESSION_COMPLETION_TIME,
) -> pd.Timestamp:
    local = _jakarta_timestamp(pd.Timestamp.now(tz=JAKARTA_TIMEZONE) if value is None else value)
    today = local.tz_localize(None).normalize()
    state = calendar_state(today)
    if state is CalendarState.UNKNOWN:
        raise CalendarCoverageError(f"IDX calendar has no coverage for {today.date().isoformat()}")
    include_today = (
        state is CalendarState.TRADING_SESSION
        and local.time().replace(tzinfo=None) >= completion_time
    )
    return previous_idx_session(today, include_date=include_today)


def trading_session_age(observed_at: Any, decision_at: Any) -> int | None:
    """Count completed IDX sessions after an observed trading session.

    A non-session observation or unavailable yearly calendar returns ``None``;
    callers must preserve UNKNOWN rather than treating it as fresh.
    """
    try:
        observed = _jakarta_timestamp(observed_at).tz_localize(None).normalize()
        decision = _jakarta_timestamp(decision_at).tz_localize(None).normalize()
    except (TypeError, ValueError):
        return None
    if calendar_state(observed) is not CalendarState.TRADING_SESSION:
        return None
    if calendar_state(decision) is CalendarState.UNKNOWN:
        return None
    if decision < observed:
        return -1
    age = 0
    day = observed + timedelta(days=1)
    while day <= decision:
        state = calendar_state(day)
        if state is CalendarState.UNKNOWN:
            return None
        if state is CalendarState.TRADING_SESSION:
            age += 1
        day += timedelta(days=1)
    return age


def idx_session_lag(
    last_date: Any,
    expected_date: Any,
    *,
    extra_open_dates: Iterable[Any] | None = None,
    extra_closed_dates: Iterable[Any] | None = None,
) -> int:
    """Count completed IDX sessions missing after ``last_date``.

    Calendar-day age is misleading across weekends and exchange holidays.  A
    result of zero means the series reaches the expected completed session.
    """
    last = pd.to_datetime(last_date, errors="coerce")
    expected = pd.to_datetime(expected_date, errors="coerce")
    if pd.isna(last) or pd.isna(expected):
        return 9999
    last = pd.Timestamp(last)
    expected = pd.Timestamp(expected)
    if last.tzinfo is not None:
        last = last.tz_convert("Asia/Jakarta").tz_localize(None)
    if expected.tzinfo is not None:
        expected = expected.tz_convert("Asia/Jakarta").tz_localize(None)
    last = last.normalize(); expected = expected.normalize()
    if calendar_state(last, extra_open_dates=extra_open_dates, extra_closed_dates=extra_closed_dates) is not CalendarState.TRADING_SESSION:
        return 9999
    if calendar_state(expected, extra_open_dates=extra_open_dates, extra_closed_dates=extra_closed_dates) is CalendarState.UNKNOWN:
        return 9999
    if last >= expected:
        return 0 if last == expected else -1
    lag = 0
    day = last + timedelta(days=1)
    for _ in range(3700):
        if day > expected:
            return lag
        if is_idx_session(day, extra_open_dates=extra_open_dates, extra_closed_dates=extra_closed_dates):
            lag += 1
        day += timedelta(days=1)
    return 9999


def filter_known_nontrading_dates(
    frame: pd.DataFrame,
    *,
    extra_open_dates: Iterable[Any] | None = None,
    extra_closed_dates: Iterable[Any] | None = None,
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy(), []
    out = frame.copy()
    index = pd.DatetimeIndex(pd.to_datetime(out.index, errors="coerce"))
    valid_index = ~index.isna()
    out = out.loc[valid_index].copy()
    index = index[valid_index]
    normalized = index.normalize()
    opens = _normalise_dates(extra_open_dates)
    closes = OFFICIAL_CLOSED_DATES | _normalise_dates(extra_closed_dates)
    # Vectorised equivalent of is_idx_session. The prior per-row path parsed
    # the same calendar sets hundreds of thousands of times for a 400-ticker
    # universe and dominated warm-cache latency.
    open_override = normalized.isin(list(opens)) if opens else np.zeros(len(normalized), dtype=bool)
    closed = normalized.isin(list(closes)) if closes else np.zeros(len(normalized), dtype=bool)
    weekday = normalized.weekday < 5
    mask = np.asarray(open_override | (weekday & ~closed), dtype=bool)
    removed = sorted(set(pd.Timestamp(day).normalize() for day in normalized[~mask]))
    out = out.loc[mask].copy()
    return out, removed


@dataclass(frozen=True)
class ConsensusHolidayAudit:
    inferred_dates: tuple[pd.Timestamp, ...]
    date_stats: pd.DataFrame


def infer_marketwide_zero_volume_dates(
    histories: Mapping[str, pd.DataFrame],
    *,
    min_observations: int = 5,
    minimum_zero_ratio: float = 0.80,
    minimum_price_unchanged_ratio: float = 0.70,
) -> ConsensusHolidayAudit:
    """Infer synthetic market-wide holiday rows conservatively.

    A date is inferred only when enough securities contain the date and most of
    them have non-positive/blank volume while OHLC is unchanged.  This avoids
    deleting a legitimate zero-volume day from one illiquid ticker.
    """
    records: list[pd.DataFrame] = []
    for ticker, frame in (histories or {}).items():
        if frame is None or frame.empty:
            continue
        local = frame.copy()
        local.index = pd.DatetimeIndex(pd.to_datetime(local.index, errors="coerce")).normalize()
        local = local.loc[~local.index.isna()]
        if local.empty:
            continue
        volume = pd.to_numeric(
            local.get("Volume", pd.Series(np.nan, index=local.index)),
            errors="coerce",
        )
        ohlc = local.reindex(
            columns=["Open", "High", "Low", "Close"],
        ).apply(pd.to_numeric, errors="coerce")
        price_unchanged = (
            ohlc.notna().all(axis=1)
            & ohlc.max(axis=1).sub(ohlc.min(axis=1)).abs().le(1e-12)
        )
        records.append(pd.DataFrame({
            "date": local.index,
            "ticker": str(ticker),
            "zero_volume": volume.isna() | volume.le(0.0),
            "price_unchanged": price_unchanged,
        }))
    if not records:
        return ConsensusHolidayAudit((), pd.DataFrame())
    raw = pd.concat(records, ignore_index=True, sort=False)
    stats = raw.groupby("date", as_index=False).agg(
        observations=("ticker", "nunique"),
        zero_volume_ratio=("zero_volume", "mean"),
        price_unchanged_ratio=("price_unchanged", "mean"),
    )
    candidates = stats[
        stats["observations"].ge(int(min_observations))
        & stats["zero_volume_ratio"].ge(float(minimum_zero_ratio))
        & stats["price_unchanged_ratio"].ge(float(minimum_price_unchanged_ratio))
    ].copy()
    # Known exchange sessions override inference; official closures are already
    # filtered and are harmless if included again.
    inferred = tuple(sorted(pd.Timestamp(value).normalize() for value in candidates["date"]))
    stats["inferred_nontrading"] = stats["date"].isin(inferred)
    return ConsensusHolidayAudit(inferred, stats.sort_values("date").reset_index(drop=True))


def sanitize_idx_histories(
    histories: Mapping[str, pd.DataFrame],
    *,
    extra_open_dates: Iterable[Any] | None = None,
    extra_closed_dates: Iterable[Any] | None = None,
    infer_consensus_holidays: bool = True,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    first_pass: dict[str, pd.DataFrame] = {}
    audit_rows: list[dict[str, Any]] = []
    for ticker, frame in (histories or {}).items():
        attrs = dict(getattr(frame, "attrs", {}) or {}) if frame is not None else {}
        clean, removed = filter_known_nontrading_dates(
            frame,
            extra_open_dates=extra_open_dates,
            extra_closed_dates=extra_closed_dates,
        )
        clean.attrs.update(attrs)
        first_pass[str(ticker)] = clean
        for day in removed:
            audit_rows.append({
                "ticker": str(ticker), "date": day, "reason": "OFFICIAL_OR_WEEKEND_NONTRADING",
                "calendar_version": CALENDAR_VERSION,
            })
    consensus = infer_marketwide_zero_volume_dates(first_pass) if infer_consensus_holidays else ConsensusHolidayAudit((), pd.DataFrame())
    inferred = set(consensus.inferred_dates)
    final: dict[str, pd.DataFrame] = {}
    for ticker, frame in first_pass.items():
        attrs = dict(getattr(frame, "attrs", {}) or {})
        if inferred and frame is not None and not frame.empty:
            normalized = pd.DatetimeIndex(frame.index).normalize()
            mask = ~normalized.isin(inferred)
            removed = sorted(set(normalized[~mask]))
            frame = frame.loc[mask].copy()
            for day in removed:
                audit_rows.append({
                    "ticker": ticker, "date": pd.Timestamp(day), "reason": "MARKETWIDE_ZERO_VOLUME_CONSENSUS",
                    "calendar_version": CALENDAR_VERSION,
                })
        frame.attrs.update(attrs)
        frame.attrs["idx_calendar_version"] = CALENDAR_VERSION
        frame.attrs["removed_nontrading_bars"] = int(sum(1 for row in audit_rows if row["ticker"] == ticker))
        final[ticker] = frame
    audit = pd.DataFrame(audit_rows)
    return final, audit


__all__ = [
    "CALENDAR_VERSION", "CALENDAR_SOURCE", "OFFICIAL_CLOSED_DATES",
    "is_idx_session", "previous_idx_session", "idx_session_lag", "filter_known_nontrading_dates",
    "infer_marketwide_zero_volume_dates", "sanitize_idx_histories",
]
