"""IDX trading-session calendar and OHLCV session sanitation.

Official exchange closures are authoritative for known calendar years.  For
older/provider-specific history, a conservative cross-universe consensus rule
can identify market-wide synthetic zero-volume holiday rows without deleting a
legitimate zero-volume observation from a single illiquid security.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

CALENDAR_VERSION = "2026.08.01-official-2026-consensus-history"
CALENDAR_SOURCE = "IDX_OFFICIAL_EXCHANGE_HOLIDAY_ANNOUNCEMENT"

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

OFFICIAL_CLOSED_DATES = frozenset(pd.Timestamp(value).normalize() for value in _OFFICIAL_CLOSED_2026)


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
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return False
    stamp = pd.Timestamp(parsed)
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("Asia/Jakarta").tz_localize(None)
    day = stamp.normalize()
    opens = _normalise_dates(extra_open_dates)
    closes = OFFICIAL_CLOSED_DATES | _normalise_dates(extra_closed_dates)
    if day in opens:
        return True
    if day.weekday() >= 5:
        return False
    return day not in closes


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
    for _ in range(370):
        if is_idx_session(
            day,
            extra_open_dates=extra_open_dates,
            extra_closed_dates=extra_closed_dates,
        ):
            return day
        day -= timedelta(days=1)
    raise RuntimeError("Tidak dapat menemukan sesi IDX sebelumnya dalam 370 hari")


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
