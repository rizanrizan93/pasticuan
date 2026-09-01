from __future__ import annotations

"""Point-in-time visibility rules for scanner-neutral factual evidence."""

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Iterable, Mapping

import pandas as pd

from idx_trading_calendar import (
    CalendarState,
    calendar_state,
    latest_expected_completed_session,
)
from shared_evidence_hub import MissingReason


class EvidenceDateKind(str, Enum):
    TRADE_DATE = "TRADE_DATE"
    REPORT_DATE = "REPORT_DATE"
    EVENT_DATE = "EVENT_DATE"
    PUBLICATION_DATE = "PUBLICATION_DATE"
    FETCHED_AT = "FETCHED_AT"


@dataclass(frozen=True)
class TemporalDecision:
    available: bool
    reason: str
    evidence_date_kind: str
    evidence_date: str | None
    available_at: str | None


def _timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or value is pd.NA or str(value).strip() == "":
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(stamp):
        return None
    if stamp.tzinfo is None:
        return stamp.tz_localize("Asia/Jakarta")
    return stamp.tz_convert("Asia/Jakarta")


def _first_timestamp(row: Mapping[str, Any], names: Iterable[str]) -> pd.Timestamp | None:
    for name in names:
        stamp = _timestamp(row.get(name))
        if stamp is not None:
            return stamp
    return None


def _iso(stamp: pd.Timestamp | None) -> str | None:
    return stamp.isoformat() if stamp is not None else None


def evidence_available_as_of(
    row: Mapping[str, Any],
    *,
    as_of: Any,
    date_kind: EvidenceDateKind | str,
) -> TemporalDecision:
    try:
        kind = date_kind if isinstance(date_kind, EvidenceDateKind) else EvidenceDateKind(str(date_kind))
    except ValueError:
        raise ValueError("UNKNOWN_EVIDENCE_DATE_KIND") from None
    cutoff = _timestamp(as_of)
    if cutoff is None:
        raise ValueError("INVALID_AS_OF")

    if kind is EvidenceDateKind.TRADE_DATE:
        observed = _first_timestamp(row, ("trade_date", "activity_date", "date"))
        if observed is None:
            return TemporalDecision(False, MissingReason.CONTEXT_REJECTED.value, kind.value, None, None)
        day = observed.tz_localize(None).normalize()
        if calendar_state(day) is not CalendarState.TRADING_SESSION:
            return TemporalDecision(False, MissingReason.WRONG_PERIOD.value, kind.value, day.date().isoformat(), None)
        latest = latest_expected_completed_session(cutoff)
        available = day <= latest
        return TemporalDecision(
            available,
            "VALID" if available else MissingReason.CONTEXT_REJECTED.value,
            kind.value,
            day.date().isoformat(),
            day.date().isoformat() if available else None,
        )

    if kind is EvidenceDateKind.REPORT_DATE:
        evidence = _first_timestamp(row, ("report_date", "report_period", "period_end", "reporting_period"))
        available_at = _first_timestamp(row, ("published_at", "publication_date", "filing_date", "available_at"))
        if evidence is None or available_at is None:
            return TemporalDecision(False, MissingReason.CONTEXT_REJECTED.value, kind.value, _iso(evidence), _iso(available_at))
        coherent = evidence.normalize() <= available_at.normalize()
    elif kind is EvidenceDateKind.EVENT_DATE:
        evidence = _first_timestamp(row, ("event_date", "effective_date", "date"))
        available_at = _first_timestamp(row, ("published_at", "publication_date", "observed_on", "fetched_at"))
        if evidence is None or available_at is None:
            return TemporalDecision(False, MissingReason.CONTEXT_REJECTED.value, kind.value, _iso(evidence), _iso(available_at))
        coherent = evidence.normalize() <= available_at.normalize()
    elif kind is EvidenceDateKind.PUBLICATION_DATE:
        evidence = _first_timestamp(row, ("published_at", "publication_date"))
        available_at = evidence
        if evidence is None:
            return TemporalDecision(False, MissingReason.CONTEXT_REJECTED.value, kind.value, None, None)
        coherent = True
    else:
        evidence = _first_timestamp(row, ("fetched_at",))
        available_at = evidence
        if evidence is None:
            return TemporalDecision(False, MissingReason.CONTEXT_REJECTED.value, kind.value, None, None)
        coherent = True

    if not coherent:
        return TemporalDecision(
            False, MissingReason.WRONG_PERIOD.value, kind.value, _iso(evidence), _iso(available_at)
        )
    available = bool(available_at <= cutoff)
    return TemporalDecision(
        available,
        "VALID" if available else MissingReason.CONTEXT_REJECTED.value,
        kind.value,
        _iso(evidence),
        _iso(available_at),
    )


def filter_evidence_as_of(
    rows: Iterable[Mapping[str, Any]],
    *,
    as_of: Any,
    date_kind: EvidenceDateKind | str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    accepted: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    for row in rows:
        decision = evidence_available_as_of(row, as_of=as_of, date_kind=date_kind)
        if decision.available:
            accepted.append(dict(row))
        else:
            reasons[decision.reason] = reasons.get(decision.reason, 0) + 1
    return accepted, dict(sorted(reasons.items()))


__all__ = [
    "EvidenceDateKind", "TemporalDecision", "evidence_available_as_of", "filter_evidence_as_of",
]
