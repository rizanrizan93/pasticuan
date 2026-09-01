from __future__ import annotations

"""Scanner-neutral factual IDX risk and market-event evidence."""

from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
import re
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

import requests

from shared_evidence_hub import (
    EvidenceKey,
    HubConfig,
    MissingReason,
    SharedEvidenceCoordinator,
    SupabaseEvidenceBackend,
)


TABLE = "evidence_risk_events"
NOTICE_LENGTH = 100
MARGIN_LENGTH = 300
FEEDS: dict[str, dict[str, str]] = {
    "uma": {
        "url": "https://api.zpi.web.id/v1/finance:idx/uma",
        "source": "IDX_UMA_VIA_ZAPI",
        "event_type": "UMA_NOTICE",
        "active_state": "UMA_ACTIVE_OR_RECENT",
    },
    "suspension": {
        "url": "https://api.zpi.web.id/v1/finance:idx/suspension",
        "source": "IDX_SUSPENSION_VIA_ZAPI",
        "event_type": "SUSPENSION_NOTICE",
        "active_state": "SUSPENSION_ACTIVE_OR_RECENT",
    },
    "margin-summary": {
        "url": "https://api.zpi.web.id/v1/finance:idx/margin-summary",
        "source": "IDX_MARGIN_SUMMARY_VIA_ZAPI",
        "event_type": "MARGIN_ELIGIBILITY",
        "active_state": "MARGIN_ELIGIBLE",
    },
    "lendable-stock": {
        "url": "https://api.zpi.web.id/v1/finance:idx/lendable-stock",
        "source": "IDX_LENDABLE_STOCK_VIA_ZAPI",
        "event_type": "LENDABLE_ELIGIBILITY",
        "active_state": "LENDABLE",
    },
}
NOTICE_FEEDS = frozenset({"uma", "suspension"})
DILUTION_EVENT_TYPES = frozenset({
    "RIGHTS_ISSUE", "RIGHTS_OFFERING", "WARRANT_EXERCISE", "CONVERSION",
    "PRIVATE_PLACEMENT", "ADDITIONAL_LISTING", "ISSUED_SHARES_OTHER",
})


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _secret(name: str) -> str:
    value = _clean(os.getenv(name, ""))
    if value:
        return value
    try:
        import streamlit as st

        return _clean(st.secrets.get(name, ""))
    except Exception:
        return ""


def _ticker(value: Any) -> str | None:
    text = _clean(value).upper().removesuffix(".JK")
    return text if re.fullmatch(r"[A-Z][A-Z0-9]{3,5}", text) else None


def _date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _clean(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _number(value: Any) -> int | float | None:
    if value is None or isinstance(value, bool) or not _clean(value):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return int(parsed) if parsed.is_integer() else parsed


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode("utf-8")
    ).hexdigest()


def _official_url(value: Any, *, fallback: str) -> str:
    text = _clean(value)
    if not text:
        return fallback
    parsed = urlparse(text)
    host = (parsed.hostname or "").lower()
    allowed = any(host == domain or host.endswith(f".{domain}") for domain in ("idx.co.id", "idx.id", "zpi.web.id"))
    if parsed.scheme != "https" or not allowed:
        raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
    return text


def _unwrap(payload: Any) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise RuntimeError(MissingReason.PARSE_FAILURE.value)
    root: Mapping[str, Any] = payload
    for key in ("content", "data"):
        nested = root.get(key)
        if isinstance(nested, Mapping) and any(field in nested for field in ("dataset", "provider", "items", "data")):
            root = nested
    return root


def _feed_rows(payload: Any, *, feed: str) -> tuple[list[Mapping[str, Any]], bool, date | None]:
    if feed not in FEEDS:
        raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
    root = _unwrap(payload)
    if _clean(root.get("dataset")).lower() != feed or _clean(root.get("provider")).lower() != "idx":
        raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
    if feed == "margin-summary":
        values = root.get("data")
    else:
        values = root.get("items")
    if not isinstance(values, list) or any(not isinstance(item, Mapping) for item in values):
        raise RuntimeError(MissingReason.PARSE_FAILURE.value)
    rows = list(values)
    if "hasMore" in root:
        if not isinstance(root.get("hasMore"), bool):
            raise RuntimeError(MissingReason.PARSE_FAILURE.value)
        has_more = bool(root.get("hasMore"))
    elif feed == "margin-summary":
        start = int(root.get("start") or 0)
        length = int(root.get("length") or MARGIN_LENGTH)
        total = int(root.get("total") or len(rows))
        has_more = start + max(1, length) < total
    else:
        has_more = False
    return rows, has_more, _date(root.get("date"))


def normalize_risk_rows(
    items: Iterable[Mapping[str, Any]],
    *,
    feed: str,
    source_period: date,
    window_end_date: date,
    observed_on: date,
    provider_date: date | None = None,
    fetched_at: datetime | None = None,
) -> list[dict[str, Any]]:
    if feed not in FEEDS or window_end_date < source_period:
        raise RuntimeError(MissingReason.WRONG_PERIOD.value)
    stamp = (fetched_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    normalized: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for item in items:
        ticker = _ticker(item.get("code") or item.get("ticker"))
        if not ticker:
            continue
        publication_date: date | None = None
        title = _clean(item.get("title")) or None
        details: dict[str, Any]
        if feed in NOTICE_FEEDS:
            event_date = _date(item.get("date"))
            if event_date is None or not source_period <= event_date <= window_end_date:
                continue
            publication_date = event_date
            date_semantics = "NOTICE_PUBLICATION_DATE"
            source_url = _official_url(item.get("attachment"), fallback=FEEDS[feed]["url"])
            details = {
                "status": _clean(item.get("status")) or None,
                "info_type": _clean(item.get("infoType")) or None,
                "announcement_no": _clean(item.get("announcementNo")) or None,
                "attachment_downloaded": False,
            }
            explicit_id = _clean(item.get("id") or item.get("announcementNo"))
            raw_id = explicit_id or _canonical_hash({
                "feed": feed, "ticker": ticker, "date": event_date,
                "attachment": _clean(item.get("attachment")),
                "info_type": _clean(item.get("infoType")),
            })
        elif feed == "margin-summary":
            event_date = provider_date
            if event_date is None or event_date != source_period or window_end_date != source_period:
                raise RuntimeError(MissingReason.WRONG_PERIOD.value)
            date_semantics = "PROVIDER_PERIOD_DATE"
            source_url = FEEDS[feed]["url"]
            details = {
                name: _number(item.get(name))
                for name in ("low", "high", "close", "value", "change", "volume", "frequency")
            }
            raw_id = ticker
        else:
            event_date = observed_on
            if source_period != observed_on or window_end_date != observed_on:
                raise RuntimeError(MissingReason.WRONG_PERIOD.value)
            date_semantics = "OBSERVED_ON"
            source_url = FEEDS[feed]["url"]
            details = {
                "volume": _number(item.get("volume")),
                "regular_borrow_fee": _clean(item.get("regularBorrowFee")) or None,
                "front_end_borrow_fee": _clean(item.get("frontEndBorrowFee")) or None,
            }
            raw_id = ticker
        payload_hash = _canonical_hash(item)
        source_id = f"{feed.upper()}:{raw_id or payload_hash}"
        row = {
            "provider": "ZAPI",
            "event_type": FEEDS[feed]["event_type"],
            "event_date": event_date.isoformat(),
            "ticker": ticker,
            "source_id": source_id,
            "publication_date": publication_date.isoformat() if publication_date else None,
            "active_state": FEEDS[feed]["active_state"],
            "source": FEEDS[feed]["source"],
            "source_feed": feed,
            "source_period": source_period.isoformat(),
            "window_end_date": window_end_date.isoformat(),
            "observed_on": observed_on.isoformat(),
            "date_semantics": date_semantics,
            "title": title,
            "details": details,
            "source_url": source_url,
            "payload_hash": payload_hash,
            "source_verified": True,
            "validation_state": "VALID",
            "fetched_at": stamp,
        }
        identity = (row["event_type"], row["event_date"], ticker, source_id)
        current = normalized.get(identity)
        if current is not None and current["payload_hash"] != payload_hash:
            raise RuntimeError(MissingReason.PARSE_FAILURE.value)
        normalized[identity] = row
    return sorted(normalized.values(), key=lambda row: (row["event_date"], row["ticker"], row["source_id"]))


def derive_recent_dilution_events(
    capital_rows: Iterable[Mapping[str, Any]], *, observed_on: date, lookback_days: int = 365
) -> list[dict[str, Any]]:
    start_ordinal = observed_on.toordinal() - max(1, int(lookback_days))
    window_start = date.fromordinal(start_ordinal)
    stamp = datetime.now(timezone.utc).isoformat()
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for capital in capital_rows:
        ticker = _ticker(capital.get("ticker"))
        event_date = _date(capital.get("event_date"))
        event_type = _clean(capital.get("event_type"))
        delta = _number(capital.get("delta_shares"))
        if (
            ticker is None or event_date is None or event_type not in DILUTION_EVENT_TYPES
            or delta is None or delta <= 0 or not start_ordinal <= event_date.toordinal() <= observed_on.toordinal()
            or capital.get("validation_state") != "VALID" or not capital.get("source_verified")
        ):
            continue
        original_id = _clean(capital.get("source_id")) or _canonical_hash(capital)
        source_id = f"CAPITAL_ACTION:{original_id}"
        details = {
            "capital_event_type": event_type,
            "pre_shares": capital.get("pre_shares"),
            "post_shares": capital.get("post_shares"),
            "delta_shares": delta,
            "delta_percent": capital.get("delta_percent"),
        }
        row = {
            "provider": "DERIVED_SHARED_EVIDENCE",
            "event_type": "RECENT_DILUTION_EVENT",
            "event_date": event_date.isoformat(),
            "ticker": ticker,
            "source_id": source_id,
            "publication_date": _date(capital.get("publication_date")).isoformat() if _date(capital.get("publication_date")) else None,
            "active_state": "RECENT_DILUTION_EVENT",
            "source": _clean(capital.get("source")) or "SHARED_CAPITAL_ACTIONS",
            "source_feed": "capital-actions",
            "source_period": window_start.isoformat(),
            "window_end_date": observed_on.isoformat(),
            "observed_on": observed_on.isoformat(),
            "date_semantics": "CAPITAL_ACTION_EVENT_DATE",
            "title": None,
            "details": details,
            "source_url": _clean(capital.get("source_url")) or None,
            "payload_hash": _canonical_hash({"ticker": ticker, "event_date": event_date, "source_id": source_id, "details": details}),
            "source_verified": True,
            "validation_state": "VALID",
            "fetched_at": stamp,
        }
        identity = (ticker, source_id)
        current = rows.get(identity)
        if current is not None and current["payload_hash"] != row["payload_hash"]:
            raise RuntimeError(MissingReason.PARSE_FAILURE.value)
        rows[identity] = row
    return sorted(rows.values(), key=lambda row: (row["event_date"], row["ticker"], row["source_id"]))


def validate_risk_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    feed: str,
    source_period: date,
    window_end_date: date,
    observed_on: date,
) -> tuple[bool, str]:
    records = [dict(row) for row in rows]
    if not records:
        return False, MissingReason.NO_REPORT.value
    identities: set[tuple[str, str, str, str, str]] = set()
    derived = feed == "capital-actions"
    for row in records:
        identity = (
            _clean(row.get("provider")), _clean(row.get("event_type")), _clean(row.get("event_date")),
            _clean(row.get("ticker")), _clean(row.get("source_id")),
        )
        if not all(identity) or identity in identities:
            return False, MissingReason.PARSE_FAILURE.value
        identities.add(identity)
        if (
            row.get("source_feed") != feed
            or row.get("source_period") != source_period.isoformat()
            or row.get("window_end_date") != window_end_date.isoformat()
            or row.get("observed_on") != observed_on.isoformat()
        ):
            return False, MissingReason.WRONG_PERIOD.value
        event_date = _date(row.get("event_date"))
        if event_date is None or not source_period <= event_date <= window_end_date:
            return False, MissingReason.WRONG_PERIOD.value
        if not _ticker(row.get("ticker")) or not row.get("source_verified") or row.get("validation_state") != "VALID":
            return False, MissingReason.CONTEXT_REJECTED.value
        if not isinstance(row.get("details"), Mapping) or not _clean(row.get("payload_hash")):
            return False, MissingReason.PARSE_FAILURE.value
        if derived:
            if row.get("provider") != "DERIVED_SHARED_EVIDENCE" or row.get("active_state") != "RECENT_DILUTION_EVENT":
                return False, MissingReason.CONTEXT_REJECTED.value
        elif (
            row.get("provider") != "ZAPI"
            or row.get("source") != FEEDS.get(feed, {}).get("source")
            or row.get("active_state") != FEEDS.get(feed, {}).get("active_state")
        ):
            return False, MissingReason.CONTEXT_REJECTED.value
    return True, "VALID"


class SharedRiskEventEvidence:
    def __init__(
        self,
        client_id: str,
        *,
        backend: Any | None = None,
        coordinator: SharedEvidenceCoordinator | None = None,
        session: Any | None = None,
        api_key: str | None = None,
    ):
        self.client_id = _clean(client_id).upper() or "UNKNOWN"
        self.config = HubConfig.from_environment(client_id=self.client_id)
        self.backend = backend or (SupabaseEvidenceBackend(self.config) if self.config.ready else None)
        self.coordinator = coordinator or (
            SharedEvidenceCoordinator(self.backend, client_id=self.client_id) if self.backend is not None else None
        )
        self.session = session or requests.Session()
        self.api_key = _secret("ZAPI_KEY") if api_key is None else _clean(api_key)

    @property
    def ready(self) -> bool:
        return self.backend is not None and self.coordinator is not None

    def _request(self, feed: str, params: Mapping[str, Any]) -> Any:
        if not self.api_key:
            raise RuntimeError(MissingReason.ENVIRONMENT_BLOCKED.value)
        try:
            response = self.session.request(
                "GET", FEEDS[feed]["url"], params=dict(params),
                headers={"Accept": "application/json", "x-api-key": self.api_key}, timeout=30,
            )
        except requests.Timeout as exc:
            raise RuntimeError(MissingReason.TIMEOUT.value) from exc
        except requests.ConnectionError as exc:
            raise RuntimeError(MissingReason.CONNECTION_ERROR.value) from exc
        status = int(getattr(response, "status_code", 0) or 0)
        if not 200 <= status < 300:
            raise RuntimeError(f"HTTP_{status}")
        if not getattr(response, "content", b""):
            raise RuntimeError(MissingReason.EMPTY_RESPONSE.value)
        try:
            return response.json()
        except (TypeError, ValueError) as exc:
            raise RuntimeError(MissingReason.PARSE_FAILURE.value) from exc

    def _get(
        self, *, feed: str, source_period: date, window_end_date: date, observed_on: date, max_pages: int
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if feed not in FEEDS or window_end_date < source_period:
            return [], {"state": MissingReason.WRONG_PERIOD.value, "api_calls": 0}
        if not self.ready:
            return [], {"state": MissingReason.ENVIRONMENT_BLOCKED.value, "api_calls": 0}
        meta: dict[str, Any] = {"api_calls": 0, "pages": 0, "feed": feed, "attachment_calls": 0}

        def read_current() -> list[dict[str, Any]]:
            return self.backend.read_rows(
                TABLE,
                {
                    "source": FEEDS[feed]["source"], "source_period": source_period.isoformat(),
                    "window_end_date": window_end_date.isoformat(), "validation_state": "VALID",
                },
                limit=50000,
            )

        def fetch() -> list[dict[str, Any]]:
            items: list[Mapping[str, Any]] = []
            provider_date: date | None = None
            completed = False
            for page in range(max(1, int(max_pages))):
                if feed in NOTICE_FEEDS:
                    params: dict[str, Any] = {
                        "page": page + 1, "length": NOTICE_LENGTH,
                        "dateFrom": source_period.strftime("%Y%m%d"),
                        "dateTo": window_end_date.strftime("%Y%m%d"),
                    }
                    if feed == "suspension":
                        params.update({"type": "SPT", "locale": "id"})
                elif feed == "margin-summary":
                    params = {"date": source_period.isoformat(), "length": MARGIN_LENGTH, "start": page * MARGIN_LENGTH}
                else:
                    params = {"sort": "code", "view": "list"}
                payload = self._request(feed, params)
                meta["api_calls"] += 1
                meta["pages"] = page + 1
                page_rows, has_more, response_date = _feed_rows(payload, feed=feed)
                if response_date is not None:
                    if provider_date is not None and response_date != provider_date:
                        raise RuntimeError(MissingReason.WRONG_PERIOD.value)
                    provider_date = response_date
                items.extend(page_rows)
                if not has_more or not page_rows or feed == "lendable-stock":
                    completed = True
                    break
            if not completed:
                raise RuntimeError(MissingReason.INSUFFICIENT_HISTORY.value)
            rows = normalize_risk_rows(
                items, feed=feed, source_period=source_period, window_end_date=window_end_date,
                observed_on=observed_on, provider_date=provider_date,
            )
            if not rows:
                raise RuntimeError(MissingReason.NO_REPORT.value)
            return rows

        result = self.coordinator.get_or_refresh(
            EvidenceKey("ZAPI", "RISK_EVENTS", f"IDX_GLOBAL_{feed}_{window_end_date.isoformat()}", source_period),
            read_current=read_current,
            fetch=fetch,
            persist=lambda rows: len(self.backend.upsert_rows(
                TABLE, rows, conflict=("provider", "event_type", "event_date", "ticker", "source_id")
            )),
            validate=lambda rows: validate_risk_rows(
                rows, feed=feed, source_period=source_period,
                window_end_date=window_end_date, observed_on=observed_on,
            ),
            minimum_rows=1,
            lease_seconds=300,
        )
        rows = [dict(row) for row in result.rows]
        return rows, {
            "state": result.reason, "rows": len(rows), "source_period": source_period.isoformat(),
            "window_end_date": window_end_date.isoformat(), "observed_on": observed_on.isoformat(),
            "cache_hit": result.cache_hit, "request_avoided": result.request_avoided,
            "lease_state": result.lease_state, **meta,
        }

    def get_notice_window(
        self, date_from: date, date_to: date, *, feed: str = "uma", observed_on: date | None = None, max_pages: int = 10
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if feed not in NOTICE_FEEDS:
            return [], {"state": MissingReason.CONTEXT_REJECTED.value, "api_calls": 0}
        return self._get(
            feed=feed, source_period=date_from, window_end_date=date_to,
            observed_on=observed_on or date_to, max_pages=max_pages,
        )

    def get_margin(self, period_date: date, *, observed_on: date | None = None, max_pages: int = 3) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return self._get(
            feed="margin-summary", source_period=period_date, window_end_date=period_date,
            observed_on=observed_on or period_date, max_pages=max_pages,
        )

    def get_lendable(self, observed_on: date) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return self._get(
            feed="lendable-stock", source_period=observed_on, window_end_date=observed_on,
            observed_on=observed_on, max_pages=1,
        )

    def persist_dilution_context(
        self, capital_rows: Iterable[Mapping[str, Any]], *, observed_on: date, lookback_days: int = 365
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not self.ready:
            return [], {"state": MissingReason.ENVIRONMENT_BLOCKED.value, "api_calls": 0}
        rows = derive_recent_dilution_events(capital_rows, observed_on=observed_on, lookback_days=lookback_days)
        if not rows:
            return [], {"state": MissingReason.NO_REPORT.value, "api_calls": 0}
        window_start = date.fromordinal(observed_on.toordinal() - max(1, int(lookback_days)))
        valid, reason = validate_risk_rows(
            rows, feed="capital-actions",
            source_period=window_start, window_end_date=observed_on, observed_on=observed_on,
        )
        if not valid:
            return [], {"state": reason, "api_calls": 0}
        written = self.backend.upsert_rows(
            TABLE, rows, conflict=("provider", "event_type", "event_date", "ticker", "source_id")
        )
        return [dict(row) for row in written], {"state": "PERSISTED", "rows": len(written), "api_calls": 0}


__all__ = [
    "DILUTION_EVENT_TYPES", "FEEDS", "MARGIN_LENGTH", "NOTICE_FEEDS", "NOTICE_LENGTH",
    "SharedRiskEventEvidence", "derive_recent_dilution_events", "normalize_risk_rows",
    "validate_risk_rows",
]
