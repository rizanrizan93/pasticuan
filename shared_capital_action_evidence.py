from __future__ import annotations

"""Scanner-neutral issued-share and capital-action evidence.

Only explicit numeric fields are persisted or combined.  Titles, ratios, and
event labels are never used to invent a share count or dilution percentage.
"""

from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
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


TABLE = "evidence_capital_actions"
PAGE_LENGTH = 200
ISSUED_HISTORY_LENGTH = 500

FEEDS: dict[str, dict[str, Any]] = {
    "issued-history": {
        "url": "https://api.zpi.web.id/v1/finance:idx/issued-history",
        "source": "IDX_ISSUED_HISTORY_VIA_ZAPI",
        "scope": "IDX_GLOBAL_ISSUED_HISTORY",
        "pagination": "offset",
    },
    "additional-listings": {
        "url": "https://api.zpi.web.id/v1/finance:idx/additional-listings",
        "source": "IDX_ADDITIONAL_LISTINGS_VIA_ZAPI",
        "scope": "IDX_GLOBAL_ADDITIONAL_LISTINGS_MONTH",
        "pagination": "page",
    },
    "rights-offerings": {
        "url": "https://api.zpi.web.id/v1/finance:idx/rights-offerings",
        "source": "IDX_RIGHTS_OFFERINGS_VIA_ZAPI",
        "scope": "IDX_GLOBAL_RIGHTS_OFFERINGS_MONTH",
        "pagination": "page",
    },
    "stock-splits": {
        "url": "https://api.zpi.web.id/v1/finance:idx/stock-splits",
        "source": "IDX_STOCK_SPLITS_VIA_ZAPI",
        "scope": "IDX_GLOBAL_STOCK_SPLITS_MONTH",
        "pagination": "page",
    },
}
MONTHLY_FEEDS = frozenset({"additional-listings", "rights-offerings", "stock-splits"})


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
    if value is None or isinstance(value, bool):
        return None
    text = _clean(value).replace("\u00a0", "").replace(" ", "")
    if not text:
        return None
    if text.endswith("%"):
        text = text[:-1]
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".") if text.rfind(",") > text.rfind(".") else text.replace(",", "")
    elif "," in text:
        tail = text.rsplit(",", 1)[-1]
        text = text.replace(",", ".") if len(tail) != 3 else text.replace(",", "")
    elif text.count(".") > 1:
        text = text.replace(".", "")
    try:
        parsed = Decimal(text)
    except (InvalidOperation, ValueError):
        return None
    if not parsed.is_finite():
        return None
    integral = parsed.to_integral_value()
    return int(integral) if parsed == integral else float(parsed)


def _first(item: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in item and item.get(name) not in (None, ""):
            return item.get(name)
    return None


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode("utf-8")
    ).hexdigest()


def _official_source_url(value: Any, *, fallback: str) -> str:
    text = _clean(value)
    if not text:
        return fallback
    parsed = urlparse(text)
    host = (parsed.hostname or "").lower()
    allowed = any(host == domain or host.endswith(f".{domain}") for domain in ("idx.co.id", "idx.id", "zpi.web.id"))
    if parsed.scheme != "https" or not allowed:
        raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
    return text


def _event_type(feed: str, raw_action: str) -> str:
    action = re.sub(r"[^a-z0-9]+", " ", raw_action.lower()).strip()
    if "reverse" in action or "stock consolidation" in action:
        return "REVERSE_STOCK_SPLIT"
    if "split" in action:
        return "STOCK_SPLIT"
    if "waran" in action or "warrant" in action:
        return "WARRANT_EXERCISE"
    if "right" in action or "hm etd" in action or "hmetd" in action:
        return "RIGHTS_ISSUE"
    if "convert" in action or "konvers" in action:
        return "CONVERSION"
    if "bonus" in action:
        return "BONUS_SHARES"
    if "private placement" in action or "non preemptive" in action:
        return "PRIVATE_PLACEMENT"
    defaults = {
        "issued-history": "ISSUED_SHARES_OTHER",
        "additional-listings": "ADDITIONAL_LISTING",
        "rights-offerings": "RIGHTS_OFFERING",
        "stock-splits": "STOCK_SPLIT",
    }
    return defaults[feed]


def _share_facts(item: Mapping[str, Any], *, feed: str) -> tuple[Any, Any, Any, Any, str]:
    pre = _number(_first(item, ("sharesBefore", "preShares", "beforeShares", "previousShares", "oldShares", "totalSharesBefore")))
    post = _number(_first(item, ("sharesAfter", "postShares", "afterShares", "totalSharesAfter")))
    delta_names: tuple[str, ...]
    if feed == "issued-history":
        delta_names = ("shares", "deltaShares", "sharesChange")
    elif feed == "additional-listings":
        delta_names = ("additionalShares", "deltaShares", "sharesChange", "shares", "numberOfShares")
    elif feed == "rights-offerings":
        delta_names = ("newSharesIssued", "offeredShares", "deltaShares", "sharesChange", "numberOfShares")
    else:
        delta_names = ("deltaShares", "sharesChange")
    delta = _number(_first(item, delta_names))
    explicit_percent = _number(_first(item, ("deltaPercent", "sharesChangePercent", "changePercent")))

    supplied = sum(value is not None for value in (pre, post, delta))
    if supplied == 3 and not math.isclose(float(post) - float(pre), float(delta), rel_tol=1e-12, abs_tol=1e-8):
        raise RuntimeError(MissingReason.PARSE_FAILURE.value)
    if pre is not None and post is not None:
        delta = post - pre
        state = "EXPLICIT_PRE_POST"
    elif delta is not None and post is not None:
        pre = post - delta
        state = "EXPLICIT_DELTA_POST_DERIVED_PRE"
    elif pre is not None and delta is not None:
        post = pre + delta
        state = "EXPLICIT_PRE_DELTA_DERIVED_POST"
    elif delta is not None:
        state = "EXPLICIT_DELTA_ONLY"
    else:
        state = "NO_SHARE_FACTS"

    delta_percent = explicit_percent
    if delta is not None and pre not in (None, 0):
        derived = float(delta) / float(pre) * 100.0
        if explicit_percent is not None and not math.isclose(float(explicit_percent), derived, rel_tol=1e-9, abs_tol=1e-8):
            raise RuntimeError(MissingReason.PARSE_FAILURE.value)
        delta_percent = derived
    return pre, post, delta, delta_percent, state


def _feed_rows(payload: Any, *, feed: str) -> tuple[list[Mapping[str, Any]], bool]:
    if feed not in FEEDS or not isinstance(payload, Mapping):
        raise RuntimeError(MissingReason.PARSE_FAILURE.value)
    root: Mapping[str, Any] = payload
    for key in ("content", "data"):
        nested = root.get(key)
        if isinstance(nested, Mapping) and any(field in nested for field in ("dataset", "provider", "items")):
            root = nested
    if _clean(root.get("dataset")).lower() != feed or _clean(root.get("provider")).lower() != "idx":
        raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
    values = root.get("items")
    if not isinstance(values, list) or any(not isinstance(item, Mapping) for item in values):
        raise RuntimeError(MissingReason.PARSE_FAILURE.value)
    rows = list(values)
    if "hasMore" in root:
        has_more = bool(root.get("hasMore"))
    elif feed == "issued-history":
        start = int(root.get("start") or 0)
        length = int(root.get("length") or ISSUED_HISTORY_LENGTH)
        total = int(root.get("total") or len(rows))
        has_more = start + max(1, length) < total
    else:
        page = int(root.get("page") or 1)
        length = int(root.get("length") or PAGE_LENGTH)
        total = int(root.get("total") or len(rows))
        has_more = page * max(1, length) < total
    return rows, has_more


def normalize_capital_actions(
    items: Iterable[Mapping[str, Any]],
    *,
    feed: str,
    source_period: date,
    observed_on: date,
    fetched_at: datetime | None = None,
) -> list[dict[str, Any]]:
    if feed not in FEEDS:
        raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
    period = source_period.replace(day=1) if feed in MONTHLY_FEEDS else source_period
    stamp = (fetched_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    normalized: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for item in items:
        ticker = _ticker(_first(item, ("code", "ticker", "stockCode", "KodeEmiten")))
        event_date = _date(_first(item, ("listingDate", "effectiveDate", "eventDate", "exDate", "date")))
        if not ticker or event_date is None:
            continue
        if feed in MONTHLY_FEEDS and (event_date.year, event_date.month) != (period.year, period.month):
            continue
        publication_date = _date(_first(item, ("publicationDate", "publishedAt", "publishedDate")))
        raw_action = _clean(_first(item, ("action", "actionType", "eventType", "type")))
        event_type = _event_type(feed, raw_action)
        pre, post, delta, delta_percent, calculation_state = _share_facts(item, feed=feed)
        ratio_before = _number(_first(item, ("ratioBefore", "oldRatio", "ratioOld")))
        ratio_after = _number(_first(item, ("ratioAfter", "newRatio", "ratioNew")))
        source_url = _official_source_url(
            _first(item, ("sourceUrl", "url", "detailUrl", "attachmentUrl")), fallback=FEEDS[feed]["url"]
        )
        payload_hash = _canonical_hash(item)
        provider_id = _clean(_first(item, ("id", "sourceId", "recordId", "dataId")))
        source_id = f"{feed.upper()}:{provider_id or payload_hash}"
        row = {
            "ticker": ticker,
            "event_type": event_type,
            "event_date": event_date.isoformat(),
            "publication_date": publication_date.isoformat() if publication_date else None,
            "pre_shares": pre,
            "post_shares": post,
            "delta_shares": delta,
            "delta_percent": delta_percent,
            "ratio_before": ratio_before,
            "ratio_after": ratio_after,
            "raw_action": raw_action or None,
            "calculation_state": calculation_state,
            "source": FEEDS[feed]["source"],
            "source_feed": feed,
            "source_period": period.isoformat(),
            "observed_on": observed_on.isoformat(),
            "source_url": source_url,
            "source_id": source_id,
            "payload_hash": payload_hash,
            "source_verified": True,
            "validation_state": "VALID",
            "fetched_at": stamp,
        }
        identity = (ticker, event_type, event_date.isoformat(), source_id)
        current = normalized.get(identity)
        if current is not None and current["payload_hash"] != payload_hash:
            raise RuntimeError(MissingReason.PARSE_FAILURE.value)
        normalized[identity] = row
    return sorted(normalized.values(), key=lambda row: (row["event_date"], row["ticker"], row["event_type"], row["source_id"]))


def validate_capital_action_rows(
    rows: Iterable[Mapping[str, Any]], *, feed: str, source_period: date, observed_on: date
) -> tuple[bool, str]:
    records = [dict(row) for row in rows]
    if not records:
        return False, MissingReason.NO_REPORT.value
    expected_period = source_period.replace(day=1) if feed in MONTHLY_FEEDS else source_period
    identities: set[tuple[str, str, str, str]] = set()
    for row in records:
        identity = (
            _clean(row.get("ticker")), _clean(row.get("event_type")),
            _clean(row.get("event_date")), _clean(row.get("source_id")),
        )
        if not all(identity) or identity in identities:
            return False, MissingReason.PARSE_FAILURE.value
        identities.add(identity)
        if (
            row.get("source") != FEEDS.get(feed, {}).get("source")
            or row.get("source_feed") != feed
            or row.get("source_period") != expected_period.isoformat()
            or row.get("observed_on") != observed_on.isoformat()
        ):
            return False, MissingReason.WRONG_PERIOD.value
        event_date = _date(row.get("event_date"))
        if event_date is None or (feed in MONTHLY_FEEDS and (event_date.year, event_date.month) != (expected_period.year, expected_period.month)):
            return False, MissingReason.WRONG_PERIOD.value
        if not _ticker(row.get("ticker")) or not row.get("source_verified") or row.get("validation_state") != "VALID":
            return False, MissingReason.CONTEXT_REJECTED.value
        numeric = [row.get(name) for name in ("pre_shares", "post_shares", "delta_shares", "delta_percent", "ratio_before", "ratio_after")]
        if any(value is not None and (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value))) for value in numeric):
            return False, MissingReason.PARSE_FAILURE.value
        pre, post, delta = row.get("pre_shares"), row.get("post_shares"), row.get("delta_shares")
        if (pre is not None and pre < 0) or (post is not None and post < 0):
            return False, MissingReason.CONTEXT_REJECTED.value
        if (
            (row.get("ratio_before") is not None and row["ratio_before"] <= 0)
            or (row.get("ratio_after") is not None and row["ratio_after"] <= 0)
        ):
            return False, MissingReason.CONTEXT_REJECTED.value
        if all(value is not None for value in (pre, post, delta)) and not math.isclose(float(post) - float(pre), float(delta), rel_tol=1e-12, abs_tol=1e-8):
            return False, MissingReason.PARSE_FAILURE.value
        if not _clean(row.get("payload_hash")) or not _clean(row.get("calculation_state")):
            return False, MissingReason.CONTEXT_REJECTED.value
    return True, "VALID"


class SharedCapitalActionEvidence:
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
        self, *, feed: str, source_period: date, observed_on: date, max_pages: int
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if feed not in FEEDS or (feed in MONTHLY_FEEDS and source_period.day != 1):
            return [], {"state": MissingReason.CONTEXT_REJECTED.value, "api_calls": 0}
        if not self.ready:
            return [], {"state": MissingReason.ENVIRONMENT_BLOCKED.value, "api_calls": 0}
        meta: dict[str, Any] = {"api_calls": 0, "pages": 0, "feed": feed}

        def read_current() -> list[dict[str, Any]]:
            filters = {
                "source": FEEDS[feed]["source"], "source_period": source_period.isoformat(),
                "observed_on": observed_on.isoformat(), "validation_state": "VALID",
            }
            return self.backend.read_rows(TABLE, filters, limit=50000)

        def fetch() -> list[dict[str, Any]]:
            if not self.api_key:
                raise RuntimeError(MissingReason.ENVIRONMENT_BLOCKED.value)
            items: list[Mapping[str, Any]] = []
            for page_index in range(max(1, int(max_pages))):
                if feed == "issued-history":
                    params = {"length": ISSUED_HISTORY_LENGTH, "start": page_index * ISSUED_HISTORY_LENGTH}
                else:
                    params = {
                        "year": source_period.year, "month": source_period.month,
                        "page": page_index + 1, "length": PAGE_LENGTH,
                    }
                payload = self._request(feed, params)
                meta["api_calls"] += 1
                meta["pages"] = page_index + 1
                page_rows, has_more = _feed_rows(payload, feed=feed)
                items.extend(page_rows)
                if not has_more or not page_rows:
                    break
            rows = normalize_capital_actions(
                items, feed=feed, source_period=source_period, observed_on=observed_on
            )
            if not rows:
                raise RuntimeError(MissingReason.NO_REPORT.value)
            return rows

        def persist(rows: list[Mapping[str, Any]]) -> int:
            written = self.backend.upsert_rows(
                TABLE, rows, conflict=("ticker", "event_type", "event_date", "source_id")
            )
            return len(written)

        result = self.coordinator.get_or_refresh(
            EvidenceKey("ZAPI", "CAPITAL_ACTIONS", FEEDS[feed]["scope"], source_period),
            read_current=read_current,
            fetch=fetch,
            persist=persist,
            validate=lambda rows: validate_capital_action_rows(
                rows, feed=feed, source_period=source_period, observed_on=observed_on
            ),
            minimum_rows=1,
            lease_seconds=300,
        )
        rows = [dict(row) for row in result.rows]
        return rows, {
            "state": result.reason, "source_period": source_period.isoformat(),
            "observed_on": observed_on.isoformat(), "rows": len(rows),
            "cache_hit": result.cache_hit, "request_avoided": result.request_avoided,
            "lease_state": result.lease_state, **meta,
        }

    def get_month(
        self, year: int, month: int, *, feed: str = "additional-listings", observed_on: date | None = None, max_pages: int = 10
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if feed not in MONTHLY_FEEDS:
            return [], {"state": MissingReason.CONTEXT_REJECTED.value, "api_calls": 0}
        try:
            period = date(int(year), int(month), 1)
        except (TypeError, ValueError):
            return [], {"state": MissingReason.WRONG_PERIOD.value, "api_calls": 0}
        return self._get(feed=feed, source_period=period, observed_on=observed_on or period, max_pages=max_pages)

    def get_issued_history(
        self, observed_on: date, *, max_pages: int = 20
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return self._get(
            feed="issued-history", source_period=observed_on,
            observed_on=observed_on, max_pages=max_pages,
        )


__all__ = [
    "FEEDS", "ISSUED_HISTORY_LENGTH", "MONTHLY_FEEDS", "PAGE_LENGTH",
    "SharedCapitalActionEvidence", "normalize_capital_actions",
    "validate_capital_action_rows",
]
