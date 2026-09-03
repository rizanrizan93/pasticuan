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
MAX_PAGES_PER_RUN = 10
REQUEST_TIMEOUT_SECONDS = 20

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
EVENT_DATE_FIELDS = ("listingDate", "effectiveDate", "eventDate", "exDate", "date")
PUBLICATION_DATE_FIELDS = ("publicationDate", "publishedAt", "publishedDate")
EVENT_DATE_KINDS = frozenset({"POINT", "RANGE_END"})


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


def _explicit_date_group(item: Mapping[str, Any], names: Iterable[str]) -> date | None:
    values: list[date] = []
    for name in names:
        if name not in item or item.get(name) in (None, ""):
            continue
        parsed = _date(item.get(name))
        if parsed is None:
            raise RuntimeError(MissingReason.PARSE_FAILURE.value)
        values.append(parsed)
    if not values:
        return None
    if len(set(values)) != 1:
        raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
    return values[0]


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
    # Only an explicit provider action field may refine the dedicated feed identity.
    # Do not classify from titles/descriptions and do not use substring matching.
    action = re.sub(r"[^a-z0-9]+", " ", raw_action.lower()).strip()
    explicit = {
        "reverse stock split": "REVERSE_STOCK_SPLIT",
        "reverse stock": "REVERSE_STOCK_SPLIT",
        "stock consolidation": "REVERSE_STOCK_SPLIT",
        "delisting": "DELISTING",
        "partial delisting": "PARTIAL_DELISTING",
        "ipo": "IPO",
        "transaksi material": "MATERIAL_TRANSACTION",
        "stock split": "STOCK_SPLIT",
        "waran": "WARRANT_EXERCISE",
        "warrant": "WARRANT_EXERCISE",
        "warrant exercise": "WARRANT_EXERCISE",
        "rights issue": "RIGHTS_ISSUE",
        "right issue": "RIGHTS_ISSUE",
        "hmetd": "RIGHTS_ISSUE",
        "hm etd": "RIGHTS_ISSUE",
        "conversion": "CONVERSION",
        "konversi": "CONVERSION",
        "bonus shares": "BONUS_SHARES",
        "saham bonus": "BONUS_SHARES",
        "private placement": "PRIVATE_PLACEMENT",
        "non preemptive": "PRIVATE_PLACEMENT",
    }
    defaults = {
        "issued-history": "ISSUED_SHARES_OTHER",
        "additional-listings": "ADDITIONAL_LISTING",
        "rights-offerings": "RIGHTS_OFFERING",
        "stock-splits": "STOCK_SPLIT",
    }
    return explicit.get(action, defaults[feed])


def _share_facts(item: Mapping[str, Any], *, feed: str) -> tuple[Any, Any, Any, Any, Any, Any, str]:
    pre = _number(_first(item, ("sharesBefore", "preShares", "beforeShares", "previousShares", "oldShares", "totalSharesBefore")))
    reported_post = _number(_first(item, ("sharesAfter", "postShares", "afterShares", "totalSharesAfter")))
    post = reported_post
    # Issued-history is historical exchange data. A small number of legacy
    # delisting rows contain a negative source-reported sharesAfter, which
    # cannot represent a usable post-action outstanding-share total. Preserve
    # the reported value separately, but do not promote it to post_shares.
    reported_post_negative = bool(feed == "issued-history" and reported_post is not None and reported_post < 0)
    if reported_post_negative:
        post = None

    # The issued-history contract exposes generic `shares` as the share
    # quantity attached to the listing action. It is not universally an
    # arithmetic delta relative to sharesAfter. Preserve it as event_shares.
    event_shares = _number(item.get("shares")) if feed == "issued-history" else None
    delta_names: tuple[str, ...]
    if feed == "issued-history":
        delta_names = ("deltaShares", "sharesChange")
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
    elif feed == "issued-history" and event_shares is not None and post is not None:
        state = "EXPLICIT_EVENT_SHARES_POST_NO_DELTA"
    elif feed == "issued-history" and event_shares is not None:
        state = "EXPLICIT_EVENT_SHARES_ONLY"
    elif post is not None:
        state = "EXPLICIT_POST_ONLY"
    elif pre is not None:
        state = "EXPLICIT_PRE_ONLY"
    else:
        state = "NO_SHARE_FACTS"

    if reported_post_negative:
        if delta is not None and pre is not None:
            state = "REPORTED_POST_NEGATIVE_EXPLICIT_PRE_DELTA"
        elif delta is not None:
            state = "REPORTED_POST_NEGATIVE_EXPLICIT_DELTA"
        elif pre is not None:
            state = "REPORTED_POST_NEGATIVE_EXPLICIT_PRE"
        elif event_shares is not None:
            state = "REPORTED_POST_NEGATIVE_EVENT_SHARES_ONLY"
        else:
            state = "REPORTED_POST_NEGATIVE_NO_USABLE_TOTAL"

    delta_percent = explicit_percent
    if delta is not None and pre not in (None, 0):
        derived = float(delta) / float(pre) * 100.0
        if explicit_percent is not None and not math.isclose(float(explicit_percent), derived, rel_tol=1e-9, abs_tol=1e-8):
            raise RuntimeError(MissingReason.PARSE_FAILURE.value)
        delta_percent = derived
    return event_shares, reported_post, pre, post, delta, delta_percent, state


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
        if "hasMore" in root:
            has_more = bool(root.get("hasMore"))
        elif "page" in root or "limit" in root:
            page = int(root.get("page") or 1)
            limit = int(root.get("limit") or ISSUED_HISTORY_LENGTH)
            total = int(root.get("total") or len(rows))
            has_more = page * max(1, limit) < total
        else:
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
        if feed == "additional-listings":
            # ZAPI/IDX additional-listings is a monthly aggregate row with an
            # explicit startDate..lastDate span, not a point-in-time listingDate.
            # Preserve the full span and use its explicit end only as the
            # deterministic event_date identity.
            if not ticker:
                raise RuntimeError(MissingReason.ISSUER_IDENTITY_MISSING.value)
            event_start_date = _explicit_date_group(item, ("startDate",))
            event_end_date = _explicit_date_group(item, ("lastDate",))
            if event_start_date is None or event_end_date is None:
                raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
            if event_start_date > event_end_date:
                raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
            if (
                (event_start_date.year, event_start_date.month) != (period.year, period.month)
                or (event_end_date.year, event_end_date.month) != (period.year, period.month)
            ):
                raise RuntimeError(MissingReason.WRONG_PERIOD.value)
            event_date = event_end_date
            event_date_kind = "RANGE_END"
        else:
            event_date = _explicit_date_group(item, EVENT_DATE_FIELDS)
            if not ticker or event_date is None:
                continue
            if feed in MONTHLY_FEEDS and (event_date.year, event_date.month) != (period.year, period.month):
                continue
            event_start_date = None
            event_end_date = None
            event_date_kind = "POINT"
        publication_date = _explicit_date_group(item, PUBLICATION_DATE_FIELDS)
        raw_action = _clean(_first(item, ("action", "actionType", "eventType", "type")))
        event_type = _event_type(feed, raw_action)
        event_shares, reported_post, pre, post, delta, delta_percent, calculation_state = _share_facts(item, feed=feed)
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
            "event_date_kind": event_date_kind,
            "event_start_date": event_start_date.isoformat() if event_start_date else None,
            "event_end_date": event_end_date.isoformat() if event_end_date else None,
            "publication_date": publication_date.isoformat() if publication_date else None,
            "event_shares": event_shares,
            "reported_post_shares": reported_post,
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
        date_kind = _clean(row.get("event_date_kind")).upper()
        if date_kind not in EVENT_DATE_KINDS:
            return False, MissingReason.CONTEXT_REJECTED.value
        event_start_date = _date(row.get("event_start_date"))
        event_end_date = _date(row.get("event_end_date"))
        if feed == "additional-listings":
            if (
                date_kind != "RANGE_END"
                or event_start_date is None
                or event_end_date is None
                or event_start_date > event_end_date
                or event_date != event_end_date
            ):
                return False, MissingReason.CONTEXT_REJECTED.value
            if (
                (event_start_date.year, event_start_date.month) != (expected_period.year, expected_period.month)
                or (event_end_date.year, event_end_date.month) != (expected_period.year, expected_period.month)
            ):
                return False, MissingReason.WRONG_PERIOD.value
        elif date_kind != "POINT" or row.get("event_start_date") is not None or row.get("event_end_date") is not None:
            return False, MissingReason.CONTEXT_REJECTED.value
        if not _ticker(row.get("ticker")) or not row.get("source_verified") or row.get("validation_state") != "VALID":
            return False, MissingReason.CONTEXT_REJECTED.value
        publication_date = _date(row.get("publication_date"))
        if row.get("publication_date") is not None and publication_date is None:
            return False, MissingReason.PARSE_FAILURE.value
        if publication_date is not None and publication_date > observed_on:
            return False, MissingReason.WRONG_PERIOD.value
        try:
            _official_source_url(row.get("source_url"), fallback=FEEDS[feed]["url"])
        except RuntimeError:
            return False, MissingReason.CONTEXT_REJECTED.value
        numeric = [row.get(name) for name in ("event_shares", "reported_post_shares", "pre_shares", "post_shares", "delta_shares", "delta_percent", "ratio_before", "ratio_after")]
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


def _validation_detail(
    row: Mapping[str, Any], *, feed: str, source_period: date, observed_on: date
) -> str:
    """Return a categorical diagnostic only; never include provider row values."""
    expected_period = source_period.replace(day=1) if feed in MONTHLY_FEEDS else source_period
    if (
        row.get("source") != FEEDS.get(feed, {}).get("source")
        or row.get("source_feed") != feed
        or row.get("source_period") != expected_period.isoformat()
        or row.get("observed_on") != observed_on.isoformat()
    ):
        return "SOURCE_PERIOD_IDENTITY"
    event_date = _date(row.get("event_date"))
    if event_date is None:
        return "EVENT_DATE_INVALID"
    if feed in MONTHLY_FEEDS and (event_date.year, event_date.month) != (expected_period.year, expected_period.month):
        return "EVENT_DATE_WRONG_PERIOD"
    date_kind = _clean(row.get("event_date_kind")).upper()
    if date_kind not in EVENT_DATE_KINDS:
        return "EVENT_DATE_KIND_INVALID"
    event_start_date = _date(row.get("event_start_date"))
    event_end_date = _date(row.get("event_end_date"))
    if feed == "additional-listings":
        if date_kind != "RANGE_END":
            return "RANGE_KIND_INVALID"
        if event_start_date is None or event_end_date is None:
            return "RANGE_DATE_MISSING"
        if event_start_date > event_end_date or event_date != event_end_date:
            return "RANGE_DATE_INCONSISTENT"
        if (
            (event_start_date.year, event_start_date.month) != (expected_period.year, expected_period.month)
            or (event_end_date.year, event_end_date.month) != (expected_period.year, expected_period.month)
        ):
            return "RANGE_WRONG_PERIOD"
    elif date_kind != "POINT" or row.get("event_start_date") is not None or row.get("event_end_date") is not None:
        return "POINT_DATE_SEMANTICS"
    if not _ticker(row.get("ticker")):
        return "TICKER_INVALID"
    if not row.get("source_verified") or row.get("validation_state") != "VALID":
        return "SOURCE_VALIDATION_STATE"
    publication_date = _date(row.get("publication_date"))
    if row.get("publication_date") is not None and publication_date is None:
        return "PUBLICATION_DATE_INVALID"
    if publication_date is not None and publication_date > observed_on:
        return "PUBLICATION_DATE_FUTURE"
    try:
        _official_source_url(row.get("source_url"), fallback=FEEDS[feed]["url"])
    except RuntimeError:
        return "SOURCE_URL_INVALID"
    numeric = [row.get(name) for name in ("event_shares", "reported_post_shares", "pre_shares", "post_shares", "delta_shares", "delta_percent", "ratio_before", "ratio_after")]
    if any(
        value is not None
        and (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
        )
        for value in numeric
    ):
        return "NUMERIC_INVALID"
    pre, post, delta = row.get("pre_shares"), row.get("post_shares"), row.get("delta_shares")
    if pre is not None and pre < 0:
        return "PRE_SHARES_NEGATIVE"
    if post is not None and post < 0:
        return "POST_SHARES_NEGATIVE"
    if row.get("ratio_before") is not None and row["ratio_before"] <= 0:
        return "RATIO_BEFORE_NONPOSITIVE"
    if row.get("ratio_after") is not None and row["ratio_after"] <= 0:
        return "RATIO_AFTER_NONPOSITIVE"
    if (
        all(value is not None for value in (pre, post, delta))
        and not math.isclose(float(post) - float(pre), float(delta), rel_tol=1e-12, abs_tol=1e-8)
    ):
        return "SHARE_ARITHMETIC_INCONSISTENT"
    if not _clean(row.get("payload_hash")):
        return "PAYLOAD_HASH_MISSING"
    if not _clean(row.get("calculation_state")):
        return "CALCULATION_STATE_MISSING"
    return "VALID"


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
                "GET",
                FEEDS[feed]["url"],
                params=dict(params),
                headers={
                    "Accept": "application/json",
                    "User-Agent": "Shared-IDX-Evidence-Hub/capital-actions",
                    "x-api-key": self.api_key,
                },
                timeout=REQUEST_TIMEOUT_SECONDS,
                allow_redirects=False,
            )
        except requests.Timeout as exc:
            raise RuntimeError(MissingReason.TIMEOUT.value) from exc
        except requests.ConnectionError as exc:
            raise RuntimeError(MissingReason.CONNECTION_ERROR.value) from exc
        status = int(getattr(response, "status_code", 0) or 0)
        if 300 <= status < 400:
            raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
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
        page_budget = min(MAX_PAGES_PER_RUN, max(1, int(max_pages)))
        meta: dict[str, Any] = {
            "api_calls": 0,
            "pages": 0,
            "feed": feed,
            "page_budget": page_budget,
            "bounded_complete": False,
            "provider_rows": 0,
        }

        def read_current() -> list[dict[str, Any]]:
            filters = {
                "source": FEEDS[feed]["source"], "source_period": source_period.isoformat(),
                "observed_on": observed_on.isoformat(), "validation_state": "VALID",
            }
            return self.backend.read_rows(TABLE, filters, limit=50000)

        def read_empty_current() -> bool:
            rows = self.backend.read_rows(
                "evidence_provider_state",
                {
                    "provider": "ZAPI",
                    "endpoint_family": "CAPITAL_ACTIONS",
                    "scope": FEEDS[feed]["scope"],
                    "target_date": source_period.isoformat(),
                    "response_state": "VALID_EMPTY",
                },
                limit=1,
            )
            return bool(rows)

        def fetch() -> list[dict[str, Any]]:
            if not self.api_key:
                raise RuntimeError(MissingReason.ENVIRONMENT_BLOCKED.value)
            items: list[Mapping[str, Any]] = []
            bounded_complete = False
            for page_index in range(page_budget):
                if feed == "issued-history":
                    # Current ZAPI contract is page/limit; start/length are
                    # deprecated compatibility parameters. Use the canonical
                    # contract while retaining parser support for legacy envelopes.
                    params = {"page": page_index + 1, "limit": ISSUED_HISTORY_LENGTH}
                else:
                    params = {
                        "year": source_period.year, "month": source_period.month,
                        "page": page_index + 1, "length": PAGE_LENGTH,
                    }
                # Count attempted provider calls even if timeout/HTTP failure occurs.
                meta["api_calls"] += 1
                payload = self._request(feed, params)
                meta["pages"] = page_index + 1
                page_rows, has_more = _feed_rows(payload, feed=feed)
                meta["provider_rows"] += len(page_rows)
                if has_more and not page_rows:
                    raise RuntimeError(MissingReason.PARSE_FAILURE.value)
                items.extend(page_rows)
                if not has_more:
                    bounded_complete = True
                    break
            if not bounded_complete:
                # Never persist a truncated global/monthly feed when the page
                # budget is exhausted while the provider says more data exists.
                raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
            meta["bounded_complete"] = True
            rows = normalize_capital_actions(
                items, feed=feed, source_period=source_period, observed_on=observed_on
            )
            if items and not rows:
                # A non-empty provider payload that cannot produce any factual
                # rows is not a valid empty month. Fail closed so mapping/date
                # drift cannot masquerade as "no corporate action".
                raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
            return rows

        def persist(rows: list[Mapping[str, Any]]) -> int:
            written = self.backend.upsert_rows(
                TABLE, rows, conflict=("ticker", "event_type", "event_date", "source_id")
            )
            return len(written)

        def validate_rows(rows: list[Mapping[str, Any]]) -> tuple[bool, str]:
            valid, reason = validate_capital_action_rows(
                rows, feed=feed, source_period=source_period, observed_on=observed_on
            )
            meta["validation_reason"] = reason
            if not valid:
                counts: dict[str, int] = {}
                detail_counts: dict[str, int] = {}
                action_counts: dict[str, int] = {}
                calculation_state_counts: dict[str, int] = {}
                share_relation_counts: dict[str, int] = {}
                valid_rows = 0
                for row in rows:
                    row_valid, row_reason = validate_capital_action_rows(
                        [row], feed=feed, source_period=source_period, observed_on=observed_on
                    )
                    if row_valid:
                        valid_rows += 1
                    else:
                        counts[row_reason] = counts.get(row_reason, 0) + 1
                        detail = _validation_detail(
                            row, feed=feed, source_period=source_period, observed_on=observed_on
                        )
                        detail_counts[detail] = detail_counts.get(detail, 0) + 1
                        action = _clean(row.get("raw_action")) or "<EMPTY>"
                        action_counts[action] = action_counts.get(action, 0) + 1
                        calc_state = _clean(row.get("calculation_state")) or "<EMPTY>"
                        calculation_state_counts[calc_state] = calculation_state_counts.get(calc_state, 0) + 1
                        post = row.get("post_shares")
                        delta = row.get("delta_shares")
                        if isinstance(post, (int, float)) and isinstance(delta, (int, float)):
                            relation = "DELTA_GT_POST" if delta > post else ("DELTA_EQ_POST" if delta == post else "DELTA_LT_POST")
                        else:
                            relation = "RELATION_UNAVAILABLE"
                        share_relation_counts[relation] = share_relation_counts.get(relation, 0) + 1
                meta["validation_valid_rows"] = valid_rows
                meta["validation_failure_counts"] = counts
                meta["validation_failure_detail_counts"] = detail_counts
                meta["validation_failure_action_counts"] = action_counts
                meta["validation_failure_calculation_state_counts"] = calculation_state_counts
                meta["validation_failure_share_relation_counts"] = share_relation_counts
            return valid, reason

        result = self.coordinator.get_or_refresh(
            EvidenceKey("ZAPI", "CAPITAL_ACTIONS", FEEDS[feed]["scope"], source_period),
            read_current=read_current,
            fetch=fetch,
            persist=persist,
            validate=validate_rows,
            minimum_rows=1,
            lease_seconds=300,
            allow_empty_valid=True,
            read_empty_current=read_empty_current,
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
    "EVENT_DATE_KINDS", "FEEDS", "ISSUED_HISTORY_LENGTH", "MAX_PAGES_PER_RUN", "MONTHLY_FEEDS", "PAGE_LENGTH",
    "REQUEST_TIMEOUT_SECONDS",
    "SharedCapitalActionEvidence", "normalize_capital_actions",
    "validate_capital_action_rows",
]
