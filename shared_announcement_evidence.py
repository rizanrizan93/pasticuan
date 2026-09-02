from __future__ import annotations

"""Scanner-neutral global IDX announcement and press-release evidence.

The producer preserves official feed metadata and attachment references.  It
does not infer contract, control, dilution, dividend, M&A, or other material
event semantics from a title.  Such semantics remain unconfirmed until a
separate attachment-validation stage supplies document evidence.
"""

from datetime import date, datetime, timedelta, timezone
import hashlib
import json
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


ZAPI_ANNOUNCEMENTS_URL = "https://api.zpi.web.id/v1/finance:idx/announcements"
ZAPI_PRESS_RELEASE_URL = "https://api.zpi.web.id/v1/finance:idx/press-release"
TABLE = "evidence_announcements"
PAGE_LENGTH = 100
MAX_PAGES_PER_RUN = 5
REQUEST_TIMEOUT_SECONDS = 20
JAKARTA_TZ = timezone(timedelta(hours=7))
CONFIRMATION_STATE = "METADATA_ONLY_NOT_DOCUMENT_CONFIRMED"

FEEDS: dict[str, dict[str, str]] = {
    "announcements": {
        "url": ZAPI_ANNOUNCEMENTS_URL,
        "source": "IDX_GLOBAL_ANNOUNCEMENTS_VIA_ZAPI",
        "scope": "IDX_GLOBAL_ANNOUNCEMENTS",
        "dataset": "announcements",
    },
    "press-release": {
        "url": ZAPI_PRESS_RELEASE_URL,
        "source": "IDX_GLOBAL_PRESS_RELEASE_VIA_ZAPI",
        "scope": "IDX_GLOBAL_PRESS_RELEASE",
        "dataset": "press-release",
    },
}


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


def _timestamp(value: Any) -> datetime | None:
    text = _clean(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=JAKARTA_TZ)


def _publication_day(value: datetime) -> date:
    return value.astimezone(JAKARTA_TZ).date()


def _official_idx_url(value: Any) -> bool:
    parsed = urlparse(_clean(value))
    host = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and any(
        host == domain or host.endswith(f".{domain}") for domain in ("idx.co.id", "idx.id")
    )


def _canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _feed_rows(payload: Any, *, feed: str) -> tuple[list[Mapping[str, Any]], bool]:
    if feed not in FEEDS or not isinstance(payload, Mapping):
        raise RuntimeError(MissingReason.PARSE_FAILURE.value)
    root: Mapping[str, Any] = payload
    for _ in range(3):
        nested = root.get("data")
        if isinstance(nested, Mapping) and any(name in nested for name in ("data", "items", "dataset", "provider")):
            root = nested
        else:
            break
    expected = FEEDS[feed]["dataset"]
    if _clean(root.get("dataset")).lower() != expected or _clean(root.get("provider")).lower() != "idx":
        raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
    key = "data" if feed == "announcements" else "items"
    values = root.get(key)
    if not isinstance(values, list):
        raise RuntimeError(MissingReason.PARSE_FAILURE.value)
    rows = [item for item in values if isinstance(item, Mapping)]
    if "hasMore" in root:
        has_more = bool(root.get("hasMore"))
    else:
        page = int(root.get("page") or 1)
        length = int(root.get("length") or PAGE_LENGTH)
        total = int(root.get("total") or len(rows))
        has_more = page * max(1, length) < total
    return rows, has_more


def normalize_feed_items(
    items: Iterable[Mapping[str, Any]],
    *,
    feed: str,
    publication_date: date,
    fetched_at: datetime | None = None,
) -> list[dict[str, Any]]:
    if feed not in FEEDS:
        raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
    stamp = (fetched_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    normalized: dict[str, dict[str, Any]] = {}
    for item in items:
        published_at = _timestamp(item.get("publishedAt"))
        if published_at is None:
            continue
        if _publication_day(published_at) != publication_date:
            continue
        raw_id = _clean(item.get("id"))
        title = _clean(item.get("title"))
        if not title:
            continue
        ticker = _ticker(item.get("code")) if feed == "announcements" else None
        if feed == "announcements" and not ticker:
            continue
        attachments = item.get("attachments") or []
        if not isinstance(attachments, list):
            raise RuntimeError(MissingReason.PARSE_FAILURE.value)
        attachment_urls: list[str] = []
        for attachment in attachments:
            if not isinstance(attachment, Mapping):
                raise RuntimeError(MissingReason.PARSE_FAILURE.value)
            url = _clean(attachment.get("url"))
            if not url or not _official_idx_url(url):
                raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
            attachment_urls.append(url)
        attachment_urls = list(dict.fromkeys(attachment_urls))
        explicit_event_at = _timestamp(item.get("eventAt") or item.get("eventDate"))
        form_id = _clean(item.get("formId"))
        if feed == "announcements":
            event_type = f"IDX_DISCLOSURE_FORM_{form_id}" if form_id else "IDX_ANNOUNCEMENT_UNCLASSIFIED"
        else:
            event_type = "IDX_PRESS_RELEASE"
        canonical = {
            "feed": feed,
            "provider_id": raw_id,
            "ticker": ticker,
            "title": title,
            "subject": _clean(item.get("subject")),
            "summary": _clean(item.get("summary")),
            "announcement_no": _clean(item.get("announcementNo")),
            "form_id": form_id,
            "published_at": published_at.astimezone(timezone.utc).isoformat(),
            "event_at": explicit_event_at.astimezone(timezone.utc).isoformat() if explicit_event_at else None,
            "attachment_urls": attachment_urls,
        }
        payload_hash = _canonical_hash(canonical)
        stable_id = raw_id or payload_hash
        source_event_id = f"IDX_{feed.upper().replace('-', '_')}:{stable_id}"
        row = {
            "source_event_id": source_event_id,
            "ticker": ticker,
            "title": title,
            "subject": canonical["subject"] or None,
            "summary": canonical["summary"] or None,
            # Never substitute createdAt/publishedAt for the underlying event date.
            "event_date": _publication_day(explicit_event_at).isoformat() if explicit_event_at else None,
            "event_at": canonical["event_at"],
            "publication_date": publication_date.isoformat(),
            "published_at": canonical["published_at"],
            "event_type": event_type,
            "event_confirmation_state": CONFIRMATION_STATE,
            "announcement_no": canonical["announcement_no"] or None,
            "form_id": form_id or None,
            "attachment_count": len(attachment_urls),
            "attachment_urls": attachment_urls,
            "source": FEEDS[feed]["source"],
            "source_url": attachment_urls[0] if attachment_urls else FEEDS[feed]["url"],
            "source_document_hash": None,
            "payload_hash": payload_hash,
            "source_verified": True,
            "validation_state": "VALID",
            "fetched_at": stamp,
        }
        current = normalized.get(source_event_id)
        if current is not None and current["payload_hash"] != payload_hash:
            raise RuntimeError(MissingReason.PARSE_FAILURE.value)
        normalized[source_event_id] = row
    return sorted(normalized.values(), key=lambda row: (row["published_at"], row["source_event_id"]))


def validate_announcement_rows(
    rows: Iterable[Mapping[str, Any]], *, feed: str, publication_date: date
) -> tuple[bool, str]:
    records = [dict(row) for row in rows]
    if not records:
        return False, MissingReason.NO_REPORT.value
    ids: set[str] = set()
    expected_source = FEEDS.get(feed, {}).get("source")
    for row in records:
        identity = _clean(row.get("source_event_id"))
        if not identity or identity in ids:
            return False, MissingReason.PARSE_FAILURE.value
        ids.add(identity)
        if row.get("publication_date") != publication_date.isoformat() or row.get("source") != expected_source:
            return False, MissingReason.WRONG_PERIOD.value
        if row.get("event_confirmation_state") != CONFIRMATION_STATE or row.get("source_document_hash") is not None:
            return False, MissingReason.CONTEXT_REJECTED.value
        if not row.get("source_verified") or not _clean(row.get("title")):
            return False, MissingReason.CONTEXT_REJECTED.value
        urls = row.get("attachment_urls")
        if not isinstance(urls, list) or any(not _official_idx_url(url) for url in urls):
            return False, MissingReason.CONTEXT_REJECTED.value
    return True, "VALID"


class SharedAnnouncementEvidence:
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

    def _request_page(self, feed: str, page: int) -> Any:
        if not self.api_key:
            raise RuntimeError(MissingReason.ENVIRONMENT_BLOCKED.value)
        params: dict[str, Any] = {"page": page, "length": PAGE_LENGTH}
        if feed == "announcements":
            params["locale"] = "id"
        try:
            response = self.session.request(
                "GET",
                FEEDS[feed]["url"],
                params=params,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "Shared-IDX-Evidence-Hub/announcement-metadata",
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
        if status in {401, 403, 404, 429}:
            raise RuntimeError(f"HTTP_{status}")
        if not 200 <= status < 300:
            raise RuntimeError(f"HTTP_{status}")
        if not getattr(response, "content", b""):
            raise RuntimeError(MissingReason.EMPTY_RESPONSE.value)
        try:
            return response.json()
        except (TypeError, ValueError) as exc:
            raise RuntimeError(MissingReason.PARSE_FAILURE.value) from exc

    def get_day(
        self,
        publication_date: date,
        *,
        feed: str = "announcements",
        max_pages: int = MAX_PAGES_PER_RUN,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if feed not in FEEDS:
            return [], {"state": MissingReason.CONTEXT_REJECTED.value, "api_calls": 0}
        if not self.ready:
            return [], {"state": MissingReason.ENVIRONMENT_BLOCKED.value, "api_calls": 0}
        page_budget = min(MAX_PAGES_PER_RUN, max(1, int(max_pages)))
        meta: dict[str, Any] = {
            "api_calls": 0,
            "pages": 0,
            "feed": feed,
            "attachment_calls": 0,
            "page_budget": page_budget,
            "bounded_complete": False,
        }

        def read_current() -> list[dict[str, Any]]:
            return self.backend.read_rows(
                TABLE,
                {"source": FEEDS[feed]["source"], "publication_date": publication_date.isoformat(), "validation_state": "VALID"},
                limit=50000,
            )

        def fetch() -> list[dict[str, Any]]:
            if not self.api_key:
                raise RuntimeError(MissingReason.ENVIRONMENT_BLOCKED.value)
            all_items: list[Mapping[str, Any]] = []
            bounded_complete = False
            for page in range(1, page_budget + 1):
                # Count attempted provider requests, including timeout/HTTP failure.
                meta["api_calls"] += 1
                payload = self._request_page(feed, page)
                meta["pages"] = page
                items, has_more = _feed_rows(payload, feed=feed)
                if items:
                    stamps = [_timestamp(item.get("publishedAt")) for item in items]
                    if any(stamp is None for stamp in stamps):
                        raise RuntimeError(MissingReason.PARSE_FAILURE.value)
                    observed = [_publication_day(stamp) for stamp in stamps if stamp is not None]
                else:
                    observed = []
                all_items.extend(items)

                if not has_more:
                    bounded_complete = True
                    break
                if not items:
                    raise RuntimeError(MissingReason.PARSE_FAILURE.value)
                if observed and min(observed) < publication_date:
                    bounded_complete = True
                    break

            if not bounded_complete:
                # The page budget ended before the feed crossed the target day.
                # Persisting would silently create partial-day evidence.
                raise RuntimeError(MissingReason.CONTEXT_REJECTED.value)
            meta["bounded_complete"] = True

            rows = normalize_feed_items(all_items, feed=feed, publication_date=publication_date)
            if not rows:
                raise RuntimeError(MissingReason.NO_REPORT.value)
            return rows

        def persist(rows: list[Mapping[str, Any]]) -> int:
            written = self.backend.upsert_rows(TABLE, rows, conflict=("source_event_id",))
            return len(written)

        result = self.coordinator.get_or_refresh(
            EvidenceKey("ZAPI", "ANNOUNCEMENTS", FEEDS[feed]["scope"], publication_date),
            read_current=read_current,
            fetch=fetch,
            persist=persist,
            validate=lambda rows: validate_announcement_rows(rows, feed=feed, publication_date=publication_date),
            minimum_rows=1,
            lease_seconds=300,
        )
        rows = [dict(row) for row in result.rows]
        return rows, {
            "state": result.reason,
            "publication_date": publication_date.isoformat(),
            "rows": len(rows),
            "cache_hit": result.cache_hit,
            "request_avoided": result.request_avoided,
            "lease_state": result.lease_state,
            **meta,
        }


__all__ = [
    "CONFIRMATION_STATE",
    "FEEDS",
    "PAGE_LENGTH",
    "MAX_PAGES_PER_RUN",
    "REQUEST_TIMEOUT_SECONDS",
    "SharedAnnouncementEvidence",
    "ZAPI_ANNOUNCEMENTS_URL",
    "ZAPI_PRESS_RELEASE_URL",
    "normalize_feed_items",
    "validate_announcement_rows",
]
