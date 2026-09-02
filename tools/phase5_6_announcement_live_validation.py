from __future__ import annotations

"""Bounded live proof for Phase 5.6 shared IDX announcement metadata."""

import argparse
from datetime import date
import hashlib
import json
from typing import Any, Mapping

from shared_announcement_evidence import (
    CONFIRMATION_STATE,
    MAX_PAGES_PER_RUN,
    SharedAnnouncementEvidence,
    validate_announcement_rows,
)


FORBIDDEN_KEYS = frozenset({
    "score",
    "rank",
    "recommendation",
    "watchlist",
    "execution_ready",
    "real_money_ready",
    "entry",
    "stop_loss",
    "take_profit",
    "contract_confirmed",
    "rights_issue_confirmed",
    "material_event_confirmed",
})


def canonical_fact_hash(rows: list[Mapping[str, Any]]) -> str:
    fields = (
        "source_event_id",
        "ticker",
        "title",
        "subject",
        "summary",
        "event_date",
        "event_at",
        "publication_date",
        "published_at",
        "event_type",
        "event_confirmation_state",
        "announcement_no",
        "form_id",
        "attachment_count",
        "attachment_urls",
        "source",
        "source_url",
        "source_document_hash",
        "payload_hash",
        "source_verified",
        "validation_state",
    )
    canonical = [
        {name: row.get(name) for name in fields}
        for row in sorted(
            (dict(row) for row in rows),
            key=lambda row: str(row.get("source_event_id") or ""),
        )
    ]
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("producer", "consumer"), required=True)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--feed", choices=("announcements", "press-release"), required=True)
    parser.add_argument("--publication-date", required=True)
    return parser.parse_args()


def main() -> int:
    args = _args()
    client_id = str(args.client_id).strip().upper()
    if client_id not in {"PASTICUAN", "EMIR"}:
        raise SystemExit("CLIENT_ID_NOT_ALLOWED")
    try:
        publication_date = date.fromisoformat(str(args.publication_date).strip())
    except ValueError as exc:
        raise SystemExit("PUBLICATION_DATE_INVALID") from exc

    evidence = SharedAnnouncementEvidence(
        client_id,
        api_key=None if args.mode == "producer" else "",
    )
    rows, meta = evidence.get_day(
        publication_date,
        feed=args.feed,
        max_pages=MAX_PAGES_PER_RUN,
    )
    rows = [dict(row) for row in rows]

    if not rows:
        raise SystemExit(f"ANNOUNCEMENT_ROWS_EMPTY:{args.feed}:{meta.get('state')}")

    valid, reason = validate_announcement_rows(
        rows,
        feed=args.feed,
        publication_date=publication_date,
    )
    if not valid:
        raise SystemExit(f"ANNOUNCEMENT_VALIDATION_FAILED:{reason}")

    forbidden = sorted({
        str(key).lower()
        for row in rows
        for key in row
        if str(key).lower() in FORBIDDEN_KEYS
    })
    if forbidden:
        raise SystemExit(f"FORBIDDEN_SHARED_SEMANTICS:{','.join(forbidden)}")
    if any(row.get("event_confirmation_state") != CONFIRMATION_STATE for row in rows):
        raise SystemExit("ANNOUNCEMENT_CONFIRMATION_STATE_INVALID")
    if any(row.get("source_document_hash") is not None for row in rows):
        raise SystemExit("ANNOUNCEMENT_DOCUMENT_HASH_UNEXPECTED")

    api_calls = int(meta.get("api_calls") or 0)
    attachment_calls = int(meta.get("attachment_calls") or 0)
    state = str(meta.get("state") or "")

    if args.mode == "producer":
        if state != "REFRESHED":
            raise SystemExit(f"PRODUCER_DID_NOT_REFRESH:{state}")
        if not (1 <= api_calls <= MAX_PAGES_PER_RUN):
            raise SystemExit(f"PRODUCER_REQUEST_BUDGET_VIOLATION:api={api_calls}")
        if attachment_calls != 0:
            raise SystemExit(
                f"PRODUCER_ATTACHMENT_BUDGET_VIOLATION:attachment={attachment_calls}"
            )
        if not bool(meta.get("bounded_complete")):
            raise SystemExit("PRODUCER_DAY_COVERAGE_NOT_BOUNDED_COMPLETE")
        if bool(meta.get("request_avoided")):
            raise SystemExit("PRODUCER_UNEXPECTED_CACHE_REUSE")
    else:
        if state not in {"CACHE_HIT", "CACHE_FILLED_BY_OTHER_CLIENT"}:
            raise SystemExit(f"CONSUMER_DID_NOT_REUSE_CACHE:{state}")
        if api_calls != 0 or attachment_calls != 0:
            raise SystemExit(
                f"CONSUMER_NETWORK_BUDGET_VIOLATION:api={api_calls},attachment={attachment_calls}"
            )
        if not bool(meta.get("request_avoided")):
            raise SystemExit("CONSUMER_REQUEST_NOT_AVOIDED")

    summary = {
        "client_id": client_id,
        "mode": args.mode,
        "feed": args.feed,
        "publication_date": publication_date.isoformat(),
        "state": state,
        "rows": len(rows),
        "ticker_rows": sum(1 for row in rows if row.get("ticker")),
        "factual_hash": canonical_fact_hash(rows),
        "api_calls": api_calls,
        "attachment_calls": attachment_calls,
        "page_budget": int(meta.get("page_budget") or MAX_PAGES_PER_RUN),
        "bounded_complete": bool(meta.get("bounded_complete")),
        "request_avoided": bool(meta.get("request_avoided")),
        "cache_hit": bool(meta.get("cache_hit")),
        "lease_state": str(meta.get("lease_state") or ""),
        "metadata_only": True,
        "forbidden_shared_semantics": len(forbidden),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
