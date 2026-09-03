from __future__ import annotations

"""Bounded live proof for Phase 5.6 shared IDX additional-listing capital-action evidence."""

import argparse
from datetime import date
import hashlib
import json
from typing import Any, Mapping

from shared_capital_action_evidence import (
    MAX_PAGES_PER_RUN,
    SharedCapitalActionEvidence,
    validate_capital_action_rows,
)


ALLOWED_FEED = "additional-listings"
FORBIDDEN_KEYS = frozenset({
    "score",
    "rank",
    "ranking",
    "recommendation",
    "watchlist",
    "execution_ready",
    "real_money_ready",
    "entry",
    "stop_loss",
    "take_profit",
    "rr",
    "signal",
    "production_gate",
    "bandar",
    "beneficial_owner",
})


def canonical_fact_hash(rows: list[Mapping[str, Any]]) -> str:
    fields = (
        "ticker",
        "event_type",
        "event_date",
        "event_date_kind",
        "event_start_date",
        "event_end_date",
        "publication_date",
        "pre_shares",
        "post_shares",
        "delta_shares",
        "delta_percent",
        "ratio_before",
        "ratio_after",
        "raw_action",
        "calculation_state",
        "source",
        "source_feed",
        "source_period",
        "observed_on",
        "source_url",
        "source_id",
        "payload_hash",
        "source_verified",
        "validation_state",
    )
    canonical = [
        {name: row.get(name) for name in fields}
        for row in sorted(
            (dict(row) for row in rows),
            key=lambda row: (
                str(row.get("event_date") or ""),
                str(row.get("ticker") or ""),
                str(row.get("event_type") or ""),
                str(row.get("source_id") or ""),
            ),
        )
    ]
    return hashlib.sha256(
        json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("producer", "consumer"), required=True)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--feed", choices=(ALLOWED_FEED,), default=ALLOWED_FEED)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--observed-on", required=True)
    return parser.parse_args()


def main() -> int:
    args = _args()
    client_id = str(args.client_id).strip().upper()
    if client_id not in {"PASTICUAN", "EMIR"}:
        raise SystemExit("CLIENT_ID_NOT_ALLOWED")
    try:
        source_period = date(int(args.year), int(args.month), 1)
        observed_on = date.fromisoformat(str(args.observed_on).strip())
    except (TypeError, ValueError) as exc:
        raise SystemExit("CAPITAL_ACTION_DATE_INVALID") from exc
    if observed_on < source_period:
        raise SystemExit("OBSERVED_ON_BEFORE_SOURCE_PERIOD")

    evidence = SharedCapitalActionEvidence(
        client_id,
        api_key=None if args.mode == "producer" else "",
    )
    rows, meta = evidence.get_month(
        source_period.year,
        source_period.month,
        feed=args.feed,
        observed_on=observed_on,
        max_pages=MAX_PAGES_PER_RUN,
    )
    rows = [dict(row) for row in rows]
    state = str(meta.get("state") or "")
    valid_empty = not rows and state in {
        "REFRESHED_EMPTY",
        "CACHE_HIT_EMPTY",
        "CACHE_FILLED_EMPTY_BY_OTHER_CLIENT",
    }

    if not rows and not valid_empty:
        diagnostic = {
            "state": state,
            "api_calls": int(meta.get("api_calls") or 0),
            "pages": int(meta.get("pages") or 0),
            "page_budget": int(meta.get("page_budget") or MAX_PAGES_PER_RUN),
            "provider_rows": int(meta.get("provider_rows") or 0),
            "bounded_complete": bool(meta.get("bounded_complete")),
        }
        print(json.dumps({"diagnostic": diagnostic}, sort_keys=True))
        raise SystemExit(f"CAPITAL_ACTION_ROWS_EMPTY:{args.feed}:{state}")

    if rows:
        valid, reason = validate_capital_action_rows(
            rows,
            feed=args.feed,
            source_period=source_period,
            observed_on=observed_on,
        )
        if not valid:
            raise SystemExit(f"CAPITAL_ACTION_VALIDATION_FAILED:{reason}")

    forbidden = sorted({
        str(key).lower()
        for row in rows
        for key in row
        if str(key).lower() in FORBIDDEN_KEYS
    })
    if forbidden:
        raise SystemExit(f"FORBIDDEN_SHARED_SEMANTICS:{','.join(forbidden)}")

    api_calls = int(meta.get("api_calls") or 0)

    if args.mode == "producer":
        if state not in {"REFRESHED", "REFRESHED_EMPTY"}:
            raise SystemExit(f"PRODUCER_DID_NOT_REFRESH:{state}")
        if not (1 <= api_calls <= MAX_PAGES_PER_RUN):
            raise SystemExit(f"PRODUCER_REQUEST_BUDGET_VIOLATION:api={api_calls}")
        if not bool(meta.get("bounded_complete")):
            raise SystemExit("PRODUCER_FEED_NOT_BOUNDED_COMPLETE")
        if bool(meta.get("request_avoided")):
            raise SystemExit("PRODUCER_UNEXPECTED_CACHE_REUSE")
    else:
        if state not in {
            "CACHE_HIT",
            "CACHE_FILLED_BY_OTHER_CLIENT",
            "CACHE_HIT_EMPTY",
            "CACHE_FILLED_EMPTY_BY_OTHER_CLIENT",
        }:
            raise SystemExit(f"CONSUMER_DID_NOT_REUSE_CACHE:{state}")
        if api_calls != 0:
            raise SystemExit(f"CONSUMER_NETWORK_BUDGET_VIOLATION:api={api_calls}")
        if not bool(meta.get("request_avoided")):
            raise SystemExit("CONSUMER_REQUEST_NOT_AVOIDED")

    summary = {
        "client_id": client_id,
        "mode": args.mode,
        "feed": args.feed,
        "source_period": source_period.isoformat(),
        "observed_on": observed_on.isoformat(),
        "state": state,
        "rows": len(rows),
        "ticker_rows": sum(1 for row in rows if row.get("ticker")),
        "factual_hash": canonical_fact_hash(rows),
        "api_calls": api_calls,
        "pages": int(meta.get("pages") or 0),
        "page_budget": int(meta.get("page_budget") or MAX_PAGES_PER_RUN),
        "provider_rows": int(meta.get("provider_rows") or 0),
        "bounded_complete": bool(meta.get("bounded_complete")),
        "request_avoided": bool(meta.get("request_avoided")),
        "cache_hit": bool(meta.get("cache_hit")),
        "lease_state": str(meta.get("lease_state") or ""),
        "forbidden_shared_semantics": len(forbidden),
        "factual_only": True,
        "valid_empty": valid_empty,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
