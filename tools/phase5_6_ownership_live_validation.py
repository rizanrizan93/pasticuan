from __future__ import annotations

"""Bounded live proof for Phase 5.6 shared ownership evidence.

The producer must perform one bounded ZAPI discovery path plus at most one
official IDX/KSEI workbook download.  The consumer is intentionally created
without a ZAPI key and must reuse the persisted factual rows without network
acquisition.
"""

import argparse
from datetime import date
import hashlib
import json
import os
from typing import Any, Mapping

from shared_ownership_evidence import (
    CATEGORIES,
    MAX_FILES_PER_PUBLICATION,
    MAX_INDEX_PAGES,
    SharedOwnershipEvidence,
    validate_ownership_rows,
)


FORBIDDEN_FACT_KEYS = frozenset({
    "beneficial_owner",
    "bandar",
    "broker_identity",
    "score",
    "rank",
    "recommendation",
    "watchlist",
    "execution_ready",
    "real_money_ready",
})


def canonical_fact_hash(rows: list[Mapping[str, Any]]) -> str:
    fields = (
        "source_file_hash",
        "category",
        "ticker",
        "holder_identity_hash",
        "holder_name",
        "report_date",
        "publication_date",
        "shares_held",
        "ownership_percentage",
        "holder_classification",
        "holder_type",
        "local_foreign_state",
        "source_url",
        "source_verified",
        "validation_state",
    )
    canonical = [
        {name: row.get(name) for name in fields}
        for row in sorted(
            (dict(row) for row in rows),
            key=lambda row: (
                str(row.get("ticker") or ""),
                str(row.get("holder_identity_hash") or ""),
            ),
        )
    ]
    payload = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("producer", "consumer"), required=True)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--category", choices=tuple(sorted(CATEGORIES)), required=True)
    parser.add_argument("--publication-date", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    publication_date = date.fromisoformat(args.publication_date)
    client_id = str(args.client_id).strip().upper()
    if client_id not in {"PASTICUAN", "EMIR"}:
        raise SystemExit("CLIENT_ID_NOT_ALLOWED")

    # Passing an explicit empty key in consumer mode proves that a successful
    # consumer run cannot fall through to ZAPI acquisition.
    evidence = SharedOwnershipEvidence(
        client_id,
        api_key=None if args.mode == "producer" else "",
    )
    rows, meta = evidence.get_publication(args.category, publication_date)
    rows = [dict(row) for row in rows]

    if not rows:
        raise SystemExit(f"OWNERSHIP_ROWS_EMPTY:{meta.get('state')}")
    valid, reason = validate_ownership_rows(rows, category=args.category)
    if not valid:
        raise SystemExit(f"OWNERSHIP_VALIDATION_FAILED:{reason}")

    report_dates = {str(row.get("report_date") or "") for row in rows}
    publication_dates = {str(row.get("publication_date") or "") for row in rows}
    source_hashes = {str(row.get("source_file_hash") or "") for row in rows}
    forbidden = sorted(
        {
            key
            for row in rows
            for key in row
            if str(key).lower() in FORBIDDEN_FACT_KEYS
        }
    )
    if len(report_dates) != 1 or "" in report_dates:
        raise SystemExit("OWNERSHIP_REPORT_DATE_NOT_UNIFORM")
    if publication_dates != {publication_date.isoformat()}:
        raise SystemExit("OWNERSHIP_PUBLICATION_DATE_MISMATCH")
    if len(source_hashes) != 1 or "" in source_hashes:
        raise SystemExit("OWNERSHIP_SOURCE_FILE_IDENTITY_AMBIGUOUS")
    if forbidden:
        raise SystemExit(f"FORBIDDEN_SHARED_SEMANTICS:{','.join(forbidden)}")

    api_calls = int(meta.get("api_calls") or 0)
    file_calls = int(meta.get("file_calls") or 0)
    state = str(meta.get("state") or "")
    if args.mode == "producer":
        if state != "REFRESHED":
            raise SystemExit(f"PRODUCER_DID_NOT_REFRESH:{state}")
        if not (1 <= api_calls <= MAX_INDEX_PAGES):
            raise SystemExit(f"PRODUCER_API_CALL_BUDGET_VIOLATION:{api_calls}")
        if file_calls != MAX_FILES_PER_PUBLICATION:
            raise SystemExit(f"PRODUCER_FILE_CALL_BUDGET_VIOLATION:{file_calls}")
        if bool(meta.get("request_avoided")):
            raise SystemExit("PRODUCER_UNEXPECTED_CACHE_REUSE")
    else:
        if state not in {"CACHE_HIT", "CACHE_FILLED_BY_OTHER_CLIENT"}:
            raise SystemExit(f"CONSUMER_DID_NOT_REUSE_CACHE:{state}")
        if api_calls != 0 or file_calls != 0:
            raise SystemExit(
                f"CONSUMER_NETWORK_BUDGET_VIOLATION:api={api_calls},file={file_calls}"
            )
        if not bool(meta.get("request_avoided")):
            raise SystemExit("CONSUMER_REQUEST_NOT_AVOIDED")

    safe_summary = {
        "client_id": client_id,
        "mode": args.mode,
        "state": state,
        "category": args.category,
        "publication_date": publication_date.isoformat(),
        "report_date": next(iter(report_dates)),
        "rows": len(rows),
        "tickers": len({str(row.get("ticker") or "") for row in rows}),
        "source_files": len(source_hashes),
        "factual_hash": canonical_fact_hash(rows),
        "api_calls": api_calls,
        "file_calls": file_calls,
        "request_avoided": bool(meta.get("request_avoided")),
        "cache_hit": bool(meta.get("cache_hit")),
        "lease_state": str(meta.get("lease_state") or ""),
        "forbidden_shared_semantics": len(forbidden),
    }
    print(json.dumps(safe_summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
