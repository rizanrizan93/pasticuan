from __future__ import annotations

"""Bounded live proof for Phase 5.6 shared financial XBRL evidence."""

import argparse
import hashlib
import json
from typing import Any, Mapping

from shared_financial_evidence import (
    SharedFinancialEvidence,
    validate_financial_facts,
)


FORBIDDEN_FACT_KEYS = frozenset({
    "score",
    "rank",
    "recommendation",
    "watchlist",
    "execution_ready",
    "real_money_ready",
    "entry",
    "stop_loss",
    "take_profit",
})


def canonical_fact_hash(rows: list[Mapping[str, Any]]) -> str:
    fields = (
        "ticker",
        "report_period",
        "fact_name",
        "fact_value",
        "currency",
        "unit_scale",
        "period_type",
        "report_date",
        "publication_date",
        "source",
        "source_url",
        "issuer_identity",
        "context_state",
        "parser_version",
        "source_document_hash",
        "validation_state",
    )
    canonical = [
        {name: row.get(name) for name in fields}
        for row in sorted(
            (dict(row) for row in rows),
            key=lambda row: str(row.get("fact_name") or ""),
        )
    ]
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("producer", "consumer"), required=True)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--year", required=True, type=int)
    parser.add_argument("--period", choices=("tw1", "tw2", "tw3", "audit"), required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    client_id = str(args.client_id).strip().upper()
    if client_id not in {"PASTICUAN", "EMIR"}:
        raise SystemExit("CLIENT_ID_NOT_ALLOWED")

    evidence = SharedFinancialEvidence(
        client_id,
        api_key=None if args.mode == "producer" else "",
    )
    rows, meta = evidence.get_period(args.ticker, args.year, args.period)
    rows = [dict(row) for row in rows]

    if not rows:
        stage = str(meta.get("failure_stage") or "UNRESOLVED_STAGE")
        raise SystemExit(f"FINANCIAL_ROWS_EMPTY:{stage}:{meta.get('state')}")

    report_period = str(rows[0].get("report_period") or "")
    valid, reason = validate_financial_facts(
        rows,
        ticker=args.ticker,
        report_period=report_period,
    )
    if not valid:
        raise SystemExit(f"FINANCIAL_VALIDATION_FAILED:{reason}")

    currencies = {str(row.get("currency") or "").upper() for row in rows}
    document_hashes = {str(row.get("source_document_hash") or "") for row in rows}
    report_dates = {str(row.get("report_date") or "") for row in rows}
    publication_dates = {str(row.get("publication_date") or "") for row in rows}
    forbidden = sorted({
        key
        for row in rows
        for key in row
        if str(key).lower() in FORBIDDEN_FACT_KEYS
    })

    if len(currencies) != 1 or "" in currencies:
        raise SystemExit("FINANCIAL_CURRENCY_AMBIGUOUS")
    if len(document_hashes) != 1 or "" in document_hashes:
        raise SystemExit("FINANCIAL_DOCUMENT_IDENTITY_AMBIGUOUS")
    if len(report_dates) != 1 or "" in report_dates:
        raise SystemExit("FINANCIAL_REPORT_DATE_AMBIGUOUS")
    if len(publication_dates) != 1 or "" in publication_dates:
        raise SystemExit("FINANCIAL_PUBLICATION_DATE_AMBIGUOUS")
    if forbidden:
        raise SystemExit(f"FORBIDDEN_SHARED_SEMANTICS:{','.join(forbidden)}")

    api_calls = int(meta.get("api_calls") or 0)
    attachment_calls = int(meta.get("attachment_calls") or 0)
    state = str(meta.get("state") or "")
    if args.mode == "producer":
        if state != "REFRESHED":
            raise SystemExit(f"PRODUCER_DID_NOT_REFRESH:{state}")
        if api_calls != 1 or attachment_calls != 1:
            raise SystemExit(
                f"PRODUCER_REQUEST_BUDGET_VIOLATION:api={api_calls},attachment={attachment_calls}"
            )
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
        "state": state,
        "ticker": str(rows[0].get("ticker") or ""),
        "report_period": report_period,
        "period_type": str(rows[0].get("period_type") or ""),
        "report_date": next(iter(report_dates)),
        "publication_date": next(iter(publication_dates)),
        "facts": len(rows),
        "currency": next(iter(currencies)),
        "source_documents": len(document_hashes),
        "factual_hash": canonical_fact_hash(rows),
        "api_calls": api_calls,
        "attachment_calls": attachment_calls,
        "request_avoided": bool(meta.get("request_avoided")),
        "cache_hit": bool(meta.get("cache_hit")),
        "lease_state": str(meta.get("lease_state") or ""),
        "forbidden_shared_semantics": len(forbidden),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
