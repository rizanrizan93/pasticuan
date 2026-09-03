from __future__ import annotations

"""Bounded Phase 5.6 live proof for scanner-neutral IDX risk evidence."""

import argparse
from datetime import date
import hashlib
import json
from typing import Any, Mapping

from shared_risk_event_evidence import SharedRiskEventEvidence, validate_risk_rows


FORBIDDEN_KEYS = frozenset({
    "score", "rank", "ranking", "recommendation", "watchlist",
    "execution_ready", "real_money_ready", "entry", "stop_loss",
    "take_profit", "rr", "signal", "production_gate", "bandar",
    "beneficial_owner", "scanner_score",
})


def _scan_forbidden(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).strip().lower()
            if lowered in FORBIDDEN_KEYS:
                found.add(lowered)
            found.update(_scan_forbidden(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_scan_forbidden(child))
    return found


def _fact_hash(rows: list[Mapping[str, Any]]) -> str:
    fields = (
        "provider", "event_type", "event_date", "ticker", "source_id",
        "publication_date", "active_state", "source", "source_feed",
        "source_period", "window_end_date", "observed_on", "date_semantics",
        "title", "details", "source_url", "payload_hash",
        "source_verified", "validation_state",
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
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode("utf-8")
    ).hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("producer", "consumer"), required=True)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--observed-on", required=True)
    parser.add_argument("--notice-from", required=True)
    parser.add_argument("--notice-to", required=True)
    parser.add_argument("--margin-date", required=True)
    return parser.parse_args()


def _parse_date(value: str, label: str) -> date:
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise SystemExit(f"{label}_INVALID") from exc


def _verify_state(mode: str, feed: str, meta: Mapping[str, Any]) -> None:
    state = str(meta.get("state") or "")
    api_calls = int(meta.get("api_calls") or 0)
    request_avoided = bool(meta.get("request_avoided"))
    if mode == "consumer":
        if state not in {"CACHE_HIT", "CACHE_FILLED_BY_OTHER_CLIENT"}:
            raise SystemExit(f"{feed}:CONSUMER_DID_NOT_REUSE_CACHE:{state}")
        if api_calls != 0 or not request_avoided:
            raise SystemExit(f"{feed}:CONSUMER_NETWORK_BUDGET_VIOLATION:{api_calls}")
    else:
        if state not in {"REFRESHED", "CACHE_HIT", "CACHE_FILLED_BY_OTHER_CLIENT"}:
            raise SystemExit(f"{feed}:PRODUCER_STATE_INVALID:{state}")
        if state == "REFRESHED" and (api_calls < 1 or request_avoided):
            raise SystemExit(f"{feed}:PRODUCER_REFRESH_AUDIT_INVALID")
        if state != "REFRESHED" and (api_calls != 0 or not request_avoided):
            raise SystemExit(f"{feed}:PRODUCER_CACHE_AUDIT_INVALID")


def main() -> int:
    args = _args()
    client_id = str(args.client_id).strip().upper()
    if client_id not in {"PASTICUAN", "EMIR"}:
        raise SystemExit("CLIENT_ID_NOT_ALLOWED")
    observed_on = _parse_date(args.observed_on, "OBSERVED_ON")
    notice_from = _parse_date(args.notice_from, "NOTICE_FROM")
    notice_to = _parse_date(args.notice_to, "NOTICE_TO")
    margin_date = _parse_date(args.margin_date, "MARGIN_DATE")
    if notice_from > notice_to or notice_to > observed_on or margin_date > observed_on:
        raise SystemExit("RISK_PROOF_DATE_CONTEXT_INVALID")

    evidence = SharedRiskEventEvidence(
        client_id, api_key=None if args.mode == "producer" else ""
    )
    results: dict[str, tuple[list[dict[str, Any]], dict[str, Any], tuple[bool, str]]] = {}

    uma_rows, uma_meta = evidence.get_notice_window(
        notice_from, notice_to, feed="uma", observed_on=observed_on, max_pages=10
    )
    results["uma"] = (
        [dict(row) for row in uma_rows], dict(uma_meta),
        validate_risk_rows(uma_rows, feed="uma", source_period=notice_from, window_end_date=notice_to, observed_on=observed_on),
    )

    suspension_rows, suspension_meta = evidence.get_notice_window(
        notice_from, notice_to, feed="suspension", observed_on=observed_on, max_pages=10
    )
    results["suspension"] = (
        [dict(row) for row in suspension_rows], dict(suspension_meta),
        validate_risk_rows(suspension_rows, feed="suspension", source_period=notice_from, window_end_date=notice_to, observed_on=observed_on),
    )

    margin_rows, margin_meta = evidence.get_margin(
        margin_date, observed_on=observed_on, max_pages=3
    )
    results["margin-summary"] = (
        [dict(row) for row in margin_rows], dict(margin_meta),
        validate_risk_rows(margin_rows, feed="margin-summary", source_period=margin_date, window_end_date=margin_date, observed_on=observed_on),
    )

    lendable_rows, lendable_meta = evidence.get_lendable(observed_on)
    results["lendable-stock"] = (
        [dict(row) for row in lendable_rows], dict(lendable_meta),
        validate_risk_rows(lendable_rows, feed="lendable-stock", source_period=observed_on, window_end_date=observed_on, observed_on=observed_on),
    )

    summary: dict[str, Any] = {
        "client_id": client_id,
        "mode": args.mode,
        "observed_on": observed_on.isoformat(),
        "notice_from": notice_from.isoformat(),
        "notice_to": notice_to.isoformat(),
        "margin_date": margin_date.isoformat(),
        "feeds": {},
        "total_api_calls": 0,
        "forbidden_shared_semantics": 0,
        "factual_only": True,
    }
    forbidden: set[str] = set()
    for feed, (rows, meta, validation) in results.items():
        if not rows:
            raise SystemExit(f"{feed}:RISK_ROWS_EMPTY:{meta.get('state')}")
        if validation != (True, "VALID"):
            raise SystemExit(f"{feed}:RISK_VALIDATION_FAILED:{validation[1]}")
        _verify_state(args.mode, feed, meta)
        feed_forbidden = _scan_forbidden(rows)
        forbidden.update(feed_forbidden)
        summary["total_api_calls"] += int(meta.get("api_calls") or 0)
        summary["feeds"][feed] = {
            "state": str(meta.get("state") or ""),
            "rows": len(rows),
            "api_calls": int(meta.get("api_calls") or 0),
            "pages": int(meta.get("pages") or 0),
            "attachment_calls": int(meta.get("attachment_calls") or 0),
            "cache_hit": bool(meta.get("cache_hit")),
            "request_avoided": bool(meta.get("request_avoided")),
            "lease_state": str(meta.get("lease_state") or ""),
            "factual_hash": _fact_hash(rows),
        }
        if int(meta.get("attachment_calls") or 0) != 0:
            raise SystemExit(f"{feed}:ATTACHMENT_CALL_NOT_ALLOWED")

    if forbidden:
        raise SystemExit(f"FORBIDDEN_SHARED_SEMANTICS:{','.join(sorted(forbidden))}")
    summary["forbidden_shared_semantics"] = len(forbidden)
    if args.mode == "consumer" and summary["total_api_calls"] != 0:
        raise SystemExit("CONSUMER_TOTAL_NETWORK_BUDGET_VIOLATION")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
