from __future__ import annotations

"""Bounded Phase 5.6 live proof for company/securities/reference facts."""

import argparse
from datetime import date
import hashlib
import json
from typing import Any, Mapping

from shared_company_evidence import (
    REFERENCE_SETS,
    SharedCompanyEvidence,
    validate_company_rows,
    validate_reference_rows,
)


FORBIDDEN_KEYS = frozenset({
    "score", "rank", "ranking", "recommendation", "watchlist",
    "execution_ready", "real_money_ready", "entry", "stop_loss",
    "take_profit", "rr", "signal", "production_gate", "scanner_score",
    "bandar", "beneficial_owner",
})


def _forbidden(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).strip().lower()
            if lowered in FORBIDDEN_KEYS:
                found.add(lowered)
            found.update(_forbidden(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_forbidden(child))
    return found


def _hash(rows: list[Mapping[str, Any]], fields: tuple[str, ...], identity: tuple[str, ...]) -> str:
    canonical = [
        {field: row.get(field) for field in fields}
        for row in sorted(
            (dict(row) for row in rows),
            key=lambda row: tuple(str(row.get(field) or "") for field in identity),
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
    return parser.parse_args()


def _state(mode: str, family: str, meta: Mapping[str, Any]) -> None:
    state = str(meta.get("state") or "")
    calls = int(meta.get("api_calls") or 0)
    avoided = bool(meta.get("request_avoided"))
    if mode == "consumer":
        if state not in {"CACHE_HIT", "CACHE_FILLED_BY_OTHER_CLIENT"}:
            raise SystemExit(f"{family}:CONSUMER_DID_NOT_REUSE_CACHE:{state}")
        if calls != 0 or not avoided:
            raise SystemExit(f"{family}:CONSUMER_NETWORK_BUDGET_VIOLATION:{calls}")
    else:
        if state == "REFRESHED":
            if calls < 1 or avoided:
                raise SystemExit(f"{family}:PRODUCER_REFRESH_AUDIT_INVALID")
        elif state in {"CACHE_HIT", "CACHE_FILLED_BY_OTHER_CLIENT"}:
            if calls != 0 or not avoided:
                raise SystemExit(f"{family}:PRODUCER_CACHE_AUDIT_INVALID")
        else:
            raise SystemExit(f"{family}:PRODUCER_STATE_INVALID:{state}")


def main() -> int:
    args = _args()
    client_id = str(args.client_id).strip().upper()
    if client_id not in {"PASTICUAN", "EMIR"}:
        raise SystemExit("CLIENT_ID_NOT_ALLOWED")
    try:
        observed_on = date.fromisoformat(str(args.observed_on).strip())
    except ValueError as exc:
        raise SystemExit("OBSERVED_ON_INVALID") from exc

    evidence = SharedCompanyEvidence(client_id, api_key=None if args.mode == "producer" else "")

    directory, directory_meta = evidence.get_directory(observed_on, max_pages=3)
    directory = [dict(row) for row in directory]
    if not directory:
        raise SystemExit(f"DIRECTORY_ROWS_EMPTY:{directory_meta.get('state')}")
    if validate_company_rows(directory, provider="IDX_COMPANY_DIRECTORY_VIA_ZAPI") != (True, "VALID"):
        raise SystemExit("DIRECTORY_VALIDATION_FAILED")
    _state(args.mode, "directory", directory_meta)

    references: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    for set_name in sorted(REFERENCE_SETS):
        rows, meta = evidence.get_reference(set_name, observed_on)
        rows = [dict(row) for row in rows]
        if not rows:
            raise SystemExit(f"REFERENCE_ROWS_EMPTY:{set_name}:{meta.get('state')}")
        if validate_reference_rows(rows, set_name=set_name) != (True, "VALID"):
            raise SystemExit(f"REFERENCE_VALIDATION_FAILED:{set_name}")
        _state(args.mode, f"reference:{set_name}", meta)
        references[set_name] = (rows, dict(meta))

    forbidden = _forbidden(directory)
    for rows, _ in references.values():
        forbidden.update(_forbidden(rows))
    if forbidden:
        raise SystemExit(f"FORBIDDEN_SHARED_SEMANTICS:{','.join(sorted(forbidden))}")

    directory_fields = (
        "provider", "ticker", "company_name", "sector", "sub_sector", "industry",
        "sub_industry", "listing_board", "listing_date", "listed_shares", "main_business",
        "profile", "profile_kind", "source_period", "observed_on", "change_state",
        "source_url", "payload_hash", "source_verified", "validation_state",
    )
    reference_fields = (
        "provider", "set_name", "value_key", "label", "source_period", "observed_on",
        "source_url", "payload_hash", "source_verified", "validation_state",
    )
    summary: dict[str, Any] = {
        "client_id": client_id,
        "mode": args.mode,
        "observed_on": observed_on.isoformat(),
        "directory": {
            "state": str(directory_meta.get("state") or ""),
            "rows": len(directory),
            "api_calls": int(directory_meta.get("api_calls") or 0),
            "pages": int(directory_meta.get("pages") or 0),
            "cache_hit": bool(directory_meta.get("cache_hit")),
            "request_avoided": bool(directory_meta.get("request_avoided")),
            "factual_hash": _hash(directory, directory_fields, ("ticker",)),
        },
        "references": {},
        "total_api_calls": int(directory_meta.get("api_calls") or 0),
        "forbidden_shared_semantics": 0,
        "factual_only": True,
    }
    for set_name, (rows, meta) in sorted(references.items()):
        summary["references"][set_name] = {
            "state": str(meta.get("state") or ""),
            "rows": len(rows),
            "api_calls": int(meta.get("api_calls") or 0),
            "cache_hit": bool(meta.get("cache_hit")),
            "request_avoided": bool(meta.get("request_avoided")),
            "factual_hash": _hash(rows, reference_fields, ("set_name", "value_key")),
        }
        summary["total_api_calls"] += int(meta.get("api_calls") or 0)

    if args.mode == "consumer" and summary["total_api_calls"] != 0:
        raise SystemExit("CONSUMER_TOTAL_NETWORK_BUDGET_VIOLATION")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
