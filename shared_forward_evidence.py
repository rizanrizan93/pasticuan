from __future__ import annotations

"""Scanner-neutral forward factual evidence contract for EMIR and PASTICUAN.

Only factual identity, provenance, point-in-time validity and transport live
here. Scanner scores, rankings, gates and recommendations are forbidden.
"""

from hashlib import sha256
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse
import math
import re

import pandas as pd

from shared_evidence_hub import HubConfig, SupabaseEvidenceBackend

CANONICAL_FORWARD_CONTRACT_VERSION = "v1"
CANONICAL_FORWARD_TABLE = "evidence_forward_events"
MAX_ACTIVE_AGE_DAYS = 540
FORBIDDEN_DECISION_TOKENS = (
    "score", "rank", "recommendation", "authorization", "execution_ready",
    "decision_state", "stop_loss", "take_profit", "target_price",
)


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def normalize_ticker(value: Any) -> str:
    text = _text(value).upper().replace(" ", "")
    return text if text.endswith(".JK") else f"{text}.JK" if text else ""


def _truthy(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else _text(value).lower() in {
        "1", "true", "yes", "y", "verified", "pass", "valid", "on"
    }


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _timestamp(value: Any) -> pd.Timestamp:
    stamp = pd.to_datetime(value, errors="coerce", utc=True)
    return pd.Timestamp(stamp) if pd.notna(stamp) else pd.NaT


def _date_text(value: Any) -> str:
    stamp = _timestamp(value)
    return stamp.date().isoformat() if pd.notna(stamp) else ""


def _https(value: Any) -> bool:
    try:
        parsed = urlparse(_text(value))
    except ValueError:
        return False
    return parsed.scheme.lower() == "https" and bool(parsed.hostname)


def _unique(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = _text(value)
        if text and text not in out:
            out.append(text)
    return out


def _split(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return _unique(value.values())
    if isinstance(value, (list, tuple, set)):
        return _unique(value)
    return _unique(part.strip() for part in re.split(r"[|,\n]+", _text(value)))


def _urls(value: Any) -> list[str]:
    candidates = list(value) if isinstance(value, (list, tuple, set)) else re.split(r"[|\n]+", _text(value))
    return _unique(item for item in candidates if _https(item))


def _kind(value: Any) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", _text(value).upper()).strip("_") or "FORWARD_EVENT"


def canonical_category(evidence_type: Any, title: Any = "") -> str:
    text = f" {_text(evidence_type)} {_text(title)} ".upper()
    if any(x in text for x in ("CANCEL", "TERMINAT", "DELAY", "POSTPON", "GUIDANCE CUT", "GUIDANCE_CUT")):
        return "ADVERSE_FORWARD"
    if any(x in text for x in ("BACKLOG", "CONTRACT", "KONTRAK", "OFFTAKE", "ORDER_VISIBILITY", "ORDER VISIBILITY")):
        return "CONTRACT_BACKLOG"
    if any(x in text for x in ("CAPEX", "EXPANSION", "EKSPANSI", "CAPACITY", "KAPASITAS", "COMMISSIONING")):
        return "CAPEX_EXPANSION"
    if any(x in text for x in ("GUIDANCE", "TARGET")):
        return "GUIDANCE"
    if any(x in text for x in ("PRODUCT", "LAUNCH", "PELUNCURAN", "NEW_MARKET", "NEW MARKET")):
        return "PRODUCT_LAUNCH"
    if any(x in text for x in ("JOINT_VENTURE", "JOINT VENTURE", " JV ", "ACQUISITION", "AKUISISI", "MERGER")):
        return "JV_MA"
    return "OTHER_FORWARD"


def canonical_event_id(*, ticker: Any, event_category: Any, evidence_type: Any = None, primary_source_url: Any) -> str:
    """Stable across scanner labels and corroboration-date drift.

    The neutral category plus primary source URL identify the event. The raw
    scanner-specific evidence_type and event date are deliberately excluded.
    This makes TSPC's issuer date (2026-07-15) and later BPOM corroboration date
    (2026-07-24) converge to one factual event rather than two records.
    """
    identity = "|".join((
        normalize_ticker(ticker), _text(event_category).upper(), _text(primary_source_url)
    ))
    return sha256(identity.encode("utf-8")).hexdigest()


def _official_family(value: Any) -> bool:
    text = _text(value).upper()
    return any(x in text for x in ("ISSUER", "IDX", "REGULATOR", "BPOM", "OJK", "KSEI", "GOVERNMENT"))


def factual_completeness_pct(row: Mapping[str, Any]) -> float:
    points = 0.0
    points += 20 if normalize_ticker(row.get("ticker")) else 0
    points += 10 if _text(row.get("event_category")) and _text(row.get("evidence_type")) else 0
    points += 10 if _date_text(row.get("evidence_date")) and _text(row.get("title")) else 0
    points += 20 if _https(row.get("primary_source_url")) else 0
    points += 15 if _truthy(row.get("source_verified")) else 0
    points += 10 if _truthy(row.get("entity_match_verified")) else 0
    points += 10 if _truthy(row.get("source_quorum_verified")) and _finite(row.get("source_quorum_count"), 0) >= 2 else 0
    points += 5 if _split(row.get("source_families")) else 0
    return round(float(points), 1)


def strict_active(row: Mapping[str, Any], *, as_of: Any = None, max_age_days: int = MAX_ACTIVE_AGE_DAYS) -> bool:
    if not (_truthy(row.get("source_verified")) and _truthy(row.get("entity_match_verified"))):
        return False
    if not _truthy(row.get("source_quorum_verified")) or _finite(row.get("source_quorum_count"), 0) < 2:
        return False
    if not _https(row.get("primary_source_url") or row.get("source_url")):
        return False
    stamp = _timestamp(row.get("evidence_date") or row.get("observed_at"))
    if pd.isna(stamp):
        return False
    now = pd.Timestamp.now(tz="UTC") if as_of is None else pd.Timestamp(as_of)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    age = (now - stamp).total_seconds() / 86400.0
    return -1.0 <= age <= float(max_age_days)


def _base_row(*, ticker: Any, evidence_type: Any, evidence_date: Any, title: Any,
              primary_source_url: Any, corroboration_urls: Iterable[Any], source_families: Iterable[Any],
              source_quorum_count: Any, source_quorum_verified: Any, entity_match_verified: Any,
              source_verified: Any, producer_client: str, producer_record_id: Any,
              observed_at: Any = None, evidence_confidence: Any = None, value_numeric: Any = None,
              unit: Any = None, horizon: Any = None, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    symbol = normalize_ticker(ticker)
    kind = _kind(evidence_type)
    category = canonical_category(kind, title)
    primary = _text(primary_source_url)
    confidence = _finite(evidence_confidence)
    numeric = _finite(value_numeric)
    observed = _timestamp(observed_at)
    client = _text(producer_client).upper()
    row = {
        "ticker": symbol,
        "event_category": category,
        "evidence_type": kind,
        "evidence_date": _date_text(evidence_date),
        "observed_at": observed.isoformat() if pd.notna(observed) else None,
        "title": _text(title) or kind.replace("_", " ").title(),
        "value_numeric": numeric if math.isfinite(numeric) else None,
        "unit": _text(unit) or None,
        "horizon": _text(horizon) or None,
        "primary_source_url": primary,
        "corroboration_urls": _unique(url for url in corroboration_urls if _https(url) and _text(url) != primary),
        "source_families": _unique(source_families),
        "source_quorum_count": max(0, int(_finite(source_quorum_count, 0))),
        "source_quorum_verified": _truthy(source_quorum_verified),
        "entity_match_verified": _truthy(entity_match_verified),
        "source_verified": _truthy(source_verified),
        "evidence_confidence": confidence if math.isfinite(confidence) else None,
        "payload": dict(payload or {}),
        "producer_clients": [client],
        "producer_records": {client: _text(producer_record_id)},
        "evidence_tier": "DIRECT_VERIFIED" if _truthy(source_verified) and _truthy(source_quorum_verified) else "RESEARCH_ONLY",
        "validation_state": "VALID",
        "contract_version": CANONICAL_FORWARD_CONTRACT_VERSION,
    }
    row["canonical_event_id"] = canonical_event_id(
        ticker=symbol, event_category=category, evidence_type=kind, primary_source_url=primary
    )
    return row


def canonicalize_emir_row(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(row.get("payload") or {}) if isinstance(row.get("payload"), Mapping) else {}
    return _base_row(
        ticker=row.get("ticker"), evidence_type=row.get("evidence_type"),
        evidence_date=row.get("evidence_date") or row.get("observed_at"), observed_at=row.get("observed_at"),
        title=row.get("title"), primary_source_url=row.get("source_url"),
        corroboration_urls=_urls(payload.get("secondary_url")) + _urls(payload.get("corroboration_urls")),
        source_families=_split(row.get("source_family")), source_quorum_count=row.get("source_quorum_count"),
        source_quorum_verified=row.get("source_quorum_verified"), entity_match_verified=row.get("entity_match_verified"),
        source_verified=row.get("source_verified"), evidence_confidence=row.get("evidence_confidence"),
        value_numeric=row.get("value_numeric"), unit=row.get("unit"), horizon=row.get("horizon"), payload=payload,
        producer_client="EMIR", producer_record_id=row.get("evidence_id"),
    )


def canonicalize_pasticuan_row(row: Mapping[str, Any]) -> dict[str, Any]:
    urls = _urls(row.get("project_source_urls") or row.get("source_url"))
    primary = urls[0] if urls else _text(row.get("source_url"))
    payload = {key: row.get(key) for key in (
        "project_completion_pct", "project_funding_secured_pct", "project_ownership_pct", "project_capex_idr",
        "project_expected_revenue_idr", "project_expected_ebitda_idr", "project_stage", "project_execution_flags",
        "review_origin",
    ) if row.get(key) not in (None, "")}
    return _base_row(
        ticker=row.get("ticker"), evidence_type=row.get("project_stage") or row.get("project_name"),
        evidence_date=row.get("evidence_date") or row.get("event_date") or row.get("last_verified_at"),
        observed_at=row.get("as_of") or row.get("last_verified_at"), title=row.get("project_name") or row.get("project_names"),
        primary_source_url=primary, corroboration_urls=urls[1:], source_families=_split(row.get("project_source_families")),
        source_quorum_count=row.get("source_quorum_count"), source_quorum_verified=row.get("project_source_quorum_verified"),
        entity_match_verified=row.get("entity_match_verified"),
        source_verified=True if _truthy(row.get("project_source_quorum_verified")) else row.get("source_verified"),
        evidence_confidence=row.get("evidence_confidence"), payload=payload,
        producer_client="PASTICUAN", producer_record_id=row.get("snapshot_id") or row.get("scan_id"),
    )


def merge_equivalent_rows(*rows: Mapping[str, Any]) -> dict[str, Any]:
    records = [dict(row) for row in rows if row]
    if not records:
        return {}
    if len({_text(row.get("canonical_event_id")) for row in records}) != 1:
        raise ValueError("CANONICAL_EVENT_ID_MISMATCH")
    out = dict(records[0])
    dates = [_date_text(row.get("evidence_date")) for row in records if _date_text(row.get("evidence_date"))]
    if dates:
        out["evidence_date"] = min(dates)
    primary = _text(out.get("primary_source_url"))
    all_urls = _unique(
        url for row in records for url in [row.get("primary_source_url"), *_urls(row.get("corroboration_urls"))] if _https(url)
    )
    out["corroboration_urls"] = [url for url in all_urls if url != primary]
    out["source_families"] = _unique(family for row in records for family in _split(row.get("source_families")))
    out["producer_clients"] = _unique(client.upper() for row in records for client in _split(row.get("producer_clients")))
    producers: dict[str, Any] = {}
    payload: dict[str, Any] = {}
    for row in records:
        if isinstance(row.get("producer_records"), Mapping):
            producers.update(dict(row.get("producer_records") or {}))
        if isinstance(row.get("payload"), Mapping):
            payload.update(dict(row.get("payload") or {}))
    out["producer_records"], out["payload"] = producers, payload
    out["source_quorum_count"] = max(int(_finite(row.get("source_quorum_count"), 0)) for row in records)
    out["source_quorum_verified"] = any(_truthy(row.get("source_quorum_verified")) for row in records)
    out["entity_match_verified"] = any(_truthy(row.get("entity_match_verified")) for row in records)
    out["source_verified"] = any(_truthy(row.get("source_verified")) for row in records)
    confidence = [_finite(row.get("evidence_confidence")) for row in records]
    confidence = [value for value in confidence if math.isfinite(value)]
    out["evidence_confidence"] = max(confidence) if confidence else None
    return out


def profile_rows(rows: Iterable[Mapping[str, Any]], *, ticker: Any = None, as_of: Any = None) -> dict[str, Any]:
    symbol = normalize_ticker(ticker) if ticker else ""
    selected = [dict(row) for row in rows if not symbol or normalize_ticker(row.get("ticker")) == symbol]
    active = [row for row in selected if strict_active(row, as_of=as_of)]
    families = _unique(family for row in active for family in _split(row.get("source_families")))
    dates = [_date_text(row.get("evidence_date")) for row in active if _date_text(row.get("evidence_date"))]
    completeness = [factual_completeness_pct(row) for row in active]
    return {
        "ticker": symbol,
        "shared_forward_event_count": len(selected),
        "shared_forward_active_direct_count": len(active),
        "shared_forward_verified_count": sum(int(_truthy(row.get("source_verified"))) for row in active),
        "shared_forward_official_count": sum(int(any(_official_family(x) for x in _split(row.get("source_families")))) for row in active),
        "shared_forward_source_quorum_verified": bool(active and all(
            _truthy(row.get("source_quorum_verified")) and _finite(row.get("source_quorum_count"), 0) >= 2 for row in active
        )),
        "shared_forward_contract_coverage_pct": round(sum(completeness) / len(completeness), 1) if completeness else 0.0,
        "shared_forward_latest_evidence_date": max(dates) if dates else None,
        "shared_forward_source_families": families,
        "shared_forward_provenance_state": "DIRECT_VERIFIED_ACTIVE" if active else ("HISTORICAL_ONLY_OR_STALE" if selected else "MISSING"),
        "shared_forward_contract_version": CANONICAL_FORWARD_CONTRACT_VERSION,
    }


def read_canonical_forward_rows(tickers: Sequence[Any] | None = None, *, client_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = HubConfig.from_environment(client_id=client_id)
    if not config.ready:
        return [], {"state": "SHARED_HUB_UNAVAILABLE", **config.status()}
    try:
        rows = SupabaseEvidenceBackend(config).read_rows(CANONICAL_FORWARD_TABLE, {}, limit=50000)
    except Exception as exc:
        return [], {"state": "READ_FAIL_SOFT", "error": f"{type(exc).__name__}: {str(exc)[:240]}"}
    wanted = {normalize_ticker(value) for value in (tickers or []) if normalize_ticker(value)}
    if wanted:
        rows = [row for row in rows if normalize_ticker(row.get("ticker")) in wanted]
    return rows, {"state": "SHARED_CANONICAL_FORWARD", "rows": len(rows), "contract_version": CANONICAL_FORWARD_CONTRACT_VERSION}


def upsert_canonical_forward_rows(rows: Iterable[Mapping[str, Any]], *, client_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = [dict(row) for row in rows]
    if not records:
        return [], {"state": "NO_ITEMS", "rows": 0}
    for row in records:
        forbidden = [key for key in row if any(token in key.lower() for token in FORBIDDEN_DECISION_TOKENS)]
        if forbidden:
            raise ValueError(f"SCANNER_DECISION_FIELDS_FORBIDDEN:{','.join(sorted(forbidden))}")
    config = HubConfig.from_environment(client_id=client_id)
    if not config.ready:
        return [], {"state": "SHARED_HUB_UNAVAILABLE", **config.status()}
    written = SupabaseEvidenceBackend(config).upsert_rows(
        CANONICAL_FORWARD_TABLE, records, conflict=("canonical_event_id",)
    )
    return written, {"state": "UPSERTED", "rows": len(written), "contract_version": CANONICAL_FORWARD_CONTRACT_VERSION}


def canonical_rows_to_pasticuan_projects(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row.get("payload") or {}) if isinstance(row.get("payload"), Mapping) else {}
        urls = _unique([row.get("primary_source_url"), *_urls(row.get("corroboration_urls"))])
        output.append({
            "ticker": normalize_ticker(row.get("ticker")), "project_name": _text(row.get("title")),
            "project_names": _text(row.get("title")), "project_stage": _text(row.get("evidence_type")),
            "project_completion_pct": payload.get("project_completion_pct"),
            "project_funding_secured_pct": payload.get("project_funding_secured_pct"),
            "project_ownership_pct": payload.get("project_ownership_pct"),
            "project_capex_idr": payload.get("project_capex_idr"),
            "project_expected_revenue_idr": payload.get("project_expected_revenue_idr", payload.get("estimated_total_revenue_idr")),
            "project_expected_ebitda_idr": payload.get("project_expected_ebitda_idr"),
            "project_data_coverage": factual_completeness_pct(row),
            "project_source_families": "|".join(_split(row.get("source_families"))),
            "project_source_urls": "|".join(urls),
            "project_source_quorum_verified": _truthy(row.get("source_quorum_verified")),
            "source_quorum_count": int(_finite(row.get("source_quorum_count"), 0)),
            "entity_match_verified": _truthy(row.get("entity_match_verified")),
            "last_verified_at": _date_text(row.get("evidence_date")), "evidence_date": _date_text(row.get("evidence_date")),
            "event_date": _date_text(row.get("evidence_date")), "review_origin": "SHARED_CANONICAL_FORWARD_EVIDENCE",
            "canonical_event_id": _text(row.get("canonical_event_id")),
            "shared_forward_contract_version": CANONICAL_FORWARD_CONTRACT_VERSION,
        })
    return pd.DataFrame(output)


def canonical_rows_to_emir_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        payload = dict(row.get("payload") or {}) if isinstance(row.get("payload"), Mapping) else {}
        corroboration = _urls(row.get("corroboration_urls"))
        if corroboration and not payload.get("secondary_url"):
            payload["secondary_url"] = corroboration[0]
        output.append({
            "evidence_id": _text(row.get("canonical_event_id")), "ticker": normalize_ticker(row.get("ticker")),
            "evidence_type": _text(row.get("evidence_type")), "evidence_date": _date_text(row.get("evidence_date")),
            "observed_at": row.get("observed_at"), "title": _text(row.get("title")),
            "value_numeric": row.get("value_numeric"), "unit": row.get("unit"), "horizon": row.get("horizon"),
            "source_url": _text(row.get("primary_source_url")),
            "source_family": "|".join(_split(row.get("source_families"))),
            "source_quorum_count": int(_finite(row.get("source_quorum_count"), 0)),
            "source_quorum_verified": _truthy(row.get("source_quorum_verified")),
            "entity_match_verified": _truthy(row.get("entity_match_verified")),
            "source_verified": _truthy(row.get("source_verified")), "evidence_confidence": row.get("evidence_confidence"),
            "payload": payload, "canonical_event_id": _text(row.get("canonical_event_id")),
            "shared_forward_contract_version": CANONICAL_FORWARD_CONTRACT_VERSION,
        })
    return output


__all__ = [
    "CANONICAL_FORWARD_CONTRACT_VERSION", "CANONICAL_FORWARD_TABLE", "MAX_ACTIVE_AGE_DAYS",
    "canonical_category", "canonical_event_id", "canonicalize_emir_row", "canonicalize_pasticuan_row",
    "canonical_rows_to_emir_rows", "canonical_rows_to_pasticuan_projects", "factual_completeness_pct",
    "merge_equivalent_rows", "normalize_ticker", "profile_rows", "read_canonical_forward_rows",
    "strict_active", "upsert_canonical_forward_rows",
]
