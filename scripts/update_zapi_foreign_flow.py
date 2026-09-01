from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import gzip
import hashlib
import json
import os
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zapi_flow_enrichment import (
    ZAPI_FOREIGN_FLOW_URL,
    ZAPI_STOCK_SUMMARY_URL,
    _fetch_direct_day,
    _normalize_history,
)
from idx_trading_calendar import latest_expected_completed_session, previous_idx_session
from shared_evidence_hub import (
    EvidenceKey,
    EvidenceState,
    HubConfig,
    MissingReason,
    SharedEvidenceCoordinator,
    SupabaseEvidenceBackend,
)


CACHE_PATH = Path("data/zapi_foreign_flow_60d.csv.gz")
STATE_PATH = Path("data/zapi_foreign_flow_state.json")
UNIVERSE_PATH = Path("tools/idx_400_production_universe_2026-07.txt")
OWNER = "PASTICUAN"


def _universe() -> list[str]:
    values: list[str] = []
    for line in UNIVERSE_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("#"):
            continue
        values.extend(part.strip().upper().removesuffix(".JK") for part in line.split(",") if part.strip())
    return sorted(set(values))


def _existing() -> pd.DataFrame:
    try:
        with gzip.open(CACHE_PATH, "rb") as stream:
            return _normalize_history(pd.read_csv(stream))
    except (FileNotFoundError, OSError, ValueError):
        return _normalize_history(pd.DataFrame())


def _sessions(count: int) -> list[pd.Timestamp]:
    current = latest_expected_completed_session()
    values: list[pd.Timestamp] = []
    while len(values) < count:
        values.append(current)
        current = previous_idx_session(current, include_date=False)
    return values


def _load_state() -> dict:
    try:
        value = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, OSError, ValueError):
        return {}


def _negative_cache_active(state: dict, day: pd.Timestamp, now: datetime) -> bool:
    item = dict((state.get("sessions") or {}).get(day.date().isoformat()) or {})
    retry_at = pd.to_datetime(item.get("retry_after"), utc=True, errors="coerce")
    return str(item.get("state") or "") in {"NO_DATA", "FAIL_SOFT"} and pd.notna(retry_at) and retry_at.to_pydatetime() > now


def _write_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _json_value(value):
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def _to_shared_rows(frame: pd.DataFrame, fetched_at: datetime) -> list[dict]:
    normalized = _normalize_history(frame)
    if normalized.empty:
        return []
    hash_columns = [
        name for name in (
            "ticker", "trade_date", "foreign_buy_shares", "foreign_sell_shares",
            "foreign_net_shares", "volume", "value", "source",
        ) if name in normalized.columns
    ]
    canonical = normalized[hash_columns].copy().sort_values(["trade_date", "ticker"], kind="stable")
    canonical["trade_date"] = pd.to_datetime(canonical["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    payload_hash = hashlib.sha256(canonical.to_json(orient="records").encode("utf-8")).hexdigest()
    rows: list[dict] = []
    for _, row in normalized.iterrows():
        source_family = str(row.get("source") or "ZAPI_IDX_FOREIGN_FLOW")
        rows.append({
            "provider": "ZAPI",
            "trade_date": pd.Timestamp(row["trade_date"]).date().isoformat(),
            "ticker": str(row["ticker"]),
            "foreign_buy_shares": _json_value(row.get("foreign_buy_shares")),
            "foreign_sell_shares": _json_value(row.get("foreign_sell_shares")),
            "foreign_net_shares": _json_value(row.get("foreign_net_shares")),
            "volume": _json_value(row.get("volume")),
            "value": _json_value(row.get("value")),
            "flow_unit": str(row.get("flow_unit") or "SHARES"),
            "source_family": source_family,
            "source_url": ZAPI_STOCK_SUMMARY_URL if "STOCK_SUMMARY" in source_family else ZAPI_FOREIGN_FLOW_URL,
            "payload_hash": payload_hash,
            "fetched_at": fetched_at.isoformat(),
            "freshness_state": "CURRENT",
            "validation_state": "VALID",
        })
    return rows


def _from_shared_rows(rows: list[dict] | tuple[dict, ...]) -> pd.DataFrame:
    if not rows:
        return _normalize_history(pd.DataFrame())
    frame = pd.DataFrame(rows).rename(columns={"source_family": "source"})
    return _normalize_history(frame)


def _provider_reason(meta: dict) -> str:
    text = json.dumps(meta, sort_keys=True).upper()
    for reason in (
        "HTTP_401", "HTTP_403", "HTTP_404", "HTTP_429", "TIMEOUT",
        "CONNECTION_ERROR", "QUOTA_EXHAUSTED", "RATE_LIMIT",
    ):
        if reason in text:
            return reason
    state = str(meta.get("state") or "").upper()
    return MissingReason.PROVIDER_NO_DATA.value if state == "NO_DATA" else MissingReason.EMPTY_RESPONSE.value


def _shared_context():
    config = HubConfig.from_environment(client_id=OWNER)
    print({"owner": OWNER, "shared_evidence_hub": config.status()["state"]})
    if not config.ready:
        return None
    backend = SupabaseEvidenceBackend(config)
    return backend, SharedEvidenceCoordinator(backend, client_id=OWNER)


def _shared_session(context, universe: list[str], cursor: pd.Timestamp):
    backend, coordinator = context
    day = cursor.date().isoformat()
    meta_box: dict = {}

    def read_current():
        return backend.read_rows(
            "evidence_foreign_flow",
            {"provider": "ZAPI", "trade_date": day},
            limit=2000,
        )

    def fetch():
        frame, meta = _fetch_direct_day(universe, cursor, str(os.environ.get("ZAPI_KEY", "")).strip())
        meta_box.update(meta)
        return _to_shared_rows(frame, datetime.now(timezone.utc))

    def persist(rows):
        written = backend.upsert_rows(
            "evidence_foreign_flow",
            rows,
            conflict=("provider", "trade_date", "ticker"),
        )
        return len(written)

    def validate(rows):
        minimum = max(1, int(len(universe) * 0.80))
        return (len(rows) >= minimum, "VALID" if len(rows) >= minimum else _provider_reason(meta_box))

    result = coordinator.get_or_refresh(
        EvidenceKey("ZAPI", "FOREIGN_FLOW", "IDX_ALL", cursor.date()),
        read_current=read_current,
        fetch=fetch,
        persist=persist,
        validate=validate,
        minimum_rows=max(1, int(len(universe) * 0.80)),
        lease_seconds=300,
    )
    meta = {
        "state": result.reason,
        "provider": "SHARED_IDX_EVIDENCE_HUB",
        "rows": len(result.rows),
        "provider_called": result.provider_called,
        "request_avoided": result.request_avoided,
        "lease_state": result.lease_state,
    }
    return _from_shared_rows(list(result.rows)), meta


def main() -> int:
    api_key = str(os.environ.get("ZAPI_KEY", "")).strip()
    shared_context = _shared_context()
    if not api_key and not shared_context:
        raise RuntimeError("ZAPI_KEY_REQUIRED_FOR_OWNED_CACHE_REFRESH")
    universe = _universe()
    existing = _existing()
    max_days = max(1, min(10, int(os.environ.get("ZAPI_CACHE_REFRESH_DAYS", "5"))))
    state = _load_state()
    state.update({"owner": OWNER, "provider": "ZAPI", "updated_at": datetime.now(timezone.utc).isoformat()})
    state.setdefault("sessions", {})
    now = datetime.now(timezone.utc)
    attempted = 0
    successes = 0
    parts = [existing] if not existing.empty else []
    cache_hits = 0
    existing_dates = set(pd.to_datetime(existing.get("trade_date"), errors="coerce").dropna().dt.normalize()) if not existing.empty else set()
    for cursor in _sessions(max_days):
        day_key = cursor.date().isoformat()
        if cursor in existing_dates or _negative_cache_active(state, cursor, now):
            continue
        if shared_context:
            frame, meta = _shared_session(shared_context, universe, cursor)
            attempted += int(bool(meta.get("provider_called")))
            cache_hits += int(bool(meta.get("request_avoided")))
        else:
            attempted += 1
            frame, meta = _fetch_direct_day(universe, cursor, api_key)
        provider_state = str(meta.get("state") or "UNKNOWN")
        print({"owner": OWNER, "trade_date": day_key, "state": provider_state, "rows": len(frame)})
        retry_hours = 6 if provider_state == "NO_DATA" else 2
        state["sessions"][day_key] = {
            "state": provider_state,
            "attempted_at": now.isoformat(),
            "retry_after": (now + timedelta(hours=retry_hours)).isoformat(),
            "rows": int(len(frame)),
        }
        if not frame.empty:
            parts.append(_normalize_history(frame))
            successes += 1
        if "RATE_LIMIT" in str(meta):
            state["quota_state"] = "QUOTA_EXHAUSTED"
            break
    else:
        state["quota_state"] = "AVAILABLE_OR_NOT_REQUIRED"
    if shared_context:
        metrics = shared_context[1].metrics()
        state["shared_cache"] = metrics
    _write_state(state)
    if not parts:
        print({"owner": OWNER, "state": "NO_DATA_FAIL_CLOSED", "attempted_days": attempted})
        return 0
    combined = _normalize_history(pd.concat(parts, ignore_index=True, sort=False))
    dates = sorted(combined["trade_date"].dropna().unique())[-60:]
    combined = combined[combined["trade_date"].isin(dates)].copy()
    allowed_sources = {"ZAPI_IDX_FOREIGN_FLOW", "ZAPI_IDX_STOCK_SUMMARY_FALLBACK"}
    combined = combined[combined["source"].isin(allowed_sources)]
    if combined.empty:
        raise RuntimeError("OWNED_ZAPI_CACHE_HAS_NO_VALID_PROVIDER_ROWS")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(CACHE_PATH, "wb") as stream:
        combined.to_csv(stream, index=False)
    print({"owner": OWNER, "state": "OWNED_CACHE_WRITTEN", "rows": len(combined), "sessions": len(dates), "attempted_days": attempted, "successful_days": successes, "shared_cache_hits": cache_hits})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
