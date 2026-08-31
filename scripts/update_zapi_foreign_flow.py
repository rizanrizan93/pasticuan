from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import gzip
import json
import os
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zapi_flow_enrichment import _fetch_direct_day, _normalize_history
from idx_trading_calendar import latest_expected_completed_session, previous_idx_session


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


def main() -> int:
    api_key = str(os.environ.get("ZAPI_KEY", "")).strip()
    if not api_key:
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
    existing_dates = set(pd.to_datetime(existing.get("trade_date"), errors="coerce").dropna().dt.normalize()) if not existing.empty else set()
    for cursor in _sessions(max_days):
        day_key = cursor.date().isoformat()
        if cursor in existing_dates or _negative_cache_active(state, cursor, now):
            continue
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
    print({"owner": OWNER, "state": "OWNED_CACHE_WRITTEN", "rows": len(combined), "sessions": len(dates), "attempted_days": attempted, "successful_days": successes})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
