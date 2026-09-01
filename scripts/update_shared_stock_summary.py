from __future__ import annotations

"""Refresh one completed IDX session in the shared stock-summary evidence hub."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from idx_trading_calendar import latest_expected_completed_session
from shared_stock_summary_evidence import SharedStockSummaryEvidence


ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_PATH = ROOT / "tools" / "idx_400_production_universe_2026-07.txt"
CLIENT_ID = "PASTICUAN"


def load_universe(path: Path = UNIVERSE_PATH) -> list[str]:
    values: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("#"):
            continue
        values.extend(
            part.strip().upper().removesuffix(".JK")
            for part in line.split(",")
            if part.strip()
        )
    return sorted(set(values))


def main() -> int:
    trade_date = latest_expected_completed_session().date()
    universe = load_universe()
    producer = SharedStockSummaryEvidence(CLIENT_ID)
    _, meta = producer.get_day(trade_date, universe)
    safe_summary = {
        "client_id": CLIENT_ID,
        "trade_date": trade_date.isoformat(),
        "universe_size": len(universe),
        "state": meta.get("state"),
        "rows": meta.get("rows", 0),
        "ticker_breadth": meta.get("ticker_breadth", 0),
        "http_calls": meta.get("http_calls", 0),
        "request_avoided": bool(meta.get("request_avoided")),
        "hub_state": meta.get("hub_state"),
        "zapi_key_state": meta.get("zapi_key_state"),
        "lease_state": meta.get("lease_state"),
    }
    if "quota_remaining" in meta:
        safe_summary["quota_remaining"] = meta["quota_remaining"]
    print(json.dumps(safe_summary, sort_keys=True))
    return 0 if meta.get("state") in {"CACHE_HIT", "CACHE_FILLED_BY_OTHER_CLIENT", "REFRESHED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
