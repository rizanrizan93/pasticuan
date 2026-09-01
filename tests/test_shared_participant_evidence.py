from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import threading
from typing import Any, Mapping

import pandas as pd
import pytest

from shared_evidence_hub import SharedEvidenceCoordinator
from shared_participant_evidence import PROVENANCE, SOURCE, SharedParticipantEvidence


DAY = date(2026, 8, 31)


class Backend:
    def __init__(self):
        self.rows: list[dict[str, Any]] = []
        self.lease: dict[str, Any] | None = None
        self.lock = threading.Lock()

    def read_rows(self, table: str, filters: Mapping[str, Any], *, limit: int):
        assert table == "evidence_participant_flow"
        return [dict(row) for row in self.rows if row["source"] == filters["source"] and row["trade_date"] == filters["trade_date"]]

    def upsert_rows(self, table: str, rows, *, conflict):
        assert tuple(conflict) == ("source", "trade_date", "ticker", "broker_code")
        by_key = {(row["source"], row["trade_date"], row["ticker"], row["broker_code"]): row for row in self.rows}
        for row in rows:
            by_key[(row["source"], row["trade_date"], row["ticker"], row["broker_code"])] = dict(row)
        self.rows = list(by_key.values())
        return [dict(row) for row in rows]

    def acquire_lease(self, key, holder: str, lease_seconds: int):
        with self.lock:
            now = datetime.now(timezone.utc)
            if self.lease and self.lease["state"] == "HELD" and self.lease["expires"] > now and self.lease["holder"] != holder:
                return {"acquired": False}
            self.lease = {"state": "HELD", "holder": holder, "expires": now + timedelta(seconds=lease_seconds)}
            return {"acquired": True}

    def complete_lease(self, key, holder: str, state: str):
        self.lease["state"] = "COMPLETED"
        return True

    def fail_lease(self, key, holder: str, reason: str):
        self.lease = {"state": "FAILED", "holder": holder, "reason": reason}
        return True

    def record_provider_state(self, row):
        pass


def _client(backend: Backend, client_id: str) -> SharedParticipantEvidence:
    client = SharedParticipantEvidence.__new__(SharedParticipantEvidence)
    client.client_id = client_id
    client.config = None
    client.backend = backend
    client.coordinator = SharedEvidenceCoordinator(backend, client_id=client_id, worker_id=f"{client_id}-worker")
    return client


def _aggregate(path, trade_date):
    return pd.DataFrame([
        {"ticker": "BBCA", "broker_code": "YP", "buy_value": 1000, "sell_value": 200, "buy_volume": 10, "sell_volume": 2, "net_value": 800, "net_volume": 8, "buy_avg": 100, "sell_avg": 100},
        {"ticker": "BBRI", "broker_code": "CC", "buy_value": 500, "sell_value": 700, "buy_volume": 5, "sell_volume": 7, "net_value": -200, "net_volume": -2, "buy_avg": 100, "sell_avg": 100},
    ])


@pytest.mark.parametrize("first,second", [("EMIR", "PASTICUAN"), ("PASTICUAN", "EMIR")])
def test_one_official_file_serves_both_scanners(tmp_path: Path, first: str, second: str) -> None:
    backend = Backend()
    calls = 0

    def download(trade_date, diagnostics):
        nonlocal calls
        calls += 1
        path = tmp_path / f"trade-{calls}.csv"
        path.write_bytes(b"official factual fixture")
        return path, "https://www.idxdata3.co.id/official/trade.csv"

    first_frame, first_meta = _client(backend, first).get_day(
        DAY, download=download, aggregate=_aggregate, minimum_ticker_breadth=2
    )
    second_frame, second_meta = _client(backend, second).get_day(
        DAY,
        download=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("duplicate IDX download")),
        aggregate=_aggregate,
        minimum_ticker_breadth=2,
    )
    assert calls == 1
    assert first_meta["provider_called"]
    assert second_meta["request_avoided"]
    assert len(first_frame) == len(second_frame) == 2
    assert set(first_frame["ticker"]) == {"BBCA", "BBRI"}
    assert all(row["source"] == SOURCE for row in backend.rows)
    assert all(row["provenance_state"] == PROVENANCE for row in backend.rows)
    assert "BENEFICIAL_OWNER" not in backend.rows[0]
    assert "BROKER_DIRECT" not in str(backend.rows)


def test_parse_failure_is_not_collapsed_to_url_not_found(tmp_path: Path) -> None:
    backend = Backend()
    path = tmp_path / "trade.csv"
    path.write_bytes(b"malformed")
    _, meta = _client(backend, "EMIR").get_day(
        DAY,
        download=lambda *args, **kwargs: (path, "https://www.idxdata3.co.id/official/trade.csv"),
        aggregate=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("IDX_TRADE_DETAIL_PARSE_FAILED")),
        minimum_ticker_breadth=1,
    )
    assert meta["state"] == "PARSE_FAILURE"
    assert meta["provider_called"]
