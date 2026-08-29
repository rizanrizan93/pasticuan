from __future__ import annotations

import gzip
from io import BytesIO

import pandas as pd
import requests

from scanner import _load_canonical_remote_ohlcv_seed


class _Response:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        return None


def test_canonical_remote_seed_normalizes_shared_flow_cache(monkeypatch):
    csv = pd.DataFrame({
        "ticker": ["TEST"] * 3,
        "date": ["2026-08-26", "2026-08-27", "2026-08-28"],
        "open": [100.0, 101.0, 102.0],
        "high": [102.0, 103.0, 104.0],
        "low": [99.0, 100.0, 101.0],
        "close": [101.0, 102.0, 103.0],
        "volume": [1_000_000, 1_100_000, 1_200_000],
    }).to_csv(index=False).encode("utf-8")
    payload = gzip.compress(csv)

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Response(payload))
    out = _load_canonical_remote_ohlcv_seed(["TEST.JK"], timeout=1)

    assert "TEST.JK" in out
    frame = out["TEST.JK"]
    assert frame.index.max().strftime("%Y-%m-%d") == "2026-08-28"
    assert {"Open", "High", "Low", "Close", "Volume"}.issubset(frame.columns)
