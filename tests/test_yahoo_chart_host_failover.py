from __future__ import annotations

import requests

import free_data_providers as fdp


class _Response:
    def __init__(self, status_code: int, payload: dict | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._payload


class _Session:
    def __init__(self):
        self.urls = []

    def get(self, url, **kwargs):
        self.urls.append(url)
        if "query1.finance.yahoo.com" in url:
            return _Response(429)
        return _Response(200, {
            "chart": {
                "result": [{
                    "timestamp": [1700000000, 1700086400],
                    "meta": {"currency": "IDR", "instrumentType": "EQUITY"},
                    "indicators": {
                        "quote": [{
                            "open": [100.0, 101.0],
                            "high": [102.0, 103.0],
                            "low": [99.0, 100.0],
                            "close": [101.0, 102.0],
                            "volume": [1000000, 1200000],
                        }],
                        "adjclose": [{"adjclose": [101.0, 102.0]}],
                    },
                }],
                "error": None,
            }
        })


def test_chart_direct_fails_over_query1_to_query2_without_unbounded_retry():
    session = _Session()
    frame, meta = fdp.yahoo_chart_direct(
        "TEST.JK", period="1mo", session=session, retry_count=0, timeout=1
    )

    assert len(frame) == 2
    assert meta["status"] == "OK"
    assert meta["host"] == "query2.finance.yahoo.com"
    assert len(session.urls) == 2
    assert "query1.finance.yahoo.com" in session.urls[0]
    assert "query2.finance.yahoo.com" in session.urls[1]
