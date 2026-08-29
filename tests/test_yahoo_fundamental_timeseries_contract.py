from __future__ import annotations

import pandas as pd

import free_data_providers as fdp


def test_yahoo_timeseries_accepts_list_meta_type_and_does_not_pad_history(monkeypatch):
    captured_params = []

    payload = {
        "timeseries": {
            "result": [
                {
                    "meta": {"type": ["quarterlyTotalRevenue"]},
                    "quarterlyTotalRevenue": [{
                        "asOfDate": "2026-06-30",
                        "reportedValue": {"raw": 1_200_000_000, "currencyCode": "IDR"},
                    }],
                },
                {
                    "meta": {"type": ["quarterlyFreeCashFlow"]},
                    "quarterlyFreeCashFlow": [{
                        "asOfDate": "2026-06-30",
                        "reportedValue": {"raw": 150_000_000, "currencyCode": "IDR"},
                    }],
                },
                {
                    "meta": {"type": ["quarterlyCapitalExpenditure"]},
                    "quarterlyCapitalExpenditure": [{
                        "asOfDate": "2026-06-30",
                        "reportedValue": {"raw": -50_000_000, "currencyCode": "IDR"},
                    }],
                },
            ]
        }
    }

    def fake_request_json(client, url, *, params, timeout, retry_count, retry_backoff):
        captured_params.append(dict(params))
        return payload, 1

    monkeypatch.setattr(fdp, "_request_json", fake_request_json)

    frame, report = fdp.yahoo_fundamental_timeseries_direct("TEST.JK", retry_count=1)

    assert report["status"] == "OK"
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["revenue"] == 1_200_000_000
    # OCF = reported FCF + absolute capex when Yahoo capex is negative.
    assert row["operating_cash_flow"] == 200_000_000
    assert "OCF_RECONSTRUCTED_FROM_REPORTED_FCF_AND_CAPEX" in row["validation_flags"]
    assert captured_params
    assert all(params["padTimeSeries"] == "false" for params in captured_params)
    assert any("quarterlyFreeCashFlow" in params["type"] for params in captured_params)


def test_series_type_handles_scalar_and_list_contracts():
    assert fdp._series_type("quarterlyTotalRevenue") == "quarterlyTotalRevenue"
    assert fdp._series_type(["quarterlyTotalRevenue"]) == "quarterlyTotalRevenue"
    assert fdp._series_type([]) == ""
