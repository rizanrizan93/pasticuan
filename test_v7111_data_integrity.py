from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import free_data_providers as providers
import scanner
from scanner_focus import (
    _qualified_multibagger_lane,
    _sort_multibagger_ranking_contract,
    allocate_multibagger_capital,
    build_multibagger_decision_summary,
)


class _Response:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise providers.requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload


class _Session:
    def __init__(self, responses: list[_Response]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def get(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.responses.pop(0)


def _series(name: str, value: float) -> dict:
    key = f"annual{name}"
    return {
        "meta": {"type": [key]},
        key: [{
            "asOfDate": "2025-12-31",
            "reportedValue": {"raw": value, "currencyCode": "IDR"},
        }],
    }


def test_timeseries_list_meta_is_parsed_without_padding_and_ocf_identity() -> None:
    payload = {"timeseries": {"result": [
        _series("TotalRevenue", 1_000.0),
        _series("NetIncome", 120.0),
        _series("FreeCashFlow", 80.0),
        _series("CapitalExpenditure", -20.0),
    ]}}
    session = _Session([_Response(payload)])

    frame, report = providers.yahoo_fundamental_timeseries_direct(
        "TEST.JK", session=session, retry_count=1,
    )

    assert report["status"] == "OK"
    assert report["rows"] == 1
    assert session.calls[0]["params"]["padTimeSeries"] == "false"
    row = frame.iloc[0]
    assert row["revenue"] == 1_000.0
    assert row["net_income"] == 120.0
    assert row["operating_cash_flow"] == 100.0
    assert "OCF_RECONSTRUCTED" in row["validation_flags"]
    assert "free_cash_flow" not in frame.columns


def test_direct_chart_retries_transient_502_then_succeeds() -> None:
    payload = {"chart": {"error": None, "result": [{
        "timestamp": [1_700_000_000, 1_700_086_400],
        "indicators": {
            "quote": [{
                "open": [100.0, 101.0], "high": [102.0, 103.0],
                "low": [99.0, 100.0], "close": [101.0, 102.0],
                "volume": [1_000.0, 1_100.0],
            }],
            "adjclose": [{"adjclose": [101.0, 102.0]}],
        },
        "events": {}, "meta": {"currency": "IDR"},
    }]}}
    session = _Session([_Response({}, 502), _Response(payload)])

    frame, report = providers.yahoo_chart_direct(
        "TEST.JK", session=session, retry_count=2, retry_backoff=0,
    )

    assert len(session.calls) == 2
    assert len(frame) == 2
    assert report["attempts"] == 2


def test_binary_ohlcv_cache_survives_missing_csv(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IDX_SCANNER_CACHE_DIR", str(tmp_path))
    index = pd.bdate_range("2024-01-02", periods=40)
    close = pd.Series(np.linspace(100.0, 120.0, len(index)), index=index)
    frame = pd.DataFrame({
        "Open": close * 0.99, "High": close * 1.01,
        "Low": close * 0.98, "Close": close,
        "Volume": np.full(len(index), 1_000_000.0),
    }, index=index)
    scanner._write_daily_ohlcv_cache(
        "TEST.JK", frame, "UNIT_TEST", now="2026-08-02 12:00:00+07:00",
    )
    csv_path = scanner._daily_ohlcv_cache_path("TEST.JK")
    binary_path = scanner._daily_ohlcv_cache_binary_path("TEST.JK")
    assert csv_path.is_file()
    assert binary_path.is_file()
    csv_path.unlink()

    restored = scanner._load_daily_ohlcv_cache("TEST.JK")

    assert len(restored) == len(frame)
    assert restored.index.equals(frame.index)


def test_ranking_contract_blocks_not_qualified_and_orders_score_within_gate() -> None:
    frame = pd.DataFrame([
        {
            "ticker": "BAD.JK", "multibagger_status": "MULTIBAGGER_NOT_QUALIFIED",
            "research_eligible": False, "multibagger_rank_eligible": True,
            "multibagger_proxy_rank_eligible": True,
            "multibagger_selection_score": 99.0,
            "effective_silent_accumulation_score": 90.0,
        },
        {
            "ticker": "DIRECT.JK", "multibagger_status": "MULTIBAGGER_B_CANDIDATE",
            "research_eligible": True, "multibagger_rank_eligible": True,
            "multibagger_proxy_rank_eligible": True,
            "multibagger_selection_score": 65.0,
            "effective_silent_accumulation_score": 55.0,
        },
        {
            "ticker": "PROXY1.JK", "multibagger_status": "MULTIBAGGER_B_CANDIDATE",
            "research_eligible": True, "multibagger_rank_eligible": False,
            "multibagger_proxy_rank_eligible": True,
            "multibagger_selection_score": 75.0,
            "effective_silent_accumulation_score": 50.0,
        },
        {
            "ticker": "PROXY2.JK", "multibagger_status": "MULTIBAGGER_B_CANDIDATE",
            "research_eligible": True, "multibagger_rank_eligible": False,
            "multibagger_proxy_rank_eligible": True,
            "multibagger_selection_score": 70.0,
            "effective_silent_accumulation_score": 80.0,
        },
    ])

    ranked = _sort_multibagger_ranking_contract(frame)

    assert ranked["ticker"].tolist() == [
        "DIRECT.JK", "PROXY1.JK", "PROXY2.JK", "BAD.JK",
    ]
    assert ranked.loc[ranked["ticker"].eq("DIRECT.JK"), "multibagger_production_rank"].iloc[0] == 1
    assert ranked.loc[ranked["ticker"].eq("BAD.JK"), "multibagger_production_rank"].isna().all()


def test_failed_turnaround_gate_cannot_replace_qualified_growth_lane() -> None:
    assert _qualified_multibagger_lane(
        "TURNAROUND_CYCLICAL",
        growth_research_eligible=True,
        turnaround_research_eligible=False,
    ) == "GROWTH_COMPOUNDER"
    assert _qualified_multibagger_lane(
        "GROWTH_COMPOUNDER",
        growth_research_eligible=False,
        turnaround_research_eligible=True,
    ) == "TURNAROUND_CYCLICAL"


def test_allocation_cannot_use_proxy_only_candidate() -> None:
    frame = pd.DataFrame([{
        "ticker": "PROXY.JK", "multibagger_status": "MULTIBAGGER_A_CANDIDATE",
        "multibagger_scoring_state": "SCORED", "multibagger_score": 92.0,
        "multibagger_quality_score": 92.0, "multibagger_rank_eligible": False,
        "emir_method_state": "DISABLED", "narrative_hard_block": False,
        "last_price": 1_000.0,
    }])

    allocated = allocate_multibagger_capital(frame, scanner.ScanConfig())

    assert not bool(allocated.iloc[0]["allocation_eligible"])
    assert float(allocated.iloc[0]["recommended_allocation_idr"]) == 0.0


def test_history_only_coverage_is_not_capped_by_missing_snapshot() -> None:
    rows = []
    for offset, period in enumerate(pd.date_range("2025-03-31", periods=5, freq="QE")):
        revenue = 1_000.0 + 100.0 * offset
        rows.append({
            "ticker": "HIST.JK", "period_end": period,
            "period_type": "Q", "statement_basis": "STANDALONE_QUARTER",
            "source_family": "YAHOO", "source_name": "unit",
            "source_url": "https://query2.finance.yahoo.com", "currency": "IDR",
            "revenue": revenue, "gross_profit": revenue * 0.4,
            "operating_income": revenue * 0.2, "ebit": revenue * 0.2,
            "ebitda": revenue * 0.24, "net_income": revenue * 0.14,
            "operating_cash_flow": revenue * 0.17, "capex": -revenue * 0.04,
            "total_assets": 5_000.0, "total_liabilities": 2_000.0,
            "equity": 3_000.0, "total_debt": 800.0, "cash": 700.0,
            "shares_outstanding": 1_000.0, "interest_expense": 20.0,
            "source_verified": False, "validation_flags": "",
        })
    enriched = scanner.enrich_fundamentals_with_history(
        pd.DataFrame({"ticker": ["HIST.JK"]}), pd.DataFrame(rows),
        now=pd.Timestamp("2026-08-02", tz="Asia/Jakarta"),
    )
    row = enriched.iloc[0]
    assert row["fundamental_coverage_snapshot"] == 0.0
    assert row["fundamental_coverage"] == row["fundamental_history_coverage"]
    assert row["fundamental_coverage"] > 55.0


def test_empty_fundamental_route_has_explicit_error_and_grade_d(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setenv("IDX_SCANNER_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("IDX_SCANNER_ENABLE_YFINANCE_FALLBACK", raising=False)
    monkeypatch.delenv("IDX_SCANNER_ENABLE_YAHOO_QUOTE_SUMMARY", raising=False)
    monkeypatch.setattr(
        providers, "yahoo_fundamental_timeseries_direct",
        lambda *args, **kwargs: (
            pd.DataFrame(), {"status": "NO_DATA", "rows": 0},
        ),
    )

    row = scanner.fetch_one_fundamental("NONE.JK")

    assert row["fundamental_data_grade"] == "D"
    assert row["fundamental_complete_for_multibagger"] is False
    assert row["fundamental_error_code"] == "ALL_SNAPSHOT_PROVIDERS_FAILED"
    assert "NO_NORMALIZED_HISTORY" in row["fundamental_error"]
    assert row["fundamental_provider"] == "Yahoo Fundamentals public routes"


def test_decision_summary_stays_compact() -> None:
    wide = pd.DataFrame([{f"debug_{i}": i for i in range(700)} | {
        "ticker": "TEST.JK", "multibagger_selection_score": 80.0,
        "selected_reason": "test",
    }])
    summary = build_multibagger_decision_summary(wide)
    assert "ticker" in summary
    assert "debug_1" not in summary
    assert len(summary.columns) <= 42
