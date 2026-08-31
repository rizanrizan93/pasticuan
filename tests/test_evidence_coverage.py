from __future__ import annotations

import pandas as pd

from evidence_coverage import CoveragePolicy, build_evidence_coverage
from idx_trading_calendar import latest_expected_completed_session, previous_idx_session

AS_OF = pd.Timestamp("2026-08-31 17:00", tz="Asia/Jakarta")


def _sessions(count: int = 20) -> list[pd.Timestamp]:
    current = latest_expected_completed_session(AS_OF)
    result = []
    while len(result) < count:
        result.append(current)
        current = previous_idx_session(current, include_date=False)
    return result


def _fundamental(**overrides) -> dict:
    row = {
        "ticker": "GOOD", "report_date": "2025-12-31", "period_type": "FY",
        "issuer_match": True, "source_verified": True, "source": "IDX_XBRL_OFFICIAL",
        "source_url": "https://www.idx.co.id/report", "revenue": 100.0,
        "net_income": 10.0, "assets": 200.0, "liabilities": 80.0,
        "equity": 120.0, "cash": 20.0, "short_term_debt": 5.0,
        "long_term_debt": 15.0, "ocf": 12.0, "capex": 2.0,
        "ocf_period_type": "FY", "capex_period_type": "FY",
    }
    row.update(overrides)
    return row


def test_complete_record_set_is_measured_from_evidence() -> None:
    sessions = _sessions()
    detail, summary = build_evidence_coverage(
        ["GOOD.JK"],
        ohlcv=pd.DataFrame([{"ticker": "GOOD", "trade_date": sessions[0], "close": 100, "volume": 10}]),
        fundamentals=pd.DataFrame([_fundamental()]),
        forward=pd.DataFrame([{"ticker": "GOOD", "evidence_date": "2026-08-01", "evidence_type": "CONTRACT", "issuer_match": True, "source_verified": True, "source_url": "https://www.idx.co.id/disclosure"}]),
        foreign=pd.DataFrame([{"ticker": "GOOD", "trade_date": day, "foreign_net_shares": 1, "source": "ZAPI_IDX_FOREIGN_FLOW"} for day in sessions]),
        as_of=AS_OF,
    )
    row = detail.iloc[0]
    assert bool(row["fully_evidence_ready"])
    assert bool(row["fundamental_official"])
    assert bool(row["fcf_available"])
    assert float(row["foreign_coverage_ratio"]) == 1.0
    assert summary["fully_evidence_ready"]["percentage"] == 100.0


def test_missing_and_provider_error_never_become_zero_evidence() -> None:
    detail, summary = build_evidence_coverage(
        ["MISS"], fundamentals=pd.DataFrame([{"ticker": "MISS", "provider_state": "ERROR"}]), as_of=AS_OF,
    )
    row = detail.iloc[0]
    assert not bool(row["fundamental_valid"])
    assert not bool(row["fundamental_revenue_available"])
    assert not bool(row["ocf_available"])
    assert not bool(row["fcf_available"])
    assert "NO_REPORT" in row["missing_reasons"]
    assert summary["fundamental_valid"]["count"] == 0


def test_wrong_ticker_period_identity_and_stale_are_rejected() -> None:
    wrong = _fundamental(ticker="OTHER", report_date="2023-12-31")
    bad = _fundamental(ticker="BAD", report_date="2023-12-31", period_type="UNKNOWN", issuer_match=False)
    detail, _ = build_evidence_coverage(["BAD"], fundamentals=pd.DataFrame([wrong, bad]), as_of=AS_OF)
    row = detail.iloc[0]
    assert not bool(row["fundamental_valid"])
    assert "IDENTITY_MISMATCH" in row["missing_reasons"]
    assert "WRONG_REPORTING_PERIOD" in row["missing_reasons"]
    assert "STALE" in row["missing_reasons"]


def test_superior_official_record_wins_and_forward_duplicates_deduplicate() -> None:
    lower = _fundamental(report_date="2026-06-30", source="SECONDARY_PROXY", source_url="", revenue=999.0)
    item = {"ticker": "GOOD", "evidence_date": "2026-08-01", "evidence_type": "CAPEX", "issuer_match": True, "source_verified": True, "source_url": "https://issuer.example/capex"}
    detail, _ = build_evidence_coverage(["GOOD"], fundamentals=pd.DataFrame([_fundamental(), lower]), forward=pd.DataFrame([item, item]), as_of=AS_OF, policy=CoveragePolicy(require_foreign=False))
    row = detail.iloc[0]
    assert row["fundamental_source"] == "IDX_XBRL_OFFICIAL"
    assert row["fundamental_report_date"] == "2025-12-31"
    assert int(row["forward_evidence_count"]) == 1


def test_fcf_requires_period_compatible_ocf_and_capex() -> None:
    incompatible = _fundamental(ocf_period_type="YTD", capex_period_type="FY")
    missing_capex = _fundamental(ticker="MISS", capex=None)
    detail, _ = build_evidence_coverage(["GOOD", "MISS"], fundamentals=pd.DataFrame([incompatible, missing_capex]), as_of=AS_OF, policy=CoveragePolicy(require_forward=False, require_foreign=False))
    indexed = detail.set_index("ticker")
    assert not bool(indexed.loc["GOOD", "fcf_available"])
    assert not bool(indexed.loc["MISS", "fcf_available"])
    assert bool(indexed.loc["MISS", "ocf_available"])


def test_foreign_holiday_row_is_not_an_observed_session() -> None:
    sessions = _sessions()
    foreign = [{"ticker": "GOOD", "trade_date": day, "source": "ZAPI_IDX_FOREIGN_FLOW"} for day in sessions[:-1]]
    foreign.append({"ticker": "GOOD", "trade_date": "2026-08-25", "source": "ZAPI_IDX_FOREIGN_FLOW"})
    detail, _ = build_evidence_coverage(["GOOD"], foreign=pd.DataFrame(foreign), as_of=AS_OF, policy=CoveragePolicy(require_forward=False))
    row = detail.iloc[0]
    assert int(row["foreign_observed_sessions"]) == 19
    assert float(row["foreign_coverage_ratio"]) == 0.95
    assert row["foreign_freshness_state"] == "FRESH"
