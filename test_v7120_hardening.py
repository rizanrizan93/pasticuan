from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd

import scanner
import scanner_focus
from emir_method_engine import _broker_summary_component
from scanner import ScanConfig, parse_broker_summary_csv
from scanner_focus import (
    _complete_core_radar_universe,
    _direct_broker_flow_mask,
    _multibagger_confidence_profile,
    scan_multibagger_candidates,
)


def _price_frame(periods: int) -> pd.DataFrame:
    index = pd.bdate_range(end="2026-07-31", periods=periods)
    close = np.linspace(100.0, 160.0, periods)
    return pd.DataFrame(
        {
            "Open": close * 0.995,
            "High": close * 1.015,
            "Low": close * 0.985,
            "Close": close,
            "Volume": np.full(periods, 2_000_000.0),
        },
        index=index,
    )


def _strong_fundamental(ticker: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "ticker": ticker,
        "fundamental_coverage": 90.0,
        "fundamental_score": 88.0,
        "fundamental_score_10": 8.8,
        "fundamental_reliability": "HIGH",
        "fundamental_data_grade": "B",
        "fundamental_source_count": 2,
        "fundamental_history_source_count": 1,
        "fundamental_history_quarters": 8,
        "fundamental_history_years": 3,
        "fundamental_history_coverage": 90.0,
        "fundamental_statement_family_coverage_pct": 90.0,
        "fundamental_history_period_coverage_pct": 90.0,
        "fundamental_income_statement_coverage_pct": 90.0,
        "fundamental_balance_sheet_coverage_pct": 90.0,
        "fundamental_cashflow_statement_coverage_pct": 90.0,
        "fundamental_complete_for_multibagger": True,
        "fundamental_consensus_score": 80.0,
        "statement_age_days": 60.0,
        "revenue_growth": 0.22,
        "earnings_growth": 0.28,
        "roe": 0.20,
        "roa": 0.11,
        "operating_margin": 0.18,
        "net_margin": 0.14,
        "debt_equity": 0.40,
        "current_ratio": 2.0,
        "cash_to_debt": 1.1,
        "operating_cash_flow": 140.0,
        "free_cash_flow": 110.0,
        "peg_ratio": 1.0,
        "fcf_yield": 0.05,
        "market_cap": 8_000_000_000_000.0,
        "history_cash_conversion": 1.15,
        "history_positive_ocf_ratio": 1.0,
        "history_positive_earnings_ratio": 1.0,
        "history_margin_stability": 0.90,
        "history_share_dilution_yoy": 0.0,
        "history_roic_proxy": 0.18,
        "history_gross_profitability": 0.25,
        "history_gross_margin": 0.38,
        "history_gross_profit_growth": 0.20,
        "history_accruals_to_assets": -0.01,
        "history_leverage_change_yoy": -0.03,
        "history_net_debt_ebitda": 0.6,
        "history_interest_coverage": 10.0,
        "fundamental_model": "GENERAL",
        "sector": "Industrials",
    }])


def _broker_rows() -> pd.DataFrame:
    dates = pd.bdate_range(end="2026-07-31", periods=10)
    return pd.DataFrame({
        "ticker": ["TEST"] * len(dates),
        "date": dates,
        "broker_code": ["YP"] * len(dates),
        "buy_value": np.full(len(dates), 2_000_000_000.0),
        "sell_value": np.full(len(dates), 1_000_000_000.0),
    })


def test_short_history_ticker_stays_visible_and_unranked() -> None:
    result = scan_multibagger_candidates(
        {"NEW.JK": _price_frame(20)},
        _strong_fundamental("NEW.JK"),
    )

    assert result["ticker"].tolist() == ["NEW.JK"]
    row = result.iloc[0]
    assert row["multibagger_scoring_state"] == (
        "DATA_NOT_SCORED_INSUFFICIENT_TECHNICAL_HISTORY"
    )
    assert int(row["technical_history_bars"]) == 20
    assert not bool(row["multibagger_rank_eligible"])
    assert pd.isna(row["multibagger_selection_score"])
    assert pd.isna(row["multibagger_selection_rank"])


def test_core_radar_appends_selector_excluded_ticker() -> None:
    radar = pd.DataFrame([{
        "ticker": "OLD.JK",
        "as_of": pd.Timestamp("2026-07-31"),
        "swing_selection_score": 70.0,
    }])
    completed = _complete_core_radar_universe(
        radar,
        {"OLD.JK": _price_frame(260), "NEW.JK": _price_frame(18)},
        {},
    )

    assert set(completed["ticker"]) == {"OLD.JK", "NEW.JK"}
    existing = completed.loc[completed["ticker"].eq("OLD.JK")].iloc[0]
    assert bool(existing["daily_session_current"])
    pending = completed.loc[completed["ticker"].eq("NEW.JK")].iloc[0]
    assert not bool(pending["selector_rank_eligible"])
    assert "INSUFFICIENT_TECHNICAL_HISTORY" in pending["selector_data_state"]


def test_manual_broker_upload_cannot_certify_direct_flow() -> None:
    parsed = parse_broker_summary_csv(
        _broker_rows(), as_of="2026-08-02 12:00:00+07:00",
    )
    row = parsed.iloc[0].to_dict()
    component = _broker_summary_component(row)

    assert row["broksum_provenance_state"] == "MANUAL_OR_UNVERIFIED_SOURCE"
    assert not bool(row["broksum_direct_evidence_eligible"])
    assert not bool(component["direct_verified"])


def test_authenticated_current_broker_route_can_certify_direct_flow() -> None:
    parsed = parse_broker_summary_csv(
        _broker_rows(),
        source_type="PROVIDER_API",
        source_name="unit-provider",
        source_verified=True,
        as_of="2026-08-02 12:00:00+07:00",
    )
    row = parsed.iloc[0].to_dict()
    component = _broker_summary_component(row)

    assert row["broksum_provenance_state"] == "DIRECT_SOURCE_VERIFIED"
    assert bool(row["broksum_direct_evidence_eligible"])
    assert bool(component["direct_verified"])


def test_direct_flow_mask_rejects_spoofed_coverage_without_provenance() -> None:
    frame = pd.DataFrame([
        {
            "broker_summary_coverage_pct": 100.0,
            "smart_money_flow_evidence_mode": "BROKER_SUMMARY_OBSERVED_UNVERIFIED",
            "broker_summary_direct_verified": False,
        },
        {
            "broker_summary_coverage_pct": 100.0,
            "smart_money_flow_evidence_mode": "DIRECT_BROKER_SUMMARY_VERIFIED",
            "broker_summary_direct_verified": True,
        },
    ])

    assert _direct_broker_flow_mask(frame).tolist() == [False, True]


def test_zero_weight_eoff_does_not_change_multibagger_confidence() -> None:
    common = dict(
        coverage=85.0,
        data_grade="B",
        reliability="HIGH",
        official_verified=False,
        history_coverage=85.0,
        consensus_score=75.0,
        project_coverage=50.0,
        management_coverage=50.0,
        future_impact_confidence="MEDIUM",
        accumulation_confidence=70.0,
        execution_readiness=68.0,
        core_execution_confidence=70.0,
    )
    weak = _multibagger_confidence_profile(
        **common,
        time_cycle_confidence=0.0,
        best_buy_confidence=0.0,
        eoff_validation_state="UNVALIDATED",
        eoff_events=0.0,
        eoff_lift=0.0,
    )
    strong = _multibagger_confidence_profile(
        **common,
        time_cycle_confidence=100.0,
        best_buy_confidence=100.0,
        eoff_validation_state="VALIDATED",
        eoff_events=200.0,
        eoff_lift=10.0,
    )

    assert weak["overall_research_confidence"] == strong["overall_research_confidence"]
    assert weak["eoff_confidence_effective_weight_pct"] == 0.0


def test_default_daily_multibagger_scan_skips_full_time_cycle() -> None:
    prepared = {"FAST.JK": scanner.prepare_indicators(_price_frame(280))}
    with mock.patch("scanner_focus.analyze_time_cycle") as timing:
        result = scan_multibagger_candidates(
            prepared,
            _strong_fundamental("FAST.JK"),
            config=ScanConfig(),
        )

    timing.assert_not_called()
    assert result.iloc[0]["time_cycle_evaluation_mode"] == (
        "DEFERRED_ZERO_PRODUCTION_WEIGHT"
    )


def test_full_multibagger_time_cycle_remains_explicit_opt_in() -> None:
    prepared = {"SHADOW.JK": scanner.prepare_indicators(_price_frame(280))}
    shadow = {
        "time_cycle_state": "LIMITED_EVIDENCE",
        "time_cycle_confidence": 25.0,
        "bullish_timing_score": 50.0,
        "continuation_timing_score": 50.0,
    }
    config = ScanConfig().replace(
        multibagger_time_cycle_full_refresh_enabled=True,
    )
    with mock.patch(
        "scanner_focus.analyze_time_cycle", return_value=shadow,
    ) as timing:
        result = scan_multibagger_candidates(
            prepared,
            _strong_fundamental("SHADOW.JK"),
            config=config,
        )

    timing.assert_called_once()
    assert result.iloc[0]["time_cycle_evaluation_mode"] == "FULL_CANDIDATE"


def test_fundamental_runtime_cache_is_safe_json(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("IDX_SCANNER_CACHE_DIR", str(tmp_path))
    history = pd.DataFrame([{
        "ticker": "SAFE.JK",
        "period_end": "2025-12-31",
        "period_type": "FY",
        "statement_basis": "ANNUAL",
        "source_family": "UNIT",
        "source_name": "unit",
        "source_url": "https://example.com/report",
        "currency": "IDR",
        "revenue": 100.0,
        "net_income": 10.0,
        "total_assets": 200.0,
        "total_liabilities": 80.0,
        "equity": 120.0,
        "operating_cash_flow": 15.0,
        "capex": -5.0,
        "source_verified": False,
        "validation_flags": "",
    }])

    scanner._write_recent_direct_fundamental_history("SAFE.JK", history)
    path = scanner._direct_fundamental_history_cache_path("SAFE.JK")
    restored = scanner._load_recent_direct_fundamental_history("SAFE.JK")

    assert path.suffix == ".json"
    assert path.is_file()
    assert not restored.empty
    assert restored.iloc[0]["ticker"] == "SAFE.JK"
