from __future__ import annotations

import numpy as np
import pandas as pd

from ai_engine import validation_events_to_memory
from emir_method_engine import build_emir_method_profile
from narrative_engine import _capital_allocation_alignment_proxy, _structured_financial_lineage
from scanner import ScanConfig


def _price_frame(n: int = 260) -> pd.DataFrame:
    idx = pd.bdate_range("2025-01-02", periods=n)
    close = pd.Series(np.linspace(100.0, 145.0, n), index=idx)
    return pd.DataFrame({
        "Open": close * 0.997,
        "High": close * 1.012,
        "Low": close * 0.988,
        "Close": close,
        "Volume": np.linspace(1_000_000, 1_800_000, n),
    }, index=idx)


def test_scan_config_defaults_cover_400_ticker_universe() -> None:
    cfg = ScanConfig()
    assert cfg.validation_max_tickers == 400
    assert cfg.fundamental_top_n == 400
    assert cfg.fundamental_history_top_n == 400
    assert cfg.full_completion_max_tickers == 400
    assert cfg.narrative_full_completion_max_tickers == 400
    assert cfg.relative_overlay_max_pct == 10.0
    assert cfg.fundamental_provider_batch_size == 40


def test_partial_structured_financial_evidence_scores_but_is_not_production() -> None:
    lineage = _structured_financial_lineage(
        {
            "fundamental_source_families": "YAHOO_FUNDAMENTALS_TIMESERIES",
            "fundamental_source_count": 1,
            "fundamental_history_coverage": 82.0,
            "fundamental_data_grade": "C",
            # reporting period intentionally omitted
        },
        {"score": 74.0, "coverage_pct": 86.0},
    )
    assert lineage["eligible"] is True
    assert lineage["production_eligible"] is False
    assert lineage["state"] == "PARTIAL_STRUCTURED_FINANCIAL_EVIDENCE"
    assert 0.0 < lineage["coverage_pct"] <= 32.0
    assert "REPORTING_PERIOD" in lineage["missing"]


def test_issuer_alignment_is_continuous_not_bucketed() -> None:
    strong = _capital_allocation_alignment_proxy({
        "roic_proxy": 0.20,
        "cash_conversion_ttm": 1.10,
        "free_cash_flow": 150.0,
        "operating_cash_flow": 260.0,
        "revenue": 1_000.0,
        "capital_expenditure": 100.0,
        "share_dilution_yoy": -0.01,
        "revenue_growth": 0.22,
        "earnings_growth": 0.28,
        "net_debt_ebitda": 0.5,
    })
    weak = _capital_allocation_alignment_proxy({
        "roic_proxy": 0.05,
        "cash_conversion_ttm": 0.45,
        "free_cash_flow": -40.0,
        "operating_cash_flow": 30.0,
        "revenue": 1_000.0,
        "capital_expenditure": 250.0,
        "share_dilution_yoy": 0.12,
        "revenue_growth": -0.05,
        "earnings_growth": -0.12,
        "net_debt_ebitda": 4.0,
    })
    assert strong["score"] > weak["score"] + 20.0
    assert strong["coverage_pct"] >= 70.0
    assert weak["coverage_pct"] >= 70.0


def _emir_profile(distribution_days: float, failed_absorption: float) -> dict:
    return build_emir_method_profile(
        ticker="TEST",
        frame=_price_frame(),
        active_events=pd.DataFrame(),
        outcomes={},
        fundamental={
            "fundamental_history_quarters": 8,
            "fundamental_source_count": 2,
        },
        silent_profile={
            "effective_silent_accumulation_score": 58.0,
            "silent_accumulation_confidence": 78.0,
            "persistent_bid_score": 58.0,
            "accumulation_persistence_score": 57.0,
            "weighted_close_location20": 0.58,
            "absorption_confirmed_days20": 1,
            "churning_support_days20": 1,
            "failed_absorption_days20": failed_absorption,
            "distribution_days20": distribution_days,
            "adtv20_idr": 8_000_000_000,
            "liquidity_bucket": "MEDIUM",
        },
        narrative_effective_score=70.0,
        narrative_evidence_coverage_pct=72.0,
        narrative_evidence_mode="STRUCTURED_FINANCIAL",
        alignment_effective_score=72.0,
        alignment_coverage_pct=70.0,
        adoption_stage="EARLY_DISCOVERY",
        crowding_risk_score=25.0,
        hard_block=False,
    )


def test_ohlcv_distribution_penalty_is_continuous() -> None:
    mild = _emir_profile(distribution_days=2, failed_absorption=0)
    severe = _emir_profile(distribution_days=6, failed_absorption=3)
    assert severe["distribution_severity_score"] > mild["distribution_severity_score"]
    assert severe["emir_swing_rank_adjustment"] < mild["emir_swing_rank_adjustment"]
    assert severe["distribution_penalty_points"] > mild["distribution_penalty_points"]


def test_validation_memory_persists_only_genuine_oos_rows() -> None:
    events = pd.DataFrame([
        {
            "ticker": "ABCD", "setup": "BREAKOUT_RETEST", "signal_date": "2026-01-02",
            "entry": 100.0, "stop_loss": 94.0, "tp1": 112.0, "tp2": 120.0,
            "is_oos": True, "filled": True, "r_multiple": 1.5,
            "validation_event_tier": "TRIGGER_CANDIDATE", "production_gate_pass": False,
        },
        {
            "ticker": "EFGH", "setup": "PULLBACK_CONTINUATION", "signal_date": "2026-01-03",
            "entry": 200.0, "stop_loss": 190.0, "tp1": 220.0,
            "is_oos": False, "filled": True, "r_multiple": 1.0,
        },
    ])
    memory = validation_events_to_memory(events)
    assert len(memory) == 1
    row = memory.iloc[0]
    assert row["ticker"] == "ABCD"
    assert row["outcome_quality"] == "CHRONOLOGICAL_OOS_CAUSAL"
    assert row["production_gate_pass"] is False or bool(row["production_gate_pass"]) is False
