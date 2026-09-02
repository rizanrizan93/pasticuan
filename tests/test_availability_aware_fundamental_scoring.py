from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

import scanner


def test_missing_fundamental_is_explicitly_not_scored() -> None:
    frame = pd.DataFrame([{
        "ticker": "TEST.JK",
        "status": "WATCHLIST_ENTRY",
        "fundamental_score": np.nan,
        "fundamental_coverage": 0.0,
        "fundamental_error": "provider unavailable",
    }])

    out = scanner.apply_fundamental_gate(frame)

    assert out.loc[0, "fundamental_tier"] == "MISSING_NOT_SCORED"
    assert np.isnan(float(out.loc[0, "fundamental_confidence"]))
    assert not bool(out.loc[0, "fundamental_critical_blocker"])
    assert "dikeluarkan dari scoring" in str(out.loc[0, "evidence_warnings"])


def test_execution_integrity_does_not_create_fundamental_coverage_failure() -> None:
    frame = pd.DataFrame([{
        "ticker": "TEST.JK",
        "status": "EXECUTION_READY",
        "technical_setup_ready": True,
        "quality_score": 90.0,
        "entry": 1000.0,
        "stop_loss": 950.0,
        "tp1": 1100.0,
        "tp2": 1200.0,
        "rr1": 2.0,
        "rr2": 4.0,
        "stop_pct": 5.0,
        "sizing_status": "OK",
        "suggested_lots": 1,
        "capital_required_idr": 100_000.0,
        "max_loss_idr": 5_000.0,
        "portfolio_selected": True,
        "market_status_confidence": 100.0,
        "news_confidence": 100.0,
        "validation_confidence": 100.0,
        "quote_confidence": 100.0,
        "universe_confidence": 100.0,
        "fundamental_score": np.nan,
        "fundamental_coverage": 0.0,
        "fundamental_confidence": np.nan,
        "fundamental_critical_blocker": False,
        "ohlcv_source_tier": "LIVE",
        "independent_price_verified": True,
        "critical_blockers": "",
    }])

    out = scanner._finalize_execution_integrity_v431(frame)

    assert out.loc[0, "fundamental_scoring_state"] == "MISSING_NOT_SCORED"
    failures = str(out.loc[0, "execution_gate_failures"])
    assert "FUNDAMENTAL_COVERAGE" not in failures
    assert "FUNDAMENTAL_DISTRESS" not in failures


def test_autopilot_and_signal_first_use_fundamental_safety_not_coverage() -> None:
    autopilot_source = inspect.getsource(scanner._autopilot_gate_evaluation)
    signal_first_source = inspect.getsource(scanner._signal_first_execution_evaluation)

    assert "FUNDAMENTAL_COVERAGE" not in autopilot_source
    assert "FUNDAMENTAL_COVERAGE" not in signal_first_source
    assert "FUNDAMENTAL_SAFETY" in autopilot_source
    assert "FUNDAMENTAL_SAFETY" in signal_first_source
