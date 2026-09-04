from __future__ import annotations

from pathlib import Path

import pandas as pd

import phase56_runtime_coverage_integrity_patch as patch


ROOT = Path(__file__).resolve().parents[1]


def test_public_ownership_context_fills_existing_null_canonical_columns() -> None:
    frame = pd.DataFrame([
        {
            "ticker": "ADMR.JK",
            "ranking_score": 77.0,
            "ownership_public_context_coverage_pct": None,
            "ownership_public_official_verified": None,
            "ownership_public_context_provenance_state": "",
        }
    ])
    context = {
        "ADMR": {
            "ownership_public_context_coverage_pct": 100.0,
            "ownership_public_official_verified": False,
            "ownership_public_context_provenance_state": "PUBLIC_PROVIDER_YAHOO_CONCENTRATION_NOT_IDX_KSEI",
        }
    }

    out = patch._coalescing_ownership_merge(frame, context)

    assert out.loc[0, "ownership_public_context_coverage_pct"] == 100.0
    assert not bool(out.loc[0, "ownership_public_official_verified"])
    assert out.loc[0, "ownership_public_context_provenance_state"] == "PUBLIC_PROVIDER_YAHOO_CONCENTRATION_NOT_IDX_KSEI"
    assert out.loc[0, "ranking_score"] == 77.0
    assert not any(str(column).endswith("_phase56") for column in out.columns)


def test_public_ownership_existing_meaningful_canonical_value_wins() -> None:
    frame = pd.DataFrame([
        {
            "ticker": "MARK.JK",
            "ownership_public_context_coverage_pct": 75.0,
            "ownership_public_context_provenance_state": "EXPLICIT_RUNTIME_CONTEXT",
        }
    ])
    context = {
        "MARK.JK": {
            "ownership_public_context_coverage_pct": 100.0,
            "ownership_public_context_provenance_state": "PUBLIC_PROVIDER_YAHOO_CONCENTRATION_NOT_IDX_KSEI",
        }
    }

    out = patch._coalescing_ownership_merge(frame, context)

    assert out.loc[0, "ownership_public_context_coverage_pct"] == 75.0
    assert out.loc[0, "ownership_public_context_provenance_state"] == "EXPLICIT_RUNTIME_CONTEXT"


def test_null_technical_audit_columns_are_filled_from_current_fields() -> None:
    frame = pd.DataFrame([
        {
            "ticker": "OMED.JK",
            "active_setup": None,
            "technical_entry_state": "",
            "strategy": "CORE_SWING",
            "setup_status": "ENTRY_PLAN_READY",
        }
    ])

    out = patch._null_aware_technical_aliases(
        frame,
        ("ticker", "active_setup", "technical_entry_state"),
    )

    assert out is not None
    assert out.loc[0, "active_setup"] == "CORE_SWING"
    assert out.loc[0, "technical_entry_state"] == "ENTRY_PLAN_READY"


def test_explicit_technical_audit_values_are_not_overwritten() -> None:
    frame = pd.DataFrame([
        {
            "ticker": "TSPC.JK",
            "active_setup": "EXPLICIT_SETUP",
            "technical_entry_state": "EXPLICIT_STATE",
            "strategy": "CORE_SWING",
            "setup_status": "WATCHLIST",
        }
    ])

    out = patch._null_aware_technical_aliases(
        frame,
        ("ticker", "active_setup", "technical_entry_state"),
    )

    assert out is not None
    assert out.loc[0, "active_setup"] == "EXPLICIT_SETUP"
    assert out.loc[0, "technical_entry_state"] == "EXPLICIT_STATE"


def test_runtime_release_installs_integrity_patch_before_ownership_telemetry() -> None:
    source = (ROOT / "runtime_release.py").read_text(encoding="utf-8")
    phase56 = source.index('"phase56_coverage_runtime_patch", "install"')
    integrity = source.index('"phase56_runtime_coverage_integrity_patch", "install"')
    telemetry = source.index('"pasticuan_ownership_calibration_telemetry_patch", "install"')
    assert phase56 < integrity < telemetry


def test_patch_cannot_create_execution_or_regulatory_authorization() -> None:
    source = (ROOT / "phase56_runtime_coverage_integrity_patch.py").read_text(encoding="utf-8")
    assert "real_money_authorization_pass" not in source
    assert "regulatory_free_float_pct" not in source
    assert "source_quorum_verified = True" not in source
