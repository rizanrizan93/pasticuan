from __future__ import annotations

import pandas as pd

from ownership_family_coalescing_patch import merge_ksei, merge_public


def test_public_zero_placeholder_family_is_replaced() -> None:
    frame = pd.DataFrame([{
        "ticker": "APII.JK",
        "ownership_public_context_coverage_pct": 0.0,
        "ownership_public_insiders_held_pct": 0.0,
        "ownership_public_institutions_held_pct": 0.0,
        "ownership_public_source_authority": "",
    }])
    context = {"APII.JK": {
        "ownership_public_context_coverage_pct": 100.0,
        "ownership_public_insiders_held_pct": 0.0,
        "ownership_public_institutions_held_pct": 31.5,
        "ownership_public_source_authority": "PUBLIC_PROVIDER",
        "ownership_public_official_verified": False,
        "ownership_public_context_state": "CONTEXT_ONLY_NOT_REGULATORY_FREE_FLOAT",
    }}
    row = merge_public(frame, context).iloc[0]
    assert row["ownership_public_context_coverage_pct"] == 100.0
    assert row["ownership_public_institutions_held_pct"] == 31.5
    assert row["ownership_public_insiders_held_pct"] == 0.0
    assert row["ownership_public_official_verified"] == False


def test_valid_public_observed_zero_is_preserved() -> None:
    frame = pd.DataFrame([{
        "ticker": "APII.JK",
        "ownership_public_context_coverage_pct": 100.0,
        "ownership_public_insiders_held_pct": 0.0,
        "ownership_public_source_authority": "PUBLIC_PROVIDER",
    }])
    context = {"APII.JK": {
        "ownership_public_context_coverage_pct": 100.0,
        "ownership_public_insiders_held_pct": 9.0,
        "ownership_public_source_authority": "PUBLIC_PROVIDER",
    }}
    row = merge_public(frame, context).iloc[0]
    assert row["ownership_public_insiders_held_pct"] == 0.0


def test_ksei_zero_placeholder_family_is_replaced_by_official_context() -> None:
    frame = pd.DataFrame([{
        "ticker": "APII.JK",
        "ownership_ksei_context_coverage_pct": 0.0,
        "ownership_ksei_scripless_pct": 0.0,
        "ownership_ksei_local_pct": 0.0,
        "ownership_ksei_foreign_pct": 0.0,
        "ownership_ksei_official_verified": False,
        "ownership_ksei_source_authority": "",
    }])
    context = {"APII.JK": {
        "ownership_ksei_context_coverage_pct": 100.0,
        "ownership_ksei_scripless_pct": 25.5283706403,
        "ownership_ksei_local_pct": 21.0478326730,
        "ownership_ksei_foreign_pct": 78.9521673270,
        "ownership_ksei_total_shares": 1075760000.0,
        "ownership_ksei_official_verified": True,
        "ownership_ksei_source_authority": "OFFICIAL_KSEI",
        "ownership_ksei_provenance_state": "KSEI_REGISTRATION_COMPOSITION_NOT_REGULATORY_FREE_FLOAT",
    }}
    row = merge_ksei(frame, context).iloc[0]
    assert row["ownership_ksei_context_coverage_pct"] == 100.0
    assert round(float(row["ownership_ksei_scripless_pct"]), 3) == 25.528
    assert round(float(row["ownership_ksei_local_pct"]), 3) == 21.048
    assert round(float(row["ownership_ksei_foreign_pct"]), 3) == 78.952
    assert bool(row["ownership_ksei_official_verified"])


def test_valid_ksei_zero_percent_is_not_overwritten() -> None:
    frame = pd.DataFrame([{
        "ticker": "ZERO.JK",
        "ownership_ksei_context_coverage_pct": 100.0,
        "ownership_ksei_scripless_pct": 100.0,
        "ownership_ksei_local_pct": 0.0,
        "ownership_ksei_foreign_pct": 100.0,
        "ownership_ksei_official_verified": True,
        "ownership_ksei_source_authority": "OFFICIAL_KSEI",
    }])
    context = {"ZERO.JK": {
        "ownership_ksei_context_coverage_pct": 100.0,
        "ownership_ksei_scripless_pct": 100.0,
        "ownership_ksei_local_pct": 10.0,
        "ownership_ksei_foreign_pct": 90.0,
        "ownership_ksei_official_verified": True,
        "ownership_ksei_source_authority": "OFFICIAL_KSEI",
    }}
    row = merge_ksei(frame, context).iloc[0]
    assert row["ownership_ksei_local_pct"] == 0.0
    assert row["ownership_ksei_foreign_pct"] == 100.0
    assert not any("free_float" in str(column).lower() for column in row.index)
