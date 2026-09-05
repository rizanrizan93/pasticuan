from __future__ import annotations

import shared_ownership_payload_recovery_patch as patch


def test_recovers_public_and_ksei_families(monkeypatch):
    monkeypatch.setattr(patch, "_shared_public_ownership", lambda tickers=None: {
        "APII.JK": {
            "ownership_public_context_coverage_pct": 100.0,
            "ownership_public_source_authority": "PUBLIC_PROVIDER",
            "ownership_public_official_verified": False,
            "ownership_public_context_state": "CONTEXT_ONLY_NOT_REGULATORY_FREE_FLOAT",
        }
    })
    monkeypatch.setattr(patch, "_shared_ksei_ownership", lambda tickers=None: {
        "APII.JK": {
            "ownership_ksei_context_coverage_pct": 100.0,
            "ownership_ksei_scripless_pct": 25.528,
            "ownership_ksei_local_pct": 21.048,
            "ownership_ksei_foreign_pct": 78.952,
            "ownership_ksei_source_authority": "OFFICIAL_KSEI",
            "ownership_ksei_official_verified": True,
            "ownership_ksei_context_state": "CONTEXT_ONLY_NOT_BENEFICIAL_OWNERSHIP_OR_FREE_FLOAT",
        }
    })
    payloads = {"multibagger_snapshots": [{"ticker": "APII.JK", "final_score": 77.0}]}
    row = patch._recover_payload_families(payloads)["multibagger_snapshots"][0]
    assert row["ownership_public_context_coverage_pct"] == 100.0
    assert row["ownership_public_official_verified"] is False
    assert row["ownership_ksei_context_coverage_pct"] == 100.0
    assert row["ownership_ksei_scripless_pct"] == 25.528
    assert row["ownership_ksei_official_verified"] is True
    assert row["final_score"] == 77.0


def test_preserves_legitimate_zero_composition(monkeypatch):
    monkeypatch.setattr(patch, "_shared_public_ownership", lambda tickers=None: {})
    monkeypatch.setattr(patch, "_shared_ksei_ownership", lambda tickers=None: {
        "ZERO.JK": {
            "ownership_ksei_context_coverage_pct": 100.0,
            "ownership_ksei_scripless_pct": 100.0,
            "ownership_ksei_local_pct": 0.0,
            "ownership_ksei_foreign_pct": 100.0,
            "ownership_ksei_source_authority": "OFFICIAL_KSEI",
            "ownership_ksei_official_verified": True,
        }
    })
    payloads = {"multibagger_snapshots": [{"ticker": "ZERO.JK"}]}
    row = patch._recover_payload_families(payloads)["multibagger_snapshots"][0]
    assert row["ownership_ksei_local_pct"] == 0.0
    assert row["ownership_ksei_foreign_pct"] == 100.0


def test_preserves_existing_valid_family(monkeypatch):
    monkeypatch.setattr(patch, "_shared_public_ownership", lambda tickers=None: {})
    monkeypatch.setattr(patch, "_shared_ksei_ownership", lambda tickers=None: {
        "KEEP.JK": {
            "ownership_ksei_context_coverage_pct": 100.0,
            "ownership_ksei_scripless_pct": 80.0,
            "ownership_ksei_local_pct": 20.0,
            "ownership_ksei_foreign_pct": 80.0,
            "ownership_ksei_source_authority": "OFFICIAL_KSEI",
            "ownership_ksei_official_verified": True,
        }
    })
    payloads = {"multibagger_snapshots": [{
        "ticker": "KEEP.JK",
        "ownership_ksei_context_coverage_pct": 100.0,
        "ownership_ksei_scripless_pct": 90.0,
        "ownership_ksei_local_pct": 0.0,
        "ownership_ksei_foreign_pct": 100.0,
        "ownership_ksei_source_authority": "OFFICIAL_KSEI",
        "ownership_ksei_official_verified": True,
        "real_money_authorization_pass": False,
    }]}
    row = patch._recover_payload_families(payloads)["multibagger_snapshots"][0]
    assert row["ownership_ksei_scripless_pct"] == 90.0
    assert row["ownership_ksei_local_pct"] == 0.0
    assert row["real_money_authorization_pass"] is False
    assert "effective_free_float_pct" not in row
