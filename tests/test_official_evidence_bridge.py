from __future__ import annotations

import numpy as np
import pandas as pd

from narrative_engine import _verified_forward_profile
from official_evidence_bridge import (
    bridge_management_evidence,
    bridge_project_events,
    corporate_events_to_narrative,
)
from runtime_integrity_patch import (
    _GOVERNED_EVIDENCE_CACHE,
    _wrap_forward_quality_reader,
    _wrap_narrative_event_reader,
)


def _project(**updates):
    row = {
        "ticker": "AAA.JK",
        "project_name": "Signed capacity expansion contract",
        "project_stage": "SIGNED_CONTRACT_IN_PROGRESS_Q4_2026",
        "project_data_coverage": 90.0,
        "project_source_urls": "https://issuer.example/a.pdf | https://issuer.example/b.pdf",
        "project_source_families": "ISSUER_DISCLOSURE|IDX_DISCLOSURE",
        "project_source_quorum_verified": True,
        "source_quorum_count": 2,
        "entity_match_verified": True,
        "last_verified_at": "2026-08-10T00:00:00Z",
        "project_capex_idr": 500_000_000_000,
        "project_execution_flags": "SIGNED;IN_PROGRESS;Q4_2026",
    }
    row.update(updates)
    return row


def test_strict_project_becomes_consumable_direct_forward_evidence():
    bridged = bridge_project_events(pd.DataFrame([_project()]), as_of="2026-08-15T00:00:00Z")
    assert len(bridged) == 1
    row = bridged.iloc[0]
    assert 0 < float(row["project_pipeline_score_observed"]) <= 88
    assert 0 < float(row["future_fundamental_impact_score_observed"]) <= 85
    assert row["official_forward_score_disclaimer"] == "SCANNER_DERIVED_NOT_ISSUER_REPORTED_SCORE"

    profile = _verified_forward_profile(bridged.to_dict("records"), pd.Timestamp("2026-08-15", tz="UTC"))
    assert profile["forward_source_quorum_verified"] is True
    assert profile["forward_evidence_state"] == "VERIFIED_DIRECT_FORWARD_EVIDENCE"
    assert np.isfinite(profile["forward_project_pipeline_score"])


def test_single_source_or_entity_mismatch_stays_fail_closed():
    single = _project(
        project_source_urls="https://issuer.example/a.pdf",
        source_quorum_count=1,
        project_source_quorum_verified=False,
    )
    mismatch = _project(entity_match_verified=False)
    bridged = bridge_project_events(pd.DataFrame([single, mismatch]), as_of="2026-08-15T00:00:00Z")
    assert bridged.empty


def test_board_roster_adds_coverage_but_never_invents_management_quality():
    roles = pd.DataFrame([{
        "ticker": "AAA.JK", "person_name": "Director A", "role": "PRESIDENT_DIRECTOR",
        "updated_at": "2026-08-10T00:00:00Z", "source_url": "https://issuer.example/board",
        "verified": True, "source_quorum_verified": True, "source_quorum_count": 2,
        "entity_match_verified": True,
    }])
    bridged = bridge_management_evidence(roles, pd.DataFrame(), pd.DataFrame(), as_of="2026-08-15T00:00:00Z")
    assert len(bridged) == 1
    assert float(bridged.iloc[0]["management_data_coverage"]) > 0
    assert pd.isna(bridged.iloc[0]["management_quality_score_observed"])
    assert bridged.iloc[0]["management_evidence_state"] == "DIRECT_FACTS_NO_QUALITY_SCORE_INFERENCE"


def test_verified_insider_ownership_is_passed_as_observed_fact():
    ownership = pd.DataFrame([
        {
            "ticker": "AAA.JK", "holder_name": "Director A", "holder_type": "INSIDER",
            "ownership_pct_after": 3.0, "report_date": "2026-07-31",
            "source_url": "https://issuer.example/ownership.pdf", "verified": True,
            "source_quorum_verified": True, "source_quorum_count": 2, "entity_match_verified": True,
        },
        {
            "ticker": "AAA.JK", "holder_name": "Commissioner B", "holder_type": "INSIDER",
            "ownership_pct_after": 2.0, "report_date": "2026-07-31",
            "source_url": "https://issuer.example/ownership.pdf", "verified": True,
            "source_quorum_verified": True, "source_quorum_count": 2, "entity_match_verified": True,
        },
    ])
    bridged = bridge_management_evidence(pd.DataFrame(), ownership, pd.DataFrame(), as_of="2026-08-15T00:00:00Z")
    assert float(bridged.iloc[0]["insider_ownership_pct"]) == 5.0
    assert pd.isna(bridged.iloc[0]["management_quality_score_observed"])


def test_strict_capital_action_enters_narrative_alignment_contract():
    corporate = pd.DataFrame([{
        "ticker": "AAA.JK", "event_type": "CASH_DIVIDEND", "event_date": "2026-07-01",
        "published_at": "2026-07-01T00:00:00Z", "materiality": 70,
        "source_url": "https://issuer.example/dividend.pdf", "verified": True,
        "source_quorum_verified": True, "source_quorum_count": 2, "entity_match_verified": True,
        "source_family": "ISSUER|KSEI", "metadata": {"title": "Cash dividend 2026"},
    }])
    events = corporate_events_to_narrative(corporate, as_of="2026-08-15T00:00:00Z")
    assert len(events) == 1
    assert events.iloc[0]["event_type"] == "DIVIDEND_OR_CAPITAL_RETURN"
    assert int(events.iloc[0]["impact_sign"]) == 1
    assert bool(events.iloc[0]["official_verified"]) is True


class _FakeSettings:
    mode = "SUPABASE_REST"


class _FakeBridge:
    settings = _FakeSettings()

    def read_forward_quality_cache(self, tickers):
        return pd.DataFrame(), pd.DataFrame([{
            "ticker": "AAA.JK", "provider": "FORWARD_QUALITY_CACHE", "database_read_state": "DATABASE_MISSING",
        }])

    def read_narrative_events(self, tickers, *, limit=10000):
        return pd.DataFrame()

    def _get_rows(self, table, params):
        tables = {
            "project_events": [_project()],
            "management_roles": [{
                "ticker": "AAA.JK", "person_name": "Director A", "role": "PRESIDENT_DIRECTOR",
                "updated_at": "2026-08-10T00:00:00Z", "source_url": "https://issuer.example/board",
                "verified": True, "source_quorum_verified": True, "source_quorum_count": 2,
                "entity_match_verified": True,
            }],
            "ownership_events": [{
                "ticker": "AAA.JK", "holder_name": "Director A", "holder_type": "INSIDER",
                "ownership_pct_after": 4.0, "report_date": "2026-07-31",
                "source_url": "https://issuer.example/ownership.pdf", "verified": True,
                "source_quorum_verified": True, "source_quorum_count": 2,
                "entity_match_verified": True,
            }],
            "corporate_events": [{
                "ticker": "AAA.JK", "event_type": "CASH_DIVIDEND", "event_date": "2026-07-01",
                "published_at": "2026-07-01T00:00:00Z", "materiality": 70,
                "source_url": "https://issuer.example/dividend.pdf", "verified": True,
                "source_quorum_verified": True, "source_quorum_count": 2,
                "entity_match_verified": True, "source_family": "ISSUER|KSEI",
                "metadata": {"title": "Cash dividend 2026"},
            }],
        }
        return tables.get(table, [])


def test_runtime_reader_consumes_raw_governed_tables_end_to_end():
    _GOVERNED_EVIDENCE_CACHE.clear()
    bridge_cls = type("RuntimeBridge", (_FakeBridge,), {})
    _wrap_forward_quality_reader(bridge_cls)
    _wrap_narrative_event_reader(bridge_cls)
    bridge = bridge_cls()

    forward, audit = bridge.read_forward_quality_cache(["AAA.JK"])
    project = forward.loc[forward.get("project_name", pd.Series(index=forward.index, dtype=object)).notna()]
    assert len(project) == 1
    assert np.isfinite(pd.to_numeric(project.iloc[0]["project_pipeline_score_observed"], errors="coerce"))
    assert bool(project.iloc[0]["project_source_quorum_verified"]) is True
    management = forward.loc[forward.get("management_evidence_state", pd.Series(index=forward.index, dtype=object)).notna()]
    assert len(management) == 1
    assert float(management.iloc[0]["insider_ownership_pct"]) == 4.0
    assert (audit.get("status", pd.Series(dtype=str)).astype(str) == "BRIDGED").any()

    events = bridge.read_narrative_events(["AAA.JK"])
    assert len(events) == 1
    assert events.iloc[0]["event_type"] == "DIVIDEND_OR_CAPITAL_RETURN"
    assert bool(events.iloc[0]["official_verified"]) is True
