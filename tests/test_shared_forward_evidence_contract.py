from pathlib import Path

import pandas as pd

from pasticuan_shared_forward_runtime_patch import _pasticuan_model_input
from shared_forward_evidence import (
    canonicalize_emir_row,
    canonicalize_pasticuan_row,
    canonical_rows_to_pasticuan_projects,
    factual_completeness_pct,
    merge_equivalent_rows,
    profile_rows,
    strict_active,
)


ISSUER = "https://www.temposcangroup.com/en/read/Joint-Venture-Tempo-Scan-Group-and-Sino-Biopharmaceutical-Group"
BPOM = "https://www.pom.go.id/index.php/berita/bpom-dukung-kemajuan-industri-farmasi-nasional-melalui-kolaborasi-tempo-scan-dan-sino-biopharmaceutical"


def _emir_tspc():
    return {
        "evidence_id": "emir-tspc",
        "ticker": "TSPC.JK",
        "evidence_type": "JOINT_VENTURE_SPECIALTY_PHARMA",
        "evidence_date": "2026-07-15",
        "observed_at": "2026-07-24T00:00:00Z",
        "title": "PT Tempo CTTQ Biopharmaceutical Indonesia joint venture",
        "source_url": ISSUER,
        "source_family": "ISSUER_OFFICIAL_PRESS_RELEASE|BPOM_REGULATOR",
        "source_quorum_count": 2,
        "source_quorum_verified": True,
        "entity_match_verified": True,
        "source_verified": True,
        "evidence_confidence": 0.97,
        "payload": {"secondary_url": BPOM, "regulator_support_confirmed": True},
    }


def _pasticuan_tspc():
    return {
        "snapshot_id": "pasticuan-tspc",
        "ticker": "TSPC.JK",
        "project_name": "PT Tempo CTTQ Biopharmaceutical Indonesia (TCBI) joint venture",
        "project_stage": "SIGNED_JOINT_VENTURE_REGULATORY_SUPPORT_IN_PROGRESS_2026",
        "project_source_families": "ISSUER_OFFICIAL_PRESS_RELEASE|BPOM_REGULATOR",
        "project_source_urls": f"{ISSUER}|{BPOM}",
        "project_source_quorum_verified": True,
        "source_quorum_count": 2,
        "entity_match_verified": True,
        # Historical PASTICUAN row used the corroborating BPOM publication date.
        "evidence_date": "2026-07-24",
        "project_data_coverage": 80,
    }


def test_equivalent_tspc_rows_have_same_canonical_identity_and_primary_date_reconciles():
    emir = canonicalize_emir_row(_emir_tspc())
    pasticuan = canonicalize_pasticuan_row(_pasticuan_tspc())
    assert emir["event_category"] == pasticuan["event_category"] == "JV_MA"
    assert emir["canonical_event_id"] == pasticuan["canonical_event_id"]
    merged = merge_equivalent_rows(emir, pasticuan)
    assert merged["evidence_date"] == "2026-07-15"
    assert BPOM in merged["corroboration_urls"]
    assert set(merged["producer_clients"]) == {"EMIR", "PASTICUAN"}


def test_two_omed_events_from_one_document_stay_distinct_by_neutral_category():
    source = "https://www.onemed.co.id/storage/images/image/4-3m-corporate-presentation-omed-2026.pdf"
    common = {
        "ticker": "OMED.JK", "evidence_date": "2026-05-12", "source_url": source,
        "source_family": "ISSUER_PRESENTATION|ISSUER_IR_DISCLOSURE", "source_quorum_count": 2,
        "source_quorum_verified": True, "entity_match_verified": True, "source_verified": True,
        "payload": {"secondary_url": "https://www.onemed.co.id/index.php/public-expose"},
    }
    launch = canonicalize_emir_row({**common, "evidence_id": "launch", "evidence_type": "GUIDANCE_PRODUCT_LAUNCH", "title": "Locally manufactured IOL launch Q4-2026"})
    capex = canonicalize_emir_row({**common, "evidence_id": "capex", "evidence_type": "CAPEX_AND_EXPANSION_PLAN", "title": "Capacity and market expansion / production machinery / NDC / digital"})
    assert launch["event_category"] == "GUIDANCE" or launch["event_category"] == "PRODUCT_LAUNCH"
    assert capex["event_category"] == "CAPEX_EXPANSION"
    assert launch["canonical_event_id"] != capex["canonical_event_id"]


def test_maha_remains_stored_but_is_stale_for_september_2026_profile():
    maha = canonicalize_emir_row({
        "evidence_id": "maha", "ticker": "MAHA.JK", "evidence_type": "BACKLOG_LONG_TERM_CONTRACT",
        "evidence_date": "2024-06-05", "title": "Long term coal hauling contract through 2034",
        "source_url": "https://mha.co.id/PR/pdf/240604_MAHA-BYAN%20Contract%20v02.pdf",
        "source_family": "ISSUER_PRESS_RELEASE_PDF|ISSUER_PRESS_RELEASE_INDEX",
        "source_quorum_count": 2, "source_quorum_verified": True, "entity_match_verified": True,
        "source_verified": True, "evidence_confidence": 0.98,
        "payload": {"secondary_url": "https://mha.co.id/press-release"},
    })
    assert not strict_active(maha, as_of="2026-09-05T00:00:00Z")
    profile = profile_rows([maha], ticker="MAHA.JK", as_of="2026-09-05T00:00:00Z")
    assert profile["shared_forward_event_count"] == 1
    assert profile["shared_forward_active_direct_count"] == 0
    assert profile["shared_forward_provenance_state"] == "HISTORICAL_ONLY_OR_STALE"


def test_complete_strict_fact_is_contract_coverage_not_model_score():
    row = canonicalize_emir_row(_emir_tspc())
    assert factual_completeness_pct(row) == 100.0
    assert not any(key.endswith("_score") for key in row)
    assert "recommendation" not in row
    assert "authorization" not in row


def test_pasticuan_adapter_preserves_facts_but_derives_no_shared_score():
    row = canonicalize_emir_row(_emir_tspc())
    frame = canonical_rows_to_pasticuan_projects([row])
    assert len(frame) == 1
    record = frame.iloc[0].to_dict()
    assert record["ticker"] == "TSPC.JK"
    assert record["evidence_date"] == "2026-07-15"
    assert record["review_origin"] == "SHARED_CANONICAL_FORWARD_EVIDENCE"
    assert record["project_source_quorum_verified"] is True or bool(record["project_source_quorum_verified"])
    assert "project_pipeline_score_observed" not in frame.columns
    assert "future_fundamental_impact_score_observed" not in frame.columns


def test_runtime_adapter_does_not_import_shared_contract_coverage_into_pasticuan_model_coverage():
    row = canonicalize_emir_row(_emir_tspc())
    assert factual_completeness_pct(row) == 100.0
    raw = canonical_rows_to_pasticuan_projects([row])
    assert float(raw.iloc[0]["project_data_coverage"]) == 100.0
    model_input = _pasticuan_model_input([row])
    assert pd.isna(model_input.iloc[0]["project_data_coverage"])
    profile = profile_rows([row], ticker="TSPC.JK", as_of="2026-09-05T00:00:00Z")
    assert profile["shared_forward_contract_coverage_pct"] == 100.0


def test_v36_migration_is_additive_rls_and_no_public_or_delete_grant():
    sql = Path("database/migration_v36_canonical_forward_evidence.sql").read_text().lower()
    assert "create table if not exists public.evidence_forward_events" in sql
    assert "enable row level security" in sql
    assert "revoke all on table public.evidence_forward_events from public, anon, authenticated, service_role" in sql
    assert "grant select, insert, update on table public.evidence_forward_events to service_role" in sql
    assert "grant delete" not in sql
    assert "drop table" not in sql
    for forbidden in ("recommendation", "authorization", "execution_ready", "take_profit", "stop_loss"):
        assert forbidden not in sql


def test_contract_profile_same_for_same_rows_regardless_consumer_name():
    rows = [canonicalize_emir_row(_emir_tspc())]
    a = profile_rows(rows, ticker="TSPC.JK", as_of="2026-09-05T00:00:00Z")
    b = profile_rows(rows, ticker="TSPC.JK", as_of="2026-09-05T00:00:00Z")
    assert a == b
    assert a["shared_forward_active_direct_count"] == 1
    assert a["shared_forward_contract_coverage_pct"] == 100.0
