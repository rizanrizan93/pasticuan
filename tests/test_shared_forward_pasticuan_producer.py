from __future__ import annotations

import pasticuan_shared_forward_producer as producer
from shared_forward_evidence import canonicalize_emir_row

ISSUER = "https://www.temposcangroup.com/en/read/Joint-Venture-Tempo-Scan-Group-and-Sino-Biopharmaceutical-Group"
BPOM = "https://www.pom.go.id/index.php/berita/bpom-dukung-kemajuan-industri-farmasi-nasional-melalui-kolaborasi-tempo-scan-dan-sino-biopharmaceutical"


def pasticuan_tspc():
    return {
        "snapshot_id": "OFFICIAL_FORWARD|TSPC|TCBI_JV_2026",
        "ticker": "TSPC.JK",
        "project_name": "PT Tempo CTTQ Biopharmaceutical Indonesia (TCBI) joint venture",
        "project_stage": "SIGNED_JOINT_VENTURE_REGULATORY_SUPPORT_IN_PROGRESS_2026",
        "project_source_families": "ISSUER_OFFICIAL_PRESS_RELEASE|BPOM_REGULATOR",
        "project_source_urls": f"{ISSUER}|{BPOM}",
        "project_source_quorum_verified": True,
        "source_quorum_count": 2,
        "entity_match_verified": True,
        "event_date": "2026-07-15",
        "evidence_date": "2026-07-24",
        "project_execution_flags": "SIGNED_JV;SPECIALTY_PHARMA",
        "review_origin": "OFFICIAL_FORWARD_RESEARCH",
        "project_capex_idr": None,
    }


def emir_tspc():
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


def test_strict_filter_rejects_research_only_or_single_source():
    good = pasticuan_tspc()
    assert producer._strict_source_row(good)
    assert not producer._strict_source_row({**good, "project_source_quorum_verified": False})
    assert not producer._strict_source_row({**good, "source_quorum_count": 1})
    assert not producer._strict_source_row({**good, "entity_match_verified": False})
    assert not producer._strict_source_row({**good, "project_source_urls": "http://example.com/not-https"})


def test_factual_only_strips_pasticuan_interpretation_payload():
    row = producer._factual_only(pasticuan_tspc())
    payload = row["payload"]
    assert "project_stage" not in payload
    assert "project_execution_flags" not in payload
    assert "review_origin" not in payload
    assert not any(key.endswith("_score") for key in row)
    assert "recommendation" not in row
    assert "authorization" not in row


def test_publish_reconciles_existing_emir_provenance(monkeypatch):
    local_source = pasticuan_tspc()
    existing = canonicalize_emir_row(emir_tspc())

    class FakeConfig:
        ready = True
        def status(self):
            return {"state": "READY"}

    class FakeBackend:
        def __init__(self, config):
            pass
        def read_rows(self, table, filters, limit=50000):
            assert table == "project_events"
            return [local_source]

    captured = {}

    def fake_read(tickers, *, client_id):
        assert client_id == "PASTICUAN"
        return [existing], {"state": "SHARED_CANONICAL_FORWARD", "rows": 1}

    def fake_upsert(rows, *, client_id):
        captured["rows"] = rows
        captured["client_id"] = client_id
        return rows, {"state": "UPSERTED", "rows": len(rows)}

    monkeypatch.setattr(producer.HubConfig, "from_environment", classmethod(lambda cls, client_id: FakeConfig()))
    monkeypatch.setattr(producer, "SupabaseEvidenceBackend", FakeBackend)
    monkeypatch.setattr(producer, "read_canonical_forward_rows", fake_read)
    monkeypatch.setattr(producer, "upsert_canonical_forward_rows", fake_upsert)

    audit = producer.publish_project_events()
    assert audit["state"] == "UPSERTED"
    assert captured["client_id"] == "PASTICUAN"
    assert len(captured["rows"]) == 1
    row = captured["rows"][0]
    assert row["evidence_date"] == "2026-07-15"
    assert set(row["producer_clients"]) == {"EMIR", "PASTICUAN"}
    assert BPOM in row["corroboration_urls"]


def test_publish_fails_closed_if_canonical_reconcile_read_unavailable(monkeypatch):
    class FakeConfig:
        ready = True
        def status(self):
            return {"state": "READY"}

    class FakeBackend:
        def __init__(self, config):
            pass
        def read_rows(self, table, filters, limit=50000):
            return [pasticuan_tspc()]

    monkeypatch.setattr(producer.HubConfig, "from_environment", classmethod(lambda cls, client_id: FakeConfig()))
    monkeypatch.setattr(producer, "SupabaseEvidenceBackend", FakeBackend)
    monkeypatch.setattr(
        producer,
        "read_canonical_forward_rows",
        lambda tickers, *, client_id: ([], {"state": "READ_FAIL_SOFT", "error": "temporary"}),
    )

    audit = producer.publish_project_events()
    assert audit["state"] == "RECONCILE_READ_UNAVAILABLE"
    assert audit["rows"] == 0
