import pandas as pd

from release_contract import SCANNER_RELEASE_VERSION
from resumable_app_engine import (
    ENGINE_VERSION,
    FULL_FINAL_PERSISTENCE_TABLES,
    LEAN_FINAL_PERSISTENCE_TABLES,
)
from scanner_database import (
    DatabaseSettings,
    ScannerDatabaseBridge,
    TABLE_FIELD_TYPES,
    V9_RANKING_INTEGER_FIELDS,
    V9_RANKING_NUMERIC_FIELDS,
    V9_RANKING_TEXT_FIELDS,
    _normalise_record,
)


def test_multibagger_v9_component_fields_are_whitelisted():
    spec = TABLE_FIELD_TYPES["multibagger_snapshots"]
    assert V9_RANKING_NUMERIC_FIELDS <= spec["numeric"]
    assert V9_RANKING_TEXT_FIELDS <= spec["text"]
    assert V9_RANKING_INTEGER_FIELDS <= spec["integer"]


def test_runtime_and_persistence_share_one_release_contract():
    assert ENGINE_VERSION == SCANNER_RELEASE_VERSION
    assert "fundamental_snapshots" in LEAN_FINAL_PERSISTENCE_TABLES
    assert "fundamental_snapshots" in FULL_FINAL_PERSISTENCE_TABLES


def test_v9_components_survive_payload_enrichment_and_normalisation():
    source = pd.DataFrame([{
        "ticker": "ELSA.JK",
        "ranking_score": 71.2,
        "research_score": 72.1,
        "ranking_score_state": "PRODUCTION_SCORE",
        "score_coverage_pct": 84.5,
        "business_quality_score": 78.0,
        "business_quality_coverage_pct": 90.0,
        "future_fundamental_score": 69.0,
        "future_fundamental_coverage_pct": 80.0,
        "valuation_mos_score": 66.0,
        "valuation_mos_coverage_pct": 75.0,
        "management_capital_score": 73.0,
        "management_capital_coverage_pct": 70.0,
        "issuer_macro_alignment_score": 81.0,
        "issuer_macro_alignment_coverage_pct": 88.0,
        "narrative_flow_score": 76.0,
        "narrative_flow_coverage_pct": 82.0,
        "technical_readiness_score": 74.0,
        "technical_readiness_coverage_pct": 100.0,
        "real_money_risk_lots_cap": 12,
    }])
    records = [{
        "snapshot_id": "snap-elsa",
        "scan_id": "scan-1",
        "ticker": "ELSA.JK",
        "as_of": "2026-08-14T00:00:00+00:00",
        "model_version": SCANNER_RELEASE_VERSION,
        "schema_version": "scanner_schema_v16",
    }]

    normalised = _normalise_record("multibagger_snapshots", {**records[0], **source.iloc[0].to_dict()})

    assert normalised is not None
    assert normalised["score_coverage_pct"] == 84.5
    assert normalised["business_quality_score"] == 78.0
    assert normalised["future_fundamental_score"] == 69.0
    assert normalised["valuation_mos_score"] == 66.0
    assert normalised["management_capital_score"] == 73.0
    assert normalised["issuer_macro_alignment_score"] == 81.0
    assert normalised["narrative_flow_score"] == 76.0
    assert normalised["technical_readiness_score"] == 74.0
    assert normalised["ranking_score"] == 71.2
    assert normalised["research_score"] == 72.1
    assert normalised["ranking_score_state"] == "PRODUCTION_SCORE"
    assert normalised["real_money_risk_lots_cap"] == 12


def test_missing_score_stays_missing_not_neutral_default():
    source = pd.DataFrame([{
        "ticker": "TEST.JK",
        "score_coverage_pct": 42.0,
        "future_fundamental_score": float("nan"),
        "future_fundamental_coverage_pct": 0.0,
    }])
    enriched = _normalise_record("multibagger_snapshots", {
        "snapshot_id": "snap-test", "scan_id": "scan-test", "ticker": "TEST.JK",
        "as_of": "2026-08-15T00:00:00+00:00", "model_version": SCANNER_RELEASE_VERSION,
        "schema_version": "scanner_schema_v16", **source.iloc[0].to_dict(),
    })

    assert enriched is not None
    assert enriched["future_fundamental_score"] is None
    assert enriched["future_fundamental_coverage_pct"] == 0.0
    assert enriched["score_coverage_pct"] == 42.0


def test_build_payloads_persists_fundamental_and_component_lineage_under_release_version():
    bridge = ScannerDatabaseBridge(DatabaseSettings())
    result = {
        "scan_id": "scan-release-lineage",
        "scanner_version": SCANNER_RELEASE_VERSION,
        "ticker_count": 1,
        "fundamentals": pd.DataFrame([{
            "ticker": "ELSA.JK", "period_end": "2026-06-30",
            "fundamental_score": 72.0, "fundamental_coverage": 91.0,
            "fundamental_official_verified": True,
        }]),
        "focus_screens": {"multibagger": pd.DataFrame([{
            "ticker": "ELSA.JK", "ranking_score": 71.2,
            "research_score": 72.1, "ranking_score_state": "PRODUCTION_SCORE",
            "score_coverage_pct": 91.5, "business_quality_score": 78.0,
            "business_quality_coverage_pct": 93.0,
        }])},
    }

    payloads = bridge.build_payloads(result)

    assert len(payloads["fundamental_snapshots"]) == 1
    assert payloads["fundamental_snapshots"][0]["scan_id"] == "scan-release-lineage"
    assert payloads["fundamental_snapshots"][0]["model_version"] == SCANNER_RELEASE_VERSION
    assert len(payloads["multibagger_snapshots"]) == 1
    assert payloads["multibagger_snapshots"][0]["ranking_score"] == 71.2
    assert payloads["multibagger_snapshots"][0]["score_coverage_pct"] == 91.5
