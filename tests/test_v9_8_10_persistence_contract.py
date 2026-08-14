import pandas as pd

import persistence_contract_patch as patch
from scanner_database import TABLE_FIELD_TYPES, _normalise_record


def test_multibagger_v9_component_fields_are_whitelisted():
    spec = TABLE_FIELD_TYPES["multibagger_snapshots"]
    assert patch.V9_NUMERIC_FIELDS <= spec["numeric"]
    assert patch.V9_TEXT_FIELDS <= spec["text"]
    assert patch.V9_INTEGER_FIELDS <= spec["integer"]


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
        "model_version": "9.8.10",
        "schema_version": "scanner_schema_v16",
    }]

    enriched = patch.enrich_multibagger_payload_records(records, source)
    normalised = _normalise_record("multibagger_snapshots", enriched[0])

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
    records = [{"ticker": "TEST.JK"}]

    enriched = patch.enrich_multibagger_payload_records(records, source)[0]

    assert enriched["future_fundamental_score"] is None
    assert enriched["future_fundamental_coverage_pct"] == 0.0
    assert enriched["score_coverage_pct"] == 42.0
