import numpy as np

from simple_focus import _narrative_flow_component


def test_broad_narrative_score_uses_broad_evidence_coverage():
    row = {
        "nar_narrative_event_effective_score": np.nan,
        "nar_narrative_event_coverage_pct": 0.0,
        "nar_narrative_effective_score": 70.0,
        "nar_narrative_evidence_coverage_pct": 82.0,
        "flow_silent_accumulation_score": 60.0,
        "flow_silent_accumulation_confidence": 50.0,
    }
    score, coverage, evidence = _narrative_flow_component(row)
    assert round(score, 1) == 66.0
    assert round(coverage, 1) == 69.2
    assert "SOURCED_NARRATIVE" in evidence


def test_event_specific_score_keeps_event_specific_coverage():
    row = {
        "nar_narrative_event_effective_score": 80.0,
        "nar_narrative_event_coverage_pct": 40.0,
        "nar_narrative_effective_score": 70.0,
        "nar_narrative_evidence_coverage_pct": 90.0,
    }
    score, coverage, _ = _narrative_flow_component(row)
    assert round(score, 1) == 80.0
    assert round(coverage, 1) == 40.0
