from __future__ import annotations

import numpy as np

from simple_focus import (
    SIMPLE_FOCUS_VERSION,
    _flow_score_from_row,
    _macro_component,
    _narrative_flow_component,
    _positive_cash_score,
)


def main() -> None:
    assert SIMPLE_FOCUS_VERSION == "9.8.4-calibration-integrity"

    score, coverage, _ = _macro_component({"mac_issuer_macro_alignment_score": 80.0})
    assert score == 80.0 and coverage == 0.0, (score, coverage)

    score, coverage = _flow_score_from_row({"flow_silent_accumulation_score": 80.0})
    assert score == 80.0 and coverage == 0.0, (score, coverage)

    score, coverage, _ = _narrative_flow_component({
        "nar_narrative_event_effective_score": 80.0,
        "flow_silent_accumulation_score": 80.0,
    })
    assert score == 80.0 and coverage == 0.0, (score, coverage)

    score, coverage, _ = _narrative_flow_component({
        "nar_narrative_event_effective_score": 80.0,
        "nar_narrative_event_coverage_pct": 100.0,
        "flow_silent_accumulation_score": 80.0,
        "flow_silent_accumulation_confidence": 50.0,
    })
    assert score == 80.0 and abs(coverage - 80.0) < 1e-9, (score, coverage)

    # Absolute rupiah magnitude must not change business-quality cash sign score.
    assert _positive_cash_score(10_000_000.0) == _positive_cash_score(10_000_000_000_000.0) == 82.0
    assert _positive_cash_score(0.0) == 45.0
    assert _positive_cash_score(-1.0) == 15.0
    assert np.isnan(_positive_cash_score(np.nan))

    print("PASS v9.8.4 calibration integrity")


if __name__ == "__main__":
    main()
