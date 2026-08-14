from pathlib import Path

path = Path("simple_focus.py")
text = path.read_text(encoding="utf-8")
old = '''def _narrative_flow_component(row: Mapping[str, Any]) -> tuple[float, float, str]:
    narrative = _first_num(row, ("nar_narrative_event_effective_score", "nar_narrative_effective_score"))
    narrative_cov = _first_num(row, ("nar_narrative_event_coverage_pct", "nar_narrative_evidence_coverage_pct"))
    flow, flow_cov = _flow_score_from_row(row)
'''
new = '''def _narrative_flow_component(row: Mapping[str, Any]) -> tuple[float, float, str]:
    # Pair score and coverage from the same evidence family.  A zero event-level
    # coverage must not shadow broad narrative evidence when the event-level score
    # itself is missing.
    event_score = _first_num(row, ("nar_narrative_event_effective_score",))
    if np.isfinite(event_score):
        narrative = event_score
        narrative_cov = _first_num(row, ("nar_narrative_event_coverage_pct",))
    else:
        narrative = _first_num(row, ("nar_narrative_effective_score",))
        narrative_cov = _first_num(row, ("nar_narrative_evidence_coverage_pct",))
    flow, flow_cov = _flow_score_from_row(row)
'''
if old in text:
    text = text.replace(old, new, 1)
elif 'Pair score and coverage from the same evidence family' not in text:
    raise SystemExit('narrative coverage anchor not found')
path.write_text(text, encoding="utf-8")
