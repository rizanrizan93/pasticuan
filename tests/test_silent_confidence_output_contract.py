from pathlib import Path


def test_simple_focus_outputs_silent_confidence_with_silent_score():
    source = Path("simple_focus.py").read_text(encoding="utf-8")
    expected = '''"silent_accumulation_confidence": _first_num(row, ("flow_silent_accumulation_confidence", "flow_silent_accumulation_data_coverage"))'''
    assert source.count(expected) >= 2


def test_database_contract_accepts_silent_confidence():
    source = Path("scanner_database.py").read_text(encoding="utf-8")
    assert '"silent_accumulation_confidence"' in source
