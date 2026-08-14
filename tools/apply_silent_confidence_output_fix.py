from pathlib import Path

path = Path("simple_focus.py")
text = path.read_text(encoding="utf-8")
old = '''            "silent_accumulation_score": _first_num(row, ("flow_silent_accumulation_score", "sig_silent_accumulation_score")),\n            "accumulation_dominance_pct": _first_num(row, ("flow_accumulation_dominance_pct",)),\n'''
new = '''            "silent_accumulation_score": _first_num(row, ("flow_silent_accumulation_score", "sig_silent_accumulation_score")),\n            "silent_accumulation_confidence": _first_num(row, ("flow_silent_accumulation_confidence", "flow_silent_accumulation_data_coverage")),\n            "accumulation_dominance_pct": _first_num(row, ("flow_accumulation_dominance_pct",)),\n'''
count = text.count(old)
if count:
    text = text.replace(old, new)
elif text.count('"silent_accumulation_confidence": _first_num(row, ("flow_silent_accumulation_confidence", "flow_silent_accumulation_data_coverage"))') < 2:
    raise SystemExit("silent confidence output anchors not found")
path.write_text(text, encoding="utf-8")
