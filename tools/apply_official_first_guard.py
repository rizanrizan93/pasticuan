from pathlib import Path

path = Path("resumable_app_engine.py")
text = path.read_text(encoding="utf-8")

import_anchor = "from issuer_classification import normalize_fundamental_classification\n"
import_line = "from official_evidence_guard import canonicalize_official_fundamental_evidence\n"
if import_line not in text:
    if import_anchor not in text:
        raise SystemExit("import anchor not found")
    text = text.replace(import_anchor, import_anchor + import_line, 1)

old_load = "snapshot = normalize_fundamental_classification(\n        _mark_history_eligible(enrich_fundamentals_with_history(snapshot, history))\n    )"
new_load = "snapshot = normalize_fundamental_classification(\n        _mark_history_eligible(\n            canonicalize_official_fundamental_evidence(\n                enrich_fundamentals_with_history(snapshot, history)\n            )\n        )\n    )"
if old_load in text:
    text = text.replace(old_load, new_load, 1)
elif "canonicalize_official_fundamental_evidence(\n                enrich_fundamentals_with_history(snapshot, history)" not in text:
    raise SystemExit("load canonicalisation anchor not found")

old_refresh = "enrich_fundamentals_with_history(_coalesce_primary_evidence(live_snapshot, fundamentals), history)"
new_refresh = "canonicalize_official_fundamental_evidence(\n                    enrich_fundamentals_with_history(_coalesce_primary_evidence(live_snapshot, fundamentals), history)\n                )"
if old_refresh in text:
    text = text.replace(old_refresh, new_refresh, 1)
elif "canonicalize_official_fundamental_evidence(\n                    enrich_fundamentals_with_history(_coalesce_primary_evidence(live_snapshot, fundamentals), history)" not in text:
    raise SystemExit("refresh canonicalisation anchor not found")

path.write_text(text, encoding="utf-8")
