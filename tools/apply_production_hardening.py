from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if new in text and old not in text:
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one old match or an already-patched new value, got {count}: {old!r}")
    p.write_text(text.replace(old, new, 1), encoding="utf-8")


# Production identity.
replace_once("app.py", 'APP_VERSION = "9.8.5-actionability-integrity"', 'APP_VERSION = "9.8.7-production-hardening"')
replace_once("app.py", 'st.set_page_config(page_title="IDX Scanner v9.8.5", page_icon="📊", layout="wide")', 'st.set_page_config(page_title="IDX Scanner v9.8.7", page_icon="📊", layout="wide")')
replace_once("app.py", 'st.title("IDX Super Scanner v9.8.5 — Actionability Integrity")', 'st.title("IDX Super Scanner v9.8.7 — Production Hardening")')
replace_once("fast_scan_engine.py", 'FAST_SCAN_VERSION = "9.8.5-actionability-integrity"', 'FAST_SCAN_VERSION = "9.8.7-production-hardening"')

# Expand evidence maintenance while preserving the 20-name hard ceiling in the
# finalizer. Decision review stays 12 names; the remaining capacity is a true
# rotating maintenance lane for missing/stale issuers.
for path in ("app.py", "fast_scan_engine.py"):
    replace_once(path, '"evidence_refresh_cap": 8,' if path == "app.py" else 'cfg.setdefault("evidence_refresh_cap", 8)',
                 '"evidence_refresh_cap": 20,' if path == "app.py" else 'cfg.setdefault("evidence_refresh_cap", 20)')
    replace_once(path, '"decision_evidence_cap": 8,' if path == "app.py" else 'cfg.setdefault("decision_evidence_cap", 8)',
                 '"decision_evidence_cap": 12,' if path == "app.py" else 'cfg.setdefault("decision_evidence_cap", 12)')
    replace_once(path, '"evidence_fundamental_cap": 8,' if path == "app.py" else 'cfg.setdefault("evidence_fundamental_cap", 8)',
                 '"evidence_fundamental_cap": 20,' if path == "app.py" else 'cfg.setdefault("evidence_fundamental_cap", 20)')
    replace_once(path, '"evidence_official_cap": 4,' if path == "app.py" else 'cfg.setdefault("evidence_official_cap", 4)',
                 '"evidence_official_cap": 12,' if path == "app.py" else 'cfg.setdefault("evidence_official_cap", 12)')
    replace_once(path, '"evidence_snapshot_cap": 6,' if path == "app.py" else 'cfg.setdefault("evidence_snapshot_cap", 6)',
                 '"evidence_snapshot_cap": 16,' if path == "app.py" else 'cfg.setdefault("evidence_snapshot_cap", 16)')
    replace_once(path, '"evidence_news_cap": 6,' if path == "app.py" else 'cfg.setdefault("evidence_news_cap", 6)',
                 '"evidence_news_cap": 10,' if path == "app.py" else 'cfg.setdefault("evidence_news_cap", 10)')
    replace_once(path, '"execution_verification_cap": 6,' if path == "app.py" else 'cfg.setdefault("execution_verification_cap", 6)',
                 '"execution_verification_cap": 8,' if path == "app.py" else 'cfg.setdefault("execution_verification_cap", 8)')

print("Super v9.8.7 production hardening patch verified/applied")
