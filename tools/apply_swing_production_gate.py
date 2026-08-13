from pathlib import Path


def rep(path, old, new):
    p=Path(path); text=p.read_text(encoding='utf-8')
    if new in text and old not in text: return
    n=text.count(old)
    if n!=1: raise RuntimeError(f'{path}: expected one match, got {n}')
    p.write_text(text.replace(old,new,1),encoding='utf-8')

rep('app.py','APP_VERSION = "9.8.9-production-lane-integrity"','APP_VERSION = "9.8.10-swing-production-gate"')
rep('app.py','IDX Scanner v9.8.9','IDX Scanner v9.8.10')
rep('app.py','IDX Super Scanner v9.8.9 — Production Lane Integrity','IDX Super Scanner v9.8.10 — Swing Production Gate')
rep('app.py','Deployment v9.8.9 tidak lengkap.','Deployment v9.8.10 tidak lengkap.')
rep('fast_scan_engine.py','FAST_SCAN_VERSION = "9.8.9-production-lane-integrity"','FAST_SCAN_VERSION = "9.8.10-swing-production-gate"')
old='''    out["production_rank_eligible"] = (\n        pd.to_numeric(out["v9_swing_score"], errors="coerce").notna()\n        & out["status"].isin(["EXECUTION_READY", "ENTRY_PLAN_READY", "WATCHLIST", "WAIT"])\n        & out["production_gate_pass"].fillna(False).astype(bool)\n        & out["methodology_gate_pass"].fillna(False).astype(bool)\n    )\n'''
new='''    out["production_rank_eligible"] = (\n        pd.to_numeric(out["v9_swing_score"], errors="coerce").notna()\n        & out["status"].isin(["EXECUTION_READY", "ENTRY_PLAN_READY", "WATCHLIST", "WAIT"])\n        & out["production_gate_pass"].fillna(False).astype(bool)\n        & out["methodology_gate_pass"].fillna(False).astype(bool)\n        & out.get("execution_plan_is_current", pd.Series(False, index=out.index)).fillna(False).astype(bool)\n    )\n'''
rep('simple_focus.py',old,new)
print('Super v9.8.10 swing production gate applied')
