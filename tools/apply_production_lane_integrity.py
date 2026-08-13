from pathlib import Path


def rep(path, old, new):
    p=Path(path); text=p.read_text(encoding='utf-8')
    if new in text and old not in text: return
    if text.count(old)!=1: raise RuntimeError(f'{path}: match={text.count(old)}')
    p.write_text(text.replace(old,new,1),encoding='utf-8')

rep('app.py','APP_VERSION = "9.8.8-execution-plan-integrity"','APP_VERSION = "9.8.9-production-lane-integrity"')
rep('app.py','IDX Scanner v9.8.8','IDX Scanner v9.8.9')
rep('app.py','IDX Super Scanner v9.8.8 — Execution Plan Integrity','IDX Super Scanner v9.8.9 — Production Lane Integrity')
rep('fast_scan_engine.py','FAST_SCAN_VERSION = "9.8.8-execution-plan-integrity"','FAST_SCAN_VERSION = "9.8.9-production-lane-integrity"')
old='''    if "order_ready" in out.columns:\n        out.loc[expired, "order_ready"] = False\n    if "recommended_allocation_idr" in out.columns:\n'''
new='''    if "order_ready" in out.columns:\n        out.loc[expired, "order_ready"] = False\n    if "actionable_rank_eligible" in out.columns:\n        out.loc[expired, "actionable_rank_eligible"] = False\n    if "production_rank_eligible" in out.columns:\n        out.loc[expired, "production_rank_eligible"] = False\n    if "production_gate_reason" in out.columns:\n        prior = out["production_gate_reason"].fillna("").astype(str)\n        suffix = np.where(prior.str.len().gt(0), prior + "; EXECUTION_PLAN_STALE_OR_INVALID", "EXECUTION_PLAN_STALE_OR_INVALID")\n        out.loc[expired, "production_gate_reason"] = pd.Series(suffix, index=out.index).loc[expired]\n    if "recommended_allocation_idr" in out.columns:\n'''
rep('decision_overlay.py',old,new)
print('Super v9.8.9 production-lane patch applied')
