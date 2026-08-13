import pandas as pd
from decision_overlay import apply_execution_plan_integrity


def test_stale_swing_cannot_remain_production_eligible():
    frame=pd.DataFrame([{
        'ticker':'DGWG.JK','status':'ENTRY_PLAN_READY','ranking_score':68.7,
        'last_price':332.0,'entry_low':286.0,'entry_high':296.0,'trigger':294.0,
        'stop_loss':270.0,'tp1':306.0,'tp2':326.0,'entry_zone_is_executable':True,
        'production_rank_eligible':True,'actionable_rank_eligible':True,
        'production_gate_reason':'',
    }])
    out=apply_execution_plan_integrity(frame,model='SWING_READY').iloc[0]
    assert out['ranking_score']==68.7
    assert out['execution_plan_integrity_state']=='STALE_TARGET_ALREADY_REACHED'
    assert not bool(out['production_rank_eligible'])
    assert not bool(out['actionable_rank_eligible'])
    assert out['status']=='WATCHLIST'
    assert 'EXECUTION_PLAN_STALE_OR_INVALID' in str(out['production_gate_reason'])
