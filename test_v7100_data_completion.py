import os
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import scanner
import scanner_focus
import selector_engine
from free_data_providers import yahoo_chart_direct, yahoo_fundamental_timeseries_direct
from narrative_engine import build_narrative_profiles


class FakeResponse:
    def __init__(self, payload): self.payload = payload
    def raise_for_status(self): return None
    def json(self): return self.payload


class FakeSession:
    def __init__(self, payload): self.payload = payload
    def get(self, *args, **kwargs): return FakeResponse(self.payload)


def synthetic_ohlcv(periods=280):
    idx = pd.bdate_range('2025-06-01', periods=periods)
    close = np.linspace(900, 1300, periods)
    frame = pd.DataFrame({
        'Open': close * 0.997, 'High': close * 1.01,
        'Low': close * 0.99, 'Close': close,
        'Volume': np.full(periods, 3_000_000.0),
    }, index=idx)
    return scanner.prepare_indicators(frame)


def history_fixture(ticker='TEST.JK'):
    rows=[]
    for i, period in enumerate(pd.date_range('2024-09-30', periods=8, freq='QE')):
        revenue=1_000_000_000_000*(1+0.08*i)
        rows.append({
            'ticker':ticker,'period_end':period,'period_type':'Q','statement_basis':'STANDALONE',
            'source_family':'YAHOO','source_name':'Yahoo Fundamentals Timeseries Direct','source_url':'https://query2.finance.yahoo.com',
            'currency':'IDR','revenue':revenue,'gross_profit':revenue*.35,'operating_income':revenue*.18,
            'ebit':revenue*.17,'ebitda':revenue*.21,'net_income':revenue*.12,
            'operating_cash_flow':revenue*.15,'capex':-revenue*.04,'total_assets':6e12+2e11*i,
            'total_liabilities':2.4e12+5e10*i,'equity':3.6e12+1.5e11*i,'total_debt':8e11,
            'cash':7e11+3e10*i,'shares_outstanding':1e9,'interest_expense':1e10,
            'source_verified':True,'validation_flags':'',
        })
    return pd.DataFrame(rows)


def test_yahoo_chart_direct_adjusts_split_and_preserves_event_metadata():
    ts=[int(pd.Timestamp('2026-07-30',tz='UTC').timestamp()),int(pd.Timestamp('2026-07-31',tz='UTC').timestamp())]
    payload={'chart':{'error':None,'result':[{
        'timestamp':ts,'meta':{'currency':'IDR','exchangeTimezoneName':'Asia/Jakarta','instrumentType':'EQUITY'},
        'indicators':{'quote':[{'open':[1000,510],'high':[1020,520],'low':[980,500],'close':[1000,510],'volume':[1e6,2e6]}],
                      'adjclose':[{'adjclose':[500,510]}]},
        'events':{'splits':{'x':{'date':ts[1],'numerator':2,'denominator':1,'splitRatio':'2:1'}}}
    }]}}
    frame, meta=yahoo_chart_direct('TEST.JK',session=FakeSession(payload))
    assert len(frame)==2
    assert frame.iloc[0]['Close']==500
    assert frame.attrs['corporate_action_split_dates']==['2026-07-31']
    assert meta['split_events']==1
    assert not any('corporate action' in issue.lower() for issue in scanner.ohlcv_quality_issues(frame))


def test_download_uses_direct_route_and_reaches_last_completed_session(monkeypatch):
    direct=pd.DataFrame({'Open':[100,101],'High':[102,103],'Low':[99,100],'Close':[101,102],'Volume':[1e6,1.1e6]},
                        index=pd.to_datetime(['2026-07-30','2026-07-31']))
    direct.attrs.update({'adjusted_prices':True,'corporate_action_split_dates':[]})
    fake_yf=SimpleNamespace(download=lambda *a,**k: pd.DataFrame(), Ticker=lambda *a,**k: SimpleNamespace(history=lambda **k: pd.DataFrame()))
    monkeypatch.setitem(sys.modules,'yfinance',fake_yf)
    monkeypatch.setattr('free_data_providers.yahoo_chart_direct',lambda *a,**k:(direct.copy(),{'rows':2}))
    with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{'IDX_SCANNER_CACHE_DIR':tmp}):
        histories, report=scanner._download_ohlcv_v431(['TEST.JK'],period='5y')
        assert 'TEST.JK' in histories
        assert histories['TEST.JK'].index[-1]==pd.Timestamp('2026-07-31')
        assert scanner._daily_cache_is_current(histories['TEST.JK'])
        assert report.source_tiers['TEST.JK']=='LIVE_YAHOO_DIRECT_FULL'


def test_direct_timeseries_contract_maps_all_statement_families():
    result=[]
    mapping={
        'quarterlyTotalRevenue':1000,'quarterlyNetIncome':120,'quarterlyOperatingCashFlow':150,
        'quarterlyCapitalExpenditure':-40,'quarterlyTotalAssets':6000,
        'quarterlyTotalLiabilitiesNetMinorityInterest':2400,'quarterlyStockholdersEquity':3600,
        'quarterlyTotalDebt':800,'quarterlyCashCashEquivalentsAndShortTermInvestments':700,
        'quarterlyGrossProfit':350,'quarterlyOperatingIncome':180,'quarterlyEBIT':170,
        'quarterlyEBITDA':210,'quarterlyOrdinarySharesNumber':1000,'quarterlyInterestExpense':10,
    }
    for kind,value in mapping.items():
        result.append({'meta':{'type':kind},kind:[{'asOfDate':'2026-06-30','reportedValue':{'raw':value,'currencyCode':'IDR'}}]})
    payload={'timeseries':{'error':None,'result':result}}
    frame,report=yahoo_fundamental_timeseries_direct('TEST.JK',session=FakeSession(payload))
    row=frame.iloc[0]
    assert report['status']=='OK'
    assert row['revenue']==1000
    assert row['operating_cash_flow']==150
    assert row['total_assets']==6000
    assert row['equity']==3600


def test_fundamental_history_populates_narrative_alignment_and_multibagger():
    history=scanner.normalize_fundamental_history(history_fixture())
    features=scanner.build_fundamental_history_features(history,now=pd.Timestamp('2026-08-01',tz='Asia/Jakarta'))
    aliases={'revenue_growth':'history_revenue_growth','earnings_growth':'history_earnings_growth','roe':'history_roe','roa':'history_roa',
             'operating_margin':'history_operating_margin','net_margin':'history_net_margin','debt_equity':'history_debt_equity',
             'operating_cash_flow':'history_ocf_ttm','free_cash_flow':'history_fcf_ttm'}
    for target,source in aliases.items(): features[target]=features[source]
    features['fundamental_coverage']=100.0; features['fundamental_score']=82.; features['fundamental_score_10']=8.2; features['fundamental_reliability']='HIGH'
    prepared={'TEST.JK':synthetic_ohlcv()}
    profiles=build_narrative_profiles(['TEST.JK'],prepared=prepared,events=pd.DataFrame(),outcomes=pd.DataFrame(),fundamentals=features,
                                      news_review=pd.DataFrame(),project_management=pd.DataFrame(),silent_profiles={},as_of=pd.Timestamp('2026-08-01',tz='Asia/Jakarta'))
    assert np.isfinite(profiles.loc[0,'narrative_score'])
    assert np.isfinite(profiles.loc[0,'issuer_alignment_score'])
    output=scanner_focus.scan_multibagger_candidates(prepared,features,config=scanner.ScanConfig(),silent_profiles=scanner_focus.current_silent_profiles(prepared))
    assert bool(output.loc[0,'fundamental_complete_for_multibagger'])
    assert np.isfinite(output.loc[0,'multibagger_score']) or np.isfinite(output.loc[0,'growth_compounder_score'])


def test_selector_empty_schema_fails_closed_without_crash():
    audit,fitted=selector_engine.evaluate_selector_challengers(pd.DataFrame({'ticker':['A.JK']}),selector_engine.SelectorConfig())
    assert not audit.empty
    assert audit['promotion_state'].eq('INSUFFICIENT_PANEL_SCHEMA').all()
    assert fitted=={}


def test_oos_trigger_candidate_is_recorded_when_live_gate_blocks(monkeypatch):
    frame=synthetic_ohlcv(300)
    mask=pd.Series(False,index=frame.index); mask.iloc[230]=True
    monkeypatch.setattr(scanner,'historical_signal_mask',lambda df,setup:mask)
    plan=scanner.SetupPlan(ticker='TEST.JK',setup='PULLBACK_CONTINUATION',detected=True,setup_score=72,entry=1200,entry_type='BUY_LIMIT',stop_loss=1150,tp1=1275,tp2=1325)
    monkeypatch.setitem(scanner.DETECTORS,'PULLBACK_CONTINUATION',lambda df,ticker:plan)
    monkeypatch.setattr(scanner,'_historical_gate_inputs_v300',lambda df,cfg:(['ADTV di bawah gate'],{}))
    monkeypatch.setattr(scanner,'_historical_gate_inputs',lambda df,cfg,accumulation_profile=None:(['ADTV di bawah gate'],{'silent_accumulation_score':55}))
    monkeypatch.setattr(scanner,'_historical_context',lambda df:scanner.MarketContext(regime='NEUTRAL'))
    class Engine:
        def __init__(self,cfg): pass
        def _finalize(self,*args,**kwargs): return {'status':'BLOCKED_CONTEXT','quality_score':72}
    monkeypatch.setattr(scanner,'ScanEngine',Engine)
    events=scanner.simulate_setup(frame,'TEST.JK','PULLBACK_CONTINUATION',scanner.ScanConfig())
    assert len(events)==1
    assert events[0].validation_event_tier=='TRIGGER_CANDIDATE'
    assert events[0].production_gate_pass is False
    assert 'ADTV' in events[0].production_gate_blockers
