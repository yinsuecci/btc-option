import numpy as np
import pandas as pd
import pytest
from config import Config
from src.core import *
from src.data import synthetic
from src.research import observations,run

@pytest.fixture(scope='module')
def data(): return synthetic(weeks=8)

@pytest.mark.parametrize('date,offset',[('2025-01-04T05:00Z',-5),('2025-07-05T04:00Z',-4),('2025-03-09T06:59Z',-5),('2025-03-09T07:00Z',-4)])
def test_timezone(date,offset): assert utc(date).tz_convert(ET).utcoffset().total_seconds()/3600==offset

def test_saturday():
    assert saturday('2025-01-05T04:59Z')
    assert not saturday('2025-01-05T05:00Z')
    assert weekend_id('2025-01-05T04:59Z')=='2025-01-04'

def test_naive_rejected():
    with pytest.raises(ValueError): utc('2025-01-04')

@pytest.mark.parametrize('s,k,kind,expected',[(100,90,'CALL',10),(100,110,'PUT',10),(100,110,'CALL',0),(100,90,'PUT',0)])
def test_payoff(s,k,kind,expected): assert payoff(s,k,kind)==expected

def test_no_lookahead(data):
    spot,chain,_=data; t=chain.timestamp.min(); a,s=market_factors(spot,t)
    altered=spot.copy(); altered.loc[altered.timestamp>t,['spot','volume']]=1e12
    assert market_factors(altered,t)==(a,s)
    rows,_=candidates(chain,spot,t)
    future=chain.copy(); future.loc[future.timestamp>t,'bid']=1e12
    b,_=candidates(future,spot,t); pd.testing.assert_frame_equal(rows,b)

def test_expiry_quote_and_costs(data):
    spot,chain,_=data; t=chain.timestamp.min(); rows,_=candidates(chain,spot,t); r=rows.iloc[0]
    assert r.hours_to_expiry==(r.expiry-t).total_seconds()/3600
    y,settle,fee,slip=label(r,spot); assert np.isclose(y,r.bid-settle-fee-slip)
    assert y<=r.bid-settle and fee>=0 and slip>0
    bad=chain.copy(); bad['bid']=bad.ask
    assert candidates(bad,spot,t)[0].empty
    bad=chain.copy(); bad['bid']=np.nan
    assert candidates(bad,spot,t)[0].empty
    assert candidates(chain,spot,t+pd.Timedelta(minutes=3))[0].empty

def test_walk_forward_beta_alpha(data):
    spot,chain,_=data; obs,_=observations(spot,chain,Config()); wid=sorted(obs.weekend_id.unique())[-1]; t=obs.loc[obs.weekend_id==wid].decision_time.min()
    for kind in ['ols','ridge']:
        model=FactorModel(kind).fit(obs,t); assert max(model.train_ids)<wid
        rows=obs.loc[obs.weekend_id==wid]; assert np.allclose(model.predict(rows),model.intercept+rows[FEATURES].to_numpy()@model.beta)

def test_optimizer(data):
    spot,chain,_=data; rows,_=candidates(chain,spot,chain.timestamp.min()); cfg=Config(); opt=optimize(rows,np.ones(len(rows))*100,cfg)
    assert opt is not None; w=opt['weights']; assert (w<=0).all(); assert abs(w).sum()<=cfg.gross_limit+1e-6; assert max(abs(w))<=cfg.max_position+1e-6
    assert np.allclose(opt['greeks'],w@rows[['delta','gamma','vega','theta']].to_numpy())
    assert abs(opt['greeks'][0])<=cfg.delta_limit+1e-6; assert abs(opt['greeks'][1])<=cfg.gamma_limit+1e-6; assert abs(opt['greeks'][2])<=cfg.vega_limit+1e-6
    assert min(opt['constraint_slacks'])>=-1e-6
    assert optimize(rows,-np.ones(len(rows))*100,cfg)['weights'].sum()>-1e-6
    assert optimize(rows.iloc[:0],np.array([])) is None

def test_pipeline(tmp_path,data):
    spot,chain,mode=data; tables,m=run(spot,chain,mode,Config(min_train_weekends=3),tmp_path)
    entries=tables['decisions'].query("action == 'ENTER'"); assert not entries.weekend_id.duplicated().any()
    assert all(saturday(t) for t in entries.decision_time)
    assert m['data_mode']=='SYNTHETIC'; assert (tmp_path/'report.html').exists()
    assert 'THIS IS NOT A REAL HISTORICAL PERFORMANCE RESULT' in (tmp_path/'report.html').read_text(encoding='utf-8')

def test_threshold_no_entry(tmp_path,data):
    spot,chain,mode=data
    tables,m=run(spot,chain,mode,Config(min_train_weekends=3,threshold=1e9),tmp_path)
    assert m['traded_weekends']==0 and m['net_pnl']==0
    assert tables['trades'].empty

def test_missing_settlement(data):
    spot,chain,_=data; rows,_=candidates(chain,spot,chain.timestamp.min()); r=rows.iloc[0]
    with pytest.raises(ValueError): label(r,spot.loc[spot.timestamp!=r.expiry])
    obs,audit=observations(spot.loc[spot.timestamp!=r.expiry],chain,Config())
    assert r.symbol in obs.symbol.to_numpy()
    assert obs.loc[obs.symbol==r.symbol,'y'].isna().all()

def test_future_labels_excluded(data):
    spot,chain,_=data; obs,_=observations(spot,chain,Config()); t=obs.decision_time.max()
    matured=obs.loc[(obs.weekend_id<weekend_id(t))&(obs.label_available_at<t)]
    m=FactorModel().fit(obs,t); n=FactorModel().fit(matured,t)
    assert np.allclose(m.beta,n.beta)
