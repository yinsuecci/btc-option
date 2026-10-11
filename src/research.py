from pathlib import Path
import json
import numpy as np
import pandas as pd
from config import Config
from src.core import FEATURES,ET,candidates,label,FactorModel,optimize,saturday,weekend_id,bs

def observations(spot,chain,cfg):
    samples=[]; audit=[]
    for t in sorted(chain.timestamp.unique()):
        if not saturday(t) or t.tz_convert(ET).hour%cfg.decision_frequency: continue
        rows,rejected=candidates(chain,spot,t,cfg); audit.extend(rejected)
        for _,r in rows.iterrows():
            try: y,settlement,fees,slip=label(r,spot,cfg)
            except ValueError as error:
                audit.append(dict(time=str(t),symbol=r.symbol,reason=str(error)))
                # Keep point-in-time candidates even if a future outcome is missing.
                y=settlement=fees=slip=np.nan
            d=r.to_dict(); d.update(y=y,settlement=settlement,fees=fees,slippage=slip,label_available_at=r.expiry)
            samples.append(d)
    if not samples: raise ValueError('No valid labeled observations')
    return pd.DataFrame(samples),pd.DataFrame(audit)

def run(spot,chain,mode,cfg=Config(),output='results'):
    p=Path(output); p.mkdir(parents=True,exist_ok=True)
    obs,audit=observations(spot,chain,cfg); audit.to_csv(p/'data_audit.csv',index=False)
    betas=[]; predictions=[]; decompositions=[]; trades=[]; decisions=[]; weekends=[]; stresses=[]
    ids=sorted({weekend_id(t) for t in chain.timestamp.unique() if saturday(t)})
    for wid in ids[cfg.min_train_weekends:]:
        test=obs.loc[obs.weekend_id==wid]; first=test.decision_time.min()
        if test.empty:
            weekends.append(dict(weekend_id=wid,net_pnl=0.,traded=False,entry_hour=np.nan))
            decisions.append(dict(weekend_id=wid,action='NO TRADE',reason='no valid labeled eligible candidates'))
            continue
        train=obs.loc[(obs.weekend_id<wid)&(obs.label_available_at<first)&obs.y.notna()]
        if cfg.rolling_weekends: train=train.loc[train.weekend_id.isin(sorted(train.weekend_id.unique())[-cfg.rolling_weekends:])]
        if train.weekend_id.nunique()<cfg.min_train_weekends: continue
        models={name:FactorModel(name).fit(train,first) for name in ['ols','ridge']}; model=models['ridge']
        for name,m in models.items():
            for f,b in zip(['intercept']+FEATURES,[m.intercept]+list(m.beta)):
                betas.append(dict(weekend_id=wid,model=name,factor_name=f,beta=b,standard_error=np.nan,t_stat=np.nan,training_sample_size=m.n,training_start=m.train_ids[0],training_end=m.train_ids[-1]))
        entered=False; weekend_pnl=0.; entry_hour=np.nan
        for t,rows in test.groupby('decision_time',sort=True):
            if entered: break
            alpha=model.predict(rows); opt=optimize(rows,alpha,cfg)
            for idx,(_,r) in enumerate(rows.iterrows()):
                predictions.append(dict(weekend_id=wid,decision_time=t,symbol=r.symbol,strike=r.strike,option_type=r.option_type,expiry=r.expiry,alpha_hat=alpha[idx],residual_sigma=model.sigma))
                decompositions.append(dict(weekend_id=wid,decision_time=t,symbol=r.symbol,intercept=model.intercept,alpha_hat=alpha[idx],**{f:r[f]*b for f,b in zip(FEATURES,model.beta)}))
            enter=opt is not None and opt['score']>=cfg.threshold and np.sum(abs(opt['weights']))>1e-5
            decisions.append(dict(weekend_id=wid,decision_time=t,decision_time_et=t.tz_convert(ET),action='ENTER' if enter else 'NO TRADE',entry_score=opt['score'] if opt else np.nan,expected_net_pnl=opt['expected'] if opt else np.nan,risk=opt['risk'] if opt else np.nan,constraint_slacks=json.dumps(dict(zip(['gross','capital','delta','gamma','vega'],opt['constraint_slacks']))) if opt else 'optimizer unavailable'))
            if not enter: continue
            # Round down to executable contract lots, then revalidate all portfolio limits.
            q=np.floor(-opt['weights']/rows.min_qty.to_numpy()+1e-9)*rows.min_qty.to_numpy()
            greeks=-q@rows[['delta','gamma','vega','theta']].to_numpy()
            if not q.sum() or any(abs(greeks[j])>limit+1e-6 for j,limit in enumerate([cfg.delta_limit,cfg.gamma_limit,cfg.vega_limit])):
                decisions[-1]['action']='NO TRADE'
                decisions[-1]['reason']='lot rounding violates limits or zero quantity'
                continue
            score=float(alpha@q)/max(float(np.linalg.norm(opt['risk_per_unit']*q)),1e-9)
            decisions[-1]['entry_score']=score
            decisions[-1]['expected_net_pnl']=float(alpha@q)
            decisions[-1]['risk']=float(np.linalg.norm(opt['risk_per_unit']*q))
            if score<cfg.threshold:
                decisions[-1]['action']='NO TRADE'; decisions[-1]['reason']='rounded portfolio below threshold'; continue
            opt['score']=score
            entered=True; entry_hour=t.tz_convert(ET).hour
            for idx,(_,r) in enumerate(rows.iterrows()):
                if q[idx]<=0: continue
                if pd.isna(r.y): raise ValueError(f'Missing settlement for entered {r.symbol}; refusing incomplete PnL')
                gross=q[idx]*(r.bid-r.settlement); net=q[idx]*r.y; weekend_pnl+=net
                trades.append(dict(r.to_dict(),entry_time_utc=t,entry_time_et=t.tz_convert(ET),decision_time_utc=t,decision_time_et=t.tz_convert(ET),expiry_time=r.expiry,alpha_hat=alpha[idx],weight=-q[idx],quantity=q[idx],portfolio_delta=greeks[0],portfolio_gamma=greeks[1],portfolio_vega=greeks[2],portfolio_theta=greeks[3],entry_score=opt['score'],gross_pnl=gross,fees=q[idx]*r.fees,spread_cost=q[idx]*(r.ask-r.bid)/2,slippage=q[idx]*r.slippage,net_pnl=net,holding_hours=r.hours_to_expiry))
            for shock in [-.3,-.2,-.1,-.05,0,.05,.1,.2,.3]:
                for ivshock in [0,.10,.25,.50]:
                    stressed=np.array([bs(r.spot*(1+shock),r.strike,r.hours_to_expiry,r.mark_iv+ivshock,r.option_type)[0] for r in rows.itertuples()])
                    stresses.append(dict(weekend_id=wid,spot_shock=shock,iv_shock=ivshock,estimated_pnl=float(q@(rows.bid.to_numpy()-stressed)-q@(rows.fees.to_numpy()+rows.slippage.to_numpy()))))
        weekends.append(dict(weekend_id=wid,net_pnl=weekend_pnl,traded=entered,entry_hour=entry_hour))
    tables=dict(beta_history=pd.DataFrame(betas),predictions=pd.DataFrame(predictions),prediction_decomposition=pd.DataFrame(decompositions),trades=pd.DataFrame(trades),decisions=pd.DataFrame(decisions),weekends=pd.DataFrame(weekends),stress=pd.DataFrame(stresses))
    for name,frame in tables.items(): frame.to_csv(p/f'{name}.csv',index=False)
    summary=obs[FEATURES].agg(['mean','std','min','max']).T
    summary['correlation_with_future_pnl']=obs[FEATURES].corrwith(obs.y)
    summary.index.name='factor'; summary.to_csv(p/'factor_summary.csv')
    from src.nautilus_adapter import replay
    engine_result=replay(tables['trades'],cfg,p)
    from src.reporting import report
    metrics=report(tables,obs,mode,p,engine_result)
    return tables,metrics
