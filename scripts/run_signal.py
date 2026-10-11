import argparse
import pandas as pd
from config import Config
from src.core import ET,FEATURES,FactorModel,candidates,optimize,saturday,weekend_id
from src.data import load
from src.research import observations

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data'); p.add_argument('--time'); a=p.parse_args(); cfg=Config()
    spot,chain,mode=load(a.data)
    t=pd.Timestamp(a.time) if a.time else (pd.Timestamp.now(tz='UTC') if mode=='REAL' else chain.timestamp.max().normalize()+pd.Timedelta(hours=5))
    if mode=='SYNTHETIC': t=chain.timestamp.max().tz_convert(ET).normalize().tz_convert('UTC') if not a.time else t
    print('DATA_MODE =',mode); print('Current ET Time:',t.tz_convert(ET))
    rows,_=candidates(chain,spot,t,cfg)
    if not saturday(t) or rows.empty: print('NO TRADE: not Saturday ET or no eligible quote'); return
    obs,_=observations(spot,chain,cfg)
    matured=obs.loc[(obs.weekend_id<weekend_id(t))&(obs.label_available_at<t)&obs.y.notna()]
    if matured.weekend_id.nunique()<cfg.min_train_weekends: print('NO TRADE: insufficient completed training weekends'); return
    model=FactorModel().fit(matured,t); alpha=model.predict(rows); opt=optimize(rows,alpha,cfg)
    print('Spot:',rows.spot.iloc[0]); print('Selected Expiry:',rows.expiry.iloc[0]); print('Hours To Expiry:',rows.hours_to_expiry.iloc[0]); print('FACTOR STATE:',rows[FEATURES].to_string(index=False)); print('MODEL beta:',dict(zip(FEATURES,model.beta)), 'intercept:',model.intercept)
    rows['alpha_hat']=alpha; print('TOP OPTION ALPHAS:',rows[['symbol','alpha_hat']].to_string(index=False))
    if opt:
        print('OPTIMAL PORTFOLIO:',dict(zip(rows.symbol,opt['weights']))); print('Portfolio Greeks:',opt['greeks']); print('Expected Net PnL:',opt['expected'],'Risk:',opt['risk'],'Entry Score:',opt['score'])
    print('ENTER (research signal only)' if opt and opt['score']>=cfg.threshold and abs(opt['weights']).sum()>1e-5 else 'NO TRADE')
if __name__=='__main__': main()
