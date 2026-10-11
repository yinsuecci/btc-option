"""Recompute every OOS decision with future labels/features removed or poisoned."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.center_vol import (CenterConfig, MEAN_FEATURES, causal_mean_residuals, fit_distribution,
                            inverse_expected_payoff, payoff_covariance)
from src.trade_research import TradeConfig, market_features, optimize_native


def main():
    p=Path('results/center_vol')
    initial_hashes={name:hashlib.sha256((p/(name+'.csv')).read_bytes()).hexdigest()
                    for name in ['forecasts','predictions']}
    formula=json.loads((p/'final_formula.json').read_text())
    cfg=CenterConfig(**formula['config']); execution=TradeConfig(**formula['execution'])
    frame=pd.read_csv(p/'price_training.csv',float_precision='round_trip')
    residuals=pd.read_csv(p/'causal_mean_residuals.csv',float_precision='round_trip')
    obs=pd.read_csv(p/'observations.csv',float_precision='round_trip')
    f=pd.read_csv(p/'forecasts.csv',float_precision='round_trip')
    pred=pd.read_csv(p/'predictions.csv',float_precision='round_trip')
    source=Path(formula.get('source_directory','data/deribit_trade_daily'))
    spot=pd.read_csv(source/'spot_daily.csv')
    for table in [frame,residuals,obs]:
        for col in ['decision_time','label_available_at','trade_time','spot_available_at','mean_last_label']:
            if col in table:
                table[col]=pd.to_datetime(table[col],utc=True,format='mixed')
    spot['timestamp']=pd.to_datetime(spot.timestamp,utc=True,format='mixed')
    checks=[]
    def check(name,value):
        if not bool(value):
            raise AssertionError(name)
        checks.append(name)
    check('current_prints_not_future',obs.trade_time.le(obs.decision_time).all())
    check('only_closed_daily_features',obs.spot_available_at.lt(obs.decision_time).all())
    check('center_features_exclude_settlement',not set(MEAN_FEATURES)&{'delivery_index','log_return','y_btc'})
    for wid,group in obs.groupby('weekend_id'):
        t=group.decision_time.iloc[0]
        with gzip.open(source/'raw'/(wid+'_trades.json.gz'),'rt',encoding='utf-8') as stream:
            raw=json.load(stream)
        check('raw_window_'+wid,raw['complete'] and all(x['timestamp']<=int(t.timestamp()*1000) for x in raw['trades']))
        ordered=sorted(raw['trades'],key=lambda x:(x['timestamp'],x['trade_id']))
        lookup={str(x['trade_id']):x for x in raw['trades']}
        check('index_from_past_print_'+wid,np.allclose(group.spot,float(ordered[-1]['index_price'])))
        check('IV_and_premium_from_past_print_'+wid,all(
            np.isclose(r.trade_iv,float(lookup[str(r.trade_id)]['iv'])/100)
            and np.isclose(r.trade_price_btc,float(lookup[str(r.trade_id)]['price']))
            and lookup[str(r.trade_id)]['timestamp']==r.trade_time.value//1_000_000
            for r in group.itertuples()))
        causal,_=market_features(spot,t)
        check('daily_recalculation_'+wid,all(np.allclose(group[k],v,rtol=1e-10,atol=1e-12) for k,v in causal.items()))
        poisoned_spot=spot.copy()
        poisoned_spot.loc[poisoned_spot.timestamp>=t,['spot','volume']]=1e12
        after,_=market_features(poisoned_spot,t)
        check('daily_future_poison_'+wid,causal==after)
    # Four complete prefix rebuilds also check the causal creation of residual labels.
    sample_indexes=sorted(set([0,len(f)//3,2*len(f)//3,len(f)-1]))
    for index in sample_indexes:
        r=f.iloc[index]; t=pd.Timestamp(r.decision_time)
        prefix=frame.loc[frame.weekend_id<=r.weekend_id].copy()
        rebuilt=causal_mean_residuals(prefix,cfg)
        past=rebuilt.loc[rebuilt.label_available_at<t]
        stored=residuals.loc[(residuals.weekend_id<r.weekend_id)&(residuals.label_available_at<t)]
        check('prefix_residuals_'+r.weekend_id,len(past)==len(stored) and np.allclose(past.residual,stored.residual,rtol=1e-9,atol=1e-12))
    for r in f.itertuples():
        t=pd.Timestamp(r.decision_time)
        prefix=frame.loc[frame.weekend_id<=r.weekend_id].copy()
        past_errors=residuals.loc[(residuals.weekend_id<r.weekend_id)&(residuals.label_available_at<t)].copy()
        mean,vol,c,_,_=fit_distribution(prefix,past_errors,t,r.weekend_id,cfg)
        current=prefix.loc[prefix.weekend_id.eq(r.weekend_id)]
        mu=float(mean.predict(current)[0])
        variance=float(np.clip(c*np.exp(np.clip(vol.predict(current)[0],-30,5)),cfg.sigma_floor**2,cfg.sigma_cap**2))
        check('prefix_distribution_'+r.weekend_id,np.allclose([mu,variance,c],[r.mu_log_return,r.variance_log_return,r.variance_calibration],rtol=1e-9,atol=1e-12))
        changed=frame.copy(); changed_errors=residuals.copy()
        changed.loc[changed.weekend_id>=r.weekend_id,'log_return']=1000.
        changed.loc[changed.weekend_id>=r.weekend_id,'delivery_index']=1e12
        changed.loc[changed.weekend_id>r.weekend_id,MEAN_FEATURES]=1e6
        changed_errors.loc[changed_errors.weekend_id>=r.weekend_id,'residual']=1e6
        m2,v2,c2,_,_=fit_distribution(changed,changed_errors,t,r.weekend_id,cfg)
        check('poisoned_distribution_'+r.weekend_id,np.allclose(mean.beta,m2.beta,rtol=1e-10,atol=1e-12)
              and np.allclose(vol.beta,v2.beta,rtol=1e-10,atol=1e-12) and np.isclose(c,c2,rtol=1e-10)
              and np.allclose(mean.predict(current),m2.predict(current),atol=1e-12)
              and np.allclose(vol.predict(current),v2.predict(current),atol=1e-12))
        # Remove all target columns before calling the optimizer: settlement cannot enter q.
        rows=obs.loc[obs.weekend_id.eq(r.weekend_id)].copy()
        rows=rows.drop(columns=['delivery_index','settlement_btc','y_btc','label_available_at'])
        payoff=inverse_expected_payoff(r.spot,rows.strike,rows.option_type,mu,variance)
        prices=rows.trade_price_btc.to_numpy()
        alpha=prices*(1-execution.slippage_fraction)-np.minimum(execution.fee_btc,execution.fee_premium_cap*prices)-payoff
        cov=payoff_covariance(r.spot,rows.strike,rows.option_type,mu,variance,cfg.quadrature_nodes)
        result=optimize_native(rows,alpha,execution,r.pre_entry_equity_btc,cov,cfg.covariance_penalty_per_btc)
        status=result['status'] if result else 'NO_EQUITY'
        quantity=result['quantity'] if status=='ENTER' else np.zeros(len(rows))
        saved=pred.loc[pred.weekend_id.eq(r.weekend_id)].set_index('symbol').loc[rows.symbol]
        if status!=r.action or not np.allclose(quantity,saved.quantity,atol=1e-10):
            print(json.dumps(dict(weekend=r.weekend_id,status=status,recorded_status=r.action,
                                  recalculated=quantity.tolist(),recorded=saved.quantity.tolist(),
                                  max_alpha_difference=float(np.max(np.abs(alpha-saved.predicted_net_per_unit_btc.to_numpy()))))))
        check('target_free_allocation_'+r.weekend_id,status==r.action and np.allclose(quantity,saved.quantity,atol=1e-10) and np.allclose(alpha,saved.predicted_net_per_unit_btc,rtol=1e-9,atol=1e-12))
    for name,digest in initial_hashes.items():
        check('unchanged_input_'+name,hashlib.sha256((p/(name+'.csv')).read_bytes()).hexdigest()==digest)
    evidence=dict(status='PASS_NO_LOOKAHEAD_DETECTED_IN_TESTED_PATHS',check_count=len(checks),checks=checks,
                  oos_decisions_recomputed=len(f),prefix_residual_rebuilds=len(sample_indexes),raw_windows_audited=obs.weekend_id.nunique(),
                  forecast_sha256=hashlib.sha256((p/'forecasts.csv').read_bytes()).hexdigest(),
                  predictions_sha256=hashlib.sha256((p/'predictions.csv').read_bytes()).hexdigest(),
                  scope='Recorded timestamps, closed daily factors, causal residuals, train-only scalers/calibration, current label poison and target-free allocation. Not a guarantee of historical vendor publication latency or executable fills.',
                  final_parameters='Full-sample retrospective only; not passed into historical decisions.')
    (p/'lookahead_audit.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in evidence.items() if k!='checks'},indent=2))


if __name__=='__main__':
    main()
