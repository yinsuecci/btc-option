"""Independent raw-window, allocation cashflow and timing-selection artifact checks."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_entry_timing import SLOTS, decision_time, split_summary
from src.center_vol_metrics import performance_summary
from src.center_vol import CenterConfig, fit_distribution


def main():
    root=Path('results/entry_timing')
    summary=json.loads((root/'summary.json').read_text())
    evidence=json.loads((root/'validation.json').read_text())
    checks=[]
    def check(name,value):
        if not bool(value):
            raise AssertionError(name)
        checks.append(name)
    check('summary_hash',hashlib.sha256((root/'summary.json').read_bytes()).hexdigest()==evidence['summary_sha256'])
    check('fixed_grid',set(SLOTS)=={m['slot'] for m in summary['summaries']})
    ids=pd.date_range('2022-01-01','2026-10-10',freq='W-SAT').strftime('%Y-%m-%d').tolist()
    for m in summary['summaries']:
        name=m['slot']; p=root/name; data=Path('data/entry_timing')/name
        read=lambda filename:pd.read_csv(p/(filename+'.csv'),float_precision='round_trip')
        w=read('weekends'); obs=read('observations'); positions=read('positions_by_weekend'); forecasts=read('forecasts')
        check(name+'_calendar',w.weekend_id.tolist()==ids and w.evaluation.sum()==198)
        check(name+'_complete_windows',len(pd.read_csv(data/'download_coverage.csv'))==250)
        for wid in ids:
            with gzip.open(data/'raw'/(wid+'_trades.json.gz'),'rt',encoding='utf-8') as stream:
                raw=json.load(stream)
            t=decision_time(wid,SLOTS[name]); end=int(t.timestamp()*1000)
            check(name+'_raw_'+wid,raw['complete'] and raw['end_ms']==end and raw['start_ms']==end-3600000
                  and all(raw['start_ms']<=r['timestamp']<=end for r in raw['trades'])
                  and len({r['trade_id'] for r in raw['trades']})==len(raw['trades']))
        for col in ['trade_time','decision_time','spot_available_at']:
            obs[col]=pd.to_datetime(obs[col],utc=True,format='mixed')
        check(name+'_observation_chronology',obs.trade_time.le(obs.decision_time).all() and obs.spot_available_at.lt(obs.decision_time).all())
        for col in ['decision_time','mean_last_label','variance_last_label']:
            forecasts[col]=pd.to_datetime(forecasts[col],utc=True,format='mixed')
        check(name+'_matured_labels',(forecasts.mean_last_label<forecasts.decision_time).all()
              and (forecasts.variance_last_label<forecasts.decision_time).all())
        training=read('price_training'); errors=read('causal_mean_residuals')
        for table in [training,errors]:
            for col in ['decision_time','label_available_at']:
                table[col]=pd.to_datetime(table[col],utc=True,format='mixed')
        for j in sorted(set([0,len(forecasts)//2,len(forecasts)-1])):
            f=forecasts.iloc[j]; cutoff=f.decision_time
            poisoned=training.copy(); poisoned_errors=errors.copy()
            poisoned.loc[poisoned.label_available_at>=cutoff,'log_return']=999.
            poisoned_errors.loc[poisoned_errors.label_available_at>=cutoff,'residual']=999.
            mean,vol,cal,_,_=fit_distribution(poisoned,poisoned_errors,cutoff,f.weekend_id,CenterConfig())
            current=training.loc[training.weekend_id.eq(f.weekend_id)]
            mu=float(mean.predict(current)[0])
            var=float(np.clip(cal*np.exp(np.clip(vol.predict(current)[0],-30,5)),1e-6,.04))
            check(name+'_future_poison_'+str(j),np.isclose(mu,f.mu_log_return,rtol=0,atol=1e-12)
                  and np.isclose(var,f.variance_log_return,rtol=0,atol=1e-12))
        joined=positions.merge(obs[['weekend_id','symbol','y_btc','trade_amount']],on=['weekend_id','symbol'],validate='one_to_one')
        check(name+'_leg_cashflow',np.allclose(joined.quantity*joined.y_btc,joined.net_pnl_btc,atol=1e-12))
        check(name+'_quantity_bounds',joined.quantity.ge(0).all()
              and (joined.quantity<=np.minimum(.5,joined.trade_amount)+1e-7).all()
              and np.allclose(joined.quantity/.1,np.round(joined.quantity/.1))
              and np.allclose(joined.omega,-joined.quantity))
        grouped=joined.groupby('weekend_id').net_pnl_btc.sum().reindex(ids,fill_value=0)
        check(name+'_weekly_cashflow',np.allclose(grouped,w.net_pnl_btc,atol=1e-12))
        check(name+'_idle',w.loc[~w.traded,'net_pnl_btc'].eq(0).all())
        metrics,curve=performance_summary(w)
        check(name+'_sharpe',np.isclose(metrics['weekly_return_sharpe'],m['weekly_return_sharpe']))
        check(name+'_drawdown',np.isclose(metrics['max_drawdown_pct'],m['max_drawdown_pct']))
        parts=split_summary(w)
        check(name+'_split',parts=={k:m[k] for k in ['discovery','holdout']})
    best=max(summary['summaries'],key=lambda m:m['net_pnl_btc'])['slot']
    selected=max(summary['summaries'],key=lambda m:m['discovery']['net_pnl_btc'])['slot']
    check('full_history_winner',best==summary['full_history_winner'])
    check('training_only_selection',selected==summary['discovery_selected_slot'])
    original=pd.read_csv('results/center_vol/weekends.csv',float_precision='round_trip')
    baseline=pd.read_csv(root/'current_sat_12ny/weekends.csv',float_precision='round_trip')
    check('baseline_reproduced',np.allclose(original.net_pnl_btc,baseline.net_pnl_btc,rtol=0,atol=1e-12))
    result=dict(status='PASS',check_count=len(checks),checks=checks,
                summary_sha256=evidence['summary_sha256'])
    (root/'file_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))


if __name__=='__main__':
    main()
