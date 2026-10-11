"""Independent file-level audit of the nine recorded parameter combinations."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


if __name__=='__main__':
    p=Path('results/risk_grid')
    grid=json.loads((p/'grid.json').read_text())
    checks=[]
    def check(name,value):
        if not bool(value):
            raise AssertionError(name)
        checks.append(name)
    check('nine_unique_cases',len(grid['summaries'])==9 and len({r['scenario'] for r in grid['summaries']})==9)
    check('current_forecasts_unchanged',grid['source_forecast_sha256']==hashlib.sha256(Path('results/center_vol/forecasts.csv').read_bytes()).hexdigest())
    source=json.loads(Path('results/center_vol/final_formula.json').read_text())
    for m in grid['summaries']:
        name=m['scenario']; case=p/name
        formula=json.loads((case/'config.json').read_text())
        for key in ['center_beta','vol_gamma','variance_calibration']:
            check(name+':unchanged_'+key,formula[key]==source[key])
        for key,value in source['execution'].items():
            if key!='risk_penalty_per_btc':
                check(name+':same_'+key,formula['execution'][key]==value)
        for key,value in source['config'].items():
            if key!='covariance_penalty_per_btc':
                check(name+':same_model_'+key,formula['config'][key]==value)
        w=pd.read_csv(case/'weekends.csv',float_precision='round_trip')
        curve=pd.read_csv(case/'performance_curve.csv',float_precision='round_trip')
        pos=pd.read_csv(case/'positions_by_weekend.csv',float_precision='round_trip')
        d=pd.read_csv(case/'decisions.csv',float_precision='round_trip')
        ledger=pd.read_csv(case/'event_ledger.csv',float_precision='round_trip')
        evaluated=w.loc[w.evaluation]
        check(name+':250_weekends',len(w)==250 and len(evaluated)==198
              and len(pos)==len(pd.read_csv('results/center_vol/observations.csv')))
        check(name+':zero_warmup',pos.loc[~pos.evaluation,'quantity'].eq(0).all() and pos.loc[~pos.evaluation,'expected_payoff_btc'].isna().all())
        check(name+':cash_reconciliation',np.isclose(pos.net_pnl_btc.sum(),ledger.cashflow_btc.sum(),atol=1e-10) and np.isclose(w.net_pnl_btc.sum(),m['net_pnl_btc'],atol=1e-10))
        check(name+':equity',np.allclose(w.equity_btc,1+w.net_pnl_btc.cumsum()))
        returns=evaluated.net_pnl_btc.to_numpy()/np.r_[1.,evaluated.equity_btc.to_numpy()[:-1]]
        check(name+':Sharpe',np.isclose(returns.mean()/returns.std(ddof=1)*np.sqrt(52),m['weekly_return_sharpe']))
        check(name+':drawdown',np.isclose(curve.drawdown_pct.min(),m['max_drawdown_pct']))
        check(name+':lot_and_sign',np.allclose(pos.quantity/.1,np.round(pos.quantity/.1)) and np.allclose(pos.omega,-pos.quantity))
        check(name+':leg_caps',pos.quantity.le(np.minimum(.5,pos.trade_amount)+1e-8).all())
        check(name+':risk_caps',d.margin_proxy_btc.le(np.minimum(.5,.5*d.pre_entry_equity_btc)+1e-8).all()
              and d.worst_stress_btc.le(np.minimum(.25,.25*d.pre_entry_equity_btc)+1e-8).all()
              and d.delta_log_btc.abs().le(.30+1e-8).all() and d.vega_btc.abs().le(.20+1e-8).all())
    check('grid_content_matches_validation',json.loads((p/'validation.json').read_text())['grid_sha256']==hashlib.sha256((p/'grid.json').read_bytes()).hexdigest())
    result=dict(status='PASS',check_count=len(checks),checks=checks,scenarios=9)
    (p/'file_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
