"""Compare the prescribed tail penalty change on identical historical decisions."""
import hashlib
import json
from pathlib import Path
import pandas as pd
import numpy as np


def summarize(path):
    m=json.loads((path/'metrics.json').read_text())
    formula=json.loads((path/'final_formula.json').read_text())
    f=pd.read_csv(path/'forecasts.csv')
    years=(pd.Timestamp(m['oos_end'])-pd.Timestamp(m['oos_start'])).days/365.25
    return dict(risk_penalty=formula['execution']['risk_penalty_per_btc'],net_pnl_btc=m['net_pnl_btc'],
                cagr=(m['final_equity_btc']/formula['execution']['initial_equity_btc'])**(1/years)-1,
                sharpe=m['weekly_return_sharpe'],max_drawdown_pct=m['max_drawdown_pct'],
                traded_weekends=m['traded_weekends'],average_margin_btc=float(f.margin_proxy_btc.mean()),
                average_quantity=float(f.quantity.mean()))


if __name__=='__main__':
    base=Path('results/center_vol_risk_002'); current=Path('results/center_vol')
    b=json.loads((base/'final_formula.json').read_text()); c=json.loads((current/'final_formula.json').read_text())
    if not c['execution']['risk_penalty_per_btc'] < b['execution']['risk_penalty_per_btc']:
        raise AssertionError('Current backtest must finish with a strictly lower tail penalty')
    for key in ['center_beta','vol_gamma','config','variance_calibration','train_start','train_end']:
        if b[key]!=c[key]:
            raise AssertionError('Forecast model changed: '+key)
    for key,value in b['execution'].items():
        if key!='risk_penalty_per_btc' and value!=c['execution'][key]:
            raise AssertionError('Other execution parameter changed: '+key)
    bf=pd.read_csv(base/'forecasts.csv'); cf=pd.read_csv(current/'forecasts.csv')
    columns=['mu_log_return','variance_log_return','variance_calibration','center_mean_usd','center_median_usd']
    if not bf.weekend_id.equals(cf.weekend_id) or not np.allclose(bf[columns],cf[columns],atol=1e-12):
        raise AssertionError('Forecasts or evaluation dates changed')
    result=dict(baseline=summarize(base),current=summarize(current),
                current_forecast_sha256=hashlib.sha256((current/'forecasts.csv').read_bytes()).hexdigest(),
                only_tail_penalty_changed=True,interpretation='Same historical period; not a fresh holdout')
    (current/'risk_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
