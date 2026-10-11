"""Re-render the dashboard without retraining or changing recorded allocations."""
import json
from pathlib import Path
import pandas as pd
from src.center_vol_report import render
from src.center_vol_metrics import export_views

if __name__=='__main__':
    p=Path('results/center_vol')
    tables={name:pd.read_csv(p/(name+'.csv')) for name in ['forecasts','weekends','predictions']}
    formula=json.loads((p/'final_formula.json').read_text())
    metrics=json.loads((p/'metrics.json').read_text())
    export_views(tables,pd.read_csv(p/'observations.csv'),metrics,p,formula['execution']['initial_equity_btc'])
    (p/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding='utf-8')
    render(tables,formula,metrics,p)
