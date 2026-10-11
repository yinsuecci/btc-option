import json
from pathlib import Path

import pandas as pd

from src.center_vol_metrics import export_views
from src.center_vol_report import render


def test_report_library_precedes_first_chart_and_all_weekends_render(tmp_path):
    source=Path('results/center_vol')
    tables={name:pd.read_csv(source/(name+'.csv')) for name in ['forecasts','weekends','predictions']}
    formula=json.loads((source/'final_formula.json').read_text())
    metrics=json.loads((source/'metrics.json').read_text())
    export_views(tables,pd.read_csv(source/'observations.csv'),metrics,tmp_path)
    render(tables,formula,metrics,tmp_path)
    page=(tmp_path/'report.html').read_text(encoding='utf-8')
    first_chart=page.index('Plotly.newPlot(')
    assert page.index('plotly.js v') < first_chart
    assert page.index('</head>') < first_chart
    assert '<script src=' not in page
    assert '年化周度 Sharpe' in page
    assert '全部 250 个周末' in page
    assert 'Drawdown from prior peak (%)' in page
    assert '周末（含训练预热）' in page
    assert '2022-01-01' in page
    assert '北京时间周六 00:00' in page and '持有 40 小时' in page
    assert '06 / 入场时间' not in page and 'entry-timing-equity' not in page
    assert f"- {formula['execution']['risk_penalty_per_btc']:g}*sum((R_i*q_i)^2)" in page
