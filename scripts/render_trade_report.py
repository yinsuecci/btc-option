import json
from pathlib import Path
import shutil

import pandas as pd

from src.trade_report import report
from src.trade_research import TradeConfig


def main():
    p = Path('results/trade_only')
    names = ['beta_history','predictions','trades','decisions','weekends','stress','baseline_trades']
    tables = {name: pd.read_csv(p/(name+'.csv')) for name in names}
    obs = pd.read_csv(p/'observations.csv')
    coverage = pd.read_csv(p/'weekend_coverage.csv')
    formula = json.loads((p/'final_formula.json').read_text())
    cfg = TradeConfig(**json.loads((p/'config.json').read_text()))
    metrics = report(tables, obs, coverage, formula, cfg, p)
    # Publish the current research at the existing report entry, preserving synthetic evidence.
    old = Path('results/report.html')
    backup = Path('results/report_synthetic.html')
    if old.exists() and not backup.exists():
        shutil.copy2(old, backup)
    shutil.copy2(p/'report.html', old)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
