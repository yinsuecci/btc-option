import json
from pathlib import Path
import pandas as pd

from src.risk_grid import PRESETS, TAIL_VALUES, COV_VALUES, scenario_id
from src.risk_grid_report import panel


def test_presets_are_fixed_and_current_balance_is_preserved():
    assert len(TAIL_VALUES)*len(COV_VALUES)==9
    assert PRESETS[(.005,.5)]=='AGGRESSIVE'
    assert PRESETS[(.01,1.)]=='BALANCED'
    assert PRESETS[(.02,2.)]=='CONSERVATIVE'
    assert scenario_id(.01,1.)=='tail_0p01_cov_1'


def test_grid_fragment_includes_all_cases_and_presets():
    source=Path('results/risk_grid')
    data=json.loads((source/'grid.json').read_text())
    result=panel(data)
    assert '九组风险参数' in result
    assert '激进' in result and '平衡（当前）' in result and '保守' in result
    assert 'grid-case' in result and 'grid-weekend' in result
    assert '<script src=' not in result
    for r in data['summaries']:
        assert r['scenario'] in result
    for name,bundle in data['bundles'].items():
        assert len(bundle['weekends'])==250
        assert len(bundle['performance_curve'])==198
        assert len(bundle['positions'])==len(pd.read_csv('results/center_vol/observations.csv'))


def test_balanced_grid_quantities_match_current_report():
    name=scenario_id(.01,1.)
    grid=pd.read_csv(Path('results/risk_grid')/name/'predictions.csv',float_precision='round_trip')
    original=pd.read_csv('results/center_vol/predictions.csv',float_precision='round_trip')
    columns=['weekend_id','symbol','quantity','omega']
    pd.testing.assert_frame_equal(grid[columns],original[columns],check_exact=True)
