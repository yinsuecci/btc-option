import numpy as np
import pandas as pd
import pytest

from src.center_vol_metrics import complete_positions, performance_summary


def test_sharpe_uses_equity_returns_including_idle_weekends():
    w=pd.DataFrame(dict(weekend_id=['2022-01-01','2022-01-08','2022-01-15'],
                        evaluation=[True]*3,net_pnl_btc=[.1,0.,-.05],equity_btc=[1.1,1.1,1.05]))
    metrics,curve=performance_summary(w)
    expected=np.array([.1,0.,-.05/1.1])
    assert metrics['weekly_return_sharpe']==pytest.approx(expected.mean()/expected.std(ddof=1)*np.sqrt(52))
    np.testing.assert_allclose(curve.weekend_return,expected)
    assert metrics['sharpe_observations']==3
    assert metrics['max_drawdown_pct']==pytest.approx((1.05/1.1-1)*100)
    assert metrics['drawdown_peak_weekend']=='2022-01-08'
    assert metrics['drawdown_trough_weekend']=='2022-01-15'


def test_drawdown_includes_initial_equity_peak():
    w=pd.DataFrame(dict(weekend_id=['2022-01-01','2022-01-08'],evaluation=[True,True],
                        net_pnl_btc=[-.1,.05],equity_btc=[.9,.95]))
    metrics,_=performance_summary(w)
    assert metrics['max_drawdown_pct']==pytest.approx(-10)
    assert metrics['maximum_drawdown_btc']==pytest.approx(-.1)
    assert metrics['drawdown_peak_weekend']=='INITIAL_EQUITY'


def test_zero_return_sharpe_is_unavailable_not_infinite():
    w=pd.DataFrame(dict(weekend_id=['2022-01-01','2022-01-08'],evaluation=[True,True],
                        net_pnl_btc=[0.,0.],equity_btc=[1.,1.]))
    metrics,_=performance_summary(w)
    assert metrics['weekly_return_sharpe'] is None


def test_warmup_positions_are_zero_but_forecasts_are_missing():
    obs=pd.DataFrame(dict(weekend_id=['2022-01-01','2022-01-08'],symbol=['A','B']))
    pred=pd.DataFrame(dict(weekend_id=['2022-01-08'],symbol=['B'],quantity=[.1],omega=[-.1],
                           relative_abs_weight=[1.],expected_payoff_btc=[.001],
                           predicted_net_per_unit_btc=[.002],net_pnl_btc=[.0002]))
    w=pd.DataFrame(dict(weekend_id=['2022-01-01','2022-01-08'],status=['WARMUP','ENTER'],evaluation=[False,True]))
    result=complete_positions(obs,pred,w)
    assert result.iloc[0].quantity==0
    assert pd.isna(result.iloc[0].expected_payoff_btc)
    assert result.iloc[1].omega==-.1
