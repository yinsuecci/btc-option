import numpy as np
import pandas as pd

from scripts.run_entry_timing import SLOTS, decision_time, split_summary


def test_time_grid_precedes_sunday_settlement_and_handles_dst():
    assert len(SLOTS)==7
    for wid in ['2022-01-01','2022-07-02']:
        expiry=pd.Timestamp(wid,tz='UTC')+pd.Timedelta(hours=32)
        for offset in SLOTS.values():
            assert decision_time(wid,offset)<expiry
    assert decision_time('2022-01-01',None).hour==17
    assert decision_time('2022-07-02',None).hour==16
    assert decision_time('2022-01-01',-8)==pd.Timestamp('2021-12-31 16:00:00Z')


def test_discovery_and_holdout_do_not_overlap():
    dates=pd.date_range('2022-12-31',periods=198,freq='W-SAT').strftime('%Y-%m-%d')
    pnl=np.r_[np.full(104,.001),np.full(94,-.0005)]
    frame=pd.DataFrame(dict(weekend_id=dates,evaluation=True,net_pnl_btc=pnl,equity_btc=1+pnl.cumsum()))
    result=split_summary(frame)
    assert result['discovery']['weekends']==104
    assert result['holdout']['weekends']==94
    assert result['discovery']['end']<result['holdout']['start']
    assert np.isclose(result['discovery']['net_pnl_btc'],.104)
    assert np.isclose(result['holdout']['net_pnl_btc'],-.047)
    assert result['holdout']['max_drawdown_pct']<0
