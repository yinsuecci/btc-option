"""OOS equity-return risk statistics and complete weekend position exports."""
import numpy as np
import pandas as pd


def performance_summary(weekends, initial_equity=1.):
    w = weekends.loc[weekends.evaluation].sort_values('weekend_id').copy()
    if w.empty:
        raise ValueError('No evaluated weekends')
    equity = w.equity_btc.to_numpy(float)
    pnl = w.net_pnl_btc.to_numpy(float)
    before = np.r_[initial_equity, equity[:-1]]
    if not np.allclose(before+pnl, equity) or np.any(before <= 0):
        raise ValueError('Equity reconciliation or positive return denominator failed')
    returns = pnl/before
    peak = np.maximum.accumulate(np.r_[initial_equity, equity])[1:]
    drawdown = equity/peak-1
    volatility = np.std(returns, ddof=1) if len(returns)>1 else 0.
    trough = int(np.argmin(drawdown))
    peak_locations = np.flatnonzero(equity[:trough+1] >= peak[trough]-1e-12)
    peak_date = str(w.weekend_id.iloc[peak_locations[-1]]) if len(peak_locations) else 'INITIAL_EQUITY'
    recovered = np.flatnonzero(equity[trough+1:] >= peak[trough]-1e-12)
    recovery_date = str(w.weekend_id.iloc[trough+1+recovered[0]]) if len(recovered) else None
    curve = pd.DataFrame(dict(weekend_id=w.weekend_id.to_numpy(), equity_before_btc=before,
                              equity_btc=equity, net_pnl_btc=pnl, weekend_return=returns,
                              running_peak_btc=peak, drawdown_btc=equity-peak,
                              drawdown_pct=drawdown*100))
    summary = dict(weekly_return_sharpe=float(np.mean(returns)/volatility*np.sqrt(52)) if volatility>0 else None,
                   sharpe_risk_free_btc=0., sharpe_annualization_periods=52,
                   sharpe_observations=len(w), sharpe_includes_no_trade_weekends=True,
                   max_drawdown_pct=float(drawdown.min()*100),
                   maximum_drawdown_btc=float(np.min(equity-peak)),
                   drawdown_peak_weekend=peak_date, drawdown_trough_weekend=str(w.weekend_id.iloc[trough]),
                   drawdown_recovery_weekend=recovery_date,
                   drawdown_basis='End-weekend settled BTC equity including initial equity; not intraday mark-to-market')
    return summary, curve


def complete_positions(obs, predictions, weekends):
    fields=['weekend_id','symbol','quantity','omega','relative_abs_weight','expected_payoff_btc',
            'predicted_net_per_unit_btc','net_pnl_btc']
    positions=obs.merge(predictions[fields],on=['weekend_id','symbol'],how='left',validate='one_to_one')
    positions=positions.merge(weekends[['weekend_id','status','evaluation']],on='weekend_id',how='left',validate='many_to_one')
    if positions.status.isna().any() or positions.loc[positions.evaluation,'quantity'].isna().any():
        raise ValueError('Missing status or evaluated allocation')
    # Warmup weekends have no position; unavailable forecasts remain NaN, not zero.
    for col in ['quantity','omega','relative_abs_weight','net_pnl_btc']:
        positions[col]=positions[col].fillna(0.)
    return positions.sort_values(['weekend_id','symbol'])


def export_views(tables, observations, metrics, output, initial_equity=1.):
    summary,curve=performance_summary(tables['weekends'],initial_equity)
    metrics.update(summary)
    tables['performance_curve']=curve
    tables['positions']=complete_positions(observations,tables['predictions'],tables['weekends'])
    curve.to_csv(output/'performance_curve.csv',index=False)
    tables['positions'].to_csv(output/'positions_by_weekend.csv',index=False)
    return tables,metrics
