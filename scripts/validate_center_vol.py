"""Independent checks on realized files for the price / variance pipeline."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--browser-verified',action='store_true')
    args=parser.parse_args()
    p=Path('results/center_vol')
    frames={name:pd.read_csv(p/(name+'.csv')) for name in ['price_training','causal_mean_residuals','forecasts','predictions','trades','weekends','event_ledger']}
    checks=[]
    def check(name,condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)
    raw=frames['price_training']; errors=frames['causal_mean_residuals']; f=frames['forecasts']
    pred=frames['predictions']; tr=frames['trades']; w=frames['weekends']; ledger=frames['event_ledger']
    cfg=json.loads((p/'final_formula.json').read_text())
    check('250_unique_weekend_labels',len(raw)==250 and raw.weekend_id.nunique()==250)
    check('real_log_return_labels',np.allclose(raw.log_return,np.log(raw.delivery_index/raw.spot)))
    check('prequential_mean_train_end',errors.mean_train_end.lt(errors.weekend_id).all())
    check('prequential_mature_labels',pd.to_datetime(errors.mean_last_label,utc=True,format='mixed').lt(pd.to_datetime(errors.decision_time,utc=True,format='mixed')).all())
    for r in f.itertuples():
        t=pd.Timestamp(r.decision_time)
        prior=raw.loc[(raw.weekend_id<r.weekend_id)&(pd.to_datetime(raw.label_available_at,utc=True)<t)]
        old_errors=errors.loc[(errors.weekend_id<r.weekend_id)&(pd.to_datetime(errors.label_available_at,utc=True)<t)]
        check('mean_training_'+r.weekend_id,len(prior)==r.train_weekends and prior.weekend_id.max()==r.train_end)
        check('variance_training_'+r.weekend_id,len(old_errors)==r.variance_rows and old_errors.weekend_id.max()==r.variance_train_end)
    check('price_expectation_identity',np.allclose(f.center_mean_usd,f.spot*np.exp(f.mu_log_return+.5*f.variance_log_return)))
    check('price_median_identity',np.allclose(f.center_median_usd,f.spot*np.exp(f.mu_log_return)))
    check('volatility_identity',np.allclose(f.sigma_weekend**2,f.variance_log_return))
    check('volatility_clip',f.sigma_weekend.between(.001-1e-10,.20+1e-10).all())
    check('alpha_from_center_and_vol',np.allclose(pred.predicted_net_per_unit_btc,.95*pred.trade_price_btc-np.minimum(.0003,.125*pred.trade_price_btc)-pred.expected_payoff_btc))
    check('omega_signed_quantity',np.allclose(pred.omega,-pred.quantity))
    check('lots',np.allclose(pred.quantity/.1,np.round(pred.quantity/.1)))
    check('leg_bound',pred.quantity.le(np.minimum(.5,pred.trade_amount)+1e-9).all())
    check('gross_bound',f.quantity.le(2+1e-8).all())
    check('margin_bound',f.margin_proxy_btc.le(np.minimum(.5,.5*f.pre_entry_equity_btc)+1e-8).all())
    check('stress_bound',f.worst_stress_btc.le(np.minimum(.25,.25*f.pre_entry_equity_btc)+1e-8).all())
    check('delta_bound',f.delta_log_btc.abs().le(.30+1e-8).all())
    check('vega_bound',f.vega_btc.abs().le(.20+1e-8).all())
    check('leg_realized_pnl',np.allclose(pred.net_pnl_btc,pred.quantity*pred.y_btc))
    check('ledger_matches_all_legs',abs(ledger.cashflow_btc.sum()-tr.net_pnl_btc.sum())<1e-10)
    check('weekend_matches_legs',abs(w.net_pnl_btc.sum()-tr.net_pnl_btc.sum())<1e-10)
    check('equity',np.allclose(w.equity_btc,1+w.net_pnl_btc.cumsum()))
    check('mean_beta_not_option_alpha','median_trade_iv' in cfg['center_beta']['coefficients'] and 'trade_price_btc' not in cfg['center_beta']['coefficients'])
    check('independent_variance_gamma','vol_gamma' in cfg and cfg['variance_rows']==len(errors))
    check('52_warmup',int((~w.evaluation).sum())==52)
    page=(p/'report.html').read_text(encoding='utf-8')
    check('offline_report','<script src=' not in page)
    check('two_parameter_formulas','mu_log_return =' in page and 'g_log_squared_error =' in page)
    curve=pd.read_csv(p/'performance_curve.csv')
    positions=pd.read_csv(p/'positions_by_weekend.csv')
    metrics=json.loads((p/'metrics.json').read_text())
    expected_returns=w.loc[w.evaluation].net_pnl_btc.to_numpy()/np.r_[1.,w.loc[w.evaluation].equity_btc.to_numpy()[:-1]]
    sharpe=np.mean(expected_returns)/np.std(expected_returns,ddof=1)*np.sqrt(52)
    check('Sharpe_from_equity_returns',np.isclose(sharpe,metrics['weekly_return_sharpe']) and np.allclose(expected_returns,curve.weekend_return))
    equity=w.loc[w.evaluation].equity_btc.to_numpy()
    peaks=np.maximum.accumulate(np.r_[1.,equity])[1:]
    check('drawdown_with_initial_peak',np.isclose(np.min(equity/peaks-1)*100,metrics['max_drawdown_pct']) and np.allclose(curve.drawdown_pct,(equity/peaks-1)*100))
    check('all_250_weekends_positions',positions.weekend_id.nunique()==250 and len(positions)==len(pd.read_csv(p/'observations.csv')))
    check('warmup_not_forecast',positions.loc[~positions.evaluation,'expected_payoff_btc'].isna().all() and positions.loc[~positions.evaluation,'quantity'].eq(0).all())
    recorded=positions.loc[positions.evaluation].set_index(['weekend_id','symbol']).sort_index()
    original=pred.set_index(['weekend_id','symbol']).sort_index()
    check('holdings_export_unchanged',recorded.index.equals(original.index) and np.allclose(recorded.quantity,original.quantity))
    check('Sharpe_and_drawdown_in_report','年化周度 Sharpe' in page and 'Drawdown from prior peak (%)' in page)
    check('all_weekend_holdings_controls','holding-search' in page and 'held-only' in page and 'positions_by_weekend.csv' in page)
    audit=json.loads((p/'lookahead_audit.json').read_text())
    check('lookahead_audit_matches_current_results',audit['status']=='PASS_NO_LOOKAHEAD_DETECTED_IN_TESTED_PATHS'
          and audit['forecast_sha256']==hashlib.sha256((p/'forecasts.csv').read_bytes()).hexdigest()
          and audit['predictions_sha256']==hashlib.sha256((p/'predictions.csv').read_bytes()).hexdigest())
    check('audit_in_report','未发现未来函数' in page and str(audit['check_count']) in page)
    grid_path=p.parent/'risk_grid'/'grid.json'
    if grid_path.exists():
        grid=json.loads(grid_path.read_text())
        grid_audit=json.loads((grid_path.parent/'file_validation.json').read_text())
        check('grid_source_matches_current',grid['source_forecast_sha256']==hashlib.sha256((p/'forecasts.csv').read_bytes()).hexdigest())
        check('grid_file_audit_pass',grid_audit['status']=='PASS' and grid_audit['scenarios']==9)
        check('nine_risk_cases_in_report','grid-case' in page and all(r['scenario'] in page for r in grid['summaries']))
    check('only_selected_entry_time','entry-timing-equity' not in page and '06 / 入场时间' not in page
          and '北京时间周六 00:00' in page and '持有 40 小时' in page)
    decision_times=pd.to_datetime(raw.decision_time,utc=True)
    check('friday_16utc_decisions',decision_times.dt.weekday.eq(4).all() and decision_times.dt.hour.eq(16).all())
    result=dict(status='PASS',check_count=len(checks),checks=checks,source_weekends=len(raw),oos_weekends=len(f),
                report_sha256=hashlib.sha256((p/'report.html').read_bytes()).hexdigest(),
                browser_desktop_mobile_verified=args.browser_verified)
    (p/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))


if __name__=='__main__':
    main()
