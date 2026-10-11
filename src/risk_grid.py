"""Prespecified 3x3 risk sensitivity on unchanged causal forecasts, not model selection."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.center_vol import inverse_expected_payoff, payoff_covariance
from src.center_vol_metrics import complete_positions, performance_summary
from src.trade_research import TradeConfig, optimize_native

TAIL_VALUES = (.005, .01, .02)
COV_VALUES = (.5, 1., 2.)
PRESETS = {(.005,.5):'AGGRESSIVE',(.01,1.):'BALANCED',(.02,2.):'CONSERVATIVE'}


def scenario_id(tail, covariance):
    return f'tail_{tail:g}_cov_{covariance:g}'.replace('.','p')


def simulate_case(obs, forecasts, warmup, tail, covariance_penalty, execution, nodes):
    cfg=replace(execution,risk_penalty_per_btc=tail)
    equity=cfg.initial_equity_btc
    weekly=warmup.to_dict('records')
    allocations=[]; decisions=[]; events=[]
    checks=[]
    for f in forecasts.itertuples():
        rows=obs.loc[obs.weekend_id.eq(f.weekend_id)].copy()
        # The optimizer receives only decision-time inputs; label cashflows stay outside it.
        decision_rows=rows.drop(columns=['delivery_index','settlement_btc','y_btc','label_available_at'])
        payoff=inverse_expected_payoff(f.spot,rows.strike,rows.option_type,f.mu_log_return,f.variance_log_return)
        cov=payoff_covariance(f.spot,rows.strike,rows.option_type,f.mu_log_return,f.variance_log_return,nodes)
        prices=rows.trade_price_btc.to_numpy()
        alpha=prices*(1-cfg.slippage_fraction)-np.minimum(cfg.fee_btc,cfg.fee_premium_cap*prices)-payoff
        result=optimize_native(decision_rows,alpha,cfg,equity,cov,covariance_penalty)
        action=result['status'] if result else 'NO_EQUITY'
        q=result['quantity'] if action=='ENTER' else np.zeros(len(rows))
        decisions.append(dict(weekend_id=f.weekend_id,action=action,pre_entry_equity_btc=equity,
                              quantity=float(q.sum()),expected_net_btc=float(alpha@q),
                              margin_proxy_btc=result.get('margin_proxy_btc',0.) if result else 0.,
                              worst_stress_btc=result.get('worst_stress_btc',0.) if result else 0.,
                              delta_log_btc=result.get('delta_log_btc',0.) if result else 0.,
                              vega_btc=result.get('vega_btc',0.) if result else 0.))
        d=decisions[-1]
        valid=(np.all(q>=0) and np.all(q<=np.minimum(cfg.max_leg_quantity,rows.trade_amount)+1e-8)
               and np.allclose(q/cfg.lot,np.round(q/cfg.lot)) and q.sum()<=cfg.gross_limit+1e-8
               and abs(d['delta_log_btc'])<=cfg.delta_log_limit+1e-8
               and abs(d['vega_btc'])<=cfg.vega_btc_limit+1e-8
               and d['margin_proxy_btc']<=min(cfg.margin_proxy_limit,.5*equity)+1e-8
               and d['worst_stress_btc']<=min(cfg.stress_loss_limit,.25*equity)+1e-8)
        if not valid:
            raise AssertionError('Constraint failure: '+f.weekend_id)
        checks.append('constraints_'+f.weekend_id)
        net=0.
        for i,(_,r) in enumerate(rows.iterrows()):
            pnl=float(q[i]*r.y_btc)
            allocations.append(dict(weekend_id=f.weekend_id,symbol=r.symbol,quantity=float(q[i]),omega=float(-q[i]),
                                     relative_abs_weight=float(q[i]/q.sum()) if q.sum() else 0.,
                                     expected_payoff_btc=float(payoff[i]),predicted_net_per_unit_btc=float(alpha[i]),net_pnl_btc=pnl))
            if q[i]>0:
                net+=pnl
                events.extend([dict(weekend_id=f.weekend_id,symbol=r.symbol,event='MODELLED_ENTRY',timestamp=r.decision_time,
                                    cashflow_btc=float(q[i]*(r.trade_price_btc-r.entry_fee_btc-r.slippage_btc))),
                               dict(weekend_id=f.weekend_id,symbol=r.symbol,event='OFFICIAL_DELIVERY',timestamp=r.expiry,
                                    cashflow_btc=float(-q[i]*r.settlement_btc))])
        equity+=net
        weekly.append(dict(weekend_id=f.weekend_id,evaluation=True,status=action,traded=bool(q.sum()),net_pnl_btc=net,equity_btc=equity))
    w=pd.DataFrame(weekly); pred=pd.DataFrame(allocations); d=pd.DataFrame(decisions)
    metrics,curve=performance_summary(w,cfg.initial_equity_btc)
    years=(pd.Timestamp(forecasts.weekend_id.iloc[-1])-pd.Timestamp(forecasts.weekend_id.iloc[0])).days/365.25
    positions=complete_positions(obs,pred,w)
    gross=float((positions.quantity*(positions.trade_price_btc-positions.settlement_btc)).sum())
    costs=float((positions.quantity*(positions.entry_fee_btc+positions.slippage_btc)).sum())
    net=float(w.net_pnl_btc.sum())
    ledger=pd.DataFrame(events,columns=['weekend_id','symbol','event','timestamp','cashflow_btc'])
    if not np.isclose(gross-costs,net,atol=1e-10) or not np.isclose(ledger.cashflow_btc.sum(),net,atol=1e-10):
        raise AssertionError('PnL or ledger mismatch')
    rng=np.random.default_rng(20261011)
    pnl=w.loc[w.evaluation].net_pnl_btc.to_numpy()
    block=min(8,len(pnl)); starts=rng.integers(0,len(pnl)-block+1,size=(5000,int(np.ceil(len(pnl)/block))))
    indexes=(starts[:,:,None]+np.arange(block)).reshape(5000,-1)[:,:len(pnl)]
    metrics.update(scenario=scenario_id(tail,covariance_penalty),preset=PRESETS.get((tail,covariance_penalty),'GRID'),
                   tail_penalty=tail,covariance_penalty=covariance_penalty,net_pnl_btc=net,
                   gross_pnl_btc=gross,costs_btc=costs,final_equity_btc=equity,
                   cagr=(equity/cfg.initial_equity_btc)**(1/years)-1,
                   traded_weekends=int(w.traded.sum()),average_quantity=float(d.quantity.mean()),
                   average_margin_btc=float(d.margin_proxy_btc.mean()),max_margin_btc=float(d.margin_proxy_btc.max()),
                   worst_weekend_btc=float(pnl.min()),trade_legs=int(positions.quantity.gt(0).sum()),
                   oos_weekends=len(forecasts),status_counts=w.loc[w.evaluation].status.value_counts().to_dict(),
                   fixed_strategy_8week_bootstrap_pnl_95ci_btc=list(map(float,np.quantile(pnl[indexes].sum(axis=1),[.025,.975]))))
    return metrics,dict(weekends=w,predictions=pred,positions=positions,performance_curve=curve,decisions=d,event_ledger=ledger),checks


def run(source='results/center_vol',output='results/risk_grid'):
    s=Path(source); p=Path(output); p.mkdir(parents=True,exist_ok=True)
    audit=json.loads((s/'lookahead_audit.json').read_text())
    hashes={name:hashlib.sha256((s/(name+'.csv')).read_bytes()).hexdigest() for name in ['forecasts','predictions']}
    if audit['status']!='PASS_NO_LOOKAHEAD_DETECTED_IN_TESTED_PATHS' or hashes['forecasts']!=audit['forecast_sha256'] or hashes['predictions']!=audit['predictions_sha256']:
        raise ValueError('Source forecasts must pass current lookahead audit')
    formula=json.loads((s/'final_formula.json').read_text())
    execution=TradeConfig(**formula['execution'])
    obs=pd.read_csv(s/'observations.csv',float_precision='round_trip')
    forecasts=pd.read_csv(s/'forecasts.csv',float_precision='round_trip')
    original_w=pd.read_csv(s/'weekends.csv',float_precision='round_trip')
    original_predictions=pd.read_csv(s/'predictions.csv',float_precision='round_trip')
    summaries=[]; bundles={}; all_checks=[]
    for tail in TAIL_VALUES:
        for covariance in COV_VALUES:
            metrics,tables,checks=simulate_case(obs,forecasts,original_w.loc[~original_w.evaluation].copy(),tail,covariance,execution,formula['config']['quadrature_nodes'])
            name=metrics['scenario']; target=p/name; target.mkdir(exist_ok=True)
            for label,table in tables.items():
                filename='positions_by_weekend' if label=='positions' else label
                table.to_csv(target/(filename+'.csv'),index=False)
            case_formula=json.loads(json.dumps(formula))
            case_formula['execution']['risk_penalty_per_btc']=tail
            case_formula['config']['covariance_penalty_per_btc']=covariance
            (target/'config.json').write_text(json.dumps(case_formula,indent=2),encoding='utf-8')
            (target/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding='utf-8')
            if metrics['preset']=='BALANCED':
                index=['weekend_id','symbol']
                a=tables['predictions'].set_index(index).sort_index(); b=original_predictions.set_index(index).sort_index()
                if not a.index.equals(b.index) or not np.allclose(a.quantity,b.quantity,atol=1e-10) or not np.allclose(tables['weekends'].net_pnl_btc,original_w.net_pnl_btc,atol=1e-10):
                    raise AssertionError('Balanced case does not reproduce current strategy')
                checks.append('balanced_exactly_reproduces_current')
            summaries.append(metrics); all_checks.extend(name+':'+c for c in checks)
            bundles[name]={key:json.loads(table.to_json(orient='records',date_format='iso')) for key,table in tables.items() if key in ['positions','performance_curve','weekends']}
            print(json.dumps({key:metrics[key] for key in ['scenario','preset','cagr','weekly_return_sharpe','max_drawdown_pct','traded_weekends']}),flush=True)
    for name,digest in hashes.items():
        if hashlib.sha256((s/(name+'.csv')).read_bytes()).hexdigest()!=digest:
            raise AssertionError('Source changed while grid was running')
    report_data=dict(summaries=summaries,bundles=bundles,source_forecast_sha256=hashes['forecasts'],
                     source_predictions_sha256=hashes['predictions'],preset_definitions={k:scenario_id(*pair) for pair,k in PRESETS.items()},
                     period=dict(start=forecasts.weekend_id.iloc[0],end=forecasts.weekend_id.iloc[-1],oos_weekends=len(forecasts)),
                     notes='Prespecified groups, unchanged causal forecasts and hard constraints. Same viewed history, not fresh holdout. No automatic strategy selection.')
    (p/'grid.json').write_text(json.dumps(report_data,allow_nan=False),encoding='utf-8')
    pd.DataFrame(summaries).drop(columns=['status_counts','fixed_strategy_8week_bootstrap_pnl_95ci_btc']).to_csv(p/'summary.csv',index=False)
    evidence=dict(status='PASS',checks=all_checks,check_count=len(all_checks),scenarios=len(summaries),balanced_matches_current=True,
                  source_forecast_sha256=hashes['forecasts'],source_lookahead_audit_checks=audit['check_count'],
                  grid_sha256=hashlib.sha256((p/'grid.json').read_bytes()).hexdigest(),
                  scope='All 1782 allocations use target-free decision inputs and preceding settled equity; constraints and cashflows checked. Forecast chronology inherited from audited common source.')
    (p/'validation.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    return report_data
