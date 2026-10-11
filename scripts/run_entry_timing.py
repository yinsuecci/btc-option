"""Fixed entry-time grid with public trade windows and chronological validation."""
import argparse
import gzip
import hashlib
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.center_vol import (CenterConfig, causal_mean_residuals, distribution_values,
                            fit_distribution, inverse_expected_payoff, payoff_covariance,
                            weekend_dataset)
from src.center_vol_metrics import performance_summary
from src.trade_data import complete_window, request_json
from src.trade_research import TradeConfig, build_observations, optimize_native

SLOTS = {'fri_16utc': -8, 'sat_00utc': 0, 'sat_08utc': 8,
         'sat_16utc': 16, 'sun_00utc': 24, 'sun_04utc': 28,
         'current_sat_12ny': None}
BASE = Path('data/deribit_trade_daily')
DATA = Path('data/entry_timing')
OUT = Path('results/entry_timing')
LOCK = threading.Lock()
LAST_REQUEST = [0.]


def decision_time(wid, offset):
    if offset is None:
        return (pd.Timestamp(wid, tz='America/New_York') + pd.DateOffset(hours=12)).tz_convert('UTC')
    return pd.Timestamp(wid, tz='UTC') + pd.Timedelta(hours=offset)


def fetch_window(job):
    name, offset, wid = job
    target = DATA/name/'raw'/(wid+'_trades.json.gz')
    t = decision_time(wid, offset)
    expiry = pd.Timestamp(wid, tz='UTC')+pd.Timedelta(hours=32)
    lo = int((t-pd.Timedelta(hours=1)).timestamp()*1000)
    hi = int(t.timestamp()*1000)
    if offset is None and not target.exists():
        shutil.copyfile(BASE/'raw'/target.name, target)
    if target.exists():
        with gzip.open(target, 'rt', encoding='utf-8') as stream:
            record = json.load(stream)
        if not record['complete'] or record['start_ms'] != lo or record['end_ms'] != hi:
            raise ValueError('Incompatible window: '+str(target))
    else:
        calls = [0]
        def fetch(a, b):
            with LOCK:
                time.sleep(max(0., .25-(time.monotonic()-LAST_REQUEST[0])))
                LAST_REQUEST[0] = time.monotonic()
            calls[0] += 1
            return request_json('https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time',
                                dict(currency='BTC', kind='option', start_timestamp=a,
                                     end_timestamp=b, count=1000, sorting='asc'))
        trades = complete_window(fetch, lo, hi)
        unique = {r['trade_id']:r for r in trades}
        record = dict(weekend_id=wid, start_ms=lo, end_ms=hi, complete=True,
                      request_count=calls[0], fetched_at=str(pd.Timestamp.now(tz='UTC')),
                      trades=sorted(unique.values(),key=lambda r:(r['timestamp'],r['trade_id'])))
        with gzip.open(target, 'wt', encoding='utf-8') as stream:
            json.dump(record, stream)
    if any(not lo<=r['timestamp']<=hi for r in record['trades']):
        raise AssertionError('Future or out-of-window trade')
    return name, dict(weekend_id=wid,decision_time=str(t),expiry=str(expiry),
                      status='DOWNLOADED_TRADE_WINDOW',window_complete=True,
                      trades=len(record['trades']),request_count=record['request_count'])


def acquire(workers=4):
    ids = pd.date_range('2022-01-01','2026-10-10',freq='W-SAT').strftime('%Y-%m-%d').tolist()
    coverage = {name:[] for name in SLOTS}
    jobs = []
    for name, offset in SLOTS.items():
        root = DATA/name
        (root/'raw').mkdir(parents=True,exist_ok=True)
        for filename in ['spot_daily.csv','delivery_prices.json']:
            shutil.copyfile(BASE/filename,root/filename)
        jobs.extend((name,offset,wid) for wid in ids)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for j,(name,row) in enumerate(pool.map(fetch_window,jobs)):
            coverage[name].append(row)
            pd.DataFrame(coverage[name]).to_csv(DATA/name/'download_coverage.csv',index=False)
            if (j+1)%25==0:
                print(f'WINDOW {j+1}/{len(jobs)} {name} {row["weekend_id"]} trades={row["trades"]}',flush=True)
    return ids


def simulate(name, offset, ids):
    p = OUT/name
    p.mkdir(parents=True,exist_ok=True)
    cfg = CenterConfig()
    execution = TradeConfig(risk_penalty_per_btc=.01)
    obs, coverage, rejected = build_observations(DATA/name,execution)
    frame = weekend_dataset(obs)
    residuals = causal_mean_residuals(frame,cfg)
    equity = 1.
    weekly, forecasts, positions, checks = [], [], [], []
    for wid in ids:
        t = decision_time(wid,offset)
        rows = obs.loc[obs.weekend_id.eq(wid)].copy()
        evaluation = wid >= '2022-12-31'
        status = 'WARMUP' if not evaluation else 'NO_ELIGIBLE_TRADES'
        q = np.zeros(len(rows))
        alpha = np.full(len(rows),np.nan)
        if evaluation and len(rows):
            train = frame.loc[(frame.weekend_id<wid)&(frame.label_available_at<t)]
            errors = residuals.loc[(residuals.weekend_id<wid)&(residuals.label_available_at<t)]
            if len(train)<52 or len(errors)<26:
                status = 'INSUFFICIENT_MATURED_TRAINING'
            else:
                mean, vol, cal, train, errors = fit_distribution(frame,residuals,t,wid,cfg)
                current = frame.loc[frame.weekend_id.eq(wid)]
                mu = float(mean.predict(current)[0])
                variance = float(np.clip(cal*np.exp(np.clip(vol.predict(current)[0],-30,5)),1e-6,.04))
                spot = float(rows.spot.iloc[0])
                payoff = inverse_expected_payoff(spot,rows.strike,rows.option_type,mu,variance)
                cov = payoff_covariance(spot,rows.strike,rows.option_type,mu,variance)
                prices = rows.trade_price_btc.to_numpy()
                alpha = .95*prices-np.minimum(.0003,.125*prices)-payoff
                clean = rows.drop(columns=['delivery_index','settlement_btc','y_btc','label_available_at'])
                result = optimize_native(clean,alpha,execution,equity,cov,1.)
                status = result['status'] if result else 'NO_EQUITY'
                q = result['quantity'] if status=='ENTER' else q
                if status=='ENTER':
                    if not (np.all(q>=0) and np.all(q<=np.minimum(.5,rows.trade_amount)+1e-7)
                            and np.allclose(q/.1,np.round(q/.1)) and q.sum()<=2+1e-7
                            and abs(result['delta_log_btc'])<=.30+1e-7
                            and abs(result['vega_btc'])<=.20+1e-7
                            and result['margin_proxy_btc']<=min(.5,.5*equity)+1e-7
                            and result['worst_stress_btc']<=min(.25,.25*equity)+1e-7):
                        raise AssertionError('Allocation constraint violation')
                checks.append(bool(train.label_available_at.max()<t and errors.label_available_at.max()<t
                                   and rows.trade_time.le(t).all() and rows.spot_available_at.lt(t).all()))
                forecasts.append(dict(weekend_id=wid,decision_time=str(t),
                                      mean_last_label=str(train.label_available_at.max()),
                                      variance_last_label=str(errors.label_available_at.max()),
                                      mean_train_rows=len(train),variance_rows=len(errors),
                                      **distribution_values(spot,mu,variance)))
        pnl = float(q@rows.y_btc.to_numpy()) if len(rows) else 0.
        equity += pnl
        weekly.append(dict(weekend_id=wid,evaluation=evaluation,status=status,traded=bool(q.sum()),
                           net_pnl_btc=pnl,equity_btc=equity,quantity=float(q.sum()),candidates=len(rows)))
        for j,leg in enumerate(rows.itertuples()):
            positions.append(dict(weekend_id=wid,symbol=leg.symbol,quantity=float(q[j]),omega=float(-q[j]),
                                  predicted_net_per_unit_btc=float(alpha[j]),net_pnl_btc=float(q[j]*leg.y_btc)))
    if not all(checks):
        raise AssertionError('Chronology check failed')
    w = pd.DataFrame(weekly)
    metric, curve = performance_summary(w)
    oos = w.loc[w.evaluation]
    years = (pd.Timestamp(ids[-1])-pd.Timestamp('2022-12-31')).days/365.25
    metric.update(slot=name,offset_utc_hours=offset,net_pnl_btc=float(oos.net_pnl_btc.sum()),
                  cagr=equity**(1/years)-1,final_equity_btc=equity,traded_weekends=int(oos.traded.sum()),
                  oos_weekends=len(oos),average_quantity=float(oos.quantity.mean()),
                  eligible_weekends=frame.shape[0],candidate_legs=len(obs),chronology_checks=len(checks),
                  status_counts=oos.status.value_counts().to_dict())
    for label,table in [('observations',obs),('coverage',coverage),('rejected',rejected),('price_training',frame),
                        ('causal_mean_residuals',residuals),('forecasts',pd.DataFrame(forecasts)),
                        ('positions_by_weekend',pd.DataFrame(positions)),('weekends',w),('performance_curve',curve)]:
        table.to_csv(p/(label+'.csv'),index=False)
    (p/'metrics.json').write_text(json.dumps(metric,indent=2,allow_nan=False),encoding='utf-8')
    print('RESULT '+json.dumps(metric),flush=True)
    return metric,w


def split_summary(w):
    evaluated = w.loc[w.evaluation].reset_index(drop=True)
    result = {}
    for name,part in [('discovery',evaluated.iloc[:104]),('holdout',evaluated.iloc[104:])]:
        part = part.copy()
        part['equity_btc'] = 1.+part.net_pnl_btc.cumsum()
        metrics,_ = performance_summary(part)
        years = (pd.Timestamp(part.weekend_id.iloc[-1])-pd.Timestamp(part.weekend_id.iloc[0])).days/365.25
        result[name] = dict(net_pnl_btc=float(part.net_pnl_btc.sum()),
                            cagr=float(part.equity_btc.iloc[-1]**(1/years)-1),
                            sharpe=metrics['weekly_return_sharpe'],max_drawdown_pct=metrics['max_drawdown_pct'],
                            weekends=len(part),start=part.weekend_id.iloc[0],end=part.weekend_id.iloc[-1])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download-only',action='store_true')
    parser.add_argument('--cached',action='store_true')
    parser.add_argument('--wait-for-download',action='store_true')
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    ids = pd.date_range('2022-01-01','2026-10-10',freq='W-SAT').strftime('%Y-%m-%d').tolist()
    if not args.cached:
        acquire(args.workers)
    if args.download_only:
        return
    summaries, bundles = [], {}
    for name,offset in SLOTS.items():
        if args.wait_for_download:
            path=DATA/name/'download_coverage.csv'
            while True:
                try:
                    if path.exists() and len(pd.read_csv(path))==len(ids):
                        break
                except (pd.errors.EmptyDataError,pd.errors.ParserError):
                    pass
                time.sleep(5)
        metric,w = simulate(name,offset,ids)
        metric.update(split_summary(w))
        summaries.append(metric)
        bundles[name] = w
    best = max(summaries,key=lambda m:m['net_pnl_btc'])['slot']
    selected = max(summaries,key=lambda m:m['discovery']['net_pnl_btc'])['slot']
    baseline = bundles['current_sat_12ny']
    original = pd.read_csv('results/center_vol/weekends.csv',float_precision='round_trip')
    match = np.allclose(baseline.net_pnl_btc,original.net_pnl_btc,rtol=0,atol=1e-12)
    if not match:
        raise AssertionError('Baseline failed to reproduce existing strategy')
    summary = dict(protocol='Seven fixed time slots; same 198 calendar weeks; missing candidates are idle; slot-specific causal refits',
                   full_history_winner=best,discovery_selected_slot=selected,
                   selection_objective='Net BTC PnL on first 104 evaluated weekends',
                   holdout_caveat='Temporal selection validation on existing historical data, not a newly collected unseen holdout',
                   baseline_reproduced=bool(match),summaries=summaries)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    pd.DataFrame([{k:v for k,v in m.items() if not isinstance(v,dict)} for m in summaries]).to_csv(OUT/'summary.csv',index=False)
    evidence = dict(status='PASS',baseline_reproduced=bool(match),slots=len(summaries),
                    chronology_decisions=sum(m['chronology_checks'] for m in summaries),
                    evaluated_weekends_per_slot=198,raw_windows=250*len(SLOTS),
                    summary_sha256=hashlib.sha256((OUT/'summary.json').read_bytes()).hexdigest(),
                    baseline_weekends_sha256=hashlib.sha256(Path('results/center_vol/weekends.csv').read_bytes()).hexdigest(),
                    source_hashes={name:hashlib.sha256((BASE/name).read_bytes()).hexdigest()
                                   for name in ['spot_daily.csv','delivery_prices.json']})
    (OUT/'validation.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    from src.entry_timing_report import render
    render(OUT)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
