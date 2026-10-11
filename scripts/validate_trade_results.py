"""Independent audit of generated research artifacts and static visual preview."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from src.core import ET
from src.trade_research import FEATURES, TradeConfig, contract, per_unit_cashflows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--browser-verified', action='store_true',
                        help='Record completed manual browser verification of this report.')
    args = parser.parse_args()
    data = Path('data/deribit_trade_daily')
    result = Path('results/trade_only')
    cfg = TradeConfig(**json.loads((result/'config.json').read_text()))
    obs = pd.read_csv(result/'observations.csv')
    w = pd.read_csv(result/'weekends.csv')
    trades = pd.read_csv(result/'trades.csv')
    beta = pd.read_csv(result/'beta_history.csv')
    decisions = pd.read_csv(result/'decisions.csv')
    ledger = pd.read_csv(result/'event_ledger.csv')
    for col in ['decision_time','trade_time','spot_available_at','label_available_at','expiry']:
        obs[col] = pd.to_datetime(obs[col], utc=True, format='mixed')
    checks = []

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)

    raw_files=[]
    raw_count=0
    for path in sorted((data/'raw').glob('*_trades.json.gz')):
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            record=json.load(stream)
        check('complete_window_'+record['weekend_id'],record['complete'])
        check('bounded_prints_'+record['weekend_id'],all(record['start_ms'] <= r['timestamp'] <= record['end_ms'] for r in record['trades']))
        check('unique_prints_'+record['weekend_id'],len({r['trade_id'] for r in record['trades']}) == len(record['trades']))
        raw_count+=len(record['trades'])
        raw_files.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),rows=len(record['trades'])))
    check('250_expired_weekends_audited',len(raw_files)==250)
    check('all_250_have_candidates',obs.weekend_id.nunique()==250)
    check('saturday_ET_12',((obs.decision_time.dt.tz_convert(ET).dt.weekday==5) & (obs.decision_time.dt.tz_convert(ET).dt.hour==12)).all())
    check('sunday_ET_expiry',obs.expiry.dt.tz_convert(ET).dt.weekday.eq(6).all())
    check('no_future_trade',obs.trade_time.le(obs.decision_time).all())
    check('no_future_daily_candle',obs.spot_available_at.lt(obs.decision_time).all())
    check('60_minute_trade_age',obs.trade_age_minutes.between(0,60).all())
    deliveries={r['date']:r['delivery_price'] for r in json.loads((data/'delivery_prices.json').read_text())}
    correct=[]
    for r in obs.itertuples():
        expiry,k,kind=contract(r.symbol)
        d=deliveries[expiry.date().isoformat()]
        cash=per_unit_cashflows(r.trade_price_btc,d,k,kind,cfg)
        correct.append(abs(cash['y_btc']-r.y_btc)<1e-12 and abs(d-r.delivery_index)<1e-8)
    check('all_labels_match_official_delivery',all(correct))
    check('training_precedes_test',beta.train_end.lt(beta.weekend_id).all())
    check('training_labels_mature',pd.to_datetime(beta.last_label_available_at,utc=True).lt(pd.to_datetime(beta.decision_time,utc=True)).all())
    check('warmup_and_test_partition',w.evaluation.sum()+w.status.eq('WARMUP').sum()==250)
    check('26_valid_weekend_warmup',w.status.eq('WARMUP').sum()==cfg.min_train_weekends)
    check('nonnegative_quantity',trades.quantity.ge(0).all())
    check('signed_omega',np.allclose(trades.weight,-trades.quantity))
    check('0_1_lot',np.allclose(trades.quantity/cfg.lot,np.round(trades.quantity/cfg.lot)))
    check('leg_limit',trades.quantity.le(cfg.max_leg_quantity+1e-8).all())
    check('past_size_proxy_limit',trades.quantity.le(trades.trade_amount+1e-8).all())
    check('gross_quantity_limit',decisions.quantity.le(cfg.gross_limit+1e-7).all())
    check('delta_constraint',decisions.delta_log_btc.abs().le(cfg.delta_log_limit+1e-7).all())
    check('vega_constraint',decisions.vega_btc.abs().le(cfg.vega_btc_limit+1e-7).all())
    before=dict(zip(w.weekend_id,np.r_[cfg.initial_equity_btc,w.equity_btc.to_numpy()[:-1]]))
    check('dynamic_margin_constraint',all(r.margin_proxy_btc<=min(cfg.margin_proxy_limit,.5*before[r.weekend_id])+1e-7 for r in decisions.itertuples()))
    check('dynamic_stress_constraint',all(r.worst_stress_btc<=min(cfg.stress_loss_limit,.25*before[r.weekend_id])+1e-7 for r in decisions.itertuples()))
    check('leg_cashflow_reconciliation',np.allclose(trades.net_pnl_btc,trades.gross_pnl_btc-trades.costs_btc))
    check('weekend_cashflow_reconciliation',abs(trades.net_pnl_btc.sum()-w.net_pnl_btc.sum())<1e-10)
    check('event_cashflow_reconciliation',abs(ledger.cashflow_btc.sum()-w.net_pnl_btc.sum())<1e-10)
    check('equity_reconciliation',np.allclose(w.equity_btc,cfg.initial_equity_btc+w.net_pnl_btc.cumsum()))
    formula=json.loads((result/'final_formula.json').read_text())
    check('formula_features',set(formula['beta_raw_units'])==set(FEATURES))
    page=(result/'report.html').read_text(encoding='utf-8')
    check('no_external_script_dependency','<script src=' not in page)
    check('light_blue_theme','--sky:#eaf5fc' in page and '--white:#fff' in page)
    check('correct_optimizer_formula','alpha × q − 0.02 ×' in page)
    check('formula_in_report',f'{formula["intercept_btc"]:.12g}' in page)
    manifest=dict(status='PASS',check_count=len(checks),checks=checks,
                  real_raw_trade_rows=raw_count,raw_weekends=len(raw_files),
                  raw_files=raw_files,observations=len(obs),future_weekends_excluded=11,
                  html_browser_runtime_verified=args.browser_verified)
    (result/'validation.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

    test=w.loc[w.evaluation]
    dates=pd.to_datetime(test.weekend_id)
    equity=cfg.initial_equity_btc+test.net_pnl_btc.cumsum()
    baseline=cfg.initial_equity_btc+test.baseline_pnl_btc.cumsum()
    peak=np.maximum.accumulate(np.r_[cfg.initial_equity_btc,equity])[1:]
    annual=test.assign(year=test.weekend_id.str[:4]).groupby('year').net_pnl_btc.sum()
    sens=pd.read_csv(result/'slippage_sensitivity.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
                         'axes.edgecolor':'#bed7e8','axes.labelcolor':'#36556a','text.color':'#203d50',
                         'xtick.color':'#687f8c','ytick.color':'#687f8c'})
    fig,axes=plt.subplots(2,2,figsize=(14,8.5),facecolor='#eaf5fc')
    fig.suptitle('BTC Weekend Options | Real Trade-only Research',fontsize=22,fontfamily='DejaVu Serif',x=.055,ha='left',y=.98)
    fig.text(.055,.922,f'2022-07-02 to 2026-10-10  |  224 OOS weekends  |  Net {test.net_pnl_btc.sum():+.6f} BTC  |  Modelled execution, not live fills',fontsize=11,color='#687f8c')
    ax=axes[0,0]; ax.plot(dates,equity,color='#2784be',label='Walk-forward Ridge',lw=2); ax.plot(dates,baseline,color='#8bacbf',ls='--',label='Trade-print baseline')
    ax.set(title='End-weekend equity',ylabel='BTC',xlabel='Saturday date (ET)'); ax.legend(fontsize=9)
    ax=axes[0,1]; ax.fill_between(dates,equity-peak,0,color='#b4d9ee'); ax.set(title='End-weekend drawdown',ylabel='BTC from prior peak',xlabel='Saturday date (ET)')
    ax=axes[1,0]; ax.bar(annual.index,annual.values,color=['#54a4d4' if x>=0 else '#8194a3' for x in annual]); ax.axhline(0,color='#bed7e8',lw=1); ax.set(title='Annual OOS net PnL',ylabel='BTC',xlabel='Calendar year (partial 2022 and 2026)')
    ax=axes[1,1]; ax.plot(sens.slippage_fraction*100,sens.net_pnl_btc,color='#2784be',marker='o'); ax.axhline(0,color='#bed7e8',lw=1); ax.set(title='Fixed-portfolio slippage sensitivity',ylabel='Total OOS net PnL (BTC)',xlabel='Premium haircut (%)')
    for ax in axes.flat:
        ax.set_facecolor('white'); ax.grid(axis='y',color='#edf4f9'); ax.set_axisbelow(True)
    for ax in axes[0]:
        ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.tight_layout(rect=[.03,.03,.99,.89],h_pad=2.8,w_pad=3)
    fig.savefig(result/'overview.png',dpi=150,facecolor=fig.get_facecolor())
    plt.close(fig)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['checks','raw_files']},indent=2))


if __name__ == '__main__':
    main()
