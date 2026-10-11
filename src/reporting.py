import json
import html
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from src.core import FEATURES

def report(tables,obs,mode,p,engine):
    w=tables['weekends']; tr=tables['trades']; pnl=w.net_pnl.to_numpy(); eq=100000+np.cumsum(pnl); peak=np.maximum.accumulate(np.r_[100000,eq])[1:]; dd=eq-peak
    traded=w.loc[w.traded,'net_pnl']; tail=pnl[pnl<=np.quantile(pnl,.05)]
    metrics=dict(data_mode=mode,number_of_weekends=len(w),traded_weekends=int(w.traded.sum()),no_trade_weekends=int((~w.traded).sum()),gross_pnl=float(tr.gross_pnl.sum()) if len(tr) else 0,net_pnl=float(pnl.sum()),average_pnl=float(pnl.mean()),median_pnl=float(np.median(pnl)),win_rate=float((traded>0).mean()) if len(traded) else None,sharpe=float(np.mean(pnl)/np.std(pnl,ddof=1)*np.sqrt(52)) if np.std(pnl)>0 else None,max_drawdown=float(dd.min()),turnover=float(tr.quantity.sum()*2) if len(tr) else 0,average_holding_time=float(tr.holding_hours.mean()) if len(tr) else None,average_transaction_cost=float((tr.fees+tr.slippage).mean()) if len(tr) else None,worst_trade=float(tr.net_pnl.min()) if len(tr) else None,best_trade=float(tr.net_pnl.max()) if len(tr) else None,worst_weekend=float(pnl.min()),expected_shortfall=float(tail.mean()),backtest_start=w.weekend_id.min(),backtest_end=w.weekend_id.max(),nautilus=engine)
    corr=obs[FEATURES].corr(); warnings=[]
    pairs=[f'{a} / {b}' for i,a in enumerate(FEATURES) for b in FEATURES[i+1:] if abs(corr.loc[a,b])>.9]
    if pairs: warnings.append('Multicollinearity (absolute correlation > 0.9): '+', '.join(pairs))
    bh=tables['beta_history']; pivot=bh.loc[bh.model=='ridge'].pivot(index='weekend_id',columns='factor_name',values='beta')
    unstable=[c for c in pivot if ((np.sign(pivot[c]).diff().abs()>0).sum()>2)]
    if unstable: warnings.append('Factor instability, frequent coefficient sign changes: '+', '.join(unstable))
    metrics['warnings']=warnings
    (p/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False),encoding='utf-8')
    figs=p/'figures'; figs.mkdir(exist_ok=True); images=[]
    def plot(name,title,draw):
        fig,ax=plt.subplots(figsize=(10,4)); draw(ax); ax.set_title(title); fig.tight_layout(); path=figs/f'{name}.png'; fig.savefig(path,dpi=120); plt.close(fig); images.append((title,path))
    plot('equity_curve','Equity (USDT), initial capital 100,000',lambda a:a.plot(eq))
    plot('drawdown','Drawdown (USDT) by test weekend',lambda a:a.fill_between(range(len(dd)),dd))
    plot('weekend_pnl','Net PnL (USDT) by test weekend',lambda a:a.bar(range(len(pnl)),pnl))
    plot('pnl_distribution','Weekend net PnL distribution (USDT)',lambda a:a.hist(pnl,bins=12))
    hourly=w.loc[w.traded].groupby('entry_hour').net_pnl.agg(['mean','count','min',lambda x:(x>0).mean()]).reindex(range(24)); hourly.columns=['average_net_pnl','trade_count','tail_loss','win_rate']; hourly.to_csv(p/'entry_time_analysis.csv')
    def entry_plot(a):
        fig=a.figure; fig.clear(); axes=fig.subplots(2,2); fig.set_size_inches(10,7)
        for ax,column,unit in zip(axes.flat,hourly.columns,['USDT','weekends','USDT','fraction']):
            ax.bar(hourly.index,hourly[column]); ax.set_xlabel('Saturday ET entry hour'); ax.set_ylabel(unit); ax.set_title(column.replace('_',' ')); ax.set_xticks(range(0,24,3))
    plot('entry_time_analysis','Entry timing diagnostics',entry_plot)
    if len(tr):
        plot('strike_weights','Signed option quantity by strike (negative = short)',lambda a:[a.scatter(g.strike,g.weight,label=k) for k,g in tr.groupby('option_type')] and a.legend())
        def greek_plot(a):
            fig=a.figure; fig.clear(); axes=fig.subplots(2,2); fig.set_size_inches(10,7)
            values=tr.groupby('weekend_id')[['portfolio_delta','portfolio_gamma','portfolio_vega','portfolio_theta']].first()
            for ax,column,unit in zip(axes.flat,values.columns,['BTC delta','BTC delta / USDT','USDT / 1.0 IV','USDT / day']):
                ax.plot(values[column].to_numpy()); ax.set_title(column); ax.set_xlabel('Traded weekend index'); ax.set_ylabel(unit)
        plot('greeks_over_time','Entry portfolio Greeks',greek_plot)
    stress=tables['stress']
    if len(stress):
        surface=stress.groupby(['iv_shock','spot_shock']).estimated_pnl.min().unstack()
        def heat(a):
            im=a.imshow(surface,aspect='auto'); a.set_xticks(range(len(surface.columns)),[f'{x:.0%}' for x in surface.columns]); a.set_yticks(range(len(surface)),[f'+{x*100:.0f} vol pts' for x in surface.index]); a.set_xlabel('Spot shock'); a.set_ylabel('IV shock'); a.figure.colorbar(im,ax=a,label='Worst portfolio PnL (USDT)')
        plot('spot_iv_stress_heatmap','Nonlinear Black-Scholes stress, not a margin model',heat)
        stress.loc[[stress.estimated_pnl.idxmin()]].to_csv(p/'worst_scenario.csv',index=False)
    plot('beta_stability','Raw-unit Ridge factor coefficients by weekend',lambda a:[a.plot(pivot[c].to_numpy(),label=c) for c in pivot if c!='intercept'] and a.legend(fontsize=5,ncol=4))
    def correlation_plot(a):
        a.figure.set_size_inches(12,10); im=a.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm'); a.set_xticks(range(len(FEATURES)),FEATURES,rotation=90,fontsize=8); a.set_yticks(range(len(FEATURES)),FEATURES,fontsize=8); a.figure.colorbar(im,ax=a,label='Pearson correlation')
    plot('factor_correlation','Historical factor correlation (descriptive)',correlation_plot)
    interactive=go.Figure(go.Scatter(x=w.weekend_id,y=eq,name='Equity (USDT)')).to_html(full_html=False,include_plotlyjs=True)
    warning='THIS IS NOT A REAL HISTORICAL PERFORMANCE RESULT' if mode!='REAL' else 'Historical research, not a profit guarantee'
    body=f'<h1>BTC Weekend Options Research</h1><strong>DATA MODE: {mode}</strong><h2>{warning}</h2><p>Saturday ET, Sunday ET contract expiry, expanding walk-forward Ridge; OLS comparison. No live trading.</p><p>Weekly Sharpe uses 52 observations/year including no-trade weekends; small-sample diagnostic only.</p><pre>{html.escape(json.dumps(metrics,indent=2))}</pre>'+interactive
    for title,path in images: body+=f'<details open><summary>{html.escape(title)}</summary><img alt="{html.escape(title)}" src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}"></details>'
    (p/'report.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><style>body{max-width:1100px;margin:40px auto;padding:20px;font:16px Georgia;background:#f6f2e9;color:#16322e}h1{font-size:36px}strong,h2{color:#a33723}img{width:100%}pre{white-space:pre-wrap}summary{padding:20px;cursor:pointer}</style>'+body,encoding='utf-8')
    return metrics
