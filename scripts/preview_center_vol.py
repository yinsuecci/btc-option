"""Static preview of the current price / variance model and OOS allocations."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd


if __name__=='__main__':
    p=Path('results/center_vol')
    f=pd.read_csv(p/'forecasts.csv')
    w=pd.read_csv(p/'weekends.csv')
    w=w.loc[w.evaluation]
    dates=pd.to_datetime(f.weekend_id)
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,
                         'axes.edgecolor':'#bed7e8','axes.labelcolor':'#36556a','text.color':'#203d50',
                         'xtick.color':'#687f8c','ytick.color':'#687f8c'})
    fig,axes=plt.subplots(2,2,figsize=(14,8.5),facecolor='#eaf5fc')
    fig.suptitle('BTC Weekend Options: Price Center / Volatility / Allocation',fontsize=21,fontfamily='DejaVu Serif',x=.055,ha='left',y=.98)
    fig.text(.055,.92,'2022-12-31 to 2026-10-10 | 198 OOS weekends | Real trades, modelled execution | Not live trading',fontsize=11,color='#687f8c')
    ax=axes[0,0]
    ax.fill_between(dates,100*(f.lower_95_usd/f.spot-1),100*(f.upper_95_usd/f.spot-1),color='#dceef9',label='95% model interval')
    ax.plot(dates,100*(f.center_mean_usd/f.spot-1),color='#2784be',label='Expected delivery change',lw=1.5)
    ax.scatter(dates,100*(f.delivery_index/f.spot-1),color='#8194a3',s=6,label='Actual delivery change')
    ax.set(title='Predicted center and delivery interval',ylabel='Change from Saturday index (%)'); ax.legend(fontsize=8)
    ax=axes[0,1]
    ax.plot(dates,f.sigma_weekend*100,color='#2784be',label='Predicted sigma')
    ax.scatter(dates,np.abs(f.log_return-f.mu_log_return)*100,color='#8194a3',s=7,label='Actual absolute error')
    ax.set(title='Holding-period forecast-error volatility',ylabel='Log-return / error (%)'); ax.legend(fontsize=8)
    ax=axes[1,0]
    ax.plot(dates,w.equity_btc,color='#2784be',lw=2)
    ax.set(title='Two-stage allocation equity',ylabel='End-weekend equity (BTC)')
    ax=axes[1,1]
    ax.bar(dates,w.net_pnl_btc,width=5,color=np.where(w.net_pnl_btc>=0,'#54a4d4','#8194a3'))
    ax.set(title='Actual settlement PnL with assumed entry costs',ylabel='Net PnL (BTC)')
    for ax in axes.flat:
        ax.set_facecolor('white'); ax.grid(axis='y',color='#edf4f9'); ax.set_axisbelow(True)
        ax.set_xlabel('Saturday decision date (ET)')
        ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.tight_layout(rect=[.03,.03,.99,.89],h_pad=2.8,w_pad=3)
    fig.savefig(p/'overview.png',dpi=150,facecolor=fig.get_facecolor())
