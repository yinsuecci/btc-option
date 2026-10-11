"""Offline, light-blue and white report for real trade-only research."""
import html
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.trade_research import FEATURES

LABELS = {
    'momentum_1d': '1-day momentum', 'momentum_3d': '3-day momentum',
    'momentum_7d': '7-day momentum', 'rv_7d': '7-day realized vol',
    'rv_30d': '30-day realized vol', 'spot_volume_z30': 'Volume z-score (30d)',
    'jump_z': 'Absolute return / 30d daily vol', 'trade_iv': 'Trade IV (decimal)',
    'iv_minus_rv30': 'IV minus RV30', 'log_moneyness': 'ln(index / strike)',
    'days_to_expiry': 'Days to expiry', 'trade_price_btc': 'Sell print premium (BTC)',
    'delta_log_model': 'Model log-spot delta (BTC)', 'gamma_log_model': 'Model log-spot gamma (BTC)',
    'vega_btc_model': 'Model vega (BTC / 1.0 IV)', 'theta_btc_model': 'Model theta (BTC / day)',
    'is_call': 'Call indicator', 'trade_age_minutes': 'Sell-print age (minutes)',
    'log_trade_amount': 'ln(1 + print amount)', 'log_window_trade_count': 'ln(1 + sell print count)',
}


def report(tables, obs, coverage, formula, cfg, p):
    all_w = tables['weekends']
    w = all_w.loc[all_w.evaluation].copy()
    if w.empty:
        raise ValueError('No out-of-sample weekends')
    pnl = w.net_pnl_btc.to_numpy()
    equity = cfg.initial_equity_btc + np.cumsum(pnl)
    peaks = np.maximum.accumulate(np.r_[cfg.initial_equity_btc, equity])[1:]
    dd = equity - peaks
    tr = tables['trades']
    prediction = tables['predictions'].dropna(subset=['realized_per_unit_btc'])
    sigma = float(np.std(pnl, ddof=1))
    span = (pd.Timestamp(w.weekend_id.iloc[-1]) - pd.Timestamp(w.weekend_id.iloc[0])).days / 365.25
    tail = pnl[pnl <= np.quantile(pnl, .05)]
    metrics = dict(data_mode='REAL_TRADES_WITH_MODELLED_EXECUTION_AND_GREEKS',
                   source='Deribit public history; official Deribit delivery index; Binance closed daily BTCUSDT candles',
                   requested_calendar_start='2022-01-01', requested_calendar_end='2026-12-31',
                   acquired_weekends=len(all_w), future_unsettled_weekends=int(coverage.status.eq('FUTURE_OR_UNSETTLED').sum()),
                   candidate_weekends=int(obs.weekend_id.nunique()), eligible_observations=len(obs),
                   warmup_weekends=int(all_w.status.eq('WARMUP').sum()),
                   oos_weekends=len(w), traded_weekends=int(w.traded.sum()),
                   no_trade_weekends=int((~w.traded).sum()), trade_legs=len(tr),
                   oos_start=w.weekend_id.iloc[0], oos_end=w.weekend_id.iloc[-1],
                   initial_equity_btc=cfg.initial_equity_btc, final_equity_btc=float(equity[-1]),
                   net_pnl_btc=float(pnl.sum()), return_pct=float((equity[-1]/cfg.initial_equity_btc-1)*100),
                   gross_pnl_btc=float(w.gross_pnl_btc.sum()), costs_btc=float(w.costs_btc.sum()),
                   weekly_sharpe=float(np.mean(pnl)/sigma*np.sqrt(52)) if sigma > 0 else None,
                   win_rate_traded=float((w.loc[w.traded, 'net_pnl_btc'] > 0).mean()) if w.traded.any() else None,
                   maximum_weekend_drawdown_btc=float(dd.min()),
                   maximum_weekend_drawdown_pct=float(np.min(dd/peaks)*100),
                   worst_weekend_btc=float(pnl.min()), best_weekend_btc=float(pnl.max()),
                   expected_shortfall_5pct_btc=float(tail.mean()),
                   baseline_net_pnl_btc=float(w.baseline_pnl_btc.sum()),
                   annualized_growth=float((equity[-1]/cfg.initial_equity_btc)**(1/span)-1) if span > 0 and equity[-1] > 0 else None,
                   formula_training_start=formula['train_start'], formula_training_end=formula['train_end'],
                   formula_training_samples=formula['fitted_samples'], formula_training_weekends=formula['fitted_weekends'],
                   event_ledger_reconciled=True, exchange_orderbook_replay=False)
    for model in ('ridge', 'ols'):
        residual = prediction[model+'_prediction_btc'] - prediction.realized_per_unit_btc
        denominator = np.sum((prediction.realized_per_unit_btc - prediction.realized_per_unit_btc.mean())**2)
        metrics[model+'_oos_rmse_btc'] = float(np.sqrt(np.mean(residual**2)))
        metrics[model+'_oos_r2'] = float(1 - np.sum(residual**2)/denominator) if denominator else None
    corr = obs[FEATURES].corr()
    pairs = [f'{a} / {b}' for i, a in enumerate(FEATURES) for b in FEATURES[i+1:] if abs(corr.loc[a,b]) > .9]
    metrics['multicollinear_pairs'] = pairs
    metrics['limitations'] = [
        'Past sell prints are execution proxies, not contemporaneous bids or guaranteed fills.',
        'Only traded contracts in the 60-minute Saturday window enter the observed universe.',
        'Greeks and stress values are model-derived; no archived bid/ask, OI or exchange margin.',
        'BTC denomination; Binance BTCUSDT daily factors are not Deribit settlement prices.',
        'Weekend-close drawdowns omit intra-weekend losses, liquidation and funding/collateral opportunity costs.',
        'Slippage sensitivity holds the base fitted model and quantities fixed; it is not a reoptimized strategy.',
        'Final full-sample coefficients are retrospective and were not used for earlier OOS predictions.',
        'Paper event cash ledger, not Nautilus orderbook execution; daily fee assumptions are not account-specific realized fees.',
    ]
    design = (obs[FEATURES].to_numpy() - np.array([formula['scaler_mean'][f] for f in FEATURES])) / np.array([formula['scaler_scale'][f] for f in FEATURES])
    metrics['standardized_design_rank'] = int(np.linalg.matrix_rank(design))
    metrics['number_of_features'] = len(FEATURES)
    metrics['structural_factor_dependency'] = 'iv_minus_rv30 = trade_iv - rv_30d; OLS coefficients are not uniquely identified. Ridge regularizes this dependence.'
    if metrics['ridge_oos_r2'] is not None and metrics['ridge_oos_r2'] < 0:
        metrics['limitations'].append('Negative OOS R-squared: the factor forecast is worse than the pooled realized-label mean in squared error; do not treat this as a validated predictive edge.')
    rng = np.random.default_rng(20261011)
    block = min(8, len(pnl))
    starts = rng.integers(0, max(len(pnl)-block+1, 1), size=(5000, int(np.ceil(len(pnl)/block))))
    indexes = (starts[:,:,None] + np.arange(block)).reshape(5000,-1)[:,:len(pnl)]
    totals = pnl[indexes].sum(axis=1)
    metrics['fixed_strategy_8week_block_bootstrap_net_pnl_95ci_btc'] = list(map(float, np.quantile(totals, [.025,.975])))
    (p/'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    year = w.assign(year=w.weekend_id.str[:4]).groupby('year').agg(
        weekends=('weekend_id','count'), traded=('traded','sum'), net_btc=('net_pnl_btc','sum'),
        gross_btc=('gross_pnl_btc','sum'), costs_btc=('costs_btc','sum'), baseline_btc=('baseline_pnl_btc','sum'))
    year.to_csv(p/'yearly_summary.csv')
    figs = p/'figures'
    figs.mkdir(exist_ok=True)
    charts = []

    def add(fig, name, title, height=360):
        fig.update_layout(template='plotly_white', title=title, height=height,
                          font=dict(family='Verdana', color='#203d50', size=12),
                          paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
                          colorway=['#2784be','#8bacbf','#4c6574'],
                          margin=dict(l=58,r=28,t=58,b=54),
                          legend=dict(orientation='h',y=1.10,x=0), hovermode='x unified')
        fig.update_xaxes(gridcolor='#edf4f9', zerolinecolor='#d4e6f2')
        fig.update_yaxes(gridcolor='#edf4f9', zerolinecolor='#d4e6f2')
        fig.write_html(figs/(name+'.html'), include_plotlyjs=True)
        charts.append((name,title,fig))
        return fig.to_html(full_html=False,include_plotlyjs=True if len(charts)==1 else False,
                           config={'responsive':True,'displaylogo':False},div_id=name)

    figure = go.Figure()
    figure.add_scatter(x=w.weekend_id,y=equity,name='Walk-forward Ridge',mode='lines',
                       line=dict(width=2.5))
    baseline = cfg.initial_equity_btc + w.baseline_pnl_btc.cumsum()
    figure.add_scatter(x=w.weekend_id,y=baseline,name='0.35-delta trade-print baseline',mode='lines',line=dict(dash='dot'))
    figure.update_xaxes(title='Saturday weekend (ET)')
    figure.update_yaxes(title='End-weekend equity (BTC)')
    equity_html = add(figure,'equity','样本外累计权益 / BTC',420)
    fig=go.Figure(go.Bar(x=w.weekend_id,y=pnl,name='Net weekend PnL',marker_color=np.where(pnl>=0,'#54a4d4','#8194a3')))
    fig.update_xaxes(title='Saturday weekend (ET)'); fig.update_yaxes(title='Net PnL (BTC)')
    pnl_html=add(fig,'pnl','逐周末净损益')
    fig=go.Figure(go.Scatter(x=w.weekend_id,y=dd,name='Drawdown',fill='tozeroy',line=dict(color='#62869e')))
    fig.update_xaxes(title='Saturday weekend (ET)'); fig.update_yaxes(title='Drawdown from prior peak (BTC)')
    dd_html=add(fig,'drawdown','周末结算权益回撤')
    contributions=pd.Series({f:formula['beta_raw_units'][f]*formula['scaler_scale'][f] for f in FEATURES}).sort_values()
    fig=go.Figure(go.Bar(x=contributions.values,y=[LABELS[f] for f in contributions.index],orientation='h',marker_color='#6eafd8'))
    fig.update_xaxes(title='Final beta x training feature SD (BTC per contract)'); fig.update_yaxes(title='Factor')
    beta_html=add(fig,'beta','最终 Ridge 参数：每一标准差的预测影响',560)
    fig=go.Figure(go.Histogram(x=pnl,nbinsx=30,name='Weekend count',marker_color='#6eafd8'))
    fig.update_xaxes(title='Net weekend PnL (BTC)'); fig.update_yaxes(title='Number of OOS weekends')
    hist_html=add(fig,'distribution','周末收益分布')
    piv=tables['beta_history'].loc[tables['beta_history'].model.eq('ridge')].pivot(index='weekend_id',columns='factor',values='beta')
    leading=list(contributions.abs().nlargest(4).index)
    fig=go.Figure()
    for f in leading:
        fig.add_scatter(x=piv.index,y=piv[f]*formula['scaler_scale'][f],name=LABELS[f],mode='lines')
    fig.update_xaxes(title='Saturday model fit date (ET)'); fig.update_yaxes(title='Beta x final training SD (BTC)')
    stability_html=add(fig,'stability','前四个因素的滚动系数',410)
    stress_html=''
    if len(tables['stress']):
        matrix=tables['stress'].groupby(['iv_shock','spot_shock']).estimated_pnl_btc.min().unstack()
        fig=go.Figure(go.Heatmap(z=matrix.values,x=matrix.columns,y=matrix.index,colorscale='Blues',
                                colorbar=dict(title='BTC'),hovertemplate='Spot %{x:.0%}<br>IV +%{y:.0%}<br>Worst PnL %{z:.6f} BTC<extra></extra>'))
        fig.update_xaxes(title='Immediate spot shock (fraction)',tickformat='.0%')
        fig.update_yaxes(title='IV shock (absolute decimal)',tickformat='.0%')
        stress_html=add(fig,'stress','模型压力情景：各周末最差组合值')
    fig=go.Figure(go.Heatmap(z=corr.values,x=[LABELS[f] for f in FEATURES],y=[LABELS[f] for f in FEATURES],
                            zmin=-1,zmax=1,colorscale=[[0,'#d7e4ed'],[.5,'#ffffff'],[1,'#2584bd']],colorbar=dict(title='Pearson r')))
    fig.update_xaxes(title='Factor'); fig.update_yaxes(title='Factor')
    corr_html=add(fig,'correlation','因子相关性：描述性诊断',700)
    fig=go.Figure()
    for model in ('ridge','ols'):
        fig.add_scatter(x=prediction.realized_per_unit_btc,y=prediction[model+'_prediction_btc'],mode='markers',
                       name=model.upper(),marker=dict(size=5,opacity=.4))
    fig.update_xaxes(title='Realized net label (BTC / unit)'); fig.update_yaxes(title='OOS predicted net label (BTC / unit)')
    comparison_html=add(fig,'prediction','样本外预测与真实结算标签')

    coefficients='\n'.join(f'{formula["beta_raw_units"][f]:+.12g} * {f}' for f in FEATURES)
    expression=f'alpha_hat_BTC = {formula["intercept_btc"]:.12g}\n'+coefficients
    final_formula = f'''# Final Trade-only Formulas

## Full-sample Ridge forecast (BTC per one BTC option notional)

Training {formula['train_start']} through {formula['train_end']}, {formula['fitted_weekends']} eligible weekends,
{formula['fitted_samples']} contract observations. Retrospective final fit, not backfilled into OOS performance.

```text
{expression}
```

## Label and cashflow

Call settlement_BTC = max(D - K, 0) / D.
Put settlement_BTC = max(K - D, 0) / D.
Fee_BTC = min(0.0003, 0.125 * P).
Net label_BTC = 0.95 * P - settlement_BTC - Fee_BTC.
Sunday daily expiry delivery fee = 0 under the research fee assumption.
Weekend net PnL_BTC = sum(q_i * net_label_i), omega_i = -q_i.
No hedge or other trading cashflow is included.

## Ridge estimation

z_j = (x_j - mean_train_j) / SD_train_j.
(a, b) = argmin sum((y - a - z*b)^2) + 10 * sum(b_j^2).
beta_j = b_j / SD_train_j; beta_0 = a - sum(beta_j * mean_train_j).
OLS uses the same standardization with no coefficient penalty.
There is an exact feature dependence: iv_minus_rv30 = trade_iv - rv_30d.
OLS coefficient decomposition is not unique; Ridge selects a regularized solution.
Every historical model uses only earlier weekends and label_available_at < decision_time.

## Model Greeks, inverse BTC value

V_BTC = BS_USD(S, K, tau, trade_IV) / S, rate = 0.
delta_log = delta_USD - V_BTC.
gamma_log = S * gamma_USD - delta_USD + V_BTC.
vega_BTC = vega_USD / S (per 1.0 absolute IV).
theta_BTC = theta_USD / S (per calendar day).

## Weight optimization

q* = argmax(alpha*q - {cfg.risk_penalty_per_btc} * sum((risk_i * q_i)^2)); omega = -q.
risk_i = max(0.001, max_scenario(V_BTC_i_scenario - effective_entry_premium_i)).
Scenario spot shocks = -30%, -15%, -5%, 0%, +5%, +15%, +30%.
Scenario IV additions = 0, 0.25, 0.50. These are MODEL shocks, not observed prices.
0 <= q_i <= min(0.5, past sell-print amount_i); sum(q) <= 2.
abs(sum(q_i * delta_log_i)) <= 0.30 BTC.
abs(sum(q_i * vega_BTC_i)) <= 0.20 BTC per 1.0 IV.
margin_i = max(0.15 - OTM_i/S, 0.10) + P_i.
sum(q_i * margin_i) <= min(0.50 BTC, 0.5 * pre-entry equity_BTC).
max_scenario sum(q_i * scenario_loss_i) <= min(0.25 BTC, 0.25 * equity_BTC).
Round quantities DOWN to 0.1 and recheck all constraints; otherwise no trade.
Enter only when rounded quantity > 0 and predicted portfolio net PnL > 0.
Past print size is NOT observed order-book capacity; margin_i is a research proxy.

## Daily market factors

momentum_nd = last_closed_daily_close / closed_close_n_days_ago - 1, n = 1,3,7.
rv_nd = sample_SD(last n daily log returns) * sqrt(365), n = 7,30.
volume_z30 = (last_daily_volume - mean_30d) / SD_30d.
jump_z = abs(last_daily_log_return) / SD_30d_daily_log_returns.
trade_IV is the actual printed IV / 100. All feature data are available by decision.

## Important limits

Sell-print execution proxy, maximum 60-minute age, Saturday 12:00 ET decision.
Daily factors, official Sunday 08:00 UTC delivery. No contemporaneous bid/ask.
No full option-chain coverage, intraday liquidation simulation, actual account fees or guaranteed fills.
``final_ols_formula.json`` contains the full-sample comparison OLS coefficients.
'''
    (p/'FORMULAS.md').write_text(final_formula,encoding='utf-8')
    rows=''.join(f'<tr><td>{html.escape(LABELS[f])}<small>{f}</small></td><td>{formula["beta_raw_units"][f]:+.9g}</td><td>{formula["scaler_mean"][f]:.6g}</td><td>{formula["scaler_scale"][f]:.6g}</td></tr>' for f in FEATURES)
    yearly=''.join(f'<tr><td>{y}</td><td>{int(r.weekends)}</td><td>{int(r.traded)}</td><td>{r.net_btc:+.6f}</td><td>{r.costs_btc:.6f}</td><td>{r.baseline_btc:+.6f}</td></tr>' for y,r in year.iterrows())
    sensitivity=pd.read_csv(p/'slippage_sensitivity.csv')
    sens=''.join(f'<tr><td>{r.slippage_fraction:.0%}</td><td>{r.net_pnl_btc:+.6f} BTC</td></tr>' for r in sensitivity.itertuples())
    limitations=''.join('<li>'+html.escape(x)+'</li>' for x in metrics['limitations'])
    breakdown=coverage.status.value_counts().to_dict()
    info=json.dumps(dict(metrics=metrics,coverage_counts=breakdown),ensure_ascii=False,indent=2)
    drill=[]
    for r in tr.itertuples():
        drill.append(dict(weekend=r.weekend_id,symbol=r.symbol,quantity=r.quantity,omega=r.weight,
                          age=r.trade_age_minutes,premium=r.trade_price_btc,alpha=r.alpha_hat_btc,
                          settlement=r.settlement_btc,pnl=r.net_pnl_btc,iv=r.trade_iv,
                          trade_time=str(r.trade_time),decision_time=str(r.decision_time),expiry=str(r.expiry)))
    drill_json=json.dumps(drill,ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    options=''.join(f'<option value="{wid}">{wid}</option>' for wid in sorted(tr.weekend_id.unique(),reverse=True)) if len(tr) else ''
    detail=f'<select id="weekend-select">{options}</select><div class="scroll"><table><thead><tr><th>合约</th><th>omega</th><th>成交价 BTC</th><th>预测净收益</th><th>结算 BTC</th><th>净损益 BTC</th><th>成交年龄</th></tr></thead><tbody id="legs"></tbody></table></div>' if options else '<p>当前模型没有选中交易组合。</p>'
    cards=f'''<div class="numbers"><div><span>样本外净收益</span><b>{metrics['net_pnl_btc']:+.6f}<small> BTC</small></b></div><div><span>期末权益 / 起始 1 BTC</span><b>{metrics['final_equity_btc']:.6f}<small> BTC</small></b></div><div><span>已交易 / 样本外周末</span><b>{metrics['traded_weekends']}<small> / {metrics['oos_weekends']}</small></b></div></div>'''
    sharpe_text = f"{metrics['weekly_sharpe']:.3f}" if metrics['weekly_sharpe'] is not None else 'n/a'
    header=f'''<p class="eyebrow">DERIBIT / REAL TRADE RESEARCH</p><h1>周末期权，<br>从真实成交到模型权重。</h1><p class="lead">浅蓝研究账本 · {metrics['oos_start']} 至 {metrics['oos_end']} · BTC 计价</p><div class="notice">真实成交与交割指数；执行价格、Greeks 和保证金均含模型假设。不是可保证成交的收益。</div>'''
    header += f'<p class="lead">预测诊断：Ridge 样本外 R² = {metrics["ridge_oos_r2"]:.4f}，未验证预测优势。20 个因素的矩阵秩为 {metrics["standardized_design_rank"]}；OLS 系数分解不唯一。8 周块 bootstrap 的净收益 95% 区间包含零。</p>'
    page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BTC 周末成交数据回测</title><style>
:root{{--sky:#eaf5fc;--blue:#247fac;--ink:#203d50;--muted:#687f8c;--line:#d9e8f1;--white:#fff}}*{{box-sizing:border-box}}body{{margin:0;background:var(--sky);color:var(--ink);font:14px/1.65 Verdana,"Microsoft YaHei",sans-serif}}nav{{position:fixed;inset:0 auto 0 0;width:205px;padding:34px 24px;background:#f6fbff;border-right:1px solid var(--line)}}.brand{{font:22px Georgia;color:var(--blue);letter-spacing:.08em}}nav p{{color:var(--muted);font-size:11px;margin-bottom:35px}}nav button{{display:block;width:100%;text-align:left;border:0;background:transparent;color:var(--muted);font:inherit;padding:13px 10px;margin-bottom:4px;cursor:pointer}}nav button.active{{background:#dfeff9;color:var(--blue)}}nav .foot{{position:absolute;bottom:30px;font-size:11px;color:var(--muted)}}main{{margin-left:205px;padding:40px 5vw 65px;max-width:1650px}}.eyebrow{{color:var(--blue);font-size:11px;letter-spacing:.16em}}h1{{font:normal clamp(30px,3.3vw,48px)/1.2 Georgia,"Microsoft YaHei",serif;margin:18px 0}}h2{{font:normal 25px Georgia,"Microsoft YaHei",serif;margin:0 0 10px}}h3{{font-weight:normal;font-size:17px}}.lead{{color:var(--muted)}}.notice{{padding:14px 18px;background:#daedf9;border-left:3px solid #6eafd8;margin:24px 0}}.numbers{{display:grid;grid-template-columns:repeat(3,1fr);margin:32px 0 38px;gap:25px}}.numbers div{{border-bottom:1px solid #a9cee5;padding-bottom:19px}}.numbers span{{display:block;color:var(--muted);font-size:12px}}.numbers b{{font:normal 31px Georgia;display:block;margin-top:9px}}.numbers small{{font-size:16px;color:var(--muted)}}.panel{{background:var(--white);padding:24px;margin-bottom:23px;border:1px solid var(--line)}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:23px}}.grid .panel{{min-width:0}}.caption{{font-size:11px;color:var(--muted);margin:0 0 15px}}.tab{{display:none}}.tab.active{{display:block}}.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:12px}}th{{font-weight:normal;color:var(--muted);text-align:left;border-bottom:1px solid var(--line);padding:11px 9px}}td{{padding:11px 9px;border-bottom:1px solid #edf4f9;font-variant-numeric:tabular-nums}}td small{{display:block;color:var(--muted);font-size:10px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f9fd;padding:18px;font:12px/1.8 Consolas,monospace}}select{{padding:10px;border:1px solid var(--line);background:white;color:var(--ink);margin:12px 0}}.equation{{font:16px/1.9 Georgia,serif;padding:16px;background:#f2f8fc;margin:15px 0}}ul{{padding-left:20px}}li{{margin:10px 0}}a{{color:var(--blue)}}footer{{font-size:11px;color:var(--muted);padding:20px 0}}@media(max-width:850px){{nav{{position:static;width:auto;padding:15px 20px;display:flex;flex-wrap:wrap;gap:8px;border-right:0;border-bottom:1px solid var(--line)}}nav p,nav .foot{{display:none}}nav button{{width:auto;margin:0;padding:8px 12px}}nav .brand{{margin-right:auto;align-self:center}}main{{margin-left:0;padding:25px 16px}}.grid{{grid-template-columns:1fr}}.numbers{{gap:10px}}.numbers b{{font-size:23px}}.panel{{padding:13px}}}}</style></head><body>
<nav><div class="brand">BLUE LEDGER</div><p>BTC WEEKEND OPTIONS</p><button class="active" data-tab="overview">01 / 回测结果</button><button data-tab="model">02 / 最终公式</button><button data-tab="data">03 / 数据与边界</button><div class="foot">研究 / 不执行实盘<br>BTC 原生结算<br>自包含离线报告</div></nav><main>{header}
<section id="overview" class="tab active">{cards}<div class="panel">{equity_html}<p class="caption">来源：Deribit 真实 sell 成交与官方交割指数。26 个有效训练周末后扩展 walk-forward；图中仅样本外周末。基准未做风险匹配。</p></div><div class="grid"><div class="panel">{pnl_html}</div><div class="panel">{dd_html}</div></div><div class="panel"><h2>年度结果</h2><div class="scroll"><table><thead><tr><th>年度</th><th>测试周末</th><th>已交易</th><th>净收益 BTC</th><th>成本 BTC</th><th>基准净收益 BTC</th></tr></thead><tbody>{yearly}</tbody></table></div></div><div class="panel"><h2>逐周末组合与 omega</h2><p class="caption">负权重表示卖出；数量上限来自过去成交量，不是订单簿可成交容量。</p>{detail}</div><div class="grid"><div class="panel">{hist_html}</div><div class="panel"><h2>滑点敏感性</h2><p class="caption">保持基准模型及仓位不变，仅改变收取权利金的折扣。</p><table><thead><tr><th>权利金滑点</th><th>样本外净收益</th></tr></thead><tbody>{sens}</tbody></table><h3>周度风险诊断</h3><p>Sharpe {metrics['weekly_sharpe']:.3f}<br>最大周末回撤 {metrics['maximum_weekend_drawdown_btc']:.6f} BTC<br>最差周末 {metrics['worst_weekend_btc']:.6f} BTC<br>5% 尾部均值 {metrics['expected_shortfall_5pct_btc']:.6f} BTC</p></div></div><div class="panel">{stress_html}<p class="caption">零利率 Black-Scholes 估算价格除以冲击后指数价；不是实际历史平仓报价或交易所保证金模型。</p></div></section>
<section id="model" class="tab"><div class="panel"><h2>最终 Ridge 预测公式</h2><p>训练区间 {formula['train_start']} 至 {formula['train_end']}；{formula['fitted_weekends']} 个有效周末，{formula['fitted_samples']} 条合约观测。</p><div class="notice">这是最终全样本研究拟合。历史回测使用当时仅由先前已结算周末训练的系数，未使用这组最终系数回填。</div><div class="equation">alpha_hat = beta_0 + sum(beta_j × x_j) · 单位 BTC / 1 BTC 合约名义量<br>beta_0 = {formula['intercept_btc']:+.12g}<br>z_j = (x_j − mean_j) / SD_j；Ridge L2 系数 = 10</div><pre>{html.escape(expression)}</pre><details><summary>全部原始单位系数与训练标准化参数</summary><div class="scroll"><table><thead><tr><th>因子</th><th>beta</th><th>训练均值</th><th>训练标准差</th></tr></thead><tbody>{rows}</tbody></table></div></details></div><div class="panel">{beta_html}</div><div class="panel"><h2>标签、结算与权重公式</h2><div class="equation">Call payoff_BTC = max(D − K, 0) / D<br>Put payoff_BTC = max(K − D, 0) / D<br>y_BTC = 0.95 × P − payoff_BTC − min(0.0003, 0.125 × P)<br>q* = argmax[alpha × q − 10 × sum((risk_i × q_i)²)]<br>omega_i = −q_i；周末净损益 = sum(q_i × y_i)</div><p>q 为非负卖出数量，单腿最多 0.5，总数量最多 2，按 0.1 向下取整；取整后重新检查 delta、vega、保证金代理与压力损失。omega 随周末和合约变化，不是一组永久不变的常数。</p><details><summary>完整推导、因子与优化约束</summary><pre>{html.escape(final_formula)}</pre></details></div><div class="panel">{stability_html}</div><div class="panel">{comparison_html}<p class="caption">Ridge RMSE {metrics['ridge_oos_rmse_btc']:.6f} BTC；OLS RMSE {metrics['ols_oos_rmse_btc']:.6f} BTC。按合约聚合的误差，不表示独立样本数量。</p></div></section>
<section id="data" class="tab"><div class="panel"><h2>真实数据与采样口径</h2><p>2022–2026 所有已到期的周六逐一审计。每周六 ET 12:00 只决策一次，使用 11:00–12:00 ET 的 BTC 期权 sell 成交，保留每合约最后一笔非 block/combo 成交；成交年龄最多 60 分钟。只接受下一日 Sunday ET 到期的 native BTC 合约。</p><p>日度因子来自已收盘 Binance BTCUSDT K 线；结算 D 来自 Deribit 官方交割指数，而非 Binance 日收盘。moneyness 在 0.90–1.10；IV 在 5%–300%；实际成交量至少 0.1。</p><pre>{html.escape(info)}</pre><p><a href="https://docs.deribit.com/api-reference/market-data/public-get_last_trades_by_currency_and_time">官方成交字段</a> · <a href="https://support.deribit.com/hc/en-us/articles/25944746248989-Fees">费用参考</a> · <a href="https://support.deribit.com/hc/en-us/articles/31424939096093-Inverse-Options">BTC 逆向期权规格</a></p></div><div class="panel"><h2>必须保留的边界</h2><ul>{limitations}</ul><p>2026 年未来或未结算周末不参与拟合；无有效成交候选的历史周末单独标记，不插值、不伪造收益。周末交易事件账本与净损益已对账。</p></div><div class="panel">{corr_html}<p class="caption">覆盖全部已观测因子，仅作描述性分析；训练期标准化仍在每个 walk-forward 分割内部独立拟合。</p></div></section><footer>真实成交数据研究版。所有收益均为 BTC；不代表实盘成交或未来收益。</footer></main>
<script>const legs={drill_json};document.querySelectorAll('nav button').forEach(b=>b.onclick=()=>{{document.querySelectorAll('nav button,.tab').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.getElementById(b.dataset.tab).classList.add('active');window.dispatchEvent(new Event('resize'));}});const select=document.getElementById('weekend-select');function renderLegs(){{if(!select)return;document.getElementById('legs').innerHTML=legs.filter(r=>r.weekend===select.value).map(r=>`<tr title="Trade ${{r.trade_time}} | Decision ${{r.decision_time}} | Expiry ${{r.expiry}}"><td>${{r.symbol}}</td><td>${{r.omega.toFixed(1)}}</td><td>${{r.premium.toFixed(6)}}</td><td>${{r.alpha.toFixed(6)}}</td><td>${{r.settlement.toFixed(6)}}</td><td>${{r.pnl.toFixed(6)}}</td><td>${{r.age.toFixed(1)}} min</td></tr>`).join('');}}if(select){{select.onchange=renderLegs;renderLegs();}}</script></body></html>'''
    page = page.replace('alpha × q − 10 ×', f'alpha × q − {cfg.risk_penalty_per_btc:g} ×')
    (p/'report.html').write_text(page,encoding='utf-8')
    return metrics
