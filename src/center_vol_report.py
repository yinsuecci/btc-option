"""Self-contained light blue dashboard for the two-stage distribution model."""
import base64
import hashlib
import html
import json

import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs


def expression(name, params):
    return name+' = '+f'{params["intercept"]:.12g}'+''.join(
        f'\n{v:+.12g} * {f}' for f, v in params['coefficients'].items())


def render(tables, formula, metrics, p):
    f, w = tables['forecasts'], tables['weekends']
    w = w.loc[w.evaluation]
    source = f'Deribit actual trades / official delivery; Binance closed daily factors. {metrics["oos_start"]} to {metrics["oos_end"]}.'
    entry_note='周五 16:00 UTC（北京时间周六 00:00）入场，周日 08:00 UTC（北京时间周日 16:00）交割，持有 40 小时'
    expressions = expression('mu_log_return', formula['center_beta'])+'\n\n'+expression('g_log_squared_error', formula['vol_gamma'])
    formulas = f'''# Price center, volatility and leg allocation

Final retrospective fit: {formula['train_start']} through {formula['train_end']}.
Price training rows: {formula['price_rows']}. Variance rows: {formula['variance_rows']}.
Each row is ONE weekend, not one option contract. Final parameters are NOT backfilled into history.

## Raw-unit fitted parameters
{expressions}

## Distribution
r = ln(D / S); mu = E[r | X].
Center beta is Ridge on r; standardization fitted on prior training data only; L2=10.
e_u = r_u - mu_hat_u, where mu_hat_u was fitted before u with at least 13 matured weekends.
Gamma is Ridge on ln(max(e_u^2, 1e-10)), L2=10, using only prior matured prequential errors.
c = mean(max(e_u^2, 1e-10) / exp(g(Z_u))) over the variance training data.
Final c = {formula['variance_calibration']:.12g}.
v = clip(c * exp(clip(g(Z), -30, 5)), 0.001^2, 0.20^2).
sigma_weekend = sqrt(v); this is NOT annualized IV or intraday realized volatility.
Conditional assumption: ln(D/S) ~ Normal(mu, v); this is a forecast-error distribution proxy.
Price median = S * exp(mu).
Price expectation E[D] = S * exp(mu + v/2).
95% predictive price interval = S * exp(mu +/- 1.95996398454 * sqrt(v)).
Neither the expectation nor this interval guarantees a reachable price.

## Factor units
momentum_nd = last closed daily close / close n days earlier - 1, n=1,3,7.
rv_nd = sample SD(last n daily log returns) * sqrt(365), n=7,30.
spot_volume_z30 = (last closed volume - 30d mean) / 30d sample SD.
jump_z = abs(last daily log return) / 30d sample SD of daily log returns.
median_trade_iv = median printed IV / 100 across eligible sell-print contracts at decision.
This median is an INPUT summary, not the predicted sigma and not the full-chain ATM IV.
days_to_expiry = actual elapsed hours / 24.

## Expected native-BTC settlement loss
a_i = ln(K_i / S); sigma = sqrt(v); A_i = (K_i/S)*exp(-mu+v/2).
H_call_i = Phi((mu-a_i)/sigma) - A_i*Phi((mu-v-a_i)/sigma).
H_put_i = A_i*Phi((a_i-mu+v)/sigma) - Phi((a_i-mu)/sigma).
alpha_i = 0.95*P_i - min(0.0003, 0.125*P_i) - H_i.
Phi is the standard normal CDF. H_i is a physical-distribution expected settlement,
not a risk-neutral Black-Scholes price. Native payoff divides intrinsic USD by D, not S.

## Allocation
Sigma_ij = Cov(H_i(D), H_j(D) | mu,v), approximated with 128-node Gauss-Hermite quadrature.
q* = argmax(alpha' q - {formula['config']['covariance_penalty_per_btc']:g}*q' Sigma q - {formula['execution']['risk_penalty_per_btc']:g}*sum((R_i*q_i)^2)).
omega_i = -q_i; relative_abs_weight_i = q_i / sum(q) when sum(q)>0, otherwise 0.
R_i = max(0.001, max_stress(V_i_BTC - net_entry_premium_i)).
Constraints: 0<=q_i<=min(0.5, past sell-print quantity); sum(q)<=2.
abs(sum(q_i*delta_log_i))<=0.30 BTC; abs(sum(q_i*vega_i))<=0.20 BTC / 1.0 IV.
margin_i = max(0.15-OTM_i/S, 0.10)+P_i.
sum(q_i*margin_i)<=min(0.50 BTC, 0.5*equity_BTC).
max_stress sum(q_i*(V_i_BTC-net_entry_premium_i))<=min(0.25 BTC, 0.25*equity_BTC).
Spot stress: -30%,-15%,-5%,0%,+5%,+15%,+30%; IV additions: 0,0.25,0.50.
V_BTC = zero-rate BS_USD / shocked S; stress/Greeks/margin are MODEL proxies.
Round q DOWN to 0.1; recheck constraints; enter only if alpha' q>0.
All allocation risk constants are fixed research assumptions, not fitted factor coefficients.

## Realized result and limits
Realized weekend PnL_BTC = sum(q_i*(0.95*P_i - actual_intrinsic_i/D - entry_fee_i)).
Decision Saturday 12:00 ET; official next Sunday 08:00 UTC settlement.
52 prior matured weekends before evaluation; variance requires 26 prior matured residuals.
No live trades. Only 60-minute historical sell prints, not executable archived bid/ask.
No complete chain; no hedging, intraday liquidation, funding or actual account fees.
Gaussian tails, residual calibration and quadrature are assumptions; report coverage and NLL.
Daily market factors and printed contract IV are causal; settlement is label only.
Mean/variance parameters are re-estimated at every historical decision with expanding past data.
'''
    (p/'FORMULAS.md').write_text(formulas, encoding='utf-8')
    charts = []

    def chart(fig, name, title, x_label, y_label, height=380):
        multiple = len(fig.data) > 1
        fig.update_layout(template='plotly_white', height=max(height,440) if multiple else height,
                          title=dict(text=title,font=dict(size=14)),
                          font=dict(family='Verdana', color='#203d50', size=12),
                          margin=dict(l=68,r=24,t=80,b=140 if multiple else 60), paper_bgcolor='white',
                          legend=dict(orientation='h',y=-.30,x=0,yanchor='top',font=dict(size=10)), hovermode='x unified',
                          colorway=['#2784be','#8bacbf','#54a4d4'])
        fig.update_xaxes(title=x_label, gridcolor='#edf4f9', automargin=True)
        fig.update_yaxes(title=y_label, gridcolor='#edf4f9', automargin=True)
        markup = fig.to_html(full_html=False, include_plotlyjs=False,
                             config={'responsive':True,'displaylogo':False}, div_id=name)
        charts.append(name)
        return '<article>'+markup+'<p class="caption">'+html.escape(source)+'</p></article>'

    fig=go.Figure()
    fig.add_scatter(x=f.weekend_id,y=(f.lower_95_usd/f.spot-1)*100,name='95% lower',line=dict(width=0))
    fig.add_scatter(x=f.weekend_id,y=(f.upper_95_usd/f.spot-1)*100,name='95% upper',fill='tonexty',fillcolor='rgba(125,190,226,.18)',line=dict(width=0))
    fig.add_scatter(x=f.weekend_id,y=(f.center_mean_usd/f.spot-1)*100,name='Predicted expected price move')
    fig.add_scatter(x=f.weekend_id,y=(f.delivery_index/f.spot-1)*100,name='Actual delivery move',mode='markers',marker=dict(size=4))
    center_chart=chart(fig,'center','价格中心与 95% 预测区间<br>相对周六指数','Saturday date (ET)','Delivery price change (%)',450)
    fig=go.Figure()
    fig.add_scatter(x=f.weekend_id,y=f.sigma_weekend*100,name='Predicted weekend sigma')
    fig.add_scatter(x=f.weekend_id,y=np.abs(f.log_return-f.mu_log_return)*100,name='Actual absolute forecast error',mode='markers',marker=dict(size=4))
    vol_chart=chart(fig,'vol','波动模型：持有期标准差与实际误差','Saturday date (ET)','Log-return / forecast error (%)')
    fig=go.Figure(go.Scatter(x=w.weekend_id,y=w.equity_btc,name='Center + volatility allocation'))
    equity_chart=chart(fig,'equity','两阶段模型的样本外权益','Saturday date (ET)','End-weekend equity (BTC)')
    curve=tables['performance_curve']
    fig=go.Figure(go.Scatter(x=curve.weekend_id,y=curve.drawdown_pct,name='Settled equity drawdown',
                             fill='tozeroy',line=dict(color='#62869e')))
    drawdown_chart=chart(fig,'drawdown','样本外最大回撤与回撤路径','Saturday date (ET)','Drawdown from prior peak (%)')
    fig=go.Figure(go.Bar(x=w.weekend_id,y=w.net_pnl_btc,name='Net PnL',marker_color=np.where(w.net_pnl_btc>=0,'#54a4d4','#8194a3')))
    pnl_chart=chart(fig,'pnl','逐周末扣费损益','Saturday date (ET)','Net PnL (BTC)')
    coefficient_charts=[]
    for name,title in [('center_beta','价格中心 beta'),('vol_gamma','波动模型 gamma')]:
        params=formula[name]
        values={f:b*params['scaler_scale'][f] for f,b in params['coefficients'].items()}
        fig=go.Figure(go.Bar(x=list(values.values()),y=list(values),orientation='h',marker_color='#6eafd8'))
        coefficient_charts.append(chart(fig,name,title+'：因子变化一个训练标准差的影响','Coefficient x feature SD','Factor',420))
    f_records=json.loads(f.to_json(orient='records',date_format='iso'))
    positions=tables['positions']
    records=json.loads(positions.to_json(orient='records',date_format='iso'))
    all_weekends=tables['weekends'].sort_values('weekend_id')
    options=''.join(f'<option>{x}</option>' for x in reversed(all_weekends.weekend_id.tolist()))
    statuses={'WARMUP':'训练预热，无持仓','ENTER':'已建仓',
              'NO_POSITIVE_ROUNDED_PORTFOLIO':'未交易：取整后无有效组合',
              'ROUNDING_INFEASIBLE':'未交易：取整后不满足约束',
              'OPTIMIZER_FAILED':'未交易：优化失败','NO_EQUITY':'未交易：无可用权益'}
    summaries=[]
    for row in all_weekends.itertuples():
        group=positions.loc[positions.weekend_id.eq(row.weekend_id)]
        held=group.loc[group.quantity>0]
        summaries.append(dict(weekend_id=row.weekend_id,status=row.status,status_label=statuses.get(row.status,row.status),
                              traded=bool(row.traded),evaluation=bool(row.evaluation),candidates=len(group),
                              held_legs=len(held),quantity=float(group.quantity.sum()),net_pnl_btc=float(row.net_pnl_btc),
                              holdings=' ; '.join(f'{r.symbol}: q={r.quantity:.1f}, omega={r.omega:.1f}' for r in held.itertuples()),
                              candidate_symbols=group.symbol.tolist()))
    csv_uri='data:text/csv;charset=utf-8;base64,'+base64.b64encode(('\ufeff'+positions.to_csv(index=False)).encode('utf-8')).decode('ascii')
    sharpe_text=f"{metrics['weekly_return_sharpe']:.3f}" if metrics['weekly_return_sharpe'] is not None else '不适用（收益率方差为零）'
    risk_note=f'''Sharpe = mean(r_w) / sample_SD(r_w) × sqrt(52)，r_w = 周末净损益 / 周末前权益。
使用全部 {metrics['sharpe_observations']} 个样本外周末，包含未交易周末；BTC 无风险收益率假设为零。不是按单笔收益或仅交易周末计算。
最大百分比回撤峰值周末 {metrics['drawdown_peak_weekend']}，谷值 {metrics['drawdown_trough_weekend']}；
恢复日期 {metrics['drawdown_recovery_weekend'] or '截至样本末尚未恢复'}。最大金额回撤 {metrics['maximum_drawdown_btc']:.6f} BTC。
回撤按周末结算权益计算，包含初始权益峰值，不是盘中逐笔盯市回撤。'''
    audit_html=''
    audit_path=p/'lookahead_audit.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text())
        if (audit['status']=='PASS_NO_LOOKAHEAD_DETECTED_IN_TESTED_PATHS'
            and audit['forecast_sha256']==hashlib.sha256((p/'forecasts.csv').read_bytes()).hexdigest()
            and audit['predictions_sha256']==hashlib.sha256((p/'predictions.csv').read_bytes()).hexdigest()):
            audit_html=f'''<article id="lookahead"><h2>未来数据检查</h2><p class="notice">已覆盖的路径中未发现未来函数：{audit['oos_decisions_recomputed']} 个样本外决策逐一截断重算，预测和持仓与记录一致；当周/未来交割标签和未来因子篡改不改变当期预测。删除交割标签后，持仓优化结果仍一致。</p><p>{audit['raw_windows_audited']} 个原始成交窗口和日线因子时间检查通过；{audit['prefix_residual_rebuilds']} 个完整历史前缀重建的残差一致。共 {audit['check_count']} 项核验。标准化及波动回转校正仅使用历史训练数据，最终全样本系数不参与历史决策。</p><p class="caption">这不是对数据供应商历史发布延迟或实际可成交价格的证明。成交代理、盘中清算缺失和模型假设仍然存在。</p></article>'''
    holdings_html=f'''<article><h2>全部 {len(all_weekends)} 个周末的持仓总览</h2><p>包括训练预热、未交易以及实际建仓周末。点击日期查看当周每条候选腿；数量 0 表示未持仓，不是数据缺失。</p><div class="filters"><label for="holding-search">日期或合约</label> <input id="holding-search" type="search" placeholder="例如 2026-09 或 82500-C"> <label><input id="held-only" type="checkbox">只看已建仓</label> <button id="download-positions" type="button">下载全部逐腿持仓 CSV</button></div><p id="holding-count" class="caption"></p><div class="scroll all-holdings"><table><thead><tr><th>周末</th><th>状态</th><th>持仓腿 / 候选腿</th><th>总卖出数量</th><th>各腿持仓 q 与 omega</th><th>实际净收益 BTC</th></tr></thead><tbody id="all-holdings"></tbody></table></div></article>'''
    comparison_html=''
    comparison_path=p/'risk_comparison.json'
    if comparison_path.exists():
        comparison=json.loads(comparison_path.read_text())
        if comparison['current_forecast_sha256']==hashlib.sha256((p/'forecasts.csv').read_bytes()).hexdigest():
            base=comparison['baseline']; current=comparison['current']
            comparison_html=f'''<article><h2>风险参数调整：{base['risk_penalty']:g} → {current['risk_penalty']:g}</h2><p>只改变尾部压力风险惩罚。协方差惩罚、单腿上限、历史成交量上限、保证金和压力损失约束均不变。价格中心 beta 与波动 gamma 不变。</p><div class="scroll"><table><thead><tr><th>版本</th><th>净收益 BTC</th><th>复合年化收益</th><th>年化周度 Sharpe</th><th>最大结算回撤</th><th>已交易周末</th><th>平均保证金代理 BTC</th></tr></thead><tbody><tr><td>原版 {base['risk_penalty']:g}</td><td>{base['net_pnl_btc']:+.6f}</td><td>{base['cagr']:.2%}</td><td>{base['sharpe']:.3f}</td><td>{base['max_drawdown_pct']:.3f}%</td><td>{base['traded_weekends']}</td><td>{base['average_margin_btc']:.6f}</td></tr><tr><td>当前 {current['risk_penalty']:g}</td><td>{current['net_pnl_btc']:+.6f}</td><td>{current['cagr']:.2%}</td><td>{current['sharpe']:.3f}</td><td>{current['max_drawdown_pct']:.3f}%</td><td>{current['traded_weekends']}</td><td>{current['average_margin_btc']:.6f}</td></tr></tbody></table></div><p class="caption">参数调整使用同一段已查看历史数据。这是回测敏感性比较，不是新的未见样本验证，也不是实际回撤承诺。预测和持仓时间顺序仍单独审计。</p></article>'''
    diagnosis=f'''价格中心 RMSE：{metrics['log_return_rmse']:.5f}；过去均值基准：{metrics['past_mean_rmse']:.5f}；相对基准 R²：{metrics['r2_vs_past_mean']:.4f}。
95% 区间实际覆盖率：{metrics['interval_95_coverage']:.1%}。Gaussian NLL：{metrics['gaussian_nll']:.4f}；历史均值/方差基准 NLL：{metrics['naive_gaussian_nll']:.4f}（越低越好）。'''
    body=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BTC 价格中心与波动分配</title><style>
:root{{--sky:#eaf5fc;--white:#fff;--ink:#203d50;--muted:#687f8c;--line:#d4e6f2;--accent:#2784be}}*{{box-sizing:border-box}}body{{margin:0;background:var(--sky);color:var(--ink);font:15px Georgia,'Microsoft YaHei',serif}}nav{{width:210px;position:fixed;inset:0 auto 0 0;background:#f7fbff;padding:30px 22px;border-right:1px solid var(--line)}}nav strong{{font-size:23px;letter-spacing:2px}}nav a{{display:block;margin-top:22px;color:var(--ink);text-decoration:none;padding:12px 0}}main{{margin-left:210px;padding:42px 5%;max-width:1600px}}h1{{font-size:42px;font-weight:400;line-height:1.25;margin:24px 0}}h2{{font-weight:400;font-size:25px}}h3{{font-size:18px}}.eyebrow{{font:11px Verdana,sans-serif;letter-spacing:3px;color:var(--muted)}}.notice{{padding:18px;background:#deedf7;border-left:3px solid #8bacbf;line-height:1.8}}.caption{{font:12px Verdana,'Microsoft YaHei',sans-serif;line-height:1.8;color:var(--muted)}}.numbers{{display:flex;gap:24px;margin:32px 0}}.numbers div{{flex:1;border-bottom:1px solid var(--line);padding-bottom:20px}}.numbers b{{display:block;font-size:28px;font-weight:400;margin-top:12px}}article{{background:var(--white);padding:18px;margin:24px 0;border:1px solid var(--line);min-width:0}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}.grid article{{margin:0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:13px Consolas,monospace;line-height:1.8;background:#f7fbff;padding:18px}}select{{padding:12px;border:1px solid var(--line);background:white;color:var(--ink)}}.scroll{{overflow:auto}}table{{border-collapse:collapse;min-width:800px;width:100%;font:12px Verdana,'Microsoft YaHei',sans-serif}}td,th{{padding:12px;text-align:left;border-bottom:1px solid var(--line)}}#distribution{{line-height:1.9}}section{{scroll-margin-top:20px}}@media(max-width:800px){{nav{{position:static;width:auto;padding:18px}}nav a{{display:inline-block;margin:0 20px 0 0}}main{{margin:0;padding:24px 16px}}h1{{font-size:30px}}.grid{{grid-template-columns:1fr}}.numbers{{gap:12px}}.numbers b{{font-size:21px}}article{{padding:12px}}}}
.numbers{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr))}}.filters{{display:flex;gap:12px;align-items:center;flex-wrap:wrap}}input[type=search]{{padding:10px;border:1px solid var(--line);background:white;color:var(--ink)}}a{{color:var(--accent)}}.all-holdings{{max-height:560px}}.all-holdings th{{position:sticky;top:0;background:#f7fbff}}.date-button{{font:inherit;border:0;background:none;color:var(--accent);cursor:pointer;padding:0}}@media(max-width:800px){{.numbers{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
</style><script>{get_plotlyjs()}</script></head><body><nav><strong style="font-size:18px;letter-spacing:0.5px;overflow-wrap:anywhere">option-backtest</strong><p class="caption">PRICE / VOL / LEGS</p><a href="#overview">01 / 模型与回测</a><a href="#allocation">02 / 逐腿持仓</a><a href="#formulas">03 / 带参公式</a><a href="#limits">04 / 数据与边界</a></nav><main>
<p class="eyebrow">DERIBIT REAL TRADE RESEARCH</p><h1>先估计价格中心，<br>再预测波动，配置各腿。</h1><p class="caption">{metrics['oos_start']} 至 {metrics['oos_end']}。BTC 计价。仅研究，不执行实盘。</p>
<div class="notice">周末市场因子 → 价格中心 beta → 样本外残差 → 波动 gamma → 交割价格分布 → 各腿预期结算损失与协方差 → omega。真实历史成交作为执行代理，不是保证可成交报价。</div>
<section id="overview"><div class="numbers"><div>样本外净收益<b>{metrics['net_pnl_btc']:+.6f} BTC</b></div><div>年化周度 Sharpe<b>{sharpe_text}</b></div><div>最大结算回撤<b>{metrics['max_drawdown_pct']:.3f}%</b></div><div>已交易 / 测试周末<b>{metrics['traded_weekends']} / {metrics['oos_weekends']}</b></div></div>{audit_html}<p class="notice">{diagnosis}</p><div class="grid">{equity_chart}{drawdown_chart}</div><article><h2>风险指标口径</h2><p class="caption">{risk_note}</p></article>{center_chart}{vol_chart}{pnl_chart}<p class="caption">周末净收益平均值 {metrics['mean_weekend_pnl_btc']:+.8f} BTC；中位数 {metrics['median_weekend_pnl_btc']:+.8f} BTC。均包含未交易的样本外周末。</p></section>
<section id="allocation"><article><h2>价格中心、波动与各腿的数量</h2><label for="weekend">周末（含训练预热）</label> <select id="weekend">{options}</select><div id="distribution"></div><div class="scroll"><table><thead><tr><th>合约</th><th>预期结算 BTC</th><th>预期净收益 / 单位</th><th>卖出数量 q</th><th>omega = -q</th><th>相对数量占比</th><th>实际净收益 BTC</th></tr></thead><tbody id="legs"></tbody></table></div><p class="caption">展示全部候选腿，包括数量为零的腿。相对占比是数量占比，不是保证金占比或资金权重。模型分布在所有腿之间共用，执行价决定每腿的结算风险。</p></article></section>
{holdings_html}
<section id="formulas"><article><h2>两套独立的带参公式</h2><p>最终全样本参数：{formula['price_rows']} 个周末价格标签，{formula['variance_rows']} 个历史滚动残差。历史回测每周重新拟合，未使用最终参数回填。</p><pre>{html.escape(expressions)}</pre><p>价格中值 = S × exp(mu)；价格期望 = S × exp(mu + v / 2)。波动 v = clip({formula['variance_calibration']:.12g} × exp(g), 0.001², 0.20²)，g 另有 -30 至 5 的数值保护。</p><details><summary>展开完整公式、参数、因子单位与持仓约束</summary><pre>{html.escape(formulas)}</pre></details><details><summary>训练标准化参数 JSON</summary><pre>{html.escape(json.dumps(formula,indent=2))}</pre></details></article><div class="grid">{''.join(coefficient_charts)}</div></section>
<section id="limits"><article><h2>数据与模型边界</h2><p>2022-01-01 至 2026-10-10 共 {metrics['source_weekends']} 个真实周末、{metrics['observations']} 条合约观测。周六 12:00 ET 决策，使用此前 60 分钟 sell 成交与已收盘日线；周日 08:00 UTC 官方指数交割。尚未到期的周末不参与。</p><p>先有 13 周末训练中心模型并生成因果残差；正式回测在 52 个已结算周末后开始，波动模型至少需要 26 个已结算残差。模型预测的是持有期交割收益分布，不是盘中最高价、最低价或一定可达价格。</p><p>预测误差的条件二阶矩被用作条件方差代理，包含均值模型偏差和估计误差。正态尾部、对数残差回转校正、波动上下限、协方差数值积分均为研究假设。波动因子 gamma、价格因子 beta 和最终数量 omega 是不同对象。</p><p>只有成交候选，没有完整链或历史 bid/ask；5% 权利金折扣、费用、Greeks、保证金和压力情景均为模型假设。没有盘中清算、动态对冲、资金费或账户实际费率。不将正收益当作预测优势。</p><pre>{html.escape(json.dumps(metrics,indent=2))}</pre></article></section></main>
<script>
const forecasts={json.dumps(f_records)},legs={json.dumps(records)},weekends={json.dumps(summaries)};
const select=document.getElementById('weekend');
function number(x,n=7){{return Number.isFinite(x)?x.toFixed(n):'未预测';}}
function renderLegs(){{
  const f=forecasts.find(x=>x.weekend_id===select.value);
  const summary=weekends.find(x=>x.weekend_id===select.value);
  const selected=legs.filter(x=>x.weekend_id===select.value);
  document.getElementById('distribution').textContent=f?`周六指数 $${{number(f.spot,2)}} · 价格期望 $${{number(f.center_mean_usd,2)}} · 价格中值 $${{number(f.center_median_usd,2)}} · 持有期波动 ${{number(100*f.sigma_weekend,3)}}% · 95%区间 $${{number(f.lower_95_usd,2)}} 至 $${{number(f.upper_95_usd,2)}} · 官方交割 $${{number(f.delivery_index,2)}} · ${{summary.status_label}}`:`${{summary.weekend_id}} · ${{summary.status_label}} · 该周末尚未进入样本外评估，不展示未训练的预测值。`;
  document.getElementById('legs').innerHTML=selected.map(x=>`<tr><td>${{x.symbol}}</td><td>${{number(x.expected_payoff_btc)}}</td><td>${{number(x.predicted_net_per_unit_btc)}}</td><td>${{number(x.quantity,1)}}</td><td>${{number(x.omega,1)}}</td><td>${{number(100*x.relative_abs_weight,1)}}%</td><td>${{number(x.net_pnl_btc)}}</td></tr>`).join('');
}}
function renderHoldings(){{
  const query=document.getElementById('holding-search').value.trim().toLowerCase();
  const heldOnly=document.getElementById('held-only').checked;
  const matching=weekends.filter(x=>(!heldOnly||x.traded)&&(!query||(x.weekend_id+' '+x.candidate_symbols.join(' ')).toLowerCase().includes(query))).slice().reverse();
  document.getElementById('holding-count').textContent=`显示 ${{matching.length}} / ${{weekends.length}} 个周末，训练预热周末也单独标识。`;
  document.getElementById('all-holdings').innerHTML=matching.map(x=>`<tr><td><button class="date-button" data-weekend="${{x.weekend_id}}">${{x.weekend_id}}</button></td><td>${{x.status_label}}</td><td>${{x.held_legs}} / ${{x.candidates}}</td><td>${{number(x.quantity,1)}}</td><td>${{x.holdings||'无持仓'}}</td><td>${{number(x.net_pnl_btc)}}</td></tr>`).join('');
  document.querySelectorAll('.date-button').forEach(b=>b.onclick=()=>{{select.value=b.dataset.weekend;renderLegs();document.getElementById('allocation').scrollIntoView({{behavior:'auto'}});}});
}}
select.onchange=renderLegs;
document.getElementById('holding-search').oninput=renderHoldings;
document.getElementById('held-only').onchange=renderHoldings;
document.getElementById('download-positions').onclick=()=>{{const a=document.createElement('a');a.href='{csv_uri}';a.download='positions_by_weekend.csv';a.click();}};
renderLegs();renderHoldings();
</script></body></html>'''
    body=body.replace('<section id="overview">','<section id="overview">'+comparison_html,1)
    body=body.replace('周六 12:00 ET 决策，使用此前 60 分钟 sell 成交与已收盘日线；周日 08:00 UTC 官方指数交割。',entry_note+'。使用决策前 60 分钟卖方成交与已收盘日线。',1)
    body=body.replace('<p class="eyebrow">DERIBIT REAL TRADE RESEARCH</p>',f'<p class="eyebrow">DERIBIT REAL TRADE RESEARCH</p><p class="notice">固定入场时间：{entry_note}。</p>',1)
    grid_path=p.parent/'risk_grid'/'grid.json'
    grid_validation=p.parent/'risk_grid'/'validation.json'
    if grid_path.exists() and grid_validation.exists():
        grid=json.loads(grid_path.read_text())
        evidence=json.loads(grid_validation.read_text())
        if (evidence['status']=='PASS' and grid['source_forecast_sha256']==hashlib.sha256((p/'forecasts.csv').read_bytes()).hexdigest()
            and evidence['grid_sha256']==hashlib.sha256(grid_path.read_bytes()).hexdigest()):
            from src.risk_grid_report import panel
            body=body.replace('<section id="overview">','<section id="overview">'+panel(grid),1)
            body=body.replace('</nav>','<a href="#risk-grid">05 / 策略分组</a></nav>',1)
    body=body.replace('<div class="notice">周末市场因子',f'<p class="caption">当前尾部风险惩罚 = {formula["execution"]["risk_penalty_per_btc"]:g}；协方差惩罚 = {formula["config"]["covariance_penalty_per_btc"]:g}。其余硬约束不变。</p><div class="notice">周末市场因子',1)
    (p/'report.html').write_text(body,encoding='utf-8')
