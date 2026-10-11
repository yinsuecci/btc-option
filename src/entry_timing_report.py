"""Entry-time sensitivity panel for the existing option-backtest dashboard."""
import html
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

LABELS={'fri_16utc':'周五 16:00 UTC / 北京周六 00:00',
        'sat_00utc':'周六 00:00 UTC / 北京周六 08:00',
        'sat_08utc':'周六 08:00 UTC / 北京周六 16:00',
        'sat_16utc':'周六 16:00 UTC / 北京周日 00:00',
        'sun_00utc':'周日 00:00 UTC / 北京周日 08:00',
        'sun_04utc':'周日 04:00 UTC / 北京周日 12:00',
        'current_sat_12ny':'当前：纽约周六 12:00 / 北京周日 00:00 或 01:00'}


def panel(root):
    root=Path(root)
    data=json.loads((root/'summary.json').read_text())
    rows=[]
    for m in sorted(data['summaries'],key=lambda x:x['net_pnl_btc'],reverse=True):
        d,h=m['discovery'],m['holdout']
        hours='15--16' if m['offset_utc_hours'] is None else str(32-m['offset_utc_hours'])
        rows.append(f'''<tr><td>{html.escape(LABELS[m['slot']])}</td><td>{hours}</td><td>{m['net_pnl_btc']:+.6f}</td>
<td>{m['cagr']:.2%}</td><td>{m['weekly_return_sharpe']:.3f}</td><td>{abs(m['max_drawdown_pct']):.2f}%</td>
<td>{m['traded_weekends']}/198</td><td>{m['average_quantity']:.3f}</td><td>{m['eligible_weekends']}/250</td>
<td>{d['net_pnl_btc']:+.6f}</td><td>{h['net_pnl_btc']:+.6f}</td><td>{h['sharpe']:.3f}</td></tr>''')
    fig=go.Figure()
    for m in data['summaries']:
        c=pd.read_csv(root/m['slot']/'performance_curve.csv')
        fig.add_trace(go.Scatter(x=c.weekend_id,y=c.equity_btc,name=LABELS[m['slot']].split(' / ')[0],mode='lines'))
    fig.update_layout(title='七组入场时间：周末结算 BTC 权益',xaxis_title='周末日期',yaxis_title='权益 / BTC',
                      template='plotly_white',height=540,legend=dict(orientation='h',y=-.25),
                      margin=dict(l=55,r=20,t=60,b=150))
    chart=fig.to_html(full_html=False,include_plotlyjs=False,div_id='entry-timing-equity',config={'responsive':True})
    winner=LABELS[data['full_history_winner']]
    selected=LABELS[data['discovery_selected_slot']]
    sample=data['summaries'][0]
    d,h=sample['discovery'],sample['holdout']
    return f'''<article id="entry-timing"><h2>06 / 周末入场时间比较</h2>
<p><b>历史收益最高：</b>{html.escape(winner)}。<b>前 104 周收益选出的时间：</b>{html.escape(selected)}。</p>
<p>固定七个候选时间，以每次决策前一小时的真实成交为输入。各时点独立进行成熟标签扩展窗口拟合，
尾部惩罚 0.01、协方差惩罚 1、5% 成交折让和全部硬约束一致。每周最多入场一次，统一周日交割。</p>
<p class="caption">相同 198 个评价周末；缺少候选腿或成熟训练不足时为空仓并计入收益分母。
完整窗口下载与有可交易合约是两项不同覆盖口径。当前策略保留不变。</p>
<div class="scroll"><table><thead><tr><th>入场时间</th><th>持有小时</th><th>净收益 BTC</th><th>复合年化</th><th>Sharpe</th>
<th>结算回撤</th><th>交易周末</th><th>平均卖出数量</th><th>有候选腿周末</th>
<th>前104周收益 BTC</th><th>后94周收益 BTC</th><th>后94周 Sharpe</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
{chart}<p class="caption">来源：Deribit 实际成交与官方交割、Binance 已收盘日线。2022-12-31 至 2026-10-10。
权益初始 1 BTC，采用模型成交与周末结算口径。</p>
<p>时间选择区间：{d['start']} 至 {d['end']}；时间切分检验区间：{h['start']} 至 {h['end']}。
后段收益从其起点重新以 1 BTC 记账用于展示，持仓仍沿用原策略连续权益路径。
全区间最高收益是七组候选中的事后结果；时间切分使用已有历史，未提供新增未见样本。</p>
<p class="caption">UTC 入场网格按固定时钟，当前纽约中午随夏令时变化。检验范围为七个时点，尚未穷举每小时。
历史成交代理不含真实盘口成交及盘中强平。明细保存在 results/entry_timing，每组包含预测、数量及权益曲线。</p></article>'''


def render(root):
    root=Path(root)
    content=panel(root)
    body=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>option-backtest / 入场时间</title><style>body{{background:#eaf5fc;color:#203d50;margin:0;font:15px Georgia,'Microsoft YaHei',serif}}
main{{max-width:1500px;margin:auto;padding:28px}}article{{background:white;padding:24px}}h1{{font-size:28px}}h2{{font-size:23px}}
p{{line-height:1.8}}.caption{{font-size:12px;color:#687f8c}}.scroll{{overflow:auto}}table{{border-collapse:collapse;min-width:1300px;width:100%;font-size:12px}}
td,th{{padding:12px;text-align:left;border-bottom:1px solid #d4e6f2}}@media(max-width:700px){{main{{padding:12px}}article{{padding:12px}}}}</style>
<script>{get_plotlyjs()}</script></head><body><main><h1>option-backtest / 周末入场时间研究</h1>{content}</main></body></html>'''
    (root/'report.html').write_text(body,encoding='utf-8')
