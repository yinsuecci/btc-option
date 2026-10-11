"""Embedded nine-case risk comparison for the existing light-blue research report."""
import json
import numpy as np
import plotly.graph_objects as go

LABELS={'AGGRESSIVE':'激进','BALANCED':'平衡（当前）','CONSERVATIVE':'保守','GRID':'参数组合'}


def panel(data):
    summary=data['summaries']; bundles=data['bundles']; presets=data['preset_definitions']
    charts=[]
    colors={'AGGRESSIVE':'#2784be','BALANCED':'#62869e','CONSERVATIVE':'#9dbed2'}
    for metric,title,ylabel in [('equity_btc','三类策略样本外权益对比','End-weekend equity (BTC)'),
                                ('drawdown_pct','三类策略结算回撤对比','Drawdown from prior peak (%)')]:
        fig=go.Figure()
        for label,name in presets.items():
            curve=bundles[name]['performance_curve']
            fig.add_scatter(x=[r['weekend_id'] for r in curve],y=[r[metric] for r in curve],name=LABELS[label],
                            line=dict(color=colors[label],width=2.2))
        fig.update_layout(template='plotly_white',height=410,title=dict(text=title,font=dict(size=14)),
                          font=dict(family='Verdana',color='#203d50',size=12),
                          margin=dict(l=65,r=20,t=60,b=110),legend=dict(orientation='h',y=-.30),hovermode='x unified')
        fig.update_xaxes(title='Saturday decision date (ET)',gridcolor='#edf4f9',automargin=True)
        fig.update_yaxes(title=ylabel,gridcolor='#edf4f9',automargin=True)
        charts.append(fig.to_html(full_html=False,include_plotlyjs=False,config={'responsive':True,'displaylogo':False},div_id='grid_'+metric)
                      +f'<p class="caption">Source: Deribit trade/official delivery research. {data["period"]["start"]} to {data["period"]["end"]}; 198 OOS weekends, fixed preset groups, BTC equity. End-weekend settlement only.</p>')
    rows=''.join(f'''<tr><td>{LABELS[r['preset']]}</td><td>{r['tail_penalty']:g}</td><td>{r['covariance_penalty']:g}</td><td>{r['net_pnl_btc']:+.6f}</td><td>{r['cagr']:.2%}</td><td>{r['weekly_return_sharpe']:.3f}</td><td>{r['max_drawdown_pct']:.3f}%</td><td>{r['traded_weekends']}</td><td>{r['average_margin_btc']:.6f}</td><td>{r['worst_weekend_btc']:.6f}</td></tr>''' for r in summary)
    ordered=sorted(summary,key=lambda r:(0 if r['preset']=='BALANCED' else 1 if r['preset']!='GRID' else 2,r['tail_penalty'],r['covariance_penalty']))
    options=''.join(f'''<option value="{r['scenario']}">{LABELS[r['preset']]}：尾部 {r['tail_penalty']:g} / 协方差 {r['covariance_penalty']:g}</option>''' for r in ordered)
    balance=presets['BALANCED']
    dates=''.join(f'<option>{r["weekend_id"]}</option>' for r in reversed(bundles[balance]['weekends']))
    compact={name:dict(positions=[{key:r[key] for key in ['weekend_id','symbol','quantity','omega','relative_abs_weight','expected_payoff_btc','predicted_net_per_unit_btc','net_pnl_btc']} for r in bundle['positions']],weekends=bundle['weekends']) for name,bundle in bundles.items()}
    return f'''<article id="risk-grid"><h2>激进、平衡、保守：九组风险参数</h2><p>固定同一组因果价格中心和波动预测，只调整尾部惩罚和协方差惩罚。预先指定激进 (0.005, 0.5)、平衡 (0.01, 1)、保守 (0.02, 2)，另外六组用于分辨两个参数的影响。当前主回测仍保持平衡策略。</p><p class="notice">分组表示模型的风险惩罚强弱，不保证历史或未来回撤按激进、平衡、保守排序。全部使用已查看的同一历史区间，不是九次独立的新样本验证，也不自动选择收益最高的组合。</p><div class="scroll"><table><thead><tr><th>分组</th><th>尾部惩罚</th><th>协方差惩罚</th><th>净收益 BTC</th><th>年化收益</th><th>Sharpe</th><th>最大结算回撤</th><th>交易周末</th><th>平均保证金代理 BTC</th><th>最差周末 BTC</th></tr></thead><tbody>{rows}</tbody></table></div><p class="caption">Source: audited Deribit trade / settlement research. {data['period']['start']} to {data['period']['end']}, {data['period']['oos_weekends']} OOS weekends per case. Initial equity 1 BTC. Same 5% premium haircut, fees, print-size limits and hard constraints. All 9 × 198 allocations checked with target columns removed; current balanced allocations reproduced.</p></article><div class="grid"><article>{charts[0]}</article><article>{charts[1]}</article></div><article><h2>按策略查看每个周末的逐腿持仓</h2><div class="filters"><label for="grid-case">策略组合</label><select id="grid-case">{options}</select><label for="grid-weekend">策略周末</label><select id="grid-weekend">{dates}</select><button id="grid-download" type="button">导出所选策略全部周末持仓 CSV</button></div><p id="grid-case-metrics"></p><p id="grid-weekend-state" class="caption"></p><div class="scroll"><table><thead><tr><th>合约</th><th>预期结算 BTC</th><th>预期净收益 / 单位</th><th>卖出数量 q</th><th>omega</th><th>相对数量占比</th><th>实际净损益 BTC</th></tr></thead><tbody id="grid-legs"></tbody></table></div><p class="caption">每个策略包含全部 250 个周末和 1751 条候选腿记录；预热周末数量为零、预测值 unavailable。数量占比不是资金占比。其他策略的持仓变化不改变当前平衡策略。</p></article>
<script>
const gridMetrics={json.dumps(summary)},gridCases={json.dumps(compact,allow_nan=False)};
const gridCase=document.getElementById('grid-case'),gridWeekend=document.getElementById('grid-weekend');
function gridNum(x,n=6){{return Number.isFinite(x)?x.toFixed(n):'未预测';}}
function gridRender(){{const m=gridMetrics.find(x=>x.scenario===gridCase.value),c=gridCases[gridCase.value],w=c.weekends.find(x=>x.weekend_id===gridWeekend.value);document.getElementById('grid-case-metrics').textContent=`尾部 ${{m.tail_penalty}} / 协方差 ${{m.covariance_penalty}}：年化收益 ${{gridNum(m.cagr*100,2)}}%，Sharpe ${{gridNum(m.weekly_return_sharpe,3)}}，最大结算回撤 ${{gridNum(m.max_drawdown_pct,3)}}%，交易周末 ${{m.traded_weekends}} / ${{m.oos_weekends}}`;document.getElementById('grid-weekend-state').textContent=`${{w.weekend_id}} · ${{w.status}} · 该周末实际净损益 ${{gridNum(w.net_pnl_btc)}} BTC`;
document.getElementById('grid-legs').innerHTML=c.positions.filter(x=>x.weekend_id===gridWeekend.value).map(x=>`<tr><td>${{x.symbol}}</td><td>${{gridNum(x.expected_payoff_btc)}}</td><td>${{gridNum(x.predicted_net_per_unit_btc)}}</td><td>${{gridNum(x.quantity,1)}}</td><td>${{gridNum(x.omega,1)}}</td><td>${{gridNum(100*x.relative_abs_weight,1)}}%</td><td>${{gridNum(x.net_pnl_btc)}}</td></tr>`).join('');}}
gridCase.onchange=gridRender;gridWeekend.onchange=gridRender;
document.getElementById('grid-download').onclick=()=>{{const columns=['weekend_id','symbol','quantity','omega','relative_abs_weight','expected_payoff_btc','predicted_net_per_unit_btc','net_pnl_btc'];const csv='\\ufeff'+columns.join(',')+'\\n'+gridCases[gridCase.value].positions.map(r=>columns.map(k=>r[k]??'').join(',')).join('\\n');const url=URL.createObjectURL(new Blob([csv],{{type:'text/csv;charset=utf-8'}}));const a=document.createElement('a');a.href=url;a.download=gridCase.value+'_positions.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}};
gridCase.value='{balance}';gridRender();
</script>'''
