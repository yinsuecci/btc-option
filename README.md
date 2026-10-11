# option-backtest

## BTC 周末期权的条件分布预测与约束多腿配置

### 摘要

本文构建 BTC 周末期权研究框架：以市场因子估计到期对数收益的条件均值，以历史逐步预测误差估计条件二阶矩，再由预测分布计算逆向 BTC 结算期权的支付期望及协方差，通过带压力风险惩罚的约束优化确定多腿卖出数量。估计采用标签成熟约束下的扩展窗口 Ridge 回归。实验包含 2022-01-01 至 2026-10-10 的 250 个周末、2,330 条候选合约记录，其中 198 个周末用于逐步预测与评价。固定北京时间周六 00:00 入场、周日 16:00 交割，持有 40 小时。平衡配置的 BTC 累计收益为 31.8841%，复合年化收益为 7.6056%，年化周度 Sharpe 为 2.2023，最大结算净值回撤为 2.1107%。结果基于真实成交记录及模拟执行成本，适用于策略研究和参数敏感性分析。

**关键词：** BTC 期权；条件价格分布；逐步预测；Ridge 回归；逆向结算；组合优化。

## 1. 数据与记号

以 $`t`$ 表示周末，$`i`$ 表示候选腿。决策时间固定为周五 16:00 UTC（北京时间周六 00:00），成交窗口为周五 15:00--16:00 UTC，合约于周日 08:00 UTC（北京时间周日 16:00）交割，持有期恒定为 40 小时。日历以所属周末的周六日期作为编号。

- $`S_t`$：窗口最后一笔成交所附带的 BTC 指数价格，单位 USD。
- $`D_t`$：官方交割指数价格，单位 USD。
- $`K_{t,i}`$：行权价，单位 USD。
- $`P_{t,i}`$：该合约窗口内最后一笔合格卖方成交价，单位 BTC。
- $`A_{t,i}`$：上述成交数量；$`I_{t,i}`$：年化成交隐含波动率。
- $`\tau_t`$：决策至到期的天数。
- $`q_{t,i}\geq0`$：卖出数量；$`\omega_{t,i}=-q_{t,i}`$：带方向持仓。
- $`W_{t-1}`$：开仓前 BTC 权益；初始权益 $`W_0=1`$。
- $`\Phi`$、$`\phi`$：标准正态累积分布函数及密度函数。

数据包括 Deribit 成交和官方交割价格，以及 Binance 已收盘现货日线。完整性口径为指定的一小时成交窗口。候选集合 $`\mathcal C_t`$ 排除大宗与组合成交，并满足

```math
0<P_i<1,\quad 0.05\leq I_i\leq3,\quad A_i\geq0.1,
\quad0\leq\mathrm{age}_i\leq60\text{ min},
\quad0.90\leq S_t/K_i\leq1.10. \tag{1}
```

到期晚于决策且相距不超过两天，其纽约本地日期为周日。现货日线须在决策前收盘，至少有 32 条记录，最近记录距决策不超过 36 小时。

## 2. 市场因子

令 $`C_n,V_n`$ 为决策前最后一个已收盘日的价格和成交量，$`r_n=\log(C_n/C_{n-1})`$。

```math
M_{h,t}=C_n/C_{n-h}-1,\qquad h\in\{1,3,7\}. \tag{2}
```

```math
\bar r_h=\frac1h\sum_{j=0}^{h-1}r_{n-j},\qquad
RV_{h,t}=\sqrt{\frac{365}{h-1}\sum_{j=0}^{h-1}(r_{n-j}-\bar r_h)^2},
\quad h\in\{7,30\}. \tag{3}
```

令 $`\bar V_{30},s_{V,30}`$ 为最近 30 日成交量的均值和样本标准差，$`s_{r,30}`$ 为最近 30 日对数收益的样本标准差：

```math
Z_{V,t}=\frac{V_n-\bar V_{30}}{\max(s_{V,30},10^{-10})},\qquad
J_t=\frac{|r_n|}{\max(s_{r,30},10^{-10})}. \tag{4}
```

```math
I_t=\mathrm{median}_{i\in\mathcal C_t}I_{t,i},\qquad
\tau_t=\frac{\text{expiry}_t-\text{decision}_t}{24\text{ h}}. \tag{5}
```

均值与方差模型的因子向量分别为

```math
x_t=(M_{1,t},M_{3,t},M_{7,t},RV_{7,t},RV_{30,t},Z_{V,t},J_t,I_t,\tau_t)^\top, \tag{6}
```

```math
u_t=(RV_{7,t},RV_{30,t},Z_{V,t},J_t,I_t,\tau_t)^\top. \tag{7}
```

动量及波动率使用小数单位，例如 50% 年化 IV 输入为 $`0.50`$。

## 3. 条件均值估计

每个周末构造一个价格标签 $`Y_t=\log(D_t/S_t)`$。令 $`a_s`$ 为标签可用时间，$`d_t`$ 为决策时间，成熟训练集合为

```math
\mathcal T_t=\{s:s<t,\ a_s<d_t\}. \tag{8}
```

在该集合内计算因子均值 $`m_{t,j}`$ 和总体标准差 $`d_{t,j}`$，令 $`z_{s,j}=(x_{s,j}-m_{t,j})/d_{t,j}`$。零方差因子的尺度设为 1。Ridge 目标为

```math
(\hat a_t,\hat b_t)=\arg\min_{a,b}
\left[\sum_{s\in\mathcal T_t}(Y_s-a-b^\top z_s)^2+10\|b\|_2^2\right]. \tag{9}
```

截距自由估计。对于中心化矩阵 $`Z_c`$ 和目标 $`Y_c`$，斜率满足

```math
\hat b_t=(Z_c^\top Z_c+10\mathbf I)^{-1}Z_c^\top Y_c. \tag{10}
```

转换为原始因子单位：

```math
\beta_{t,j}=\hat b_{t,j}/d_{t,j},\qquad
\beta_{t,0}=\hat a_t-\sum_jm_{t,j}\beta_{t,j},\qquad
\mu_t=\beta_{t,0}+\beta_t^\top x_t. \tag{11}
```

## 4. 条件预测误差模型

在至少 13 个成熟历史周末后，对每个历史时点 $`s`$ 用其当时的训练前缀生成预测 $`\hat\mu_s`$，结算后得到

```math
e_s=Y_s-\hat\mu_s,\qquad h_s=\max(e_s^2,10^{-10}). \tag{12}
```

令 $`\mathcal E_t`$ 为决策前已成熟且已生成逐步误差的历史集合。以 $`\log h_s`$ 为目标、$`u_s`$ 为输入，采用式 (9) 的标准化 Ridge 方法，得到原始单位预测

```math
g_t(u)=\gamma_{t,0}+\gamma_t^\top u,\qquad
\tilde g_t(u)=\mathrm{clip}(g_t(u),-30,5). \tag{13}
```

指数还原校准及方差预测为

```math
c_t=\frac1{|\mathcal E_t|}\sum_{s\in\mathcal E_t}h_s e^{-\tilde g_t(u_s)}, \tag{14}
```

```math
v_t^{raw}=c_te^{\tilde g_t(u_t)},\qquad
v_t=\mathrm{clip}(v_t^{raw},10^{-6},0.04),\qquad \sigma_t=\sqrt{v_t}. \tag{15}
```

该方法将条件预测误差二阶矩作为分布方差的近似。$`\sigma_t`$ 对应决策至到期的持有期，输入的 $`RV`$ 和 $`I`$ 为年化量。

## 5. 到期价格分布

给定决策信息 $`\mathcal F_t`$，假设

```math
Y_t\mid\mathcal F_t\sim\mathcal N(\mu_t,v_t),\qquad
D_t=S_te^{\mu_t+\sigma_t Z},\quad Z\sim\mathcal N(0,1). \tag{16}
```

价格期望、中值与方差为

```math
E[D_t\mid\mathcal F_t]=S_te^{\mu_t+v_t/2},\qquad
\mathrm{Median}(D_t\mid\mathcal F_t)=S_te^{\mu_t}, \tag{17}
```

```math
\mathrm{Var}(D_t\mid\mathcal F_t)=S_t^2e^{2\mu_t+v_t}(e^{v_t}-1). \tag{18}
```

95% 预测区间为

```math
[L_t,U_t]=[S_te^{\mu_t-1.959963985\sigma_t},S_te^{\mu_t+1.959963985\sigma_t}]. \tag{19}
```

## 6. 逆向期权支付与收益优势

一单位合约的 BTC 到期支付为

```math
H_C(D,K)=\frac{(D-K)^+}{D},\qquad H_P(D,K)=\frac{(K-D)^+}{D}. \tag{20}
```

截断正态指数矩满足

```math
E[e^{-Y}\mathbf1_{\{Y>a\}}]=e^{-\mu+v/2}\Phi\left(\frac{\mu-v-a}{\sqrt v}\right). \tag{21}
```

令 $`a_i=\log(K_i/S_t)`$、$`B_i=(K_i/S_t)e^{-\mu_t+v_t/2}`$，可得

```math
\bar H_{C,i}=\Phi\left(\frac{\mu_t-a_i}{\sigma_t}\right)
-B_i\Phi\left(\frac{\mu_t-v_t-a_i}{\sigma_t}\right), \tag{22}
```

```math
\bar H_{P,i}=B_i\Phi\left(\frac{a_i-\mu_t+v_t}{\sigma_t}\right)
-\Phi\left(\frac{a_i-\mu_t}{\sigma_t}\right). \tag{23}
```

上述支付期望采用式 (16) 的预测概率分布。每单位开仓手续费、扣成本收入及收益优势定义为

```math
f_i=\min(0.0003,0.125P_i),\qquad p_i^{eff}=0.95P_i-f_i,
\qquad\alpha_i=p_i^{eff}-\bar H_i. \tag{24}
```

5% 为模拟成交价格折让。组合预期净收益为 $`E[\Pi_t\mid\mathcal F_t]=\alpha_t^\top q_t`$。

## 7. 协方差与压力风险

### 7.1 支付协方差

```math
\Sigma_{ij}=E[H_iH_j]-E[H_i]E[H_j],\qquad
\mathrm{Var}(\Pi_t\mid\mathcal F_t)=q_t^\top\Sigma_tq_t. \tag{25}
```

实现采用 128 节点 Gauss--Hermite 积分。令 $`(x_k,\tilde w_k)`$ 为节点和权重、$`w_k=\tilde w_k/\sqrt\pi`$：

```math
D_k=S_te^{\mu_t+\sqrt{2v_t}x_k},\qquad
\bar H_i^{GH}=\sum_{k=1}^{128}w_kH_i(D_k), \tag{26}
```

```math
\Sigma_{ij}\approx\sum_{k=1}^{128}w_k
(H_i(D_k)-\bar H_i^{GH})(H_j(D_k)-\bar H_j^{GH}). \tag{27}
```

### 7.2 情景估值

压力情景为现货相对冲击与年化 IV 绝对增量的笛卡尔积：

```math
\mathcal S=\{-0.30,-0.15,-0.05,0,0.05,0.15,0.30\}\times\{0,0.25,0.50\}. \tag{28}
```

使用零利率 Black--Scholes 模型，$`T=\text{剩余小时数}/8760`$：

```math
d_1=\frac{\log(S/K)+I^2T/2}{I\sqrt T},\qquad d_2=d_1-I\sqrt T, \tag{29}
```

```math
C_{USD}=S\Phi(d_1)-K\Phi(d_2),\qquad
P_{USD}=K\Phi(-d_2)-S\Phi(-d_1),\qquad V^{BTC}=V_{USD}/S. \tag{30}
```

情景 $`\ell=(a,b)`$ 保持剩余期限不变：

```math
V_{\ell,i}^{BTC}=\frac{BS_{USD}(S_t(1+a),K_i,T_i,I_i+b)}{S_t(1+a)},
\quad L_{\ell,i}=V_{\ell,i}^{BTC}-p_i^{eff},
\quad R_i=\max(0.001,\max_\ell L_{\ell,i}). \tag{31}
```

### 7.3 敏感度

```math
\Delta_{USD}=\Phi(d_1)-\mathbf1_{\{Put\}},\quad
\Gamma_{USD}=\frac{\phi(d_1)}{SI\sqrt T},\quad Vega_{USD}=S\phi(d_1)\sqrt T, \tag{32}
```

```math
\Delta_{\log}=\Delta_{USD}-V^{BTC},\quad
\Gamma_{\log}=S\Gamma_{USD}-\Delta_{USD}+V^{BTC},\quad
Vega^{BTC}=Vega_{USD}/S. \tag{33}
```

实现同时记录日 Theta：$`Theta^{BTC}=-\phi(d_1)I/(2\sqrt T\cdot365)`$。当前硬约束使用 Delta 和 Vega。

## 8. 约束多腿配置

### 8.1 优化目标

```math
q_t^*=\arg\max_q\left[\alpha_t^\top q-\lambda_\Sigma q^\top\Sigma_tq
-\lambda_R\sum_i(R_{t,i}q_i)^2\right]. \tag{34}
```

$`\lambda_\Sigma`$ 为协方差惩罚，$`\lambda_R`$ 为压力风险惩罚，数值采用 BTC 单位约定。令 $`Q=\lambda_\Sigma\Sigma+\lambda_R\mathrm{diag}(R_i^2)`$。在省略约束且 $`Q`$ 可逆时，一阶条件给出 $`q=Q^{-1}\alpha/2`$；实际配置求解完整约束问题。

### 8.2 硬约束

```math
0\leq q_i\leq\min(0.5,A_{t,i}),\qquad\sum_iq_i\leq2. \tag{35}
```

定义价外距离与单位保证金代理：

```math
OTM_i=\begin{cases}(K_i-S_t)^+/S_t,&Call,\\(S_t-K_i)^+/S_t,&Put,\end{cases}
\qquad m_i=\max(0.15-OTM_i,0.10)+P_i. \tag{36}
```

```math
m^\top q\leq\min(0.50,0.5W_{t-1}),\qquad
|\Delta_{\log}^\top q|\leq0.30,\qquad |(Vega^{BTC})^\top q|\leq0.20, \tag{37}
```

```math
L_\ell^\top q\leq\min(0.25,0.25W_{t-1}),\qquad\forall\ell\in\mathcal S. \tag{38}
```

求解采用 SLSQP、零向量初值、500 次最大迭代和 $`10^{-10}`$ 目标容差；目标和梯度乘以 1000 调整数值尺度。

### 8.3 离散数量

```math
\tilde q_i=0.1\left\lfloor\frac{\max(q_i^*,0)}{0.1}+10^{-9}\right\rfloor,
\qquad\omega_i=-\tilde q_i. \tag{39}
```

取整后检查全部约束，允许数值误差 $`10^{-7}`$。检查通过、$`\alpha^\top\tilde q>0`$ 且总数量为正时开仓，其余情形记录空仓。相对数量占比为 $`w_i^{abs}=\tilde q_i/\sum_j\tilde q_j`$，空仓时设为 0。带方向组合 Greeks 为 $`-\Delta_{\log}^\top\tilde q`$ 和 $`-(Vega^{BTC})^\top\tilde q`$。

## 9. 逐步回测与评价

### 9.1 时间推进

1. 用成熟历史标签拟合均值模型，收集历史逐步预测误差。
2. 成熟价格标签至少 52 个、成熟误差至少 26 个时，估计当周条件分布。
3. 用可见因子及候选成交计算支付期望、协方差和压力风险。
4. 求解数量，完成离散约束检查。
5. 到期后按官方交割价格记账，更新权益并推进下一周。

### 9.2 现金流与绩效

```math
\Pi_{t,i}=\tilde q_{t,i}[p_{t,i}^{eff}-H_i(D_t)],\quad
\Pi_t=\sum_i\Pi_{t,i},\quad W_t=W_{t-1}+\Pi_t. \tag{40}
```

开仓现金流为 $`\tilde q_i p_i^{eff}`$，交割现金流为 $`-\tilde q_iH_i(D_t)`$。周日到期合约的交割费用设为 0。逐腿、逐周及事件账本进行金额勾稽。

对全部正式评价周末，包括空仓周末，定义 $`r_t^p=\Pi_t/W_{t-1}`$。令 $`N`$ 为评价周末数，$`Y`$ 为首末评价日期间隔除以 365.25 天：

```math
R_{total}=W_N/W_0-1,\quad CAGR=(W_N/W_0)^{1/Y}-1,\quad
Sharpe=\sqrt{52}\,\bar r^p/s(r^p), \tag{41}
```

其中 $`s(r^p)`$ 使用分母 $`N-1`$ 的样本标准差，BTC 无风险收益设为 0。令 $`Peak_t=\max(W_0,\ldots,W_t)`$：

```math
DD_t=W_t/Peak_t-1,\qquad MDD=-\min_tDD_t. \tag{42}
```

### 9.3 预测诊断

历史均值基准为 $`\mu_t^0=|\mathcal T_t|^{-1}\sum_{s\in\mathcal T_t}Y_s`$：

```math
RMSE=\sqrt{\frac1N\sum_t(Y_t-\mu_t)^2},\quad
R^2_{baseline}=1-\frac{\sum_t(Y_t-\mu_t)^2}{\sum_t(Y_t-\mu_t^0)^2}, \tag{43}
```

```math
NLL=\frac1{2N}\sum_t\left[\log(2\pi v_t)+\frac{(Y_t-\mu_t)^2}{v_t}\right],
\quad Coverage_{95}=\frac1N\sum_t\mathbf1_{\{L_t\leq D_t\leq U_t\}}. \tag{44}
```

额外诊断包括 8 周连续区块、5,000 次重采样的固定策略总收益区间，以及保持历史持仓数量不变的 0%、5%、10%、20% 成交折让分析。

## 10. 实验结果

### 10.1 平衡配置

数据截面截至 2026-10-11。预热 52 个周末，正式评价 198 个周末，所属周末区间为 2022-12-31 至 2026-10-10。182 个周末开仓，6 个周末取整后缺少正预期组合，10 个周末取整后约束检查未通过。

- 净收益：0.318841491 BTC；期末权益：1.318841491 BTC。
- 累计收益：31.8841%；复合年化收益：7.6056%。
- 年化周度 Sharpe：2.2023；最大结算净值回撤：2.1107%。
- 中心预测 RMSE：0.0201473；历史均值基准 RMSE：0.0187923。
- 相对基准 R²：-0.149406；95% 预测区间覆盖率：95.9596%。
- 高斯 NLL：-2.54097；历史均值及方差基准 NLL：-2.50381。

均值预测误差高于历史均值基准，区间覆盖接近标称水平；分布及策略稳健性有待独立后续样本检验。

### 10.2 风险参数敏感性

固定网格 $`\lambda_R\in\{0.005,0.01,0.02\}`$、$`\lambda_\Sigma\in\{0.5,1,2\}`$，共九组。各组共享逐步分布预测，独立顺序更新数量、收益及权益。代表配置在计算前定义为：

- 激进配置：尾部惩罚 0.005、协方差惩罚 0.5；年化 8.6376%，Sharpe 2.3896，最大结算回撤 2.7135%。
- 平衡配置：尾部惩罚 0.01、协方差惩罚 1；年化 7.6056%，Sharpe 2.2023，最大结算回撤 2.1107%。
- 保守配置：尾部惩罚 0.02、协方差惩罚 2；年化 6.8152%，Sharpe 2.0954，最大结算回撤 2.5267%。

分组描述惩罚强度，实际回撤排序取决于组合与历史路径。比较使用同一段已查看历史，属于参数敏感性研究。完整九组摘要位于 `results/risk_grid/summary.csv`。

### 10.3 入场时间的选择与评价范围

当前仓库的主回测固定北京时间周六 00:00 入场，每周一次持仓至周日交割。该时间通过已查看历史的入场时间比较选出。模型参数仍按成熟历史逐周拟合，但时点选择属于同样本研究；对整个策略的独立验证需追加后续数据。主报告仅展示这一入场时间，风险参数分组也使用同一时间。

## 11. 最终研究系数

最终均值模型使用全部 250 个成熟周末，误差模型使用 237 个逐步预测误差。原始因子单位系数的舍入展示如下：

```math
\begin{aligned}
\mu={}&0.00766394053005+0.128736853408M_1-0.0359468603205M_3\\
&+0.0312491580746M_7+0.00570256554321RV_7+0.000577030345893RV_{30}\\
&+0.00180508547307Z_V+0.00127208496881J-0.0263097513357I.
\end{aligned} \tag{45}
```

```math
\begin{aligned}
g={}&-11.9119452191-0.129114223731RV_7+1.62705077587RV_{30}\\
&+0.236721084137Z_V+0.0956487231686J+3.72937929167I.
\end{aligned} \tag{46}
```

```math
c=5.122779836848,\qquad
v=\mathrm{clip}(c\exp[\mathrm{clip}(g,-30,5)],10^{-6},0.04). \tag{47}
```

式 (45)--(47) 是最终数据截面的研究摘要，历史评价使用各自决策时点的参数。持有期恒为 40 小时，到期天数因子为常数，其最终斜率为 0。完整精度保存在 `final_formula.json`，历史参数保存在 `coefficients.csv`。

## 12. 实现与复现

### 12.1 环境

Python 3.12 或更高版本。Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Linux/macOS：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

`requirements.txt` 保存实验环境版本；`run.ps1` 为原始本机运行器，使用本机 Codex Python 路径。

### 12.2 固定截面复算

在项目根目录运行：

```bash
python -m scripts.download_trade_history --start 2022-01-01 --end 2026-10-10
python -m scripts.run_center_vol
python -m scripts.audit_center_vol_lookahead
python -m scripts.run_risk_grid
python -m scripts.validate_risk_grid
python -m scripts.render_center_vol
python -m scripts.validate_center_vol
python -m scripts.publish_center_vol
python -m pytest -q
```

下载依赖公共端点的可用性，原始数据及覆盖记录应随研究截面归档。显式调整尾部惩罚并另存结果：

```bash
python -m scripts.run_center_vol --risk-penalty 0.02 --output results/center_vol_custom
```

报告预览：

```bash
python -m http.server 8766 --directory results/center_vol
```

地址为 `http://127.0.0.1:8766/report.html`；自包含 HTML 亦支持本地文件浏览。

### 12.3 实现与输出

- `src/trade_data.py`：历史数据下载及窗口覆盖。
- `src/trade_research.py`：候选过滤、日线因子、Greeks、压力情景与约束优化。
- `src/center_vol.py`：逐步均值及方差拟合、解析支付期望及协方差积分。
- `src/center_vol_metrics.py`：权益收益、回撤和全候选腿持仓导出。
- `src/risk_grid.py`：九组风险参数的顺序回测。
- `src/center_vol_report.py`、`src/risk_grid_report.py`：交互式报告。

`results/center_vol` 保存训练数据、误差、预测、系数、交易、完整持仓、权益曲线、事件账本、最终参数、指标及报告。主要文件为 `price_training.csv`、`causal_mean_residuals.csv`、`forecasts.csv`、`coefficients.csv`、`predictions.csv`、`trades.csv`、`positions_by_weekend.csv`、`weekends.csv`、`performance_curve.csv`、`event_ledger.csv`、`final_formula.json`、`metrics.json`、`report.html`。持仓导出涵盖全部 250 个周末的候选腿，预热期间数量为 0、预测值为空。

`results/risk_grid` 保存同一入场时间下的九组风险参数摘要、结果目录和验证记录。`lookahead_audit.json` 保存历史前缀重算、未来标签扰动、无目标配置及原始窗口检查；`validation.json` 保存输出一致性检查。原始成交数据位于 `data/entry_timing/fri_16utc`。其他时点的历史结果已移入 `results/archive_before_friday_entry`，不参与主报告。

### 12.4 GitHub 发布

发布范围建议包括源代码、测试、README、依赖声明及适量结果摘要。原始数据和大型报告按许可与仓库容量单独管理，并记录数据截面及覆盖校验。公式使用 GitHub 支持的 math 围栏；行内符号使用带反引号的数学定界符，以减少 Markdown 转义冲突。本机虚拟环境、依赖缓存、归档和临时目录保留在忽略规则中。

## 13. 假设与研究边界

条件高斯对数收益及误差二阶矩校准构成分布近似，偏度、厚尾、跳跃和状态切换可作为扩展方向。执行采用历史卖方成交代理、5% 折让和固定手续费；盘口深度、排队及成交概率属于后续执行研究。数量上限借助历史成交量，保证金使用研究代理，压力估值采用固定期限的零利率模型。收益与回撤使用 BTC 周末结算权益；持仓期间盯市、动态对冲、融资及强平过程留待独立建模。后续评价应使用独立时间区间与预先固定的参数协议。

## 附录：历史实验与资料

仓库保留早期直接期权净收益回归及合成报价链实验。本文模型和结果对应 `results/center_vol`，历史报告保存在 `report_netalpha.html` 与 `results/report_synthetic.html`，供实现比较和回归测试使用。

数据配置见 `RESEARCH_SOURCE.json`，验证说明见 `VALIDATION.md`。接口资料参阅 [Deribit 历史成交 API](https://docs.deribit.com/api-reference/market-data/public-get_last_trades_by_currency_and_time)。
