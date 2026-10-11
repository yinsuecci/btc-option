# Final Trade-only Formulas

## Full-sample Ridge forecast (BTC per one BTC option notional)

Training 2022-01-01 through 2026-10-10, 250 eligible weekends,
1751 contract observations. Retrospective final fit, not backfilled into OOS performance.

```text
alpha_hat_BTC = 0.0027246716509
+0.00390601389373 * momentum_1d
-0.00571363883341 * momentum_3d
+0.00364608996041 * momentum_7d
-0.000660223924526 * rv_7d
-0.000840451393102 * rv_30d
-0.000249137319842 * spot_volume_z30
+0.000453796483411 * jump_z
-0.000530651841786 * trade_iv
-3.50775383831e-05 * iv_minus_rv30
+0.000220832468035 * log_moneyness
-0.00260122616409 * days_to_expiry
-0.0595639078971 * trade_price_btc
+0.000605038860925 * delta_log_model
+2.03407104603e-05 * gamma_log_model
-0.152273630099 * vega_btc_model
-0.644599677453 * theta_btc_model
-0.000158226215202 * is_call
-2.47931951845e-06 * trade_age_minutes
-0.000284207195326 * log_trade_amount
-0.000146049852812 * log_window_trade_count
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

q* = argmax(alpha*q - 0.02 * sum((risk_i * q_i)^2)); omega = -q.
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
