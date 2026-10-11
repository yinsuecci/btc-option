# Price center, volatility and leg allocation

Final retrospective fit: 2022-01-01 through 2026-10-10.
Price training rows: 250. Variance rows: 237.
Each row is ONE weekend, not one option contract. Final parameters are NOT backfilled into history.

## Raw-unit fitted parameters
mu_log_return = 0.00766394053005
+0.128736853408 * momentum_1d
-0.0359468603205 * momentum_3d
+0.0312491580746 * momentum_7d
+0.00570256554321 * rv_7d
+0.000577030345893 * rv_30d
+0.00180508547307 * spot_volume_z30
+0.00127208496881 * jump_z
-0.0263097513357 * median_trade_iv
+0 * days_to_expiry

g_log_squared_error = -11.9119452191
-0.129114223731 * rv_7d
+1.62705077587 * rv_30d
+0.236721084137 * spot_volume_z30
+0.0956487231686 * jump_z
+3.72937929167 * median_trade_iv
+0 * days_to_expiry

## Distribution
r = ln(D / S); mu = E[r | X].
Center beta is Ridge on r; standardization fitted on prior training data only; L2=10.
e_u = r_u - mu_hat_u, where mu_hat_u was fitted before u with at least 13 matured weekends.
Gamma is Ridge on ln(max(e_u^2, 1e-10)), L2=10, using only prior matured prequential errors.
c = mean(max(e_u^2, 1e-10) / exp(g(Z_u))) over the variance training data.
Final c = 5.12277983685.
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
q* = argmax(alpha' q - 1*q' Sigma q - 0.01*sum((R_i*q_i)^2)).
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
