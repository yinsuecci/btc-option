"""Causal trade-only, native-BTC weekend research. Not an executable-book replay."""
import gzip
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from src.core import ET, bs
from src.deribit import settlement_btc

FEATURES = ['momentum_1d', 'momentum_3d', 'momentum_7d', 'rv_7d', 'rv_30d',
            'spot_volume_z30', 'jump_z', 'trade_iv', 'iv_minus_rv30',
            'log_moneyness', 'days_to_expiry', 'trade_price_btc',
            'delta_log_model', 'gamma_log_model', 'vega_btc_model',
            'theta_btc_model', 'is_call', 'trade_age_minutes',
            'log_trade_amount', 'log_window_trade_count']
MONTHS = {name: index for index, name in enumerate(
    ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'], 1)}


@dataclass
class TradeConfig:
    min_train_weekends: int = 26
    ridge_alpha: float = 10.
    slippage_fraction: float = .05
    fee_btc: float = .0003
    fee_premium_cap: float = .125
    max_trade_age_minutes: float = 60.
    initial_equity_btc: float = 1.
    max_leg_quantity: float = .5
    gross_limit: float = 2.
    lot: float = .1
    delta_log_limit: float = .30
    vega_btc_limit: float = .20
    margin_proxy_limit: float = .50
    stress_loss_limit: float = .25
    risk_penalty_per_btc: float = .02


def contract(symbol):
    match = re.fullmatch(r'BTC-(\d{1,2})([A-Z]{3})(\d{2})-(\d+(?:\.\d+)?)-([CP])', symbol)
    if not match:
        raise ValueError('Not a native BTC vanilla option')
    day, month, year, strike, kind = match.groups()
    expiry = pd.Timestamp(datetime(2000 + int(year), MONTHS[month], int(day), 8, tzinfo=timezone.utc))
    return expiry, float(strike), 'CALL' if kind == 'C' else 'PUT'


def model_value(s, k, hours, iv, kind):
    return bs(s, k, hours, iv, kind)[0] / s


def inverse_greeks(s, k, hours, iv, kind):
    usd, delta, gamma, vega, theta = bs(s, k, hours, iv, kind)
    value = usd / s
    # Derivatives with respect to log(S) of V_USD/S, not mislabeled USD Greeks.
    return dict(delta_log_model=delta - value,
                gamma_log_model=s * gamma - delta + value,
                vega_btc_model=vega / s, theta_btc_model=theta / s)


def market_features(spot, t):
    history = spot.loc[spot.timestamp < t].sort_values('timestamp')
    if len(history) < 32 or t - history.timestamp.iloc[-1] > pd.Timedelta(hours=36):
        raise ValueError('Insufficient or stale closed daily spot history')
    prices = history.spot.to_numpy()
    returns = np.diff(np.log(prices))
    result = {f'momentum_{n}d': prices[-1] / prices[-1-n] - 1 for n in (1, 3, 7)}
    result.update(rv_7d=np.std(returns[-7:], ddof=1) * np.sqrt(365),
                  rv_30d=np.std(returns[-30:], ddof=1) * np.sqrt(365))
    volumes = history.volume.iloc[-30:]
    result['spot_volume_z30'] = (volumes.iloc[-1] - volumes.mean()) / max(volumes.std(), 1e-10)
    result['jump_z'] = abs(returns[-1]) / max(np.std(returns[-30:], ddof=1), 1e-10)
    return result, history.timestamp.iloc[-1]


def per_unit_cashflows(price, delivery, strike, kind, cfg):
    settlement = settlement_btc(delivery, strike, kind)
    fee = min(cfg.fee_btc, cfg.fee_premium_cap * price)
    slippage = cfg.slippage_fraction * price
    # Sunday expiries are daily options, exempt from delivery fees in this model.
    return dict(settlement_btc=settlement, entry_fee_btc=fee,
                slippage_btc=slippage, delivery_fee_btc=0.,
                y_btc=price - settlement - fee - slippage)


def build_observations(directory, cfg=TradeConfig()):
    root = Path(directory)
    coverage = pd.read_csv(root / 'download_coverage.csv')
    spot = pd.read_csv(root / 'spot_daily.csv')
    spot['timestamp'] = pd.to_datetime(spot.timestamp, utc=True)
    deliveries = {r['date']: float(r['delivery_price']) for r in json.loads((root / 'delivery_prices.json').read_text())}
    observations, audits, rejected = [], [], []
    for r in coverage.itertuples():
        if r.status != 'DOWNLOADED_TRADE_WINDOW':
            audits.append(dict(weekend_id=r.weekend_id, status=r.status, candidates=0))
            continue
        t = pd.Timestamp(r.decision_time)
        with gzip.open(root / 'raw' / (r.weekend_id + '_trades.json.gz'), 'rt', encoding='utf-8') as stream:
            raw = json.load(stream)
        if not raw['complete']:
            raise ValueError('Incomplete trade window')
        trades = raw['trades']
        if any(row['timestamp'] > int(t.timestamp() * 1000) for row in trades):
            raise ValueError('Future trade in decision window')
        factors, spot_available_at = market_features(spot, t)
        ordered = sorted(trades, key=lambda row: (row['timestamp'], row['trade_id']))
        if not ordered:
            audits.append(dict(weekend_id=r.weekend_id, status='NO_WINDOW_TRADES', candidates=0))
            continue
        index_price = float(ordered[-1]['index_price'])
        latest = {}
        counts = {}
        for row in ordered:
            if row.get('direction') != 'sell' or row.get('block_trade_id') or row.get('combo_trade_id'):
                continue
            name = row['instrument_name']
            latest[name] = row
            counts[name] = counts.get(name, 0) + 1
        accepted = 0
        for name, row in latest.items():
            reason = None
            try:
                expiry, strike, kind = contract(name)
            except (ValueError, KeyError):
                continue
            if expiry <= t or expiry.tz_convert(ET).weekday() != 6 or expiry - t > pd.Timedelta(days=2):
                continue
            price = float(row['price'])
            iv = float(row.get('iv') or np.nan) / 100
            trade_time = pd.to_datetime(row['timestamp'], unit='ms', utc=True)
            age = (t - trade_time).total_seconds() / 60
            amount = float(row['amount'])
            if not np.isfinite([price, iv, amount, index_price]).all():
                reason = 'nonfinite_trade'
            elif not 0 < price < 1 or not .05 <= iv <= 3 or amount < cfg.lot:
                reason = 'invalid_trade_or_iv_or_size'
            elif age < 0 or age > cfg.max_trade_age_minutes:
                reason = 'stale_trade'
            elif not .90 <= index_price / strike <= 1.10:
                reason = 'outside_moneyness_filter'
            if reason:
                rejected.append(dict(weekend_id=r.weekend_id, symbol=name, reason=reason))
                continue
            hours = (expiry - t).total_seconds() / 3600
            greeks = inverse_greeks(index_price, strike, hours, iv, kind)
            date = expiry.date().isoformat()
            delivery = deliveries.get(date)
            d = dict(weekend_id=r.weekend_id, decision_time=t, trade_time=trade_time,
                     spot_available_at=spot_available_at, label_available_at=expiry,
                     expiry=expiry, symbol=name, strike=strike, option_type=kind,
                     spot=index_price, trade_id=row['trade_id'], trade_price_btc=price,
                     trade_iv=iv, trade_amount=amount, trade_age_minutes=age,
                     trade_direction='sell', days_to_expiry=hours / 24,
                     log_moneyness=np.log(index_price / strike),
                     iv_minus_rv30=iv - factors['rv_30d'],
                     is_call=float(kind == 'CALL'), log_trade_amount=np.log1p(amount),
                     log_window_trade_count=np.log1p(counts[name]),
                     delivery_index=delivery, **factors, **greeks)
            if delivery is not None:
                d.update(per_unit_cashflows(price, delivery, strike, kind, cfg))
            else:
                d.update(y_btc=np.nan, settlement_btc=np.nan, entry_fee_btc=np.nan,
                         slippage_btc=np.nan, delivery_fee_btc=np.nan)
            observations.append(d)
            accepted += 1
        audits.append(dict(weekend_id=r.weekend_id, status='CANDIDATES' if accepted else 'NO_ELIGIBLE_SELL_PRINT',
                           candidates=accepted, raw_trades=len(trades),
                           latest_index_timestamp=pd.to_datetime(ordered[-1]['timestamp'], unit='ms', utc=True)))
    return pd.DataFrame(observations), pd.DataFrame(audits), pd.DataFrame(rejected)


class TradeModel:
    def __init__(self, kind='ridge', alpha=10.):
        self.kind = kind
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha) if kind == 'ridge' else LinearRegression()

    def fit(self, observations, t, weekend_id):
        train = observations.loc[(observations.weekend_id < weekend_id) &
                                 (observations.label_available_at < t) & observations.y_btc.notna()]
        if train.empty:
            raise ValueError('No matured prior-weekend labels')
        x = train[FEATURES].to_numpy(float)
        self.model.fit(self.scaler.fit_transform(x), train.y_btc.to_numpy())
        self.beta = self.model.coef_ / self.scaler.scale_
        self.intercept = float(self.model.intercept_ - self.scaler.mean_ @ self.beta)
        self.sigma = float(np.std(train.y_btc.to_numpy() - self.predict(train), ddof=1))
        self.n = len(train)
        self.weekends = sorted(train.weekend_id.unique())
        self.last_label = train.label_available_at.max()
        return self

    def predict(self, rows):
        return self.intercept + rows[FEATURES].to_numpy(float) @ self.beta


def optimize_native(rows, alpha, cfg=TradeConfig(), equity=1., covariance=None, covariance_penalty=0.):
    if len(rows) == 0 or equity <= 0:
        return None
    covariance = np.zeros((len(rows), len(rows))) if covariance is None else np.asarray(covariance, dtype=float)
    if covariance.shape != (len(rows), len(rows)) or not np.isfinite(covariance).all():
        raise ValueError('Invalid payoff covariance')
    if covariance_penalty < 0 or not np.allclose(covariance, covariance.T) or np.linalg.eigvalsh(covariance).min() < -1e-10:
        raise ValueError('Covariance must be positive semidefinite and penalty nonnegative')
    shocks = [(s, v) for s in (-.30, -.15, -.05, 0., .05, .15, .30) for v in (0., .25, .50)]
    values = np.array([[model_value(r.spot * (1+s), r.strike, r.days_to_expiry * 24,
                                    r.trade_iv + v, r.option_type) for r in rows.itertuples()]
                       for s, v in shocks])
    prices = rows.trade_price_btc.to_numpy()
    fees = np.minimum(cfg.fee_btc, prices * cfg.fee_premium_cap)
    effective = prices * (1 - cfg.slippage_fraction) - fees
    losses = values - effective
    risk = np.maximum(losses.max(axis=0), .001)
    otm = np.where(rows.option_type.eq('CALL'), np.maximum(rows.strike - rows.spot, 0),
                   np.maximum(rows.spot - rows.strike, 0)) / rows.spot
    margin = np.maximum(.15 - otm, .10).to_numpy() + prices
    g = rows[['delta_log_model', 'vega_btc_model']].to_numpy(float)
    margin_limit = min(cfg.margin_proxy_limit, .5 * equity)
    stress_limit = min(cfg.stress_loss_limit, .25 * equity)
    bounds = [(0., min(cfg.max_leg_quantity, float(r.trade_amount))) for r in rows.itertuples()]
    constraints = [
        {'type': 'ineq', 'fun': lambda q: cfg.gross_limit - q.sum()},
        {'type': 'ineq', 'fun': lambda q: margin_limit - margin @ q},
        {'type': 'ineq', 'fun': lambda q: cfg.delta_log_limit - abs(g[:, 0] @ q)},
        {'type': 'ineq', 'fun': lambda q: cfg.vega_btc_limit - abs(g[:, 1] @ q)},
        {'type': 'ineq', 'fun': lambda q: stress_limit - losses @ q},
    ]
    scale = 1000.
    result = minimize(lambda q: scale * (-alpha @ q + cfg.risk_penalty_per_btc * np.sum((risk * q) ** 2) + covariance_penalty * q @ covariance @ q),
                      np.zeros(len(rows)), jac=lambda q: scale * (-alpha + 2 * cfg.risk_penalty_per_btc * risk ** 2 * q + 2 * covariance_penalty * covariance @ q),
                      method='SLSQP', bounds=bounds, constraints=constraints,
                      options={'maxiter': 500, 'ftol': 1e-10})
    if not result.success:
        return dict(status='OPTIMIZER_FAILED', message=result.message)
    q = np.floor(np.maximum(result.x, 0) / cfg.lot + 1e-9) * cfg.lot
    feasible = all(np.min(np.asarray(c['fun'](q))) >= -1e-7 for c in constraints)
    # Signed-delta cancellation can change after per-leg rounding.
    if not feasible:
        return dict(status='ROUNDING_INFEASIBLE', message='No trade after lot rounding')
    if alpha @ q <= 0 or not q.sum():
        return dict(status='NO_POSITIVE_ROUNDED_PORTFOLIO')
    return dict(status='ENTER', quantity=q, weights=-q, risk=risk,
                expected_net_btc=float(alpha @ q),
                delta_log_btc=float(-g[:, 0] @ q), vega_btc=float(-g[:, 1] @ q),
                margin_proxy_btc=float(margin @ q), worst_stress_btc=float(np.max(losses @ q)),
                stress_values=values, shocks=shocks, effective_premium=effective)


def run(directory='data/deribit_trade_daily', output='results/trade_only', cfg=TradeConfig()):
    p = Path(output)
    p.mkdir(parents=True, exist_ok=True)
    obs, coverage, rejected = build_observations(directory, cfg)
    if obs.empty:
        raise ValueError('No eligible real trade observations')
    if obs.y_btc.isna().any():
        raise ValueError('Historical candidate lacks official delivery index; finish settlement acquisition before fitting')
    obs.to_csv(p / 'observations.csv', index=False)
    coverage.to_csv(p / 'weekend_coverage.csv', index=False)
    rejected.to_csv(p / 'rejected_trades.csv', index=False)
    beta, predictions, legs, decisions, weekends, stress, baselines = [], [], [], [], [], [], []
    equity = cfg.initial_equity_btc
    for c in coverage.itertuples():
        if c.status == 'FUTURE_OR_UNSETTLED':
            continue
        rows = obs.loc[obs.weekend_id.eq(c.weekend_id)].copy()
        if rows.empty:
            empty_time = (pd.Timestamp(c.weekend_id, tz=ET) + pd.DateOffset(hours=12)).tz_convert('UTC')
            prior_count = obs.loc[(obs.weekend_id < c.weekend_id) &
                                  (obs.label_available_at < empty_time) & obs.y_btc.notna()].weekend_id.nunique()
            weekends.append(dict(weekend_id=c.weekend_id, status=c.status, net_pnl_btc=0.,
                                 gross_pnl_btc=0., costs_btc=0., baseline_pnl_btc=0., equity_btc=equity,
                                 traded=False, evaluation=prior_count >= cfg.min_train_weekends, candidates=0))
            continue
        t = rows.decision_time.iloc[0]
        train = obs.loc[(obs.weekend_id < c.weekend_id) & (obs.label_available_at < t) & obs.y_btc.notna()]
        matured = train.weekend_id.nunique()
        if matured < cfg.min_train_weekends:
            weekends.append(dict(weekend_id=c.weekend_id, status='WARMUP', net_pnl_btc=0.,
                                 gross_pnl_btc=0., costs_btc=0., baseline_pnl_btc=0., equity_btc=equity,
                                 traded=False, evaluation=False, candidates=len(rows)))
            continue
        models = {name: TradeModel(name, cfg.ridge_alpha).fit(obs, t, c.weekend_id) for name in ['ols', 'ridge']}
        m = models['ridge']
        for name, model in models.items():
            for f, b in zip(['intercept'] + FEATURES, [model.intercept] + list(model.beta)):
                beta.append(dict(weekend_id=c.weekend_id, model=name, factor=f, beta=float(b),
                                 training_samples=model.n, training_weekends=len(model.weekends),
                                 train_start=model.weekends[0], train_end=model.weekends[-1],
                                 last_label_available_at=model.last_label, decision_time=t))
        alpha = m.predict(rows)
        ols = models['ols'].predict(rows)
        result = optimize_native(rows, alpha, cfg, equity)
        action = result['status'] if result else 'NO_EQUITY'
        q = result['quantity'] if action == 'ENTER' else np.zeros(len(rows))
        for i, r in enumerate(rows.itertuples()):
            predictions.append(dict(weekend_id=c.weekend_id, symbol=r.symbol, decision_time=t,
                                    ridge_prediction_btc=float(alpha[i]), ols_prediction_btc=float(ols[i]),
                                    realized_per_unit_btc=float(r.y_btc), weight=float(-q[i]),
                                    intercept=m.intercept,
                                    **{f + '_contribution': float(getattr(r, f) * b) for f, b in zip(FEATURES, m.beta)}))
        decisions.append(dict(weekend_id=c.weekend_id, decision_time=t, action=action,
                              training_weekends=len(m.weekends), training_samples=m.n,
                              expected_net_btc=result.get('expected_net_btc', 0.) if result else 0.,
                              quantity=float(q.sum()), margin_proxy_btc=result.get('margin_proxy_btc', 0.) if result else 0.,
                              worst_stress_btc=result.get('worst_stress_btc', 0.) if result else 0.,
                              delta_log_btc=result.get('delta_log_btc', 0.) if result else 0.,
                              vega_btc=result.get('vega_btc', 0.) if result else 0.))
        gross = net = fees = 0.
        for i, (_, r) in enumerate(rows.iterrows()):
            if q[i] <= 0:
                continue
            if not np.isfinite(r.y_btc):
                raise ValueError('Missing delivery for selected position; refusing zero-PnL fill')
            leg_net = q[i] * r.y_btc
            leg_gross = q[i] * (r.trade_price_btc - r.settlement_btc)
            leg_cost = q[i] * (r.entry_fee_btc + r.slippage_btc)
            gross += leg_gross
            net += leg_net
            fees += leg_cost
            legs.append(dict(r.to_dict(), quantity=q[i], weight=-q[i], alpha_hat_btc=alpha[i],
                             gross_pnl_btc=leg_gross, costs_btc=leg_cost, net_pnl_btc=leg_net))
        baseline_net = 0.
        for kind in ['CALL', 'PUT']:
            group = rows.loc[rows.option_type.eq(kind)].copy()
            if group.empty:
                continue
            group['target_distance'] = abs(abs(group.delta_log_model) - .35)
            selected = group.sort_values(['target_distance', 'symbol']).iloc[0]
            if np.isfinite(selected.y_btc):
                amount = np.floor(min(.5, selected.trade_amount) / cfg.lot + 1e-9) * cfg.lot
                baseline_net += amount * selected.y_btc
                baselines.append(dict(weekend_id=c.weekend_id, symbol=selected.symbol,
                                      quantity=amount, net_pnl_btc=amount * selected.y_btc))
        if action == 'ENTER':
            for j, (s, v) in enumerate(result['shocks']):
                stress.append(dict(weekend_id=c.weekend_id, spot_shock=s, iv_shock=v,
                                   estimated_pnl_btc=float(q @ (result['effective_premium'] - result['stress_values'][j]))))
        equity += net
        weekends.append(dict(weekend_id=c.weekend_id, status=action, net_pnl_btc=net,
                             gross_pnl_btc=gross, costs_btc=fees, baseline_pnl_btc=baseline_net,
                             equity_btc=equity, traded=bool(q.sum()), evaluation=True, candidates=len(rows)))
    tables = dict(beta_history=pd.DataFrame(beta), predictions=pd.DataFrame(predictions),
                  trades=pd.DataFrame(legs), decisions=pd.DataFrame(decisions), weekends=pd.DataFrame(weekends),
                  stress=pd.DataFrame(stress), baseline_trades=pd.DataFrame(baselines))
    if tables['beta_history'].empty:
        raise ValueError('Too few matured weekends for walk-forward fitting')
    for name, frame in tables.items():
        frame.to_csv(p / (name + '.csv'), index=False)
    # Final research fit is separate from all earlier out-of-sample predictions.
    final_time = obs.label_available_at.max() + pd.Timedelta(seconds=1)
    final = TradeModel('ridge', cfg.ridge_alpha).fit(obs, final_time, '9999-12-31')
    formula = dict(intercept_btc=final.intercept, beta_raw_units=dict(zip(FEATURES, map(float, final.beta))),
                   feature_order=FEATURES, fitted_samples=final.n, fitted_weekends=len(final.weekends),
                   train_start=final.weekends[0], train_end=final.weekends[-1],
                   label_cutoff=str(final_time), ridge_alpha=cfg.ridge_alpha,
                   scaler_mean=dict(zip(FEATURES, map(float, final.scaler.mean_))),
                   scaler_scale=dict(zip(FEATURES, map(float, final.scaler.scale_))),
                   meaning='Retrospective full-sample final research formula; NOT used to produce preceding OOS backtest.')
    (p / 'final_formula.json').write_text(json.dumps(formula, indent=2), encoding='utf-8')
    final_ols = TradeModel('ols').fit(obs, final_time, '9999-12-31')
    (p / 'final_ols_formula.json').write_text(json.dumps(dict(intercept_btc=final_ols.intercept,
        beta_raw_units=dict(zip(FEATURES, map(float, final_ols.beta))),
        fitted_samples=final_ols.n, fitted_weekends=len(final_ols.weekends)), indent=2), encoding='utf-8')
    (p / 'config.json').write_text(json.dumps(asdict(cfg), indent=2), encoding='utf-8')
    sensitivity = []
    if len(tables['trades']):
        tr = tables['trades']
        for fraction in (0., .05, .10, .20):
            pnl = tr.quantity * (tr.trade_price_btc * (1-fraction) - tr.settlement_btc - tr.entry_fee_btc)
            sensitivity.append(dict(slippage_fraction=fraction, net_pnl_btc=float(pnl.sum()),
                                    portfolio_policy='Base fitted model and fixed recorded quantities'))
    pd.DataFrame(sensitivity, columns=['slippage_fraction', 'net_pnl_btc', 'portfolio_policy']).to_csv(p / 'slippage_sensitivity.csv', index=False)
    events = []
    for r in tables['trades'].itertuples():
        entry = r.quantity * (r.trade_price_btc - r.entry_fee_btc - r.slippage_btc)
        events.append(dict(timestamp=r.decision_time, weekend_id=r.weekend_id, symbol=r.symbol,
                           event='MODELLED_ENTRY_AT_PAST_SELL_PRINT', cashflow_btc=entry))
        events.append(dict(timestamp=r.expiry, weekend_id=r.weekend_id, symbol=r.symbol,
                           event='OFFICIAL_INDEX_DELIVERY', cashflow_btc=-r.quantity*r.settlement_btc))
    ledger = pd.DataFrame(events, columns=['timestamp', 'weekend_id', 'symbol', 'event', 'cashflow_btc'])
    ledger = ledger.sort_values(['timestamp', 'symbol', 'event'])
    ledger['cash_balance_btc'] = cfg.initial_equity_btc + ledger.cashflow_btc.cumsum()
    ledger.to_csv(p / 'event_ledger.csv', index=False)
    if abs(ledger.cashflow_btc.sum() - tables['weekends'].net_pnl_btc.sum()) > 1e-10:
        raise ValueError('Event cash ledger does not reconcile with weekend PnL')
    from src.trade_report import report
    return report(tables, obs, coverage, formula, cfg, p)
