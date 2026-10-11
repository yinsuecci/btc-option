"""Causal price-center / forecast-error variance models and inverse option allocation."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.special import ndtr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.trade_research import TradeConfig, build_observations, optimize_native

MEAN_FEATURES = ['momentum_1d', 'momentum_3d', 'momentum_7d', 'rv_7d', 'rv_30d',
                 'spot_volume_z30', 'jump_z', 'median_trade_iv', 'days_to_expiry']
VOL_FEATURES = ['rv_7d', 'rv_30d', 'spot_volume_z30', 'jump_z', 'median_trade_iv', 'days_to_expiry']
DEFAULT_TAIL_PENALTY = .01
DEFAULT_SOURCE = 'data/entry_timing/fri_16utc'


@dataclass
class CenterConfig:
    mean_min_train: int = 13
    evaluation_min_train: int = 52
    variance_min_residuals: int = 26
    ridge_alpha: float = 10.
    residual_square_floor: float = 1e-10
    sigma_floor: float = .001
    sigma_cap: float = .20
    covariance_penalty_per_btc: float = 1.
    quadrature_nodes: int = 128


class FactorFit:
    def __init__(self, features, alpha=10.):
        self.features = features
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def fit(self, frame, target):
        self.model.fit(self.scaler.fit_transform(frame[self.features].to_numpy(float)), np.asarray(target))
        self.beta = self.model.coef_ / self.scaler.scale_
        self.intercept = float(self.model.intercept_ - self.scaler.mean_ @ self.beta)
        return self

    def predict(self, frame):
        return self.intercept + frame[self.features].to_numpy(float) @ self.beta

    def parameters(self):
        return dict(intercept=self.intercept, coefficients=dict(zip(self.features, map(float, self.beta))),
                    scaler_mean=dict(zip(self.features, map(float, self.scaler.mean_))),
                    scaler_scale=dict(zip(self.features, map(float, self.scaler.scale_))))


def weekend_dataset(obs):
    records = []
    for wid, group in obs.groupby('weekend_id', sort=True):
        for col in ['spot', 'delivery_index', 'decision_time', 'label_available_at', 'days_to_expiry']:
            if group[col].nunique() != 1:
                raise ValueError(f'Multiple {col} in weekend {wid}')
        r = group.iloc[0]
        records.append(dict(weekend_id=wid, decision_time=r.decision_time,
                            label_available_at=r.label_available_at, spot=r.spot,
                            delivery_index=r.delivery_index,
                            log_return=float(np.log(r.delivery_index/r.spot)),
                            median_trade_iv=float(group.trade_iv.median()),
                            **{f: float(r[f]) for f in MEAN_FEATURES if f != 'median_trade_iv'}))
    return pd.DataFrame(records)


def causal_mean_residuals(frame, cfg):
    records = []
    for i, r in frame.iterrows():
        train = frame.loc[(frame.weekend_id < r.weekend_id) & (frame.label_available_at < r.decision_time)]
        if len(train) < cfg.mean_min_train:
            continue
        model = FactorFit(MEAN_FEATURES, cfg.ridge_alpha).fit(train, train.log_return)
        mu = float(model.predict(frame.loc[[i]])[0])
        records.append(dict(r.to_dict(), mean_prediction=mu, residual=r.log_return-mu,
                            mean_train_end=train.weekend_id.iloc[-1],
                            mean_last_label=train.label_available_at.max()))
    return pd.DataFrame(records)


def fit_distribution(frame, residuals, decision_time, weekend_id, cfg):
    train = frame.loc[(frame.weekend_id < weekend_id) & (frame.label_available_at < decision_time)]
    errors = residuals.loc[(residuals.weekend_id < weekend_id) & (residuals.label_available_at < decision_time)]
    if len(train) < cfg.evaluation_min_train or len(errors) < cfg.variance_min_residuals:
        raise ValueError('Insufficient matured mean / variance training rows')
    mean = FactorFit(MEAN_FEATURES, cfg.ridge_alpha).fit(train, train.log_return)
    squared = np.maximum(errors.residual.to_numpy()**2, cfg.residual_square_floor)
    vol = FactorFit(VOL_FEATURES, cfg.ridge_alpha).fit(errors, np.log(squared))
    # Retransformation corrects log-square bias using past matured errors only.
    calibration = float(np.mean(squared / np.exp(np.clip(vol.predict(errors), -30, 5))))
    return mean, vol, calibration, train, errors


def distribution_values(spot, mu, variance):
    sigma = np.sqrt(variance)
    return dict(mu_log_return=mu, variance_log_return=variance, sigma_weekend=sigma,
                center_mean_usd=spot*np.exp(mu+.5*variance),
                center_median_usd=spot*np.exp(mu),
                lower_95_usd=spot*np.exp(mu-1.959963984540054*sigma),
                upper_95_usd=spot*np.exp(mu+1.959963984540054*sigma))


def inverse_expected_payoff(spot, strikes, kinds, mu, variance):
    """Physical-distribution E[intrinsic(D)/D], not a risk-neutral BS price."""
    strikes = np.asarray(strikes, dtype=float)
    a = np.log(strikes/spot)
    if variance <= 0:
        delivery = spot*np.exp(mu)
        return np.where(np.asarray(kinds) == 'CALL', np.maximum(delivery-strikes, 0)/delivery,
                        np.maximum(strikes-delivery, 0)/delivery)
    sigma = np.sqrt(variance)
    multiplier = strikes/spot*np.exp(-mu+.5*variance)
    call = ndtr((mu-a)/sigma) - multiplier*ndtr((mu-variance-a)/sigma)
    put = multiplier*ndtr((a-mu+variance)/sigma) - ndtr((a-mu)/sigma)
    return np.maximum(np.where(np.asarray(kinds) == 'CALL', call, put), 0.)


def payoff_covariance(spot, strikes, kinds, mu, variance, nodes=128):
    x, weights = hermgauss(nodes)
    weights = weights / np.sqrt(np.pi)
    delivery = spot*np.exp(mu+np.sqrt(2*variance)*x)
    strikes = np.asarray(strikes)
    payoffs = np.where(np.asarray(kinds)[None,:] == 'CALL',
                       np.maximum(delivery[:,None]-strikes, 0)/delivery[:,None],
                       np.maximum(strikes-delivery[:,None], 0)/delivery[:,None])
    centered = payoffs - weights @ payoffs
    return (centered.T * weights) @ centered


def run(source=DEFAULT_SOURCE, output='results/center_vol', cfg=CenterConfig(), execution=None):
    execution = execution or TradeConfig(risk_penalty_per_btc=DEFAULT_TAIL_PENALTY)
    if not np.isfinite(execution.risk_penalty_per_btc) or execution.risk_penalty_per_btc < 0:
        raise ValueError('Tail risk penalty must be finite and nonnegative')
    p = Path(output)
    p.mkdir(parents=True, exist_ok=True)
    obs, coverage, rejected = build_observations(source, execution)
    if obs.empty or obs.y_btc.isna().any():
        raise ValueError('No observations or missing official settlement')
    frame = weekend_dataset(obs)
    times=pd.to_datetime(frame.decision_time,utc=True)
    if not (times.dt.weekday.eq(4).all() and times.dt.hour.eq(16).all()
            and times.dt.minute.eq(0).all() and np.allclose(frame.days_to_expiry*24,40)):
        raise ValueError('Selected strategy requires Friday 16:00 UTC entry and 40-hour expiry horizon')
    residuals = causal_mean_residuals(frame, cfg)
    frame.to_csv(p/'price_training.csv', index=False)
    residuals.to_csv(p/'causal_mean_residuals.csv', index=False)
    obs.to_csv(p/'observations.csv', index=False)
    coverage.to_csv(p/'coverage.csv', index=False)
    rejected.to_csv(p/'rejected.csv', index=False)
    forecasts, predictions, legs, weekends, beta, events, sensitivity = [], [], [], [], [], [], []
    equity = execution.initial_equity_btc
    for _, r in frame.iterrows():
        train = frame.loc[(frame.weekend_id < r.weekend_id) & (frame.label_available_at < r.decision_time)]
        errors = residuals.loc[(residuals.weekend_id < r.weekend_id) & (residuals.label_available_at < r.decision_time)]
        if len(train) < cfg.evaluation_min_train or len(errors) < cfg.variance_min_residuals:
            weekends.append(dict(weekend_id=r.weekend_id, evaluation=False, status='WARMUP',
                                 traded=False, net_pnl_btc=0., equity_btc=equity))
            continue
        mean, vol, calibration, train, errors = fit_distribution(frame, residuals, r.decision_time, r.weekend_id, cfg)
        current = frame.loc[frame.weekend_id.eq(r.weekend_id)]
        mu = float(mean.predict(current)[0])
        raw_var = float(calibration*np.exp(np.clip(vol.predict(current)[0], -30, 5)))
        variance = float(np.clip(raw_var, cfg.sigma_floor**2, cfg.sigma_cap**2))
        distribution = distribution_values(r.spot, mu, variance)
        forecast = dict(r.to_dict(), **distribution, raw_variance=raw_var, variance_clipped=raw_var != variance,
                        train_weekends=len(train), train_end=train.weekend_id.iloc[-1],
                        last_train_label=train.label_available_at.max(), variance_rows=len(errors),
                        variance_train_end=errors.weekend_id.iloc[-1],
                        variance_last_label=errors.label_available_at.max(), variance_calibration=calibration,
                        naive_mu=float(train.log_return.mean()), naive_variance=float(train.log_return.var(ddof=1)))
        rows = obs.loc[obs.weekend_id.eq(r.weekend_id)].copy()
        payoff = inverse_expected_payoff(r.spot, rows.strike, rows.option_type, mu, variance)
        covariance = payoff_covariance(r.spot, rows.strike, rows.option_type, mu, variance, cfg.quadrature_nodes)
        prices = rows.trade_price_btc.to_numpy()
        alpha = prices*(1-execution.slippage_fraction)-np.minimum(execution.fee_btc, execution.fee_premium_cap*prices)-payoff
        result = optimize_native(rows, alpha, execution, equity, covariance, cfg.covariance_penalty_per_btc)
        action = result['status'] if result else 'NO_EQUITY'
        q = result['quantity'] if action == 'ENTER' else np.zeros(len(rows))
        forecast.update(action=action, quantity=float(q.sum()), expected_net_btc=float(alpha @ q),
                        payoff_variance_btc2=float(q @ covariance @ q),
                        margin_proxy_btc=result.get('margin_proxy_btc', 0.) if result else 0.,
                        worst_stress_btc=result.get('worst_stress_btc', 0.) if result else 0.,
                        delta_log_btc=result.get('delta_log_btc', 0.) if result else 0.,
                        vega_btc=result.get('vega_btc', 0.) if result else 0., pre_entry_equity_btc=equity)
        forecasts.append(forecast)
        for name, model in [('center_beta', mean), ('vol_gamma', vol)]:
            for f, value in [('intercept', model.intercept)] + list(zip(model.features, model.beta)):
                beta.append(dict(weekend_id=r.weekend_id, model=name, factor=f, coefficient=float(value)))
        net = 0.
        for i, (_, leg) in enumerate(rows.iterrows()):
            rec = dict(leg.to_dict(), **distribution, expected_payoff_btc=float(payoff[i]),
                       predicted_net_per_unit_btc=float(alpha[i]), quantity=float(q[i]), omega=float(-q[i]),
                       relative_abs_weight=float(q[i]/q.sum()) if q.sum() else 0., net_pnl_btc=float(q[i]*leg.y_btc))
            predictions.append(rec)
            if q[i] <= 0:
                continue
            legs.append(rec)
            net += rec['net_pnl_btc']
            events.extend([dict(weekend_id=r.weekend_id, symbol=leg.symbol, timestamp=leg.decision_time,
                                event='MODELLED_ENTRY', cashflow_btc=float(q[i]*(leg.trade_price_btc-leg.entry_fee_btc-leg.slippage_btc))),
                           dict(weekend_id=r.weekend_id, symbol=leg.symbol, timestamp=leg.expiry,
                                event='OFFICIAL_DELIVERY', cashflow_btc=float(-q[i]*leg.settlement_btc))])
        equity += net
        weekends.append(dict(weekend_id=r.weekend_id, evaluation=True, status=action, traded=bool(q.sum()),
                             net_pnl_btc=net, equity_btc=equity))
    if not forecasts:
        raise ValueError('No OOS forecasts')
    tables = dict(forecasts=pd.DataFrame(forecasts), predictions=pd.DataFrame(predictions),
                  trades=pd.DataFrame(legs), weekends=pd.DataFrame(weekends), coefficients=pd.DataFrame(beta),
                  event_ledger=pd.DataFrame(events))
    for name, table in tables.items():
        table.to_csv(p/(name+'.csv'), index=False)
    final_time = frame.label_available_at.max()+pd.Timedelta(seconds=1)
    mean, vol, calibration, train, errors = fit_distribution(frame, residuals, final_time, '9999-12-31', cfg)
    formula = dict(source_directory=str(source),entry_time_utc='Friday 16:00',
                   entry_time_beijing='Saturday 00:00',holding_hours=40,
                   center_beta=mean.parameters(), vol_gamma=vol.parameters(), variance_calibration=calibration,
                   train_start=train.weekend_id.iloc[0], train_end=train.weekend_id.iloc[-1],
                   price_rows=len(train), variance_rows=len(errors), label_cutoff=str(final_time),
                   retrospective=True, config=asdict(cfg), execution=asdict(execution))
    (p/'final_formula.json').write_text(json.dumps(formula, indent=2), encoding='utf-8')
    w = tables['weekends'].loc[tables['weekends'].evaluation]
    f = tables['forecasts']
    realized = f.log_return.to_numpy()
    error = realized-f.mu_log_return.to_numpy()
    var = f.variance_log_return.to_numpy()
    rmse = float(np.sqrt(np.mean(error**2)))
    naive_error = realized-f.naive_mu.to_numpy()
    pnl = w.net_pnl_btc.to_numpy()
    peaks = np.maximum.accumulate(np.r_[execution.initial_equity_btc, w.equity_btc])[1:]
    metrics = dict(model='PRICE_CENTER_THEN_CONDITIONAL_ERROR_VARIANCE_THEN_ALLOCATION',
                   source_weekends=len(frame), observations=len(obs), warmup_weekends=int((~tables['weekends'].evaluation).sum()),
                   oos_weekends=len(w), oos_start=w.weekend_id.iloc[0], oos_end=w.weekend_id.iloc[-1],
                   traded_weekends=int(w.traded.sum()), net_pnl_btc=float(pnl.sum()), final_equity_btc=float(equity),
                   return_pct=float((equity/execution.initial_equity_btc-1)*100),
                   max_drawdown_pct=float(np.min((w.equity_btc.to_numpy()-peaks)/peaks)*100),
                   median_weekend_pnl_btc=float(np.median(pnl)), mean_weekend_pnl_btc=float(np.mean(pnl)),
                   log_return_rmse=rmse, zero_drift_rmse=float(np.sqrt(np.mean(realized**2))),
                   past_mean_rmse=float(np.sqrt(np.mean(naive_error**2))),
                   r2_vs_past_mean=float(1-np.sum(error**2)/np.sum(naive_error**2)),
                   interval_95_coverage=float(np.mean((f.delivery_index >= f.lower_95_usd)&(f.delivery_index <= f.upper_95_usd))),
                   gaussian_nll=float(np.mean(.5*(np.log(2*np.pi*var)+error**2/var))),
                   naive_gaussian_nll=float(np.mean(.5*(np.log(2*np.pi*f.naive_variance)+naive_error**2/f.naive_variance))),
                   variance_clipped_weekends=int(f.variance_clipped.sum()),
                   optimizer_status_counts=w.status.value_counts().to_dict())
    rng = np.random.default_rng(20261011)
    block = min(8, len(pnl))
    starts = rng.integers(0, len(pnl)-block+1, size=(5000, int(np.ceil(len(pnl)/block))))
    indexes = (starts[:,:,None]+np.arange(block)).reshape(5000,-1)[:,:len(pnl)]
    metrics['fixed_strategy_8week_block_bootstrap_pnl_95ci_btc'] = list(map(float,np.quantile(pnl[indexes].sum(axis=1),[.025,.975])))
    tr = tables['trades']
    for slip in [0., .05, .10, .20]:
        value = float((tr.quantity*(tr.trade_price_btc*(1-slip)-tr.settlement_btc-tr.entry_fee_btc)).sum()) if len(tr) else 0.
        sensitivity.append(dict(slippage=slip, net_pnl_btc=value))
    tables['slippage'] = pd.DataFrame(sensitivity)
    tables['slippage'].to_csv(p/'slippage.csv', index=False)
    from src.center_vol_metrics import export_views
    export_views(tables, obs, metrics, p, execution.initial_equity_btc)
    (p/'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    from src.center_vol_report import render
    render(tables, formula, metrics, p)
    return metrics
