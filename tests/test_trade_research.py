import numpy as np
import pandas as pd
import pytest

from src.trade_data import complete_window
from src.trade_research import (FEATURES, TradeConfig, TradeModel, contract,
                                inverse_greeks, market_features, model_value,
                                optimize_native, per_unit_cashflows)


def test_split_pagination_retains_all_tied_prints():
    records = [{'trade_id': str(i), 'timestamp': t} for i, t in enumerate([0, 0, 1, 1, 2, 3])]

    def fetch(a, b):
        selected = [r for r in records if a <= r['timestamp'] <= b]
        return {'result': {'trades': selected[:2], 'has_more': len(selected) > 2}}

    assert complete_window(fetch, 0, 3) == records


def test_impossible_single_ms_page_raises():
    with pytest.raises(ValueError, match='single ms'):
        complete_window(lambda a,b: {'result': {'trades': [], 'has_more': True}}, 3, 3)


def test_contract_expiry_and_inverse_settlement():
    expiry, strike, kind = contract('BTC-2JAN22-48000-C')
    assert expiry == pd.Timestamp('2022-01-02T08:00:00Z')
    assert expiry.tz_convert('America/New_York').weekday() == 6
    cfg = TradeConfig()
    out = per_unit_cashflows(.01, 50000, strike, kind, cfg)
    assert out['settlement_btc'] == pytest.approx(.04)
    assert out['y_btc'] == pytest.approx(.0095 - .04 - .0003)
    assert out['delivery_fee_btc'] == 0


def test_inverse_model_greeks_finite_difference():
    s, k, h, iv, kind = 50000., 51000., 16., .6, 'CALL'
    g = inverse_greeks(s, k, h, iv, kind)
    eps = .0001
    v = model_value(s, k, h, iv, kind)
    up = model_value(s*np.exp(eps), k, h, iv, kind)
    down = model_value(s*np.exp(-eps), k, h, iv, kind)
    assert g['delta_log_model'] == pytest.approx((up-down)/(2*eps), rel=1e-4)
    assert g['gamma_log_model'] == pytest.approx((up-2*v+down)/eps**2, rel=1e-4)


def test_closed_candles_only():
    times = pd.date_range('2022-01-01', periods=45, freq='D', tz='UTC')
    frame = pd.DataFrame(dict(timestamp=times, spot=50000*np.exp(np.arange(45)*.001), volume=np.arange(45)+100))
    t = times[40] + pd.Timedelta(hours=12)
    first, stamp = market_features(frame, t)
    frame.loc[frame.timestamp > t, 'spot'] = 1e12
    second, _ = market_features(frame, t)
    assert first == second
    assert stamp < t
    assert first['momentum_1d'] == pytest.approx(np.exp(.001)-1)


def test_no_future_labels_or_same_weekend_training():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(rng.normal(size=(80,len(FEATURES))), columns=FEATURES)
    frame['weekend_id'] = ['2022-01-01']*40 + ['2022-02-05']*40
    frame['label_available_at'] = pd.to_datetime(['2022-01-02T08:00Z']*40 + ['2022-02-06T08:00Z']*40, utc=True)
    frame['y_btc'] = rng.normal(0,.01,80)
    t = pd.Timestamp('2022-02-05T17:00Z')
    first = TradeModel().fit(frame,t,'2022-02-05')
    frame.loc[frame.weekend_id.eq('2022-02-05'),'y_btc'] = 1e6
    second = TradeModel().fit(frame,t,'2022-02-05')
    assert first.n == 40
    np.testing.assert_allclose(first.beta,second.beta)
    assert first.last_label < t
    np.testing.assert_allclose(first.predict(frame),first.model.predict(first.scaler.transform(frame[FEATURES].to_numpy())))


def test_optimizer_native_constraints_and_lots():
    rows = []
    for kind,k in [('CALL',51000),('PUT',49000)]:
        rows.append(dict(option_type=kind,strike=k,spot=50000,trade_price_btc=.005,
                         trade_iv=.7,days_to_expiry=16/24,trade_amount=1.,
                         **inverse_greeks(50000,k,16,.7,kind)))
    frame = pd.DataFrame(rows)
    cfg = TradeConfig()
    result = optimize_native(frame,np.array([.003,.003]),cfg)
    assert result['status'] == 'ENTER'
    assert np.all(result['weights'] <= 0)
    assert result['quantity'].sum() <= cfg.gross_limit + 1e-8
    assert abs(result['delta_log_btc']) <= cfg.delta_log_limit + 1e-8
    assert abs(result['vega_btc']) <= cfg.vega_btc_limit + 1e-8
    assert result['worst_stress_btc'] <= cfg.stress_loss_limit + 1e-8
    np.testing.assert_allclose(result['quantity']/cfg.lot,np.round(result['quantity']/cfg.lot))


def test_negative_alpha_does_not_enter():
    r = dict(option_type='CALL',strike=51000,spot=50000,trade_price_btc=.005,
             trade_iv=.7,days_to_expiry=16/24,trade_amount=1.,
             **inverse_greeks(50000,51000,16,.7,'CALL'))
    assert optimize_native(pd.DataFrame([r]),np.array([-.01]))['status'] != 'ENTER'
