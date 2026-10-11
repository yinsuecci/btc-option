import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from src.center_vol import (MEAN_FEATURES, CenterConfig, causal_mean_residuals,
                            distribution_values, fit_distribution,
                            inverse_expected_payoff, payoff_covariance, run)
from src.trade_research import TradeConfig


def sample_weekends(n=65):
    rng=np.random.default_rng(35)
    dates=pd.date_range('2022-01-01',periods=n,freq='7D',tz='UTC')
    frame=pd.DataFrame(rng.normal(size=(n,len(MEAN_FEATURES))),columns=MEAN_FEATURES)
    frame['weekend_id']=dates.strftime('%Y-%m-%d')
    frame['decision_time']=dates+pd.Timedelta(hours=17)
    frame['label_available_at']=dates+pd.Timedelta(days=1,hours=8)
    frame['log_return']=rng.normal(0,.01,n)
    return frame


@pytest.mark.parametrize('kind',['CALL','PUT'])
def test_inverse_expected_payoff_matches_independent_integration(kind):
    spot,strike,mu,var=50000.,50500.,.002,.0004
    sigma=np.sqrt(var)
    def integrand(z):
        d=spot*np.exp(mu+sigma*z)
        payoff=max(d-strike,0)/d if kind=='CALL' else max(strike-d,0)/d
        return payoff*norm.pdf(z)
    boundary=(np.log(strike/spot)-mu)/sigma
    expected=quad(integrand,-12,boundary,epsabs=1e-12)[0]+quad(integrand,boundary,12,epsabs=1e-12)[0]
    actual=inverse_expected_payoff(spot,[strike],[kind],mu,var)[0]
    assert actual==pytest.approx(expected,abs=1e-10)


def test_zero_variance_native_payoff():
    result=inverse_expected_payoff(50000,[49000,51000],['CALL','PUT'],0,0)
    np.testing.assert_allclose(result,[.02,.02])


def test_price_center_mean_median_and_interval():
    values=distribution_values(50000,.003,.0004)
    assert values['center_mean_usd'] > values['center_median_usd']
    assert values['center_median_usd']==pytest.approx(50000*np.exp(.003))
    assert values['lower_95_usd'] < values['center_median_usd'] < values['upper_95_usd']


def test_prequential_errors_and_two_stage_no_future_labels():
    frame=sample_weekends()
    cfg=CenterConfig()
    residuals=causal_mean_residuals(frame,cfg)
    assert residuals.mean_train_end.lt(residuals.weekend_id).all()
    assert residuals.mean_last_label.lt(residuals.decision_time).all()
    row=frame.iloc[55]
    first=fit_distribution(frame,residuals,row.decision_time,row.weekend_id,cfg)
    altered=frame.copy()
    altered.loc[altered.weekend_id>=row.weekend_id,'log_return']=9999
    errors=causal_mean_residuals(altered,cfg)
    second=fit_distribution(altered,errors,row.decision_time,row.weekend_id,cfg)
    np.testing.assert_allclose(first[0].beta,second[0].beta)
    np.testing.assert_allclose(first[1].beta,second[1].beta)
    assert first[2]==pytest.approx(second[2])
    assert first[3].label_available_at.max()<row.decision_time
    assert first[4].label_available_at.max()<row.decision_time


def test_covariance_psd_and_distribution_affects_each_leg():
    strikes=[49000,51000]
    kinds=['PUT','CALL']
    cov=payoff_covariance(50000,strikes,kinds,0,.0004)
    np.testing.assert_allclose(cov,cov.T,atol=1e-12)
    assert np.linalg.eigvalsh(cov).min()>=-1e-12
    assert cov[0,1]<0
    quiet=inverse_expected_payoff(50000,strikes,kinds,0,.00001)
    volatile=inverse_expected_payoff(50000,strikes,kinds,0,.0004)
    assert (volatile>quiet).all()
    bullish=inverse_expected_payoff(50000,strikes,kinds,.01,.0004)
    assert bullish[0]<volatile[0] and bullish[1]>volatile[1]


def test_parameter_export_raw_units_matches_standardized_model():
    frame=sample_weekends()
    cfg=CenterConfig()
    residuals=causal_mean_residuals(frame,cfg)
    row=frame.iloc[-1]
    mean,vol,c,_,_=fit_distribution(frame,residuals,row.decision_time,row.weekend_id,cfg)
    for model in [mean,vol]:
        np.testing.assert_allclose(model.predict(frame),model.model.predict(model.scaler.transform(frame[model.features].to_numpy())))
    assert c>0


@pytest.mark.parametrize('penalty',[-.01,float('nan')])
def test_invalid_tail_penalty_rejected_before_data_access(penalty):
    with pytest.raises(ValueError,match='Tail risk penalty'):
        run(source='missing-source',execution=TradeConfig(risk_penalty_per_btc=penalty))
