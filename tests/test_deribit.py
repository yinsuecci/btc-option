import pytest
from src.deribit import settlement_btc,short_pnl_btc,history_request,public
import pandas as pd

def test_inverse_settlement():
    assert settlement_btc(100000,90000,'CALL')==pytest.approx(.1)
    assert settlement_btc(50000,60000,'PUT')==pytest.approx(.2)
    assert settlement_btc(50000,60000,'CALL')==0
    assert short_pnl_btc(.25,50000,60000,'PUT',.001,.0005,.0001)==pytest.approx(.0484)
    with pytest.raises(ValueError): settlement_btc(0,60000,'PUT')

def test_request_secret_not_in_url(monkeypatch):
    monkeypatch.setenv('TARDIS_API_KEY','test-secret')
    request=history_request(pd.Timestamp('2022-01-02'))
    assert 'test-secret' not in request.full_url
    assert request.get_header('Authorization')=='Bearer test-secret'

def test_no_trade_endpoint():
    with pytest.raises(ValueError): public('buy',instrument_name='BTC')
