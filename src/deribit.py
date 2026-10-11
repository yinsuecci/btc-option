"""Public Deribit data and native BTC cash-flow helpers. No trading endpoints."""
import gzip
import io
import os
from urllib.parse import urlencode
from urllib.request import Request,urlopen
from src.data import get_json
from src.core import payoff

def public(method,**params):
    allowed={'get_instruments','get_order_book','get_delivery_prices','get_index_price'}
    if method not in allowed: raise ValueError('Only allowlisted public research endpoints permitted')
    response=get_json('https://www.deribit.com/api/v2/public/'+method+'?'+urlencode(params))
    if 'error' in response: raise ValueError(str(response['error']))
    return response['result']

def settlement_btc(delivery_price,strike,option_type):
    if delivery_price<=0: raise ValueError('Delivery price must be positive')
    return payoff(delivery_price,strike,option_type)/delivery_price

def short_pnl_btc(bid_btc,delivery_price,strike,option_type,entry_fee_btc,delivery_fee_btc,slippage_btc):
    if min(bid_btc,entry_fee_btc,delivery_fee_btc,slippage_btc)<0: raise ValueError('Invalid price/cost')
    return bid_btc-settlement_btc(delivery_price,strike,option_type)-entry_fee_btc-delivery_fee_btc-slippage_btc

def history_request(date):
    stamp=date.strftime('%Y/%m/%d')
    headers={}
    token=os.environ.get('TARDIS_API_KEY')
    if token: headers['Authorization']='Bearer '+token
    return Request(f'https://datasets.tardis.dev/v1/deribit/options_chain/{stamp}/OPTIONS.csv.gz',headers=headers)

def history_sample(date,n=10):
    # Stream only enough compressed bytes for schema validation, not an entire daily file.
    with urlopen(history_request(date),timeout=30) as response:
        with gzip.GzipFile(fileobj=response) as zipped:
            with io.TextIOWrapper(zipped,encoding='utf-8') as stream:
                lines=[stream.readline()]
                for _ in range(10000):
                    line=stream.readline()
                    if not line: break
                    if line.split(',',2)[1].startswith('BTC-'):
                        lines.append(line)
                    if len(lines)>=n+1: break
    return ''.join(lines)
