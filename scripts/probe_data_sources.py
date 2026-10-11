"""Record read-only public access evidence for candidate historical datasets."""
import html
import json
import re
from pathlib import Path
from urllib.request import urlopen,Request
from urllib.error import HTTPError,URLError

TARGETS={
    'optionsdx_product':'https://www.optionsdx.com/product/btc-option-chains/',
    'optionsdx_sample':'https://www.optionsdx.com/wp-content/uploads/2022/01/btc_sample.csv',
    'laevitas_history':'https://apiv2.laevitas.ch/api/v1/options/ticker-history?exchange=deribit&currency=BTC&start=2022-01-01&end=2022-01-02',
    'supagamma_catalog':'https://api.supagamma.com/v1/series',
    'supagamma_deribit_day':'https://api.supagamma.com/v1/download/options?series_id=deribit:btc-options&start=2022-01-01&end=2022-01-02&format=parquet',
    'coinapi_history':'https://rest.coinapi.io/v1/quotes/DERIBIT_OPT_BTC_USD_220102_45000_P/history?time_start=2022-01-01T00:00:00&limit=1',
    'coindesk_orderbook':'https://data-api.coindesk.com/options/v1/historical/orderbook/l2/snapshots/minute?market=deribit&instrument=BTC-2JAN22-45000-P&limit=1&to_ts=1641081600',
    'kaggle_catalog':'https://www.kaggle.com/api/v1/datasets/list?search=deribit',
    'volar_history':'https://api.volardata.com/v1/chains/BTC?at=2022-01-01T12%3A00%3A00Z',
    'deribit_old_trades':'https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time?currency=BTC&kind=option&start_timestamp=1640995200000&end_timestamp=1641081600000&count=2',
}

def main():
    p=Path('results/source_search'); p.mkdir(parents=True,exist_ok=True); evidence=[]
    for name,url in TARGETS.items():
        record={'source':name,'url':url}
        try:
            with urlopen(Request(url,headers={'User-Agent':'BTCOptionsResearch/0.1'}),timeout=30) as response:
                body=response.read(2000000); record.update(http_status=response.status,content_type=response.headers.get('Content-Type'))
            (p/(name+'.txt')).write_bytes(body)
            if name=='optionsdx_product':
                page=body.decode('utf-8'); match=re.search(r'data-product_variations=[\"\x27](.*?)[\"\x27]',page,re.S)
                if match:
                    variants=json.loads(html.unescape(match.group(1)))
                    record['variants']=[{'attributes':v['attributes'],'price':v.get('display_price'),'purchasable':v.get('is_purchasable')} for v in variants] if isinstance(variants,list) else variants
            elif name=='deribit_old_trades': record['response']=json.loads(body)
        except HTTPError as error:
            status={401:'AUTH_REQUIRED',402:'SUBSCRIPTION_REQUIRED',403:'ACCESS_FORBIDDEN',404:'NOT_FOUND'}.get(error.code,'HTTP_ERROR')
            record.update(http_status=error.code,status=status)
        except (URLError,TimeoutError,ValueError) as error: record.update(status='ERROR',reason=str(error))
        evidence.append(record)
    (p/'access_probes.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(evidence,indent=2))

if __name__=='__main__': main()
