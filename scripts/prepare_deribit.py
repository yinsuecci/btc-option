import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError,URLError
import pandas as pd
from src.core import ET
from src.deribit import public,history_sample

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--start',default='2022-01-01'); parser.add_argument('--end',default='2026-12-31'); parser.add_argument('--output',default='results/deribit_2022_2026'); args=parser.parse_args()
    p=Path(args.output); p.mkdir(parents=True,exist_ok=True)
    now=pd.Timestamp.now(tz='UTC'); audit={'exchange':'DERIBIT','data_mode':'REAL_ACCESS_CHECK','checked_at_utc':str(now),'requested_start':args.start,'requested_end':args.end,'beta':None,'omega':None,'tardis_key_configured':bool(os.environ.get('TARDIS_API_KEY'))}
    instruments=public('get_instruments',currency='BTC',kind='option',expired='false')
    (p/'current_instruments.json').write_text(json.dumps(instruments,indent=2),encoding='utf-8')
    native=[r for r in instruments if r.get('settlement_currency')=='BTC']
    audit['current_btc_options']=len(native)
    audit['sunday_et_expiries']=sorted({str(pd.to_datetime(r['expiration_timestamp'],unit='ms',utc=True)) for r in native if pd.to_datetime(r['expiration_timestamp'],unit='ms',utc=True).tz_convert(ET).weekday()==6})
    if native:
        book=public('get_order_book',instrument_name=native[0]['instrument_name'],depth=1)
        (p/'current_book.json').write_text(json.dumps(book,indent=2),encoding='utf-8')
    deliveries=public('get_delivery_prices',index_name='btc_usd',offset=0,count=1000)
    (p/'delivery_prices.json').write_text(json.dumps(deliveries,indent=2),encoding='utf-8')
    audit['probes']=[]
    for date in [pd.Timestamp(args.start),pd.Timestamp(args.start)+pd.Timedelta(days=1)]:
        record={'date':date.date().isoformat()}
        try:
            sample=history_sample(date)
            (p/f'{date.date()}_chain_schema_sample.csv').write_text(sample,encoding='utf-8')
            record.update(status='SAMPLE_ACCESSIBLE',columns=sample.splitlines()[0].split(','))
        except HTTPError as e:
            record.update(status='ACCESS_DENIED' if e.code in (401,403) else 'HTTP_ERROR',http_status=e.code)
        except (URLError,TimeoutError) as e: record.update(status='NETWORK_ERROR',reason=str(e))
        audit['probes'].append(record)
    coverage=[]
    for saturday in pd.date_range(args.start,args.end,freq='W-SAT',tz=ET):
        end=saturday+pd.DateOffset(days=2)
        status='FUTURE' if saturday.tz_convert('UTC')>now else 'INCOMPLETE_WEEKEND' if end.tz_convert('UTC')>now else 'NOT_DOWNLOADED_FULL_CHAIN_INDEX_VOLUME_AND_SETTLEMENT'
        coverage.append({'weekend_id':saturday.date().isoformat(),'status':status,'beta_available':False,'omega_available':False})
    pd.DataFrame(coverage).to_csv(p/'weekend_coverage.csv',index=False)
    audit['status']='BLOCKED_FULL_HISTORY_REQUIRED'
    audit['warning']='Current snapshots and first-day schema samples are NOT a historical backtest. Native BTC options cannot be passed to the USDT linear engine.'
    (p/'access_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))

if __name__=='__main__': main()
