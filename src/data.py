import json
from pathlib import Path
from urllib.request import urlopen
import numpy as np
import pandas as pd
from src.core import ET,bs

def get_json(url):
    with urlopen(url,timeout=20) as response: return json.load(response)

def exchange_chain():
    raw=get_json('https://eapi.binance.com/eapi/v1/exchangeInfo')
    return pd.DataFrame([dict(symbol=r['symbol'],strike=float(r['strikePrice']),option_type=r['side'],expiry=pd.to_datetime(r['expiryDate'],unit='ms',utc=True),unit=float(r['unit']),min_qty=float(r['minQty'])) for r in raw['optionSymbols'] if r['underlying']=='BTCUSDT'])

def binance_spot_history(start,end):
    """Closed BTCUSDT hourly candles; NOT an options settlement-index archive."""
    start=pd.Timestamp(start); end=pd.Timestamp(end)
    if start.tzinfo is None or end.tzinfo is None: raise ValueError('UTC-aware bounds required')
    cursor=int(start.timestamp()*1000); finish=int(end.timestamp()*1000); records=[]
    while cursor<finish:
        batch=get_json(f'https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&startTime={cursor}&endTime={finish}&limit=1000')
        if not batch: break
        records.extend(dict(timestamp=pd.to_datetime(r[6],unit='ms',utc=True),spot=float(r[4]),volume=float(r[5])) for r in batch if r[6]<=finish)
        next_cursor=int(batch[-1][0])+3600000
        if next_cursor<=cursor: raise ValueError('Non-advancing spot pagination')
        cursor=next_cursor
    return pd.DataFrame(records)

def synthetic(weeks=20,seed=42):
    rng=np.random.default_rng(seed)
    start=pd.Timestamp('2025-01-04',tz=ET).tz_convert('UTC')
    ts=pd.date_range(start-pd.Timedelta(days=8),start+pd.Timedelta(weeks=weeks),freq='h')
    returns=rng.normal(0,0.0025,len(ts)); returns[::263]+=rng.choice([-0.06,0.06],len(returns[::263]))
    prices=60000*np.exp(np.cumsum(returns))
    spot=pd.DataFrame(dict(timestamp=ts,spot=prices,volume=rng.lognormal(8,0.4,len(ts))))
    rows=[]
    for w in range(weeks):
        saturday=(start.tz_convert(ET)+pd.DateOffset(weeks=w)).normalize()
        expiry=(saturday+pd.DateOffset(days=1,hours=16)).tz_convert('UTC')
        anchor=float(spot.loc[spot.timestamp<=saturday.tz_convert('UTC'),'spot'].iloc[-1])
        strikes=np.round(anchor*np.array([0.94,0.98,1.02,1.06])/100)*100
        for hour in range(24):
            t=(saturday+pd.Timedelta(hours=hour)).tz_convert('UTC'); s=float(spot.loc[spot.timestamp==t,'spot'].iloc[0])
            for k in strikes:
                for kind in ['PUT','CALL']:
                    iv=0.65+0.15*abs(np.log(s/k))+0.05*np.sin(w)
                    mark,d,g,v,theta=bs(s,k,(expiry-t).total_seconds()/3600,iv,kind)
                    spread=max(2,mark*0.04)
                    rows.append(dict(timestamp=t,symbol=f'MOCK-{w}-{int(k)}-{kind}',option_type=kind,strike=k,expiry=expiry,bid=max(0.01,mark-spread/2),ask=max(0.02,mark+spread/2),bid_size=2.,ask_size=2.,mark_price=mark,mark_iv=iv,bid_iv=iv-0.02,ask_iv=iv+0.02,delta=d,gamma=g,vega=v,theta=theta,volume=100.,open_interest=500.,unit=1.,min_qty=0.001))
    return spot,pd.DataFrame(rows),'SYNTHETIC'

def load(directory=None):
    if directory is None: return synthetic()
    p=Path(directory); manifest=json.loads((p/'manifest.json').read_text())
    if manifest['data_mode']!='REAL': raise ValueError('Imported history must explicitly declare REAL')
    if manifest.get('settlement_currency')=='BTC' or manifest.get('is_inverse'):
        raise ValueError('Native inverse BTC data requires an inverse execution engine; refusing the USDT linear pipeline')
    if not manifest.get('source') or not manifest.get('settlement_index_source'): raise ValueError('Provenance is required')
    spot=pd.read_csv(p/'spot.csv'); chain=pd.read_csv(p/'chain.csv')
    required={'timestamp','symbol','option_type','strike','expiry','bid','ask','bid_size','ask_size','mark_price','mark_iv','bid_iv','ask_iv','delta','gamma','vega','theta','volume','open_interest','unit','min_qty'}
    if not required.issubset(chain.columns) or not {'timestamp','spot','volume'}.issubset(spot.columns): raise ValueError('Incomplete historical schema')
    if not chain.option_type.isin(['CALL','PUT']).all(): raise ValueError('Invalid option type')
    if not np.isfinite(spot[['spot','volume']]).all().all() or (spot.spot<=0).any(): raise ValueError('Invalid underlying data')
    for frame,col in [(spot,'timestamp'),(chain,'timestamp'),(chain,'expiry')]:
        if any(pd.Timestamp(v).tzinfo is None for v in frame[col]): raise ValueError('Raw timestamps must carry UTC offset')
        frame[col]=pd.to_datetime(frame[col],utc=True)
    if spot.timestamp.duplicated().any() or chain[['timestamp','symbol']].duplicated().any(): raise ValueError('Duplicate observations')
    return spot,chain,'REAL'
