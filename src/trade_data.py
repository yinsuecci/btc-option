"""Free public trade acquisition, deterministic split pagination and provenance."""
import gzip
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from src.core import ET


def request_json(base, params, attempts=4):
    url = base + '?' + urlencode(params)
    for attempt in range(attempts):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'BTC-Weekend-Research/1.0'}), timeout=40) as response:
                result = json.load(response)
            if 'error' in result:
                raise ValueError(f'Public API error: {result["error"]}')
            return result
        except (HTTPError, URLError, TimeoutError) as error:
            if isinstance(error, HTTPError) and error.code in (401, 403):
                raise
            if attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


def complete_window(fetch, start_ms, end_ms):
    """Inclusive integer-ms splitting cannot drop ties at page boundaries."""
    response = fetch(start_ms, end_ms)
    rows = response['result']['trades']
    if not response['result'].get('has_more', False):
        if any(not start_ms <= r['timestamp'] <= end_ms for r in rows):
            raise ValueError('Trade outside requested bounds')
        return rows
    if start_ms == end_ms:
        raise ValueError('More than one page of trades within a single ms; refusing incomplete window')
    mid = (start_ms + end_ms) // 2
    return complete_window(fetch, start_ms, mid) + complete_window(fetch, mid + 1, end_ms)


def download(directory, start='2022-01-01', end='2026-12-31', decision_hour=12, lookback_minutes=60,
             decision_offset_utc_hours=None):
    root = Path(directory)
    raw = root / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz='UTC')
    audit = []
    for day in pd.date_range(start, end, freq='W-SAT', tz=ET):
        t = (day + pd.DateOffset(hours=decision_hour)).tz_convert('UTC')
        if decision_offset_utc_hours is not None:
            t = pd.Timestamp(day.date(),tz='UTC')+pd.Timedelta(hours=decision_offset_utc_hours)
        expiry = pd.Timestamp(day.date() + pd.Timedelta(days=1), tz='UTC') + pd.Timedelta(hours=8)
        wid = str(day.date())
        row = dict(weekend_id=wid, decision_time=str(t), expiry=str(expiry),
                   window_complete=False, trades=0, elapsed_seconds=0.)
        if expiry > now:
            row['status'] = 'FUTURE_OR_UNSETTLED'
            audit.append(row)
            continue
        path = raw / (wid + '_trades.json.gz')
        begin = int((t - pd.Timedelta(minutes=lookback_minutes)).timestamp() * 1000)
        finish = int(t.timestamp() * 1000)
        tick = time.monotonic()
        try:
            if path.exists():
                with gzip.open(path, 'rt', encoding='utf-8') as stream:
                    record = json.load(stream)
                if record['start_ms'] != begin or record['end_ms'] != finish or not record['complete']:
                    raise ValueError('Cached download has incompatible window')
            else:
                calls = 0

                def fetch(lo, hi):
                    nonlocal calls
                    calls += 1
                    data = request_json('https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time',
                                        dict(currency='BTC', kind='option', start_timestamp=lo,
                                             end_timestamp=hi, count=1000, sorting='asc'))
                    time.sleep(.15)
                    return data

                trades = complete_window(fetch, begin, finish)
                unique = {r['trade_id']: r for r in trades}
                record = dict(weekend_id=wid, start_ms=begin, end_ms=finish,
                              complete=True, request_count=calls,
                              fetched_at=datetime.now(timezone.utc).isoformat(),
                              trades=sorted(unique.values(), key=lambda r: (r['timestamp'], r['trade_id'])))
                with gzip.open(path, 'wt', encoding='utf-8') as stream:
                    json.dump(record, stream)
            row.update(status='DOWNLOADED_TRADE_WINDOW', window_complete=True,
                       trades=len(record['trades']), request_count=record['request_count'])
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError) as error:
            row.update(status='DOWNLOAD_ERROR', reason=str(error))
        row['elapsed_seconds'] = time.monotonic() - tick
        audit.append(row)
        pd.DataFrame(audit).to_csv(root / 'download_coverage.csv', index=False)
        if len(audit) % 10 == 0:
            print(f'Downloaded/audited {len(audit)} weekends; latest {wid}: {row["status"]}, {row["trades"]} trades', flush=True)
    pd.DataFrame(audit).to_csv(root / 'download_coverage.csv', index=False)
    if any(r['status'] == 'DOWNLOAD_ERROR' for r in audit):
        raise ValueError('Some historical windows failed; resume before fitting')

    deliveries_path = root / 'delivery_prices.json'
    deliveries = json.loads(deliveries_path.read_text()) if deliveries_path.exists() else []
    if not deliveries or min(r['date'] for r in deliveries) > start:
        offset = len(deliveries)
        while True:
            response = request_json('https://www.deribit.com/api/v2/public/get_delivery_prices',
                                    dict(index_name='btc_usd', offset=offset, count=100))
            chunk = response['result']['data']
            deliveries.extend(chunk)
            deliveries_path.write_text(json.dumps(deliveries, indent=2), encoding='utf-8')
            if not chunk or min(r['date'] for r in chunk) <= start:
                break
            offset += len(chunk)
    spot_path = root / 'spot_daily.csv'
    if not spot_path.exists():
        cursor = int(pd.Timestamp(start, tz='UTC').timestamp() * 1000) - 45 * 86400000
        finish = min(int(now.timestamp() * 1000), int(pd.Timestamp(end, tz='UTC').timestamp() * 1000) + 86400000)
        candles = []
        while cursor < finish:
            batch = request_json('https://api.binance.com/api/v3/klines',
                                 dict(symbol='BTCUSDT', interval='1d', startTime=cursor, endTime=finish, limit=1000))
            if not batch:
                break
            candles.extend(dict(timestamp=pd.to_datetime(r[6], unit='ms', utc=True),
                                spot=float(r[4]), volume=float(r[5])) for r in batch if r[6] < finish)
            next_cursor = int(batch[-1][0]) + 86400000
            if next_cursor <= cursor:
                raise ValueError('Non-advancing candle pagination')
            cursor = next_cursor
        pd.DataFrame(candles).to_csv(spot_path, index=False)
    return pd.DataFrame(audit)
