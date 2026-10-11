"""Verify public weekend trade access, never certify quote-chain completeness."""
import argparse
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from src.core import ET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2022-01-01')
    parser.add_argument('--end', default='2024-01-07')
    parser.add_argument('--output', default='results/free_source_search')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(args.start, args.end, freq='W-SAT', tz=ET)
    evidence = []
    for day in days:
        name = day.date().isoformat()
        cached = output / (name + '_trade_sample.json')
        begin = int(day.timestamp() * 1000)
        end = int((day + pd.DateOffset(days=2)).timestamp() * 1000) - 1
        params = dict(currency='BTC', kind='option', start_timestamp=begin,
                      end_timestamp=end, count=1, sorting='asc')
        url = 'https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time?' + urlencode(params)
        record = {'weekend_id': name, 'url': url,
                  'quote_chain_complete': False, 'full_trade_tape_downloaded': False}
        try:
            if cached.exists():
                response = json.loads(cached.read_text())
            else:
                request = Request(url, headers={'User-Agent': 'BTC-options-research'})
                with urlopen(request, timeout=25) as result:
                    response = json.load(result)
                cached.write_text(json.dumps(response, indent=2), encoding='utf-8')
            if 'error' in response:
                record.update(status='API_ERROR', error=response['error'])
            else:
                trades = response.get('result', {}).get('trades', [])
                valid = bool(trades) and all(
                    begin <= row['timestamp'] <= end and
                    row['instrument_name'].startswith('BTC-') and
                    row['instrument_name'].endswith(('-C', '-P'))
                    for row in trades
                )
                record.update(status='REAL_TRADE_SAMPLE_ACCESSIBLE' if valid else 'NO_VALID_SAMPLE',
                              sample_rows=len(trades),
                              fields=sorted(trades[0]) if trades else [],
                              has_more=response.get('result', {}).get('has_more'))
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError) as error:
            record.update(status='ERROR', error=str(error))
        evidence.append(record)
        summary = {
            'source': 'DERIBIT_PUBLIC_HISTORY', 'authentication_used': False,
            'payment_required_for_probed_endpoint': False,
            'requested_first_saturday': str(days[0].date()),
            'requested_last_saturday': str(days[-1].date()),
            'requested_weekends': len(days), 'checked_weekends': len(evidence),
            'accessible_trade_sample_weekends': sum(r['status'] == 'REAL_TRADE_SAMPLE_ACCESSIBLE' for r in evidence),
            'full_quote_chain_verified': False, 'beta': None, 'omega': None,
            'warning': 'One trade sample per weekend proves access only, not complete trades, option chains, Sunday-expiry eligibility, bids/asks, Greeks or executable PnL.',
            'weekends': evidence,
        }
        (output / 'weekend_access.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        if len(evidence) % 10 == 0 or len(evidence) == len(days):
            print(json.dumps({key: val for key, val in summary.items() if key != 'weekends'}), flush=True)
        time.sleep(0.25)


if __name__ == '__main__':
    main()
