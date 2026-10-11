from pathlib import Path

import pandas as pd

from src.optionsdx import read_quotes, saturday_daily_candidates


def test_vendor_sample_native_units(tmp_path):
    # The prior network audit intentionally capped the public sample at 2 MB.
    lines = Path('results/source_search/optionsdx_sample.txt').read_text().splitlines()
    sample = tmp_path / 'complete_rows.csv'
    sample.write_text('\n'.join(lines[:9]) + '\n')
    frame = read_quotes(sample)
    assert frame.base_currency.eq('BTC').all()
    assert frame.timestamp.iloc[0] == pd.Timestamp('2021-06-01T04:00:46Z')
    assert frame.mark_iv.iloc[0] > 1
    assert 'bid_iv' not in frame and 'mark_price' not in frame
    assert frame.bid.max() < 10


def test_saturday_snapshot_is_causal_and_common():
    frame = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-06T04:00Z', '2024-01-07T03:00Z',
                                     '2024-01-07T03:00Z', '2024-01-07T05:30Z'], utc=True),
        'expiry': pd.to_datetime(['2024-01-07T08:00Z'] * 4, utc=True),
        'symbol': ['friday', 'call', 'put', 'sunday'],
    })
    rows = saturday_daily_candidates(frame)
    assert rows.symbol.tolist() == ['call', 'put']
    assert rows.weekend_id.tolist() == ['2024-01-06'] * 2


def test_no_sunday_expiry_no_candidates():
    frame = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-06T23:00Z'], utc=True),
        'expiry': pd.to_datetime(['2024-01-12T08:00Z'], utc=True),
        'symbol': ['weekly_friday'],
    })
    assert saturday_daily_candidates(frame).empty
