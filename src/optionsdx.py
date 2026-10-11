"""Native BTC optionsDX ingestion; intentionally separate from the linear engine."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.core import ET

FIELDS = {
    'QUOTE_UNIXTIME': 'timestamp', 'INSTRUMENT_NAME': 'symbol',
    'BASE_CURRENCY': 'base_currency', 'CONTRACT_SIZE': 'unit',
    'UNDERLYING_INDEX': 'underlying_index', 'UNDERLYING_PRICE': 'underlying_price',
    'EXPIRY_UNIX': 'expiry', 'OPTION_RIGHT': 'option_type', 'STRIKE': 'strike',
    'BID_SIZE': 'bid_size', 'BID_PRICE': 'bid', 'ASK_PRICE': 'ask',
    'ASK_SIZE': 'ask_size', 'OPEN_INTEREST': 'open_interest', 'VOLUME': 'volume',
    'DELTA': 'delta', 'GAMMA': 'gamma', 'VEGA': 'vega', 'THETA': 'theta',
    'RHO': 'rho', 'MARK_IV': 'mark_iv',
}


def read_quotes(path):
    frame = pd.read_csv(Path(path), skipinitialspace=True, na_values=[' ', ''])
    frame.columns = [c.strip().strip('[]') for c in frame.columns]
    missing = set(FIELDS) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing optionsDX columns: {sorted(missing)}')
    frame = frame[list(FIELDS)].rename(columns=FIELDS)
    frame['timestamp'] = pd.to_datetime(pd.to_numeric(frame.timestamp), unit='s', utc=True)
    frame['expiry'] = pd.to_datetime(pd.to_numeric(frame.expiry), unit='s', utc=True)
    frame['option_type'] = frame.option_type.str.strip().str.upper()
    frame['base_currency'] = frame.base_currency.str.strip().str.upper()
    for col in set(FIELDS.values()) - {
        'timestamp', 'expiry', 'symbol', 'base_currency', 'underlying_index', 'option_type'
    }:
        frame[col] = pd.to_numeric(frame[col], errors='raise')
    if not frame.base_currency.eq('BTC').all():
        raise ValueError('This research adapter only supports native BTC contracts')
    if not frame.option_type.isin(['CALL', 'PUT']).all():
        raise ValueError('Invalid option right')
    if frame[['timestamp', 'symbol']].duplicated().any():
        raise ValueError('Duplicate quote timestamps/instruments')
    if not np.isfinite(frame[['strike', 'underlying_price', 'unit']]).all().all():
        raise ValueError('Missing contract price or multiplier')
    if (frame[['strike', 'underlying_price', 'unit']] <= 0).any().any():
        raise ValueError('Invalid contract price or multiplier')
    frame['mark_iv'] /= 100
    frame['source_file'] = str(Path(path).resolve())
    return frame.sort_values(['timestamp', 'symbol']).reset_index(drop=True)


def saturday_daily_candidates(frame):
    """Latest supplied daily snapshot on Saturday ET, future Sunday ET expiry.

    Selection uses actual quote timestamps, not the vendor's local date label.
    Never borrows Friday quotes or independently picks a later time per strike.
    Missing daily files/weekends remain missing, not interpolated observations.
    """
    local = frame.timestamp.dt.tz_convert(ET)
    saturday = frame.loc[local.dt.weekday.eq(5)].copy()
    saturday['weekend_id'] = local.loc[saturday.index].dt.date.astype(str)
    last = saturday.groupby('weekend_id').timestamp.transform('max')
    saturday = saturday.loc[saturday.timestamp.eq(last)]
    eligible = (saturday.expiry > saturday.timestamp) & (
        saturday.expiry.dt.tz_convert(ET).dt.weekday.eq(6)
    )
    return saturday.loc[eligible].copy()
