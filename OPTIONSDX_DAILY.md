# optionsDX Daily Research

Superseded by the user's source-change request. The selected source is now
Tardis/Deribit as recorded in RESEARCH_SOURCE.json. Keep this adapter and scope
note only for reproducibility; optionsDX acquisition is no longer active.

User-approved scope: optionsDX End of Day files, one Saturday snapshot per
weekend rather than hourly timing. The original hourly strategy is not equivalent.

## Acquisition

The user handles account login. Obtain legitimate account downloads for all
available months from January 2022 onward, initially through September 2024
as previously listed by the product. Check the current catalogue after login;
do not promise 2025/2026 coverage before verifying actual files.
No paid intraday data, signup, or binding checkout terms are pre-authorized.

## Time And Native Units

Use QUOTE_UNIXTIME and EXPIRY_UNIX in UTC; classify Saturday entry and Sunday
expiry in America/New_York, including DST. Verify the vendor EOD sampling
convention from the full downloaded files. Select one common latest supplied
Saturday snapshot, not an arbitrary fixed midnight time or separate best time
per strike. Keep missing weekends and unavailable Sunday-expiry contracts in
the coverage audit rather than filling them.

Premiums and settlement PnL remain native BTC. Expiry payoff is USD intrinsic
divided by the actual Deribit delivery index, multiplied by contract size.
UNDERLYING_PRICE is a contract underlying reference, not automatically the
official delivery index. A daily close must never replace that delivery index.

## Factors And Parameter Computation

The sample includes bid/ask, size, open interest, volume, mark IV and Greeks.
MARK_IV is supplied in percent and normalized to decimal. Missing bid IV,
ask IV and mark price are not synthesized. Drop unsupported factors from the
daily model and rename market-return/realized-volatility windows to daily
windows; do not label daily features as 6-hour factors.

OLS/Ridge fitting must use only previous weekends with already matured labels.
Keep beta regression coefficients separate from signed option portfolio omega.
Retain realistic fees, short-only size limits, Greeks and risk constraints in
consistent BTC units. Do not reuse the USDT linear engine on these files.

## Current Status

`src/optionsdx.py` implements raw-file normalization and timestamp-based daily
Saturday/Sunday candidate selection. Tests cover native units, percent IV,
Friday/Sunday exclusion and common snapshot timing. Full monthly files have
not yet been imported; two-year completeness, real beta/omega, daily inverse
optimization and a daily backtest are not yet verified or computed.
