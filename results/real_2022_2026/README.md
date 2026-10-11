# REAL 2022-2026 request: data audit, not a backtest

No real beta or omega was estimated. Existing results/report.html and beta_history.csv
remain SYNTHETIC and must not be used as answers to this request.

As checked on 2026-10-11 Asia/Shanghai, the ET calendar contains 261 Saturdays in
2022-2026: 249 fully completed weekends (2022-01-01 through 2026-10-03), one ongoing
weekend (2026-10-10), and 11 future weekends. weekend_coverage.csv explicitly records
missing data, not NO TRADE or zero returns. Monday 00:00 ET defines full-weekend completion.

Direct enumeration of Binance's public S3 archive found:
- BVOLIndex: 1,182 BTC files, 2023-06-20 through 2026-10-09. Its schema is a volatility
  index, not per-contract execution quotes or supervised option-PnL labels.
- EOHSummary: 147 BTC files, 2023-05-18 through 2023-10-23. It includes bid/ask, sizes,
  IV and Greeks, but does not cover the requested period. The inspected sample contains
  missing best_buy_iv. Hour timestamp semantics, expiry metadata and exact settlement
  index provenance still require validation before this partial dataset can be traded.

Actual downloaded schema samples and archive evidence are in this directory.
data_availability.json intentionally records beta=null and omega=null, not fabricated
values. A source date range does not guarantee daily completeness.

Tardis documents Binance European Options data since 2023-06-15; free samples cover
the first day of a month, not all weekends. Therefore this provider alone cannot fulfill
the full 2022-2026 requirement, even with access to its later-period archive.
Source: https://docs.tardis.dev/historical-data-details/binance-european-options
Archive: https://data.binance.vision/?prefix=data/option/daily/

Required to calculate the requested quantities: historical point-in-time option
constituents/expiry metadata, executable bid/ask and sizes, all model features and
exchange settlement index/records. Full CSV schema and REAL import interface are
documented in the project's README. No vendor purchase or account authorization
was performed. Spot candles or BVOL alone cannot identify per-option net-PnL beta
and constrained portfolio omega without changing the model.

The correct eventual output is weekly walk-forward beta and hourly candidate omega,
with at most one executed portfolio each Saturday; not a single hindsight-fitted
coefficient/weight vector across all years. Initial weekends remain training warmup.

Reproduce the read-only availability check with:
`./run.ps1 -m scripts.audit_real_history`
