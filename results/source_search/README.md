# Multi-source search: not complete, no fabricated coefficients

Checked on 2026-10-11. The acceptance criterion is at least two years of real
historical weekend data, with hourly Saturday chain observations and usable
execution quotes, features, actual expiries and settlement/exit outcomes. A vendor
coverage claim is not downloaded data. A long trade tape is not a quote archive.

## Read-only probes actually performed

- Tardis/Deribit: January 1, 2022 schema sample accessible; January 2 returns 401.
- optionsDX: public monthly BTC product includes June 2021 through September 2024.
  Price endpoint confirms EOD variants for January 2022, January 2023 and September
  2024 are $0, while 30-minute variants are $5/month. Download sample was obtained.
  Complete free data requires account/checkout; EOD is not hourly coverage. No account
  was created, no legal terms accepted and no purchase made.
- Laevitas: correct v2 options ticker-history endpoint returned HTTP 402.
- Volar: 2022 historical chain request returned HTTP 401. Vendor describes a real
  June 2021-September 2024 BTC archive; full archive requires annual Pro access.
  Its October 2024-April 2026 modeled bridge is not accepted as REAL.
- CoinAPI: historical quote request returned HTTP 401.
- CoinDesk: historical options L2 request returned HTTP 401.
- SupaGamma: public catalogue accessible; Deribit daily download returned HTTP 401.
- Deribit history API: actual January 2022 trade records accessible, including
  execution price, index, IV and size. No bid/ask or quote sizes; cannot reconstruct
  contemporaneous executable books or fill missing Greeks without assumptions.
- Kaggle: downloaded hsergeyfrolov/deribit-btc-options-information; 1,008 rows, not
  two years of history. Actual schema and bounds are recorded in open_dataset_schema.json.
- Hugging Face crypto-market-datasets: its author documents Deribit options only
  from April 2026 onward. The multi-year headline refers to Binance spot/futures.
- QuantLet hedging_cc: complete untruncated repository inventory contains 459 ticker
  CSVs dated April 1, 2019 through July 1, 2020; header/sample verified. Under two years.
- UWA academic dataset landing page: terminal request returned HTTP 403. No access
  controls were bypassed; the listed article/dataset does not certify two years of
  complete hourly weekend books.
- Deribit/Cryptarbitrage free archive is documented as January 13-July 27, 2024,
  under seven months; its article's multi-year backtest does not release that full data.
- OKX official public option order-book archive requests on April 1, 2023, January
  6, 2024 and April 5, 2025 all returned HTTP 403 in this environment.
- CryptoStruct and Amberdata advertise long histories but complete archives are
  commercial. DeepTick free access is selected sample days. No full data acquired.

## Outcome

No accessible dataset has passed two-year weekend completeness validation. Real beta
and omega remain unavailable. Existing results/report.html is SYNTHETIC, unchanged.
No raw trade, daily OHLC, fitted surface, model bridge, absent day or zero quote was
silently substituted for the requested quote data. No authorization controls were
bypassed. Account registration/terms, paid access and network access are external
conditions; further retries of the same denied endpoints cannot create those permissions.

## Evidence and reproducibility

access_probes.json, optionsdx_prices.json, quantlet_audit.json,
open_dataset_schema.json, okx_archive_probes.json, kaggle_deribit.zip and the original
public response/schema files are in this directory. Commands:

```
./run.ps1 -m scripts.probe_data_sources
./run.ps1 -m scripts.probe_optionsdx
./run.ps1 -m scripts.inspect_open_datasets
./run.ps1 -m scripts.inspect_quantlet
./run.ps1 -m scripts.probe_okx_archive
```

Primary source links:
- https://www.optionsdx.com/product/btc-option-chains/
- https://www.optionsdx.com/faq/
- https://www.volardata.com/
- https://apiv2.laevitas.ch/slate/
- https://github.com/QuantLet/hedging_cc
- https://huggingface.co/datasets/3ltrashpanda/crypto-market-datasets/raw/main/README.md
- https://insights.deribit.com/education/option-backtest-selling-weekend-vol/
- https://www.okx.com/historical-data
