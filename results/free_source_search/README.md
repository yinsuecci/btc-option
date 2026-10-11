# Free BTC Option Data Search

Checked on 2026-10-11. Current requirement: free BTC option data covering at
least two years of weekends; daily observations are acceptable. The original
strategy still requires actual bid/ask, contract eligibility and settlement.

## Verified Free Multi-Year Source: Trades Only

Deribit public history endpoint:
https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time

Actual checks completed for all 106 consecutive Saturday-start ET weekends
from 2022-01-01 through 2024-01-06. Every weekend returned a valid BTC option
trade within its requested Saturday/Sunday ET window, without credentials or
payment. Raw JSON responses and requested URLs are retained here. This confirms
multi-year public trade access, NOT complete market data or strategy eligibility.

The audit requests ONE trade per weekend, not all trades. It does not assert
that every Sunday-expiry strike traded, that every day has records, or that all
quotes are available. Response fields include price, amount, direction, IV,
index price and mark price. They do not include historical bid/ask, quote size,
open interest or complete Greeks. Never use this sample as a training dataset.

Official schema:
https://docs.deribit.com/api-reference/market-data/public-get_last_trades_by_currency_and_time

`weekend_access.json` contains the completed 106/106 access audit and explicit
false completeness flags. This is a verified free option TRADE data source,
not a replacement for the original executable quote-based backtest.

## Additional Candidates Checked

- Mendeley hfd4b8zfw6/1: CC0, 5-minute Deribit snapshots only December 9-18,
  2018. Ten days, not two years. Public page fetched successfully.
  https://data.mendeley.com/datasets/hfd4b8zfw6/1
- CryptoDataDownload: multi-year maturity OHLCV downloads on the current
  Deribit page link to Join Plus+. Its quote archive starts 2025-12-31;
  does not establish two years of free quotes.
  https://www.cryptodatadownload.com/data/deribit/
  https://api.cryptodatadownload.com/static/catalog/CDD_Data_Catalog.html
- Blockchain Research Center: request/member workflow; paper's actual
  option-book sample spans 2021-04-01 to 2022-04-01, one year. No freely
  downloadable two-year full option-chain archive verified.
  https://blockchain-research-center.com/blockchain-explorer/request-data/
  https://www.mdpi.com/2227-9091/11/5/85
- Azulcoder Hugging Face: indexed card describes a recent keyless capture,
  but current metadata and directory requests return 401. Indexed schema
  also lacks option bid/ask. No two-year quote history obtained.
  https://huggingface.co/datasets/azulcoder/btc-quant-ticks
- Uforia Polynomial: public landing request returned 503; neither time
  bounds nor downloadable complete history could be verified.
  https://uforia.polynomial.fi/datasets?dataset=option_chain_snapshot
- Loris: official pricing lists history since September 2025 and no free
  historical lookback. Not two years of free option data.
  https://loris.tools/pricing
- Amberdata: free analytics UI advertises historical IV since 2019;
  this does not establish free export of complete historical contract books.
  https://www.amberdata.io/hubfs/AmberdataFreeAccess.pdf
- Stochvolmodels: weekly BTC calibrations April 2019-October 2023 are
  explicitly marked not distributable; bundled chain is one 2021 snapshot.
  https://stochvolmodels.readthedocs.io/en/latest/analytics_gallery.html
- Deep-hedging paper's five-year headline is monthly first-day windows,
  not continuous data or all weekends. Do not confuse it with full history.
  https://arxiv.org/html/2608.29025v1

Public Kaggle and Hugging Face catalog responses are saved under `catalogs/`.
New catalog candidates did not establish two-year complete BTC option books.
Previous probes remain under `results/source_search/`.

## Outcome

Two-year free option trade access verified. Two-year free complete quote-chain
coverage NOT found or verified. Real strategy beta and omega remain uncomputed.
No synthetic spread/Greek, monthly sample interpolation, absent-day fill, paid
subscription, signup, or access-control workaround was used.

## Reproduce

```
./run.ps1 scripts/audit_free_weekends.py
./run.ps1 scripts/inspect_free_catalogs.py
```

The access audit is resumable from saved raw responses. Both scripts passed
Python syntax compilation. These evidence files are not backtest outputs.
