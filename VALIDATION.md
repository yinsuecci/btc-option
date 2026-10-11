# Validation, 2026-10-11 (Asia/Shanghai)

- Project editable build/install succeeded using run.ps1 and local dependencies.
- pytest: 18 passed; two upstream Nautilus Timestamp.utcnow deprecation warnings.
- inspect_chain: SYNTHETIC fixture succeeded.
- inspect_chain --real: Binance public exchangeInfo succeeded (current metadata only).
- download_chain: REAL exchangeInfo, mark and ticker snapshots archived under data/snapshots.
- run_signal: completed with factors, Ridge beta, option alphas, portfolio Greeks and a research-only ENTER signal.
- run_backtest: completed with NautilusTrader 1.231.0 and 146 fills.
- Timezone tests cover EST, EDT, spring DST boundary, Saturday cutoff and weekend ID.
- Factor tests perturb future spot, volume and option quotes without changing current features.
- Model tests verify prior weekend grouping, matured labels and raw-unit alpha decomposition.
- Portfolio tests verify short-alpha sign, per-leg/gross/capital/Greeks constraints and aggregate Greeks.
- Execution tests verify bid entry, exact settlement, fees/slippage, invalid/missing/stale books.
- Pipeline tests verify one entry per weekend, threshold no-entry and self-contained synthetic report warning.
- All 10 requested PNGs and report.html are generated; stress and entry-time PNGs visually inspected.

## SYNTHETIC demo only

20 source weekends; first 6 used for warmup. Test period 2025-02-15 through 2025-05-17:
14 test weekends, 13 traded, 1 no trade, 73 option legs / 146 engine fills.
Research net PnL 121.787908 USDT; engine fill-adjusted net 121.781250 USDT.
Gross engine/ledger discrepancy -0.006658 USDT, within 0.019405 rounding tolerance.
Worst weekend / empirical 5% expected shortfall -50.303575 USDT.
Worst entry-time stress -1495.095245 USDT (spot -30%, IV +50 vol points).
These results are NOT real historical performance, and do not establish profitability.

## Limitations and blockers

Complete real historical executable Binance option chains and settlement-index archive
were not supplied. Current metadata snapshots must never be substituted for them.
Imported real data must include point-in-time constituents, executable books, timestamped
Greeks and the exact official settlement index; schema and manifest are in README.
No live execution path is present.

Nautilus is a causal-decision quote/order replay: research decisions are computed before
replay using strict historical filters, rather than refitting sklearn inside event callbacks.
Terminal intrinsic quote simulates cash settlement one second before expiry; zero intrinsic
uses 0.01 USDT and prices round to 0.01. Fees and slippage are explicit research-ledger
adjustments, not native exchange commission or liquidation modeling. No dynamic hedging,
liquidation, exchange-specific margin, order book queue or partial-fill model is included.
The margin proxy does not imply Binance naked-short eligibility.
OLS/Ridge coefficient t statistics and calibrated predictive uncertainty are not provided.
Entry-hour diagnostics describe selected entries, not unbiased comparisons of all hours.
Factor collinearity and hours_to_expiry sign instability are reported.
