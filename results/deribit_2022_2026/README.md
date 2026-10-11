# Deribit migration status

Real data access was exercised, not just assumed. On 2026-10-11:
- Public metadata returned 938 current BTC-settled option instruments.
- Current Sunday ET expiry: 2026-10-11 08:00 UTC. This is not historical constituent evidence.
- A public current book and official delivery prices were archived.
- Tardis 2022-01-01 options_chain schema sample was accessible.
- Tardis 2022-01-02 returned HTTP 401; TARDIS_API_KEY was not configured.
- Three new tests passed: inverse settlement/PnL, credential handling and no-trade endpoint guard.

No full Deribit historical archive was downloaded, no Deribit beta/omega fitted,
and no Deribit Nautilus backtest was run. access_audit.json retains null beta/omega
and weekend_coverage.csv distinguishes missing history from future/incomplete weekends.
The existing results/report.html remains synthetic.

Full authorized options_chain/quote history, underlying index and actual trading
volume history, official delivery records, historical lot/multiplier metadata and
an inverse BTC-account execution adapter are required before claiming a real run.
Do not multiply historical BTC premium by entry spot and then subtract USD payoff:
that implicitly hedges the premium's BTC FX exposure and changes the strategy.
