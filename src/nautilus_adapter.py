"""Read-only research replay; never instantiates a live trading node."""
import numpy as np
from nautilus_trader.backtest.engine import BacktestEngine
from nautilus_trader.config import BacktestEngineConfig,LoggingConfig
from nautilus_trader.model.instruments import CryptoOption
from nautilus_trader.model.identifiers import InstrumentId,Symbol,Venue
from nautilus_trader.model.enums import AssetClass,OptionKind,AccountType,OmsType,OrderSide
from nautilus_trader.model.objects import Price,Quantity,Money,Currency
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.trading.strategy import Strategy

class ReplayStrategy(Strategy):
    def __init__(self,schedule):
        super().__init__(); self.schedule=schedule; self.done=set()
    def on_start(self):
        for instrument_id in self.schedule: self.subscribe_quote_ticks(instrument_id)
    def on_quote_tick(self,tick):
        entry,exit_,quantity=self.schedule[tick.instrument_id]
        phase='entry' if tick.ts_event==entry else 'exit' if tick.ts_event==exit_ else None
        key=(tick.instrument_id,phase)
        if phase is None or key in self.done: return
        self.done.add(key)
        self.submit_order(self.order_factory.market(instrument_id=tick.instrument_id,order_side=OrderSide.SELL if phase=='entry' else OrderSide.BUY,quantity=Quantity.from_str(f'{quantity:.3f}')))

def replay(trades,cfg,p):
    engine=BacktestEngine(BacktestEngineConfig(logging=LoggingConfig(log_level='ERROR')))
    venue=Venue('RESEARCH'); currency=Currency.from_str('USDT')
    engine.add_venue(venue=venue,oms_type=OmsType.NETTING,account_type=AccountType.MARGIN,base_currency=currency,starting_balances=[Money(100000,currency)])
    ticks=[]; schedule={}
    for r in trades.itertuples():
        iid=InstrumentId(Symbol(r.symbol),venue); entry=r.entry_time_utc.value; exit_=r.expiry.value-1_000_000_000
        instrument=CryptoOption(instrument_id=iid,raw_symbol=Symbol(r.symbol),underlying=Currency.from_str('BTC'),quote_currency=currency,settlement_currency=currency,is_inverse=False,price_precision=2,size_precision=3,price_increment=Price.from_str('0.01'),size_increment=Quantity.from_str('0.001'),multiplier=Quantity.from_str('1'),lot_size=Quantity.from_str('0.001'),option_kind=OptionKind.CALL if r.option_type=='CALL' else OptionKind.PUT,strike_price=Price.from_str(f'{r.strike:.2f}'),activation_ns=entry-1,expiration_ns=r.expiry.value,ts_event=entry-1,ts_init=entry-1)
        engine.add_instrument(instrument); schedule[iid]=(entry,exit_,r.quantity)
        for time,bid,ask in [(entry,r.bid,r.ask),(exit_,max(r.settlement,.01),max(r.settlement,.01))]:
            ticks.append(QuoteTick(instrument_id=iid,bid_price=Price.from_str(f'{bid:.2f}'),ask_price=Price.from_str(f'{ask:.2f}'),bid_size=Quantity.from_str('100.000'),ask_size=Quantity.from_str('100.000'),ts_event=time,ts_init=time))
    if not ticks:
        engine.dispose(); return {'engine':'NautilusTrader','fills':0,'gross_pnl':0,'reconciliation_error':0}
    engine.add_strategy(ReplayStrategy(schedule)); engine.add_data(sorted(ticks,key=lambda t:t.ts_init)); engine.run()
    frames={'orders':engine.trader.generate_orders_report(),'fills':engine.trader.generate_order_fills_report(),'positions':engine.trader.generate_positions_report(),'account':engine.trader.generate_account_report(venue)}
    for name,frame in frames.items(): frame.to_csv(p/f'nautilus_{name}.csv')
    fills=frames['fills']
    if len(fills)!=2*len(trades): raise RuntimeError(f'Nautilus incomplete fills: {len(fills)} / {2*len(trades)}')
    # Reconcile actual engine fill prices and quantities against the research ledger.
    actual=0.
    for _,row in fills.iterrows():
        qty=float(str(row['filled_qty']).replace(',','')); price=float(str(row['avg_px']).replace(',',''))
        actual+=qty*price*(1 if str(row['side']).upper()=='SELL' else -1)
    expected=float(trades.gross_pnl.sum()); error=actual-expected
    tolerance=float(trades.quantity.sum()*.021+1e-6)
    if abs(error)>tolerance: raise RuntimeError(f'Engine gross PnL mismatch {error}, tolerance {tolerance}')
    result={'engine':'NautilusTrader','fills':len(fills),'gross_pnl':actual,'reconciliation_error':error,'tolerance':tolerance,'net_pnl_after_explicit_costs':actual-float((trades.fees+trades.slippage).sum())}
    engine.dispose(); return result
