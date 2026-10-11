import argparse
import json
from src.center_vol import DEFAULT_SOURCE, DEFAULT_TAIL_PENALTY, run
from src.trade_research import TradeConfig

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--risk-penalty',type=float,default=DEFAULT_TAIL_PENALTY)
    parser.add_argument('--output',default='results/center_vol')
    parser.add_argument('--source',default=DEFAULT_SOURCE)
    args=parser.parse_args()
    print(json.dumps(run(source=args.source,output=args.output,execution=TradeConfig(risk_penalty_per_btc=args.risk_penalty)), indent=2))
