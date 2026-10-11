import argparse
import json
from src.trade_research import run, TradeConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/deribit_trade_daily')
    parser.add_argument('--output', default='results/trade_only')
    parser.add_argument('--min-train-weekends', type=int, default=26)
    args = parser.parse_args()
    result = run(args.data, args.output, TradeConfig(min_train_weekends=args.min_train_weekends))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
