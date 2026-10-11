import argparse
from src.trade_data import download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2022-01-01')
    parser.add_argument('--end', default='2026-10-10')
    parser.add_argument('--output', default='data/entry_timing/fri_16utc')
    args = parser.parse_args()
    audit = download(args.output, args.start, args.end,decision_offset_utc_hours=-8)
    print(audit.status.value_counts().to_string())


if __name__ == '__main__':
    main()
