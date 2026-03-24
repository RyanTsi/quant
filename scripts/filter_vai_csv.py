"""Filter top-N liquidity stocks per year from local CSV directory.

Usage:
    python -m scripts.filter_vai_csv [--data_dir DIR] [--output PATH]
"""

import argparse
import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from config.settings import settings

START_YEAR = 2010
END_YEAR = 2026
MIN_TRADING_DAYS = 5
TOP_N = 500

_data_path = None


def load_stock(file):
    symbol = file.replace('.csv', '')
    df = pd.read_csv(os.path.join(_data_path, file), usecols=['date', 'amount', 'tradestatus'])
    df = df[df['tradestatus'] != 0]
    df['year'] = pd.to_datetime(df['date']).dt.year
    yearly = df.groupby('year').agg(count=('amount', 'size'), turnover=('amount', 'sum')).reset_index()
    yearly = yearly[yearly['count'] >= MIN_TRADING_DAYS]
    yearly['symbol'] = symbol
    return yearly[['symbol', 'year', 'turnover']]


def main():
    global _data_path

    parser = argparse.ArgumentParser(description="Filter top liquidity stocks from CSV data")
    parser.add_argument("--data_dir", default=None, help="CSV directory (default: latest in .data/)")
    parser.add_argument("--output", default=None, help="Output file path")
    args = parser.parse_args()

    _data_path = args.data_dir
    if _data_path is None:
        candidates = sorted(
            d for d in os.listdir(settings.data_path)
            if os.path.isdir(os.path.join(settings.data_path, d)) and "-" in d
        )
        if not candidates:
            print("No data directory found.")
            return
        _data_path = os.path.join(settings.data_path, candidates[-1])

    output_path = args.output or os.path.join(settings.data_path, 'top_500_liquidity_stocks.txt')
    index_list_path = os.path.join(settings.data_path, 'index_code_list')

    index_codes = set()
    if os.path.exists(index_list_path):
        with open(index_list_path, 'r') as f:
            index_codes = {line.strip() for line in f if line.strip()}

    files = [f for f in os.listdir(_data_path) if f.endswith('.csv') and f.replace('.csv', '') not in index_codes]
    print(f"Total files: {len(files)}")

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(load_stock, files))

    df_all = pd.concat(results, ignore_index=True)

    top_stocks = (
        df_all
        .groupby('year', group_keys=False)
        .apply(lambda g: g.nlargest(TOP_N, 'turnover'))
        .reset_index(drop=True)
    )

    top_stocks = top_stocks[(top_stocks['year'] >= START_YEAR) & (top_stocks['year'] <= END_YEAR)]
    with open(output_path, 'w') as out:
        for _, row in top_stocks.iterrows():
            y = int(row['year'])
            out.write(f"{row['symbol']}\t{y}-01-01\t{y}-12-31\n")

    print(f"Done. Saved {len(top_stocks)} rows to {output_path}")
    for year in range(START_YEAR, END_YEAR + 1):
        count = len(top_stocks[top_stocks['year'] == year])
        print(f"  {year}: {count} stocks")


if __name__ == '__main__':
    main()
