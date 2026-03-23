import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor

DATA_PATH = "C:/Users/sola/Documents/quant/.data/20100101-20260319"
INDEX_LIST_PATH = "C:/Users/sola/Documents/quant/.data/index_code_list"
OUTPUT_PATH = "C:/Users/sola/Documents/quant/.data/top_500_liquidity_stocks.txt"
START_YEAR = 2010
END_YEAR = 2026
MIN_TRADING_DAYS = 5
TOP_N = 500


def load_stock(file):
    symbol = file.replace('.csv', '')
    df = pd.read_csv(os.path.join(DATA_PATH, file), usecols=['date', 'amount', 'tradestatus'])
    df = df[df['tradestatus'] != 0]
    df['year'] = pd.to_datetime(df['date']).dt.year
    yearly = df.groupby('year').agg(count=('amount', 'size'), turnover=('amount', 'sum')).reset_index()
    yearly = yearly[yearly['count'] >= MIN_TRADING_DAYS]
    yearly['symbol'] = symbol
    return yearly[['symbol', 'year', 'turnover']]


if __name__ == '__main__':
    with open(INDEX_LIST_PATH, 'r') as f:
        index_codes = {line.strip() for line in f if line.strip()}
    files = [f for f in os.listdir(DATA_PATH) if f.endswith('.csv') and f.replace('.csv', '') not in index_codes]
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
    with open(OUTPUT_PATH, 'w') as out:
        for _, row in top_stocks.iterrows():
            y = int(row['year'])
            out.write(f"{row['symbol']}\t{y}-01-01\t{y}-12-31\n")
    print(f"Done. Saved {len(top_stocks)} rows to {OUTPUT_PATH}")
    for year in range(START_YEAR, END_YEAR + 1):
        count = len(top_stocks[top_stocks['year'] == year])
        print(f"  {year}: {count} stocks")
