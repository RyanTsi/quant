from config.settings import settings
from data_pipeline.database import DBClient
import pandas as pd
import utils.io
import os


def filter_top_liquidity(start_year=2010, end_year=2026, top_n=500):
    db_client = DBClient(settings.db_host, settings.db_port)
    all_stock_code_list_path = os.path.join(settings.data_path, 'stock_code_list')
    all_stock_list = utils.io.read_file_lines(all_stock_code_list_path)
    all_results = []

    for year in range(start_year, end_year + 1):
        yearly_liquidity = []
        for stock_code in all_stock_list:
            response = db_client.query_data(stock_code, f"{year}-01-01", f"{year}-12-31")
            if response is None or response.status_code != 200:
                continue
            body = response.json()
            rows = body.get("data", [])
            if len(rows) < 60:
                continue
            total_turnover = sum(item['close'] * item['volume'] for item in rows)
            yearly_liquidity.append({
                'symbol': stock_code,
                'year': year,
                'turnover': total_turnover
            })
        top_stocks = sorted(yearly_liquidity, key=lambda x: x['turnover'], reverse=True)[:top_n]
        all_results.extend(top_stocks)
        print(f"Year {year} processed, top {top_n} stocks selected.")

    df_top_stocks = pd.DataFrame(all_results)
    output_path = os.path.join(settings.data_path, 'top_500_liquidity_stocks.csv')
    df_top_stocks.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    filter_top_liquidity()
