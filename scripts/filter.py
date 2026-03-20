from config.settings import settings
from data_pipeline.database import DBClient
import pandas as pd
import utils.io
import os

db_client = DBClient(settings.db_host, settings.db_port)
all_stock_code_list_path = os.path.join(settings.data_path, 'stock_code_list')
all_stock_list = utils.io.read_file_lines(all_stock_code_list_path)
start_year = 2010
end_year = 2026
all_results = []

for year in range(start_year, end_year + 1):
    yearly_liquidity = []
    for stock_code in all_stock_list:
        response = db_client.query_data(stock_code, f"{year}-01-01", f"{year}-12-31")
        if response.status_code == 200:
            data = response.json()
            if len(data) < 60:
                continue
            total_turnover = sum(item['close'] * item['volume'] for item in data)
            yearly_liquidity.append({
                'symbol': stock_code,
                'year': year,
                'turnover': total_turnover
            })
    top_500_this_year = sorted(yearly_liquidity, key=lambda x: x['turnover'], reverse=True)[:500]
    all_results.extend(top_500_this_year)
    print(f"Year {year} processed, top 500 stocks selected.")

df_top_stocks = pd.DataFrame(all_results)
df_top_stocks.to_csv(os.path.join(settings.data_path, 'top_500_liquidity_stocks.csv'), index=False)