from data_pipeline.fetcher import DataFetcher


date = '312'
fetcher = DataFetcher()
df = fetcher.fetch_current_index_spot_df()
df.to_csv(f'index_spot_{date}.csv', index=False)
df = fetcher.fetch_current_stock_spot_df()
df.to_csv(f'stock_spot_{date}.csv', index=False)