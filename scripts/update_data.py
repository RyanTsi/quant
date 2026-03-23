from data_pipeline.fetcher import StockDataFetcher
from utils.run_tracker import record_run, get_last_date, today


def main():
    data_fetcher = StockDataFetcher()

    start_date = get_last_date("fetch_stock", default="20100101")
    end_date = today()

    print(f"Fetching data: {start_date} -> {end_date}")
    
    data_fetcher.fetch_all_stock_history(start_date, end_date)
    data_fetcher.fetch_all_index_history(start_date, end_date)

    record_run("fetch_stock", start_date=start_date, end_date=end_date)
    print(f"Done. Run recorded: fetch_stock -> {end_date}")


if __name__ == '__main__':
    main()
