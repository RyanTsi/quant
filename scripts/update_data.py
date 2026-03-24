from data_pipeline.fetcher import StockDataFetcher
from utils.run_tracker import record_run, get_last_run, today


def main():
    data_fetcher = StockDataFetcher()

    last = get_last_run("fetch_stock")
    start_date = last.get("end_date", "20100101") if last else "20100101"
    end_date = today()

    print(f"Fetching data: {start_date} -> {end_date}")
    
    data_fetcher.fetch_all_stock_history(start_date, end_date)
    data_fetcher.fetch_all_index_history(start_date, end_date)

    record_run("fetch_stock", start_date=start_date, end_date=end_date)
    print(f"Done. Run recorded: fetch_stock -> {end_date}")


if __name__ == '__main__':
    main()
