import logging
import os
from datetime import datetime, timedelta

from scheduler.decorator import task
from utils.run_tracker import record_run, get_last_run, today, today_dash
from utils.io import package_data
from config.settings import settings

logger = logging.getLogger("scheduler")


@task("fetch_data")
def fetch_data():
    """Fetch stock & index data. start_date = last_run - 7 days, end_date = today."""
    from data_pipeline.fetcher import StockDataFetcher

    fetcher = StockDataFetcher()

    last = get_last_run("fetch_stock")
    last_str = "20100108" if last is None or "end_date" not in last else last["end_date"]
    last_dt = datetime.strptime(last_str, "%Y%m%d")
    start_date = (last_dt - timedelta(days=7)).strftime("%Y%m%d")
    end_date = today()

    logger.info(f"  Fetching: {start_date} -> {end_date} (last_run={last}, lookback 7d)")
    save_dir = os.path.join(settings.data_path, f'{start_date}-{end_date}')
    fetcher.fetch_all_stock_history(start_date, end_date, save_dir)
    fetcher.fetch_all_index_history(start_date, end_date, save_dir)
    package_data(save_dir, settings.send_buffer_path)

    record_run("fetch_stock", start_date=start_date, end_date=end_date)


@task("ingest_to_db")
def ingest_to_db():
    """Push CSV data to the C++ data gateway via HTTP."""
    from data_pipeline.ingest import ingest_directory

    server_url = f"http://{settings.db_host}:{settings.db_port}"
    data_dir = settings.send_buffer_path

    if not os.path.isdir(data_dir):
        logger.warning("  No data directory found, skipping.")
        return

    logger.info(f"  Server: {server_url}")
    logger.info(f"  Data:   {data_dir}")
    ingest_directory(server_url, data_dir)

    record_run("ingest_to_db", data_dir=data_dir)


@task("export_from_db")
def export_from_db():
    """Export all symbols' full history (2010-today) from DB to per-symbol CSVs."""
    import pandas as pd
    from data_pipeline.database import DBClient

    START_DATE = "2010-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    output_dir = os.path.join(settings.data_path, "db_export")
    os.makedirs(output_dir, exist_ok=True)

    client = DBClient(settings.db_host, settings.db_port)
    health = client.health()
    if health.get("status") != "healthy":
        logger.error(f"  DB unreachable: {health}")
        return

    resp = client.list_symbols()
    if resp is None or resp.status_code != 200:
        logger.error("  Failed to list symbols from DB")
        return
    symbols = resp.json().get("symbols", [])
    total = len(symbols)
    logger.info(f"  Exporting {total} symbols: {START_DATE} -> {end_date}")

    csv_columns = ["date", "open", "high", "low", "close",
                   "volume", "amount", "turn", "tradestatus", "isST"]
    db_to_csv = {"is_st": "isST"}

    exported = 0
    for i, symbol in enumerate(symbols):
        all_rows = []
        offset = 0
        page_size = 5000
        while True:
            r = client.query_data(symbol, START_DATE, end_date,
                                  limit=page_size, offset=offset)
            if r is None or r.status_code != 200:
                break
            data = r.json().get("data", [])
            all_rows.extend(data)
            if len(data) < page_size:
                break
            offset += page_size

        if not all_rows:
            continue

        df = pd.DataFrame(all_rows)
        df.rename(columns=db_to_csv, inplace=True)
        if "date" in df.columns:
            df["date"] = df["date"].astype(str).str[:10]
        if "symbol" in df.columns:
            df.drop(columns=["symbol"], inplace=True)
        existing = [c for c in csv_columns if c in df.columns]
        df = df[existing]
        df.to_csv(os.path.join(output_dir, f"{symbol}.csv"), index=False)
        exported += 1

        if (i + 1) % 500 == 0:
            logger.info(f"  Progress: {i + 1}/{total}")

    logger.info(f"  Exported {exported}/{total} symbols to {output_dir}")
    record_run("export_from_db", start_date=START_DATE, end_date=end_date,
               output_dir=output_dir, symbols=exported)
