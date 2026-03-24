import logging
import traceback
import os
from datetime import datetime, timedelta
from functools import wraps

from utils.format import format_date
from utils.run_tracker import record_run, get_last_run, today, today_dash
from config.settings import settings

logger = logging.getLogger("scheduler")


def task(name: str):
    """Decorator that wraps a task with logging, timing, and error handling."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = datetime.now()
            logger.info(f"[{name}] started at {start:%H:%M:%S}")
            try:
                result = func(*args, **kwargs)
                elapsed = (datetime.now() - start).total_seconds()
                logger.info(f"[{name}] finished in {elapsed:.1f}s")
                return result
            except Exception:
                logger.error(f"[{name}] failed:\n{traceback.format_exc()}")
                return None
        wrapper.task_name = name
        return wrapper
    return decorator


# ─── Individual Tasks ───────────────────────────────────────────────


@task("fetch_data")
def fetch_data():
    """Fetch stock & index data. start_date = last_run - 7 days, end_date = today."""
    from data_pipeline.fetcher import StockDataFetcher

    fetcher = StockDataFetcher()

    last = get_last_run("fetch_stock")["last_run"][:10]
    last = format_date(last,"YYYYMMDD")
    last_dt = datetime.strptime(last, "%Y%m%d")
    start_date = (last_dt - timedelta(days=7)).strftime("%Y%m%d")
    end_date = today()

    logger.info(f"  Fetching: {start_date} -> {end_date} (last_run={last}, lookback 7d)")
    fetcher.fetch_all_stock_history(start_date, end_date)
    fetcher.fetch_all_index_history(start_date, end_date)

    record_run("fetch_stock", start_date=start_date, end_date=end_date)


@task("ingest_to_db")
def ingest_to_db():
    """Push CSV data to the C++ data gateway via HTTP."""
    from scripts.put_data import ingest_directory

    server_url = f"http://{settings.db_host}:{settings.db_port}"
    data_dir = os.path.join(settings.data_path, f"{get_last_date('fetch_stock', '20100101')}-{today()}")

    if not os.path.isdir(data_dir):
        candidates = [d for d in os.listdir(settings.data_path)
                      if os.path.isdir(os.path.join(settings.data_path, d)) and "-" in d]
        if candidates:
            data_dir = os.path.join(settings.data_path, sorted(candidates)[-1])
        else:
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


@task("dump_to_qlib")
def dump_to_qlib():
    """Convert CSV data to Qlib binary format."""
    import subprocess
    import sys

    csv_dir = os.path.join(settings.data_path, "db_export")
    qlib_dir = os.path.join(settings.data_path, "qlib_data")

    if not os.path.isdir(csv_dir) or not os.listdir(csv_dir):
        logger.warning("  No CSV data in db_export/, skipping.")
        return

    logger.info(f"  Source: {csv_dir}")
    logger.info(f"  Target: {qlib_dir}")

    subprocess.run([
        sys.executable, "scripts/dump_bin.py", "dump_all",
        f"--csv_path={csv_dir}",
        f"--qlib_dir={qlib_dir}",
        "--date_field_name=date",
        "--include_fields=open,high,low,close,volume,amount,turn",
    ], check=True)

    record_run("dump_to_qlib", csv_dir=csv_dir, qlib_dir=qlib_dir)


@task("predict")
def predict():
    """Generate stock predictions using the latest trained model."""
    import subprocess
    import sys
    subprocess.run([sys.executable, "scripts/predict.py"], check=True)
    record_run("predict", date=today_dash())


@task("train_model")
def train_model():
    """Train the Transformer model via Qlib workflow."""
    from alpha_models.qlib_workflow import main as qlib_main
    qlib_main()
    record_run("train_model", date=today_dash())


# ─── Pipelines ──────────────────────────────────────────────────────

# 18:15 — fetch from network + ingest to DB
EVENING_PIPELINE = [
    fetch_data,
    ingest_to_db,
]

# 14:00 — pull from DB, update qlib data, predict
AFTERNOON_PIPELINE = [
    export_from_db,
    dump_to_qlib,
    predict,
]

# TODO: Weekly retrain pipeline — schedule for e.g. Saturday 02:00
# Blocked: model retraining workflow not yet finalized.
# When ready, uncomment and wire up in main.py setup_schedule().
#
# WEEKLY_RETRAIN_PIPELINE = [
#     fetch_data,
#     ingest_to_db,
#     dump_to_qlib,
#     train_model,
#     predict,
# ]

FULL_PIPELINE = [
    fetch_data,
    ingest_to_db,
    dump_to_qlib,
    train_model,
    predict,
]


def run_pipeline(pipeline: list):
    """Run a list of tasks sequentially."""
    logger.info(f"{'='*50}")
    logger.info(f"Pipeline started at {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"Tasks: {[t.task_name for t in pipeline]}")
    logger.info(f"{'='*50}")

    for task_func in pipeline:
        task_func()
    logger.info(f"Pipeline completed at {datetime.now():%Y-%m-%d %H:%M:%S}\n")
