from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

import pandas as pd

from data_pipeline.database import DBClient
from data_pipeline.ingest import ingest_directory
from quantcore.history import RunHistoryStore
from quantcore.settings import AppSettings
from utils.io import package_data


class DataPipelineService:
    def __init__(self, settings: AppSettings, *, history: RunHistoryStore | None = None):
        self.settings = settings
        self.history = history or RunHistoryStore(os.path.join(settings.data_path, "run_history.json"))
        self.logger = logging.getLogger("scheduler")

    def fetch_data(self, *, lookback_days: int = 7) -> dict[str, str]:
        """Fetch stock/index data then package CSV chunks for ingest."""
        from data_pipeline.fetcher import StockDataFetcher

        fetcher = StockDataFetcher()
        last = self.history.get("fetch_stock")
        last_end = "20100108" if last is None or "end_date" not in last else str(last["end_date"])
        last_dt = datetime.strptime(last_end, "%Y%m%d")
        start_date = (last_dt - timedelta(days=lookback_days)).strftime("%Y%m%d")
        end_date = datetime.now().strftime("%Y%m%d")

        save_dir = os.path.join(self.settings.data_path, f"{start_date}-{end_date}")
        os.makedirs(save_dir, exist_ok=True)
        self.logger.info("  Fetching: %s -> %s (last_run=%s, lookback=%dd)", start_date, end_date, last, lookback_days)
        fetcher.fetch_all_stock_history(start_date, end_date, save_dir)
        fetcher.fetch_all_index_history(start_date, end_date, save_dir)
        package_data(save_dir, self.settings.send_buffer_path)

        self.history.record("fetch_stock", start_date=start_date, end_date=end_date)
        return {"start_date": start_date, "end_date": end_date, "save_dir": save_dir}

    def ingest_to_db(self, *, delete_after_ingest: bool = True) -> dict[str, str] | None:
        """Push packaged CSV chunks to the C++ gateway."""
        data_dir = self.settings.send_buffer_path
        if not os.path.isdir(data_dir):
            self.logger.warning("  No data directory found, skipping.")
            return None

        server_url = f"http://{self.settings.db_host}:{self.settings.db_port}"
        self.logger.info("  Server: %s", server_url)
        self.logger.info("  Data:   %s", data_dir)
        ingest_directory(server_url, data_dir, delete_after_ingest=delete_after_ingest)
        self.history.record("ingest_to_db", data_dir=data_dir, server_url=server_url)
        return {"data_dir": data_dir, "server_url": server_url}

    def export_from_db(self, *, start_date: str = "2010-01-01") -> dict[str, int | str]:
        """
        Export all symbols from DB to per-symbol CSV files in receive buffer.
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        output_dir = self.settings.receive_buffer_path
        os.makedirs(output_dir, exist_ok=True)

        client = DBClient(self.settings.db_host, self.settings.db_port)
        health = client.health()
        if health.get("status") != "healthy":
            raise RuntimeError(f"DB unreachable: {health}")

        resp = client.list_symbols()
        if resp is None or resp.status_code != 200:
            raise RuntimeError("Failed to list symbols from DB.")

        symbols = resp.json().get("symbols", [])
        total = len(symbols)
        self.logger.info("  Exporting %d symbols: %s -> %s", total, start_date, end_date)

        csv_columns = ["date", "open", "high", "low", "close", "volume", "amount", "turn", "isST", "factor"]
        db_to_csv = {"is_st": "isST"}

        exported = 0
        for i, symbol in enumerate(symbols):
            all_rows = []
            offset = 0
            page_size = 3000
            while True:
                r = client.query_data(symbol, start_date, end_date, limit=page_size, offset=offset)
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
            if "factor" not in df.columns:
                df["factor"] = 1.0
            if "date" in df.columns:
                df["date"] = df["date"].astype(str).str[:10]
            if "symbol" in df.columns:
                df.drop(columns=["symbol"], inplace=True)
            if "tradestatus" in df.columns:
                df = df[df["tradestatus"] != 0]
            existing = [c for c in csv_columns if c in df.columns]
            df = df[existing]
            df.to_csv(os.path.join(output_dir, f"{symbol}.csv"), index=False)
            exported += 1

            if (i + 1) % 500 == 0:
                self.logger.info("  Progress: %d/%d", i + 1, total)

        self.logger.info("  Exported %d/%d symbols to %s", exported, total, output_dir)
        self.history.record("export_from_db", output_dir=output_dir, exported=exported, total=total)
        return {"output_dir": output_dir, "exported": exported, "total": total}
