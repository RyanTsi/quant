import requests
import os
import sys
from config.settings import settings
import utils.io

BATCH_SIZE = 4096


def ingest_directory(base_url: str, data_dir: str):
    if not os.path.isdir(data_dir):
        print(f"Directory not found: {data_dir}")
        return

    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    total = len(files)
    print(f"Found {total} CSV files in {data_dir}")

    for idx, filename in enumerate(files, 1):
        filepath = os.path.join(data_dir, filename)
        symbol = filename.split(".")[0]
        lines = utils.io.read_file_lines(filepath)
        if len(lines) < 2:
            continue

        header = lines[0].split(",")
        col_map = {k.strip(): v for v, k in enumerate(header)}

        data_list = []
        for line in lines[1:]:
            cols = line.split(",")
            if len(cols) < len(header):
                continue
            payload = {
                "date":        cols[col_map["date"]],
                "symbol":      symbol,
                "open":        float(cols[col_map["open"]]),
                "high":        float(cols[col_map["high"]]),
                "low":         float(cols[col_map["low"]]),
                "close":       float(cols[col_map["close"]]),
                "volume":      float(cols[col_map.get("volume", col_map.get("volume", 0))]),
                "amount":      float(cols[col_map["amount"]]) if "amount" in col_map else 0.0,
                "turn":        float(cols[col_map["turn"]]) if "turn" in col_map else 0.0,
                "tradestatus": int(cols[col_map["tradestatus"]]) if "tradestatus" in col_map else 1,
                "isST":        int(cols[col_map["isST"]]) if "isST" in col_map else 0,
            }
            data_list.append(payload)

        for i in range(0, len(data_list), BATCH_SIZE):
            batch = data_list[i:i + BATCH_SIZE]
            try:
                resp = requests.post(f"{base_url}/api/v1/ingest/daily", json=batch, timeout=30)
                if resp.status_code != 200:
                    print(f"  [{symbol}] batch {i//BATCH_SIZE+1} failed: {resp.status_code} {resp.text}")
            except Exception as e:
                print(f"  [{symbol}] batch {i//BATCH_SIZE+1} error: {e}")

        print(f"[{idx}/{total}] {symbol}: {len(data_list)} rows sent.")


if __name__ == "__main__":
    server_url = f"http://{settings.db_host}:{settings.db_port}"
    data_dir = os.path.join(settings.data_path, "20100101-20260319")

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    print(f"Server: {server_url}")
    print(f"Data:   {data_dir}")
    ingest_directory(server_url, data_dir)
