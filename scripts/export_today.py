"""Export market data from the database to CSV files.

Usage:
    python -m scripts.export_today                        # export today
    python -m scripts.export_today --date 2026-03-18      # specific date
    python -m scripts.export_today --date 2026-03-18 --output ./out.csv
"""

import argparse
import sys
import os
import requests
import pandas as pd
from datetime import datetime

from config.settings import settings
from data_pipeline.database import DBClient


def fetch_all_by_date(client: DBClient, date: str, page_size=5000):
    """Paginate through /query/daily/all and return a combined DataFrame."""
    session = requests.Session()
    all_rows = []
    offset = 0
    while True:
        try:
            resp = session.get(
                f"{client.base_url}/query/daily/all",
                params={"date": date, "limit": page_size, "offset": offset},
                timeout=30,
            )
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return None
        if resp.status_code != 200:
            print(f"Error: server returned {resp.status_code} — {resp.text}")
            return None
        body = resp.json()
        data = body.get("data", [])
        all_rows.extend(data)
        if len(data) < page_size:
            break
        offset += page_size
    if not all_rows:
        return None
    return pd.DataFrame(all_rows)


def export_date_to_csv(date: str, output_dir: str = None) -> str | None:
    """Export a single date's market data to CSV. Returns the output path or None."""
    client = DBClient(settings.db_host, settings.db_port)

    health = client.health()
    if health.get("status") == "unreachable":
        print(f"Cannot reach server at {settings.db_host}:{settings.db_port}")
        return None

    print(f"Fetching market data for {date} ...")
    df = fetch_all_by_date(client, date)

    if df is None or df.empty:
        print(f"No data found for {date}")
        return None

    if "date" in df.columns:
        df["date"] = df["date"].astype(str).str[:10]

    column_order = ["symbol", "date", "open", "high", "low", "close",
                    "volume", "amount", "turn", "tradestatus", "is_st"]
    existing = [c for c in column_order if c in df.columns]
    df = df[existing]

    out_dir = output_dir or settings.data_path
    output_path = os.path.join(out_dir, f"market_{date}.csv")
    df.to_csv(output_path, index=False)
    print(f"Done. {len(df)} rows saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export a single day's market data to CSV")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                        help="Date to export, e.g. 2026-03-23 (default: today)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: .data/market_<date>.csv)")
    args = parser.parse_args()

    result = export_date_to_csv(args.date, output_dir=args.output)
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
