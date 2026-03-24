"""CLI wrapper: ingest CSV files into the data gateway.

Usage:
    python -m scripts.put_data [DATA_DIR]
"""

import os
import sys

from config.settings import settings
from data_pipeline.ingest import ingest_directory


if __name__ == "__main__":
    server_url = f"http://{settings.db_host}:{settings.db_port}"
    data_dir = os.path.join(settings.data_path, "20260312-20260324")

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]

    print(f"Server: {server_url}")
    print(f"Data:   {data_dir}")
    ingest_directory(server_url, data_dir)
