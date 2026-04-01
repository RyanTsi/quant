"""CLI wrapper: ingest CSV data into the C++ gateway."""

from __future__ import annotations

import argparse

from data_pipeline.ingest import ingest_directory
from quantcore.settings import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest CSV data to gateway")
    parser.add_argument("--data_dir", type=str, default=None, help="CSV directory. Default: .data/send_buffer")
    parser.add_argument(
        "--delete_after_ingest",
        action="store_true",
        help="Delete local CSV files after successful ingest.",
    )
    args = parser.parse_args()

    settings = get_settings(refresh=True)
    server_url = f"http://{settings.db_host}:{settings.db_port}"
    data_dir = args.data_dir or settings.send_buffer_path

    print(f"Server: {server_url}")
    print(f"Data:   {data_dir}")
    ingest_directory(server_url, data_dir, delete_after_ingest=args.delete_after_ingest)


if __name__ == "__main__":
    main()
