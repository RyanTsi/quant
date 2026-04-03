"""CLI wrapper: ingest CSV data into the C++ gateway."""

from __future__ import annotations

import argparse

from runtime.services import build_data_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest CSV data to gateway")
    parser.add_argument("--data_dir", type=str, default=None, help="CSV directory. Default: .data/send_buffer")
    parser.add_argument(
        "--delete_after_ingest",
        action="store_true",
        help="Delete local CSV files after ingest attempt (including failed files).",
    )
    args = parser.parse_args()

    service = build_data_service(refresh_settings=True)
    server_url = f"http://{service.settings.db_host}:{service.settings.db_port}"
    data_dir = args.data_dir or service.settings.send_buffer_path
    print(f"Server: {server_url}")
    print(f"Data:   {data_dir}")

    result = service.ingest_to_db(
        data_dir=args.data_dir,
        delete_after_ingest=args.delete_after_ingest,
    )
    if result is None:
        return


if __name__ == "__main__":
    main()
