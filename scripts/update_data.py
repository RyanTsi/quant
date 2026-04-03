"""CLI wrapper: fetch latest stock/index data and package chunks."""

from __future__ import annotations

from runtime.services import build_data_service


def main() -> None:
    service = build_data_service(refresh_settings=True)
    result = service.fetch_data(lookback_days=7)
    print(f"Fetched {result['start_date']} -> {result['end_date']}")
    print(f"Packed data directory: {result['send_buffer_dir']}")


if __name__ == "__main__":
    main()
