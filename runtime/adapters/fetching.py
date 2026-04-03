"""Runtime-owned adapter for market-data fetch orchestration."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from typing import Any

from utils.io import package_data

DEFAULT_LAST_END_DATE = "20100108"


def _default_fetcher_factory() -> Any:
    # Import locally so tests can run without importing baostock globally.
    from data_pipeline.fetcher import StockDataFetcher

    return StockDataFetcher()


def resolve_fetch_window(*, lookback_days: int, last_history: Mapping[str, Any] | None, now: datetime | None = None) -> dict[str, str]:
    """Resolve the incremental fetch window with legacy-compatible defaults."""
    last_end_date = DEFAULT_LAST_END_DATE
    if last_history is not None and "end_date" in last_history:
        last_end_date = str(last_history["end_date"])

    last_dt = datetime.strptime(last_end_date, "%Y%m%d")
    start_date = (last_dt - timedelta(days=lookback_days)).strftime("%Y%m%d")
    end_date = (now or datetime.now()).strftime("%Y%m%d")
    return {
        "start_date": start_date,
        "end_date": end_date,
        "last_end_date": last_end_date,
    }


def fetch_and_package_market_data(
    *,
    data_root: str,
    send_buffer_dir: str,
    lookback_days: int,
    last_history: Mapping[str, Any] | None,
    logger: logging.Logger | None = None,
    fetcher_factory: Callable[[], Any] | None = None,
    package_data_fn: Callable[[str, str], Any] | None = None,
    now: datetime | None = None,
) -> dict[str, str | int]:
    """Fetch stock/index bars, package CSV chunks, and return stable metadata."""
    window = resolve_fetch_window(lookback_days=lookback_days, last_history=last_history, now=now)
    start_date = window["start_date"]
    end_date = window["end_date"]
    last_end_date = window["last_end_date"]

    save_dir = os.path.join(data_root, f"{start_date}-{end_date}")
    os.makedirs(save_dir, exist_ok=True)

    active_logger = logger or logging.getLogger("scheduler")
    active_logger.info(
        "  Fetching: %s -> %s (last_run=%s, lookback=%dd)",
        start_date,
        end_date,
        last_history,
        lookback_days,
    )

    fetcher = (fetcher_factory or _default_fetcher_factory)()
    fetcher.fetch_all_stock_history(start_date, end_date, save_dir)
    fetcher.fetch_all_index_history(start_date, end_date, save_dir)
    (package_data_fn or package_data)(save_dir, send_buffer_dir)

    return {
        "start_date": start_date,
        "end_date": end_date,
        "last_end_date": last_end_date,
        "lookback_days": lookback_days,
        "save_dir": save_dir,
        "send_buffer_dir": send_buffer_dir,
    }


__all__ = [
    "DEFAULT_LAST_END_DATE",
    "fetch_and_package_market_data",
    "resolve_fetch_window",
]
