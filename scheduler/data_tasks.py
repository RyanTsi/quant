from __future__ import annotations

import logging

from quantcore.factory import build_data_service
from scheduler.decorator import task

logger = logging.getLogger("scheduler")


@task("fetch_data")
def fetch_data() -> None:
    service = build_data_service(refresh_settings=True)
    service.fetch_data(lookback_days=7)


@task("ingest_to_db")
def ingest_to_db() -> None:
    service = build_data_service(refresh_settings=True)
    service.ingest_to_db(delete_after_ingest=True)


@task("export_from_db")
def export_from_db() -> None:
    service = build_data_service(refresh_settings=True)
    service.export_from_db(start_date="2010-01-01")
