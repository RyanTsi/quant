from __future__ import annotations

from collections.abc import Callable

from runtime.services import build_data_service, build_model_service


def _named_task(name: str) -> Callable[[Callable[[], None]], Callable[[], None]]:
    def decorator(fn: Callable[[], None]) -> Callable[[], None]:
        fn.task_name = name  # type: ignore[attr-defined]
        return fn

    return decorator


@_named_task("fetch_data")
def fetch_data() -> None:
    service = build_data_service(refresh_settings=True)
    service.fetch_data(lookback_days=7)


@_named_task("ingest_to_db")
def ingest_to_db() -> None:
    service = build_data_service(refresh_settings=True)
    service.ingest_to_db(delete_after_ingest=True)


@_named_task("export_from_db")
def export_from_db() -> None:
    service = build_data_service(refresh_settings=True)
    service.export_from_db(start_date="2010-01-01")


@_named_task("dump_to_qlib")
def dump_to_qlib() -> None:
    service = build_model_service(refresh_settings=True)
    service.dump_to_qlib()


@_named_task("predict")
def predict() -> None:
    service = build_model_service(refresh_settings=True)
    service.predict()


@_named_task("build_portfolio")
def build_portfolio() -> None:
    service = build_model_service(refresh_settings=True)
    service.build_portfolio()


@_named_task("train_model")
def train_model() -> None:
    service = build_model_service(refresh_settings=True)
    service.train_model()


__all__ = [
    "build_portfolio",
    "dump_to_qlib",
    "export_from_db",
    "fetch_data",
    "ingest_to_db",
    "predict",
    "train_model",
]
