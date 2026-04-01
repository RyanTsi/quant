from __future__ import annotations

from quantcore.factory import build_model_service
from scheduler.decorator import task


@task("dump_to_qlib")
def dump_to_qlib() -> None:
    service = build_model_service(refresh_settings=True)
    service.dump_to_qlib()


@task("predict")
def predict() -> None:
    service = build_model_service(refresh_settings=True)
    service.predict()


@task("build_portfolio")
def build_portfolio() -> None:
    service = build_model_service(refresh_settings=True)
    service.build_portfolio()


@task("train_model")
def train_model() -> None:
    service = build_model_service(refresh_settings=True)
    service.train_model()
