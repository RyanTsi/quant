from __future__ import annotations

import os

from quantcore.history import RunHistoryStore
from quantcore.services import DataPipelineService, ModelPipelineService
from quantcore.settings import get_settings


def build_history_store(*, refresh_settings: bool = False) -> RunHistoryStore:
    settings = get_settings(refresh=refresh_settings)
    return RunHistoryStore(os.path.join(settings.data_path, "run_history.json"))


def build_data_service(*, refresh_settings: bool = False) -> DataPipelineService:
    settings = get_settings(refresh=refresh_settings)
    history = build_history_store(refresh_settings=False)
    return DataPipelineService(settings, history=history)


def build_model_service(*, refresh_settings: bool = False) -> ModelPipelineService:
    settings = get_settings(refresh=refresh_settings)
    history = build_history_store(refresh_settings=False)
    return ModelPipelineService(settings, history=history)
