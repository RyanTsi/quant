from __future__ import annotations

import logging
import os

from runtime.config import get_settings
from runtime.constants import PIPELINE_TASK_NAMES
from runtime.orchestrator import SequentialOrchestrator
from runtime.registry import RuntimeRegistry


def cooldown_seconds() -> float:
    env = os.getenv("PIPELINE_COOLDOWN_SECONDS")
    if env is not None:
        return float(env)
    return float(get_settings(refresh=True).pipeline_cooldown_seconds)


def build_default_orchestrator() -> SequentialOrchestrator:
    return SequentialOrchestrator(logger=logging.getLogger("scheduler"), cooldown_provider=cooldown_seconds)


def build_default_registry() -> RuntimeRegistry:
    from runtime.tasks import (
        build_portfolio,
        dump_to_qlib,
        export_from_db,
        fetch_data,
        filter_training_universe,
        ingest_to_db,
        predict,
        train_model,
    )

    task_map = {
        "fetch": fetch_data,
        "ingest": ingest_to_db,
        "export": export_from_db,
        "dump": dump_to_qlib,
        "filter": filter_training_universe,
        "train": train_model,
        "predict": predict,
        "portfolio": build_portfolio,
    }
    return RuntimeRegistry(
        task_map=task_map,
        pipeline_map=dict(PIPELINE_TASK_NAMES),
        orchestrator=build_default_orchestrator(),
    )
