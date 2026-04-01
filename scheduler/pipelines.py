from __future__ import annotations

import logging
import os
from collections.abc import Callable

from quantcore.pipeline import PipelineRunner, TaskSpec
from quantcore.settings import get_settings
from scheduler.data_tasks import export_from_db, fetch_data, ingest_to_db
from scheduler.model_tasks import build_portfolio, dump_to_qlib, predict, train_model

logger = logging.getLogger("scheduler")

EVENING_PIPELINE = [fetch_data, ingest_to_db]
AFTERNOON_PIPELINE = [export_from_db, dump_to_qlib, predict, build_portfolio]
FULL_PIPELINE = [fetch_data, ingest_to_db, export_from_db, dump_to_qlib, train_model, predict, build_portfolio]


def _cooldown_seconds() -> float:
    env = os.getenv("PIPELINE_COOLDOWN_SECONDS")
    if env is not None:
        return float(env)
    return float(get_settings(refresh=True).pipeline_cooldown_seconds)


def run_pipeline(pipeline: list[Callable[[], None]]) -> bool:
    specs = [TaskSpec(name=getattr(task_func, "task_name", task_func.__name__), fn=task_func) for task_func in pipeline]
    runner = PipelineRunner(logger=logger, cooldown_provider=_cooldown_seconds)
    return runner.run(specs)
