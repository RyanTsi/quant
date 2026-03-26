import logging
import time
from datetime import datetime

from scheduler.data_tasks import fetch_data, ingest_to_db, export_from_db
from scheduler.model_tasks import dump_to_qlib, predict, train_model
from scheduler.decorator import TaskFailed
from config.settings import settings

logger = logging.getLogger("scheduler")

# 18:15 — fetch from network + ingest to DB
EVENING_PIPELINE = [
    fetch_data,
    ingest_to_db,
]

# 14:00 — pull from DB, update qlib data, predict
AFTERNOON_PIPELINE = [
    export_from_db,
    dump_to_qlib,
    predict,
]

# TODO: Weekly retrain pipeline — schedule for e.g. Saturday 02:00
# Blocked: model retraining workflow not yet finalized.
# When ready, uncomment and wire up in main.py setup_schedule().
#
# WEEKLY_RETRAIN_PIPELINE = [
#     fetch_data,
#     ingest_to_db,
#     dump_to_qlib,
#     train_model,
#     predict,
# ]

# Export DB -> per-symbol CSVs under .data/db_export before dump_to_qlib (otherwise dump is a no-op).
FULL_PIPELINE = [
    fetch_data,
    ingest_to_db,
    export_from_db,
    dump_to_qlib,
    train_model,
    predict,
]


def run_pipeline(pipeline: list):
    """Run a list of tasks sequentially."""
    logger.info(f"{'='*50}")
    logger.info(f"Pipeline started at {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"Tasks: {[t.task_name for t in pipeline]}")
    logger.info(f"{'='*50}")

    cooldown_s = float(settings.pipeline_cooldown_seconds)
    for i, task_func in enumerate(pipeline):
        try:
            task_func()
        except TaskFailed as e:
            logger.error(f"Pipeline aborted due to task failure: {e.task_name}")
            logger.info(f"Pipeline stopped at {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            return
        if i < len(pipeline) - 1 and cooldown_s > 0:
            logger.info(f"Cooldown {cooldown_s:.1f}s before next task...")
            time.sleep(cooldown_s)
    logger.info(f"Pipeline completed at {datetime.now():%Y-%m-%d %H:%M:%S}\n")
