"""
Quant System - Main Scheduler

Usage:
    python main.py                     # Start the scheduler (daily scheduled tasks)
    python main.py --run evening       # Run the evening pipeline once (fetch + ingest)
    python main.py --run afternoon     # Run the afternoon pipeline once (export + dump + predict)
    python main.py --run full          # Run full pipeline once (fetch, ingest, export, dump, train, predict)
    python main.py --run fetch         # Run a single task and exit
    python main.py --run ingest
    python main.py --run export
    python main.py --run dump
    python main.py --run train
    python main.py --run predict
    python main.py --status            # Show last run status for all tasks
"""

import argparse
import logging
import time
import signal
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scheduler.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("main")


def setup_schedule():
    import schedule
    from scheduler.pipelines import (
        EVENING_PIPELINE, AFTERNOON_PIPELINE, run_pipeline,
    )

    weekdays = [
        schedule.every().monday,
        schedule.every().tuesday,
        schedule.every().wednesday,
        schedule.every().thursday,
        schedule.every().friday,
    ]

    for day in weekdays:
        day.at("18:15").do(run_pipeline, EVENING_PIPELINE)
        day.at("14:00").do(run_pipeline, AFTERNOON_PIPELINE)

    # TODO: Weekly model retrain — enable when retraining workflow is ready
    # from scheduler.pipelines import WEEKLY_RETRAIN_PIPELINE
    # schedule.every().saturday.at("02:00").do(run_pipeline, WEEKLY_RETRAIN_PIPELINE)

    return schedule


def run_scheduler():
    sched = setup_schedule()

    logger.info("Scheduler started. Upcoming jobs:")
    for job in sched.get_jobs():
        logger.info(f"  {job}")
    logger.info("Press Ctrl+C to stop.\n")

    running = True
    def handle_stop(signum, frame):
        nonlocal running
        logger.info("Shutting down scheduler...")
        running = False

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    while running:
        sched.run_pending()
        time.sleep(30)

    logger.info("Scheduler stopped.")


def run_once(task_name: str):
    from scheduler.data_tasks import fetch_data, ingest_to_db, export_from_db
    from scheduler.model_tasks import dump_to_qlib, train_model, predict
    from scheduler.pipelines import (
        EVENING_PIPELINE, AFTERNOON_PIPELINE, FULL_PIPELINE,
        run_pipeline,
    )

    task_map = {
        "fetch":   fetch_data,
        "ingest":  ingest_to_db,
        "export":  export_from_db,
        "dump":    dump_to_qlib,
        "train":   train_model,
        "predict": predict,
    }

    pipeline_map = {
        "evening":   EVENING_PIPELINE,
        "afternoon": AFTERNOON_PIPELINE,
        "full":      FULL_PIPELINE,
    }

    if task_name in pipeline_map:
        run_pipeline(pipeline_map[task_name])
    elif task_name in task_map:
        task_map[task_name]()
    else:
        all_names = list(pipeline_map.keys()) + list(task_map.keys())
        print(f"Unknown task: '{task_name}'")
        print(f"Available: {', '.join(all_names)}")
        sys.exit(1)


def show_status():
    from utils.run_tracker import _load

    data = _load()
    if not data:
        print("No run history found.")
        return

    print(f"\n{'Task':<20} {'Last Run':<22} {'Details'}")
    print("-" * 70)
    for task_name, info in data.items():
        last_run = info.get("last_run", "?")
        details = {k: v for k, v in info.items() if k not in ("last_run", "last_date")}
        detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
        print(f"{task_name:<20} {last_run:<22} {detail_str}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Quant System Scheduler")
    parser.add_argument("--run", type=str, metavar="TASK",
                        help="Run a task/pipeline once: evening, afternoon, full, "
                             "fetch, ingest, export, dump, train, predict")
    parser.add_argument("--status", action="store_true",
                        help="Show last run status for all tasks")
    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.run:
        run_once(args.run)
    else:
        run_scheduler()


if __name__ == "__main__":
    main()
