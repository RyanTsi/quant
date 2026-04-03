"""
Quant System - Main Scheduler and CLI.

Usage:
    python main.py
    python main.py --run evening
    python main.py --run afternoon
    python main.py --run full
    python main.py --run fetch|ingest|export|dump|filter|train|predict|portfolio
    python main.py --status
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

from runtime.bootstrap import build_default_registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler("scheduler.log", encoding="utf-8")],
)
logger = logging.getLogger("main")


def setup_schedule():
    import schedule
    registry = build_default_registry()

    weekdays = [
        schedule.every().monday,
        schedule.every().tuesday,
        schedule.every().wednesday,
        schedule.every().thursday,
        schedule.every().friday,
    ]

    for day in weekdays:
        day.at("18:15").do(registry.run, "evening")
        day.at("14:00").do(registry.run, "afternoon")
    return schedule


def run_scheduler() -> None:
    sched = setup_schedule()
    logger.info("Scheduler started. Upcoming jobs:")
    for job in sched.get_jobs():
        logger.info("  %s", job)
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


def run_once(task_name: str) -> None:
    registry = build_default_registry()
    try:
        ok = registry.run(task_name)
    except KeyError as exc:
        print(str(exc))
        sys.exit(1)
    except Exception as exc:
        print(f"Task '{task_name}' failed: {exc}")
        sys.exit(1)

    if not ok:
        print(f"Pipeline '{task_name}' failed.")
        sys.exit(1)


def show_status() -> None:
    from runtime.runlog import load_run_history

    data = load_run_history()
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Quant System Scheduler")
    parser.add_argument(
        "--run",
        type=str,
        metavar="TASK",
        help="Run once: evening, afternoon, full, fetch, ingest, export, dump, filter, train, predict, portfolio",
    )
    parser.add_argument("--status", action="store_true", help="Show last run status for all tasks")
    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.run:
        run_once(args.run)
    else:
        run_scheduler()


if __name__ == "__main__":
    main()
