import logging
import os

from scheduler.decorator import task
from utils.run_tracker import record_run, today_dash
from config.settings import settings

logger = logging.getLogger("scheduler")


@task("dump_to_qlib")
def dump_to_qlib():
    """Convert CSV data to Qlib binary format."""
    import subprocess
    import sys

    csv_dir = settings.receive_buffer_path
    qlib_dir = settings.qlib_data_path

    if not os.path.isdir(csv_dir) or not os.listdir(csv_dir):
        logger.warning("  No CSV data in db_export/, skipping.")
        return

    logger.info(f"  Source: {csv_dir}")
    logger.info(f"  Target: {qlib_dir}")

    subprocess.run(
        [
            sys.executable,
            "scripts/dump_bin.py",
            "dump_all",
            f"--data_path={csv_dir}",
            f"--qlib_dir={qlib_dir}",
            "--include_fields=open,high,low,close,volume,amount,turn,isST,factor",
            "--file_suffix=.csv",
        ],
        check=True,
    )

    record_run("dump_to_qlib", csv_dir=csv_dir, qlib_dir=qlib_dir)


@task("predict")
def predict():
    """Generate stock predictions using the latest trained model."""
    import subprocess
    import sys
    subprocess.run(
        [sys.executable, "-m", "scripts.predict"],
        check=True,
    )
    record_run("predict", date=today_dash())


@task("train_model")
def train_model():
    """Train the Transformer model via Qlib workflow."""
    from alpha_models.qlib_workflow import main as qlib_main
    qlib_main()
    record_run("train_model", date=today_dash())
