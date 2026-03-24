import json
import os
from datetime import datetime
from config.settings import settings

TRACKER_FILE = os.path.join(settings.data_path, "run_history.json")


def _load():
    if os.path.exists(TRACKER_FILE):
        with open(TRACKER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save(data):
    os.makedirs(os.path.dirname(TRACKER_FILE), exist_ok=True)
    with open(TRACKER_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def record_run(task_name: str, **kwargs):
    """Record a task run with timestamp and optional metadata."""
    data = _load()
    entry = {
        "last_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **kwargs,
    }
    data[task_name] = entry
    _save(data)
    return entry


def get_last_run(task_name: str):
    """Get the last run info for a task. Returns None if never run."""
    data = _load()
    return data.get(task_name)


def get_last_date(task_name: str, default: str = "20100101") -> str:
    """Get the last_date from the last run of a task, or *default* if never run."""
    entry = get_last_run(task_name)
    if entry is None:
        return default
    return entry.get("last_date", default)


def today() -> str:
    return datetime.now().strftime("%Y%m%d")


def today_dash() -> str:
    return datetime.now().strftime("%Y-%m-%d")
