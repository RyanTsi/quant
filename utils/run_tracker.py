import json
import os
from datetime import datetime

_tracker_file = None


def init(data_path: str):
    """Set the directory where run_history.json is stored."""
    global _tracker_file
    _tracker_file = os.path.join(data_path, "run_history.json")


def _get_tracker_file() -> str:
    global _tracker_file
    if _tracker_file is None:
        from config.settings import settings
        init(settings.data_path)
    return _tracker_file


def _load():
    path = _get_tracker_file()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save(data):
    path = _get_tracker_file()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
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


def today() -> str:
    return datetime.now().strftime("%Y%m%d")


def today_dash() -> str:
    return datetime.now().strftime("%Y-%m-%d")
