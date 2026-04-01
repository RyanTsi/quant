from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from quantcore.history import RunHistoryStore

_tracker_file: str | None = None


def init(data_path: str) -> None:
    """Set the directory where run_history.json is stored."""
    global _tracker_file
    _tracker_file = os.path.join(data_path, "run_history.json")


def _get_tracker_file() -> str:
    global _tracker_file
    if _tracker_file is None:
        from config.settings import settings

        init(settings.data_path)
    return str(_tracker_file)


def _store() -> RunHistoryStore:
    return RunHistoryStore(_get_tracker_file())


def _load() -> dict[str, Any]:
    return _store().load()


def _save(data: dict[str, Any]) -> None:
    _store().save(data)


def record_run(task_name: str, **kwargs: Any) -> dict[str, Any]:
    """Record a task run with timestamp and optional metadata."""
    return _store().record(task_name, **kwargs)


def get_last_run(task_name: str) -> dict[str, Any] | None:
    """Get the last run info for a task. Returns None if never run."""
    return _store().get(task_name)


def today() -> str:
    return datetime.now().strftime("%Y%m%d")


def today_dash() -> str:
    return datetime.now().strftime("%Y-%m-%d")
