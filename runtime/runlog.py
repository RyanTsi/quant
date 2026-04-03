from __future__ import annotations

import contextlib
import json
import os
from datetime import datetime
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None


SCHEMA_VERSION = 2


class RunLogStore:
    """Run-history store with legacy-read compatibility and atomic writes."""

    def __init__(self, path: str):
        self.path = path

    @property
    def _parent_dir(self) -> str:
        return os.path.dirname(self.path) or "."

    @property
    def _lock_path(self) -> str:
        return f"{self.path}.lock"

    @staticmethod
    def _empty_document() -> dict[str, Any]:
        return {
            "_meta": {"schema_version": SCHEMA_VERSION},
            "tasks": {},
        }

    @staticmethod
    def _normalize(raw: Any) -> dict[str, Any]:
        doc = RunLogStore._empty_document()
        if not isinstance(raw, dict):
            return doc

        def _coerce_schema_version(value: Any) -> int:
            try:
                parsed = int(str(value).strip())
            except (TypeError, ValueError):
                return SCHEMA_VERSION
            return parsed if parsed > 0 else SCHEMA_VERSION

        tasks_raw = raw.get("tasks")
        if isinstance(tasks_raw, dict):
            meta = raw.get("_meta")
            if isinstance(meta, dict):
                doc["_meta"].update(meta)
            doc["_meta"]["schema_version"] = _coerce_schema_version(doc["_meta"].get("schema_version"))
            doc["tasks"] = {
                str(task_name): dict(task_entry)
                for task_name, task_entry in tasks_raw.items()
                if isinstance(task_entry, dict)
            }
            return doc

        # Legacy format: top-level map where each key is a task name.
        doc["_meta"]["migrated_from"] = "legacy_flat"
        doc["tasks"] = {
            str(task_name): dict(task_entry)
            for task_name, task_entry in raw.items()
            if isinstance(task_entry, dict)
        }
        return doc

    @contextlib.contextmanager
    def _locked_file(self):
        os.makedirs(self._parent_dir, exist_ok=True)
        with open(self._lock_path, "a+", encoding="utf-8") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield lock_file
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _load_document_unlocked(self) -> dict[str, Any]:
        if not os.path.exists(self.path):
            return self._empty_document()

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except json.JSONDecodeError:
            return self._empty_document()
        return self._normalize(raw)

    def _load_document(self) -> dict[str, Any]:
        with self._locked_file():
            return self._load_document_unlocked()

    def load(self) -> dict[str, Any]:
        return self._load_document()["tasks"]

    def _write_document_unlocked(self, data: dict[str, Any]) -> None:
        os.makedirs(self._parent_dir, exist_ok=True)
        normalized = self._normalize({"tasks": data})
        normalized["_meta"]["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        tmp_path = f"{self.path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(normalized, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.path)

    def save(self, data: dict[str, Any]) -> None:
        with self._locked_file():
            self._write_document_unlocked(data)

    def record(self, task_name: str, **payload: Any) -> dict[str, Any]:
        with self._locked_file():
            data = self._load_document_unlocked()["tasks"]
            entry = {"last_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **payload}
            data[task_name] = entry
            self._write_document_unlocked(data)
            return entry

    def get(self, task_name: str) -> dict[str, Any] | None:
        return self.load().get(task_name)


def _default_store(*, path: str | None = None, refresh_settings: bool = False) -> RunLogStore:
    if path is not None:
        return RunLogStore(path)

    from runtime.config import get_settings

    settings = get_settings(refresh=refresh_settings)
    return RunLogStore(os.path.join(settings.data_path, "run_history.json"))


def load_run_history(*, path: str | None = None, refresh_settings: bool = False) -> dict[str, Any]:
    return _default_store(path=path, refresh_settings=refresh_settings).load()


def save_run_history(
    data: dict[str, Any],
    *,
    path: str | None = None,
    refresh_settings: bool = False,
) -> None:
    _default_store(path=path, refresh_settings=refresh_settings).save(data)


def record_run(
    task_name: str,
    *,
    path: str | None = None,
    refresh_settings: bool = False,
    **payload: Any,
) -> dict[str, Any]:
    return _default_store(path=path, refresh_settings=refresh_settings).record(task_name, **payload)


def get_last_run(
    task_name: str,
    *,
    path: str | None = None,
    refresh_settings: bool = False,
) -> dict[str, Any] | None:
    return _default_store(path=path, refresh_settings=refresh_settings).get(task_name)


def today() -> str:
    return datetime.now().strftime("%Y%m%d")


def today_dash() -> str:
    return datetime.now().strftime("%Y-%m-%d")


__all__ = [
    "RunLogStore",
    "SCHEMA_VERSION",
    "get_last_run",
    "load_run_history",
    "record_run",
    "save_run_history",
    "today",
    "today_dash",
]
