from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any


class RunHistoryStore:
    def __init__(self, path: str):
        self.path = path

    def load(self) -> dict[str, Any]:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save(self, data: dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, self.path)

    def record(self, task_name: str, **payload: Any) -> dict[str, Any]:
        data = self.load()
        entry = {"last_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **payload}
        data[task_name] = entry
        self.save(data)
        return entry

    def get(self, task_name: str) -> dict[str, Any] | None:
        return self.load().get(task_name)
