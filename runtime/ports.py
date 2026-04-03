from __future__ import annotations

from typing import Any, Protocol


class CooldownProvider(Protocol):
    def __call__(self) -> float: ...


class RunLogPort(Protocol):
    def load(self) -> dict[str, Any]: ...

    def save(self, data: dict[str, Any]) -> None: ...

    def record(self, task_name: str, **payload: Any) -> dict[str, Any]: ...

    def get(self, task_name: str) -> dict[str, Any] | None: ...
