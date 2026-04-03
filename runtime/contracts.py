from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

TaskCallable = Callable[[], None]


@dataclass(frozen=True)
class RunRequest:
    name: str


@dataclass(frozen=True)
class TaskSpec:
    name: str
    fn: TaskCallable


@dataclass(frozen=True)
class TaskResult:
    name: str
    success: bool
    started_at: datetime
    finished_at: datetime
    error: str | None = None

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()


@dataclass(frozen=True)
class PipelineResult:
    name: str
    success: bool
    tasks: tuple[TaskResult, ...] = field(default_factory=tuple)
