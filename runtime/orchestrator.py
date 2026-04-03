from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Sequence

from runtime.contracts import PipelineResult, TaskResult, TaskSpec
from runtime.ports import CooldownProvider


class SequentialOrchestrator:
    """Sequential task runner with shared logging/cooldown semantics."""

    def __init__(self, *, logger: logging.Logger, cooldown_provider: CooldownProvider):
        self._logger = logger
        self._cooldown_provider = cooldown_provider

    def run_task(self, task: TaskSpec, *, raise_on_failure: bool = False) -> TaskResult:
        started_at = datetime.now()
        self._logger.info("[%s] started at %s", task.name, started_at.strftime("%H:%M:%S"))
        try:
            task.fn()
            finished_at = datetime.now()
            result = TaskResult(name=task.name, success=True, started_at=started_at, finished_at=finished_at)
            self._logger.info("[%s] finished in %.1fs", task.name, result.duration_seconds)
            return result
        except Exception as exc:
            finished_at = datetime.now()
            result = TaskResult(
                name=task.name,
                success=False,
                started_at=started_at,
                finished_at=finished_at,
                error=str(exc),
            )
            self._logger.exception("[%s] failed", task.name)
            if raise_on_failure:
                raise
            return result

    def run_pipeline(self, name: str, tasks: Sequence[TaskSpec]) -> PipelineResult:
        self._logger.info("%s", "=" * 50)
        self._logger.info("Pipeline started at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self._logger.info("Pipeline: %s", name)
        self._logger.info("Tasks: %s", [t.name for t in tasks])
        self._logger.info("%s", "=" * 50)

        cooldown_s = max(0.0, float(self._cooldown_provider()))
        task_results: list[TaskResult] = []
        for idx, task in enumerate(tasks):
            result = self.run_task(task, raise_on_failure=False)
            task_results.append(result)
            if not result.success:
                self._logger.error("Pipeline aborted due to task failure: %s", task.name)
                self._logger.info("Pipeline stopped at %s\n", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                return PipelineResult(name=name, success=False, tasks=tuple(task_results))

            if idx < len(tasks) - 1 and cooldown_s > 0:
                self._logger.info("Cooldown %.1fs before next task...", cooldown_s)
                time.sleep(cooldown_s)

        self._logger.info("Pipeline completed at %s\n", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        return PipelineResult(name=name, success=True, tasks=tuple(task_results))
