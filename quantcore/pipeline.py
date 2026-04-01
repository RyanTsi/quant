from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Sequence


@dataclass(frozen=True)
class TaskSpec:
    name: str
    fn: Callable[[], None]


class PipelineRunner:
    def __init__(self, *, logger: logging.Logger, cooldown_provider: Callable[[], float]):
        self._logger = logger
        self._cooldown_provider = cooldown_provider

    def run(self, tasks: Sequence[TaskSpec]) -> bool:
        self._logger.info("%s", "=" * 50)
        self._logger.info("Pipeline started at %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self._logger.info("Tasks: %s", [t.name for t in tasks])
        self._logger.info("%s", "=" * 50)

        cooldown_s = max(0.0, float(self._cooldown_provider()))
        for i, task in enumerate(tasks):
            try:
                task.fn()
            except Exception as exc:
                self._logger.error("Pipeline aborted due to task failure: %s", task.name)
                self._logger.info("Pipeline stopped at %s\n", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                self._logger.debug("Task %s error: %s", task.name, exc)
                return False

            if i < len(tasks) - 1 and cooldown_s > 0:
                self._logger.info("Cooldown %.1fs before next task...", cooldown_s)
                time.sleep(cooldown_s)

        self._logger.info("Pipeline completed at %s\n", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        return True
