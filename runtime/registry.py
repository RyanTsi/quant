from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Sequence

from runtime.contracts import TaskSpec
from runtime.orchestrator import SequentialOrchestrator

TaskCallable = Callable[[], None]


def _default_orchestrator() -> SequentialOrchestrator:
    from runtime.bootstrap import cooldown_seconds

    return SequentialOrchestrator(logger=logging.getLogger("scheduler"), cooldown_provider=cooldown_seconds)


@dataclass(frozen=True)
class RuntimeRegistry:
    task_map: dict[str, TaskCallable]
    pipeline_map: dict[str, Sequence[str | TaskCallable]]
    orchestrator: SequentialOrchestrator = field(default_factory=_default_orchestrator)

    @classmethod
    def build_default(cls) -> "RuntimeRegistry":
        from runtime.bootstrap import build_default_registry

        registry = build_default_registry()
        if isinstance(registry, cls):
            return registry
        return cls(
            task_map=dict(registry.task_map),
            pipeline_map=dict(registry.pipeline_map),
            orchestrator=registry.orchestrator,
        )

    @staticmethod
    def _task_spec(default_name: str, task_fn: TaskCallable) -> TaskSpec:
        return TaskSpec(name=getattr(task_fn, "task_name", default_name), fn=task_fn)

    def _resolve_pipeline(self, name: str) -> list[TaskSpec]:
        specs: list[TaskSpec] = []
        for task_item in self.pipeline_map[name]:
            if isinstance(task_item, str):
                task_fn = self.task_map.get(task_item)
                if task_fn is None:
                    raise KeyError(f"Pipeline '{name}' references unknown task '{task_item}'.")
                specs.append(self._task_spec(task_item, task_fn))
            else:
                specs.append(self._task_spec(task_item.__name__, task_item))
        return specs

    def run(self, name: str) -> bool:
        if name in self.pipeline_map:
            specs = self._resolve_pipeline(name)
            result = self.orchestrator.run_pipeline(name, specs)
            return bool(result.success)

        if name in self.task_map:
            self.orchestrator.run_task(self._task_spec(name, self.task_map[name]), raise_on_failure=True)
            return True

        available = list(self.pipeline_map.keys()) + list(self.task_map.keys())
        raise KeyError(f"Unknown task '{name}'. Available: {', '.join(available)}")
