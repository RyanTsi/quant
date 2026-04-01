from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from scheduler.data_tasks import export_from_db, fetch_data, ingest_to_db
from scheduler.model_tasks import build_portfolio, dump_to_qlib, predict, train_model
from scheduler.pipelines import AFTERNOON_PIPELINE, EVENING_PIPELINE, FULL_PIPELINE, run_pipeline


@dataclass(frozen=True)
class RuntimeRegistry:
    task_map: dict[str, Callable[[], None]]
    pipeline_map: dict[str, list[Callable[[], None]]]

    @classmethod
    def build_default(cls) -> "RuntimeRegistry":
        task_map = {
            "fetch": fetch_data,
            "ingest": ingest_to_db,
            "export": export_from_db,
            "dump": dump_to_qlib,
            "train": train_model,
            "predict": predict,
            "portfolio": build_portfolio,
        }
        pipeline_map = {
            "evening": EVENING_PIPELINE,
            "afternoon": AFTERNOON_PIPELINE,
            "full": FULL_PIPELINE,
        }
        return cls(task_map=task_map, pipeline_map=pipeline_map)

    def run(self, name: str) -> None:
        if name in self.pipeline_map:
            run_pipeline(self.pipeline_map[name])
            return
        if name in self.task_map:
            self.task_map[name]()
            return
        available = list(self.pipeline_map.keys()) + list(self.task_map.keys())
        raise KeyError(f"Unknown task '{name}'. Available: {', '.join(available)}")
