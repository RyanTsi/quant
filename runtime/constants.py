"""Stable runtime names for tasks and pipelines."""

TASK_NAMES: tuple[str, ...] = (
    "fetch",
    "ingest",
    "export",
    "dump",
    "train",
    "predict",
    "portfolio",
)

PIPELINE_TASK_NAMES: dict[str, list[str]] = {
    "evening": ["fetch", "ingest"],
    "afternoon": ["export", "dump", "predict", "portfolio"],
    "full": ["fetch", "ingest", "export", "dump", "train", "predict", "portfolio"],
}

PIPELINE_NAMES: tuple[str, ...] = tuple(PIPELINE_TASK_NAMES.keys())
