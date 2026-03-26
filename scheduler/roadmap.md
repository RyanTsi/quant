# Roadmap: scheduler/

## 1. Overview

Task orchestration: defines individual tasks, composes them into pipelines, runs on cron (via `main.py`).

## 2. Architecture

```
EVENING_PIPELINE   (18:15 weekdays): fetch_data → ingest_to_db
AFTERNOON_PIPELINE (14:00 weekdays): export_from_db → dump_to_qlib → predict
FULL_PIPELINE:                      all tasks end-to-end
```

## 3. File-Role Mapping

| File / Subdirectory | Role / Description |
| :--- | :--- |
| `decorator.py` | `@task` decorator — logging, timing, error catching |
| `data_tasks.py` | Data tasks: `fetch_data`, `ingest_to_db`, `export_from_db` |
| `model_tasks.py` | Model tasks: `dump_to_qlib`, `train_model`, `predict` |
| `pipelines.py` | Pipeline definitions + `run_pipeline` sequential runner |
| `__init__.py` | Re-exports public symbols |

## 5. Navigation

| If you want to... | Go to... |
| :--- | :--- |
| Add a new task with consistent logging | `decorator.py` |
| Change data ingest/export schedule logic | `data_tasks.py` |
| Change model train/predict/dump logic | `model_tasks.py` |
| Add/modify a pipeline (task ordering) | `pipelines.py` |
| See how cron is wired and triggered | `../main.py` |

## 6. Conventions

- Every task function is decorated with `@task("name")`.
- Tasks call `record_run()` on success for tracking.
- Pipelines are plain lists of task functions; `run_pipeline` runs them sequentially.
- `ingest_to_db` imports from `data_pipeline.ingest` (not `scripts`).
