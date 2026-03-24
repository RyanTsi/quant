# scheduler/

Task orchestration: defines individual tasks, composes them into pipelines, runs on cron.

## Files

| File             | Role                                                  |
|------------------|-------------------------------------------------------|
| `decorator.py`   | `@task` decorator — logging, timing, error catching   |
| `data_tasks.py`  | Data tasks: `fetch_data`, `ingest_to_db`, `export_from_db` |
| `model_tasks.py` | Model tasks: `dump_to_qlib`, `train_model`, `predict` |
| `pipelines.py`   | Pipeline definitions + `run_pipeline` sequential runner |
| `__init__.py`    | Re-exports all public symbols                         |

## Pipeline Structure

```
EVENING_PIPELINE  (18:15 weekdays):  fetch_data → ingest_to_db
AFTERNOON_PIPELINE (14:00 weekdays): export_from_db → dump_to_qlib → predict
FULL_PIPELINE:                       all tasks end-to-end
```

## Conventions

- Every task function is decorated with `@task("name")`.
- Tasks call `record_run()` on success for tracking.
- Pipelines are plain lists of task functions; `run_pipeline` runs them sequentially.
- `ingest_to_db` imports from `data_pipeline.ingest` (not `scripts`).

## See Also

- `main.py` — wires pipelines to `schedule` library
- `data_pipeline/` — underlying data operations
- `alpha_models/` — underlying model operations
