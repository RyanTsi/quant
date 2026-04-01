# Workflow Runner Control Modes

`alpha_models/workflow/runner.py` now supports two control modes:

1. YAML-driven (backward compatible):
   - `QlibWorkflowRunner.run_from_yaml(...)`
2. Code-driven (new):
   - `QlibWorkflowRunner.run_from_config(...)`

This follows the component-by-component control style shown in `docs/detailed_workflow.ipynb`.

## New Programmatic Entry

```python
from alpha_models.workflow.runner import QlibWorkflowRunner

cfg = {
    "qlib_init": {"provider_uri": ".data/qlib_data", "region": "cn"},
    "task": {
        "model": {...},
        "dataset": {...},
        "record": [...],
    },
}

runner = QlibWorkflowRunner()
result = runner.run_from_config(
    config=cfg,
    source_label="python_config",
    task_overrides={"model": {"kwargs": {"dropout": 0.2}}},
)
```

## Runtime Control Knobs

- `config_overrides`: deep-merge at full config level (`qlib_init`, `task`, etc.)
- `task_overrides`: deep-merge only in task section (`model`, `dataset`, `record`)
- `qlib_init_overrides`: deep-merge only in qlib init section
- `provider_uri_override`, `mlruns_uri`, `experiment_name`: runtime-only overrides

## Compatibility

- Existing `run_from_yaml` behavior is kept.
- Internally both YAML and code configs go through the same `run(...)` path.
