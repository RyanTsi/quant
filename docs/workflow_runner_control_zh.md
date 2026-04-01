# Workflow Runner 控制模式

`alpha_models/workflow/runner.py` 现在支持两种控制方式：

1. YAML 驱动（向后兼容）：
   - `QlibWorkflowRunner.run_from_yaml(...)`
2. 代码驱动（新增）：
   - `QlibWorkflowRunner.run_from_config(...)`

该设计参考了 `docs/detailed_workflow.ipynb` 中“按组件逐步控制”的工作流风格。

## 新增代码入口

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

## 运行时可控项

- `config_overrides`：在完整配置层深度合并（`qlib_init`、`task` 等）
- `task_overrides`：仅在 `task` 区域深度合并（`model`、`dataset`、`record`）
- `qlib_init_overrides`：仅在 `qlib_init` 区域深度合并
- `provider_uri_override`、`mlruns_uri`、`experiment_name`：运行时覆盖项

## 兼容性说明

- 现有 `run_from_yaml` 行为保留。
- YAML 与代码配置最终统一走同一个 `run(...)` 运行主路径。
