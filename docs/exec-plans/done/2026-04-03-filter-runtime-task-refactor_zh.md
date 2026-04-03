# 执行计划：Filter 运行时任务重构

- 状态：已于 2026-04-03 完成

## 目标

把训练股票池的主要构建流程从 `scripts/filter.py` 迁到 `model_function/`
中，保留脚本作为面向操作方的薄封装，并新增一个由 runtime 管理的 task，
让这条流程能够通过规范的 registry 调度。

## 范围

- `model_function/universe.py`
- `model_function/__init__.py`
- `runtime/services.py`
- `runtime/tasks.py`
- `runtime/bootstrap.py`
- `runtime/constants.py`
- `scripts/filter.py`
- 与新 runtime task 对应的聚焦测试和最小必要文档更新

## 假设

- 除非本任务中的测试明确要求调整，否则当前训练股票池行为仍然视为真值。
- 重构后 `scripts/filter.py` 仍需保持 CLI 可用。
- 新增 runtime task 只做增量扩展；除非显式接入，否则不改现有 pipeline。

## 步骤

1. 将按月滞后的训练股票池构建流程提取为可复用的 `model_function` helper。
2. 把 `scripts/filter.py` 重构为共享 helper 的薄包装。
3. 为这条 filter 流程增加 `ModelPipelineService` 入口，以及 runtime 的
   task/registry 映射。
4. 更新聚焦测试，覆盖模型 helper、runtime task 接线和脚本兼容性。
5. 更新最小必要文档和追踪产物；验证完成后把本计划移到
   `docs/exec-plans/done/`。

## 验收标准

- `scripts/filter.py` 不再承载主要训练股票池实现。
- 训练股票池核心构建路径位于 `model_function/`。
- `main.py --run filter` 能通过 runtime registry 正常调度。
- 覆盖本次重构的聚焦单元测试全部通过。

## 验证

- `python -m unittest test.test_model_function_universe test.test_filter_stocks test.test_model_cli_wrappers test.test_runtime_tasks test.test_runtime_bootstrap test.test_model_pipeline_service`
- `python -m unittest discover -s test -p 'test_*.py'`

## 回滚说明

- 如果提取后的 helper 造成行为偏差，优先恢复 `scripts/filter.py` 旧实现。
- 如果新 runtime task 的对外语义不稳定或容易误导，则移除其注册映射。
