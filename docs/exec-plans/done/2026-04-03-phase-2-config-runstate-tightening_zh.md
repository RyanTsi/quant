# 执行计划：阶段 2 配置与运行状态边界收紧

## 目标

通过把 runtime 归属的状态与 fallback 解析收回到显式的 runtime 侧 helper 和
service 中，减少模型流水线里对 `get_settings()`、`get_last_run()` 和
`record_run()` 的直接穿透访问。

## 范围

- `runtime/services.py`
- 新增的 runtime 归属模型状态 helper 模块
- `alpha_models/qlib_workflow.py`
- `runtime/adapters/modeling.py`
- `scripts/view.py`
- `scripts/eval_test.py`
- `scripts/build_portfolio.py`
- 聚焦测试，以及被直接影响的文档

## 假设

- Phase 1 在 `model_function/qlib.py` 中完成的 helper 抽取继续作为可复用的
  模型域边界。
- run-history 写入应尽量由 runtime services 负责。
- 面向操作的 CLI 契约保持稳定。
- 本阶段仍然通过保持 wrapper 入口不变来保证可回滚。

## 步骤

1. 引入 runtime 归属的模型状态解析 helper 边界。
   - 新增一个小型 runtime 模块，统一处理：
     - 训练 workflow 所需 settings 输入解析
     - 基于 env/runlog/settings 的最新训练 recorder identity 解析
     - 训练 run-history payload 写入

2. 收紧训练归属边界。
   - 让 `alpha_models/qlib_workflow.py` 成为不再直接写 `qlib_train`
     的 workflow wrapper。
   - 把 `qlib_train` 的写入迁回 `runtime.services.ModelPipelineService`。

3. 让 predict/view/eval/portfolio wrapper 使用 runtime 归属的状态解析。
   - 在可由 runtime helper 提供同等能力的地方，去掉脚本/adapter 对
     `get_last_run()` 的直接访问。
   - 让 `scripts.build_portfolio.py` 走 `ModelPipelineService` 路径，把正常操作
     路径上的组合 runlog 写入从 `runtime/adapters/modeling.py` 收回到 service。

4. 补齐或更新测试。
   - 为新的 runtime 侧 helper 模块增加直接测试。
   - 更新 wrapper/service 测试，覆盖新的委托关系和状态归属边界。

5. 更新文档与 trace。
   - 如果训练/view/eval 的归属描述有变化，同步更新架构/runtime 文档。

## 验收标准

- `alpha_models/qlib_workflow.py` 不再直接写 `qlib_train`。
- runtime 归属 helper 可以为 view/eval/predict/portfolio 提供 recorder/config
  fallback 解析。
- `runtime/adapters/modeling.py` 在正常操作路径上不再直接写 runlog 状态。
- 新的 runtime 状态边界和受影响 wrapper 都有测试保护。

## 验证

- 新 runtime 侧 helper 模块的直接测试
- 更新后的聚焦模型/runtime 测试，包括：
  - `test/test_model_pipeline_service.py`
  - `test/test_qlib_workflow.py`
  - `test/test_view_script.py`
  - `test/test_eval_test_script.py`
  - `test/test_model_cli_wrappers.py`
  - `test/test_modeling_adapter_contract.py`

## 回滚说明

- 保持入口签名稳定。
- 如果状态 helper 抽取带来过多扰动，可以保留兼容包装层，但仍应保留
  service 归属的 runlog 写入和直测。
