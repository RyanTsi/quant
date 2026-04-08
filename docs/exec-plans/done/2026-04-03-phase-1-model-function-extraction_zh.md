# 执行计划：阶段 1 将训练与预测能力抽取到 model_function

## 目标

在保持当前 runtime 行为不变的前提下，把 `alpha_models/*` 和
`runtime/adapters/modeling.py` 中可复用的训练、recorder 解析、预测以及训练后
分析逻辑抽取到 `model_function/`。

## 范围

- `model_function/`
- `runtime/adapters/modeling.py`
- `alpha_models/qlib_workflow.py`
- `scripts/view.py`
- `scripts/eval_test.py`
- `test/` 下相关测试
- 仅更新描述新归属关系所必需的文档

## 假设

- `alpha_models/workflow/runner.py` 在本阶段继续保留为底层 Qlib workflow
  runner。
- `predict` 与 `build_portfolio` 的 run-history 写入仍由 runtime services
  负责。
- 本阶段不改变训练模型配置、输出文件名或用户可见 CLI 命令。
- 可复用的模型域逻辑归属 `model_function/`，runtime 编排仍归属 `runtime/`。

## 步骤

1. 为训练工作流编排创建 model-function helper。
   - 增加可复用训练入口，负责解析配置来源、运行 workflow runner，并返回结构化
     训练元数据。
   - 将训练后可视化的复用逻辑迁移到模型域归属下，避免 workflow 代码再导入
     `scripts.view`。

2. 为预测与 recorder 访问创建 model-function helper。
   - 增加可复用的 recorder id 解析与训练模型加载 helper。
   - 增加与 notebook 概念一致的 Alpha158 / `TSDatasetH` 预测数据集装配
     helper。
   - 现有确定性 universe 选择逻辑继续保留在 `universe` 模块。

3. 重构包装层，让它们向下委托。
   - 让 `alpha_models/qlib_workflow.py` 成为新训练 helper 之上的薄入口。
   - 让 `runtime/adapters/modeling.py` 将预测相关的 Qlib 装配委托给
     `model_function/`。
   - 让 `scripts/view.py` 和 `scripts/eval_test.py` 在行为重叠的地方复用
     `model_function` helper，保持脚本层轻量。

4. 补齐或更新测试。
   - 为新的 `model_function` helper 添加直接测试。
   - 让 wrapper 测试聚焦委托关系与稳定的公共行为。
   - 保持或增强现有 runtime service 与 adapter contract 测试。

5. 更新 trace 文档。
   - 记录阶段日志与 reviewer 报告。
   - 只有当归属关系声明发生变化时才更新架构类文档。

## 验收标准

- `alpha_models/qlib_workflow.py` 不再导入 `scripts.view`。
- 预测数据集装配与 recorder 解析能力可在
  `runtime/adapters/modeling.py` 之外复用。
- 预测、训练入口和 view 生成的 runtime 行为通过测试保持不变。
- 改动后的模块更小，职责边界更清晰。

## 验证

- 新 `model_function` helper 的窄范围单元测试
- 现有 wrapper/service 测试，覆盖：
  - `test/test_qlib_workflow.py`
  - `test/test_model_pipeline_service.py`
  - `test/test_model_cli_wrappers.py`
  - `test/test_view_script.py`
  - `test/test_workflow_runner.py`
  - `test/test_modeling_adapter_contract.py`

## 回滚说明

- 如果抽取后不稳定，可在 `alpha_models/` 与
  `runtime/adapters/modeling.py` 保留兼容包装层，转发到旧实现形态，同时保留
  新测试。
