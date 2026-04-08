# 执行计划：架构整改路线图

## 目标

针对
`docs/logs/architecture-review-2026-04-03-16c156d1-a014-4fda-856a-1a1282797353.md`
中指出的高杠杆架构问题，采用小步、分阶段的重构方式逐步整改，在保持现有
runtime 行为稳定的前提下，改善模块边界、可审查性和运维清晰度。

## 范围

- `model_function/`、`runtime/adapters/modeling.py`、
  `runtime/services.py`、`alpha_models/*`、`scripts/view.py`、
  `scripts/eval_test.py` 等 Python 模型/runtime 边界相关改造
- 为保持代码与文档一致所需的测试与文档更新
- 每个阶段对应的子计划、结构化日志和 subagent review 闭环

## 假设

- 当前代码与测试仍然是行为真相来源。
- 除非某个阶段明确要求改变，否则重构应保持现有 CLI/runtime 契约不变。
- 第一阶段优先聚焦于把 `alpha_models/*` 与
  `runtime/adapters/modeling.py` 中可复用的训练/预测逻辑迁移到
  `model_function/`，并参考 `docs/detailed_workflow.ipynb` 中的概念拆分。
- 每个实现阶段都会重新创建两个全新的 GPT-5.4 subagent：
  一个负责编码，一个负责 review。

## 阶段

### 阶段 1：将训练与预测能力抽取到 model_function

- 把与 notebook 对应的 Qlib workflow 装配、recorder 解析、预测数据集构造
  迁移到 `model_function/`。
- 将 `runtime/adapters/modeling.py` 收敛成更薄的一层 runtime adapter。
- 将 `alpha_models/qlib_workflow.py` 收敛成更薄的一层 workflow 入口。
- 通过迁移训练后可视化复用逻辑，消除 workflow 反向依赖 `scripts/view.py`
  的层次倒置问题。

### 阶段 2：收紧配置与运行状态边界

- 减少 `alpha_models/*`、`runtime/adapters/*` 与 `scripts/*` 中对
  `get_settings()`、`get_last_run()`、`record_run()` 的直接穿透访问。
- 引入更清晰的 helper 边界来处理 recorder 查找与模型 artifact 解析。
- 在可行范围内，让 runtime services 继续充当主要的状态记录边界。

### 阶段 3：统一交易日与执行上下文语义

- 引入更清晰的执行上下文契约，区分自然日与交易日语义。
- 让预测与组合构建在读取上一日状态时遵循一致的交易日历规则。
- 在需要可重复重跑的路径上，用显式日期替代 wall-clock 回退。

### 阶段 4：运维语义加固与文档对账

- 在不进行大改写的前提下，处理 scheduler 与 destructive ingest 的后续问题。
- 清理 runtime-first 收敛后遗留的过时架构/设计/测试命名残留。
- 保持中英文文档与导航状态和新结构一致。

## 验收标准

- 每个阶段都有独立子计划、验证范围和结构化日志。
- 模型流水线的高责任逻辑被拆分到更小、更清晰的模块中，整体更易 review。
- workflow 到 script 的依赖倒置被移除。
- 新 helper 边界有测试保护，便于后续阶段继续迭代。
- 剩余问题被明确记录，而不是继续堆积在单一热点模块中。

## 回滚说明

- 每个阶段都应通过保持 wrapper 层契约稳定来保证可回滚。
- 如果某个阶段改动面过大，应先停在 helper 抽取和兼容包装层，不要强行一次做完。
