# 执行计划：Python 运行时稳定性重构

- 日期：2026-04-01
- 任务 UUID：`70554382-1a2b-4e2b-b774-79d96320d03a`

## 目标
在不改动 `server/` 与 `news_module/` 的前提下，重建 Python 侧运行时架构，使系统在稳定性、可测试性、可维护性方面明显提升，并让日常使用路径更清晰可靠。

## 范围
- 范围内：
  - 以 typed execution context、typed task result、显式 ports、单一 orchestrator 为核心，替换当前较薄的 `quantcore` 编排层。
  - 将数据流、模型流、CLI 与调度入口统一到同一条执行栈上。
  - 对冗余或明显不好用的接口进行清理和删除。
  - 强化任务级、工作流级、失败路径级自动化测试。
  - 同步更新中英文文档、guide、计划和日志，确保与实现一致。
- 范围外：
  - `server/` C++ 网关改动。
  - `news_module/` 启用或重设计。

## 假设
- 当文档与代码冲突时，以当前代码与测试行为为准。
- 若新的接口设计能显著提升稳定性与可用性，则不需要保留向后兼容。
- `baostock`、网关 HTTP、Qlib/MLflow 等重依赖应通过 adapter 隔离，并尽量提供离线可验证测试。
- `conda quant` 是本次验证必须使用的运行环境。

## 步骤
1. 审计当前 Python 运行时，结合 subagent review 冻结目标架构、风险点与迁移阶段。
2. 引入新的 runtime 包，包含：
   - typed domain models
   - 结构化运行日志原语
   - port 接口
   - 单一 workflow orchestrator
3. 先迁移模型侧执行链路（`dump`、`train`、`predict`、`portfolio`），去掉基于 subprocess/script 的核心编排。
4. 再迁移数据侧执行链路（`fetch`、`ingest`、`export`），补齐明确的 schema 归一化与 artifact 处理。
5. 将 `main.py`、CLI 入口和调度路径切换到新的 orchestrator，并删除冗余旧编排代码。
6. 在 `conda quant` 中扩展验证：补齐聚焦单测、工作流集成测试与失败路径测试。
7. 输出新的双语 guide 和使用说明。
8. 当重构、文档、日志、critic 报告与验证全部完成后，将本计划移动到 `docs/exec-plans/done/`。

## 验收标准
- Python 运行时由单一 orchestrator 驱动，而不是分裂的 scheduler/script 路径。
- 任务边界显式且类型化，具备结构化 task result 与持久化运行元数据。
- 模型流和数据流在核心运行路径上不再依赖 subprocess 编排。
- 冗余旧编排代码被删除，或仅保留为非关键路径上的薄兼容层。
- 变更行为具备自动化测试覆盖，包含关键失败场景。
- 文档包含架构更新以及可实际使用的中英文 guide。
- 结构化日志具备 UUID 可追踪性，并附带另一 subagent 的 critic 报告。

## 回滚说明
- 如果切换后稳定性不达标，则回滚新的 runtime 包、入口改线以及相关测试/文档。
- 如果新的 orchestrator 未通过验收，则恢复原有 `quantcore` + `scheduler` 运行路径。
- 若执行完整回滚，需同时移除本 UUID 对应的计划/日志产物，并恢复相关导航或工作流状态文档。
