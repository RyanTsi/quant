# 执行计划：运行时兼容层清理

- 日期：2026-04-02
- 任务 UUID：`0c5ab334-1f6f-4f4e-be77-922c86df198d`

## 目标
删除 Python 侧剩余仅为过渡保留的兼容层，使主运行路径直接由规范 runtime 模块持有，而不再依赖 `quantcore`、`config.settings`、`utils.run_tracker`、`scheduler` 等旧包装层。

## 范围
- 范围内：
  - 删除当前仅用于兼容的 runtime shim / wrapper。
  - 将运行时分发、任务入口、脚本和测试改线到规范 runtime 模块。
  - 在安全前提下删除冗余旧导入路径、兼容包装层和 adapter-only 模块。
  - 完成最后一批仍在活跃使用的配置访问、run-history helper 和 service 归属切换。
  - 同步更新中英文文档、日志和计划，反映更精简的 post-compat 架构。
- 范围外：
  - `server/`
  - `news_module/`
  - 与本次兼容层清理无关的产品行为变更

## 假设
- 对这些 Python 导入路径，仓库内消费者不再要求向后兼容；只要规范 runtime 路径清晰且有测试覆盖即可。
- 当前代码与测试仍是行为事实来源。
- 本次清理应保留任务名、流水线名和脚本命令面，除非只是内部实现改线。
- 只要在同一改动中把直接消费者迁移到规范 runtime 模块，本任务可以一并删除剩余活跃兼容表面。

## 冻结边界
- 本任务删除：
  - `quantcore/__init__.py`
  - `quantcore/services/__init__.py`
  - `quantcore/services/data_service.py`
  - `quantcore/services/model_service.py`
  - `scheduler/data_tasks.py`
  - `scheduler/model_tasks.py`
  - `scheduler/pipelines.py`
  - `scheduler/decorator.py`
  - `scheduler/__init__.py`
  - `quantcore/factory.py`
  - `quantcore/pipeline.py`
  - `quantcore/settings.py`
  - `quantcore/history.py`
  - `quantcore/registry.py`
  - `data_pipeline/ingest.py`
  - `config/settings.py`
  - `utils/run_tracker.py`
  - 只服务于已删除旧脚本或兼容导入路径的遗留测试
- 本任务暂缓：
  - 与兼容层清理无关的更大范围运行时行为改动
  - 需要保留历史语境的旧计划 / 旧日志

## 步骤
1. 冻结兼容层删除边界，并记录“本轮删除”与“暂缓项”。
2. 将剩余 service、config consumer 和 run-history helper 归并到 `runtime/*`。
3. 将测试从旧导入目标迁移到 runtime 原生模块，同时保持行为契约不变。
4. 在 runtime 原生替代通过验证后，一次性删除 scheduler / quantcore / config / run-tracker / data_pipeline shim 链。
5. 删除仅服务于已删除旧脚本的遗留测试。
6. 在 `conda quant` 中重新执行定向测试与全量验证。
7. 更新双语文档/日志，并在 critic review 与验收完成后将本计划移入 `docs/exec-plans/done/`。

## 验收标准
- 运行时分发不再依赖仅用于兼容的 scheduler / task wrapper。
- `scheduler` / `quantcore` 中重复的 pipeline / orchestration wrapper 被删除，并由 runtime 原生模块接管。
- 活跃脚本与 workflow 不再依赖 `config/settings.py` 或 `utils/run_tracker.py`。
- 本任务删除的兼容模块在代码与测试中都不再被引用。
- 文档能够准确描述更精简的规范 runtime 架构，且中英文一致。
- `conda quant` 中全量测试通过。
- 结构化日志包含 UUID 追踪和 critic 报告。

## 回滚说明
- 如果直接切换破坏任务分发或脚本行为，则恢复被删除的兼容模块和旧运行时接线。
- 若回滚代码，也需一并回滚对应文档与日志更新。
