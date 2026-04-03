# 产品规格：Python Runtime V2

- 版本：v2
- 日期：2026-03-31
- 适用范围：Python 编排与运行时模块

## 1. 目标

1. 提供稳定、可测试的数据 -> 模型 -> 组合流水线编排能力。
2. 保持现有命令面兼容。
3. 支持使用轻量替身快速验证。

## 2. 用户侧契约

### CLI 兼容性

`main.py` 保持以下命令：
- `--run evening`
- `--run afternoon`
- `--run full`
- `--run fetch|ingest|export|dump|filter|train|predict|portfolio`
- `--status`

### 脚本兼容性

核心脚本继续支持 `python -m scripts.<name>` 调用方式。

## 3. 功能需求

### FR-1 配置

- 运行时配置需从 `.env` 读取并有默认值。
- `runtime.config` 是规范配置实现。
- cooldown 设置可在进程内通过环境变量刷新生效。

### FR-2 运行历史

- 任务运行记录必须持久化到 `.data/run_history.json`。
- `runtime.runlog.RunLogStore` 是规范运行历史存储实现。
- 新结构写入时仍需兼容旧扁平 run-history 文件的读取。
- 当前受支持的便捷 API 位于 `runtime.runlog`（`load_run_history`、`save_run_history`、`record_run`、`get_last_run`、`today`、`today_dash`）。

### FR-3 流水线执行

- 流水线按顺序执行任务。
- 任一任务失败会中断流水线。
- 单次 CLI 运行流水线时，失败必须向调用方暴露为非零退出语义。
- 相邻任务之间应用 cooldown。

### FR-4 入库安全

- 底层 ingest adapter 默认不得删除源 CSV。
- 破坏性删除行为必须在调用点通过 `delete_after_ingest=True` 显式开启。
- 面向操作的 runtime 路径可以为了“一次性消费打包缓冲区”而有意启用破坏性 ingest。
- 数据侧运行时行为（`fetch`、`ingest`、`export`）必须通过 runtime 持有的 adapters 执行，并返回显式结果元数据。

### FR-5 可测试性

- 重操作必须能在服务层边界被 mock。
- 只要 runtime adapter 已存在，数据侧和模型侧运行时行为都应通过 direct runtime adapter 执行，而不是 subprocess shell 或 service 内联编排。
- 项目必须包含完整单元测试与整体流程测试。

## 4. 非功能需求

- 保持模块边界清晰、依赖方向稳定。
- 保持 CLI 与脚本命令面的稳定，同时允许删除已经过时的 Python 导入兼容壳。
- 重构过程可回滚。

## 5. 验收标准

满足以下条件视为通过：

1. 现有 CLI 契约仍可用。
2. 新架构模块（`runtime/*`）已进入主运行路径。
3. 原先由兼容层持有的数据/模型 service、配置访问与运行历史辅助能力，已直接由 `runtime.services`、`runtime.config` 与 `runtime.runlog` 持有。
4. `fetch`、`ingest`、`export`、`dump`、`filter`、`predict`、`portfolio` 的工作流逻辑已由 runtime 统一管理。
5. `conda quant` 环境下全量测试通过。
6. 中英文文档与实际实现一致，并提供可实际使用的 runtime guide。
