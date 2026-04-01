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
- `--run fetch|ingest|export|dump|train|predict|portfolio`
- `--status`

### 脚本兼容性

核心脚本继续支持 `python -m scripts.<name>` 调用方式。

## 3. 功能需求

### FR-1 配置

- 运行时配置需从 `.env` 读取并有默认值。
- cooldown 设置可在进程内通过环境变量刷新生效。

### FR-2 运行历史

- 任务运行记录必须持久化到 `.data/run_history.json`。
- 旧的 `utils.run_tracker` API 必须继续可用。

### FR-3 流水线执行

- 流水线按顺序执行任务。
- 任一任务失败会中断流水线。
- 相邻任务之间应用 cooldown。

### FR-4 入库安全

- ingest 默认不得删除源 CSV。
- 删除行为必须通过 `delete_after_ingest=True` 显式开启。

### FR-5 可测试性

- 重操作必须能在服务层边界被 mock。
- 项目必须包含完整单元测试与整体流程测试。

## 4. 非功能需求

- 保持模块边界清晰、依赖方向稳定。
- 保持关键导入路径向后兼容。
- 重构过程可回滚。

## 5. 验收标准

满足以下条件视为通过：

1. 现有 CLI 契约仍可用。
2. 新架构模块（`quantcore/*`）已进入主运行路径。
3. `conda quant` 环境下全量测试通过。
4. 中英文文档与实际实现一致。
