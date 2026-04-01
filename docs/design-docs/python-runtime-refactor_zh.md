# 设计文档：Python 运行时重构

- 日期：2026-03-31
- 范围：仅 Python 侧（不包含 `server/` 与 `news_module/`）

## 问题背景

重构前，编排职责分散在：
- scheduler 任务模块，
- 独立脚本，
- 直接产生副作用的工具函数中。

这导致逻辑重复、测试困难、配置与状态耦合高。

## 设计目标

1. 将运行时编排集中到可复用核心层。
2. 保持现有 CLI/任务契约兼容。
3. 让重副作用操作可被测试打桩。
4. 保持改造低风险且可回滚。

## 目标架构

### 核心运行时层（`quantcore/`）

- `settings.py`
  - 类型化运行配置（`AppSettings`）
  - 支持缓存与刷新读取
- `history.py`
  - JSON 持久化运行历史（`RunHistoryStore`）
  - 保存时使用原子替换
- `pipeline.py`
  - 统一流水线执行器（`PipelineRunner`）
- `services/data_service.py`
  - 抓取、入库、导出编排
- `services/model_service.py`
  - dump、训练、预测、组合编排
- `registry.py`
  - CLI 分发用任务/流水线注册表

### 兼容层

- `config/settings.py`
  - 对旧导入路径暴露新配置模型
- `utils/run_tracker.py`
  - 保留旧 API，内部委托给 `RunHistoryStore`

### 适配层

- `scheduler/*`
  - 任务函数变为薄服务调用
  - 流水线统一执行，cooldown 改为运行时读取
- `main.py`
  - 单次执行改为注册表驱动分发

## 关键行为变化

1. 每次运行都可读取最新 `PIPELINE_COOLDOWN_SECONDS`。
2. ingest 默认不删除本地 CSV（`delete_after_ingest=False`），由调用方显式开启。
3. 副作用集中在服务层，边界更清晰。

## 测试策略

单元测试覆盖：
- 核心配置
- 数据/模型服务
- 注册表与分发
- 既有工具与领域模块

整体流程测试覆盖：
- 使用 mock 服务验证全流水线执行顺序
- 覆盖 `main.run_once -> registry` 分发链路

重操作策略：
- 全市场抓取与真实训练不在自动化测试中直跑，统一 mock。

## 备选方案对比

1. 全模块重命名并迁移全部路径
   - 放弃原因：改动面过大，兼容性风险高。
2. 仅修补当前失败测试
   - 放弃原因：无法解决结构与可维护性问题。

## 回滚方案

回滚以下改动：
- `quantcore/`
- scheduler/config/utils/main 的重构文件
- 本次新增测试与文档
