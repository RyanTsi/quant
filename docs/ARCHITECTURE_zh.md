# QuantFrame 架构说明（Python 侧重构后）

本文描述本次重构后的当前架构（不包含 `server/`，且 `news_module/` 继续保持弃用隔离状态）。

## 1. 系统边界

运行时分为三层边界：

1. Python 编排与建模层（`main.py`、`quantcore/`、`scheduler/`、`scripts/`）
2. C++ 网关层（`server/`）
3. TimescaleDB 存储层（由网关代理访问）

硬约束：
- 生产链路中，Python 不直接访问 PostgreSQL。
- `news_module/` 不参与主流程。

## 2. 分层架构

Python 侧改为清晰分层：

1. 核心运行时层（`quantcore/`）
   - `settings.py`：统一环境与路径配置（`AppSettings`）
   - `history.py`：运行历史持久化（`RunHistoryStore`）
   - `pipeline.py`：通用流水线执行器（`PipelineRunner`）
   - `services/`：数据与模型服务编排
   - `registry.py`：任务/流水线注册与分发

2. 调度适配层（`scheduler/`）
   - `decorator.py`：任务包装与失败标准化
   - `data_tasks.py`、`model_tasks.py`：薄适配层，调用核心服务
   - `pipelines.py`：流水线定义与运行入口

3. CLI/脚本适配层（`main.py`、`scripts/`）
   - `main.py`：守护调度 + 单次执行入口
   - `scripts/*`：面向场景的薄 CLI 包装

4. 领域与基础设施层（保留）
   - `data_pipeline/`：抓取、入库、网关客户端
   - `alpha_models/`：Qlib 工作流
   - `backtesting/`：组合构建
   - `utils/`：工具模块与 run-tracker 兼容接口

依赖方向：
- `quantcore` 可依赖领域/基础设施模块。
- `scheduler` 与 `scripts` 依赖 `quantcore`。
- `utils` 维持叶子模块属性，不反向依赖上层。

## 3. 核心运行流

### 晚间流水线

`fetch_data -> ingest_to_db`

- `DataPipelineService.fetch_data` 抓取股票/指数，写入本地 CSV，并打包 send buffer。
- `DataPipelineService.ingest_to_db` 将 CSV 分片批量推送至网关。

### 午后流水线

`export_from_db -> dump_to_qlib -> predict -> build_portfolio`

- 从网关导出全市场历史到 receive buffer。
- 转换为 Qlib 二进制数据。
- 加载最新模型并生成预测。
- 构建目标权重与调仓指令。

### 全流程流水线

`fetch -> ingest -> export -> dump -> train -> predict -> portfolio`

训练后钩子：
- 每次训练成功后会立即触发 `scripts/view.py` 生成可视化结果。

## 4. 配置与状态

配置：
- 统一配置模型：`quantcore.settings.AppSettings`
- 兼容旧入口：`config/settings.py`
- `PIPELINE_COOLDOWN_SECONDS` 支持运行时环境变量覆盖

状态：
- 运行历史通过 `RunHistoryStore` 持久化至 `.data/run_history.json`
- `utils/run_tracker.py` 保留兼容 API

## 5. 可靠性与测试策略

重构后的关键保障：

- 副作用集中在服务层，边界清晰。
- 流水线统一通过执行器运行（日志/失败路径一致）。
- ingest 默认不删除源文件（`delete_after_ingest=False`），由调用方显式决定。

测试策略：
- 单元测试覆盖核心层（配置、服务、注册表、工具）。
- 整体流程测试覆盖全流水线顺序（重操作统一 mock）。
- 全市场抓取与真实训练不在自动化测试中直跑。

## 6. 兼容性说明

保持兼容：
- `main.py --run ...` 与调度行为。
- 任务名、流水线名。
- `config.settings.settings` 与 `utils.run_tracker` 的历史导入路径。

有意调整：
- Python 编排逻辑迁移到 `quantcore/` 的服务化架构。
- pipeline cooldown 可在运行时响应环境变量。
- ingest 删除行为改为显式开关 `delete_after_ingest`。
