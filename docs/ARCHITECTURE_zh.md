# QuantFrame 架构说明

本文档说明当前代码库的端到端工作方式、各模块职责边界，以及不应打破的架构约束。

文档依据：
- 当前仓库实现代码。
- 各目录下已有的 `roadmap.md`。

本文档以“实现现状”为主（而非愿景），并明确标注 roadmap 与代码不一致的地方。

## 1. 系统总览

QuantFrame 是一个面向 A 股日频场景的量化工作流，包含三个运行时：

1. Python 编排层（`main.py`、`scheduler/`、`scripts/`）
2. C++ 网关服务（`server/main.cc`，基于 Drogon）
3. TimescaleDB 存储层（`server/sql/market_data_daily.sql`）

核心链路：

```text
baostock -> 分股票 CSV -> 打包 CSV 分片 -> C++ HTTP 网关 -> TimescaleDB
TimescaleDB -> 分股票 CSV 导出 -> Qlib 二进制 -> 训练/预测 -> 选股结果 -> 组合权重/调仓指令
```

主调度窗口（工作日，本地进程时间）：
- `14:00`：afternoon pipeline
- `18:15`：evening pipeline

## 2. 架构不变量

以下约束是当前系统最重要的架构规则。

1. Python 在生产链路中不直连 PostgreSQL。
   - 数据访问统一走 `server/main.cc` 暴露的 HTTP API。
   - Python 侧网关客户端在 `data_pipeline/database.py`。

2. `scheduler` 只做编排，不承载业务细节。
   - 任务包装、流水线编排在 `scheduler/`。
   - 业务逻辑下沉到 `data_pipeline/`、`alpha_models/`、`scripts/`。

3. 任务失败是显式语义，且会中断流水线。
   - `@task` 将异常统一包装为 `TaskFailed`。
   - `run_pipeline` 遇到失败后立即终止后续任务。

4. 运行状态以文件方式持久化。
   - `.data/run_history.json` 记录任务最近一次运行元数据。
   - `output/` 存放预测与组合构建产物。

5. `utils/` 保持叶子模块属性。
   - 不应反向依赖业务层模块。
   - 允许被上层模块依赖。

## 3. 代码地图（职责归属）

### 3.1 入口与调度

- `main.py`
  - CLI 与调度守护进程总入口。
  - 支持单次任务执行（`--run`）与状态查看（`--status`）。
  - 通过 `schedule` 注册工作日定时任务。

- `scheduler/decorator.py`
  - `@task(name)`：日志、耗时统计、失败语义统一。

- `scheduler/pipelines.py`
  - 定义 `EVENING_PIPELINE`、`AFTERNOON_PIPELINE`、`FULL_PIPELINE`。
  - 按顺序执行任务，任务间冷却由 `PIPELINE_COOLDOWN_SECONDS` 控制。

- `scheduler/data_tasks.py`
  - 数据抓取、入库、导出任务。

- `scheduler/model_tasks.py`
  - Qlib dump、训练、预测、组合构建任务。

### 3.2 数据链路层

- `data_pipeline/fetcher.py`
  - 用 `baostock` 抓取行情。
  - 输出分股票 CSV 到日期区间目录。

- `utils/io.py::package_data`
  - 将分股票 CSV 合并为分片文件（`all_data_*.csv`）写入发送缓冲区。

- `data_pipeline/ingest.py`
  - 读取分片 CSV，批量 POST 到 `/api/v1/ingest/daily`。
  - 成功处理后删除对应 CSV。

- `data_pipeline/database.py`
  - 网关 API 的 Python 客户端封装（查询、管理等）。

- `server/main.cc`
  - 内存缓冲 + 定时刷盘（2 秒）到 TimescaleDB。
  - 提供查询、统计、符号列表、删除、健康检查接口。

### 3.3 模型与信号层

- `alpha_models/qlib_workflow.py`
  - Qlib 训练入口。
  - 将实验 ID / 记录器 ID 写入 run tracker。

- `alpha_models/workflow/runner.py`
  - YAML 配置加载与合并、工作流执行。
  - 负责 Qlib 初始化、训练、record 生成、信号指标提取。

- `alpha_models/workflow_config_transformer_Alpha158.yaml`
  - 当前主训练配置（Alpha158 + Transformer + record）。

- `scripts/predict.py`
  - 从 MLflow recorder 加载模型。
  - 构建流动性驱动的当日股票池并输出 `output/top_picks_<date>.csv`。

### 3.4 组合构建层

- `backtesting/portfolio.py`
  - 将选股结果转为目标权重（含单票权重上限与归一化）。
  - 对比上一期目标，生成调仓指令。

- `scripts/build_portfolio.py`
  - 读取 picks，输出：
    - `output/target_weights_<date>.csv`
    - `output/orders_<date>.csv`

### 3.5 配置、状态与工具层

- `config/settings.py`
  - 从 `.env` 加载全局配置的单例入口。
  - 同时定义本地数据路径并创建核心目录。

- `utils/run_tracker.py`
  - JSON 持久化运行记录（`record_run`、`get_last_run`）。

- `utils/format.py`
  - 股票代码与日期格式标准化工具。

### 3.6 隔离 / WIP 区域

- `news_module/`
  - 明确标记为 WIP，且与主流水线隔离。
  - 当前未接入 scheduler。
  - 存在已知问题（如缺失 `news_module.config`、`summary` 字段漂移）。

- `rl_portfolio/`
  - 占位包，尚未实装。

- `data_pipeline/preprocesser.py`
  - 已弃用链路，现役特征工程由 Qlib handler 承担。

## 4. 运行时数据流

### 4.1 Evening Pipeline（`fetch_data -> ingest_to_db`）

1. `fetch_data`
   - 读取 `fetch_stock` 最近运行，基于其结束日期回看 7 天。
   - 抓取股票与指数数据。
   - 写入 `.data/<start>-<end>/` 分股票 CSV。
   - 打包到 `.data/send_buffer/all_data_*.csv`。

2. `ingest_to_db`
   - 读取 send buffer 分片文件。
   - 按批次（4096）POST 到 `/api/v1/ingest/daily`。
   - 上传后删除对应分片，避免重复入库。

### 4.2 Afternoon Pipeline（`export_from_db -> dump_to_qlib -> predict -> build_portfolio`）

1. `export_from_db`
   - 先做网关健康检查，再拉取 symbols，按 symbol 分页查询。
   - 输出分股票 CSV 到 `.data/receive_buffer/`。

2. `dump_to_qlib`
   - 调用 `scripts/dump_bin.py`，生成 `.data/qlib_data/` 二进制数据。

3. `predict`
   - 默认用本地最新交易日（也支持 CLI 指定日期）。
   - 按近期流动性构建动态股票池。
   - 按环境变量 ID 或 `run_history.json` 回退加载模型。
   - 输出 `output/top_picks_<date>.csv`。

4. `build_portfolio`
   - 构建目标权重与调仓指令。
   - 输出 `output/target_weights_<date>.csv` 与 `output/orders_<date>.csv`。

### 4.3 Full Pipeline

`FULL_PIPELINE` 会在 dump 与 predict 之间插入训练：

`fetch -> ingest -> export -> dump -> train -> predict -> portfolio`

## 5. 服务边界：C++ Gateway

网关接口统一前缀为 `/api/v1`。

Ingest：
- `POST /ingest/daily`
- `POST /ingest/daily/single`

Query：
- `GET /query/daily/all`
- `GET /query/daily/symbol`
- `POST /query/daily/symbols`
- `GET /query/daily/latest`

Stats 与 Meta：
- `GET /stats/summary`
- `GET /symbols`
- `GET /health`

管理接口：
- `DELETE /data/daily`

存储模型：
- 表 `market_data_daily`（hypertable），主键 `(symbol, date)`。
- 通过 `ON CONFLICT` 实现 upsert。
- 压缩策略由 SQL 初始化脚本配置。

## 6. 配置面

运行配置单一入口：`config/settings.py`，从 `.env` 加载。

关键配置项：
- `DB_HOST`, `DB_PORT`
- `GATEWAY_LIST_SYMBOLS_TIMEOUT`
- `PIPELINE_COOLDOWN_SECONDS`
- `QLIB_PROVIDER_URI`, `QLIB_MLRUNS_URI`, `QLIB_EXPERIMENT_NAME`
- `QLIB_WORKFLOW_CONFIG`
- `QLIB_EXPERIMENT_ID`, `QLIB_RECORDER_ID`

路径约定：
- `.data/send_buffer/`：待入库分片
- `.data/receive_buffer/`：数据库导出
- `.data/qlib_data/`：Qlib 二进制数据
- `.data/run_history.json`：任务运行元数据
- `output/`：预测与组合产物
- `scheduler.log`：调度与任务日志

## 7. 失败模型与可观测性

失败处理：
- 任务异常统一为 `TaskFailed`。
- 流水线遇到任务失败立即中止。
- 导出路径有健康检查前置保护。

可观测性：
- `run_history.json` 记录任务维度状态。
- 单测覆盖关键行为（失败中断、冷却、ingest 解析、DB client、filter、portfolio）。

## 8. Roadmap 对齐与当前偏差

当前 roadmap 与代码的已知偏差：

1. `alpha_models/roadmap.md` 提到 `LSTM.py`、`quantTransformer.py`，但仓库中暂无这两个文件。
2. `test/roadmap.md` 提到 `test_qlib_workflow_refactor.py`，但当前不存在该测试文件。
3. `scheduler/roadmap.md` 尚未体现代码中已接入 `build_portfolio`（afternoon/full）。
4. `news_module` 仍为隔离中的 WIP，当前不具备生产接入条件。

本文档以代码现状为事实来源；建议后续同步更新各目录 roadmap。

## 9. 扩展建议

新增能力时，建议遵循以下模式：

1. 新增调度能力：
   - 业务实现放在 `scheduler/` 之外。
   - 用 `@task` 包装。
   - 在 `scheduler/pipelines.py` 编排顺序。

2. 新增网关能力：
   - 在 `server/main.cc` 增加 endpoint。
   - 在 `data_pipeline/database.py` 补充客户端方法。
   - 保持 API 合约稳定（或同步更新脚本/任务调用）。

3. 新增模型实验：
   - 优先改 YAML 配置，不先硬编码。
   - 工作流编排保持在 `alpha_models/workflow/runner.py`。
   - 运行元数据通过 `utils/run_tracker` 落盘。

4. 新增工具函数：
   - 保持 `utils/` 单向依赖（叶子模块）。

遵守这些边界可以避免编排层、业务层与 I/O 层耦合失控，提升系统可维护性与可调试性。
