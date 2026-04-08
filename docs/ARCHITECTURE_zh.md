# QuantFrame 架构说明（当前 Python 运行时）

本文描述 2026-04-02 完成 runtime-first 收敛后的当前 Python 侧架构。

## 1. 系统边界

系统分为三层边界：

1. Python 编排与建模层（`main.py`、`runtime/`、`scripts/`、`model_function/`、`data_pipeline/`、`alpha_models/`、`backtesting/`、`utils/`）
2. C++ 网关层（`server/`），负责市场数据 HTTP API
3. TimescaleDB 存储层，由网关代理访问

硬约束：
- 生产链路中，Python 不直接访问 PostgreSQL。
- `server/` 是 TimescaleDB 的唯一写入通道。
- 当前活跃的 Python 运行时已不再经过 `scheduler/*`、`quantcore/*`、`config/settings.py`、`utils/run_tracker.py`、`data_pipeline/ingest.py` 等已删除兼容层。

## 2. 分层架构

Python 侧当前采用 runtime-first 分层设计：

1. 规范运行时基础层（`runtime/`）
   - `config.py`：环境、路径加载与目录创建（`AppSettings`）
   - `constants.py`：稳定的任务名和流水线名
   - `contracts.py`、`ports.py`：运行时契约与协议边界
   - `runlog.py`：结构化运行历史持久化（`RunLogStore`）
   - `orchestrator.py`：统一顺序执行语义
   - `registry.py`：任务与流水线分发
   - `bootstrap.py`：默认运行时装配
   - `tasks.py`：供 registry 使用的 runtime 自有 task callable
   - `services.py`：数据/模型 service 装配辅助
   - `adapters/*`：规范的 fetch、ingest、export、dump、predict、portfolio 工作流逻辑

2. 入口层（`main.py`、`scripts/`）
   - `main.py`：守护调度与单次命令入口，内部通过 `runtime.bootstrap.build_default_registry()`
   - `scripts/*`：面向操作的薄包装，调用 runtime adapter 或 service 表面
   - 这一层应保持足够薄，不重复实现核心逻辑

3. 领域与基础设施层
   - `model_function/`：可复用的模型域逻辑，例如确定性股票池构建、预测数据集与 workflow 装配、recorder/模型访问以及分析 helper
   - `data_pipeline/`：底层 BaoStock 抓取 provider 与网关 HTTP 客户端
   - `alpha_models/`：Qlib 工作流与 workflow runner
   - `backtesting/`：组合构建与订单生成
   - `server/`：C++ 网关与 TimescaleDB 部署资源
   - `utils/`：共享的格式化、IO 与预处理叶子工具

依赖方向：
- `runtime/` 是规范控制面。
- `main.py` 与 `scripts/*` 依赖 `runtime/`，而不是反向依赖。
- `model_function/`、`data_pipeline/`、`alpha_models/`、`backtesting/` 继续作为 runtime 编排之下的领域/基础设施模块。
- `utils/` 继续保持叶子级工具定位，不再持有编排或状态归属。

## 3. 核心运行流

### 晚间流水线

`fetch -> ingest`

- 顺序由 `runtime.constants.PIPELINE_TASK_NAMES["evening"]` 定义。
- `runtime.tasks.fetch_data()` 构建 data service，并委托给 `runtime.adapters.fetching.fetch_and_package_market_data`。
- `runtime.tasks.ingest_to_db()` 通过 `DataPipelineService.ingest_to_db()` 委托给 `runtime.adapters.ingest.ingest_directory`。
- fetch 会写出原始 CSV 和打包后的 send-buffer；ingest 负责把打包 CSV 批量上传到网关。

### 午后流水线

`export -> dump -> predict -> portfolio`

- `runtime.tasks.export_from_db()` 把全市场数据从网关导出到 `.data/receive_buffer/`。
- `runtime.tasks.dump_to_qlib()` 将导出 CSV 转为 Qlib 二进制数据。
- `runtime.tasks.predict()` 通过 direct-call 预测流程，结合 `model_function/universe.py` 中的确定性滞后流动性预测池规则，以及 `model_function/qlib.py` 中共享的 recorder/模型与数据集装配 helper，写出 `output/top_picks_<date>.csv`。
- `runtime.tasks.build_portfolio()` 读取预测结果，应用 `model_function/universe.py` 中的显式买入/持有缓冲规则，再写出目标权重与调仓订单。

### 全流程流水线

`fetch -> ingest -> export -> dump -> train -> predict -> portfolio`

- `runtime.tasks.train_model()` 通过 `ModelPipelineService.train_model()` 调用 Qlib 工作流。
- `ModelPipelineService.train_model()` 会在工作流成功完成后记录 `qlib_train` 元数据，供后续预测、评估和可视化命令使用。
- 训练成功后还会通过共享的 `model_function/qlib.py` 分析 helper 自动生成 view，而 `scripts/view.py` 继续保持为薄包装。
- 测试集评估仍是独立操作命令（`scripts.eval_test`）。

## 4. 配置与状态

配置：
- `runtime.config.AppSettings` 是规范配置模型。
- `PIPELINE_COOLDOWN_SECONDS` 可通过 `runtime.bootstrap.cooldown_seconds()` 在运行时动态刷新。
- `qlib_provider_uri` 由 `.data/qlib_data` 推导，不再通过单独环境变量读取。

状态：
- 运行元数据通过 `runtime.runlog.RunLogStore` 持久化到 `.data/run_history.json`。
- 在 POSIX 平台上，runlog 写入会使用文件锁与原子替换；在其他平台上则仍通过原子替换写入，并兼容旧扁平 JSON 结构读取。
- `load_run_history`、`record_run`、`get_last_run`、`today`、`today_dash` 等便捷 API 现在直接位于 `runtime.runlog`。

## 5. 可靠性与测试策略

当前架构的关键保障：

- 运行时分发统一经过 registry/orchestrator 路径。
- 流水线执行共享日志、cooldown 与失败传播语义。
- 底层 ingest adapter 默认是非破坏性的，但面向操作的 runtime 任务路径（`main.py --run ingest`、`evening`、`full`）会显式启用 `delete_after_ingest=True`，用于一次性消费打包缓冲区。
- fetch、ingest、export、predict、portfolio 都返回显式结构化结果元数据。
- 模型侧运行时行为使用 direct-call adapter，而不是 subprocess 编排。
- 相同数据快照下的预测股票池成员资格是确定性的；训练股票池的降采样也在 `model_function/` 内保持可复现。

测试策略：
- 单元测试覆盖 `runtime.config`、`runtime.runlog`、`runtime.registry`、`runtime.orchestrator`、`runtime.tasks` 等基础模块。
- adapter 与 service 测试覆盖 fetch/ingest/export/model 行为，重依赖统一 mock。
- CLI wrapper 测试保护用户侧脚本接口。
- pipeline 测试验证顺序、cooldown 和失败中断语义，而不依赖真实抓取或训练。

## 6. 兼容性说明

保持兼容：
- `main.py --run ...` 与 `main.py --status`
- `runtime.constants` 中的稳定任务名与流水线名
- `python -m scripts.<name>` 入口

有意调整：
- 运行时任务归属现在位于 `runtime.tasks`，不再由已删除的 scheduler wrapper 持有。
- 先前位于 `quantcore.services` 的数据/模型 service 类已并入 `runtime.services`。
- `quantcore/*`、`config/settings.py`、`utils/run_tracker.py`、`scheduler/*`、`data_pipeline/ingest.py` 等历史兼容模块已不再属于活跃架构。
