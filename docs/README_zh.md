# QuantFrame

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![C++](https://img.shields.io/badge/C++-17-blue?logo=cplusplus)
![Qlib](https://img.shields.io/badge/Qlib-0.9.7-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![TimescaleDB](https://img.shields.io/badge/TimescaleDB-PostgreSQL-green?logo=postgresql)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

面向中国 A 股市场的端到端量化交易框架。覆盖日线数据采集、时序存储、基于 Transformer 的 Alpha 信号生成（通过 Qlib）及自动化调度，底层采用 C++ / Drogon REST 网关 + TimescaleDB。

> **English version**: [README.md](../README.md)

## 目录

- [概述](#概述)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [使用方法](#使用方法)
- [C++ 数据网关 API](#c-数据网关-api)
- [配置说明](#配置说明)
- [开发进度](#开发进度)
- [更新日志](#更新日志)
- [许可证](#许可证)

## 概述

系统将四个阶段串联为可重复执行的日常工作流：

1. **采集** — 从 baostock 拉取全 A 股和指数日线数据（备选：akshare），保存为按股票分组的 CSV。
2. **存储** — 将 CSV 批量 POST 到 C++ REST 网关，网关以 upsert 方式写入 TimescaleDB 超表。
3. **转换** — 从数据库导出并转换为 Qlib 二进制格式，构建 Alpha158 特征。
4. **建模** — 通过 Qlib 工作流训练 Transformer（MLflow 跟踪 + 信号/回测记录）。
5. **预测与执行** — 在动态流动性股票池上预测，并生成目标权重与调仓指令。

内置调度器（`main.py`）在工作日自动编排：**晚间流水线**（18:15 采集 + 入库）和**午后流水线**（14:00 导出 + 转换 + 预测 + 组合执行）。也可通过 CLI 按需触发单个任务。

```mermaid
graph LR
    BS[baostock] --> F[fetcher]
    F --> CSV[CSV 文件]
    CSV --> ING[put_data.py]
    ING --> GW["C++ 网关 :8080"]
    GW --> TS[(TimescaleDB)]
    TS --> EXP[export_from_db]
    EXP --> DUMP[dump_to_qlib]
    DUMP --> QLIB[Qlib Transformer]
    QLIB --> PRED[top_picks_DATE.csv]
    PRED --> PORT[target_weights + orders]
```

## 项目结构

| 路径 | 作用 |
|------|------|
| `main.py` | 统一 CLI 与调度入口 |
| `config/` | 环境变量与全局运行配置 |
| `data_pipeline/` | 数据采集 / 入库 / 导出 |
| `alpha_models/` | Qlib 训练工作流与模型配置 |
| `scripts/` | 独立工具脚本（`predict`、`filter`、`dump_bin`、`build_portfolio` 等） |
| `scheduler/` | 任务封装与流水线编排 |
| `backtesting/` | 组合构建与执行基线 |
| `server/` | C++ Drogon 网关与 TimescaleDB 部署资源 |
| `test/` | 单元测试 |
| `docs/` | 教程与补充文档 |

## 快速开始

### 环境要求

| 依赖 | 版本 | 备注 |
|------|------|------|
| Python | >= 3.12 | 推荐 conda 或 venv |
| C++17 编译器 | GCC / Clang / MSVC | 编译数据网关 |
| CMake | >= 3.15 | 网关构建 |
| Docker | — | 用于 TimescaleDB |
| TA-Lib C 库 | — | [安装指南](https://ta-lib.github.io/ta-lib-python/install.html) |

### 1. 安装 Python 依赖

```bash
git clone <repo-url> && cd quant
pip install -r requirements.txt
```

> `requirements.txt` 锁定顶层包版本。`torch`、`pandas`、`requests`、`python-dotenv` 等传递依赖由 `pyqlib` 带入。

### 2. 配置环境变量

```bash
cp .env.template .env
```

编辑 `.env`，填入 TuShare Token 和网关地址：

```
TU_TOKEN = <你的 TuShare Token>
DB_HOST  = 127.0.0.1
DB_PORT  = 8080
```

### 3. 启动 TimescaleDB

```bash
cd server/docker
cp .env.template .env   # 填入 Postgres 凭据
docker compose up -d
```

将创建 `market_data_daily` 超表，并启用 7 天压缩策略。

### 4. 编译并运行 C++ 数据网关

```bash
cd server
mkdir build && cd build
cmake ..
make -j$(nproc)
cp ../config.json .     # 编辑 config.json 中的数据库凭据
./quantDataBase
```

网关默认监听 `http://0.0.0.0:8080`。

## 使用方法

### 统一入口 (`main.py`)

```bash
# ─── 运行单个任务 ────────────────────────────────────
python main.py --run fetch       # 通过 baostock 采集股票和指数日线
python main.py --run ingest      # 将本地 CSV POST 到 C++ 网关
python main.py --run export      # 从数据库导出所有股票到单独 CSV
python main.py --run dump        # 将 CSV 转换为 Qlib 二进制格式
python main.py --run train       # 通过 Qlib 工作流训练 Transformer
python main.py --run predict     # 用最新模型生成预测
python main.py --run portfolio   # 基于预测结果生成目标权重和调仓指令

# ─── 运行流水线 ──────────────────────────────────────
python main.py --run evening     # fetch → ingest
python main.py --run afternoon   # export → dump → predict → portfolio
python main.py --run full        # fetch → ingest → export → dump → train → predict → portfolio

# ─── 查看状态 ────────────────────────────────────────
python main.py --status          # 打印每个任务的最后运行时间和元数据

# ─── 守护进程模式 ────────────────────────────────────
python main.py                   # 启动调度器 — 工作日定时：
                                 #   18:15 晚间流水线
                                 #   14:00 午后流水线
```

所有任务运行会记录到 `scheduler.log` 并持久化到 `.data/run_history.json`。

### 独立脚本

```bash
python -m scripts.update_data                             # 增量获取全部股票历史
python -m scripts.put_data [data_dir]                     # 导入 CSV 目录
python scripts/dump_bin.py dump_all --data_path=.data/receive_buffer --qlib_dir=.data/qlib_data
python -m scripts.predict --date 2026-03-25 --out output/top_picks_2026-03-25.csv
python -m scripts.build_portfolio --date 2026-03-25
python -m scripts.eval_test --config alpha_models/workflow_config_transformer_Alpha158.yaml
python -m scripts.filter                                  # 生成按季度滞后筛选的 txt 股票池
python -m scripts.view                                    # 生成 Plotly 绩效报告
```

### 运行测试

```bash
python -m pytest test/
```

## C++ 数据网关 API

所有端点前缀为 `/api/v1`。网关使用线程安全队列缓冲写入数据，每 2 秒通过 `INSERT … ON CONFLICT DO UPDATE` 刷入 TimescaleDB。

| 方法 | 端点 | 说明 |
|------|------|------|
| `POST` | `/ingest/daily` | 批量写入日线数据（JSON 数组） |
| `POST` | `/ingest/daily/single` | 写入单条日线数据 |
| `GET` | `/query/daily/all?date=&limit=&offset=` | 查询指定日期所有股票 |
| `GET` | `/query/daily/symbol?symbol=&start_date=&end_date=&limit=&offset=` | 按股票代码和日期范围查询 |
| `POST` | `/query/daily/symbols` | 多股票查询 `{"symbols":[], "start_date":"", "end_date":""}` |
| `GET` | `/query/daily/latest?symbol=&n=` | 获取指定股票最近 N 条数据 |
| `GET` | `/stats/summary?symbol=&start_date=&end_date=` | 聚合统计（均价、总成交量等） |
| `GET` | `/symbols` | 列出所有股票代码 |
| `DELETE` | `/data/daily?symbol=&start_date=&end_date=` | 按股票代码和可选日期范围删除 |
| `GET` | `/health` | 健康检查（`SELECT 1`） |

## 配置说明

### `.env`（项目根目录）

| 变量 | 读取方 | 说明 |
|------|--------|------|
| `TU_TOKEN` | `config/settings.py` | TuShare API Token |
| `DB_HOST` | `config/settings.py` | C++ 网关主机（默认 `127.0.0.1`） |
| `DB_PORT` | `config/settings.py` | C++ 网关端口（默认 `8080`） |
| `GATEWAY_LIST_SYMBOLS_TIMEOUT` | `config/settings.py` | 网关股票列表查询超时（秒） |
| `PIPELINE_COOLDOWN_SECONDS` | `scheduler/pipelines.py` | 流水线任务之间冷却时间 |
| `QLIB_PROVIDER_URI` | 训练/预测流程 | Qlib 数据目录 |
| `QLIB_MLRUNS_URI` | 训练流程 | MLflow 跟踪 URI |
| `QLIB_EXPERIMENT_NAME` | 训练流程 | 训练实验名称 |
| `QLIB_WORKFLOW_CONFIG` | `alpha_models/qlib_workflow.py` | 训练 YAML 配置路径 |
| `QLIB_EXPERIMENT_ID` | `scripts/predict.py`, `scripts/eval_test.py` | 可选模型定位参数 |
| `QLIB_RECORDER_ID` | `scripts/predict.py`, `scripts/eval_test.py` | 可选模型定位参数 |
| `QLIB_TORCH_DATALOADER_WORKERS` | workflow runner | Windows DataLoader 进程数覆盖 |
| `DB_USER` | — | 预留 |
| `DB_PASSWORD` | — | 预留 |
| `DB_NAME` | — | 预留 |

### `server/docker/.env`（TimescaleDB）

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `TSDB_HOST` | PostgreSQL 主机 | `127.0.0.1` |
| `TSDB_PORT` | PostgreSQL 端口 | `5432` |
| `TSDB_USER` | PostgreSQL 用户 | `postgres` |
| `TSDB_PASSWORD` | PostgreSQL 密码 | — |
| `TSDB_DB` | 数据库名 | `postgres` |

### `server/config.json`

Drogon 配置文件：HTTP 监听（8080 端口）、PostgreSQL 连接池和线程数。网关直连 TimescaleDB，Python 端仅与网关的 HTTP API 交互。

## 开发进度

| 模块 | 状态 | 备注 |
|------|------|------|
| 数据采集（baostock/akshare） | 可用 | 增量采集、重试、ST 过滤 |
| C++ 网关 + TimescaleDB | 可用 | Upsert、查询、统计、Docker 部署 |
| 调度器与 CLI | 可用 | 晚间 / 午后 / 全量流水线、运行历史 |
| Qlib Transformer 工作流 | 可用 | Alpha158 配置化训练、MLflow 模型落盘、信号指标提取 |
| 预测流程 | 可用 | 支持 `--date` / `--out`，60 日流动性基础池 + 前一日结果扩充到 500 |
| 组合执行基线 | 可用 | 基于预测结果生成目标权重和调仓单 |
| 测试集评估脚本 | 可用 | `scripts/eval_test.py` 对整段 test 计算 IC/ICIR |
| 特征工程（TA-Lib） | 可用 | 20+ 特征、截面 z-score |
| DB HTTP 客户端 | 可用 | 完整 CRUD、GET 重试 |
| 自定义 QuantTransformer | 已实现 | 独立训练器，支持早停 |
| LSTM 模型 | 未完成 | 依赖缺失的 `src.data_loader` |
| 新闻模块 | 早期阶段 | 仅 Mock 爬虫；`BaseScraper` 缺少 config 模块 |
| 流动性筛选脚本 | 可用 | 按季度滞后（防未来数据）筛选并输出 txt 股票池 |
| 强化学习组合 | 计划中 | 空包；`gymnasium` + `stable-baselines3` 已在依赖中 |
| 测试 | 已扩展 | 覆盖 settings、pipeline、filter、portfolio、DB client、ingest/export |

## 更新日志

### 2026-03-26
- 在午后/全量流水线中加入 `portfolio` 后处理步骤
- 新增 `scripts/build_portfolio.py` + `backtesting/portfolio.py`，输出目标权重与调仓指令
- 升级预测股票池逻辑：60 日流动性分段抽样基础池 + 前一日高分股票扩充至 500
- 新增可配置预测/评估命令（`--date`、`--out`、整段测试集评估）
- 更新流动性筛选为季度滞后抽样，并输出 txt 股票池格式以规避未来函数
- 在 `config/settings.py` 统一环境变量读取，并补全 `.env.template`

### 2026-03-23
- 新增调度器系统：`@task` 装饰器（日志、计时、异常处理）
- 新增工作日定时流水线：晚间（18:15）、午后（14:00）
- 新增统一 CLI 入口（`--run`、`--status`、守护进程模式）
- 新增 `utils/run_tracker.py` 任务执行历史持久化
- 新增 `utils/format.py` 股票代码与日期格式化工具
- 新增 run_tracker 和 DBClient 单元测试
- 修复 C++ 数据网关问题

### 2026-03-20
- 更新 C++ 数据网关服务器

### 2026-03-19
- 新增 C++ 数据网关（Drogon + TimescaleDB、Docker Compose）
- 新增基于 baostock 的数据采集器
- 移除历史遗留代码

### 更早
- 实现 Qlib Transformer 工作流（Alpha158/Alpha360）
- 构建自定义 QuantTransformer 和 LSTM 模型
- 构建数据管道（akshare 采集器 + TA-Lib 预处理器）
- 创建新闻舆情模块骨架
- 新增股票筛选和预测脚本

## 许可证

本项目采用 MIT License。详情见 [LICENSE](../LICENSE)。
