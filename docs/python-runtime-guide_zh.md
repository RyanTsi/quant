# Python 运行时使用指南

本指南描述 2026-04-02 兼容层清理完成后的当前 runtime-first Python 工作流。

## 1. 规范归属

- `runtime/` 是规范控制面。
- `runtime/bootstrap.py`、`runtime/registry.py`、`runtime/tasks.py`、`runtime/constants.py` 持有分发、任务名和流水线顺序。
- `runtime/adapters/fetching.py`、`runtime/adapters/ingest.py`、`runtime/adapters/exporting.py`、`runtime/adapters/modeling.py` 持有工作流逻辑。
- `runtime/services.py` 持有规范的数据/模型 service 类与 builder。
- `runtime/runlog.py` 同时持有 `RunLogStore` 与便捷 API（`load_run_history`、`record_run`、`get_last_run`、`today`、`today_dash`）。
- `runtime/config.py` 是当前活跃 Python 运行时代码唯一支持的配置入口。
- `main.py` 与 `scripts/*` 是面向操作的薄入口。
- `data_pipeline/fetcher.py` 与 `data_pipeline/database.py` 继续作为 runtime 编排之下的底层 provider/client。

## 2. 关键产物目录

| 路径 | 含义 |
|---|---|
| `.data/<start>-<end>/` | 原始抓取得到的市场 CSV 目录（`save_dir`） |
| `.data/send_buffer/` | 打包后的入库目录，供网关上传（`send_buffer_dir`） |
| `.data/receive_buffer/` | 从网关导出的逐股票 CSV 目录 |
| `.data/qlib_data/` | Qlib 二进制数据目录 |
| `output/` | 预测 CSV、目标权重、调仓单与 view 产物 |
| `.data/run_history.json` | 结构化运行历史文件 |

## 3. 任务语义

### Fetch

- 入口：`main.py --run fetch`、`python -m scripts.update_data`
- 规范实现：`runtime/adapters/fetching.py`
- 保持不变的窗口计算：
  - 无历史时使用 `last_end_date = "20100108"`
  - `start_date = last_end_date - lookback_days`
  - `end_date = today`
- 成功后会把以下元数据写入 history：
  - `start_date`
  - `end_date`
  - `last_end_date`
  - `lookback_days`
  - `save_dir`
  - `send_buffer_dir`

### Ingest

- 入口：`main.py --run ingest`、`python -m scripts.put_data`
- 规范实现：`runtime/adapters/ingest.py`
- 结构化结果字段：
  - `data_dir`
  - `server_url`
  - `files_found`
  - `files_ingested`
  - `skipped_files`
  - `rows_sent`
  - `failed_files`
  - `failed_batches`
  - `deleted_files`
- 关键行为：
  - 目录缺失时，`DataPipelineService.ingest_to_db()` 返回 `None`，且不写 history
  - `rows_sent` 只统计 HTTP 200 成功批次中的行数
  - 底层 adapter 默认值是 `delete_after_ingest=False`
  - `python -m scripts.put_data` 默认非破坏性，只有显式传入 `--delete_after_ingest` 才删除文件
  - `main.py --run ingest` 以及 `evening` / `full` 流水线会显式使用 `delete_after_ingest=True`
  - 破坏性 ingest 会在 ingest attempt 之后删除已处理文件，包括批次失败的文件

### Export

- 入口：`main.py --run export`
- 规范实现：`runtime/adapters/exporting.py`
- 结构化结果字段：
  - `output_dir`
  - `exported`
  - `total`
  - `failed_symbols`
  - `partial_symbols`
- 关键行为：
  - `failed_symbols` 表示该 symbol 完全没有导出产物
  - `partial_symbols` 表示 CSV 已写出，但后续分页失败

### Dump

- 入口：`main.py --run dump`、`python -m scripts.dump_bin ...`
- 规范实现：`runtime/adapters/modeling.dump_to_qlib_data`
- 关键行为：
  - 当 `.data/receive_buffer/` 缺失或为空时，`ModelPipelineService.dump_to_qlib()` 返回 `None`
  - 成功执行后会记录 `dump_to_qlib` history，包含 `csv_dir` 与 `qlib_dir`

### Train

- 入口：`main.py --run train`
- 运行路径：`runtime.tasks.train_model()` -> `ModelPipelineService.train_model()` -> `alpha_models.qlib_workflow.main()`
- 关键行为：
  - Qlib 工作流会记录 `qlib_train` 元数据，供后续预测、评估和可视化命令使用
  - runtime service 也会记录一个 `train_model` 执行日期项
  - 训练成功后还会基于生成的 experiment/recorder id 自动调用训练后 view 生成器

### Predict

- 入口：`main.py --run predict`、`python -m scripts.predict`
- 规范实现：`runtime/adapters/modeling.generate_predictions`
- 关键行为：
  - 若未传 `--date`，默认使用本地最新交易日
  - 当同时设置 `QLIB_RECORDER_ID` 和 `QLIB_EXPERIMENT_ID` 时优先用环境变量选模型，否则回退到 `run_history.json -> qlib_train`
  - 默认输出文件为 `output/top_picks_<date>.csv`

### Portfolio

- 入口：`main.py --run portfolio`、`python -m scripts.build_portfolio`
- 规范实现：`runtime/adapters/modeling.build_portfolio_outputs`
- 关键行为：
  - 输入文件为 `output/top_picks_<date>.csv`
  - 输出文件为 `output/target_weights_<date>.csv` 与 `output/orders_<date>.csv`
  - `build_portfolio` history 会记录 picks 文件、输出文件和订单摘要统计

## 4. 常用命令

```bash
# 统一 runtime 入口
python main.py --run fetch
python main.py --run ingest
python main.py --run export
python main.py --run dump
python main.py --run train
python main.py --run predict
python main.py --run portfolio
python main.py --run full

# 薄脚本包装
python -m scripts.update_data
python -m scripts.put_data --data_dir /path/to/csvs
python -m scripts.dump_bin dump_all --data_path=.data/receive_buffer --qlib_dir=.data/qlib_data
python -m scripts.predict --date 2026-04-01 --out output/top_picks_2026-04-01.csv
python -m scripts.build_portfolio --date 2026-04-01
python -m scripts.eval_test --config alpha_models/workflow_config_transformer_Alpha158.yaml
```

## 5. 验证

更新文档时常用的聚焦 runtime 检查：

```bash
conda run -n quant python -m unittest \
  test.test_runtime_bootstrap \
  test.test_runtime_tasks \
  test.test_runtime_runlog \
  test.test_data_cli_wrappers \
  test.test_model_cli_wrappers
```

全量测试主命令：

```bash
conda run -n quant python -m unittest discover -s test -p 'test_*.py'
```

## 6. 排障提示

- ingest 时出现 `No data directory found, skipping.`：
  - 说明 `send_buffer_dir` 缺失；先重新执行 fetch，或者把 `scripts.put_data --data_dir` 指向有效 CSV 目录。
- export 结果里出现 `partial_symbols`：
  - 说明 CSV 已生成，但部分分页失败，对应 symbol 数据不完整。
- predict 报 `No trained model found`：
  - 需要先设置 `QLIB_RECORDER_ID` 和 `QLIB_EXPERIMENT_ID`，或者先执行 `train` 让 `qlib_train` history 落盘。
- fetch 脚本打印 `Packed data directory`：
  - 这里指的是打包后的入库目录（`send_buffer_dir`），不是原始抓取目录。

相关文档：
- [README](../README.md)
- [ARCHITECTURE](../ARCHITECTURE.md)
- [Python 运行时产品规格](product-specs/python-runtime-v2_zh.md)
