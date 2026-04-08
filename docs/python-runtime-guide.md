# Python Runtime Guide

This guide describes the current runtime-first Python workflow after the
compatibility cleanup completed on 2026-04-02.

## 1. Canonical Ownership

- `runtime/` is the canonical control plane.
- `runtime/bootstrap.py`, `runtime/registry.py`, `runtime/tasks.py`, and `runtime/constants.py` own dispatch, task names, and pipeline order.
- `runtime/adapters/fetching.py`, `runtime/adapters/ingest.py`, `runtime/adapters/exporting.py`, and `runtime/adapters/modeling.py` own workflow logic.
- `runtime/services.py` owns the canonical data/model service classes and builders.
- `runtime/runlog.py` owns both `RunLogStore` and the helper API (`load_run_history`, `record_run`, `get_last_run`, `today`, `today_dash`).
- `runtime/config.py` is the only supported config entry for active Python runtime code.
- `main.py` and `scripts/*` are thin operator-facing entrypoints.
- `data_pipeline/fetcher.py` and `data_pipeline/database.py` remain low-level provider/client modules beneath runtime orchestration.

## 2. Key Artifact Directories

| Path | Meaning |
|---|---|
| `.data/<start>-<end>/` | raw fetched market CSV directory (`save_dir`) |
| `.data/send_buffer/` | packaged ingest directory for gateway upload (`send_buffer_dir`) |
| `.data/receive_buffer/` | exported per-symbol CSV directory from the gateway |
| `.data/qlib_data/` | dumped Qlib binary directory |
| `output/` | prediction CSVs, target weights, orders, and view artifacts |
| `.data/run_history.json` | structured runtime run-history file |

## 3. Task Semantics

### Fetch

- Entry points: `main.py --run fetch`, `python -m scripts.update_data`
- Canonical logic: `runtime/adapters/fetching.py`
- Preserved window math:
  - fallback `last_end_date = "20100108"` when no prior history exists
  - `start_date = last_end_date - lookback_days`
  - `end_date = today`
- Successful fetch writes run history with:
  - `start_date`
  - `end_date`
  - `last_end_date`
  - `lookback_days`
  - `save_dir`
  - `send_buffer_dir`

### Ingest

- Entry points: `main.py --run ingest`, `python -m scripts.put_data`
- Canonical logic: `runtime/adapters/ingest.py`
- Structured result fields:
  - `data_dir`
  - `server_url`
  - `files_found`
  - `files_ingested`
  - `skipped_files`
  - `rows_sent`
  - `failed_files`
  - `failed_batches`
  - `deleted_files`
- Important behavior:
  - missing directory returns `None` from `DataPipelineService.ingest_to_db()` and does not write history
  - `rows_sent` counts rows from successful HTTP 200 batches only
  - the low-level adapter default is `delete_after_ingest=False`
  - `python -m scripts.put_data` stays non-destructive unless `--delete_after_ingest` is supplied
  - `main.py --run ingest` plus the `evening` / `full` pipelines explicitly use `delete_after_ingest=True`
  - destructive ingest deletes processed files after the ingest attempt, including files with failed batches

### Export

- Entry point: `main.py --run export`
- Canonical logic: `runtime/adapters/exporting.py`
- Structured result fields:
  - `output_dir`
  - `exported`
  - `total`
  - `failed_symbols`
  - `partial_symbols`
- Important behavior:
  - export prefers local symbol artifacts in `.data/stock_code_list`, `.data/index_code_list`, and `.data/qlib_data/instruments/all.txt`; if those are unavailable, it falls back to `GET /api/v1/symbols`
  - `failed_symbols` means no artifact exported for that symbol
  - `partial_symbols` means CSV was written but later pages failed

### Dump

- Entry points: `main.py --run dump`, `python -m scripts.dump_bin ...`
- Canonical logic: `runtime/adapters/modeling.dump_to_qlib_data`
- Important behavior:
  - `ModelPipelineService.dump_to_qlib()` returns `None` if `.data/receive_buffer/` is missing or empty
  - successful runs record `dump_to_qlib` history with `csv_dir` and `qlib_dir`

### Filter

- Entry points: `main.py --run filter`, `python -m scripts.filter`
- Canonical logic: `runtime/services.ModelPipelineService.build_training_universe()` -> `runtime/adapters/modeling.build_training_universe_file()` -> `model_function/universe.py`
- Important behavior:
  - `scripts/filter.py` is a thin wrapper and no longer owns the main training-universe implementation
  - the runtime-managed path writes the canonical instrument artifact under `qlib_data/instruments/my_800_stocks.txt`
  - successful runs record `filter_training_universe` history with the input parameters, output path, effective end date, month count, and unique-symbol count for the merged artifact

### Train

- Entry point: `main.py --run train`
- Runtime path: `runtime.tasks.train_model()` -> `ModelPipelineService.train_model()` -> `alpha_models.qlib_workflow.run_training()`
- Important behavior:
  - `ModelPipelineService` records `qlib_train` metadata used by later predict/eval/view commands
  - the runtime service also records a `train_model` run entry with the execution date
  - successful training automatically calls the post-train view generator with the resulting experiment/recorder ids

### Predict

- Entry points: `main.py --run predict`, `python -m scripts.predict`
- Canonical logic: `runtime/services.ModelPipelineService.predict()` -> `runtime/adapters/modeling.generate_predictions`
- Important behavior:
  - if `--date` is omitted, the latest local trading date is used
  - recorder selection comes from `QLIB_RECORDER_ID` + `QLIB_EXPERIMENT_ID` when both are set, otherwise from `run_history.json -> qlib_train`
  - if neither env ids nor `qlib_train` history exist, predict fails fast with the operator-facing “run train first” error
  - the default output artifact is `output/top_picks_<date>.csv`

### Portfolio

- Entry points: `main.py --run portfolio`, `python -m scripts.build_portfolio`
- Canonical logic: `runtime/services.ModelPipelineService.build_portfolio()` -> `runtime/adapters/modeling.build_portfolio_outputs`
- Important behavior:
  - if `--date` is omitted, the latest local trading date is used
  - input is `output/top_picks_<date>.csv`
  - outputs are `output/target_weights_<date>.csv` and `output/orders_<date>.csv`
  - previous holdings continuity reads `target_weights_<prev_date>.csv` from the previous local trading day, not the previous natural day
  - the normal operator path records `build_portfolio` history at the service boundary with the picks file, output files, and order summary stats

## 4. Common Commands

```bash
# Runtime entry
python main.py --run fetch
python main.py --run ingest
python main.py --run export
python main.py --run dump
python main.py --run filter
python main.py --run train
python main.py --run predict
python main.py --run portfolio
python main.py --run full

# Thin script wrappers
python -m scripts.update_data
python -m scripts.put_data --data_dir /path/to/csvs
python -m scripts.dump_bin dump_all --data_path=.data/receive_buffer --qlib_dir=.data/qlib_data
python -m scripts.filter --start_year 2010 --end_year 2026 --top_n 2200 --random_seed 42
python -m scripts.predict --date 2026-04-01 --out output/top_picks_2026-04-01.csv
python -m scripts.build_portfolio --date 2026-04-01
python -m scripts.eval_test --config alpha_models/workflow_config_transformer_Alpha158.yaml
```

## 5. Verification

Focused runtime checks commonly used while updating docs:

```bash
conda run -n quant python -m unittest \
  test.test_runtime_bootstrap \
  test.test_runtime_tasks \
  test.test_runtime_runlog \
  test.test_data_cli_wrappers \
  test.test_model_cli_wrappers
```

Primary full-suite command:

```bash
conda run -n quant python -m unittest discover -s test -p 'test_*.py'
```

## 6. Troubleshooting

- `No data directory found, skipping.` during ingest:
  - `send_buffer_dir` is missing; re-run fetch or point `scripts.put_data --data_dir` at a valid CSV directory.
- Export reports `partial_symbols`:
  - CSVs exist, but some gateway pagination failed; treat those symbols as incomplete.
- Export stalls or fails while listing symbols:
  - the runtime now prefers local symbol artifacts and only queries `GET /api/v1/symbols` when those artifacts are unavailable, but stale local code lists can still miss newly ingested symbols; re-run `fetch` if coverage looks wrong.
- Predict fails with `No trained model found`:
  - set `QLIB_RECORDER_ID` and `QLIB_EXPERIMENT_ID`, or run `train` first so `qlib_train` history is recorded by `ModelPipelineService`.
- Fetch script prints `Packed data directory`:
  - this is the packaged ingest directory (`send_buffer_dir`), not the raw fetch directory.

Cross-reference:
- [README](../README.md)
- [ARCHITECTURE](../ARCHITECTURE.md)
- [Python runtime product spec](product-specs/python-runtime-v2.md)
