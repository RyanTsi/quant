# QuantFrame Architecture

This document explains how the current codebase works end-to-end, what each module owns, and which boundaries should not be broken.

It is based on:
- Current implementation in this repository.
- Existing `roadmap.md` files across modules.

It is intentionally implementation-oriented (not aspirational) and highlights known gaps where roadmap and code diverge.

## 1. System At A Glance

QuantFrame is a daily A-share quant workflow with three runtimes:

1. Python orchestration (`main.py`, `scheduler/`, `scripts/`)
2. C++ gateway service (`server/main.cc`, Drogon)
3. TimescaleDB storage (`server/sql/market_data_daily.sql`)

Core pipeline:

```text
baostock -> per-symbol CSV -> packaged CSV chunks -> C++ HTTP gateway -> TimescaleDB
TimescaleDB -> per-symbol CSV export -> Qlib binary -> train/predict -> picks -> portfolio targets/orders
```

Main scheduler windows (weekday, local process time):
- `14:00`: afternoon pipeline
- `18:15`: evening pipeline

## 2. Architectural Invariants

These are the most important architecture rules in the current code.

1. Python never talks to PostgreSQL directly in production flow.
   - Data access goes through HTTP gateway endpoints in `server/main.cc`.
   - Python DB client lives in `data_pipeline/database.py`.

2. `scheduler` is orchestration, not domain logic.
   - Task wrapper and sequencing live in `scheduler/`.
   - Heavy logic is delegated to `data_pipeline/`, `alpha_models/`, and `scripts/`.

3. Task failures are explicit and stop pipelines.
   - `@task` wraps exceptions as `TaskFailed`.
   - `run_pipeline` aborts on first failed task.

4. Runtime state is file-backed and append-friendly.
   - `.data/run_history.json` stores last run metadata.
   - `output/` stores prediction and portfolio artifacts.

5. `utils/` should remain a leaf dependency.
   - It should not import business modules.
   - Higher-level modules may depend on it.

## 3. Code Map (Ownership)

### 3.1 Entry And Scheduling

- `main.py`
  - CLI and scheduler daemon entry.
  - Supports one-shot task runs (`--run`) and status inspection (`--status`).
  - Registers weekday cron jobs via `schedule`.

- `scheduler/decorator.py`
  - `@task(name)` wrapper for logging, timing, and failure normalization.

- `scheduler/pipelines.py`
  - Defines `EVENING_PIPELINE`, `AFTERNOON_PIPELINE`, `FULL_PIPELINE`.
  - Executes tasks sequentially with configurable cooldown (`PIPELINE_COOLDOWN_SECONDS`).

- `scheduler/data_tasks.py`
  - Fetch, ingest, export tasks.

- `scheduler/model_tasks.py`
  - Dump-to-Qlib, train, predict, and portfolio build tasks.

### 3.2 Data Plane

- `data_pipeline/fetcher.py`
  - Uses `baostock` to fetch bars.
  - Writes per-symbol CSV files to date-ranged directories.

- `utils/io.py::package_data`
  - Merges per-symbol CSVs into chunked files (`all_data_*.csv`) under send buffer.

- `data_pipeline/ingest.py`
  - Reads chunk CSVs and POSTs batches to gateway endpoint `/api/v1/ingest/daily`.
  - Deletes each CSV after ingestion.

- `data_pipeline/database.py`
  - HTTP client abstraction for gateway query and management endpoints.

- `server/main.cc`
  - In-memory buffer + periodic flush to TimescaleDB (2s interval).
  - Query, stats, symbol list, delete, and health endpoints.

### 3.3 Modeling And Signals

- `alpha_models/qlib_workflow.py`
  - Entry for Qlib training workflow.
  - Persists experiment and recorder IDs to run tracker.

- `alpha_models/workflow/runner.py`
  - YAML loader/merger and workflow runtime.
  - Handles Qlib init, model fit, records, and signal metric extraction.

- `alpha_models/workflow_config_transformer_Alpha158.yaml`
  - Canonical training config (Alpha158 + Transformer + records).

- `scripts/predict.py`
  - Loads trained model from MLflow recorder.
  - Builds liquidity-based daily universe and generates `output/top_picks_<date>.csv`.

### 3.4 Portfolio Construction

- `backtesting/portfolio.py`
  - Converts picks to normalized target weights with per-stock cap.
  - Builds rebalance orders against previous target.

- `scripts/build_portfolio.py`
  - Reads latest picks, writes:
    - `output/target_weights_<date>.csv`
    - `output/orders_<date>.csv`

### 3.5 Config, State, And Utility

- `config/settings.py`
  - Singleton configuration object from `.env`.
  - Also defines local data paths and creates core directories.

- `utils/run_tracker.py`
  - JSON-backed run metadata (`record_run`, `get_last_run`).

- `utils/format.py`
  - Stock code and date normalization helpers.

### 3.6 Isolated / WIP Areas

- `news_module/`
  - Explicitly marked WIP and isolated.
  - Not integrated into scheduler pipelines.
  - Contains known issues (missing `news_module.config`, model/schema drift around `summary`).

- `rl_portfolio/`
  - Placeholder package.

- `data_pipeline/preprocesser.py`
  - Deprecated path; active feature engineering now comes from Qlib handlers.

## 4. Runtime Data Flow

### 4.1 Evening Pipeline (`fetch_data -> ingest_to_db`)

1. `fetch_data`
   - Determines fetch window from last run (`fetch_stock`) with 7-day lookback.
   - Fetches stock and index bars.
   - Writes per-symbol CSVs into `.data/<start>-<end>/`.
   - Packages into `.data/send_buffer/all_data_*.csv`.

2. `ingest_to_db`
   - Reads send buffer files.
   - Posts batches (size 4096) to `/api/v1/ingest/daily`.
   - Deletes uploaded files to avoid re-ingestion.

### 4.2 Afternoon Pipeline (`export_from_db -> dump_to_qlib -> predict -> build_portfolio`)

1. `export_from_db`
   - Health-checks gateway, lists all symbols, paginates query by symbol.
   - Writes per-symbol CSVs to `.data/receive_buffer/`.

2. `dump_to_qlib`
   - Calls `scripts/dump_bin.py` to generate Qlib binary data under `.data/qlib_data/`.

3. `predict`
   - Uses latest trading date (or CLI date).
   - Builds a dynamic pool from recent liquidity.
   - Loads model by env IDs or `run_history.json` fallback.
   - Writes `output/top_picks_<date>.csv`.

4. `build_portfolio`
   - Builds target weights and rebalance orders.
   - Writes `output/target_weights_<date>.csv` and `output/orders_<date>.csv`.

### 4.3 Full Pipeline

`FULL_PIPELINE` adds training between dump and predict:

`fetch -> ingest -> export -> dump -> train -> predict -> portfolio`

## 5. Service Boundary: C++ Gateway

Gateway endpoints are prefixed with `/api/v1`.

Ingest:
- `POST /ingest/daily`
- `POST /ingest/daily/single`

Query:
- `GET /query/daily/all`
- `GET /query/daily/symbol`
- `POST /query/daily/symbols`
- `GET /query/daily/latest`

Stats and meta:
- `GET /stats/summary`
- `GET /symbols`
- `GET /health`

Management:
- `DELETE /data/daily`

Storage model:
- Table `market_data_daily` hypertable keyed by `(symbol, date)`.
- Upsert semantics via `ON CONFLICT`.
- Compression policy configured in SQL bootstrap.

## 6. Configuration Surface

Single source of runtime config: `config/settings.py` loaded from `.env`.

Most important keys:
- `DB_HOST`, `DB_PORT`
- `GATEWAY_LIST_SYMBOLS_TIMEOUT`
- `PIPELINE_COOLDOWN_SECONDS`
- `QLIB_PROVIDER_URI`, `QLIB_MLRUNS_URI`, `QLIB_EXPERIMENT_NAME`
- `QLIB_WORKFLOW_CONFIG`
- `QLIB_EXPERIMENT_ID`, `QLIB_RECORDER_ID`

Path conventions:
- `.data/send_buffer/`: ingest payload chunks
- `.data/receive_buffer/`: DB exports
- `.data/qlib_data/`: Qlib binary dataset
- `.data/run_history.json`: task run metadata
- `output/`: prediction and portfolio outputs
- `scheduler.log`: scheduler and task logging

## 7. Failure Model And Observability

Failure handling:
- Task exceptions become `TaskFailed`.
- Pipelines stop on first failed task.
- Health checks guard DB-export path.

Observability:
- Structured task logs in `scheduler.log`.
- Per-task run metadata in `run_history.json`.
- Unit tests cover key behavior (pipeline stop-on-error, cooldown, ingest parsing, DB client calls, filter logic, portfolio logic).

## 8. Roadmap Alignment And Current Drift

The following mismatches exist between scattered roadmap docs and current repository state:

1. `alpha_models/roadmap.md` references `LSTM.py` and `quantTransformer.py`, but these files are not in the current tree.
2. `test/roadmap.md` references `test_qlib_workflow_refactor.py`, but that file is not present.
3. `scheduler/roadmap.md` does not yet reflect that `build_portfolio` is part of afternoon/full pipelines in code.
4. `news_module` remains intentionally isolated and currently broken for production use.

This document reflects the code as source of truth; roadmap files should be updated to match.

## 9. Extension Guidelines

When adding features, keep these patterns:

1. New scheduled behavior:
   - Implement logic outside `scheduler/`.
   - Wrap with `@task`.
   - Add to pipeline list in `scheduler/pipelines.py`.

2. New gateway behavior:
   - Add endpoint in `server/main.cc`.
   - Extend `data_pipeline/database.py`.
   - Keep API contract stable for scripts and scheduler tasks.

3. New modeling experiment:
   - Prefer YAML config change first.
   - Keep workflow orchestration in `alpha_models/workflow/runner.py`.
   - Persist model run metadata through `utils/run_tracker`.

4. New utility helper:
   - Keep `utils/` dependency direction one-way (leaf).

Following these boundaries keeps the system debuggable and prevents orchestration, domain logic, and I/O concerns from collapsing into one layer.
