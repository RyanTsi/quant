# QuantFrame Architecture (Python Side Refactor)

This document describes the current architecture after the Python-side refactor
(excluding `server/` and deprecated `news_module/`).

## 1. System Boundaries

Runtime is split into three boundaries:

1. Python orchestration and modeling (`main.py`, `quantcore/`, `scheduler/`, `scripts/`)
2. C++ gateway (`server/`) for market-data API
3. TimescaleDB storage (behind the gateway)

Hard boundary:
- Python does not directly operate PostgreSQL in production flow.
- `news_module/` remains isolated and not part of runtime pipelines.

## 2. Layered Architecture

The Python side now follows a layered design:

1. Core runtime layer (`quantcore/`)
   - `settings.py`: environment + path loading (`AppSettings`)
   - `history.py`: run-history persistence (`RunHistoryStore`)
   - `pipeline.py`: generic pipeline executor (`PipelineRunner`)
   - `services/`: data/model service orchestration
   - `registry.py`: task/pipeline registry for CLI dispatch

2. Scheduler adapter layer (`scheduler/`)
   - `decorator.py`: task wrapper and failure normalization
   - `data_tasks.py`, `model_tasks.py`: thin adapters to core services
   - `pipelines.py`: pipeline definitions + execution wiring

3. CLI/script adapter layer (`main.py`, `scripts/`)
   - `main.py`: scheduler daemon + one-shot command entry
   - `scripts/*`: focused CLI wrappers for data/model operations

4. Domain and infrastructure modules (existing)
   - `data_pipeline/`: fetch, ingest, gateway client
   - `alpha_models/`: Qlib workflow
   - `backtesting/`: portfolio construction
   - `utils/`: helper utilities and run-tracker compatibility API

Dependency direction:
- `quantcore` may depend on domain/infrastructure modules.
- `scheduler` and `scripts` depend on `quantcore`.
- `utils` stays as a leaf helper and should not depend on higher layers.

## 3. Key Runtime Flows

### Evening Pipeline

`fetch_data -> ingest_to_db`

- `DataPipelineService.fetch_data` fetches stock/index bars, writes local CSV, and packs send-buffer chunks.
- `DataPipelineService.ingest_to_db` pushes chunked CSV payloads to gateway ingest API.

### Afternoon Pipeline

`export_from_db -> dump_to_qlib -> predict -> build_portfolio`

- Export full symbol history from gateway to receive buffer.
- Dump CSV to Qlib binary.
- Generate predictions with latest trained model.
- Build target weights and rebalance orders.

### Full Pipeline

`fetch -> ingest -> export -> dump -> train -> predict -> portfolio`

Post-train hook:
- After each successful training run, visualization is generated immediately via `scripts/view.py`.

## 4. Configuration and State

Configuration:
- Single runtime model: `quantcore.settings.AppSettings`.
- Backward-compatible import path: `config/settings.py`.
- Cooldown supports dynamic env override at run-time (`PIPELINE_COOLDOWN_SECONDS`).

State:
- Run metadata persists to `.data/run_history.json` via `RunHistoryStore`.
- Backward-compatible API remains in `utils/run_tracker.py`.

## 5. Reliability and Test Strategy

The refactor enforces:

- Explicit service-layer boundaries for side effects.
- Pipeline execution through a shared runner (single logging/failure path).
- Non-destructive default ingest behavior (`delete_after_ingest=False`) unless caller opts in.

Testing strategy:
- Unit tests for core layers (`quantcore.settings`, services, registry, helpers).
- Integration-style tests for end-to-end pipeline order with mocked heavy operations.
- Heavy operations (full market fetch/model training) are mocked in tests; production behavior is unchanged.

## 6. Compatibility Notes

Kept compatible:
- `main.py --run ...` and scheduler behavior.
- Existing task names and pipeline names.
- Existing `config.settings.settings` and `utils.run_tracker` imports.

Changed intentionally:
- Python orchestration moved to `quantcore/` service-centric design.
- Pipeline cooldown can react to runtime env changes.
- CSV ingest file deletion is now explicit via `delete_after_ingest`.
