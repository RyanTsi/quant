# QuantFrame Architecture (Current Python Runtime)

This document describes the current Python-side architecture after the
runtime-first consolidation completed on 2026-04-02.

## 1. System Boundaries

The system is split into three boundaries:

1. Python orchestration and modeling (`main.py`, `runtime/`, `scripts/`, `model_function/`, `data_pipeline/`, `alpha_models/`, `backtesting/`, `utils/`)
2. C++ gateway (`server/`) for market-data HTTP APIs
3. TimescaleDB storage behind the gateway

Hard boundaries:
- Python does not directly operate PostgreSQL in the production flow.
- `server/` remains the only write path to TimescaleDB.
- The active Python runtime no longer routes through deleted compatibility layers such as `scheduler/*`, `quantcore/*`, `config/settings.py`, `utils/run_tracker.py`, or `data_pipeline/ingest.py`.

## 2. Layered Architecture

The Python side now follows a runtime-first layered design:

1. Canonical runtime foundation (`runtime/`)
   - `config.py`: environment, path loading, and directory creation (`AppSettings`)
   - `constants.py`: stable task and pipeline names
   - `contracts.py`, `ports.py`: runtime contracts and protocol boundaries
   - `runlog.py`: structured run-history persistence (`RunLogStore`)
   - `orchestrator.py`: shared sequential execution semantics
   - `registry.py`: task and pipeline dispatch
   - `bootstrap.py`: default runtime assembly
   - `tasks.py`: runtime-owned task callables used by the registry
   - `services.py`: assembly helpers for data/model services
   - `adapters/*`: canonical fetch, ingest, export, dump, predict, and portfolio workflow logic

2. Entry-point layer (`main.py`, `scripts/`)
   - `main.py`: scheduler daemon plus one-shot command entry using `runtime.bootstrap.build_default_registry()`
   - `scripts/*`: thin operator-facing wrappers over runtime adapters or service surfaces
   - These files should stay shallow and delegate behavior downward

3. Domain and infrastructure modules
   - `model_function/`: reusable model-domain logic such as deterministic universe construction, prediction dataset/workflow assembly, recorder/model access, and analysis helpers
   - `data_pipeline/`: low-level BaoStock fetch provider and gateway HTTP client
   - `alpha_models/`: Qlib workflow and workflow runner
   - `backtesting/`: portfolio construction and order generation
   - `server/`: C++ gateway and TimescaleDB deployment assets
   - `utils/`: shared leaf helpers for formatting, IO, and preprocessing

Dependency direction:
- `runtime/` is the canonical control plane.
- `main.py` and `scripts/*` depend on `runtime/`, not the reverse.
- `model_function/`, `data_pipeline/`, `alpha_models/`, and `backtesting/` stay domain/infrastructure focused beneath runtime orchestration.
- `utils/` stays leaf-level and should not grow orchestration or state ownership.

## 3. Key Runtime Flows

### Evening Pipeline

`fetch -> ingest`

- `runtime.constants.PIPELINE_TASK_NAMES["evening"]` defines the order.
- `runtime.tasks.fetch_data()` builds a data service and delegates to `runtime.adapters.fetching.fetch_and_package_market_data`.
- `runtime.tasks.ingest_to_db()` delegates to `runtime.adapters.ingest.ingest_directory` through `DataPipelineService.ingest_to_db()`.
- Fetch writes raw CSVs plus packaged send-buffer artifacts; ingest uploads packaged CSV batches to the gateway.

### Afternoon Pipeline

`export -> dump -> predict -> portfolio`

- `runtime.tasks.export_from_db()` exports all symbols from the gateway into `.data/receive_buffer/`.
- `runtime.tasks.dump_to_qlib()` converts export CSVs into Qlib binary data.
- `runtime.tasks.predict()` runs direct-call prediction generation, using `model_function/universe.py` for the deterministic lagged-liquidity prediction universe and `model_function/qlib.py` for shared recorder/model and dataset assembly helpers, and writes `output/top_picks_<date>.csv`.
- `runtime.tasks.build_portfolio()` reads prediction output, applies explicit buy/hold buffer rules from `model_function/universe.py`, and writes target weights plus rebalance orders.

### Full Pipeline

`fetch -> ingest -> export -> dump -> train -> predict -> portfolio`

- `runtime.tasks.train_model()` calls the Qlib workflow through `ModelPipelineService.train_model()`.
- `ModelPipelineService.train_model()` records `qlib_train` metadata for later predict/eval/view flows after the workflow completes successfully.
- Successful training also triggers automatic view generation through the shared `model_function/qlib.py` analysis helper, with `scripts/view.py` remaining a thin wrapper.
- Test-set evaluation remains a separate operator command (`scripts.eval_test`).

## 4. Configuration and State

Configuration:
- `runtime.config.AppSettings` is the canonical configuration model.
- `PIPELINE_COOLDOWN_SECONDS` can be refreshed dynamically through `runtime.bootstrap.cooldown_seconds()`.
- `qlib_provider_uri` is derived from `.data/qlib_data`; it is not read from a dedicated environment variable.

State:
- Run metadata persists to `.data/run_history.json` via `runtime.runlog.RunLogStore`.
- On POSIX platforms, runlog writes use file locking plus atomic replace; on other platforms they still use atomic replace and keep legacy flat JSON readable.
- Convenience helpers such as `load_run_history`, `record_run`, `get_last_run`, `today`, and `today_dash` now live in `runtime.runlog`.

## 5. Reliability and Test Strategy

The current architecture enforces:

- Runtime-first dispatch through one registry/orchestrator path.
- Shared pipeline execution semantics for logging, cooldown, and failure propagation.
- The low-level ingest adapter is non-destructive by default, but the operator-facing runtime task path (`main.py --run ingest`, `evening`, and `full`) explicitly enables `delete_after_ingest=True` for one-shot buffer consumption.
- Explicit structured result metadata for fetch, ingest, export, predict, and portfolio outputs.
- Direct-call model-side runtime adapters instead of subprocess-based orchestration.
- Deterministic prediction-pool membership for the same data snapshot, with reproducible training-universe downsampling isolated inside `model_function/`.

Testing strategy:
- Unit tests cover runtime foundation modules (`runtime.config`, `runtime.runlog`, `runtime.registry`, `runtime.orchestrator`, `runtime.tasks`).
- Adapter and service tests cover fetch/ingest/export/model behavior with mocked heavy dependencies.
- CLI-wrapper tests protect the user-facing script surface.
- Pipeline tests verify ordering, cooldown, and failure semantics without requiring real market-data or training runs.

## 6. Compatibility Notes

Kept compatible:
- `main.py --run ...` and `main.py --status`
- Stable task names and pipeline names from `runtime.constants`
- `python -m scripts.<name>` entrypoints

Changed intentionally:
- Runtime task ownership now lives in `runtime.tasks` rather than deleted scheduler wrappers.
- `runtime.services` now owns the data/model service classes that previously sat behind `quantcore.services`.
- Historical compatibility modules such as `quantcore/*`, `config/settings.py`, `utils/run_tracker.py`, `scheduler/*`, and `data_pipeline/ingest.py` are no longer part of the active architecture.
