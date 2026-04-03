# Product Spec: Python Runtime V2

- Version: v2
- Date: 2026-03-31
- Applies to: Python orchestration/runtime modules

## 1. Goals

1. Provide stable, testable orchestration for data -> model -> portfolio pipelines.
2. Keep existing command surface compatible.
3. Support efficient validation with lightweight test doubles.

## 2. User-Facing Contract

### CLI Compatibility

`main.py` keeps:
- `--run evening`
- `--run afternoon`
- `--run full`
- `--run fetch|ingest|export|dump|train|predict|portfolio`
- `--status`

### Script Compatibility

Core scripts remain callable via `python -m scripts.<name>`.

## 3. Functional Requirements

### FR-1 Configuration

- Runtime config must be loaded from `.env` with defaults.
- `runtime.config` is the canonical config implementation.
- Runtime can refresh env-based cooldown settings without process restart.

### FR-2 Run History

- Task runs must persist to `.data/run_history.json`.
- `runtime.runlog.RunLogStore` is the canonical run-history store.
- The store must keep legacy flat run-history files readable while writing the structured format.
- The supported helper API now lives in `runtime.runlog` (`load_run_history`, `save_run_history`, `record_run`, `get_last_run`, `today`, `today_dash`).

### FR-3 Pipeline Execution

- Pipelines run tasks sequentially.
- Any task failure aborts the pipeline.
- One-shot CLI pipeline runs must surface failure to the caller as a non-zero process exit.
- Cooldown is applied between consecutive tasks.

### FR-4 Data Ingest Safety

- The low-level ingest adapter must keep source CSVs by default.
- Destructive ingest behavior must remain explicit at the call site via `delete_after_ingest=True`.
- The operator-facing runtime path may intentionally enable destructive ingest for one-shot packaged-buffer consumption.
- Data-side runtime behavior (`fetch`, `ingest`, `export`) must execute through runtime-owned adapters with explicit result metadata.

### FR-5 Testability

- Heavy operations must be mockable at service boundaries.
- Data and model runtime behavior should execute through direct runtime adapters instead of subprocess shells or inline service orchestration where a runtime adapter exists.
- Project must include both unit tests and integration-style pipeline tests.

## 4. Non-Functional Requirements

- Keep module boundaries clear and dependency direction predictable.
- Preserve the CLI and script command surface while allowing obsolete Python import shims to be deleted.
- Keep refactor reversible.

## 5. Acceptance

The implementation is accepted when:

1. Existing CLI contracts still work.
2. New architecture modules (`runtime/*`) are active in the main runtime path.
3. Former compatibility-owned data/model services, config access, and run-history helpers are owned directly by `runtime.services`, `runtime.config`, and `runtime.runlog`.
4. Data-side and model-side workflow logic is runtime-adapter owned for `fetch`, `ingest`, `export`, `dump`, `predict`, and `portfolio`.
5. Full test suite passes in `conda quant` environment.
6. Bilingual documentation reflects the implemented behavior, including a practical runtime guide.
