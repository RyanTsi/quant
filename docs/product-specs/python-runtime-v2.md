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
- Runtime can refresh env-based cooldown settings without process restart.

### FR-2 Run History

- Task runs must persist to `.data/run_history.json`.
- Existing helper API (`utils.run_tracker`) must continue working.

### FR-3 Pipeline Execution

- Pipelines run tasks sequentially.
- Any task failure aborts the pipeline.
- Cooldown is applied between consecutive tasks.

### FR-4 Data Ingest Safety

- Ingest API must not delete source CSV by default.
- Deletion behavior must be explicit via `delete_after_ingest=True`.

### FR-5 Testability

- Heavy operations must be mockable at service boundaries.
- Project must include both unit tests and integration-style pipeline tests.

## 4. Non-Functional Requirements

- Keep module boundaries clear and dependency direction predictable.
- Preserve backward compatibility for key import paths.
- Keep refactor reversible.

## 5. Acceptance

The implementation is accepted when:

1. Existing CLI contracts still work.
2. New architecture modules (`quantcore/*`) are active in runtime path.
3. Full test suite passes in `conda quant` environment.
4. Bilingual documentation reflects the implemented behavior.
