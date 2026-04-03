# Exec Plan: Runtime Compatibility Removal

- Date: 2026-04-02
- Task UUID: `0c5ab334-1f6f-4f4e-be77-922c86df198d`

## Goal
Delete the remaining Python-side compatibility layers that were kept only as transitional shims, so the runtime path is directly owned by canonical runtime modules instead of `quantcore`, `config.settings`, `utils.run_tracker`, `scheduler`, and other legacy wrappers.

## Scope
- In scope:
  - Remove compatibility-only runtime shims where current code no longer needs them.
  - Rewire runtime dispatch, task entrypoints, scripts, and tests onto canonical runtime modules.
  - Delete redundant legacy imports, wrappers, and adapter-only modules when safe.
  - Finish the last active compatibility cutover for config access, run-history helpers, and service placement.
  - Update English and Chinese docs, logs, and plans to reflect the slimmer post-compat architecture.
- Out of scope:
  - `server/`
  - `news_module/`
  - unrelated product behavior changes

## Assumptions
- Backward compatibility is no longer required for these Python import paths for in-repo consumers if the canonical runtime path remains clear and tested.
- Current code and tests remain the behavioral source of truth.
- The compatibility cleanup should preserve task names, pipelines, and script command surfaces unless a cleaner direct path is strictly internal.
- Remaining compatibility surfaces may be deleted in this task if their direct consumers are migrated onto canonical runtime modules in the same change.

## Frozen Boundary
- Delete in this task:
  - `quantcore/__init__.py`
  - `quantcore/services/__init__.py`
  - `quantcore/services/data_service.py`
  - `quantcore/services/model_service.py`
  - `scheduler/data_tasks.py`
  - `scheduler/model_tasks.py`
  - `scheduler/pipelines.py`
  - `scheduler/decorator.py`
  - `scheduler/__init__.py`
  - `quantcore/factory.py`
  - `quantcore/pipeline.py`
  - `quantcore/settings.py`
  - `quantcore/history.py`
  - `quantcore/registry.py`
  - `data_pipeline/ingest.py`
  - `config/settings.py`
  - `utils/run_tracker.py`
  - stale tests for removed legacy scripts and compatibility-only import paths
- Defer from this task:
  - broader runtime behavior changes unrelated to compatibility removal
  - historical docs/logs whose purpose is to preserve past architecture state

## Steps
1. Freeze the compatibility-removal boundary and record which surfaces are deleted now versus deferred.
2. Rehome the remaining service classes, config consumers, and run-history helpers into `runtime/*`.
3. Migrate tests from legacy import targets to runtime-native modules while preserving behavior contracts.
4. Delete the scheduler / quantcore / config / run-tracker / data_pipeline shim chain in one coherent slice once runtime-native replacements are green.
5. Remove stale tests that only target already-deleted legacy scripts.
6. Re-run targeted and full verification in `conda quant`.
7. Update bilingual docs/logs and move this plan to `docs/exec-plans/done/` after critic review and acceptance.

## Acceptance Criteria
- Runtime dispatch no longer depends on compatibility-only scheduler/task wrappers.
- Duplicate pipeline/orchestration wrappers in `scheduler` / `quantcore` are removed and replaced by runtime-native ownership.
- Active scripts and workflows no longer depend on `config/settings.py` or `utils/run_tracker.py`.
- Compatibility-only modules removed in this task are no longer referenced anywhere in code or tests.
- Docs describe the slimmer canonical runtime architecture accurately in English and Chinese.
- Full test suite passes in `conda quant`.
- Structured logs include UUID traceability and a critic report.

## Rollback Notes
- Restore deleted compatibility modules and previous runtime wiring if direct cutover breaks task dispatch or script behavior.
- Revert doc/log updates together with the code if the cleanup is rolled back.
