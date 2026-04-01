# Exec Plan: Python Core System Refactor

- Date: 2026-03-31
- Task UUID: f731efc3-eb41-48f8-959e-b4a518abf941

## Goal
Refactor the Python side of the project into a clearer, layered architecture (excluding `server/` and deprecated `news_module/`) with stronger maintainability, compatibility, and complete automated tests.

## Scope
- In scope:
  - Rebuild orchestration architecture for config, runtime state, task execution, and pipeline composition.
  - Migrate `scheduler/`, `scripts/`, `data_pipeline/`, `alpha_models/` call paths onto shared services.
  - Keep existing CLI behavior compatible where possible.
  - Add unit tests and end-to-end style pipeline tests using small-scale data/mocked heavy operations.
  - Update bilingual documentation to match new architecture.
- Out of scope:
  - `server/` C++ gateway refactor.
  - `news_module/` feature revival (deprecated).

## Assumptions
- Production behavior remains based on current code/tests where docs and code diverge.
- Heavy tasks (full market fetch and real model training) are replaced with lightweight test doubles in automated tests.
- Existing dirty working tree contains unrelated changes and should not be reverted.

## Steps
1. Build new core modules for settings, run-history persistence, task interfaces, and pipeline runner.
2. Refactor scheduler tasks into shared service layer and remove duplicated orchestration logic.
3. Refactor scripts to use service layer entry points with thin CLI wrappers.
4. Fix known test fragility uncovered in baseline (`ingest` cleanup coupling, static cooldown reading).
5. Add/upgrade unit tests for each core layer and integration tests for full pipeline execution with mocks.
6. Update architecture and usage docs in English and Chinese.
7. Write structured bilingual task logs and close plan by moving to `docs/exec-plans/done/`.

## Acceptance Criteria
- Python architecture is modular and layered with clear responsibilities.
- Existing command entry points still run expected tasks.
- Test suite includes complete unit and integration coverage for refactored flow.
- Baseline test failures are resolved.
- Bilingual docs reflect the refactor and are UTF-8 encoded.

## Rollback Notes
- Revert refactor files under:
  - `config/`, `scheduler/`, `scripts/`, `data_pipeline/`, `alpha_models/`, `utils/`, `test/`, and updated docs.
- Restore previous task wiring by rolling back `main.py` and scheduler modules.
- Remove plan/log artifacts for this UUID if rollback is full.
