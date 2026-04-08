# Exec Plan: Phase 2 Config and Run-State Boundary Tightening

## Goal

Reduce direct `get_settings()`, `get_last_run()`, and `record_run()`
reach-through across the model pipeline by moving runtime-owned state and
fallback resolution into explicit runtime-side helpers and services.

## Scope

- `runtime/services.py`
- new runtime-owned model state helper module(s)
- `alpha_models/qlib_workflow.py`
- `runtime/adapters/modeling.py`
- `scripts/view.py`
- `scripts/eval_test.py`
- `scripts/build_portfolio.py`
- focused tests and any directly affected docs

## Assumptions

- Phase 1 helper extraction in `model_function/qlib.py` remains the reusable
  model-domain boundary.
- Runtime services should own run-history writes wherever practical.
- Operator-facing CLI contracts should stay stable.
- The change should stay reversible by keeping wrapper entrypoints intact.

## Steps

1. Introduce runtime-owned helper boundaries for model-state resolution.
   - Add a small runtime module that resolves:
     - training workflow settings inputs
     - latest trained recorder identity from env/runlog/settings
     - training run-history payload recording

2. Refactor training ownership.
   - Make `alpha_models/qlib_workflow.py` a workflow wrapper that no longer
     writes `qlib_train` directly.
   - Move `qlib_train` recording into `runtime.services.ModelPipelineService`.

3. Refactor predict/view/eval/portfolio wrappers to use runtime-owned state
   resolution.
   - Remove direct `get_last_run()` reach-through from scripts/adapters where
     the runtime helper can supply the same behavior.
   - Move portfolio run-history writes out of `runtime/adapters/modeling.py`
     and into the service path by routing the CLI through `ModelPipelineService`
     where appropriate.

4. Add or update tests.
   - Add direct tests for the new runtime-side helper module.
   - Update wrapper/service tests to assert the new delegation and state
     ownership boundaries.

5. Update docs and trace artifacts.
   - Keep architecture/runtime docs aligned if training/view/eval ownership
     wording changes.

## Acceptance Criteria

- `alpha_models/qlib_workflow.py` no longer writes `qlib_train` directly.
- Runtime-owned helper(s) provide the recorder/config fallback resolution now
  needed by view/eval/predict/portfolio paths.
- `runtime/adapters/modeling.py` no longer writes runlog state directly for the
  normal operator path.
- Tests cover the new runtime-side state boundary and the affected wrappers.

## Verification

- Direct tests for the new runtime-side helper module
- Updated focused model/runtime tests, including:
  - `test/test_model_pipeline_service.py`
  - `test/test_qlib_workflow.py`
  - `test/test_view_script.py`
  - `test/test_eval_test_script.py`
  - `test/test_model_cli_wrappers.py`
  - `test/test_modeling_adapter_contract.py`

## Rollback Notes

- Keep entrypoint signatures stable.
- If the state-helper extraction causes too much churn, keep compatibility
  wrappers but preserve service-owned runlog writes and direct tests.
