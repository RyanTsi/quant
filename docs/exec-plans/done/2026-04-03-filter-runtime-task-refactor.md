# Exec Plan: Filter Runtime Task Refactor

- Status: completed on 2026-04-03

## Goal

Move the main training-universe build flow out of `scripts/filter.py` into
`model_function/`, keep the script as a thin operator-facing wrapper, and add a
runtime-managed task so the flow can be dispatched through the canonical
registry.

## Scope

- `model_function/universe.py`
- `model_function/__init__.py`
- `runtime/services.py`
- `runtime/tasks.py`
- `runtime/bootstrap.py`
- `runtime/constants.py`
- `scripts/filter.py`
- focused tests and minimal doc updates tied to the new runtime task

## Assumptions

- The current training-universe behavior remains the source of truth unless
  changed intentionally by tests in this task.
- `scripts/filter.py` should remain CLI-compatible after the refactor.
- The new runtime task should be additive and should not change existing
  pipelines unless explicitly wired into one.

## Steps

1. Extract the month-lagged training-universe build flow into reusable
   `model_function` helpers.
2. Refactor `scripts/filter.py` into a thin wrapper over the shared helper.
3. Add a `ModelPipelineService` entry plus a runtime task/registry mapping for
   the filter flow.
4. Update focused tests for the model helper, runtime task wiring, and script
   compatibility.
5. Update minimal docs and trace artifacts, then move this plan to
   `docs/exec-plans/done/` after verification.

## Acceptance Criteria

- `scripts/filter.py` no longer owns the main training-universe implementation.
- The core training-universe construction path lives in `model_function/`.
- `main.py --run filter` dispatches through the runtime registry successfully.
- Focused unit tests covering the refactor pass.

## Verification

- `python -m unittest test.test_model_function_universe test.test_filter_stocks test.test_model_cli_wrappers test.test_runtime_tasks test.test_runtime_bootstrap test.test_model_pipeline_service`
- `python -m unittest discover -s test -p 'test_*.py'`

## Rollback Notes

- Restore the previous `scripts/filter.py` implementation if the extracted
  helper changes behavior unexpectedly.
- Remove the new runtime task registration if the task surface proves
  misleading or unstable.
