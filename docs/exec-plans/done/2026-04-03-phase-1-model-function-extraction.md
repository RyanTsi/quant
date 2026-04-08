# Exec Plan: Phase 1 Model-Function Extraction

## Goal

Extract the reusable training, recorder-resolution, prediction, and
post-training analysis logic from `alpha_models/*` and
`runtime/adapters/modeling.py` into `model_function/`, while preserving current
runtime behavior.

## Scope

- `model_function/`
- `runtime/adapters/modeling.py`
- `alpha_models/qlib_workflow.py`
- `scripts/view.py`
- `scripts/eval_test.py`
- related tests under `test/`
- only the docs needed to describe the new ownership

## Assumptions

- `alpha_models/workflow/runner.py` remains the low-level Qlib workflow runner in
  this phase.
- Runtime services keep owning run-history writes for `predict` and
  `build_portfolio`.
- The phase should not change the trained model config, output file names, or
  user-facing CLI commands.
- Reusable model-domain logic belongs in `model_function/`; runtime orchestration
  remains in `runtime/`.

## Steps

1. Create model-function helpers for training workflow orchestration.
   - Add a reusable training entry that resolves config source, runs the
     workflow runner, and returns structured training metadata.
   - Move reusable post-train visualization behavior under model-domain
     ownership so workflow code no longer imports `scripts.view`.

2. Create model-function helpers for prediction and recorder access.
   - Add reusable recorder-id resolution and trained-model loading helpers.
   - Add notebook-aligned prediction dataset assembly helpers for Alpha158 /
     `TSDatasetH` style scoring.
   - Keep deterministic universe selection in the existing universe module.

3. Refactor wrappers to delegate downward.
   - Make `alpha_models/qlib_workflow.py` a thin entrypoint over the new
     training helper.
   - Make `runtime/adapters/modeling.py` delegate prediction-specific Qlib
     assembly to `model_function/`.
   - Make `scripts/view.py` and `scripts/eval_test.py` thin wrappers over shared
     model-function helpers where behavior overlaps.

4. Add or update tests.
   - Add direct tests for the new model-function helpers.
   - Keep wrapper tests focused on delegation and stable public behavior.
   - Preserve or improve the existing runtime service and adapter contract tests.

5. Update trace docs.
   - Record the phase log and reviewer report.
   - Update architecture-facing docs only if ownership claims changed.

## Acceptance Criteria

- `alpha_models/qlib_workflow.py` no longer imports `scripts.view`.
- Prediction dataset assembly and recorder resolution are reusable outside
  `runtime/adapters/modeling.py`.
- Runtime-facing behavior for prediction, training entry, and view generation is
  preserved by tests.
- The changed modules are smaller and have clearer responsibility boundaries.

## Verification

- Narrow unit tests for new `model_function` helpers
- Existing wrapper/service tests covering:
  - `test/test_qlib_workflow.py`
  - `test/test_model_pipeline_service.py`
  - `test/test_model_cli_wrappers.py`
  - `test/test_view_script.py`
  - `test/test_workflow_runner.py`
  - `test/test_modeling_adapter_contract.py`

## Rollback Notes

- If the extraction becomes unstable, keep compatibility wrappers in
  `alpha_models/` and `runtime/adapters/modeling.py` that forward to the
  previous implementation shape while retaining the new tests.
