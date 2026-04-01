# Exec Plan: Pool Logic Reuse + Predict Index Exclusion + Label Alignment

- Date: 2026-03-31
- Task UUID: f7968da5-dae1-4754-893f-acf535b97fd8

## Goal
Extract shared stock-pool preprocessing steps into reusable function(s), ensure prediction pool excludes index instruments, and align prediction label expression with training label.

## Scope
- In scope:
  - Shared utility extraction for pool sampling/liquidity-related common logic.
  - Apply shared utility in `scripts/filter.py` and `scripts/predict.py`.
  - Exclude index symbols from prediction pool generation and previous-day pool expansion.
  - Align prediction label expression with training config label.
  - Add/update targeted tests.
- Out of scope:
  - Redesigning pool strategy logic.
  - Changing model architecture or retraining workflow.

## Assumptions
- `settings.data_path/index_code_list` is the canonical local list of index symbols.
- Current training label in `alpha_models/workflow_config_transformer_Alpha158.yaml` is the source-of-truth label semantics.
- Minimal, reversible changes are preferred.

## Steps
1. Add shared utility function(s) under `utils/` for common ranked-pool sampling and reusable label constant.
2. Refactor `scripts/filter.py` and `scripts/predict.py` to use shared utility.
3. Add index exclusion in prediction pool ranking and previous-day expansion.
4. Align prediction label string to training label constant.
5. Add/adjust unit tests for sampling reuse, index exclusion, and label alignment.
6. Run narrow tests and verify.

## Acceptance Criteria
- Prediction pool does not include symbols from `index_code_list`.
- Training and prediction use the same Alpha158 label expression.
- Shared logic is implemented once and reused by both training-pool and prediction-pool generation paths.
- Relevant tests pass.

## Rollback Notes
- Revert touched files:
  - `utils/preprocess.py`
  - `scripts/filter.py`
  - `scripts/predict.py`
  - test files added/modified in `test/`
- Remove this plan if rollout is canceled.
