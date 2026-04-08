# Exec Plan: Phase 3 Trading-Date and Execution-Context Normalization

## Goal

Normalize model-pipeline date handling so prediction and portfolio stages use
the same trading-calendar semantics and the remaining date defaults are routed
through a clearer execution-context boundary.

## Scope

- `runtime/adapters/modeling.py`
- `runtime/services.py`
- `model_function/qlib.py` and/or a small shared helper module if needed
- focused model/runtime tests
- directly affected runtime and architecture docs

## Assumptions

- Phase 1 and Phase 2 boundaries remain the source of truth for helper and
  run-state ownership.
- Operator-facing CLI contracts should stay stable unless a deterministic
  rerun contract requires an explicit date.
- The smallest useful change is to centralize trading-date resolution instead
  of rewriting the full model pipeline.
- Prediction-pool continuity and portfolio continuity should both resolve
  previous-state artifacts from the same trading calendar.

## Steps

1. Introduce a shared trading-date / execution-context helper boundary.
   - Normalize:
     - trading-date selection
     - previous trading-date lookup
     - default execution-date fallback behavior

2. Refactor predict and portfolio paths to use the shared boundary.
   - Keep prediction-pool previous holdings lookup on trading-calendar
     semantics.
   - Replace portfolio natural-day previous-target lookup with the same
     trading-calendar semantics.

3. Tighten deterministic date defaults where needed.
   - Remove or narrow wall-clock fallbacks in paths that should prefer explicit
     dates or a shared runtime date resolver.
   - Preserve current operator UX where the fallback is still intentionally
     “latest available trading day”.

4. Add or update focused tests.
   - Cover weekend/holiday continuity behavior.
   - Cover shared helper failure semantics when a requested date is not in the
     local trading calendar.
   - Cover the service/adapter contract if any new execution-context arguments
     are introduced.

5. Update docs and trace artifacts.
   - Keep EN/ZH docs aligned on trading-date semantics and rerun behavior.

## Acceptance Criteria

- Prediction and portfolio previous-state lookup use the same prior trading-day
  semantics.
- The model pipeline no longer mixes trading-day and natural-day previous-file
  resolution for adjacent predict/portfolio stages.
- Shared helper tests make the date contract reviewable without reopening the
  entire adapter hotspot.
- Runtime docs describe the resulting default-date behavior accurately.

## Verification

- Focused model/runtime tests for:
  - `test/test_predict_pool.py`
  - `test/test_modeling_adapter_contract.py`
  - `test/test_model_pipeline_service.py`
- Any new targeted tests added for trading-date helpers or execution-context
  behavior

## Rollback Notes

- Keep public entrypoint signatures stable where possible.
- If a shared helper adds too much churn, keep it adapter-local but retain the
  unified trading-date semantics and direct tests.
