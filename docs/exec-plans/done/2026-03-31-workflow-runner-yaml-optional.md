# Exec Plan: Workflow Runner Refactor (YAML Optional)

- Date: 2026-03-31
- Task UUID: 3e83f7da-4b24-41d8-887e-d768bbaba772

## Goal
Refactor `alpha_models/workflow/runner.py` so training workflow can be controlled programmatically and is no longer fully dependent on YAML files, while preserving backward compatibility.

## Scope
- In scope:
  - Add config-loading path from Python dict/config objects.
  - Keep existing YAML path working.
  - Add override/merge support for runtime control.
  - Add tests for new non-YAML and compatibility behavior.
  - Update user-facing docs/changelog briefly.
- Out of scope:
  - Redesigning model architecture or training logic.
  - Scheduler pipeline redesign.

## Assumptions
- Existing `run_from_yaml` behavior must remain valid.
- `docs/detailed_workflow.ipynb` indicates desired step-by-step, code-driven control style.

## Steps
1. Extend runner config model and add generic config composition entry.
2. Add `run_from_config` and make `run_from_yaml` delegate to shared runtime logic.
3. Keep `alpha_models/qlib_workflow.py` unchanged or minimally touched for compatibility.
4. Add unit tests focused on config composition/merge and API compatibility.
5. Run targeted tests in `conda quant`.
6. Update changelog docs (EN + ZH).

## Acceptance Criteria
- Runner supports training execution from dict/config object without YAML file.
- Existing YAML workflow path still works.
- Runtime overrides are merged predictably.
- Added/updated tests pass.

## Rollback Notes
- Revert modified files:
  - `alpha_models/workflow/runner.py`
  - `test/test_workflow_runner.py` (new)
  - optional doc updates (`README.md`, `docs/README_zh.md`)
- Remove this plan if canceled.
