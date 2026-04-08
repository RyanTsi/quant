# Exec Plan: Architecture Remediation Roadmap

## Goal

Resolve the highest-leverage architecture issues identified in
`docs/logs/architecture-review-2026-04-03-16c156d1-a014-4fda-856a-1a1282797353.md`
through small, staged refactors that keep runtime behavior stable while
improving module boundaries, reviewability, and operational clarity.

## Scope

- Python model/runtime boundary work across `model_function/`,
  `runtime/adapters/modeling.py`, `runtime/services.py`, `alpha_models/*`,
  `scripts/view.py`, and `scripts/eval_test.py`
- Supporting test and documentation updates required to keep code and docs in
  sync
- Per-phase plans, structured logs, and subagent review loops

## Assumptions

- The current code and tests remain the source of truth for behavior.
- Refactors should preserve existing CLI/runtime contracts unless a phase
  explicitly changes them.
- The first phase should focus on extracting reusable training/prediction logic
  from `alpha_models/*` and `runtime/adapters/modeling.py` into
  `model_function/`, following the concepts documented in
  `docs/detailed_workflow.ipynb`.
- Each implementation phase will use two fresh GPT-5.4 subagents:
  one coding worker and one reviewer.

## Phases

### Phase 1: Model-function extraction for training and prediction

- Move notebook-aligned Qlib workflow assembly, recorder resolution, and
  prediction-dataset construction into `model_function/`.
- Reduce `runtime/adapters/modeling.py` to a thinner runtime adapter.
- Reduce `alpha_models/qlib_workflow.py` to a thinner workflow entrypoint.
- Remove the workflow-to-script dependency inversion by moving reusable
  post-train visualization behavior out of `scripts/view.py` ownership.

### Phase 2: Config and run-state boundary tightening

- Reduce direct `get_settings()`, `get_last_run()`, and `record_run()`
  reach-through across `alpha_models/*`, `runtime/adapters/*`, and `scripts/*`.
- Introduce clearer helper boundaries for recorder lookup and model artifact
  resolution.
- Keep runtime services as the primary state-recording boundary wherever
  practical.

### Phase 3: Trading-date and execution-context normalization

- Introduce a clearer execution-context contract for natural date versus trading
  date handling.
- Align prediction and portfolio previous-state lookup around the same trading
  calendar semantics.
- Replace wall-clock fallbacks where deterministic reruns need explicit dates.

### Phase 4: Operational hardening and documentation reconciliation

- Address scheduler and destructive-ingest follow-up items that can be improved
  without a broad rewrite.
- Reconcile stale architecture/design/test naming residue left from the
  runtime-first consolidation.
- Keep EN/ZH docs and navigation status aligned with the new structure.

## Acceptance Criteria

- Each phase has its own sub-plan, verification scope, and structured log.
- The model pipeline becomes easier to review because high-responsibility logic
  is split into smaller modules with clearer ownership.
- The workflow-to-script dependency inversion is removed.
- Tests cover new helper boundaries so later phases can continue from stable
  seams.
- Remaining follow-up items are explicitly tracked rather than hidden inside a
  single hotspot module.

## Rollback Notes

- Each phase should remain reversible by keeping wrapper-level contracts stable.
- If a phase introduces too much churn, stop after extracting helpers and keep
  compatibility wrappers in place until the next phase.
