# Exec Plan: Python Runtime Stability Overhaul

- Date: 2026-04-01
- Task UUID: `70554382-1a2b-4e2b-b774-79d96320d03a`

## Goal
Rebuild the Python-side runtime into a clearer, more stable architecture focused on operational reliability, testability, and maintainability, while keeping `server/` and `news_module/` untouched.

## Scope
- In scope:
  - Replace the current thin `quantcore` orchestration with a new runtime architecture centered on typed execution context, typed task results, explicit ports, and a single orchestrator.
  - Refactor data, model, CLI, and scheduling paths onto one shared execution stack.
  - Remove redundant or awkward interfaces where they are no longer useful.
  - Strengthen automated tests for task units, workflow integration, and failure paths.
  - Update English and Chinese docs, guides, plans, and logs to reflect the new system.
- Out of scope:
  - `server/` C++ gateway changes.
  - `news_module/` activation or redesign.

## Assumptions
- The current code and tests are the behavioral source of truth where docs disagree.
- Backward compatibility is not required if a cleaner interface materially improves stability and usability.
- Heavy external dependencies (`baostock`, gateway HTTP, Qlib/MLflow) should be isolated behind adapters and covered by offline-friendly tests where possible.
- The `conda quant` environment is the required verification environment.

## Steps
1. Audit the current Python runtime and freeze the target architecture, risks, and migration phases with subagent review.
2. Introduce a new runtime package with:
   - typed domain models
   - structured run logging primitives
   - port interfaces
   - a single workflow orchestrator
3. Migrate model-side execution (`dump`, `train`, `predict`, `portfolio`) off subprocess/script-driven orchestration and onto direct Python use cases.
4. Migrate data-side execution (`fetch`, `ingest`, `export`) onto typed adapters with explicit schema normalization and artifact handling.
5. Cut `main.py`, CLI entry points, and scheduling paths over to the new orchestrator, then remove redundant legacy orchestration code.
6. Expand verification with focused unit tests, workflow integration tests, and failure-path tests in `conda quant`.
7. Publish a clear bilingual guide for the new runtime structure and usage.
8. Move this plan to `docs/exec-plans/done/` when the refactor, docs, logs, critic report, and verification are complete.

## Acceptance Criteria
- Python runtime flow is driven by one orchestrator instead of split scheduler/script paths.
- Task boundaries are explicit and typed, with structured task results and persistent run metadata.
- Model and data workflows no longer depend on subprocess-based orchestration for core runtime behavior.
- Redundant legacy orchestration code is removed or reduced to thin compatibility shims outside the critical path.
- Changed behavior is covered by automated tests, including key failure scenarios.
- Documentation includes architecture updates plus a practical guide in English and Chinese.
- Structured logs include UUID traceability and a critic report from another subagent.

## Rollback Notes
- Revert the new runtime package, entry-point rewiring, and updated tests/docs if the cutover proves unstable.
- Restore the prior `quantcore` + `scheduler` orchestration path if the new orchestrator fails acceptance.
- If rolling back fully, remove the plan/log artifacts for this UUID and restore the previous navigation/workstream status documents.
