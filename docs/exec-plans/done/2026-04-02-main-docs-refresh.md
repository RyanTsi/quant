# Exec Plan: Main Documentation Refresh

- Date: 2026-04-02
- Task UUID: `9db9b42a-0c47-4c94-815b-180cec6fc376`

## Goal
Refresh the project's main English and Chinese documentation so it matches the current runtime-first codebase after the compatibility-removal changes.

## Scope
- In scope:
  - `README.md`
  - `docs/README_zh.md`
  - `ARCHITECTURE.md`
  - `docs/ARCHITECTURE_zh.md`
  - `docs/index.md`
  - `docs/index_zh.md`
  - `docs/navigation-docs/*`
  - `docs/python-runtime-guide*.md`
  - `docs/product-specs/python-runtime-v2*.md` when active contract wording drifts from code
  - trace artifacts for this documentation task
- Out of scope:
  - behavior changes in Python runtime, training, or gateway code
  - historical plans/logs that intentionally capture past architecture states

## Assumptions
- Current code and tests are the source of truth for present behavior.
- Historical documents may remain historical; only active/main docs need present-tense updates.
- The current runtime no longer depends on `scheduler/`, `quantcore.settings`, `quantcore.history`, `quantcore.registry`, `data_pipeline/ingest.py`, or `news_module/`.

## Steps
1. Read navigation docs and current runtime entrypoints to identify factual drift.
2. Update the main EN/ZH docs to reflect the current file layout, runtime ownership, CLI entrypoints, and configuration surface.
3. Refresh navigation content so module routing matches the live repository tree.
4. Reconcile any critic-found drift in active runtime/product docs before closure.
5. Run focused verification for doc-linked runtime surfaces and consistency checks.
6. Record bilingual structured logs, attach critic review, and move this plan to `docs/exec-plans/done/`.

## Acceptance Criteria
- Main docs describe the current runtime-first architecture accurately in English and Chinese.
- Navigation docs no longer route through deleted modules.
- Config and usage sections reference live code paths and supported environment variables.
- Verification covers the main runtime entrypoints discussed in the docs.
- Structured logs include UUID traceability and critic feedback.

## Rollback Notes
- Revert the documentation-only changes and trace artifacts together if the refresh proves inaccurate.
