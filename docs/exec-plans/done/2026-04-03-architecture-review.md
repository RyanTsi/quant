# Exec Plan: Architecture Review

## Goal

Review the current project structure and software architecture, compare the
documented runtime-first design with the implementation, and prepare a grounded
assessment of strengths, weaknesses, and high-leverage improvement ideas.

## Scope

- Documentation entrypoints and navigation content required by `AGENTS.md`
- Representative modules across:
  - `main.py`
  - `runtime/*`
  - `runtime/adapters/*`
  - `model_function/*`
  - `data_pipeline/*`
  - `alpha_models/*`
  - `scripts/*`
  - `backtesting/*`
  - `test/test_*.py`
- Trace artifacts for this review under `docs/exec-plans/*` and `docs/logs/*`

## Assumptions

- The user requested an architectural review, not a production behavior change.
- Current code and passing tests are treated as the source of truth when older
  docs or naming residue differ.
- The worktree may contain in-progress changes, so conclusions should separate
  stable architectural signals from active refactor noise.

## Steps

1. Read navigation and core architecture docs in the required order.
2. Inspect representative runtime, data, model, CLI, and test modules to map
   actual dependency direction and layering.
3. Run focused architecture-related tests to validate the main runtime and CLI
   surfaces.
4. Record bilingual trace artifacts and summarize strengths, weaknesses, and
   prioritized improvement ideas for the user.

## Acceptance Criteria

- The review is grounded in the current repository implementation, not only the
  intended docs.
- Strengths and weaknesses reference concrete architectural surfaces.
- Focused verification results are recorded.
- Bilingual trace artifacts are stored under `docs/exec-plans/*` and
  `docs/logs/*`.

## Rollback Notes

- Documentation-only task. Remove the added trace files if they are no longer
  needed.
