# Navigation Content: Workstream Backlog

This file captures active status and backlog summaries for the current codebase.

## 1. Active Workstreams

| Workstream | Scope | Current status |
|---|---|---|
| Python runtime architecture | `main.py`, `runtime`, `scripts`, `data_pipeline`, `alpha_models`, `utils` | Runtime-first ownership is complete; the remaining compatibility shims were removed on 2026-04-02 and active work is now limited to behavior polish plus doc/test alignment |
| Data pipeline reliability | `runtime/adapters/fetching.py`, `runtime/adapters/ingest.py`, `runtime/adapters/exporting.py`, `data_pipeline/*` | Active; focus remains on fetch-window correctness, ingest safety, and gateway failure reporting |
| Model pipeline usability | `model_function`, `alpha_models`, `runtime/adapters/modeling.py`, `scripts/filter.py`, `scripts/predict.py`, `scripts/build_portfolio.py`, `scripts/view.py`, `scripts/eval_test.py` | Active; deterministic universe construction now lives in `model_function/`, and the current focus is keeping train/predict/portfolio behavior and operator UX aligned with that shared contract |
| Test completeness | `test/*` | Active; runtime-centric coverage now includes bootstrap, registry, runlog, adapters, CLI wrappers, and pipeline semantics |
| Documentation navigation | `docs/NAVIGATION*`, `docs/navigation-docs/*`, `docs/python-runtime-guide*.md`, main docs | Refreshed on 2026-04-02 and actively reconciled against code; keep EN/ZH docs synchronized as runtime details continue to settle |

## 2. Deferred / Out-of-Scope Streams

| Stream | Reason |
|---|---|
| `server/*` deep refactor | excluded from current Python documentation/runtime tasks |
| RL portfolio productionization | placeholder stage only |

## 3. Update Rule

When status changes:
1. Update this file.
2. If scope or routing changed, update `module-index`.
3. If topology changed, update `system-map`.
