# Navigation Content: Workstream Backlog

This file captures active status/backlog summaries.

## 1. Active Workstreams

| Workstream | Scope | Current status |
|---|---|---|
| Python runtime architecture | `quantcore`, `scheduler`, `scripts`, `config`, `utils` | Refactored and active |
| Data pipeline reliability | `data_pipeline/*` | Active, keep incremental hardening |
| Model pipeline usability | `alpha_models`, `scripts/predict.py`, `scripts/view.py` | Active, post-train view enforced |
| Test completeness | `test/*` | Active, broad unit/integration coverage |
| Documentation navigation | `docs/NAVIGATION*`, `docs/navigation-docs/*` | Active and authoritative |

## 2. Deferred / Out-of-Scope Streams

| Stream | Reason |
|---|---|
| `server/*` deep refactor | excluded by current Python refactor scope |
| `news_module/*` activation | module is deprecated/WIP and isolated |
| RL portfolio productionization | placeholder stage |

## 3. Update Rule

When status changes:
1. Update this file.
2. If scope/module routing changed, update `module-index`.
3. If system topology changed, update `system-map`.
