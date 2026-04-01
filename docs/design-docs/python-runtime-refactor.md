# Design Doc: Python Runtime Refactor

- Date: 2026-03-31
- Scope: Python side only (`server/` and `news_module/` excluded)

## Problem

Before refactor, orchestration responsibilities were spread across:
- scheduler task modules,
- standalone scripts,
- utility modules with direct side effects.

This caused duplicated logic, harder testing, and configuration/state coupling.

## Design Goals

1. Centralize runtime orchestration in one reusable core layer.
2. Keep existing CLI/task contracts compatible.
3. Make heavy side effects mockable for tests.
4. Keep migration reversible and low-risk.

## Target Architecture

### Core Runtime (`quantcore/`)

- `settings.py`
  - Typed runtime config (`AppSettings`)
  - Cached + refreshable settings access
- `history.py`
  - JSON-backed run history store (`RunHistoryStore`)
  - Atomic file replacement on save
- `pipeline.py`
  - Unified pipeline executor (`PipelineRunner`)
- `services/data_service.py`
  - Fetch, ingest, export orchestration
- `services/model_service.py`
  - Dump, train, predict, portfolio orchestration
- `registry.py`
  - Task/pipeline name registry for CLI dispatch

### Compatibility Layer

- `config/settings.py`
  - Re-export new settings model for legacy imports.
- `utils/run_tracker.py`
  - Keep old helper API; delegate to `RunHistoryStore`.

### Adapters

- `scheduler/*`
  - Task wrappers are now thin service calls.
  - Pipelines use a shared runner and runtime cooldown lookup.
- `main.py`
  - Uses registry-driven dispatch for one-shot runs.

## Key Behavior Changes

1. Pipeline cooldown can pick up runtime env overrides (`PIPELINE_COOLDOWN_SECONDS`) per run.
2. Ingest no longer deletes local CSV by default (`delete_after_ingest=False`); callers opt in explicitly.
3. Side effects are concentrated in service layer methods.

## Testing Strategy

Unit tests:
- Core settings
- Data/model services
- Registry and dispatch
- Existing utility/domain tests

Integration-style tests:
- Full pipeline order using mocked services
- End-to-end dispatch path (`main.run_once` -> registry)

Heavy operations:
- Full market fetch and real model training are mocked in automated tests.

## Alternatives Considered

1. Full package rename + path migration for all modules.
   - Rejected: high churn, weak backward compatibility.
2. Keep old architecture and patch only failing tests.
   - Rejected: does not address maintainability/structure goals.

## Rollback Plan

Revert:
- `quantcore/`
- updated scheduler/config/utils/main modules
- tests and docs added for the refactor
