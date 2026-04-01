# Exec Plan: Navigation Rules Upgrade

- Date: 2026-03-31
- Task UUID: 1bb80e51-c329-42c8-9c12-bccb536118ae

## Goal
Establish a unified Navigation rule system with explicit graph structure and clear execution procedure.

## Scope
- In scope:
  - Add `docs/NAVIGATION.md` as the top Navigation contract.
  - Add `docs/navigation-docs/*` as graph and rule detail docs.
  - Add Chinese counterparts under `docs/` with `_zh` suffix.
  - Update `AGENTS.md` to explicitly require reading Navigation docs.
- Out of scope:
  - Code/runtime behavior changes.
  - Server/news module changes.

## Assumptions
- Navigation docs become the authoritative doc-routing system.

## Steps
1. Create top-level Navigation docs (EN + ZH) with graph + execution rules.
2. Create detailed navigation docs under `docs/navigation-docs/` (EN + ZH).
3. Update `AGENTS.md` source-of-truth order and explicit Navigation rule.
4. Update docs index links.
5. Create task logs and move plan to `done/`.

## Acceptance Criteria
- `docs/NAVIGATION.md` and `docs/navigation-docs/*` exist with clear graph-based guidance.
- `AGENTS.md` explicitly instructs readers to follow Navigation docs.
- NAVIGATION includes explicit “how to execute this rule set”.
- Chinese versions are present and consistent.

## Rollback Notes
- Revert:
  - `AGENTS.md`
  - `docs/NAVIGATION.md`, `docs/NAVIGATION_zh.md`
  - `docs/navigation-docs/*`
  - optional index link updates
