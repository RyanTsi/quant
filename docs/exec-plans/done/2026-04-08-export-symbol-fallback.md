# Execution Plan: Export Symbol Fallback

- Goal: Keep `main.py --run export` working when the gateway `GET /api/v1/symbols` call times out or disconnects.
- Scope: Python export adapter, data service wiring, focused tests, and runtime-guide documentation.
- Assumptions:
  - Per-symbol gateway queries remain healthier than the full symbol-list query.
  - Local symbol artifacts under `.data/` are recent enough to serve as the preferred export symbol source.
- Steps:
  1. Confirm the failing runtime step and identify the adapter/service boundary.
  2. Prefer local symbol artifacts for export symbol resolution and keep gateway listing as a secondary fallback.
  3. Update tests for adapter behavior and service wiring.
  4. Document the fallback and record trace artifacts.
  5. Run narrow verification plus one fallback-based real export probe.
- Acceptance Criteria:
  - Export no longer hard-fails when gateway symbol listing fails.
  - Existing export result metadata stays unchanged.
  - Focused tests cover fallback loading and service argument wiring.
- Rollback Notes:
  - Revert the export-adapter fallback helper and the service wiring if the fallback causes incorrect symbol coverage.
