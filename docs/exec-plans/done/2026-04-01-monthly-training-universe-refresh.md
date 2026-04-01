# Exec Plan: Monthly Training Universe Refresh

## Goal

Change training stock-pool refresh cadence from quarter-based updates to month-based updates.

## Scope

- `scripts/filter.py` period routing and selection logic
- `test/test_filter_stocks.py` related behavior tests
- Documentation wording in `README.md` and `docs/README_zh.md`

## Assumptions

- Anti-lookahead design remains: source period determines next period membership.
- ST exclusion logic should remain active after cadence change.
- Output format (`my_800_stocks.txt`) remains unchanged.

## Steps

1. Refactor filter period helpers from quarter to month.
2. Keep one-period lag and partial-period merge behavior aligned to monthly periods.
3. Update tests to reflect monthly routing and ST exclusion on monthly periods.
4. Update docs mentioning quarter-lag behavior.
5. Run focused test suite and adjust if needed.
6. Move this plan to `docs/exec-plans/done/` after completion.

## Acceptance Criteria

- Training pool selection updates by month (not quarter).
- Lag behavior remains anti-lookahead (previous month -> next month).
- ST exclusion still applies under monthly cadence.
- Updated tests pass.

## Rollback Notes

- Revert changes in `scripts/filter.py`, `test/test_filter_stocks.py`, and docs if monthly cadence causes downstream issues.

