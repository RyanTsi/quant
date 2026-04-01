# Exec Plan: Monthly Pool with Stability, Volatility Filter, and Weighted Grouping

## Goal

Enhance monthly training stock-pool logic with:

- past-quarter liquidity stability scoring
- volatility extreme filtering (remove top/bottom 5%)
- 10-group non-uniform top-heavy selection with per-group minimum

## Scope

- `scripts/filter.py`
- `test/test_filter_stocks.py`
- `README.md`, `docs/README_zh.md`

## Assumptions

- Monthly refresh and anti-lookahead behavior remain unchanged.
- ST exclusion remains enabled.
- Output contract (`my_800_stocks.txt`) remains unchanged.

## Steps

1. Add past-quarter lookback metric computation per symbol per source month.
2. Add volatility band filter (5%-95%).
3. Replace uniform group sampling with weighted 10-group allocation while enforcing minimum coverage.
4. Update tests for new behavior.
5. Update docs and run focused tests.

## Acceptance Criteria

- Monthly pool uses past-quarter liquidity stability for ranking.
- Symbols in volatility top/bottom 5% are excluded per source month.
- 10 groups remain, with top-heavy non-uniform selection and minimum picks per group.
- Related tests pass.

## Rollback Notes

- Revert modified files if the new selection behavior causes downstream regressions.

