# Exec Plan: Exclude ST Stocks in Training and Prediction

## Goal

Ensure both model training and prediction exclude ST stocks by default in the project runtime flow.

## Scope

- `scripts/filter.py` (training universe generation path)
- `scripts/predict.py` (prediction pool construction path)
- Related unit tests in `test/test_filter_stocks.py` and `test/test_predict_pool.py`
- Lightweight documentation updates

## Assumptions

- Source market rows include `isST` (ingest DB side), and dumped Qlib features expose it as `$isst`.
- Training universe is generated via `scripts/filter.py` and consumed by workflow config market `my_800_stocks`.
- ST exclusion should apply conservatively: if a stock is ST for a relevant period/snapshot, exclude it.

## Steps

1. Add ST-quarter exclusion to liquidity-based universe selection in `scripts/filter.py`.
2. Add ST-symbol exclusion to prediction pool ranking and previous-day carryover in `scripts/predict.py`.
3. Keep output path handling robust for tests and runtime (ensure instrument output directory exists).
4. Add/update tests for:
   - filter behavior when ST appears in quarter rows
   - prediction pool exclusion with `$isst`
5. Run focused tests and adjust as needed.
6. Update docs and trace logs, then move this plan to `docs/exec-plans/done/`.

## Acceptance Criteria

- Training-universe generation excludes ST quarters/symbols from selected output.
- Prediction pool excludes ST symbols both from same-day liquidity candidates and carryover candidates.
- Existing related tests pass with new ST exclusions.
- Docs/logs reflect the behavior update.

## Rollback Notes

- Revert changes in `scripts/filter.py`, `scripts/predict.py`, and updated tests if exclusion behavior causes unexpected downstream regressions.

