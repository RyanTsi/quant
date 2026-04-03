# Exec Plan: A-Share Low-Compute Universe Redesign

- Status: completed on 2026-04-03

## Goal

Define and later implement a more stable, deterministic, and compute-bounded stock-universe policy for the A-share OHLCVA-only model so that:

- training remains feasible on limited hardware
- prediction remains reproducible and non-random
- turnover control is handled explicitly by holding rules instead of by ad hoc pool merging
- future backtests can compare the incumbent and redesigned policies on the same contract

## Scope

- Research and implementation planning for:
  - shared model-domain universe contract under `model_function/`
  - monthly lagged training-universe construction
  - monthly lagged prediction-universe construction
  - explicit holding-buffer / rebalance-capacity rules
  - shared reusable universe helpers in the existing pool-generation path
  - targeted tests and documentation updates
- Implemented surfaces confirmed against live code/tests:
  - `model_function/universe.py`
  - `runtime/adapters/modeling.py`
  - `scripts/filter.py`
  - `scripts/build_portfolio.py`
  - related tests under `test/`
  - user-facing documentation in `README.md`, `docs/README_zh.md`, `ARCHITECTURE.md`, `docs/ARCHITECTURE_zh.md`, and `docs/navigation-docs/*`

Out of scope:

- adding non-OHLCVA features
- changing model architecture, labels, or training framework
- intraday execution simulation or broker integration
- rewriting the broader runtime pipeline

## Assumptions

- The project remains focused on China A-share daily-frequency stock selection.
- Anti-lookahead behavior must remain strict: any month `t` membership decision must be derived only from data available by the end of month `t-1`.
- Existing ST exclusion and index-exclusion safeguards remain active unless explicitly changed in a later task.
- Limited hardware remains a hard constraint, so the solution must preserve an upper bound on both training and prediction workload.
- The current monthly pool logic documented in prior plans remains the incumbent baseline until the redesigned policy is implemented and validated.
- Because this plan was created under a minimal-code-inspection constraint, any file ownership, metric availability, and validation capability assumptions remain provisional until confirmed by code/tests.

## Target Policy Contract

The numeric bands and metric choices below are starting hypotheses for later evaluation, not implementation-ready fixed standards. They are included to make the future work concrete and testable, but they must be confirmed against live code, available data fields, and current runtime ownership before rollout.

1. Separate the three decision layers.
   - Training universe answers: which stocks are eligible for model fitting.
   - Prediction universe answers: which stocks receive model scores today.
   - Holding buffer answers: which existing positions are allowed to survive rank drift.

2. Keep randomness only in training-time downsampling.
   - Training may use stratified downsampling to stay within compute limits.
   - Prediction must be deterministic for the same date snapshot and config.

3. Use a lagged liquidity metric that is robust to one-off spikes.
   - Prefer rolling median traded value over a short-to-medium lookback window such as 20 to 60 trading days.
   - If direct traded value is unavailable, derive it from close and volume using the data already available in the project.

4. Replace ad hoc previous-day merge logic with explicit buffer zones.
   - Example target contract for evaluation:
     - prediction entry band: top 1000 by lagged liquidity
     - prediction exit band: top 1200 by lagged liquidity
     - buy band: top 300 by model rank
     - hold band: top 500 by model rank
   - Final thresholds stay provisional until validation confirms they are workable for this project.

5. Keep the training universe broader than the prediction universe.
   - Example target contract for evaluation:
     - training entry band: top 1800 by lagged liquidity
     - training exit band: top 2200 by lagged liquidity
     - deterministic stratified sample down to 800 names if full training remains too expensive

## Steps

1. Confirm the real implementation ownership before any behavior change.
   - Re-run the navigation path as a module-change task when implementation starts.
   - Confirm whether live ownership sits in `scripts/*`, runtime adapters, shared utilities, or a different surface.
   - Confirm the exact tests and artifacts that currently define stock-pool behavior.
   - Do not treat the provisional file/module list in this plan as authoritative until that confirmation is complete.

2. Freeze the incumbent behavior and evidence surface.
   - Document the current training-pool, prediction-pool, and previous-day merge rules as the baseline contract.
   - Record the current monthly pool sizes, prediction-day candidate sizes, and final holding capacity assumptions.
   - Store the frozen baseline in a dedicated bilingual contract pair:
     - `docs/product-specs/a-share-universe-contract.md`
     - `docs/product-specs/a-share-universe-contract_zh.md`
   - Record current pain points:
     - randomness in prediction eligibility
     - mixed responsibility between universe selection and turnover control
     - train/predict distribution mismatch

3. Define the redesigned universe contract in the same bilingual artifact pair.
   - Specify the exact lag convention, liquidity metric, buffer semantics, and deterministic sampling rule.
   - Explicitly state which steps are allowed to use randomness and how reproducibility is guaranteed.
   - Keep incumbent and redesigned contracts side by side in the same named artifact pair until rollout is complete.
   - Keep the contract small enough to be testable from file outputs alone.

4. Refactor shared universe logic before changing behavior.
   - Centralize common filters and ranking helpers used by training and prediction pool generation.
   - Ensure the shared path supports:
     - lagged liquidity ranking
     - deterministic stratified sampling for training
     - deterministic entry/exit buffers for prediction
     - index and ST exclusion hooks

5. Implement the training-universe redesign.
   - Preserve monthly refresh cadence and anti-lookahead semantics.
   - Replace the incumbent grouping logic only after the new deterministic contract is available.
   - If downsampling remains necessary, make it reproducible by a fixed seed or date/symbol hash rule.

6. Implement the prediction-universe and holding-buffer redesign.
   - Remove random prediction-pool shrink logic.
   - Replace previous-day prediction merge behavior with an explicit hold-band rule.
   - Keep the final capacity cap explicit and testable.

7. Add focused verification.
   - Add unit tests for:
     - lag correctness
     - deterministic outputs on repeated runs
     - entry/exit buffer behavior
     - capacity cap behavior
     - no index leakage and no ST leakage
   - Add at least one targeted regression fixture comparing incumbent vs redesigned output surfaces on a fixed snapshot.

8. Run the minimum useful validation before rollout.
   - Compare incumbent vs redesigned policy on:
     - pool stability
     - prediction-day churn
     - holding turnover proxy
     - final name count
     - any ranking/backtest summary only after confirming the repo currently produces that evidence surface
   - Do not promote the redesigned policy if it improves determinism but materially harms implementability or downstream signal quality.

9. Update docs and trace artifacts, then move the plan when complete.
   - Update `README.md` and `docs/README_zh.md` if the operator-visible stock-pool rules change.
   - Keep this plan in `docs/exec-plans/active/` while the redesign is still in progress.
   - Move it to `docs/exec-plans/done/` only after implementation and validation are complete.

## Acceptance Criteria

- A documented and bilingual universe contract exists for training, prediction, and holding-buffer logic.
- Prediction-pool membership is deterministic for the same date snapshot and configuration.
- Training downsampling, if still required, is explicitly reproducible and separated from prediction logic.
- Turnover control is expressed through an explicit hold-band / capacity rule rather than an ad hoc previous-day prediction merge.
- Lagged liquidity ranking remains anti-lookahead and monthly-refresh compatible.
- Implementation ownership and test coverage are confirmed against live code before behavior changes are merged.
- Targeted tests cover lag behavior, determinism, exclusions, buffer logic, and capacity limits.
- Incumbent vs redesigned policy comparison is recorded before rollout.

## Rollback Notes

- Keep the incumbent monthly pool logic available until the redesigned contract passes focused validation.
- If the redesign increases instability, hidden lookahead risk, or downstream regressions, revert the behavior changes while keeping any reusable helper extraction and documentation that still reflect the incumbent path.
- If validation shows the thresholds are too aggressive for current hardware or market coverage, keep the three-layer contract and adjust only the numeric bands.
