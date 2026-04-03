# Exec Plan: Model Production Readiness Validation

## Goal

Establish a repeatable long-term validation program that can show whether the current Transformer alpha model, and future retrained models, are strong enough to promote from research to paper trading and then to production.

## Scope

- Evaluation protocol for `alpha_models/*`, `scripts/eval_test.py`, `scripts/view.py`, and related MLflow artifacts
- Research and validation artifacts under `docs/`, `.data/`, and experiment outputs
- Operational readiness checks for prediction, portfolio construction, and daily runtime behavior
- Promotion gates for current and future model candidates

## Assumptions

- The project remains focused on China A-share daily-frequency modeling and portfolio construction.
- Current signal usage stays ranking-first, not precise return regression.
- Promotion decisions should be driven by out-of-sample evidence, implementation realism, and operational stability together.
- New model candidates may change architecture or label design, but must pass the same evaluation gates.

## Operating Cadence

- Daily:
  - verify prediction delivery, artifact completeness, and portfolio generation success
- Monthly:
  - retrain the challenger model and compare it with the incumbent on the frozen evaluation contract
- Quarterly:
  - rerun the full walk-forward panel, robustness suite, and baseline comparison pack
- Before any production promotion:
  - complete an uninterrupted 8 to 12 week paper-trading / shadow-production run

## Steps

1. Freeze the evaluation contract.
   - Store the contract in a dedicated bilingual document pair under `docs/` and treat it as the only valid comparison standard.
   - Version the evaluation window definitions, benchmark, cost assumptions, universe rules, and portfolio policy.
   - Require any future contract change to go through a new dated exec plan and include side-by-side backfill results on the latest 3 completed walk-forward windows.
   - Require every training run to save the exact config, signal metrics, portfolio metrics, turnover, and artifact links.
   - Freeze the exact baseline suite that every candidate must beat:
     - incumbent Transformer using the frozen reference config
     - linear Alpha158 baseline on the same label, windows, and portfolio policy
     - LightGBM Alpha158 baseline on the same label, windows, and portfolio policy

2. Fix data and simulator trust issues before trusting backtests.
   - Remove or explain recurring warnings around `$close` `nan`, future calendar fallback, and empty-slice statistics.
   - Add a data-quality report for coverage, missing labels, suspended stocks, ST exclusions, and pool membership drift.
   - Add a backtest sanity checklist:
     - no lookahead leakage
     - correct trade calendar handling
     - realistic fees, slippage proxy, and limit-up/limit-down handling
   - Do not advance to the full walk-forward matrix until all of the following are true:
     - 3 consecutive evaluation runs finish without unexplained data or calendar warnings
     - the leakage checklist is signed off
     - the data-quality report is complete for the target evaluation windows

3. Build the core out-of-sample experiment matrix.
   - Run walk-forward training on multiple non-overlapping test windows, not just one recent year.
   - Minimum recommended matrix:
     - rolling 12M train / 3M valid / 3M test
     - rolling 24M train / 6M valid / 6M test
     - at least 6 to 10 consecutive test windows
   - For each window, record:
     - IC, ICIR, Rank IC, Rank ICIR
     - excess return with and without cost
     - information ratio, max drawdown, turnover, win-rate months
     - number of tradable names and average holding concentration

4. Add robustness and ablation experiments.
   - Compare against baselines on the same windows and same portfolio policy.
   - Run feature and label ablations:
     - Alpha158 only vs reduced feature subsets
     - current weighted multi-day label vs simpler next-day / 5-day labels
     - Transformer vs simpler model families
   - Run hyperparameter sensitivity tests:
     - seed stability
     - step length
     - model width/depth
     - dropout and batch size
   - A model should not be considered production-ready if performance only exists in one narrow parameter corner.

5. Prove portfolio implementability.
   - Measure turnover, capacity proxy, concentration, style exposure, and industry exposure.
   - Stress the strategy under stricter assumptions:
     - higher fees
     - delayed execution
     - smaller tradable universe
     - topk and dropout variants
   - Add pass/fail thresholds for implementability:
     - turnover within an operationally acceptable band
     - no single name or industry dominates portfolio risk
     - cost-adjusted alpha remains positive under mild stress

6. Introduce stability monitoring across retrains.
   - Retrain on a fixed cadence, such as monthly, and compare each new candidate to the incumbent.
   - Maintain a model registry table with:
     - train/valid/test periods
     - config hash
     - signal metrics
     - cost-adjusted portfolio metrics
     - turnover and drawdown metrics
     - pass/fail by gate
   - Reject a new model if it only wins on one headline metric while degrading stability or operational risk.

7. Run paper-trading and shadow-production validation.
   - Before live promotion, shadow the full daily pipeline for at least 8 to 12 weeks.
   - Record:
     - prediction delivery timeliness
     - missing artifact rate
     - portfolio generation success rate
     - drift between expected and realized tradable universe
     - realized slippage proxy and turnover versus backtest assumptions
   - Investigate all days with abnormal deviations, failed runs, or unexplained portfolio jumps.

8. Define promotion gates and decision rules.
   - Research gate:
     - positive mean Rank IC across most windows
     - positive cost-adjusted excess return across the full walk-forward panel
     - beats baseline median performance, not only best-case performance
   - Robustness gate:
     - acceptable degradation under fee/execution stress
     - no single-period dependence for the full thesis
     - stable results across seeds and small hyperparameter changes
   - Paper-trading gate:
     - stable daily operations for at least 8 weeks
     - no major data-quality or scheduling incidents
     - realized operational metrics remain close to backtest assumptions
   - Production gate:
     - incumbent replacement only if the challenger improves a weighted score of alpha, stability, and implementability
     - human review required for architecture changes, large universe changes, or risk-profile changes

## Phased Delivery

- Phase 1: Evaluation contract and trust blockers
  - target horizon: 1 to 2 weeks
  - output: frozen contract, frozen baseline suite, data/backtest trust exit criteria
- Phase 2: Walk-forward and registry foundation
  - target horizon: 2 to 4 weeks
  - output: multi-window evaluation runner, model registry table, baseline comparison pack
- Phase 3: Robustness and implementability
  - target horizon: 2 to 4 weeks
  - output: stress-test pack, ablation results, exposure and turnover review
- Phase 4: Paper trading
  - target horizon: 8 to 12 weeks
  - output: shadow-run incident log, operational scorecard, promotion recommendation
- Each phase must name one accountable owner before work starts, even if execution is shared.

## Starting Gate Thresholds

- These are initial operating thresholds for the first validation cycle and should be tuned only after enough history is collected.
- Research gate:
  - walk-forward mean Rank IC > 0
  - walk-forward median Rank IC >= 0.05
  - positive cost-adjusted excess return in at least 70% of test windows
  - full-panel cost-adjusted annualized excess return >= 8%
  - full-panel cost-adjusted information ratio >= 0.8
- Robustness gate:
  - under +50% fee stress, full-panel cost-adjusted excess return remains positive
  - under 1-day delayed execution proxy, full-panel alpha degradation is less than 35%
  - across at least 3 random seeds, Rank IC standard deviation stays within 25% of the mean absolute Rank IC
- Implementability gate:
  - annualized turnover does not exceed 1.5x the incumbent unless alpha improvement is materially higher
  - no single stock exceeds 10% target weight on rebalance
  - no single industry exceeds 30% of target portfolio weight without explicit approval
- Paper-trading gate:
  - prediction and portfolio jobs succeed on at least 95% of trading days
  - missing artifact rate stays below 1%
  - median realized tradable-universe drift stays within 10% of the expected shadow universe
  - median realized turnover proxy stays within 20% of the shadow backtest expectation
  - no unresolved abnormal jump days remain in the review log
- Promotion gate:
  - challenger beats the incumbent on a weighted scorecard across the latest 3 monthly retrain cycles:
    - 40% alpha quality
    - 25% drawdown and return stability
    - 20% turnover and implementability
    - 15% operational reliability
  - no blocker remains in data quality, backtest trust, or paper-trading operations

## Acceptance Criteria

- A documented validation framework exists for current and future models.
- Every candidate model is evaluated on the same walk-forward, baseline, robustness, and paper-trading gates.
- Promotion decisions can be explained with stored artifacts instead of one-off backtest screenshots.
- The team can clearly answer:
  - Does the model generalize across time?
  - Does it survive trading frictions?
  - Is it operationally stable enough to run every day?
  - Is it meaningfully better than the incumbent and simple baselines?

## Rollback Notes

- This plan is documentation-only. If the staged validation program proves too heavy, reduce cadence or matrix size, but keep the same gate structure and artifact discipline.
