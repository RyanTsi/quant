# Product Spec: A-Share Universe Contract

- Version: v1
- Date: 2026-04-03
- Applies to: `model_function/universe.py`, `scripts/filter.py`, `runtime/adapters/modeling.py`, `scripts/build_portfolio.py`

## 1. Purpose

This document freezes the incumbent universe behavior and the redesigned contract for the low-compute A-share model pipeline so train, predict, and portfolio stages can be compared against one shared policy surface.

## 2. Shared Terms

- Turnover proxy: `amount` when available, otherwise `close * volume`.
- Lag rule: any month or day membership decision must use only data visible at or before that source snapshot.
- Index exclusion: symbols from `index_code_list` are never eligible.
- ST exclusion: symbols with active ST status at the relevant source snapshot are never eligible.

## 3. Incumbent Contract

### Training Universe

- Monthly refresh with one-month lag.
- Ranking used quarter-lookback liquidity stability plus a volatility extreme filter.
- Final output used weighted group selection to write `my_800_stocks.txt`.
- There was no explicit monthly entry/exit buffer for prior training members.

### Prediction Universe

- Rank symbols from a recent liquidity window.
- Randomly sample the top-liquidity names into a smaller pool.
- If the sampled pool was smaller than the target size, expand it with previous-day prediction output until 500 names.
- This made prediction eligibility partly random and mixed turnover control into pool construction.

### Holding / Turnover Control

- No explicit model-rank hold band existed.
- Turnover smoothing happened indirectly through previous-day prediction merge behavior and the downstream `top_k` cap.

## 4. Redesigned Contract

### Training Universe

- Source period: previous month.
- Liquidity score:
  - `lagged_liquidity(symbol, month_t) = median(turnover)` over the trailing 60 trading observations ending on the last available trading day in `month_t`.
  - Minimum observations: 20.
- Membership bands:
  - entry band: top 1800 by `lagged_liquidity`
  - exit band: top 2200 by `lagged_liquidity`
- Retention rule:
  - new names may enter only from the entry band
  - prior training members may remain only while they stay inside the exit band
- Compute cap:
  - final monthly output is deterministically downsampled to `TrainingUniverseConfig.sample_size` names
  - the current live default in code is `1200`; the artifact name remains `my_800_stocks.txt` for backward compatibility
  - downsampling is segment-based, reproducible, and keyed by `source_month` plus the configured seed

### Prediction Universe

- Source snapshot: prediction day using the trailing 60 trading observations ending on that day.
- Liquidity score:
  - per-symbol median turnover over that 60-day window
- Membership bands:
  - entry band: top 1000 by lagged liquidity
  - exit band: top 1200 by lagged liquidity
- Retention rule:
  - all entry-band names are scored
  - previous holdings from the previous trading day's `target_weights_<prev_date>.csv` are also scored if they remain inside the exit band
- Determinism rule:
  - no random sampling is allowed in prediction-pool construction

### Holding Buffer / Portfolio Selection

- Buy band: top 300 by model score.
- Hold band: top 500 by model score for symbols already present in the previous target portfolio.
- Continuity source:
  - the previous target portfolio is loaded from the previous trading day's `target_weights_<prev_date>.csv`
- Final capacity:
  - after applying buy/hold bands, the existing `top_k` portfolio cap is applied
  - default `top_k` remains 80 unless overridden by CLI/runtime arguments

## 5. Output Surfaces

- Training universe artifact: `.data/qlib_data/instruments/my_800_stocks.txt`
- Prediction artifact: `output/top_picks_<date>.csv`
- Portfolio artifacts:
  - `output/target_weights_<date>.csv`
  - `output/orders_<date>.csv`

## 6. Acceptance Surface

The contract is accepted when:

1. Training universe membership is month-lagged, deterministic, index/ST-safe, and bounded by the configured compute cap (`TrainingUniverseConfig.sample_size`; current default `1200`).
2. Prediction universe membership is deterministic for the same snapshot and configuration.
3. Holding retention is expressed through explicit buy/hold score bands rather than previous-day prediction merge logic.
4. Train, predict, and portfolio stages all route through the shared helpers in `model_function/universe.py`.
