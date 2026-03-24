# Quant System — Root

A-share quantitative trading system: data acquisition → storage → feature engineering → model training → prediction, with scheduled automation.

## Architecture Overview

```
main.py              ← Entry point: scheduler + CLI runner
config/              ← Global settings, env loading
data_pipeline/       ← Fetch market data, preprocess features, DB client
alpha_models/        ← ML models (LSTM, Transformer) and Qlib workflows
scheduler/           ← Task definitions, pipelines, cron scheduling
scripts/             ← Standalone CLI tools (ingest, export, dump, predict)
server/              ← C++ HTTP gateway (Drogon) for PostgreSQL
news_module/         ← Financial news scraping service (WIP)
utils/               ← I/O helpers, run tracker, formatting
test/                ← Unit tests
harness/             ← Logging and dev tooling
backtesting/         ← (Placeholder) backtesting framework
rl_portfolio/        ← (Placeholder) RL-based portfolio optimization
docs/                ← Documentation and tutorials
```

## Data Flow

```
akshare / baostock  →  CSV files  →  C++ gateway  →  PostgreSQL
                                          ↓
                                     DB export CSV  →  Qlib binary  →  Model train / predict
```

## Navigation

| Want to...                     | Go to                  |
|--------------------------------|------------------------|
| Understand data fetching       | `data_pipeline/`       |
| See how tasks are scheduled    | `scheduler/`           |
| Read model architecture        | `alpha_models/`        |
| Understand the HTTP gateway    | `server/`              |
| Run a one-off script           | `scripts/`             |
| Check news scraping            | `news_module/`         |
| Look at utility functions      | `utils/`               |
| Change global config           | `config/`              |

## Conventions

- Python 3.12+. Dependencies in `requirements.txt`.
- Config via `.env` (see `.env.template`).
- All runtime data under `.data/` (gitignored).
