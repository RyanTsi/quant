# Roadmap: Quant System — Root

## 1. Overview

A-share quantitative trading system: data acquisition → storage → feature engineering → model training → prediction, with scheduled automation.

## 2. Architecture

### 2.1. Module map

```
main.py              ← Entry point: scheduler + CLI runner
config/              ← Global settings, env loading
data_pipeline/       ← Fetch market data, ingest/export, DB HTTP client
alpha_models/        ← ML models (LSTM, Transformer) and Qlib workflows
scheduler/           ← Task definitions, pipelines, cron scheduling
scripts/             ← Standalone CLI tools (ingest/export/filter/dump/predict)
server/              ← C++ HTTP gateway (Drogon) for PostgreSQL
news_module/         ← Financial news scraping service (WIP, isolated)
utils/               ← Leaf utilities (I/O, formatting, run tracker)
test/                ← Unit tests
.harness/            ← Change logs and operational notes (see `.harness/log/`)
backtesting/         ← Backtesting package placeholder (currently minimal)
rl_portfolio/        ← RL portfolio package placeholder (currently minimal)
docs/                ← Documentation and tutorials
```

### 2.2. Data flow

```
akshare / baostock  →  per-symbol CSV  →  gateway  →  PostgreSQL
                               ↓
                        export CSV  →  Qlib binary  →  train / predict
```

## 3. File-Role Mapping

| File / Subdirectory | Role / Description |
| :--- | :--- |
| `main.py` | System entrypoint |
| `config/` | Global settings (`.env` loading) |
| `data_pipeline/` | Fetch/ingest/export + gateway client |
| `scheduler/` | Scheduled tasks and pipelines |
| `alpha_models/` | Models + Qlib workflow configs |
| `scripts/` | One-off tools |
| `server/` | C++ gateway in front of PostgreSQL |
| `utils/` | Low-level helpers |
| `news_module/` | News scraping WIP |
| `test/` | Unit tests |
| `.harness/log/changelog.md` | Change log (newest first) |
| `backtesting/` | Placeholder package |
| `rl_portfolio/` | Placeholder package |
| `docs/` | Docs and tutorials |

## 5. Navigation

| If you want to... | Go to... |
| :--- | :--- |
| Run scheduled pipelines | `main.py` / `scheduler/` |
| Fetch & ingest market data | `data_pipeline/` |
| Change cron/pipeline ordering | `scheduler/pipelines.py` |
| Train/predict models | `alpha_models/` / `scheduler/model_tasks.py` |
| Understand / modify gateway APIs | `server/` |
| Run one-off export/filter scripts | `scripts/` |
| Update configuration / env vars | `config/settings.py` / `.env.template` |
| See recent changes | `.harness/log/changelog.md` |

## 6. Conventions

- Python 3.12+. Dependencies in `requirements.txt`.
- Config via `.env` (see `.env.template`).
- Runtime data under `.data/` (gitignored).
