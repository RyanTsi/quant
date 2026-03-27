# Roadmap: server/

## 1. Overview

C++ HTTP gateway (Drogon framework) that sits in front of PostgreSQL.

## 2. Architecture

- `DataBuffer<T>` — thread-safe buffer for incoming data
- `MarketDataManager` — handles ingest HTTP, buffers rows
- `MarketDataStorage` — periodic flush to PostgreSQL via async SQL
- Flush interval: 2 seconds

## 3. File-Role Mapping


| File / Subdirectory         | Role / Description                            |
| --------------------------- | --------------------------------------------- |
| `main.cc`                   | All handlers: ingest, query, stats, health    |
| `config.json`               | Drogon config (port, DB connection, threads)  |
| `CMakeLists.txt`            | Build configuration                           |
| `sql/market_data_daily.sql` | DDL for the `market_data_daily` table         |
| `docker/`                   | Docker Compose setup for PostgreSQL + gateway |


## 5. Navigation


| If you want to...                       | Go to...                       |
| --------------------------------------- | ------------------------------ |
| Change an API endpoint behavior         | `main.cc`                      |
| Update DB schema / table definition     | `sql/market_data_daily.sql`    |
| Change gateway port / DB connection     | `config.json`                  |
| Build or adjust build flags             | `CMakeLists.txt`               |
| Run via Docker Compose                  | `docker/`                      |
| Update Python client usage of endpoints | `../data_pipeline/database.py` |


## 6. Conventions

- API contracts are consumed by `data_pipeline/database.py` and `scripts/put_data.py`; keep paths stable or update Python clients together.

## API Endpoints


| Method | Path                          | Purpose                |
| ------ | ----------------------------- | ---------------------- |
| POST   | `/api/v1/ingest/daily`        | Batch ingest           |
| GET    | `/api/v1/query/daily/symbol`  | Query by symbol + date |
| POST   | `/api/v1/query/daily/symbols` | Multi-symbol query     |
| GET    | `/api/v1/query/daily/latest`  | Latest N bars          |
| GET    | `/api/v1/symbols`             | List all symbols       |
| GET    | `/api/v1/health`              | Health check           |
| DELETE | `/api/v1/data/daily`          | Delete by symbol       |


