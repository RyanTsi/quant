# server/

C++ HTTP gateway (Drogon framework) that sits in front of PostgreSQL.

## Files

| File / Dir                | Role                                          |
|---------------------------|-----------------------------------------------|
| `main.cc`                 | All handlers: ingest, query, stats, health     |
| `config.json`             | Drogon config (port, DB connection, threads)   |
| `CMakeLists.txt`          | Build configuration                            |
| `sql/market_data_daily.sql` | DDL for the `market_data_daily` table        |
| `docker/`                 | Docker Compose setup for PostgreSQL + gateway  |

## API Endpoints

| Method | Path                           | Purpose                |
|--------|--------------------------------|------------------------|
| POST   | `/api/v1/ingest/daily`         | Batch ingest           |
| GET    | `/api/v1/query/daily/symbol`   | Query by symbol + date |
| POST   | `/api/v1/query/daily/symbols`  | Multi-symbol query     |
| GET    | `/api/v1/query/daily/latest`   | Latest N bars          |
| GET    | `/api/v1/symbols`              | List all symbols       |
| GET    | `/api/v1/health`               | Health check           |
| DELETE | `/api/v1/data/daily`           | Delete by symbol       |

## Architecture

- `DataBuffer<T>` — thread-safe buffer for incoming data
- `MarketDataManager` — handles ingest HTTP, buffers rows
- `MarketDataStorage` — periodic flush to PostgreSQL via async SQL
- Flush interval: 2 seconds

## See Also

- `data_pipeline/database.py` — Python HTTP client that talks to this gateway
- `scripts/put_data.py` — bulk CSV ingest script
