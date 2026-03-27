# Roadmap: data_pipeline/

## 1. Overview

Market data acquisition, ingestion, and database access (HTTP client to the C++ gateway).

## 2. Architecture

```
fetcher  →  per-symbol CSV  →  ingest  →  gateway  →  PostgreSQL
                                              ↑
                               scheduler/data_tasks orchestrates all three
```

## 3. File-Role Mapping

| File / Subdirectory | Role / Description |
| :--- | :--- |
| `fetcher.py` | `StockDataFetcher` — pulls OHLCV via baostock |
| `ingest.py` | `ingest_directory` — batch-POST CSV data to the C++ gateway |
| `database.py` | `DBClient` — HTTP client for the C++ data gateway |
| `preprocesser.py` | Deprecated TA-Lib feature generator (superseded by Qlib Alpha158/Alpha360) |

## 5. Navigation

| If you want to... | Go to... |
| :--- | :--- |
| Fetch daily bars from data source | `fetcher.py` |
| Ingest a directory of CSVs into DB (via gateway) | `ingest.py` |
| Query / export data from the gateway | `database.py` |
| Wire end-to-end fetch→ingest→export in schedules | `../scheduler/data_tasks.py` |
| Understand gateway API contracts | `../server/roadmap.md` / `../server/main.cc` |

## 6. Conventions

- All fetcher methods return `pd.DataFrame` or `None`.
- Column names are English-standardized via `rename_and_clean_df_columns`.
- Uses `logging` module — no raw `print` calls.
