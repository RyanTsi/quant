# data_pipeline/

Market data acquisition, ingestion, and database access.

## Files

| File              | Role                                                        |
|-------------------|-------------------------------------------------------------|
| `fetcher.py`      | `StockDataFetcher` — pulls OHLCV via baostock               |
| `ingest.py`       | `ingest_directory` — batch-POST CSV data to the C++ gateway  |
| `database.py`     | `DBClient` — HTTP client for the C++ data gateway            |
| `preprocesser.py` | **(Deprecated)** TA-Lib feature generator; superseded by Qlib Alpha158 |

## Data Flow

```
fetcher  →  per-symbol CSV  →  ingest  →  gateway  →  PostgreSQL
                                              ↑
                               scheduler/data_tasks orchestrates all three
```

## Conventions

- All fetcher methods return `pd.DataFrame` or `None`.
- Column names are English-standardized via `rename_and_clean_df_columns`.
- Uses `logging` module — no raw `print` calls.

## See Also

- `scheduler/data_tasks.py` — orchestrates fetch / ingest / export
- `server/` — the C++ gateway that `DBClient` talks to
- `config/settings.py` — paths and connection params
