# data_pipeline/

Market data acquisition, preprocessing, and database access.

## Files

| File              | Role                                                        |
|-------------------|-------------------------------------------------------------|
| `fetcher.py`      | `StockDataFetcher` — pulls OHLCV via akshare & baostock    |
| `preprocesser.py` | `Preprocesser` — generates TA features, cross-sectional z-score, clipping |
| `database.py`     | `DBClient` — HTTP client for the C++ data gateway           |

## Data Flow

```
fetcher  →  per-symbol CSV  →  (scripts/put_data)  →  DBClient  →  gateway
                                                          ↑
                                       scheduler/data_tasks calls DBClient for export
```

## Conventions

- All fetcher methods return `pd.DataFrame` or `None`.
- Column names are English-standardized via `rename_and_clean_df_columns`.
- Feature generation in `preprocesser.py` uses TA-Lib; each feature section starts with an English comment block explaining the math.

## See Also

- `scheduler/data_tasks.py` — orchestrates fetch / ingest / export
- `server/` — the C++ gateway that `DBClient` talks to
- `config/settings.py` — paths and connection params
