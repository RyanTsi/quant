# scripts/

Standalone CLI tools, each runnable as `python -m scripts.<name>`.

## Files

| File               | Purpose                                            |
|--------------------|----------------------------------------------------|
| `put_data.py`      | CLI wrapper for `data_pipeline.ingest`              |
| `dump_bin.py`      | Convert CSV to Qlib binary format                   |
| `export_today.py`  | Export a single day's market data from DB to CSV     |
| `predict.py`       | Run model prediction using trained Qlib model        |
| `update_data.py`   | Incremental data update via baostock                 |
| `filter.py`        | Filter top-500 liquidity stocks via DB queries       |
| `filter_vai_csv.py`| Filter top-500 liquidity stocks from local CSV files |
| `view.py`          | Visualize Qlib experiment results (IC/IR, returns)   |

## Conventions

- Each script is self-contained with `if __name__ == "__main__"` entry.
- Scripts import from `config`, `data_pipeline`, `utils` — never from `scheduler`.
- Hardcoded paths and experiment IDs have been moved to `config/settings.py`.

## See Also

- `scheduler/` — automated versions of these tasks
- `data_pipeline/` — underlying data operations
