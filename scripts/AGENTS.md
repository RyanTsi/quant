# scripts/

Standalone CLI tools, each runnable as `python scripts/<name>.py`.

## Files

| File               | Purpose                                           |
|--------------------|---------------------------------------------------|
| `put_data.py`      | Bulk ingest per-symbol CSVs into the C++ gateway   |
| `dump_bin.py`      | Convert CSV to Qlib binary format                  |
| `export_today.py`  | Export today's data snapshot                        |
| `predict.py`       | Run model prediction                               |
| `update_data.py`   | Incremental data update                            |
| `filter.py`        | Filter stock universe                              |
| `filter_vai_csv.py`| Filter via CSV criteria                            |
| `view.py`          | Quick data viewer for debugging                    |

## Conventions

- Each script is self-contained with `if __name__ == "__main__"` entry.
- Scripts import from `config`, `data_pipeline`, `utils` — never from `scheduler`.

## See Also

- `scheduler/` — automated versions of these tasks
- `data_pipeline/` — underlying data operations
