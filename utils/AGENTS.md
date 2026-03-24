# utils/

Low-level utility functions. No business logic. No imports from higher-level modules.

## Files

| File              | Role                                              |
|-------------------|---------------------------------------------------|
| `io.py`           | File I/O: `read_file_lines`, `get_df_from_csv`, `package_data` |
| `run_tracker.py`  | JSON-based task run history (`record_run`, `get_last_run`, `today`) |
| `format.py`       | Stock code formatting helpers                      |

## Dependency Rule

This module is a leaf in the dependency graph. It may import only stdlib and third-party libraries. `run_tracker.py` uses lazy fallback to `config.settings` only if `init(data_path)` was not called explicitly.

## See Also

- Every other module depends on `utils/`.
