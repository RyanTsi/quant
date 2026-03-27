# Roadmap: utils/

## 1. Overview

Low-level utility functions (I/O, formatting, run tracking). Keep this module as a “leaf” to avoid upward dependency leakage.

## 3. File-Role Mapping

| File / Subdirectory | Role / Description |
| :--- | :--- |
| `io.py` | File I/O: `read_file_lines`, `get_df_from_csv`, `package_data` |
| `run_tracker.py` | JSON-based task run history (`record_run`, `get_last_run`, `today`) |
| `format.py` | Stock code formatting helpers |

## 5. Navigation

| If you want to... | Go to... |
| :--- | :--- |
| Read/write CSV helpers | `io.py` |
| Track scheduled task runs | `run_tracker.py` |
| Format stock codes | `format.py` |

## 6. Conventions

- Prefer pure helpers. Avoid importing business layers (`scheduler/`, `data_pipeline/`, etc.).
- `run_tracker.py` currently falls back to `config.settings` only when `init(data_path)` was not called; for strict leaf behavior, call `init(...)` at process startup (e.g. in `main.py`).
