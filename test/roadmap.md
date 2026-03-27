# Roadmap: test/

## 1. Overview

Unit tests (pytest / unittest). Prefer mocks for HTTP and filesystem under `tempfile`.

## 3. File-Role Mapping

| File / Subdirectory | Role / Description |
| :--- | :--- |
| `test_run_tracker.py` | Covers `utils/run_tracker` — `init`, `record_run`, `get_last_run`, `today` |
| `test_fetch_data_from_db.py` | Covers `data_pipeline/database.DBClient` |
| `test_ingest.py` | Covers `data_pipeline/ingest` — `_safe_float` / `_safe_int`, `ingest_directory` |
| `test_export_today.py` | Covers `scripts/export_today` — `fetch_all_by_date`, `export_date_to_csv` |
| `test_filter_stocks.py` | Covers `scripts/filter.filter_top_liquidity` |
| `test_filter_vai_csv.py` | Covers `scripts/filter_vai_csv.load_stock` |
| `test_config_settings.py` | Covers `config/settings` — types and path invariants |
| `test_qlib_workflow_refactor.py` | Covers modular workflow builders/registry/runtime guards (no live qlib run) |
| `test_scheduler_pipeline.py` | Covers scheduler pipeline cooldown + stop-on-error behavior |

## 5. Navigation

| If you want to... | Go to... |
| :--- | :--- |
| Add tests for a new module | Create `test_<area>.py` under this dir |
| Update DBClient tests | `test_fetch_data_from_db.py` |
| Update ingest parsing tests | `test_ingest.py` |
| Update export logic tests | `test_export_today.py` |
| Update settings invariants tests | `test_config_settings.py` |

## 6. Conventions

- Test files named `test_<area>.py`.
- Mock external HTTP (`requests`) and gateway; avoid live `baostock` / Qlib in CI unless marked optional.
