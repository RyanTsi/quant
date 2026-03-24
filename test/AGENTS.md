# test/

Unit tests (pytest / unittest). Prefer mocks for HTTP and filesystem under `tempfile`.

## Files

| File | Covers |
|------|--------|
| `test_run_tracker.py` | `utils/run_tracker` — `init`, `record_run`, `get_last_run`, `today` |
| `test_fetch_data_from_db.py` | `data_pipeline/database.DBClient` |
| `test_ingest.py` | `data_pipeline/ingest` — `_safe_float` / `_safe_int`, `ingest_directory` |
| `test_export_today.py` | `scripts/export_today` — `fetch_all_by_date`, `export_date_to_csv` |
| `test_filter_stocks.py` | `scripts/filter.filter_top_liquidity` |
| `test_filter_vai_csv.py` | `scripts/filter_vai_csv.load_stock` |
| `test_config_settings.py` | `config/settings` — types and path invariants |

## Conventions

- Test files named `test_<area>.py`.
- Mock external HTTP (`requests`) and gateway; avoid live `baostock` / Qlib in CI unless marked optional.

## See Also

- Modules under test (see table above).
