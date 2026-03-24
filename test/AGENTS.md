# test/

Unit and integration tests.

## Files

| File                       | Tests                         |
|----------------------------|-------------------------------|
| `test_run_tracker.py`      | `utils/run_tracker.py`        |
| `test_fetch_data_from_db.py` | `data_pipeline/database.py` |

## Conventions

- Test files named `test_<module>.py`.
- Use pytest.
- Tests should be runnable offline where possible; mock external services.

## See Also

- The module each test covers (see table above).
