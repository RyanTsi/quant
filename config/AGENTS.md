# config/

Global configuration. Single source of truth for paths, API tokens, connection settings, and Qlib parameters.

## Files

| File          | Role                                              |
|---------------|---------------------------------------------------|
| `settings.py` | `Settings` class — loads `.env`, exposes all config as attributes |

## Usage

```python
from config.settings import settings
settings.data_path            # .data/
settings.db_host              # from .env or default 127.0.0.1
settings.db_port              # from .env or default 8080
settings.qlib_provider_uri    # Qlib binary data path
settings.qlib_recorder_id     # MLflow recorder for model loading
settings.qlib_experiment_id   # MLflow experiment ID
settings.qlib_mlruns_uri      # MLflow tracking URI
```

## Conventions

- All secrets come from `.env` (never hardcoded).
- Paths are relative to project root via `BASE_DIR`.
- Add new config as attributes on `Settings`, not as module-level globals.

## See Also

- `.env.template` — environment variable reference
- Root `AGENTS.md` — system overview
