# config/

Global configuration. Single source of truth for paths, API tokens, and connection settings.

## Files

| File          | Role                                              |
|---------------|---------------------------------------------------|
| `settings.py` | `Settings` class — loads `.env`, exposes all config as attributes |

## Usage

```python
from config.settings import settings
settings.data_path      # .data/
settings.db_host        # from .env or default 127.0.0.1
settings.db_port        # from .env or default 8080
```

## Conventions

- All secrets come from `.env` (never hardcoded).
- Paths are relative to project root via `BASE_DIR`.
- Add new config as attributes on `Settings`, not as module-level globals.

## See Also

- `.env.template` — environment variable reference
- Root `AGENTS.md` — system overview
