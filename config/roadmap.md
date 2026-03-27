# Roadmap: config/

## 1. Overview

Global configuration. Single source of truth for paths, API tokens, connection settings, and Qlib parameters.

## 3. File-Role Mapping


| File / Subdirectory | Role / Description                                                |
| ------------------- | ----------------------------------------------------------------- |
| `settings.py`       | `Settings` class — loads `.env`, exposes all config as attributes |
| `.env.template`     | Environment variable reference (root file)                        |


## 5. Navigation


| If you want to...                      | Go to...        |
| -------------------------------------- | --------------- |
| Add / change an environment variable   | `.env.template` |
| Add a new setting (type-safe)          | `settings.py`   |
| Find DB gateway host/port and timeouts | `settings.py`   |
| Find Qlib / MLflow configuration       | `settings.py`   |


## 6. Conventions

- All secrets come from `.env` (never hardcoded).
- Paths are relative to project root via `BASE_DIR`.
- Add new config as attributes on `Settings`, not as module-level globals.

