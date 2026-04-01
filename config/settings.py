"""Compatibility adapter for legacy imports.

Use `quantcore.settings` for the new runtime architecture.
"""

from quantcore.settings import AppSettings, BASE_DIR, get_settings, load_settings

# Keep historical singleton import path: `from config.settings import settings`.
settings: AppSettings = get_settings()

__all__ = ["AppSettings", "BASE_DIR", "get_settings", "load_settings", "settings"]
