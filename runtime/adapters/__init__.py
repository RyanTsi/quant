"""Runtime adapter entry points with lazy imports.

This keeps package import lightweight so data-side imports do not eagerly
pull modeling-side dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "build_portfolio_outputs",
    "dump_to_qlib_data",
    "export_from_gateway",
    "generate_predictions",
    "get_predict_conf",
    "normalize_gateway_rows",
]

_ATTR_MODULE_MAP = {
    "build_portfolio_outputs": ("runtime.adapters.modeling", "build_portfolio_outputs"),
    "dump_to_qlib_data": ("runtime.adapters.modeling", "dump_to_qlib_data"),
    "export_from_gateway": ("runtime.adapters.exporting", "export_from_gateway"),
    "generate_predictions": ("runtime.adapters.modeling", "generate_predictions"),
    "get_predict_conf": ("runtime.adapters.modeling", "get_predict_conf"),
    "normalize_gateway_rows": ("runtime.adapters.exporting", "normalize_gateway_rows"),
}


def __getattr__(name: str) -> Any:
    target = _ATTR_MODULE_MAP.get(name)
    if target is None:
        raise AttributeError(f"module 'runtime.adapters' has no attribute '{name}'")

    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
