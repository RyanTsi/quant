"""Regression tests for runtime.adapters package lazy import behavior."""

from __future__ import annotations

import importlib
import sys
import unittest


def _purge_runtime_adapters_modules() -> None:
    for name in list(sys.modules):
        if name == "runtime.adapters" or name.startswith("runtime.adapters."):
            sys.modules.pop(name, None)


def _snapshot_runtime_adapters_modules() -> dict[str, object]:
    return {
        name: module
        for name, module in sys.modules.items()
        if name == "runtime.adapters" or name.startswith("runtime.adapters.")
    }


def _restore_runtime_adapters_modules(snapshot: dict[str, object]) -> None:
    _purge_runtime_adapters_modules()
    for name, module in snapshot.items():
        sys.modules[name] = module


class TestRuntimeAdaptersPackage(unittest.TestCase):
    def test_import_runtime_adapters_does_not_eagerly_import_modeling(self):
        snapshot = _snapshot_runtime_adapters_modules()
        try:
            _purge_runtime_adapters_modules()
            importlib.import_module("runtime.adapters")
            self.assertIn("runtime.adapters", sys.modules)
            self.assertNotIn("runtime.adapters.modeling", sys.modules)
        finally:
            _restore_runtime_adapters_modules(snapshot)

    def test_exporting_symbol_lookup_does_not_require_modeling_import(self):
        snapshot = _snapshot_runtime_adapters_modules()
        try:
            _purge_runtime_adapters_modules()
            adapters = importlib.import_module("runtime.adapters")
            normalized = adapters.normalize_gateway_rows([])
            self.assertIn("runtime.adapters.exporting", sys.modules)
            self.assertNotIn("runtime.adapters.modeling", sys.modules)
            self.assertEqual(
                list(normalized.columns),
                ["date", "open", "high", "low", "close", "volume", "amount", "turn", "isST", "factor"],
            )
        finally:
            _restore_runtime_adapters_modules(snapshot)


if __name__ == "__main__":
    unittest.main()
