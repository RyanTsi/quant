"""Integration-style tests for runtime registry and CLI dispatch."""

import unittest
from unittest.mock import MagicMock, patch

import main
from quantcore.registry import RuntimeRegistry


class TestRuntimeRegistry(unittest.TestCase):
    def test_registry_run_single_task(self):
        seen = []

        def fake_task():
            seen.append("task")

        reg = RuntimeRegistry(task_map={"fetch": fake_task}, pipeline_map={})
        reg.run("fetch")
        self.assertEqual(seen, ["task"])

    @patch("quantcore.registry.run_pipeline")
    def test_registry_run_pipeline(self, mock_run_pipeline):
        pipeline = [lambda: None]
        reg = RuntimeRegistry(task_map={}, pipeline_map={"full": pipeline})
        reg.run("full")
        mock_run_pipeline.assert_called_once_with(pipeline)

    def test_registry_run_unknown_raises(self):
        reg = RuntimeRegistry(task_map={}, pipeline_map={})
        with self.assertRaises(KeyError):
            reg.run("unknown")


class TestMainDispatch(unittest.TestCase):
    @patch("main.RuntimeRegistry.build_default")
    def test_run_once_delegates_to_registry(self, mock_build):
        mock_registry = MagicMock()
        mock_build.return_value = mock_registry
        main.run_once("fetch")
        mock_registry.run.assert_called_once_with("fetch")

    @patch("main.RuntimeRegistry.build_default")
    def test_run_once_unknown_exits(self, mock_build):
        mock_registry = MagicMock()
        mock_registry.run.side_effect = KeyError("bad")
        mock_build.return_value = mock_registry
        with self.assertRaises(SystemExit):
            main.run_once("bad")


if __name__ == "__main__":
    unittest.main()
