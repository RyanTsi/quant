"""Integration-style tests for runtime registry and CLI dispatch."""

import logging
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import main
from runtime.orchestrator import SequentialOrchestrator
from runtime.registry import RuntimeRegistry
from runtime.runlog import RunLogStore


class TestRuntimeRegistry(unittest.TestCase):
    def test_registry_run_single_task(self):
        seen = []

        def fake_task():
            seen.append("task")

        fake_task.task_name = "fetch_data"
        reg = RuntimeRegistry(task_map={"fetch": fake_task}, pipeline_map={})
        reg.run("fetch")
        self.assertEqual(seen, ["task"])

    def test_registry_run_pipeline(self):
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_pipeline.return_value = SimpleNamespace(success=True)

        pipeline = ["fetch"]
        fake_task = lambda: None
        fake_task.task_name = "fetch_data"
        reg = RuntimeRegistry(task_map={"fetch": fake_task}, pipeline_map={"full": pipeline}, orchestrator=mock_orchestrator)
        ok = reg.run("full")

        mock_orchestrator.run_pipeline.assert_called_once()
        args, _kwargs = mock_orchestrator.run_pipeline.call_args
        self.assertEqual(args[0], "full")
        self.assertEqual([t.name for t in args[1]], ["fetch_data"])
        self.assertTrue(ok)

    def test_registry_run_pipeline_failure_returns_false(self):
        mock_orchestrator = MagicMock()
        mock_orchestrator.run_pipeline.return_value = SimpleNamespace(success=False)

        fake_task = lambda: None
        reg = RuntimeRegistry(task_map={"fetch": fake_task}, pipeline_map={"full": ["fetch"]}, orchestrator=mock_orchestrator)

        ok = reg.run("full")
        self.assertFalse(ok)

    def test_registry_run_single_task_prefers_callable_task_name(self):
        mock_orchestrator = MagicMock()

        def fake_task():
            return None

        fake_task.task_name = "predict_task"
        reg = RuntimeRegistry(task_map={"predict": fake_task}, pipeline_map={}, orchestrator=mock_orchestrator)
        reg.run("predict")

        mock_orchestrator.run_task.assert_called_once()
        task_spec = mock_orchestrator.run_task.call_args.args[0]
        self.assertEqual(task_spec.name, "predict_task")

    def test_registry_run_unknown_raises(self):
        reg = RuntimeRegistry(task_map={}, pipeline_map={})
        with self.assertRaises(KeyError):
            reg.run("unknown")

    def test_pipeline_failure_keeps_prior_runlog_entry_readable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runlog_path = os.path.join(tmpdir, "run_history.json")
            store = RunLogStore(runlog_path)

            def persist_task():
                store.record("persist_task", status="ok")

            def fail_task():
                raise RuntimeError("forced failure")

            reg = RuntimeRegistry(
                task_map={"persist": persist_task, "boom": fail_task},
                pipeline_map={"smoke": ["persist", "boom"]},
                orchestrator=SequentialOrchestrator(
                    logger=logging.getLogger("test-runtime-smoke"),
                    cooldown_provider=lambda: 0.0,
                ),
            )

            ok = reg.run("smoke")

            self.assertFalse(ok)
            persisted = store.get("persist_task")
            self.assertIsNotNone(persisted)
            self.assertEqual(persisted["status"], "ok")
            self.assertIsNone(store.get("boom"))


class TestMainDispatch(unittest.TestCase):
    @patch("main.build_default_registry")
    def test_run_once_delegates_to_registry(self, mock_build):
        mock_registry = MagicMock()
        mock_build.return_value = mock_registry
        main.run_once("fetch")
        mock_registry.run.assert_called_once_with("fetch")

    @patch("main.build_default_registry")
    def test_run_once_unknown_exits(self, mock_build):
        mock_registry = MagicMock()
        mock_registry.run.side_effect = KeyError("bad")
        mock_build.return_value = mock_registry
        with self.assertRaises(SystemExit):
            main.run_once("bad")

    @patch("main.build_default_registry")
    def test_run_once_pipeline_failure_exits(self, mock_build):
        mock_registry = MagicMock()
        mock_registry.run.return_value = False
        mock_build.return_value = mock_registry

        with self.assertRaises(SystemExit):
            main.run_once("full")


if __name__ == "__main__":
    unittest.main()
