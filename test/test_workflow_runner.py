"""Tests for alpha_models.workflow.runner."""

import sys
import types
import unittest
from unittest.mock import patch

from alpha_models.workflow.runner import (
    LoadedWorkflowConfig,
    QlibWorkflowRunner,
    TrainResult,
)


class TestWorkflowRunner(unittest.TestCase):
    def setUp(self):
        self.runner = QlibWorkflowRunner()

    def test_compose_config_from_dict(self):
        cfg = {
            "qlib_init": {"provider_uri": "~/.qlib_data"},
            "market": "my_market",
            "benchmark": "SH000001",
            "task": {"model": {"class": "M"}, "dataset": {"class": "D"}},
        }
        loaded = self.runner.compose_config(config=cfg, source_label="inline_cfg")

        self.assertEqual(loaded.config_source, "inline_cfg")
        self.assertEqual(loaded.market, "my_market")
        self.assertEqual(loaded.benchmark, "SH000001")
        self.assertEqual(loaded.qlib_init["provider_uri"], "~/.qlib_data")
        self.assertIn("model", loaded.task)
        self.assertIn("dataset", loaded.task)

    def test_compose_config_applies_overrides(self):
        cfg = {
            "task": {
                "model": {"class": "M", "kwargs": {"a": 1, "b": 2}},
                "dataset": {"class": "D"},
            }
        }
        overrides = {"task": {"model": {"kwargs": {"b": 99, "c": 3}}}}
        loaded = self.runner.compose_config(
            config=cfg,
            source_label="inline_cfg",
            config_overrides=overrides,
        )

        kwargs = loaded.task["model"]["kwargs"]
        self.assertEqual(kwargs["a"], 1)
        self.assertEqual(kwargs["b"], 99)
        self.assertEqual(kwargs["c"], 3)
        self.assertEqual(loaded.config_source, "inline_cfg+override")

    def test_compose_config_rejects_ambiguous_sources(self):
        with self.assertRaises(ValueError):
            self.runner.compose_config(config_source="a.yaml", config={})

    def test_compose_config_rejects_invalid_config_type(self):
        with self.assertRaises(TypeError):
            self.runner.compose_config(config=["not", "a", "dict"])  # type: ignore[arg-type]

    def test_run_from_config_delegates_to_run(self):
        cfg = {"task": {"model": {"class": "M"}, "dataset": {"class": "D"}}}
        expected = TrainResult(
            experiment_id="exp",
            recorder_id="rec",
            metrics=None,
            config_source="python_cfg",
        )
        with patch.object(self.runner, "run", return_value=expected) as run_mock:
            result = self.runner.run_from_config(config=cfg, source_label="python_cfg")

        self.assertIs(result, expected)
        passed_loaded = run_mock.call_args.kwargs["loaded_config"]
        self.assertEqual(passed_loaded.config_source, "python_cfg")
        self.assertIn("model", passed_loaded.task)
        self.assertIn("dataset", passed_loaded.task)

    def test_run_from_yaml_keeps_compatibility(self):
        loaded = LoadedWorkflowConfig(
            qlib_init={},
            market="all",
            benchmark="",
            task={"model": {"class": "M"}, "dataset": {"class": "D"}},
            config_source="alpha_models/workflow_config.yaml",
        )
        expected = TrainResult(
            experiment_id="exp",
            recorder_id="rec",
            metrics=None,
            config_source=loaded.config_source,
        )
        with patch.object(QlibWorkflowRunner, "load_yaml_config", return_value=loaded) as load_mock:
            with patch.object(self.runner, "run", return_value=expected) as run_mock:
                result = self.runner.run_from_yaml(config_source="alpha_models/workflow_config.yaml")

        self.assertIs(result, expected)
        load_mock.assert_called_once_with("alpha_models/workflow_config.yaml")
        passed_loaded = run_mock.call_args.kwargs["loaded_config"]
        self.assertEqual(passed_loaded.config_source, "alpha_models/workflow_config.yaml")

    def test_yaml_and_dict_paths_pass_equivalent_task_to_run(self):
        cfg = {"task": {"model": {"class": "M"}, "dataset": {"class": "D"}}}
        loaded_yaml = LoadedWorkflowConfig(
            qlib_init={},
            market="all",
            benchmark="",
            task={"model": {"class": "M"}, "dataset": {"class": "D"}},
            config_source="sample.yaml",
        )
        expected = TrainResult(experiment_id="e", recorder_id="r", metrics=None, config_source="x")

        with patch.object(self.runner, "run", return_value=expected) as run_mock:
            self.runner.run_from_config(config=cfg, source_label="dict_cfg")
            task_from_dict = run_mock.call_args.kwargs["loaded_config"].task

        with patch.object(QlibWorkflowRunner, "load_yaml_config", return_value=loaded_yaml):
            with patch.object(self.runner, "run", return_value=expected) as run_mock:
                self.runner.run_from_yaml(config_source="sample.yaml")
                task_from_yaml = run_mock.call_args.kwargs["loaded_config"].task

        self.assertEqual(task_from_dict, task_from_yaml)

    def test_run_rejects_missing_model_or_dataset(self):
        # lightweight qlib stubs to reach task validation branch without importing real qlib
        qlib_mod = types.ModuleType("qlib")
        qlib_mod.init = lambda *args, **kwargs: None
        qlib_utils = types.ModuleType("qlib.utils")
        qlib_utils.init_instance_by_config = lambda cfg: cfg
        qlib_workflow = types.ModuleType("qlib.workflow")
        qlib_workflow.R = types.SimpleNamespace(set_uri=lambda *_a, **_k: None)

        bad = LoadedWorkflowConfig(
            qlib_init={},
            market="all",
            benchmark="",
            task={"dataset": {"class": "D"}},
            config_source="inline",
        )
        with patch.dict(
            sys.modules,
            {"qlib": qlib_mod, "qlib.utils": qlib_utils, "qlib.workflow": qlib_workflow},
            clear=False,
        ):
            with self.assertRaises(ValueError):
                self.runner.run(loaded_config=bad)


if __name__ == "__main__":
    unittest.main()
