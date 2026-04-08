"""Direct tests for model_function.qlib helper boundaries."""

from __future__ import annotations

import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from alpha_models.workflow.runner import TrainMetrics, TrainResult
from model_function import qlib as qlib_helpers


def _install_qlib_stubs() -> dict[str, object]:
    """Install lightweight qlib stubs so helper tests stay unit-level."""

    qlib_mod = types.ModuleType("qlib")
    qlib_mod.init = MagicMock()

    qlib_config = types.ModuleType("qlib.config")
    qlib_config.REG_CN = "cn"

    recorder = MagicMock()
    workflow_r = types.SimpleNamespace(
        set_uri=MagicMock(),
        get_recorder=MagicMock(return_value=recorder),
    )
    qlib_workflow = types.ModuleType("qlib.workflow")
    qlib_workflow.R = workflow_r

    qlib_utils = types.ModuleType("qlib.utils")
    qlib_utils.init_instance_by_config = MagicMock(return_value="dataset_instance")

    analysis_model = types.SimpleNamespace(model_performance_graph=MagicMock(return_value=[]))
    analysis_position = types.SimpleNamespace(report_graph=MagicMock(return_value=[]))
    qlib_contrib_report = types.ModuleType("qlib.contrib.report")
    qlib_contrib_report.analysis_model = analysis_model
    qlib_contrib_report.analysis_position = analysis_position

    signal_record = types.SimpleNamespace(generate_label=MagicMock())
    qlib_record_temp = types.ModuleType("qlib.workflow.record_temp")
    qlib_record_temp.SignalRecord = signal_record

    qlib_contrib_eva_alpha = types.ModuleType("qlib.contrib.eva.alpha")
    qlib_contrib_eva_alpha.calc_ic = MagicMock(return_value=(np.array([1.0]), np.array([1.0])))

    qlib_contrib = types.ModuleType("qlib.contrib")
    qlib_contrib_model = types.ModuleType("qlib.contrib.model")
    qlib_contrib_model_ts = types.ModuleType("qlib.contrib.model.pytorch_transformer_ts")
    qlib_contrib_eva = types.ModuleType("qlib.contrib.eva")

    patcher = patch.dict(
        sys.modules,
        {
            "qlib": qlib_mod,
            "qlib.config": qlib_config,
            "qlib.workflow": qlib_workflow,
            "qlib.utils": qlib_utils,
            "qlib.contrib": qlib_contrib,
            "qlib.contrib.report": qlib_contrib_report,
            "qlib.contrib.model": qlib_contrib_model,
            "qlib.contrib.model.pytorch_transformer_ts": qlib_contrib_model_ts,
            "qlib.workflow.record_temp": qlib_record_temp,
            "qlib.contrib.eva": qlib_contrib_eva,
            "qlib.contrib.eva.alpha": qlib_contrib_eva_alpha,
        },
        clear=False,
    )
    patcher.start()
    return {
        "patcher": patcher,
        "qlib": qlib_mod,
        "workflow_r": workflow_r,
        "recorder": recorder,
        "utils": qlib_utils,
        "analysis_model": analysis_model,
        "analysis_position": analysis_position,
        "signal_record": signal_record,
        "calc_ic": qlib_contrib_eva_alpha.calc_ic,
    }


class TestResolveRecorderIdentity(unittest.TestCase):
    def test_explicit_ids_take_priority(self):
        identity = qlib_helpers.resolve_recorder_identity(
            experiment_id="exp_explicit",
            recorder_id="rec_explicit",
            env={"QLIB_EXPERIMENT_ID": "exp_env", "QLIB_RECORDER_ID": "rec_env"},
            runlog_entry={"experiment_id": "exp_hist", "recorder_id": "rec_hist"},
            fallback_experiment_id="exp_settings",
            fallback_recorder_id="rec_settings",
        )

        self.assertEqual(identity.experiment_id, "exp_explicit")
        self.assertEqual(identity.recorder_id, "rec_explicit")

    def test_env_ids_take_priority_after_explicit(self):
        identity = qlib_helpers.resolve_recorder_identity(
            env={"QLIB_EXPERIMENT_ID": "exp_env", "QLIB_RECORDER_ID": "rec_env"},
            runlog_entry={"experiment_id": "exp_hist", "recorder_id": "rec_hist"},
            fallback_experiment_id="exp_settings",
            fallback_recorder_id="rec_settings",
        )

        self.assertEqual(identity.experiment_id, "exp_env")
        self.assertEqual(identity.recorder_id, "rec_env")

    def test_runlog_used_after_env(self):
        identity = qlib_helpers.resolve_recorder_identity(
            env={"QLIB_EXPERIMENT_ID": "", "QLIB_RECORDER_ID": ""},
            runlog_entry={"experiment_id": "exp_hist", "recorder_id": "rec_hist"},
            fallback_experiment_id="exp_settings",
            fallback_recorder_id="rec_settings",
        )

        self.assertEqual(identity.experiment_id, "exp_hist")
        self.assertEqual(identity.recorder_id, "rec_hist")

    def test_fallback_ids_used_last(self):
        identity = qlib_helpers.resolve_recorder_identity(
            env={"QLIB_EXPERIMENT_ID": "", "QLIB_RECORDER_ID": ""},
            runlog_entry={},
            fallback_experiment_id="exp_settings",
            fallback_recorder_id="rec_settings",
        )

        self.assertEqual(identity.experiment_id, "exp_settings")
        self.assertEqual(identity.recorder_id, "rec_settings")

    def test_failure_when_no_resolution_source_exists(self):
        with self.assertRaises(RuntimeError):
            qlib_helpers.resolve_recorder_identity(
                env={"QLIB_EXPERIMENT_ID": "", "QLIB_RECORDER_ID": ""},
                runlog_entry={},
            )


class TestLoadTrainedModel(unittest.TestCase):
    def test_load_trained_model_prefers_primary_artifact(self):
        recorder = MagicMock()
        recorder.load_object.side_effect = lambda key: {
            "trained_model": "model_primary",
            "trained_model.pkl": "model_secondary",
            "params.pkl": "model_legacy",
        }[key]

        result = qlib_helpers.load_trained_model(recorder)

        self.assertEqual(result, "model_primary")
        self.assertEqual(recorder.load_object.call_args_list[0].args[0], "trained_model")

    def test_load_trained_model_falls_back_across_historical_keys(self):
        recorder = MagicMock()

        def _load_object(key: str):
            if key == "trained_model":
                raise FileNotFoundError("missing primary")
            if key == "trained_model.pkl":
                return "model_secondary"
            raise AssertionError(f"Unexpected key: {key}")

        recorder.load_object.side_effect = _load_object

        result = qlib_helpers.load_trained_model(recorder)

        self.assertEqual(result, "model_secondary")
        self.assertEqual(
            [call.args[0] for call in recorder.load_object.call_args_list],
            ["trained_model", "trained_model.pkl"],
        )

    def test_load_trained_model_uses_params_as_last_fallback(self):
        recorder = MagicMock()

        def _load_object(key: str):
            if key in {"trained_model", "trained_model.pkl"}:
                raise FileNotFoundError("missing")
            if key == "params.pkl":
                return "model_legacy"
            raise AssertionError(f"Unexpected key: {key}")

        recorder.load_object.side_effect = _load_object

        result = qlib_helpers.load_trained_model(recorder)

        self.assertEqual(result, "model_legacy")
        self.assertEqual(
            [call.args[0] for call in recorder.load_object.call_args_list],
            ["trained_model", "trained_model.pkl", "params.pkl"],
        )


class TestGenerateAnalysisView(unittest.TestCase):
    def setUp(self):
        self.stub_state = _install_qlib_stubs()

    def tearDown(self):
        self.stub_state["patcher"].stop()

    def test_generate_analysis_view_writes_model_and_portfolio_charts(self):
        model_fig = MagicMock()
        portfolio_fig = MagicMock()
        self.stub_state["analysis_model"].model_performance_graph.return_value = [model_fig]
        self.stub_state["analysis_position"].report_graph.return_value = [portfolio_fig]

        pred_df = pd.DataFrame({"score": [0.1]}, index=[0])
        label_df = pd.DataFrame({"label": [0.2]}, index=[0])
        self.stub_state["recorder"].load_object.side_effect = lambda key: {
            "pred.pkl": pred_df,
            "label.pkl": label_df,
            "portfolio_analysis/report_normal_1day.pkl": pd.DataFrame({"return": [0.3]}),
        }[key]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = qlib_helpers.generate_analysis_view(
                identity=qlib_helpers.RecorderIdentity(experiment_id="exp1", recorder_id="rec1"),
                provider_uri="provider://test",
                analysis_path=tmpdir,
                mlruns_uri="mlruns://test",
            )

        self.assertEqual(Path(output_dir).name, "rec1")
        self.assertEqual(Path(output_dir).parent, Path(tmpdir))
        self.stub_state["qlib"].init.assert_called_once_with(provider_uri="provider://test", region="cn")
        self.stub_state["workflow_r"].set_uri.assert_called_once_with("mlruns://test")
        self.stub_state["workflow_r"].get_recorder.assert_called_once_with(
            experiment_id="exp1",
            recorder_id="rec1",
        )
        self.assertTrue(model_fig.write_html.called)
        self.assertTrue(portfolio_fig.write_html.called)

    def test_generate_analysis_view_tolerates_missing_portfolio_artifact(self):
        model_fig = MagicMock()
        self.stub_state["analysis_model"].model_performance_graph.return_value = [model_fig]

        def _load_object(key: str):
            if key == "pred.pkl":
                return pd.DataFrame({"score": [0.1]}, index=[0])
            if key == "label.pkl":
                return pd.DataFrame({"label": [0.2]}, index=[0])
            raise FileNotFoundError("missing portfolio artifact")

        self.stub_state["recorder"].load_object.side_effect = _load_object

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = qlib_helpers.generate_analysis_view(
                identity=qlib_helpers.RecorderIdentity(experiment_id="exp2", recorder_id="rec2"),
                provider_uri="provider://test",
                analysis_path=tmpdir,
            )

        self.assertEqual(Path(output_dir).name, "rec2")
        self.assertTrue(model_fig.write_html.called)
        self.assertFalse(self.stub_state["analysis_position"].report_graph.called)


class TestEvaluateTestPredictions(unittest.TestCase):
    def setUp(self):
        self.stub_state = _install_qlib_stubs()

    def tearDown(self):
        self.stub_state["patcher"].stop()

    @patch("model_function.qlib.QlibWorkflowRunner.load_yaml_config")
    def test_evaluate_test_predictions_generates_metrics_from_normalized_predictions(self, mock_load_yaml):
        pred_series = pd.Series([0.2, 0.4], index=pd.Index([0, 1], name="row"))
        model = MagicMock()
        model.predict.return_value = pred_series

        self.stub_state["recorder"].load_object.side_effect = lambda key: {
            "trained_model": model,
        }[key]
        self.stub_state["signal_record"].generate_label.return_value = pd.DataFrame({"label": [0.1, 0.3]})
        self.stub_state["calc_ic"].return_value = (
            np.array([1.0, 3.0]),
            np.array([2.0, 4.0]),
        )
        mock_load_yaml.return_value = types.SimpleNamespace(task={"dataset": {"class": "Dataset"}})

        metrics = qlib_helpers.evaluate_test_predictions(
            config_source="cfg.yaml",
            identity=qlib_helpers.RecorderIdentity(experiment_id="exp_eval", recorder_id="rec_eval"),
            provider_uri="provider://eval",
            mlruns_uri="mlruns://eval",
        )

        self.stub_state["qlib"].init.assert_called_once_with(provider_uri="provider://eval", region="cn")
        self.stub_state["workflow_r"].set_uri.assert_called_once_with("mlruns://eval")
        self.stub_state["workflow_r"].get_recorder.assert_called_once_with(
            experiment_id="exp_eval",
            recorder_id="rec_eval",
        )
        self.stub_state["utils"].init_instance_by_config.assert_called_once_with({"class": "Dataset"})
        self.stub_state["signal_record"].generate_label.assert_called_once_with("dataset_instance")
        self.assertEqual(metrics["IC"], 2.0)
        self.assertEqual(metrics["Rank IC"], 3.0)
        self.assertTrue(math.isclose(metrics["ICIR"], 2.0, rel_tol=1e-9))
        self.assertTrue(math.isclose(metrics["Rank ICIR"], 3.0, rel_tol=1e-9))


class TestRunTrainingWorkflow(unittest.TestCase):
    def setUp(self):
        self.stub_state = _install_qlib_stubs()

    def tearDown(self):
        self.stub_state["patcher"].stop()

    def test_run_training_workflow_uses_default_config_and_runtime_defaults(self):
        runner = MagicMock()
        expected = TrainResult(
            experiment_id="exp_train",
            recorder_id="rec_train",
            metrics=None,
            config_source="cfg.yaml",
        )
        runner.run_from_yaml.return_value = expected

        result = qlib_helpers.run_training_workflow(
            default_provider_uri="provider://default",
            default_mlruns_uri="mlruns://default",
            default_experiment_name="exp_name",
            runner=runner,
        )

        self.assertIs(result, expected)
        runner.run_from_yaml.assert_called_once_with(
            config_source=qlib_helpers.default_training_config_path(),
            provider_uri_override="provider://default",
            mlruns_uri="mlruns://default",
            experiment_name="exp_name",
        )


if __name__ == "__main__":
    unittest.main()
