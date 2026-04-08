"""Tests for runtime.model_state."""

from __future__ import annotations

import os
import tempfile
import types
import unittest
from dataclasses import replace

from runtime.config import load_settings
from runtime.model_state import NO_TRAINED_MODEL_ERROR, ModelRuntimeState, build_model_runtime_state
from runtime.runlog import RunLogStore


class TestModelRuntimeState(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        base_settings = load_settings(env={})
        self.settings = replace(
            base_settings,
            data_path=self.tmp,
            analysis_path=os.path.join(self.tmp, "analysis"),
            qlib_data_path=os.path.join(self.tmp, "qlib_data"),
            qlib_workflow_config="settings.yaml",
            qlib_mlruns_uri="mlruns://default",
            qlib_provider_uri="provider://default",
            qlib_experiment_name="exp_name",
            qlib_experiment_id="exp_settings",
            qlib_recorder_id="rec_settings",
        )
        os.makedirs(self.settings.analysis_path, exist_ok=True)
        os.makedirs(self.settings.qlib_data_path, exist_ok=True)
        self.history = RunLogStore(os.path.join(self.tmp, "run_history.json"))
        self.runtime_state = ModelRuntimeState(settings=self.settings, history=self.history)

    def tearDown(self):
        for root, dirs, files in os.walk(self.tmp, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.tmp)

    def test_resolve_training_workflow_inputs_uses_runtime_settings_fallbacks(self):
        inputs = self.runtime_state.resolve_training_workflow_inputs()

        self.assertEqual(inputs.config_source, "settings.yaml")
        self.assertEqual(inputs.provider_uri, "provider://default")
        self.assertEqual(inputs.mlruns_uri, "mlruns://default")
        self.assertEqual(inputs.experiment_name, "exp_name")

    def test_resolve_recorder_identity_prefers_run_history_before_settings_opt_in(self):
        self.history.record("qlib_train", experiment_id="exp_hist", recorder_id="rec_hist")

        history_identity = self.runtime_state.resolve_recorder_identity(env={})
        self.assertEqual(history_identity.experiment_id, "exp_hist")
        self.assertEqual(history_identity.recorder_id, "rec_hist")

        env_identity = self.runtime_state.resolve_recorder_identity(
            env={
                "QLIB_EXPERIMENT_ID": "exp_env",
                "QLIB_RECORDER_ID": "rec_env",
            }
        )
        self.assertEqual(env_identity.experiment_id, "exp_env")
        self.assertEqual(env_identity.recorder_id, "rec_env")

        settings_identity = self.runtime_state.resolve_recorder_identity(
            env={},
            allow_settings_fallback=True,
        )
        self.assertEqual(settings_identity.experiment_id, "exp_hist")
        self.assertEqual(settings_identity.recorder_id, "rec_hist")

    def test_resolve_recorder_identity_raises_without_env_or_runlog_by_default(self):
        with self.assertRaisesRegex(RuntimeError, NO_TRAINED_MODEL_ERROR):
            self.runtime_state.resolve_recorder_identity(env={})

    def test_resolve_recorder_identity_allows_settings_fallback_when_opted_in(self):
        identity = self.runtime_state.resolve_recorder_identity(
            env={},
            allow_settings_fallback=True,
        )

        self.assertEqual(identity.experiment_id, "exp_settings")
        self.assertEqual(identity.recorder_id, "rec_settings")

    def test_record_training_result_persists_canonical_runlog_payload(self):
        result = types.SimpleNamespace(
            config_source="config.yaml",
            experiment_id="exp_train",
            recorder_id="rec_train",
            metrics=types.SimpleNamespace(
                ic=0.1,
                icir=0.2,
                rank_ic=0.3,
                rank_icir=0.4,
            ),
        )

        entry = self.runtime_state.record_training_result(result)

        self.assertEqual(entry["config_source"], "config.yaml")
        self.assertEqual(entry["experiment_id"], "exp_train")
        self.assertEqual(entry["recorder_id"], "rec_train")
        self.assertEqual(entry["IC"], 0.1)
        self.assertEqual(self.history.get("qlib_train")["Rank ICIR"], 0.4)

    def test_build_model_runtime_state_reuses_supplied_settings_and_history(self):
        runtime_state = build_model_runtime_state(
            settings=self.settings,
            history=self.history,
        )

        self.assertIs(runtime_state.settings, self.settings)
        self.assertIs(runtime_state.history, self.history)


if __name__ == "__main__":
    unittest.main()
