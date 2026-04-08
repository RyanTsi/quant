"""Tests for scripts.eval_test."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

from scripts import eval_test
from runtime.model_state import NO_TRAINED_MODEL_ERROR


class TestEvalTestScript(unittest.TestCase):
    @patch("scripts.eval_test.evaluate_test_predictions", return_value={"IC": 0.1, "Rank IC": 0.2})
    @patch("scripts.eval_test.build_model_runtime_state")
    def test_main_resolves_inputs_and_delegates_to_shared_helper(
        self,
        mock_build_runtime_state,
        mock_evaluate_test_predictions,
    ):
        runtime_state = Mock()
        runtime_state.settings.qlib_workflow_config = "settings.yaml"
        runtime_state.resolve_training_workflow_inputs.return_value = Mock(
            config_source="resolved.yaml",
            provider_uri="provider://default",
            mlruns_uri="mlruns://default",
        )
        runtime_state.resolve_recorder_identity.return_value = Mock(
            experiment_id="exp_final",
            recorder_id="rec_final",
        )
        mock_build_runtime_state.return_value = runtime_state

        buffer = io.StringIO()
        with patch("sys.argv", ["eval_test.py", "--config", "cli.yaml"]):
            with redirect_stdout(buffer):
                eval_test.main()

        mock_build_runtime_state.assert_called_once_with(refresh_settings=True)
        runtime_state.resolve_training_workflow_inputs.assert_called_once_with(config_source="cli.yaml")
        runtime_state.resolve_recorder_identity.assert_called_once_with(
            missing_error_message=NO_TRAINED_MODEL_ERROR,
        )
        mock_evaluate_test_predictions.assert_called_once_with(
            config_source="resolved.yaml",
            identity=runtime_state.resolve_recorder_identity.return_value,
            provider_uri="provider://default",
            mlruns_uri="mlruns://default",
        )
        output = buffer.getvalue()
        self.assertIn("== Full test segment metrics ==", output)
        self.assertIn("IC: 0.1", output)

    @patch("scripts.eval_test.evaluate_test_predictions")
    @patch("scripts.eval_test.build_model_runtime_state")
    def test_main_fails_fast_when_no_env_or_trained_run_exists(
        self,
        mock_build_runtime_state,
        mock_evaluate_test_predictions,
    ):
        runtime_state = Mock()
        runtime_state.settings.qlib_workflow_config = "settings.yaml"
        runtime_state.resolve_training_workflow_inputs.return_value = Mock(
            config_source="resolved.yaml",
            provider_uri="provider://default",
            mlruns_uri="mlruns://default",
        )
        runtime_state.resolve_recorder_identity.side_effect = RuntimeError(NO_TRAINED_MODEL_ERROR)
        mock_build_runtime_state.return_value = runtime_state

        with patch("sys.argv", ["eval_test.py"]):
            with self.assertRaisesRegex(RuntimeError, NO_TRAINED_MODEL_ERROR):
                eval_test.main()

        mock_evaluate_test_predictions.assert_not_called()


if __name__ == "__main__":
    unittest.main()
