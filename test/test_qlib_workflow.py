"""Tests for alpha_models.qlib_workflow."""

import types
import unittest
from unittest.mock import patch

from alpha_models import qlib_workflow


class TestQlibWorkflow(unittest.TestCase):
    @patch("alpha_models.qlib_workflow._run_post_train_view")
    @patch("alpha_models.qlib_workflow.run_training_workflow")
    def test_run_training_uses_runtime_state_and_runs_view(
        self,
        mock_run_training_workflow,
        mock_post_view,
    ):
        runtime_state = types.SimpleNamespace(
            resolve_training_workflow_inputs=lambda: types.SimpleNamespace(
                config_source="cfg.yaml",
                provider_uri="provider://default",
                mlruns_uri="mlruns://default",
                experiment_name="exp_name",
            ),
        )
        mock_run_training_workflow.return_value = types.SimpleNamespace(
            config_source="cfg.yaml",
            experiment_id="exp_1",
            recorder_id="rec_1",
            metrics=None,
        )

        result = qlib_workflow.run_training(runtime_state=runtime_state)

        mock_run_training_workflow.assert_called_once_with(
            config_source="cfg.yaml",
            provider_uri="provider://default",
            mlruns_uri="mlruns://default",
            experiment_name="exp_name",
        )
        mock_post_view.assert_called_once_with(
            runtime_state,
            experiment_id="exp_1",
            recorder_id="rec_1",
        )
        self.assertEqual(result.experiment_id, "exp_1")

    @patch("alpha_models.qlib_workflow.run_training")
    def test_main_delegates_to_run_training(self, mock_run_training):
        qlib_workflow.main()

        mock_run_training.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
