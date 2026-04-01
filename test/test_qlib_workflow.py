"""Tests for alpha_models.qlib_workflow."""

import types
import unittest
from unittest.mock import patch

from alpha_models import qlib_workflow


class TestQlibWorkflow(unittest.TestCase):
    @patch("alpha_models.qlib_workflow._run_post_train_view")
    @patch("alpha_models.qlib_workflow.record_run")
    @patch("alpha_models.qlib_workflow.QlibWorkflowRunner")
    def test_main_runs_view_after_training(self, mock_runner_cls, mock_record_run, mock_post_view):
        mock_runner = mock_runner_cls.return_value
        mock_runner.run_from_yaml.return_value = types.SimpleNamespace(
            config_source="cfg.yaml",
            experiment_id="exp_1",
            recorder_id="rec_1",
            metrics=None,
        )

        qlib_workflow.main()

        mock_record_run.assert_called_once()
        mock_post_view.assert_called_once_with("exp_1", "rec_1")


if __name__ == "__main__":
    unittest.main()
