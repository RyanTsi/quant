"""Tests for scripts.view."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from scripts import view


class TestViewScript(unittest.TestCase):
    @patch("scripts.view.build_model_runtime_state")
    @patch("scripts.view.generate_analysis_view", return_value="/tmp/analysis/rec1")
    def test_generate_view_delegates_to_shared_helper(
        self,
        mock_generate_analysis_view,
        mock_build_runtime_state,
    ):
        runtime_state = Mock()
        runtime_state.settings.qlib_provider_uri = "provider://default"
        runtime_state.settings.qlib_mlruns_uri = "mlruns://default"
        runtime_state.settings.analysis_path = "/tmp/analysis"
        runtime_state.resolve_recorder_identity.return_value = view.RecorderIdentity(
            experiment_id="exp1",
            recorder_id="rec1",
        )
        mock_build_runtime_state.return_value = runtime_state

        result = view.generate_view(
            experiment_id="exp1",
            recorder_id="rec1",
            provider_uri="provider://test",
            mlruns_uri="mlruns://test",
        )

        self.assertEqual(result, "/tmp/analysis/rec1")
        mock_build_runtime_state.assert_called_once_with(refresh_settings=True)
        runtime_state.resolve_recorder_identity.assert_called_once_with(
            experiment_id="exp1",
            recorder_id="rec1",
            allow_settings_fallback=True,
        )
        mock_generate_analysis_view.assert_called_once_with(
            identity=view.RecorderIdentity(experiment_id="exp1", recorder_id="rec1"),
            provider_uri="provider://test",
            analysis_path="/tmp/analysis",
            mlruns_uri="mlruns://test",
        )

    @patch("scripts.view.generate_view")
    def test_main_forwards_cli_args(self, mock_generate_view):
        with patch("sys.argv", ["view.py", "--experiment_id", "exp2", "--recorder_id", "rec2"]):
            view.main()

        mock_generate_view.assert_called_once_with(experiment_id="exp2", recorder_id="rec2")


if __name__ == "__main__":
    unittest.main()
