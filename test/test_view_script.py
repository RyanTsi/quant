"""Tests for scripts.view."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts import view


class TestResolveRecorderIds(unittest.TestCase):
    def test_explicit_ids_preferred(self):
        exp_id, rec_id = view._resolve_recorder_ids("exp_x", "rec_x")
        self.assertEqual(exp_id, "exp_x")
        self.assertEqual(rec_id, "rec_x")

    def test_env_ids_used_when_explicit_missing(self):
        with patch.dict("os.environ", {"QLIB_EXPERIMENT_ID": "exp_env", "QLIB_RECORDER_ID": "rec_env"}, clear=False):
            exp_id, rec_id = view._resolve_recorder_ids(None, None)
        self.assertEqual(exp_id, "exp_env")
        self.assertEqual(rec_id, "rec_env")

    @patch("scripts.view.get_last_run", return_value={"experiment_id": "exp_hist", "recorder_id": "rec_hist"})
    def test_run_history_used_after_env(self, _mock_last):
        with patch.dict("os.environ", {"QLIB_EXPERIMENT_ID": "", "QLIB_RECORDER_ID": ""}, clear=False):
            exp_id, rec_id = view._resolve_recorder_ids(None, None)
        self.assertEqual(exp_id, "exp_hist")
        self.assertEqual(rec_id, "rec_hist")


class TestGenerateView(unittest.TestCase):
    @patch("scripts.view.analysis_position.report_graph")
    @patch("scripts.view.analysis_model.model_performance_graph")
    @patch("scripts.view.R")
    @patch("scripts.view.qlib.init")
    def test_generate_view_writes_figures(
        self,
        mock_init,
        mock_r,
        mock_model_graph,
        mock_position_graph,
    ):
        recorder = MagicMock()
        mock_r.get_recorder.return_value = recorder

        model_fig = MagicMock()
        port_fig = MagicMock()
        mock_model_graph.return_value = [model_fig]
        mock_position_graph.return_value = [port_fig]

        with tempfile.TemporaryDirectory() as tmp:
            with patch("scripts.view.settings") as mock_settings:
                mock_settings.analysis_path = tmp
                mock_settings.qlib_provider_uri = "provider_default"
                mock_settings.qlib_mlruns_uri = "mlruns_default"
                # Build label/pred DataFrame behavior expected by generate_view.
                import pandas as pd

                pred_df = pd.DataFrame({"score": [0.1]}, index=[0])
                label_df = pd.DataFrame({"label": [0.2]}, index=[0])
                recorder.load_object.side_effect = lambda key: {
                    "pred.pkl": pred_df,
                    "label.pkl": label_df,
                    "portfolio_analysis/report_normal_1day.pkl": pd.DataFrame({"return": [0.1]}),
                }[key]

                out = view.generate_view(
                    experiment_id="exp1",
                    recorder_id="rec1",
                    provider_uri="provider_x",
                    mlruns_uri="mlruns_x",
                )

        self.assertEqual(os.path.basename(out), "rec1")
        mock_init.assert_called_once()
        mock_r.set_uri.assert_called_once_with("mlruns_x")
        mock_r.get_recorder.assert_called_once_with(experiment_id="exp1", recorder_id="rec1")
        self.assertTrue(model_fig.write_html.called)
        self.assertTrue(port_fig.write_html.called)

    @patch("scripts.view.analysis_position.report_graph")
    @patch("scripts.view.analysis_model.model_performance_graph")
    @patch("scripts.view.R")
    @patch("scripts.view.qlib.init")
    def test_generate_view_skips_missing_portfolio_artifact(
        self,
        _mock_init,
        mock_r,
        mock_model_graph,
        _mock_position_graph,
    ):
        import pandas as pd

        recorder = MagicMock()

        def _load(key):
            if key == "pred.pkl":
                return pd.DataFrame({"score": [0.1]}, index=[0])
            if key == "label.pkl":
                return pd.DataFrame({"label": [0.2]}, index=[0])
            raise FileNotFoundError("missing portfolio artifact")

        recorder.load_object.side_effect = _load
        mock_r.get_recorder.return_value = recorder

        fig = MagicMock()
        mock_model_graph.return_value = [fig]

        with tempfile.TemporaryDirectory() as tmp:
            with patch("scripts.view.settings") as mock_settings:
                mock_settings.analysis_path = tmp
                mock_settings.qlib_provider_uri = "provider_default"
                mock_settings.qlib_mlruns_uri = "mlruns_default"
                out = view.generate_view(experiment_id="exp2", recorder_id="rec2")

        self.assertEqual(os.path.basename(out), "rec2")
        self.assertTrue(fig.write_html.called)


if __name__ == "__main__":
    unittest.main()
