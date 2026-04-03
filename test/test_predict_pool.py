"""Tests for prediction pool and label alignment."""

import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import yaml

from model_function.universe import PredictionUniverseConfig
from utils.preprocess import ALPHA158_WEIGHTED_5D_LABEL


def _install_qlib_stubs_if_missing():
    if "qlib" in sys.modules:
        return

    qlib_mod = types.ModuleType("qlib")
    qlib_mod.init = lambda *args, **kwargs: None

    qlib_data = types.ModuleType("qlib.data")
    qlib_data.D = types.SimpleNamespace()

    qlib_workflow = types.ModuleType("qlib.workflow")
    qlib_workflow.R = types.SimpleNamespace()

    qlib_config = types.ModuleType("qlib.config")
    qlib_config.REG_CN = "cn"

    qlib_utils = types.ModuleType("qlib.utils")
    qlib_utils.init_instance_by_config = lambda conf: conf

    qlib_contrib = types.ModuleType("qlib.contrib")
    qlib_contrib_report = types.ModuleType("qlib.contrib.report")
    qlib_contrib_report.analysis_model = types.SimpleNamespace(
        model_performance_graph=lambda *args, **kwargs: []
    )
    qlib_contrib_report.analysis_position = types.SimpleNamespace(
        report_graph=lambda *args, **kwargs: []
    )
    qlib_contrib_model = types.ModuleType("qlib.contrib.model")
    qlib_contrib_model_pts = types.ModuleType("qlib.contrib.model.pytorch_transformer_ts")

    sys.modules["qlib"] = qlib_mod
    sys.modules["qlib.data"] = qlib_data
    sys.modules["qlib.workflow"] = qlib_workflow
    sys.modules["qlib.config"] = qlib_config
    sys.modules["qlib.utils"] = qlib_utils
    sys.modules["qlib.contrib"] = qlib_contrib
    sys.modules["qlib.contrib.report"] = qlib_contrib_report
    sys.modules["qlib.contrib.model"] = qlib_contrib_model
    sys.modules["qlib.contrib.model.pytorch_transformer_ts"] = qlib_contrib_model_pts


_install_qlib_stubs_if_missing()
from runtime.adapters.modeling import _build_today_pool, generate_predictions, get_predict_conf


class TestPredictPool(unittest.TestCase):
    def test_predict_label_aligned_with_training_config(self):
        conf = get_predict_conf("2026-01-01", "2026-01-31", ["SH600000"])
        pred_label = conf["kwargs"]["handler"]["kwargs"]["label"]

        train_cfg = yaml.safe_load(
            Path("alpha_models/workflow_config_transformer_Alpha158.yaml").read_text(encoding="utf-8")
        )
        train_label = train_cfg["data_handler_config"]["label"]

        self.assertEqual(pred_label, [ALPHA158_WEIGHTED_5D_LABEL])
        self.assertEqual(pred_label, train_label)

    @patch("runtime.adapters.modeling._get_qlib_data_client")
    def test_prediction_pool_excludes_index_symbols_and_st(self, mock_get_d):
        mock_d = MagicMock()
        mock_get_d.return_value = mock_d
        calendar = [pd.Timestamp("2026-03-30"), pd.Timestamp("2026-03-31")]
        feats = pd.DataFrame(
            {
                "$amount": [1000.0, 900.0, 800.0],
                "$close": [1.0, 1.0, 1.0],
                "$volume": [1.0, 1.0, 1.0],
                "$isst": [0.0, 1.0, 0.0],
            },
            index=pd.MultiIndex.from_tuples(
                [
                    (pd.Timestamp("2026-03-31"), "SH000001"),
                    (pd.Timestamp("2026-03-31"), "SH600000"),
                    (pd.Timestamp("2026-03-31"), "SH600010"),
                ],
                names=["datetime", "instrument"],
            ),
        )
        mock_d.features.return_value = feats
        mock_d.instruments.return_value = "all"

        original_cwd = os.getcwd()
        tmp = tempfile.mkdtemp()
        try:
            os.chdir(tmp)
            os.makedirs("output", exist_ok=True)
            os.makedirs(".data", exist_ok=True)
            Path(".data/index_code_list").write_text("SH000001\n", encoding="utf-8")

            with patch("runtime.adapters.modeling.settings") as mock_settings:
                mock_settings.data_path = ".data"
                pool = _build_today_pool(calendar, "2026-03-31", seed=42)

            self.assertEqual(pool, ["SH600010"])
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(tmp, ignore_errors=True)

    @patch("runtime.adapters.modeling._get_qlib_data_client")
    def test_prediction_pool_retains_previous_holdings_within_exit_band(self, mock_get_d):
        mock_d = MagicMock()
        mock_get_d.return_value = mock_d
        calendar = [pd.Timestamp("2026-03-30"), pd.Timestamp("2026-03-31")]
        instruments = ["A", "B", "C", "D", "E", "F"]
        feats = pd.DataFrame(
            {
                "$amount": [600.0, 500.0, 400.0, 300.0, 200.0, 100.0],
                "$close": [1.0] * 6,
                "$volume": [1.0] * 6,
                "$isst": [0.0] * 6,
            },
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2026-03-31"), instrument) for instrument in instruments],
                names=["datetime", "instrument"],
            ),
        )
        mock_d.features.return_value = feats
        mock_d.instruments.return_value = "all"

        original_cwd = os.getcwd()
        tmp = tempfile.mkdtemp()
        try:
            os.chdir(tmp)
            os.makedirs("output", exist_ok=True)
            pd.DataFrame({"instrument": ["D", "F"], "target_weight": [0.5, 0.5]}).to_csv(
                "output/target_weights_2026-03-30.csv",
                index=False,
            )
            os.makedirs(".data", exist_ok=True)
            Path(".data/index_code_list").write_text("", encoding="utf-8")

            with patch("runtime.adapters.modeling.PREDICTION_UNIVERSE_DEFAULTS", PredictionUniverseConfig(entry_limit=3, exit_limit=5)):
                with patch("runtime.adapters.modeling.settings") as mock_settings:
                    mock_settings.data_path = ".data"
                    pool = _build_today_pool(calendar, "2026-03-31", seed=42)

            self.assertEqual(pool, ["A", "B", "C", "D"])
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(tmp, ignore_errors=True)

    @patch("runtime.adapters.modeling._get_qlib_data_client")
    def test_prediction_pool_ignores_unreadable_previous_target_file(self, mock_get_d):
        mock_d = MagicMock()
        mock_get_d.return_value = mock_d
        calendar = [pd.Timestamp("2026-03-30"), pd.Timestamp("2026-03-31")]
        instruments = ["A", "B", "C"]
        feats = pd.DataFrame(
            {
                "$amount": [300.0, 200.0, 100.0],
                "$close": [1.0, 1.0, 1.0],
                "$volume": [1.0, 1.0, 1.0],
                "$isst": [0.0, 0.0, 0.0],
            },
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2026-03-31"), instrument) for instrument in instruments],
                names=["datetime", "instrument"],
            ),
        )
        mock_d.features.return_value = feats
        mock_d.instruments.return_value = "all"

        original_cwd = os.getcwd()
        tmp = tempfile.mkdtemp()
        try:
            os.chdir(tmp)
            os.makedirs("output", exist_ok=True)
            Path("output/target_weights_2026-03-30.csv").write_text("bad csv", encoding="utf-8")
            os.makedirs(".data", exist_ok=True)
            Path(".data/index_code_list").write_text("", encoding="utf-8")

            with patch("runtime.adapters.modeling.pd.read_csv", side_effect=ValueError("bad csv")):
                with patch("runtime.adapters.modeling.settings") as mock_settings:
                    mock_settings.data_path = ".data"
                    pool = _build_today_pool(calendar, "2026-03-31", seed=42)

            self.assertEqual(pool, ["A", "B", "C"])
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(tmp, ignore_errors=True)

    @patch("runtime.adapters.modeling._get_qlib_data_client")
    def test_prediction_pool_is_deterministic_for_same_snapshot(self, mock_get_d):
        mock_d = MagicMock()
        mock_get_d.return_value = mock_d
        calendar = [pd.Timestamp("2026-03-30"), pd.Timestamp("2026-03-31")]
        instruments = [f"SH600{i:03d}" for i in range(15)]
        feats = pd.DataFrame(
            {
                "$amount": list(range(1500, 1485, -1)),
                "$close": [1.0] * len(instruments),
                "$volume": [1.0] * len(instruments),
            },
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2026-03-31"), instrument) for instrument in instruments],
                names=["datetime", "instrument"],
            ),
        )
        mock_d.features.return_value = feats
        mock_d.instruments.return_value = "all"

        original_cwd = os.getcwd()
        tmp = tempfile.mkdtemp()
        try:
            os.chdir(tmp)
            os.makedirs(".data", exist_ok=True)
            Path(".data/index_code_list").write_text("", encoding="utf-8")

            with patch("runtime.adapters.modeling.settings") as mock_settings:
                mock_settings.data_path = ".data"
                first = _build_today_pool(calendar, "2026-03-31", seed=42)
                second = _build_today_pool(calendar, "2026-03-31", seed=7)

            self.assertEqual(first, second)
            self.assertTrue(set(first).issubset(set(instruments)))
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(tmp, ignore_errors=True)

    @patch("runtime.adapters.modeling._resolve_recorder_ids", return_value=("rec_1", "exp_1"))
    @patch("runtime.adapters.modeling._build_today_pool", return_value=["SH600000"])
    @patch("runtime.adapters.modeling._get_qlib_data_client")
    @patch("runtime.adapters.modeling._get_qlib_runtime")
    def test_generate_predictions_direct_call_writes_output(
        self,
        mock_get_runtime,
        mock_get_d,
        _mock_pool,
        _mock_resolve_ids,
    ):
        calendar = [pd.Timestamp("2026-03-30"), pd.Timestamp("2026-03-31")]
        mock_d = MagicMock()
        mock_d.calendar.return_value = calendar
        mock_get_d.return_value = mock_d

        model = MagicMock()
        pred_index = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2026-03-31"), "SH600000")],
            names=["datetime", "instrument"],
        )
        model.predict.return_value = pd.Series([0.9], index=pred_index)

        recorder = MagicMock()
        recorder.load_object.return_value = model
        recorder_client = MagicMock()
        recorder_client.get_recorder.return_value = recorder
        mock_get_runtime.return_value = (MagicMock(), "cn", recorder_client, lambda _conf: "dataset")

        with tempfile.TemporaryDirectory() as tmp:
            with patch("runtime.adapters.modeling.settings") as mock_settings:
                mock_settings.qlib_provider_uri = "provider://default"
                mock_settings.qlib_mlruns_uri = "mlruns://default"
                mock_settings.data_path = ".data"

                result = generate_predictions(date="2026-03-31", output_dir=tmp)

            out_path = Path(result["output_path"])
            self.assertTrue(out_path.exists())
            out_df = pd.read_csv(out_path)
            self.assertIn("Score", out_df.columns)
            self.assertEqual(result["predict_date"], "2026-03-31")
            self.assertEqual(result["pool_size"], 1)


if __name__ == "__main__":
    unittest.main()
