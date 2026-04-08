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
from runtime.model_state import NO_TRAINED_MODEL_ERROR
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
from runtime.adapters.modeling import (
    TradingDateContext,
    _build_today_pool,
    _resolve_trading_date_context,
    generate_predictions,
    get_predict_conf,
)


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

            with patch("runtime.adapters.modeling.build_model_runtime_state") as mock_runtime_state:
                mock_runtime_state.return_value = types.SimpleNamespace(
                    settings=types.SimpleNamespace(data_path=".data")
                )
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
                with patch("runtime.adapters.modeling.build_model_runtime_state") as mock_runtime_state:
                    mock_runtime_state.return_value = types.SimpleNamespace(
                        settings=types.SimpleNamespace(data_path=".data")
                    )
                    pool = _build_today_pool(calendar, "2026-03-31", seed=42)

            self.assertEqual(pool, ["A", "B", "C", "D"])
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(tmp, ignore_errors=True)

    @patch("runtime.adapters.modeling._get_qlib_data_client")
    def test_prediction_pool_uses_previous_trading_day_across_weekend_gap(self, mock_get_d):
        mock_d = MagicMock()
        mock_get_d.return_value = mock_d
        calendar = [
            pd.Timestamp("2026-04-02"),
            pd.Timestamp("2026-04-03"),
            pd.Timestamp("2026-04-06"),
        ]
        instruments = ["A", "B", "C", "D", "E", "F"]
        feats = pd.DataFrame(
            {
                "$amount": [600.0, 500.0, 400.0, 300.0, 200.0, 100.0],
                "$close": [1.0] * 6,
                "$volume": [1.0] * 6,
                "$isst": [0.0] * 6,
            },
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2026-04-06"), instrument) for instrument in instruments],
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
            pd.DataFrame({"instrument": ["D"], "target_weight": [1.0]}).to_csv(
                "output/target_weights_2026-04-03.csv",
                index=False,
            )
            os.makedirs(".data", exist_ok=True)
            Path(".data/index_code_list").write_text("", encoding="utf-8")

            with patch("runtime.adapters.modeling.PREDICTION_UNIVERSE_DEFAULTS", PredictionUniverseConfig(entry_limit=3, exit_limit=5)):
                with patch("runtime.adapters.modeling.build_model_runtime_state") as mock_runtime_state:
                    mock_runtime_state.return_value = types.SimpleNamespace(
                        settings=types.SimpleNamespace(data_path=".data")
                    )
                    pool = _build_today_pool(calendar, "2026-04-06", seed=42)

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
                with patch("runtime.adapters.modeling.build_model_runtime_state") as mock_runtime_state:
                    mock_runtime_state.return_value = types.SimpleNamespace(
                        settings=types.SimpleNamespace(data_path=".data")
                    )
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

            with patch("runtime.adapters.modeling.build_model_runtime_state") as mock_runtime_state:
                mock_runtime_state.return_value = types.SimpleNamespace(
                    settings=types.SimpleNamespace(data_path=".data")
                )
                first = _build_today_pool(calendar, "2026-03-31", seed=42)
                second = _build_today_pool(calendar, "2026-03-31", seed=7)

            self.assertEqual(first, second)
            self.assertTrue(set(first).issubset(set(instruments)))
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(tmp, ignore_errors=True)

    def test_resolve_trading_date_context_defaults_to_latest_local_trading_day(self):
        calendar = [pd.Timestamp("2026-04-02"), pd.Timestamp("2026-04-03"), pd.Timestamp("2026-04-06")]

        context = _resolve_trading_date_context(calendar, None)

        self.assertEqual(
            context,
            TradingDateContext(
                trading_date="2026-04-06",
                previous_trading_date="2026-04-03",
            ),
        )

    def test_resolve_trading_date_context_rejects_non_trading_day(self):
        calendar = [pd.Timestamp("2026-04-02"), pd.Timestamp("2026-04-03"), pd.Timestamp("2026-04-06")]

        with self.assertRaisesRegex(ValueError, "not in local trading calendar"):
            _resolve_trading_date_context(calendar, "2026-04-04")

    @patch("runtime.adapters.modeling._build_today_pool", return_value=["SH600000"])
    @patch("runtime.adapters.modeling._get_qlib_data_client")
    @patch("runtime.adapters.modeling._get_qlib_runtime")
    @patch("runtime.adapters.modeling.build_model_runtime_state")
    def test_generate_predictions_direct_call_writes_output(
        self,
        mock_build_runtime_state,
        mock_get_runtime,
        mock_get_d,
        _mock_pool,
    ):
        calendar = [pd.Timestamp("2026-03-30"), pd.Timestamp("2026-03-31")]
        mock_d = MagicMock()
        mock_d.calendar.return_value = calendar
        mock_get_d.return_value = mock_d
        mock_build_runtime_state.return_value = types.SimpleNamespace(
            settings=types.SimpleNamespace(
                qlib_provider_uri="provider://default",
                qlib_mlruns_uri="mlruns://default",
                data_path=".data",
            ),
            resolve_recorder_identity=lambda **_: types.SimpleNamespace(
                recorder_id="rec_1",
                experiment_id="exp_1",
            ),
        )

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
            result = generate_predictions(date="2026-03-31", output_dir=tmp)

            out_path = Path(result["output_path"])
            self.assertTrue(out_path.exists())
            out_df = pd.read_csv(out_path)
            self.assertIn("Score", out_df.columns)
            self.assertEqual(result["predict_date"], "2026-03-31")
            self.assertEqual(result["pool_size"], 1)

    @patch("runtime.adapters.modeling._build_today_pool", return_value=["SH600000"])
    @patch("runtime.adapters.modeling._get_qlib_data_client")
    @patch("runtime.adapters.modeling._get_qlib_runtime")
    @patch("runtime.adapters.modeling.build_model_runtime_state")
    def test_generate_predictions_fails_fast_without_env_or_training_run(
        self,
        mock_build_runtime_state,
        mock_get_runtime,
        mock_get_d,
        _mock_pool,
    ):
        calendar = [pd.Timestamp("2026-03-30"), pd.Timestamp("2026-03-31")]
        mock_d = MagicMock()
        mock_d.calendar.return_value = calendar
        mock_get_d.return_value = mock_d
        mock_get_runtime.return_value = (MagicMock(), "cn", MagicMock(), lambda _conf: "dataset")
        mock_build_runtime_state.return_value = types.SimpleNamespace(
            settings=types.SimpleNamespace(
                qlib_provider_uri="provider://default",
                qlib_mlruns_uri="mlruns://default",
                data_path=".data",
            ),
            resolve_recorder_identity=lambda **_: (_ for _ in ()).throw(RuntimeError(NO_TRAINED_MODEL_ERROR)),
        )

        with self.assertRaisesRegex(RuntimeError, NO_TRAINED_MODEL_ERROR):
            generate_predictions(date="2026-03-31", output_dir="output")


if __name__ == "__main__":
    unittest.main()
