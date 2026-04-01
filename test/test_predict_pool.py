"""Tests for prediction pool and label alignment."""

import os
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

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
from scripts.predict import _build_today_pool, get_predict_conf


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

    @patch("scripts.predict.D")
    def test_prediction_pool_excludes_index_symbols(self, mock_d):
        calendar = [pd.Timestamp("2026-03-30"), pd.Timestamp("2026-03-31")]
        feats_idx = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2026-03-31"), "SH000001"),
                (pd.Timestamp("2026-03-31"), "SH600000"),
                (pd.Timestamp("2026-03-31"), "SZ000001"),
            ],
            names=["datetime", "instrument"],
        )
        feats = pd.DataFrame(
            {
                "$amount": [1000.0, 900.0, 800.0],
                "$close": [1.0, 1.0, 1.0],
                "$volume": [1.0, 1.0, 1.0],
                "$isst": [0.0, 1.0, 0.0],
            },
            index=feats_idx,
        )
        mock_d.features.return_value = feats
        mock_d.instruments.return_value = "all"

        original_cwd = os.getcwd()
        tmp = tempfile.mkdtemp()
        try:
            os.chdir(tmp)
            os.makedirs("output", exist_ok=True)
            pd.DataFrame(
                {
                    "instrument": ["SH000001", "SH600000", "SH600010"],
                    "Score": [0.99, 0.95, 0.88],
                }
            ).to_csv("output/top_picks_2026-03-30.csv", index=False)

            os.makedirs(".data", exist_ok=True)
            Path(".data/index_code_list").write_text("SH000001\n", encoding="utf-8")

            with patch("scripts.predict.settings") as mock_settings:
                mock_settings.data_path = ".data"
                pool = _build_today_pool(calendar, "2026-03-31", seed=42)

            self.assertNotIn("SH000001", pool)
            self.assertNotIn("SH600000", pool)
            self.assertIn("SH600010", pool)
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(tmp, ignore_errors=True)

    @patch("scripts.predict.D")
    def test_prediction_pool_fallback_for_small_universe(self, mock_d):
        calendar = [pd.Timestamp("2026-03-30"), pd.Timestamp("2026-03-31")]
        instruments = [f"SH600{i:03d}" for i in range(15)]
        feats_idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2026-03-31"), sym) for sym in instruments],
            names=["datetime", "instrument"],
        )
        feats = pd.DataFrame(
            {
                "$amount": list(range(1500, 1485, -1)),
                "$close": [1.0] * len(instruments),
                "$volume": [1.0] * len(instruments),
            },
            index=feats_idx,
        )
        mock_d.features.return_value = feats
        mock_d.instruments.return_value = "all"

        original_cwd = os.getcwd()
        tmp = tempfile.mkdtemp()
        try:
            os.chdir(tmp)
            os.makedirs(".data", exist_ok=True)
            Path(".data/index_code_list").write_text("", encoding="utf-8")

            with patch("scripts.predict.settings") as mock_settings:
                mock_settings.data_path = ".data"
                pool = _build_today_pool(calendar, "2026-03-31", seed=42)

            self.assertGreater(len(pool), 0)
            self.assertTrue(set(pool).issubset(set(instruments)))
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
