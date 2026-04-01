"""Tests for quantcore.services.model_service."""

import os
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch

from quantcore.history import RunHistoryStore
from quantcore.services.model_service import ModelPipelineService
from quantcore.settings import load_settings


class TestModelPipelineService(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        base = load_settings(env={})
        self.settings = replace(
            base,
            data_path=self.tmp,
            receive_buffer_path=os.path.join(self.tmp, "receive_buffer"),
            qlib_data_path=os.path.join(self.tmp, "qlib_data"),
        )
        os.makedirs(self.settings.receive_buffer_path, exist_ok=True)
        os.makedirs(self.settings.qlib_data_path, exist_ok=True)
        self.history = RunHistoryStore(os.path.join(self.tmp, "run_history.json"))
        self.service = ModelPipelineService(self.settings, history=self.history)

    def tearDown(self):
        for root, dirs, files in os.walk(self.tmp, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.tmp)

    @patch("quantcore.services.model_service.subprocess.run")
    def test_dump_to_qlib_skip_when_no_csv(self, mock_run):
        result = self.service.dump_to_qlib()
        self.assertIsNone(result)
        mock_run.assert_not_called()

    @patch("quantcore.services.model_service.subprocess.run")
    def test_dump_to_qlib_runs_when_csv_exists(self, mock_run):
        path = os.path.join(self.settings.receive_buffer_path, "SH600000.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("date,open\n2026-03-31,10\n")

        result = self.service.dump_to_qlib()
        self.assertIsNotNone(result)
        mock_run.assert_called_once()
        self.assertIsNotNone(self.history.get("dump_to_qlib"))

    @patch("quantcore.services.model_service.subprocess.run")
    def test_predict_and_portfolio_use_subprocess(self, mock_run):
        self.service.predict()
        self.service.build_portfolio()
        self.assertEqual(mock_run.call_count, 2)

    @patch("quantcore.services.model_service._run_qlib_training")
    def test_train_model_calls_workflow_entry(self, mock_main):
        self.service.train_model()
        mock_main.assert_called_once()
        self.assertIsNotNone(self.history.get("train_model"))


if __name__ == "__main__":
    unittest.main()
