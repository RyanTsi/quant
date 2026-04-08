"""Tests for runtime.services.ModelPipelineService."""

import os
import tempfile
import types
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from runtime.config import load_settings
from runtime.model_state import NO_TRAINED_MODEL_ERROR
from runtime.runlog import RunLogStore
import runtime.services as model_service_module
from runtime.services import ModelPipelineService


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
        self.history = RunLogStore(os.path.join(self.tmp, "run_history.json"))
        self.service = ModelPipelineService(self.settings, history=self.history)

    def tearDown(self):
        for root, dirs, files in os.walk(self.tmp, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.tmp)

    @patch("runtime.services.modeling.dump_to_qlib_data")
    def test_dump_to_qlib_skip_when_no_csv(self, mock_dump):
        result = self.service.dump_to_qlib()
        self.assertIsNone(result)
        mock_dump.assert_not_called()

    @patch("runtime.services.modeling.dump_to_qlib_data")
    def test_dump_to_qlib_runs_when_csv_exists(self, mock_dump):
        path = os.path.join(self.settings.receive_buffer_path, "SH600000.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("date,open\n2026-03-31,10\n")

        result = self.service.dump_to_qlib()
        self.assertIsNotNone(result)
        mock_dump.assert_called_once_with(
            csv_dir=self.settings.receive_buffer_path,
            qlib_dir=self.settings.qlib_data_path,
            include_fields=model_service_module.modeling.DEFAULT_DUMP_INCLUDE_FIELDS,
            file_suffix=model_service_module.modeling.DEFAULT_DUMP_FILE_SUFFIX,
        )
        self.assertIsNotNone(self.history.get("dump_to_qlib"))

    @patch("runtime.services.modeling.dump_to_qlib_data")
    def test_dump_to_qlib_failure_does_not_record_success_entry(self, mock_dump):
        path = os.path.join(self.settings.receive_buffer_path, "SH600000.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("date,open\n2026-03-31,10\n")

        mock_dump.side_effect = RuntimeError("dump failed")
        with self.assertRaises(RuntimeError):
            self.service.dump_to_qlib()

        self.assertIsNone(self.history.get("dump_to_qlib"))

    @patch("runtime.services.modeling.build_portfolio_outputs")
    @patch("runtime.services.modeling.generate_predictions")
    def test_predict_and_portfolio_use_direct_adapter(
        self,
        mock_predict,
        mock_build_portfolio,
    ):
        mock_runtime_state = Mock()
        recorder_identity = types.SimpleNamespace(
            experiment_id="exp-1",
            recorder_id="rec-1",
        )
        mock_runtime_state.resolve_recorder_identity.return_value = recorder_identity
        self.service.runtime_state = mock_runtime_state
        mock_predict.return_value = {
            "predict_date": "2026-04-01",
            "lookback_start": "2025-10-01",
            "pool_size": 500,
            "recorder_id": "rec-1",
            "experiment_id": "exp-1",
            "output_path": "output/top_picks_2026-04-01.csv",
        }
        mock_build_portfolio.return_value = {
            "date": "2026-04-01",
            "picks_path": "output/top_picks_2026-04-01.csv",
            "target_path": "output/target_weights_2026-04-01.csv",
            "orders_path": "output/orders_2026-04-01.csv",
            "stats": {"turnover": 0.1, "order_count": 12},
        }

        predict_result = self.service.predict()
        portfolio_result = self.service.build_portfolio()
        mock_runtime_state.resolve_recorder_identity.assert_called_once_with()
        mock_predict.assert_called_once_with(
            date=None,
            out=None,
            runtime_state=mock_runtime_state,
            recorder_identity=recorder_identity,
        )
        mock_build_portfolio.assert_called_once_with(
            date=None,
            top_k=80,
            max_weight=0.02,
            rebalance_threshold=0.002,
            buy_rank=300,
            hold_rank=500,
            track_run=False,
            runtime_state=mock_runtime_state,
        )
        self.assertEqual(predict_result["predict_date"], "2026-04-01")
        self.assertEqual(portfolio_result["target_path"], "output/target_weights_2026-04-01.csv")

        predict_entry = self.history.get("predict")
        self.assertIsNotNone(predict_entry)
        self.assertEqual(predict_entry["predict_date"], "2026-04-01")
        self.assertEqual(predict_entry["output_path"], "output/top_picks_2026-04-01.csv")
        self.assertEqual(predict_entry["pool_size"], 500)

        portfolio_entry = self.history.get("build_portfolio")
        self.assertIsNotNone(portfolio_entry)
        self.assertEqual(portfolio_entry["target_file"], "output/target_weights_2026-04-01.csv")
        self.assertEqual(portfolio_entry["orders_file"], "output/orders_2026-04-01.csv")
        self.assertEqual(portfolio_entry["turnover"], 0.1)

    @patch("runtime.services.modeling.generate_predictions")
    def test_predict_uses_service_owned_runtime_state_for_recorder_resolution(self, mock_predict):
        mock_runtime_state = Mock()
        recorder_identity = types.SimpleNamespace(
            experiment_id="exp-service",
            recorder_id="rec-service",
        )
        mock_runtime_state.resolve_recorder_identity.return_value = recorder_identity
        self.service.runtime_state = mock_runtime_state
        mock_predict.return_value = {
            "predict_date": "2026-04-01",
            "lookback_start": "2025-10-01",
            "pool_size": 10,
            "recorder_id": "rec-service",
            "experiment_id": "exp-service",
            "output_path": "output/top_picks_2026-04-01.csv",
        }

        self.service.predict()

        mock_runtime_state.resolve_recorder_identity.assert_called_once_with()
        mock_predict.assert_called_once_with(
            date=None,
            out=None,
            runtime_state=mock_runtime_state,
            recorder_identity=recorder_identity,
        )

    @patch("runtime.services.modeling.generate_predictions")
    def test_predict_forwards_explicit_args(self, mock_predict):
        mock_runtime_state = Mock()
        recorder_identity = types.SimpleNamespace(
            experiment_id="exp-explicit",
            recorder_id="rec-explicit",
        )
        mock_runtime_state.resolve_recorder_identity.return_value = recorder_identity
        self.service.runtime_state = mock_runtime_state
        mock_predict.return_value = {
            "predict_date": "2026-04-02",
            "lookback_start": "2025-10-02",
            "pool_size": 12,
            "recorder_id": "rec-explicit",
            "experiment_id": "exp-explicit",
            "output_path": "output/custom.csv",
        }

        self.service.predict(date="2026-04-02", out="output/custom.csv")

        mock_predict.assert_called_once_with(
            date="2026-04-02",
            out="output/custom.csv",
            runtime_state=mock_runtime_state,
            recorder_identity=recorder_identity,
        )

    @patch("runtime.services.modeling.generate_predictions")
    def test_predict_fails_fast_when_no_env_or_training_run_exists(self, mock_predict):
        mock_runtime_state = Mock()
        mock_runtime_state.resolve_recorder_identity.side_effect = RuntimeError(NO_TRAINED_MODEL_ERROR)
        self.service.runtime_state = mock_runtime_state

        with self.assertRaisesRegex(RuntimeError, NO_TRAINED_MODEL_ERROR):
            self.service.predict()

        mock_runtime_state.resolve_recorder_identity.assert_called_once_with()
        mock_predict.assert_not_called()
        self.assertIsNone(self.history.get("predict"))

    @patch("runtime.services.modeling.build_training_universe_file")
    def test_build_training_universe_uses_direct_adapter(self, mock_build):
        mock_build.return_value = {
            "output_path": "output/my_800_stocks.txt",
            "start_year": 2011,
            "end_year": 2020,
            "top_n": 1800,
            "random_seed": 9,
            "effective_end": "2026-04-03",
            "source_month_count": 3,
            "range_count": 10,
            "symbol_count": 1200,
        }

        result = self.service.build_training_universe(start_year=2011, end_year=2020, top_n=1800, random_seed=9)

        mock_build.assert_called_once_with(
            start_year=2011,
            end_year=2020,
            top_n=1800,
            random_seed=9,
            data_path=self.settings.data_path,
            qlib_dir=self.settings.qlib_data_path,
            db_host=self.settings.db_host,
            db_port=self.settings.db_port,
        )
        self.assertEqual(result["output_path"], "output/my_800_stocks.txt")
        history_entry = self.history.get("filter_training_universe")
        self.assertIsNotNone(history_entry)
        self.assertEqual(history_entry["output_path"], "output/my_800_stocks.txt")
        self.assertEqual(history_entry["top_n"], 1800)
        self.assertEqual(history_entry["random_seed"], 9)
        self.assertEqual(history_entry["symbol_count"], 1200)

    def test_model_service_module_has_no_subprocess_dependency(self):
        self.assertFalse(hasattr(model_service_module, "subprocess"))

    @patch("runtime.services.modeling.build_portfolio_outputs")
    @patch("runtime.services.modeling.generate_predictions")
    def test_direct_call_failure_recovery_keeps_runlog_consistent(
        self,
        mock_predict,
        mock_build_portfolio,
    ):
        mock_runtime_state = Mock()
        recorder_identity = types.SimpleNamespace(
            experiment_id="exp-1",
            recorder_id="rec-1",
        )
        mock_runtime_state.resolve_recorder_identity.return_value = recorder_identity
        self.service.runtime_state = mock_runtime_state
        # Predict fail first: no success record should be written.
        mock_predict.side_effect = [
            RuntimeError("predict failed"),
            {
                "predict_date": "2026-04-01",
                "lookback_start": "2025-10-01",
                "pool_size": 500,
                "recorder_id": "rec-1",
                "experiment_id": "exp-1",
                "output_path": "output/top_picks_2026-04-01.csv",
            },
        ]
        with self.assertRaises(RuntimeError):
            self.service.predict()
        self.assertIsNone(self.history.get("predict"))

        # Predict recovers: success record should appear.
        predict_result = self.service.predict()
        self.assertEqual(predict_result["predict_date"], "2026-04-01")
        predict_entry = self.history.get("predict")
        self.assertIsNotNone(predict_entry)
        self.assertEqual(predict_entry["output_path"], "output/top_picks_2026-04-01.csv")
        saved_predict_entry = dict(predict_entry)

        # Portfolio fail after predict success: existing predict record must stay intact.
        mock_build_portfolio.side_effect = RuntimeError("portfolio failed")
        with self.assertRaises(RuntimeError):
            self.service.build_portfolio()
        self.assertIsNone(self.history.get("build_portfolio"))
        self.assertEqual(self.history.get("predict"), saved_predict_entry)

        # Portfolio recovers with the expected artifact-shaped adapter payload.
        mock_build_portfolio.side_effect = None
        mock_build_portfolio.return_value = {
            "date": "2026-04-01",
            "picks_path": "output/top_picks_2026-04-01.csv",
            "target_path": "output/target_weights_2026-04-01.csv",
            "orders_path": "output/orders_2026-04-01.csv",
            "stats": {"turnover": 0.0},
        }
        portfolio_result = self.service.build_portfolio()
        self.assertEqual(portfolio_result["target_path"], "output/target_weights_2026-04-01.csv")
        build_portfolio_entry = self.history.get("build_portfolio")
        self.assertIsNotNone(build_portfolio_entry)
        self.assertEqual(build_portfolio_entry["target_file"], "output/target_weights_2026-04-01.csv")
        self.assertEqual(build_portfolio_entry["turnover"], 0.0)
        self.assertEqual(mock_build_portfolio.call_count, 2)
        self.assertIn("target_path", mock_build_portfolio.return_value)
        self.assertIn("orders_path", mock_build_portfolio.return_value)

    @patch("runtime.services._run_qlib_training")
    def test_train_model_calls_workflow_entry(self, mock_main):
        mock_main.return_value = types.SimpleNamespace(
            config_source="cfg.yaml",
            experiment_id="exp_1",
            recorder_id="rec_1",
            metrics=None,
        )

        with patch.object(self.service, "_today_dash", return_value="2026-04-01"):
            result = self.service.train_model()

        mock_main.assert_called_once_with(self.service.runtime_state)
        self.assertEqual(result, {"date": "2026-04-01"})
        self.assertEqual(self.history.get("train_model")["date"], "2026-04-01")
        qlib_train_entry = self.history.get("qlib_train")
        self.assertIsNotNone(qlib_train_entry)
        self.assertEqual(qlib_train_entry["config_source"], "cfg.yaml")
        self.assertEqual(qlib_train_entry["experiment_id"], "exp_1")

    @patch("runtime.services.modeling.build_portfolio_outputs")
    def test_build_portfolio_forwards_explicit_args(self, mock_build_portfolio):
        mock_build_portfolio.return_value = {
            "date": "2026-04-02",
            "picks_path": "output/top_picks_2026-04-02.csv",
            "target_path": "output/target_weights_2026-04-02.csv",
            "orders_path": "output/orders_2026-04-02.csv",
            "stats": {"turnover": 0.2},
        }

        self.service.build_portfolio(
            date="2026-04-02",
            top_k=60,
            buy_rank=250,
            hold_rank=400,
            max_weight=0.03,
            rebalance_threshold=0.005,
        )

        mock_build_portfolio.assert_called_once_with(
            date="2026-04-02",
            top_k=60,
            max_weight=0.03,
            rebalance_threshold=0.005,
            buy_rank=250,
            hold_rank=400,
            track_run=False,
            runtime_state=self.service.runtime_state,
        )
        self.assertEqual(self.history.get("build_portfolio")["target_file"], "output/target_weights_2026-04-02.csv")


if __name__ == "__main__":
    unittest.main()
