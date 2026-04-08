"""Tests for runtime.services.DataPipelineService."""

import os
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch

from runtime.config import load_settings
from runtime.runlog import RunLogStore
from runtime.services import DataPipelineService


class TestDataPipelineService(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        base = load_settings(env={})
        self.settings = replace(
            base,
            data_path=self.tmp,
            analysis_path=os.path.join(self.tmp, "analysis"),
            send_buffer_path=os.path.join(self.tmp, "send_buffer"),
            receive_buffer_path=os.path.join(self.tmp, "receive_buffer"),
            qlib_data_path=os.path.join(self.tmp, "qlib_data"),
        )
        os.makedirs(self.settings.send_buffer_path, exist_ok=True)
        os.makedirs(self.settings.receive_buffer_path, exist_ok=True)
        self.history = RunLogStore(os.path.join(self.tmp, "run_history.json"))
        self.service = DataPipelineService(self.settings, history=self.history)

    def tearDown(self):
        for root, dirs, files in os.walk(self.tmp, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.tmp)

    @patch("runtime.services.fetch_and_package_market_data")
    def test_fetch_data_delegates_to_adapter_and_records_run(self, mock_fetch):
        mock_fetch.return_value = {
            "start_date": "20260331",
            "end_date": "20260401",
            "last_end_date": "20260401",
            "lookback_days": 1,
            "save_dir": os.path.join(self.tmp, "20260331-20260401"),
            "send_buffer_dir": self.settings.send_buffer_path,
        }

        result = self.service.fetch_data(lookback_days=1)

        mock_fetch.assert_called_once_with(
            data_root=self.settings.data_path,
            send_buffer_dir=self.settings.send_buffer_path,
            lookback_days=1,
            last_history=None,
            logger=self.service.logger,
        )
        self.assertEqual(result["start_date"], "20260331")
        history_entry = self.history.get("fetch_stock")
        self.assertIsNotNone(history_entry)
        self.assertEqual(history_entry["start_date"], "20260331")
        self.assertEqual(history_entry["end_date"], "20260401")
        self.assertEqual(history_entry["last_end_date"], "20260401")
        self.assertEqual(history_entry["lookback_days"], 1)
        self.assertEqual(history_entry["save_dir"], os.path.join(self.tmp, "20260331-20260401"))
        self.assertEqual(history_entry["send_buffer_dir"], self.settings.send_buffer_path)

    @patch("runtime.services.fetch_and_package_market_data", side_effect=RuntimeError("fetch failed"))
    def test_fetch_data_does_not_record_history_when_adapter_fails(self, _mock_fetch):
        with self.assertRaisesRegex(RuntimeError, "fetch failed"):
            self.service.fetch_data(lookback_days=1)

        self.assertIsNone(self.history.get("fetch_stock"))

    @patch("runtime.services.ingest_directory")
    def test_ingest_to_db_passes_delete_flag(self, mock_ingest):
        mock_ingest.return_value = {
            "data_dir": self.settings.send_buffer_path,
            "server_url": f"http://{self.settings.db_host}:{self.settings.db_port}",
            "files_found": 2,
            "files_ingested": 1,
            "skipped_files": ["EMPTY.csv"],
            "rows_sent": 10,
            "failed_files": ["BROKEN.csv"],
            "failed_batches": [{"file": "BROKEN.csv", "batch": 1}],
            "deleted_files": ["EMPTY.csv", "BROKEN.csv"],
        }
        result = self.service.ingest_to_db(delete_after_ingest=True)
        mock_ingest.assert_called_once()
        self.assertTrue(mock_ingest.call_args.kwargs["delete_after_ingest"])
        self.assertEqual(mock_ingest.call_args.args[0], f"http://{self.settings.db_host}:{self.settings.db_port}")
        self.assertEqual(mock_ingest.call_args.args[1], self.settings.send_buffer_path)
        self.assertEqual(mock_ingest.call_args.kwargs["logger_override"], self.service.logger)
        self.assertEqual(result["files_found"], 2)
        history_entry = self.history.get("ingest_to_db")
        self.assertIsNotNone(history_entry)
        self.assertEqual(history_entry["files_ingested"], 1)
        self.assertEqual(history_entry["failed_files"], ["BROKEN.csv"])

    @patch("runtime.services.ingest_directory")
    def test_ingest_to_db_returns_none_when_data_dir_missing(self, mock_ingest):
        os.rmdir(self.settings.send_buffer_path)

        result = self.service.ingest_to_db(delete_after_ingest=True)

        self.assertIsNone(result)
        mock_ingest.assert_not_called()
        self.assertIsNone(self.history.get("ingest_to_db"))

    @patch("runtime.services.export_from_gateway")
    @patch("runtime.services.DBClient")
    def test_export_from_db_delegates_to_adapter(self, mock_client_cls, mock_export):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_export.return_value = {
            "output_dir": self.settings.receive_buffer_path,
            "exported": 1,
            "total": 2,
            "failed_symbols": ["SH600001"],
            "partial_symbols": ["SH600000"],
        }
        result = self.service.export_from_db(start_date="2026-01-01")

        mock_client_cls.assert_called_once_with(self.settings.db_host, self.settings.db_port)
        mock_export.assert_called_once()
        call_kwargs = mock_export.call_args.kwargs
        self.assertEqual(call_kwargs["start_date"], "2026-01-01")
        self.assertEqual(call_kwargs["output_dir"], self.settings.receive_buffer_path)
        self.assertEqual(call_kwargs["logger"], self.service.logger)
        self.assertEqual(
            call_kwargs["symbol_fallback_paths"],
            (
                os.path.join(self.settings.data_path, "stock_code_list"),
                os.path.join(self.settings.data_path, "index_code_list"),
                os.path.join(self.settings.qlib_data_path, "instruments", "all.txt"),
            ),
        )
        self.assertTrue(call_kwargs["prefer_local_symbol_fallback"])
        self.assertRegex(call_kwargs["end_date"], r"^\d{4}-\d{2}-\d{2}$")

        self.assertEqual(result["exported"], 1)
        self.assertEqual(result["failed_symbols"], ["SH600001"])
        self.assertEqual(result["partial_symbols"], ["SH600000"])
        history_entry = self.history.get("export_from_db")
        self.assertIsNotNone(history_entry)
        self.assertEqual(history_entry["failed"], 1)
        self.assertEqual(history_entry["partial"], 1)


if __name__ == "__main__":
    unittest.main()
