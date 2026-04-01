"""Tests for quantcore.services.data_service."""

import os
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import MagicMock, patch

from quantcore.history import RunHistoryStore
from quantcore.services.data_service import DataPipelineService
from quantcore.settings import load_settings


def _mock_response(payload, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    return resp


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
        self.history = RunHistoryStore(os.path.join(self.tmp, "run_history.json"))
        self.service = DataPipelineService(self.settings, history=self.history)

    def tearDown(self):
        for root, dirs, files in os.walk(self.tmp, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.tmp)

    @patch("quantcore.services.data_service.package_data")
    @patch("data_pipeline.fetcher.StockDataFetcher")
    def test_fetch_data_calls_fetcher_and_records_run(self, mock_fetcher_cls, mock_package_data):
        fetcher = MagicMock()
        mock_fetcher_cls.return_value = fetcher

        result = self.service.fetch_data(lookback_days=1)

        fetcher.fetch_all_stock_history.assert_called_once()
        fetcher.fetch_all_index_history.assert_called_once()
        mock_package_data.assert_called_once()
        self.assertIn("start_date", result)
        self.assertIn("end_date", result)
        self.assertIsNotNone(self.history.get("fetch_stock"))

    @patch("quantcore.services.data_service.ingest_directory")
    def test_ingest_to_db_passes_delete_flag(self, mock_ingest):
        self.service.ingest_to_db(delete_after_ingest=True)
        mock_ingest.assert_called_once()
        self.assertTrue(mock_ingest.call_args.kwargs["delete_after_ingest"])

    @patch("quantcore.services.data_service.DBClient")
    def test_export_from_db_writes_symbol_csv(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.health.return_value = {"status": "healthy"}
        mock_client.list_symbols.return_value = _mock_response({"symbols": ["SH600000"]})
        mock_client.query_data.return_value = _mock_response(
            {
                "data": [
                    {
                        "symbol": "SH600000",
                        "date": "2026-03-31",
                        "open": 10,
                        "high": 11,
                        "low": 9,
                        "close": 10.5,
                        "volume": 1000,
                        "amount": 1e6,
                        "turn": 0.01,
                        "tradestatus": 1,
                        "is_st": 0,
                    }
                ]
            }
        )
        mock_client_cls.return_value = mock_client

        result = self.service.export_from_db(start_date="2026-01-01")

        out_file = os.path.join(self.settings.receive_buffer_path, "SH600000.csv")
        self.assertTrue(os.path.isfile(out_file))
        self.assertEqual(result["exported"], 1)
        self.assertIsNotNone(self.history.get("export_from_db"))


if __name__ == "__main__":
    unittest.main()
