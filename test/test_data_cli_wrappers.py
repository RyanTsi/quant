"""CLI wrapper compatibility tests for data-side scripts."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

from scripts import put_data, update_data


class TestPutDataCliWrapper(unittest.TestCase):
    @patch("scripts.put_data.build_data_service")
    def test_put_data_cli_forwards_args(self, mock_build_service):
        service = MagicMock()
        service.settings.db_host = "127.0.0.1"
        service.settings.db_port = 8080
        service.settings.send_buffer_path = "/tmp/send_buffer"
        service.ingest_to_db.return_value = {
            "data_dir": "/tmp/custom",
            "server_url": "http://127.0.0.1:8080",
            "files_found": 2,
            "files_ingested": 2,
            "skipped_files": [],
            "rows_sent": 12,
            "failed_files": [],
            "failed_batches": [],
            "deleted_files": ["A.csv", "B.csv"],
        }
        mock_build_service.return_value = service

        buf = io.StringIO()
        with patch("sys.argv", ["put_data.py", "--data_dir", "/tmp/custom", "--delete_after_ingest"]):
            with redirect_stdout(buf):
                put_data.main()

        mock_build_service.assert_called_once_with(refresh_settings=True)
        service.ingest_to_db.assert_called_once_with(
            data_dir="/tmp/custom",
            delete_after_ingest=True,
        )
        output = buf.getvalue()
        self.assertIn("Server:", output)
        self.assertIn("Data:", output)

    @patch("scripts.put_data.build_data_service")
    def test_put_data_cli_handles_missing_directory_result(self, mock_build_service):
        service = MagicMock()
        service.settings.db_host = "127.0.0.1"
        service.settings.db_port = 8080
        service.settings.send_buffer_path = "/tmp/missing"
        service.ingest_to_db.return_value = None
        mock_build_service.return_value = service

        buf = io.StringIO()
        with patch("sys.argv", ["put_data.py"]):
            with redirect_stdout(buf):
                put_data.main()

        service.ingest_to_db.assert_called_once_with(
            data_dir=None,
            delete_after_ingest=False,
        )
        output = buf.getvalue()
        self.assertIn("Server: http://127.0.0.1:8080", output)
        self.assertIn("Data:   /tmp/missing", output)


class TestUpdateDataCliWrapper(unittest.TestCase):
    @patch("scripts.update_data.build_data_service")
    def test_update_data_cli_forwards_args_and_prints_summary(self, mock_build_service):
        service = MagicMock()
        service.fetch_data.return_value = {
            "start_date": "20260324",
            "end_date": "20260401",
            "last_end_date": "20260331",
            "lookback_days": 7,
            "save_dir": "/tmp/data/20260324-20260401",
            "send_buffer_dir": "/tmp/data/send_buffer",
        }
        mock_build_service.return_value = service

        buf = io.StringIO()
        with patch("sys.argv", ["update_data.py"]):
            with redirect_stdout(buf):
                update_data.main()

        mock_build_service.assert_called_once_with(refresh_settings=True)
        service.fetch_data.assert_called_once_with(lookback_days=7)
        output = buf.getvalue()
        self.assertIn("Fetched 20260324 -> 20260401", output)
        self.assertIn("Packed data directory: /tmp/data/send_buffer", output)


if __name__ == "__main__":
    unittest.main()
