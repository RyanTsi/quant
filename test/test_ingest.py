"""Tests for runtime.adapters.ingest."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import runtime.adapters.ingest as ingest


class TestIngestHelpers(unittest.TestCase):
    def test_safe_float_valid(self):
        self.assertEqual(ingest._safe_float("1.5"), 1.5)
        self.assertEqual(ingest._safe_float("  2.0  "), 2.0)

    def test_safe_float_empty(self):
        self.assertEqual(ingest._safe_float("", default=99.0), 99.0)
        self.assertEqual(ingest._safe_float("   "), 0.0)

    def test_safe_float_invalid(self):
        self.assertEqual(ingest._safe_float("not_a_number"), 0.0)

    def test_safe_int_valid(self):
        self.assertEqual(ingest._safe_int("42"), 42)
        self.assertEqual(ingest._safe_int("3.7"), 3)

    def test_safe_int_invalid(self):
        self.assertEqual(ingest._safe_int("x", default=-1), -1)


class TestIngestDirectory(unittest.TestCase):
    def test_missing_dir_returns_early(self):
        with self.assertLogs(ingest.logger, level="WARNING") as cm:
            result = ingest.ingest_directory("http://127.0.0.1:8080", "/nonexistent_dir_ingest_xyz")
        self.assertTrue(any("Directory not found" in m for m in cm.output))
        self.assertEqual(result["files_found"], 0)
        self.assertEqual(result["files_ingested"], 0)
        self.assertEqual(result["failed_files"], [])

    def test_builds_payload_and_posts(self):
        tmp = tempfile.mkdtemp()
        try:
            csv_path = os.path.join(tmp, "SH600000.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(
                    "date,symbol,open,high,low,close,volume,amount,turn,tradestatus,isST\n"
                    "2026-03-01,SH600000,10,11,9,10.5,1000,1e6,0.01,1,0\n"
                )

            mock_resp = MagicMock()
            mock_resp.status_code = 200

            with patch("runtime.adapters.ingest.requests.post", return_value=mock_resp) as mock_post:
                result = ingest.ingest_directory("http://127.0.0.1:8080", tmp)

            self.assertTrue(mock_post.called)
            args, kwargs = mock_post.call_args
            self.assertIn("/api/v1/ingest/daily", args[0])
            batch = kwargs["json"]
            self.assertEqual(len(batch), 1)
            self.assertEqual(batch[0]["symbol"], "SH600000")
            self.assertEqual(batch[0]["close"], 10.5)
            self.assertEqual(result["files_found"], 1)
            self.assertEqual(result["files_ingested"], 1)
            self.assertEqual(result["rows_sent"], 1)
            self.assertEqual(result["failed_batches"], [])
            self.assertEqual(result["deleted_files"], [])
        finally:
            os.remove(csv_path)
            os.rmdir(tmp)

    def test_skips_short_file(self):
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, "SHORT.csv")
            with open(path, "w", encoding="utf-8") as f:
                f.write("date,open\n")

            with patch("runtime.adapters.ingest.requests.post") as mock_post:
                result = ingest.ingest_directory("http://127.0.0.1:8080", tmp)

            mock_post.assert_not_called()
            self.assertEqual(result["files_found"], 1)
            self.assertEqual(result["files_ingested"], 0)
            self.assertEqual(result["skipped_files"], ["SHORT.csv"])
            self.assertEqual(result["rows_sent"], 0)
        finally:
            os.remove(path)
            os.rmdir(tmp)

    def test_delete_after_ingest_removes_file(self):
        tmp = tempfile.mkdtemp()
        try:
            csv_path = os.path.join(tmp, "SH600000.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(
                    "date,symbol,open,high,low,close,volume,amount,turn,tradestatus,isST\n"
                    "2026-03-01,SH600000,10,11,9,10.5,1000,1e6,0.01,1,0\n"
                )

            mock_resp = MagicMock()
            mock_resp.status_code = 200
            with patch("runtime.adapters.ingest.requests.post", return_value=mock_resp):
                result = ingest.ingest_directory(
                    "http://127.0.0.1:8080",
                    tmp,
                    delete_after_ingest=True,
                )

            self.assertFalse(os.path.exists(csv_path))
            self.assertEqual(result["deleted_files"], ["SH600000.csv"])
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)
            os.rmdir(tmp)

    def test_failed_batch_is_reported(self):
        tmp = tempfile.mkdtemp()
        try:
            csv_path = os.path.join(tmp, "SH600000.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(
                    "date,symbol,open,high,low,close,volume,amount,turn,tradestatus,isST\n"
                    "2026-03-01,SH600000,10,11,9,10.5,1000,1e6,0.01,1,0\n"
                )

            mock_resp = MagicMock()
            mock_resp.status_code = 500
            mock_resp.text = "boom"
            with patch("runtime.adapters.ingest.requests.post", return_value=mock_resp):
                result = ingest.ingest_directory("http://127.0.0.1:8080", tmp)

            self.assertEqual(result["files_ingested"], 0)
            self.assertEqual(result["failed_files"], ["SH600000.csv"])
            self.assertEqual(len(result["failed_batches"]), 1)
            self.assertEqual(result["rows_sent"], 0)
            failed = result["failed_batches"][0]
            self.assertEqual(failed["symbol"], "SH600000")
            self.assertEqual(failed["batch"], 1)
            self.assertEqual(failed["status_code"], 500)
        finally:
            os.remove(csv_path)
            os.rmdir(tmp)

    def test_failed_batch_still_deletes_file_when_flag_is_enabled(self):
        tmp = tempfile.mkdtemp()
        try:
            csv_path = os.path.join(tmp, "SH600000.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(
                    "date,symbol,open,high,low,close,volume,amount,turn,tradestatus,isST\n"
                    "2026-03-01,SH600000,10,11,9,10.5,1000,1e6,0.01,1,0\n"
                )

            mock_resp = MagicMock()
            mock_resp.status_code = 500
            mock_resp.text = "boom"
            with patch("runtime.adapters.ingest.requests.post", return_value=mock_resp):
                result = ingest.ingest_directory(
                    "http://127.0.0.1:8080",
                    tmp,
                    delete_after_ingest=True,
                )

            self.assertFalse(os.path.exists(csv_path))
            self.assertEqual(result["failed_files"], ["SH600000.csv"])
            self.assertEqual(result["deleted_files"], ["SH600000.csv"])
            self.assertEqual(result["rows_sent"], 0)
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)
            os.rmdir(tmp)

    def test_rows_sent_tracks_only_successful_batches_in_mixed_outcome(self):
        tmp = tempfile.mkdtemp()
        try:
            csv_path = os.path.join(tmp, "SH600000.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("date,symbol,open,high,low,close,volume,amount,turn,tradestatus,isST\n")
                f.write("2026-03-01,SH600000,10,11,9,10.5,1000,1e6,0.01,1,0\n")
                f.write("2026-03-02,SH600000,10,11,9,10.5,1000,1e6,0.01,1,0\n")
                f.write("2026-03-03,SH600000,10,11,9,10.5,1000,1e6,0.01,1,0\n")

            ok_resp = MagicMock()
            ok_resp.status_code = 200
            fail_resp = MagicMock()
            fail_resp.status_code = 500
            fail_resp.text = "boom"

            with patch("runtime.adapters.ingest.requests.post", side_effect=[ok_resp, fail_resp]):
                result = ingest.ingest_directory(
                    "http://127.0.0.1:8080",
                    tmp,
                    batch_size=2,
                )

            self.assertEqual(result["files_found"], 1)
            self.assertEqual(result["files_ingested"], 0)
            self.assertEqual(result["failed_files"], ["SH600000.csv"])
            self.assertEqual(result["rows_sent"], 2)
            self.assertEqual(len(result["failed_batches"]), 1)
            failed = result["failed_batches"][0]
            self.assertEqual(failed["batch"], 2)
            self.assertEqual(failed["status_code"], 500)
        finally:
            os.remove(csv_path)
            os.rmdir(tmp)


if __name__ == "__main__":
    unittest.main()
