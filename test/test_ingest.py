"""Tests for data_pipeline.ingest."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from data_pipeline import ingest


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
            ingest.ingest_directory("http://127.0.0.1:8080", "/nonexistent_dir_ingest_xyz")
        self.assertTrue(any("Directory not found" in m for m in cm.output))

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

            with patch("data_pipeline.ingest.requests.post", return_value=mock_resp) as mock_post:
                ingest.ingest_directory("http://127.0.0.1:8080", tmp)

            self.assertTrue(mock_post.called)
            args, kwargs = mock_post.call_args
            self.assertIn("/api/v1/ingest/daily", args[0])
            batch = kwargs["json"]
            self.assertEqual(len(batch), 1)
            self.assertEqual(batch[0]["symbol"], "SH600000")
            self.assertEqual(batch[0]["close"], 10.5)
        finally:
            os.remove(csv_path)
            os.rmdir(tmp)

    def test_skips_short_file(self):
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, "SHORT.csv")
            with open(path, "w", encoding="utf-8") as f:
                f.write("date,open\n")

            with patch("data_pipeline.ingest.requests.post") as mock_post:
                ingest.ingest_directory("http://127.0.0.1:8080", tmp)

            mock_post.assert_not_called()
        finally:
            os.remove(path)
            os.rmdir(tmp)


if __name__ == "__main__":
    unittest.main()
