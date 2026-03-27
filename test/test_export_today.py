"""Tests for scripts.export_today."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from scripts import export_today


class TestFetchAllByDate(unittest.TestCase):
    def test_single_page_returns_dataframe(self):
        body = {
            "data": [
                {"symbol": "SH600000", "date": "2026-03-20", "open": 10.0, "close": 10.5, "volume": 100},
            ],
        }

        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = body
        mock_session.get.return_value = mock_resp

        with patch("scripts.export_today.requests.Session", return_value=mock_session):
            client = MagicMock()
            client.base_url = "http://127.0.0.1:8080/api/v1"
            df = export_today.fetch_all_by_date(client, "2026-03-20", page_size=5000)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        mock_session.get.assert_called()

    def test_empty_data_returns_none(self):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": []}
        mock_session.get.return_value = mock_resp

        with patch("scripts.export_today.requests.Session", return_value=mock_session):
            client = MagicMock()
            client.base_url = "http://127.0.0.1:8080/api/v1"
            df = export_today.fetch_all_by_date(client, "2026-03-20")

        self.assertIsNone(df)

    def test_non_200_returns_none(self):
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "error"
        mock_session.get.return_value = mock_resp

        with patch("scripts.export_today.requests.Session", return_value=mock_session):
            client = MagicMock()
            client.base_url = "http://127.0.0.1:8080/api/v1"
            df = export_today.fetch_all_by_date(client, "2026-03-20")

        self.assertIsNone(df)


class TestExportDateToCsv(unittest.TestCase):
    def test_unreachable_returns_none(self):
        with patch("scripts.export_today.DBClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.health.return_value = {"status": "unreachable"}
            mock_cls.return_value = mock_client

            result = export_today.export_date_to_csv("2026-03-20")

        self.assertIsNone(result)

    def test_writes_csv_when_data_ok(self):
        tmp = tempfile.mkdtemp()
        try:
            df = pd.DataFrame(
                [
                    {
                        "symbol": "SH600000",
                        "date": "2026-03-20",
                        "open": 1.0,
                        "high": 2.0,
                        "low": 0.5,
                        "close": 1.5,
                        "volume": 100.0,
                    },
                ]
            )

            with patch("scripts.export_today.DBClient") as mock_cls:
                mock_client = MagicMock()
                mock_client.health.return_value = {"status": "healthy"}
                mock_cls.return_value = mock_client

                with patch("scripts.export_today.fetch_all_by_date", return_value=df):
                    out = export_today.export_date_to_csv("2026-03-20", output_dir=tmp)

            self.assertIsNotNone(out)
            self.assertTrue(os.path.isfile(out))
            self.assertIn("market_2026-03-20.csv", out)
        finally:
            for name in os.listdir(tmp):
                os.remove(os.path.join(tmp, name))
            os.rmdir(tmp)


if __name__ == "__main__":
    unittest.main()
