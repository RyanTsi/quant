"""Contract tests for runtime.adapters.exporting."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock

import pandas as pd

from runtime.adapters import exporting


def _mock_response(payload, status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    return response


class TestNormalizeGatewayRows(unittest.TestCase):
    def test_normalize_gateway_rows_applies_export_schema(self):
        rows = [
            {
                "symbol": "SH600000",
                "date": "2026-03-31T00:00:00",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 1000,
                "amount": 1_000_000,
                "turn": 0.01,
                "tradestatus": 1,
                "is_st": 1,
            },
            {
                "symbol": "SH600000",
                "date": "2026-03-30 00:00:00",
                "open": 9.8,
                "high": 10.3,
                "low": 9.2,
                "close": 9.5,
                "volume": 800,
                "amount": 800_000,
                "turn": 0.008,
                "tradestatus": 0,
                "is_st": 0,
            },
        ]

        normalized = exporting.normalize_gateway_rows(rows)

        self.assertEqual(list(normalized.columns), exporting.DEFAULT_EXPORT_CSV_COLUMNS)
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized.iloc[0]["date"], "2026-03-31")
        self.assertEqual(normalized.iloc[0]["isST"], 1)
        self.assertEqual(normalized.iloc[0]["factor"], 1.0)


class TestExportFromGateway(unittest.TestCase):
    def test_export_from_gateway_raises_when_health_check_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = MagicMock()
            client.health.return_value = {"status": "unhealthy"}

            with self.assertRaisesRegex(RuntimeError, "DB unreachable"):
                exporting.export_from_gateway(
                    client,
                    start_date="2026-01-01",
                    end_date="2026-04-01",
                    output_dir=tmpdir,
                )

    def test_export_from_gateway_raises_when_list_symbols_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = MagicMock()
            client.health.return_value = {"status": "healthy"}
            client.list_symbols.return_value = _mock_response({"symbols": []}, status_code=503)

            with self.assertRaisesRegex(RuntimeError, "Failed to list symbols"):
                exporting.export_from_gateway(
                    client,
                    start_date="2026-01-01",
                    end_date="2026-04-01",
                    output_dir=tmpdir,
                )

    def test_export_from_gateway_writes_normalized_csv_and_tracks_failed_symbols(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = MagicMock()
            client.health.return_value = {"status": "healthy"}
            client.list_symbols.return_value = _mock_response({"symbols": ["SH600000", "SH600001"]})
            client.query_data.side_effect = [
                _mock_response(
                    {
                        "data": [
                            {
                                "symbol": "SH600000",
                                "date": "2026-03-31T00:00:00",
                                "open": 10.0,
                                "high": 11.0,
                                "low": 9.0,
                                "close": 10.5,
                                "volume": 1000,
                                "amount": 1_000_000,
                                "turn": 0.01,
                                "tradestatus": 1,
                                "is_st": 0,
                            },
                            {
                                "symbol": "SH600000",
                                "date": "2026-03-30T00:00:00",
                                "open": 9.8,
                                "high": 10.3,
                                "low": 9.2,
                                "close": 9.5,
                                "volume": 800,
                                "amount": 800_000,
                                "turn": 0.008,
                                "tradestatus": 0,
                                "is_st": 0,
                            },
                        ]
                    }
                ),
                _mock_response({"data": []}),
                _mock_response({"data": []}, status_code=503),
            ]
            logger = MagicMock()

            result = exporting.export_from_gateway(
                client,
                start_date="2026-01-01",
                end_date="2026-04-01",
                output_dir=tmpdir,
                logger=logger,
                page_size=2,
            )

            self.assertEqual(result["exported"], 1)
            self.assertEqual(result["total"], 2)
            self.assertEqual(result["failed_symbols"], ["SH600001"])
            self.assertEqual(result["partial_symbols"], [])

            first_symbol_path = os.path.join(tmpdir, "SH600000.csv")
            self.assertTrue(os.path.isfile(first_symbol_path))
            self.assertFalse(os.path.exists(os.path.join(tmpdir, "SH600001.csv")))

            exported_df = pd.read_csv(first_symbol_path)
            self.assertEqual(list(exported_df.columns), exporting.DEFAULT_EXPORT_CSV_COLUMNS)
            self.assertEqual(len(exported_df), 1)
            self.assertEqual(exported_df.iloc[0]["date"], "2026-03-31")
            self.assertEqual(exported_df.iloc[0]["isST"], 0)
            self.assertEqual(exported_df.iloc[0]["factor"], 1.0)
            self.assertGreaterEqual(logger.warning.call_count, 1)

    def test_export_symbol_csvs_marks_partial_symbols_when_later_page_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            client = MagicMock()
            client.query_data.side_effect = [
                _mock_response(
                    {
                        "data": [
                            {
                                "symbol": "SH600000",
                                "date": "2026-03-31",
                                "open": 10.0,
                                "high": 11.0,
                                "low": 9.0,
                                "close": 10.5,
                                "volume": 1000,
                                "amount": 1_000_000,
                                "turn": 0.01,
                                "tradestatus": 1,
                                "is_st": 0,
                            },
                            {
                                "symbol": "SH600000",
                                "date": "2026-03-30",
                                "open": 9.8,
                                "high": 10.3,
                                "low": 9.2,
                                "close": 9.5,
                                "volume": 800,
                                "amount": 800_000,
                                "turn": 0.008,
                                "tradestatus": 1,
                                "is_st": 0,
                            },
                        ]
                    }
                ),
                _mock_response({"data": []}, status_code=500),
            ]
            logger = MagicMock()

            result = exporting.export_symbol_csvs(
                client,
                symbols=["SH600000"],
                start_date="2026-01-01",
                end_date="2026-04-01",
                output_dir=tmpdir,
                logger=logger,
                page_size=2,
            )

            self.assertEqual(result["exported"], 1)
            self.assertEqual(result["failed_symbols"], [])
            self.assertEqual(result["partial_symbols"], ["SH600000"])

            out_path = os.path.join(tmpdir, "SH600000.csv")
            exported_df = pd.read_csv(out_path)
            self.assertEqual(len(exported_df), 2)
            self.assertGreaterEqual(logger.warning.call_count, 1)


if __name__ == "__main__":
    unittest.main()
