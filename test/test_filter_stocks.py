"""Tests for scripts.filter.filter_top_liquidity."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts import filter as filter_script


def _mock_response(rows, status_code=200):
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = {"data": rows}
    return r


class TestFilterTopLiquidity(unittest.TestCase):
    def test_uses_data_rows_not_dict_length(self):
        tmp = tempfile.mkdtemp()
        try:
            stock_list_path = os.path.join(tmp, "stock_code_list")
            with open(stock_list_path, "w", encoding="utf-8") as f:
                f.write("SH600000\n")

            rows = [{"close": 10.0, "volume": 5.0} for _ in range(60)]

            with patch("scripts.filter.settings") as mock_settings:
                mock_settings.data_path = tmp
                mock_settings.db_host = "127.0.0.1"
                mock_settings.db_port = 8080

                with patch("scripts.filter.DBClient") as mock_client_cls:
                    mock_client = MagicMock()
                    mock_client.query_data.return_value = _mock_response(rows)
                    mock_client_cls.return_value = mock_client

                    with patch("scripts.filter.utils.io.read_file_lines", return_value=["SH600000"]):
                        filter_script.filter_top_liquidity(start_year=2020, end_year=2020, top_n=500)

            out_csv = os.path.join(tmp, "top_500_liquidity_stocks.csv")
            self.assertTrue(os.path.isfile(out_csv))
        finally:
            for name in os.listdir(tmp):
                os.remove(os.path.join(tmp, name))
            os.rmdir(tmp)

    def test_skips_when_response_none(self):
        tmp = tempfile.mkdtemp()
        try:
            with patch("scripts.filter.settings") as mock_settings:
                mock_settings.data_path = tmp
                mock_settings.db_host = "127.0.0.1"
                mock_settings.db_port = 8080

                with patch("scripts.filter.DBClient") as mock_client_cls:
                    mock_client = MagicMock()
                    mock_client.query_data.return_value = None
                    mock_client_cls.return_value = mock_client

                    with patch("scripts.filter.utils.io.read_file_lines", return_value=["SH600001"]):
                        filter_script.filter_top_liquidity(start_year=2021, end_year=2021, top_n=10)

            out_csv = os.path.join(tmp, "top_500_liquidity_stocks.csv")
            self.assertTrue(os.path.isfile(out_csv))
            content = open(out_csv, encoding="utf-8").read().strip()
            self.assertEqual(content, "")
        finally:
            for name in os.listdir(tmp):
                os.remove(os.path.join(tmp, name))
            os.rmdir(tmp)


if __name__ == "__main__":
    unittest.main()
