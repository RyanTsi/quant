"""Tests for the training-universe build path behind scripts.filter."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from runtime.adapters import modeling


def _mock_response(rows, status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {"data": rows}
    return response


class TestBuildTrainingUniverseFile(unittest.TestCase):
    def test_uses_data_rows_not_dict_length(self):
        tmp = tempfile.mkdtemp()
        try:
            rows = [
                {"date": f"2020-01-{(idx % 28) + 1:02d}", "close": 10.0, "volume": 5.0, "amount": 50.0}
                for idx in range(60)
            ]

            with patch("runtime.adapters.modeling.DBClient") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.query_data.return_value = _mock_response(rows)
                mock_client_cls.return_value = mock_client

                with patch("runtime.adapters.modeling.io_utils.read_file_lines", return_value=["SH600000"]):
                    result = modeling.build_training_universe_file(
                        start_year=2020,
                        end_year=2020,
                        top_n=500,
                        data_path=tmp,
                        qlib_dir=tmp,
                        db_host="127.0.0.1",
                        db_port=8080,
                    )

            out_txt = os.path.join(tmp, "instruments", "my_800_stocks.txt")
            self.assertTrue(os.path.isfile(out_txt))
            self.assertEqual(result["output_path"], out_txt)
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_skips_when_response_none(self):
        tmp = tempfile.mkdtemp()
        try:
            with patch("runtime.adapters.modeling.DBClient") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.query_data.return_value = None
                mock_client_cls.return_value = mock_client

                with patch("runtime.adapters.modeling.io_utils.read_file_lines", return_value=["SH600001"]):
                    modeling.build_training_universe_file(
                        start_year=2021,
                        end_year=2021,
                        top_n=10,
                        data_path=tmp,
                        qlib_dir=tmp,
                        db_host="127.0.0.1",
                        db_port=8080,
                    )

            out_txt = os.path.join(tmp, "instruments", "my_800_stocks.txt")
            self.assertTrue(os.path.isfile(out_txt))
            with open(out_txt, encoding="utf-8") as handle:
                content = handle.read().strip()
            self.assertEqual(content, "")
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_st_months_excluded_from_selection(self):
        tmp = tempfile.mkdtemp()
        try:
            rows = []
            for idx in range(50):
                rows.append(
                    {
                        "date": f"2020-01-{(idx % 28) + 1:02d}",
                        "amount": 10000.0,
                        "close": 10.0,
                        "volume": 1.0,
                        "isST": 0,
                    }
                )
            for idx in range(25):
                rows.append(
                    {
                        "date": f"2020-02-{(idx % 28) + 1:02d}",
                        "amount": 20000.0,
                        "close": 10.0,
                        "volume": 1.0,
                        "isST": 1,
                    }
                )

            with patch("runtime.adapters.modeling.DBClient") as mock_client_cls:
                mock_client = MagicMock()
                mock_client.query_data.return_value = _mock_response(rows)
                mock_client_cls.return_value = mock_client

                with patch("runtime.adapters.modeling.io_utils.read_file_lines", return_value=["SH600000"]):
                    modeling.build_training_universe_file(
                        start_year=2020,
                        end_year=2020,
                        top_n=10,
                        random_seed=1,
                        data_path=tmp,
                        qlib_dir=tmp,
                        db_host="127.0.0.1",
                        db_port=8080,
                    )

            out_txt = os.path.join(tmp, "instruments", "my_800_stocks.txt")
            self.assertTrue(os.path.isfile(out_txt))
            with open(out_txt, encoding="utf-8") as handle:
                content = handle.read()
            self.assertIn("2020-02-01", content)
            self.assertNotIn("2020-03-01", content)
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
