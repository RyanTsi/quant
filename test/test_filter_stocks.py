"""Tests for scripts.filter.filter_top_liquidity."""

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from scripts import filter as filter_script


def _mock_response(rows, status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {"data": rows}
    return response


class TestFilterTopLiquidity(unittest.TestCase):
    def test_target_source_month_pairs_shift_one_month(self):
        months = [
            (filter_script.pd.Timestamp("2026-01-01"), filter_script.pd.Timestamp("2026-01-31")),
            (filter_script.pd.Timestamp("2026-02-01"), filter_script.pd.Timestamp("2026-02-28")),
            (filter_script.pd.Timestamp("2026-03-01"), filter_script.pd.Timestamp("2026-03-31")),
        ]
        pairs = filter_script._target_source_month_pairs(months)

        self.assertEqual(len(pairs), 2)
        self.assertEqual(str(pairs[0][0]), "2026-01")
        self.assertEqual(pairs[0][1].strftime("%Y-%m-%d"), "2026-02-01")
        self.assertEqual(pairs[0][2].strftime("%Y-%m-%d"), "2026-02-28")
        self.assertEqual(str(pairs[1][0]), "2026-02")
        self.assertEqual(pairs[1][1].strftime("%Y-%m-%d"), "2026-03-01")
        self.assertEqual(pairs[1][2].strftime("%Y-%m-%d"), "2026-03-31")

    def test_sampling_uses_entry_exit_buffer_and_is_deterministic(self):
        df_m = filter_script.pd.DataFrame(
            {
                "symbol": [f"S{i:04d}" for i in range(1, 11)],
                "lagged_liquidity": [float(1000 - i) for i in range(10)],
            }
        )

        selected_first = filter_script._sample_symbols_for_month(
            df_m,
            rng=filter_script.random.Random(7),
            top_n=10,
            segment_count=2,
            total_select=4,
            min_per_segment=1,
            previous_symbols={"S0005"},
            source_month=filter_script.pd.Period("2026-01", freq="M"),
            entry_limit=3,
            exit_limit=5,
            seed=11,
        )
        selected_second = filter_script._sample_symbols_for_month(
            df_m,
            rng=filter_script.random.Random(99),
            top_n=10,
            segment_count=2,
            total_select=4,
            min_per_segment=1,
            previous_symbols={"S0005"},
            source_month=filter_script.pd.Period("2026-01", freq="M"),
            entry_limit=3,
            exit_limit=5,
            seed=11,
        )

        self.assertEqual(selected_first, selected_second)
        self.assertEqual(len(selected_first), 4)
        self.assertTrue(set(selected_first).issubset({"S0001", "S0002", "S0003", "S0004", "S0005"}))

    def test_merge_contiguous_ranges_merges_adjacent_windows(self):
        df_ranges = filter_script.pd.DataFrame(
            [
                {"symbol": "SH600000", "start_date": "2026-01-01", "end_date": "2026-03-31"},
                {"symbol": "SH600000", "start_date": "2026-04-01", "end_date": "2026-06-30"},
                {"symbol": "SH600000", "start_date": "2026-10-01", "end_date": "2026-12-31"},
            ]
        )
        merged = filter_script._merge_contiguous_ranges(df_ranges)

        self.assertEqual(len(merged), 2)
        first = merged.iloc[0].to_dict()
        second = merged.iloc[1].to_dict()
        self.assertEqual(first["start_date"], "2026-01-01")
        self.assertEqual(first["end_date"], "2026-06-30")
        self.assertEqual(second["start_date"], "2026-10-01")
        self.assertEqual(second["end_date"], "2026-12-31")

    def test_uses_data_rows_not_dict_length(self):
        tmp = tempfile.mkdtemp()
        try:
            stock_list_path = os.path.join(tmp, "stock_code_list")
            with open(stock_list_path, "w", encoding="utf-8") as handle:
                handle.write("SH600000\n")

            rows = [
                {"date": f"2020-01-{(idx % 28) + 1:02d}", "close": 10.0, "volume": 5.0, "amount": 50.0}
                for idx in range(60)
            ]

            with patch("scripts.filter.settings") as mock_settings:
                mock_settings.data_path = tmp
                mock_settings.qlib_data_path = tmp
                mock_settings.db_host = "127.0.0.1"
                mock_settings.db_port = 8080

                with patch("scripts.filter.DBClient") as mock_client_cls:
                    mock_client = MagicMock()
                    mock_client.query_data.return_value = _mock_response(rows)
                    mock_client_cls.return_value = mock_client

                    with patch("scripts.filter.utils.io.read_file_lines", return_value=["SH600000"]):
                        filter_script.filter_top_liquidity(start_year=2020, end_year=2020, top_n=500)

            out_txt = os.path.join(tmp, "instruments", "my_800_stocks.txt")
            self.assertTrue(os.path.isfile(out_txt))
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)

    def test_skips_when_response_none(self):
        tmp = tempfile.mkdtemp()
        try:
            with patch("scripts.filter.settings") as mock_settings:
                mock_settings.data_path = tmp
                mock_settings.qlib_data_path = tmp
                mock_settings.db_host = "127.0.0.1"
                mock_settings.db_port = 8080

                with patch("scripts.filter.DBClient") as mock_client_cls:
                    mock_client = MagicMock()
                    mock_client.query_data.return_value = None
                    mock_client_cls.return_value = mock_client

                    with patch("scripts.filter.utils.io.read_file_lines", return_value=["SH600001"]):
                        filter_script.filter_top_liquidity(start_year=2021, end_year=2021, top_n=10)

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

            with patch("scripts.filter.settings") as mock_settings:
                mock_settings.data_path = tmp
                mock_settings.qlib_data_path = tmp
                mock_settings.db_host = "127.0.0.1"
                mock_settings.db_port = 8080

                with patch("scripts.filter.DBClient") as mock_client_cls:
                    mock_client = MagicMock()
                    mock_client.query_data.return_value = _mock_response(rows)
                    mock_client_cls.return_value = mock_client

                    with patch("scripts.filter.utils.io.read_file_lines", return_value=["SH600000"]):
                        filter_script.filter_top_liquidity(start_year=2020, end_year=2020, top_n=10, random_seed=1)

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
