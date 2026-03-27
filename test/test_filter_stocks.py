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
    def test_target_source_quarter_pairs_shift_one_quarter(self):
        quarters = [
            (filter_script.pd.Timestamp("2026-01-01"), filter_script.pd.Timestamp("2026-03-31")),
            (filter_script.pd.Timestamp("2026-04-01"), filter_script.pd.Timestamp("2026-06-30")),
            (filter_script.pd.Timestamp("2026-07-01"), filter_script.pd.Timestamp("2026-09-30")),
        ]
        pairs = filter_script._target_source_quarter_pairs(quarters)

        self.assertEqual(len(pairs), 2)
        self.assertEqual(str(pairs[0][0]), "2026Q1")
        self.assertEqual(pairs[0][1].strftime("%Y-%m-%d"), "2026-04-01")
        self.assertEqual(pairs[0][2].strftime("%Y-%m-%d"), "2026-06-30")
        self.assertEqual(str(pairs[1][0]), "2026Q2")
        self.assertEqual(pairs[1][1].strftime("%Y-%m-%d"), "2026-07-01")
        self.assertEqual(pairs[1][2].strftime("%Y-%m-%d"), "2026-09-30")

    def test_sampling_drops_heads_tails_then_random_from_middle(self):
        df_q = filter_script.pd.DataFrame(
            [{"symbol": f"S{i:03d}", "turnover": float(1000 - i)} for i in range(200)]
        )
        rng = filter_script.random.Random(7)
        selected = filter_script._sample_symbols_for_quarter(
            df_q, rng=rng, top_n=200, segment_count=1
        )

        self.assertEqual(len(selected), 80)
        forbidden = {f"S{i:03d}" for i in list(range(20)) + list(range(180, 200))}
        self.assertTrue(set(selected).isdisjoint(forbidden))

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

            out_txt = os.path.join(tmp, "my_800_stocks.txt")
            self.assertTrue(os.path.isfile(out_txt))
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

            out_txt = os.path.join(tmp, "my_800_stocks.txt")
            self.assertTrue(os.path.isfile(out_txt))
            content = open(out_txt, encoding="utf-8").read().strip()
            self.assertEqual(content, "")
        finally:
            for name in os.listdir(tmp):
                os.remove(os.path.join(tmp, name))
            os.rmdir(tmp)


if __name__ == "__main__":
    unittest.main()
