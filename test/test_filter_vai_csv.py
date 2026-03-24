"""Tests for scripts.filter_vai_csv.load_stock."""

import os
import tempfile
import unittest

import pandas as pd

import scripts.filter_vai_csv as filter_vai


class TestLoadStock(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        for name in os.listdir(self.tmp):
            os.remove(os.path.join(self.tmp, name))
        os.rmdir(self.tmp)

    def test_load_stock_aggregates_by_year(self):
        filter_vai._data_path = self.tmp
        path = os.path.join(self.tmp, "SH600000.csv")
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10, freq="D"),
                "amount": [1e6] * 10,
                "tradestatus": [1] * 10,
            }
        )
        df.to_csv(path, index=False)

        yearly = filter_vai.load_stock("SH600000.csv")
        self.assertFalse(yearly.empty)
        self.assertIn("symbol", yearly.columns)
        self.assertIn("year", yearly.columns)
        self.assertIn("turnover", yearly.columns)
        self.assertTrue((yearly["symbol"] == "SH600000").all())

    def test_filters_zero_tradestatus(self):
        filter_vai._data_path = self.tmp
        path = os.path.join(self.tmp, "SZ000001.csv")
        df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=10, freq="D"),
                "amount": [100.0] * 10,
                "tradestatus": [0] * 10,
            }
        )
        df.to_csv(path, index=False)

        yearly = filter_vai.load_stock("SZ000001.csv")
        self.assertTrue(yearly.empty)


if __name__ == "__main__":
    unittest.main()
