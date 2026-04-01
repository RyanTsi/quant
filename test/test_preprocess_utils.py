"""Tests for shared preprocessing helpers."""

import random
import unittest

import pandas as pd

from utils.preprocess import compute_liquidity, exclude_symbols, sample_ranked_symbols


class TestPreprocessUtils(unittest.TestCase):
    def test_compute_liquidity_prefers_amount(self):
        df = pd.DataFrame(
            {
                "amount": [100.0, None, 300.0],
                "close": [10.0, 20.0, 30.0],
                "volume": [1.0, 1.0, 1.0],
            }
        )
        liq = compute_liquidity(df, amount_col="amount", close_col="close", volume_col="volume")
        self.assertEqual(liq.tolist(), [100.0, 0.0, 300.0])

    def test_compute_liquidity_falls_back_to_close_volume(self):
        df = pd.DataFrame({"close": [10.0, None, 3.0], "volume": [2.0, 4.0, None]})
        liq = compute_liquidity(df, amount_col="amount", close_col="close", volume_col="volume")
        self.assertEqual(liq.tolist(), [20.0, 0.0, 0.0])

    def test_sample_ranked_symbols_trims_edges(self):
        ranked = [f"S{i:03d}" for i in range(200)]
        selected = sample_ranked_symbols(
            ranked,
            segment_count=1,
            segment_size=200,
            sample_per_segment=80,
            trim_count=20,
            min_segment_size=41,
            rng=random.Random(7),
        )
        self.assertEqual(len(selected), 80)
        forbidden = {f"S{i:03d}" for i in list(range(20)) + list(range(180, 200))}
        self.assertTrue(set(selected).isdisjoint(forbidden))

    def test_exclude_symbols(self):
        symbols = ["SH000001", "SH600000", "SZ000001"]
        excluded = {"SH000001", "SZ000001"}
        self.assertEqual(exclude_symbols(symbols, excluded), ["SH600000"])


if __name__ == "__main__":
    unittest.main()
