"""Focused contract tests for runtime model adapter outputs."""

from __future__ import annotations

import os
import tempfile
import unittest

import pandas as pd

from runtime.adapters.modeling import build_portfolio_outputs


class TestModelingAdapterContract(unittest.TestCase):
    def test_build_portfolio_outputs_writes_expected_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            date = "2026-03-31"
            picks_path = os.path.join(tmpdir, f"top_picks_{date}.csv")
            picks_df = pd.DataFrame(
                {
                    "instrument": ["SH600000", "SH600001", "SH600002"],
                    "Score": [0.9, 0.6, 0.3],
                }
            )
            picks_df.to_csv(picks_path, index=False)

            result = build_portfolio_outputs(
                date=date,
                output_dir=tmpdir,
                top_k=3,
                max_weight=0.8,
                rebalance_threshold=0.001,
                track_run=False,
            )

            self.assertEqual(result["date"], date)
            self.assertTrue(os.path.isfile(result["target_path"]))
            self.assertTrue(os.path.isfile(result["orders_path"]))
            self.assertIn("stats", result)

            target_df = pd.read_csv(result["target_path"])
            self.assertIn("instrument", target_df.columns)
            self.assertIn("target_weight", target_df.columns)
            self.assertIn("score", target_df.columns)

            orders_df = pd.read_csv(result["orders_path"])
            self.assertIn("instrument", orders_df.columns)
            self.assertIn("action", orders_df.columns)

            stats = result["stats"]
            self.assertIn("buy_count", stats)
            self.assertIn("sell_count", stats)
            self.assertIn("hold_count", stats)
            self.assertIn("turnover", stats)

    def test_build_portfolio_outputs_raises_if_prediction_file_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                build_portfolio_outputs(
                    date="2026-03-31",
                    output_dir=tmpdir,
                    track_run=False,
                )

    def test_build_portfolio_outputs_applies_hold_band_before_capacity_cap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            date = "2026-03-31"
            picks_path = os.path.join(tmpdir, f"top_picks_{date}.csv")
            pd.DataFrame(
                {
                    "instrument": ["A", "B", "C", "D", "E", "F"],
                    "Score": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                }
            ).to_csv(picks_path, index=False)

            prev_target_path = os.path.join(tmpdir, "target_weights_2026-03-30.csv")
            pd.DataFrame(
                {
                    "instrument": ["D", "E"],
                    "target_weight": [0.6, 0.4],
                }
            ).to_csv(prev_target_path, index=False)

            result = build_portfolio_outputs(
                date=date,
                output_dir=tmpdir,
                top_k=3,
                buy_rank=2,
                hold_rank=4,
                max_weight=0.8,
                rebalance_threshold=0.001,
                track_run=False,
            )

            target_df = pd.read_csv(result["target_path"])
            self.assertEqual(target_df["instrument"].tolist(), ["A", "B", "D"])


if __name__ == "__main__":
    unittest.main()
