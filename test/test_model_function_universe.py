"""Focused tests for model_function universe helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from model_function.universe import (
    PredictionUniverseConfig,
    TrainingUniverseConfig,
    apply_entry_exit_buffer,
    apply_portfolio_hold_buffer,
    build_training_source_month_pairs,
    build_prediction_pool_from_features,
    collect_training_month_liquidity,
    merge_contiguous_symbol_ranges,
)


class TestModelFunctionUniverse(unittest.TestCase):
    def test_build_training_source_month_pairs_shift_one_month(self):
        months = [
            (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-31")),
            (pd.Timestamp("2026-02-01"), pd.Timestamp("2026-02-28")),
            (pd.Timestamp("2026-03-01"), pd.Timestamp("2026-03-31")),
        ]

        pairs = build_training_source_month_pairs(months)

        self.assertEqual(len(pairs), 2)
        self.assertEqual(str(pairs[0][0]), "2026-01")
        self.assertEqual(pairs[0][1].strftime("%Y-%m-%d"), "2026-02-01")
        self.assertEqual(pairs[0][2].strftime("%Y-%m-%d"), "2026-02-28")
        self.assertEqual(str(pairs[1][0]), "2026-02")
        self.assertEqual(pairs[1][1].strftime("%Y-%m-%d"), "2026-03-01")
        self.assertEqual(pairs[1][2].strftime("%Y-%m-%d"), "2026-03-31")

    def test_apply_entry_exit_buffer_retains_previous_symbols_inside_exit_band(self):
        ranked = ["A", "B", "C", "D", "E", "F"]
        selected = apply_entry_exit_buffer(ranked, {"E", "F"}, entry_limit=3, exit_limit=5)

        self.assertEqual(selected, ["A", "B", "C", "E"])

    def test_collect_training_month_liquidity_is_lagged(self):
        rows = []
        for idx in range(25):
            rows.append({"date": f"2020-01-{idx + 1:02d}", "amount": 10.0, "close": 1.0, "volume": 1.0, "isST": 0})
        for idx in range(25):
            rows.append({"date": f"2020-02-{idx + 1:02d}", "amount": 1000.0, "close": 1.0, "volume": 1.0, "isST": 0})

        records = collect_training_month_liquidity(
            symbol="SH600000",
            rows=rows,
            source_months=[pd.Period("2020-01", freq="M")],
            start_year=2020,
            end_year=2020,
            config=TrainingUniverseConfig(liquidity_lookback_days=60, liquidity_min_periods=20),
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["symbol"], "SH600000")
        self.assertEqual(str(records[0]["source_month"]), "2020-01")
        self.assertAlmostEqual(records[0]["lagged_liquidity"], 10.0)

    def test_build_prediction_pool_from_features_uses_buffer_and_filters_st(self):
        instruments = ["A", "B", "C", "D", "E", "F"]
        feats = pd.DataFrame(
            {
                "$amount": [600.0, 500.0, 400.0, 300.0, 200.0, 100.0],
                "$close": [1.0] * 6,
                "$volume": [1.0] * 6,
                "$isst": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            },
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2026-03-31"), instrument) for instrument in instruments],
                names=["datetime", "instrument"],
            ),
        )

        pool = build_prediction_pool_from_features(
            feats,
            previous_holdings=["D", "E", "F"],
            excluded_symbols={"B"},
            config=PredictionUniverseConfig(entry_limit=2, exit_limit=3),
        )

        self.assertEqual(pool, ["A", "C", "D"])

    def test_apply_portfolio_hold_buffer_keeps_existing_positions_inside_hold_band(self):
        picks = pd.DataFrame(
            {
                "instrument": ["A", "B", "C", "D", "E", "F"],
                "Score": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            }
        )

        eligible = apply_portfolio_hold_buffer(
            picks,
            current_instruments=["D", "E"],
            buy_rank=2,
            hold_rank=4,
        )

        self.assertEqual(eligible["instrument"].tolist(), ["A", "B", "D"])

    def test_merge_contiguous_symbol_ranges_merges_adjacent_windows(self):
        df_ranges = pd.DataFrame(
            [
                {"symbol": "SH600000", "start_date": "2026-01-01", "end_date": "2026-03-31"},
                {"symbol": "SH600000", "start_date": "2026-04-01", "end_date": "2026-06-30"},
                {"symbol": "SH600000", "start_date": "2026-10-01", "end_date": "2026-12-31"},
            ]
        )

        merged = merge_contiguous_symbol_ranges(df_ranges)

        self.assertEqual(len(merged), 2)
        first = merged.iloc[0].to_dict()
        second = merged.iloc[1].to_dict()
        self.assertEqual(first["start_date"], "2026-01-01")
        self.assertEqual(first["end_date"], "2026-06-30")
        self.assertEqual(second["start_date"], "2026-10-01")
        self.assertEqual(second["end_date"], "2026-12-31")


if __name__ == "__main__":
    unittest.main()
