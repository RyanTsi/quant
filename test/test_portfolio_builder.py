import unittest

import pandas as pd

from backtesting.portfolio import (
    PortfolioConfig,
    build_rebalance_orders,
    build_target_weights,
    summarize_orders,
)


class TestPortfolioBuilder(unittest.TestCase):
    def test_build_target_weights_basic_constraints(self):
        picks = pd.DataFrame(
            {
                "instrument": [f"S{i:03d}" for i in range(120)],
                "Score": [120 - i for i in range(120)],
            }
        )
        cfg = PortfolioConfig(top_k=80, max_weight=0.03, rebalance_threshold=0.002)
        target = build_target_weights(picks, cfg)

        self.assertEqual(len(target), 80)
        self.assertAlmostEqual(float(target["target_weight"].sum()), 1.0, places=6)
        self.assertTrue((target["target_weight"] <= 1.0).all())

    def test_rebalance_orders_and_summary(self):
        target = pd.DataFrame(
            {
                "instrument": ["A", "B", "C"],
                "target_weight": [0.5, 0.3, 0.2],
                "score": [1.0, 0.9, 0.8],
            }
        )
        current = pd.DataFrame(
            {
                "instrument": ["A", "D"],
                "target_weight": [0.4, 0.6],
            }
        )
        orders = build_rebalance_orders(target, current, threshold=0.05)
        stats = summarize_orders(orders)

        self.assertIn("BUY", set(orders["action"]))
        self.assertIn("SELL", set(orders["action"]))
        self.assertGreater(stats["turnover"], 0.0)


if __name__ == "__main__":
    unittest.main()

