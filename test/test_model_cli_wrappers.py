"""CLI wrapper compatibility tests for model-side scripts."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from scripts import build_portfolio, predict


class TestPredictCliWrapper(unittest.TestCase):
    @patch("scripts.predict.generate_predictions")
    def test_predict_cli_forwards_args(self, mock_generate):
        mock_generate.return_value = {
            "predict_date": "2026-03-31",
            "lookback_start": "2025-10-01",
            "pool_size": 123,
            "result_df": "mock_df",
            "output_path": "output/top_picks_2026-03-31.csv",
        }

        buf = io.StringIO()
        with patch("sys.argv", ["predict.py", "--date", "2026-03-31", "--out", "output/custom.csv"]):
            with redirect_stdout(buf):
                predict.main()

        mock_generate.assert_called_once_with(date="2026-03-31", out="output/custom.csv")
        output = buf.getvalue()
        self.assertIn("Predict date:", output)
        self.assertIn("Saved to:", output)

    @patch("scripts.predict.generate_predictions", side_effect=RuntimeError("missing model"))
    def test_predict_cli_shows_friendly_error(self, _mock_generate):
        buf = io.StringIO()
        with patch("sys.argv", ["predict.py"]):
            with redirect_stdout(buf):
                with self.assertRaises(SystemExit) as ctx:
                    predict.main()

        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("Prediction failed: missing model", buf.getvalue())


class TestBuildPortfolioCliWrapper(unittest.TestCase):
    @patch("scripts.build_portfolio.build_portfolio_outputs")
    def test_build_portfolio_cli_forwards_args(self, mock_build):
        mock_build.return_value = {
            "target_path": "output/target_weights_2026-03-31.csv",
            "orders_path": "output/orders_2026-03-31.csv",
            "stats": {"buy_count": 1, "sell_count": 2, "hold_count": 3, "turnover": 0.1},
        }

        buf = io.StringIO()
        with patch(
            "sys.argv",
            [
                "build_portfolio.py",
                "--date",
                "2026-03-31",
                "--top_k",
                "60",
                "--max_weight",
                "0.03",
                "--rebalance_threshold",
                "0.005",
            ],
        ):
            with redirect_stdout(buf):
                build_portfolio.main()

        mock_build.assert_called_once_with(
            date="2026-03-31",
            top_k=60,
            buy_rank=300,
            hold_rank=500,
            max_weight=0.03,
            rebalance_threshold=0.005,
        )
        output = buf.getvalue()
        self.assertIn("Target weights saved:", output)
        self.assertIn("Orders saved:", output)

    @patch("scripts.build_portfolio.build_portfolio_outputs", side_effect=RuntimeError("missing picks"))
    def test_build_portfolio_cli_shows_friendly_error(self, _mock_build):
        buf = io.StringIO()
        with patch("sys.argv", ["build_portfolio.py"]):
            with redirect_stdout(buf):
                with self.assertRaises(SystemExit) as ctx:
                    build_portfolio.main()

        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("Portfolio build failed: missing picks", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
