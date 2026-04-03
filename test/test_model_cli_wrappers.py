"""CLI wrapper compatibility tests for model-side scripts."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from scripts import build_portfolio, filter as filter_script, predict


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


class TestFilterCliWrapper(unittest.TestCase):
    @patch("scripts.filter.build_model_service")
    def test_filter_top_liquidity_delegates_to_model_service(self, mock_build):
        service = mock_build.return_value
        service.build_training_universe.return_value = {"output_path": "output/mock.txt"}

        result = filter_script.filter_top_liquidity(start_year=2020, end_year=2021, top_n=300, random_seed=7)

        mock_build.assert_called_once_with(refresh_settings=True)
        service.build_training_universe.assert_called_once_with(
            start_year=2020,
            end_year=2021,
            top_n=300,
            random_seed=7,
        )
        self.assertEqual(result["output_path"], "output/mock.txt")

    @patch("scripts.filter.filter_top_liquidity")
    def test_filter_cli_forwards_args(self, mock_filter):
        mock_filter.return_value = {
            "output_path": "output/my_800_stocks.txt",
            "effective_end": "2026-04-03",
            "source_month_count": 2,
            "symbol_count": 5,
        }

        buf = io.StringIO()
        with patch(
            "sys.argv",
            [
                "filter.py",
                "--start_year",
                "2015",
                "--end_year",
                "2020",
                "--top_n",
                "1800",
                "--random_seed",
                "99",
            ],
        ):
            with redirect_stdout(buf):
                filter_script.main()

        mock_filter.assert_called_once_with(start_year=2015, end_year=2020, top_n=1800, random_seed=99)
        output = buf.getvalue()
        self.assertIn("Saved to:", output)
        self.assertIn("Unique symbols in artifact:", output)

    @patch("scripts.filter.filter_top_liquidity", side_effect=RuntimeError("db unavailable"))
    def test_filter_cli_shows_friendly_error(self, _mock_filter):
        buf = io.StringIO()
        with patch("sys.argv", ["filter.py"]):
            with redirect_stdout(buf):
                with self.assertRaises(SystemExit) as ctx:
                    filter_script.main()

        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("Filter build failed: db unavailable", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
