"""Focused contract tests for runtime model adapter outputs."""

from __future__ import annotations

import os
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from runtime.adapters.modeling import TradingDateContext, build_portfolio_outputs


class TestModelingAdapterContract(unittest.TestCase):
    @patch(
        "runtime.adapters.modeling._resolve_execution_trading_context",
        return_value=TradingDateContext(
            trading_date="2026-03-31",
            previous_trading_date="2026-03-30",
        ),
    )
    def test_build_portfolio_outputs_writes_expected_artifacts(self, _mock_context):
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

    @patch(
        "runtime.adapters.modeling._resolve_execution_trading_context",
        return_value=TradingDateContext(
            trading_date="2026-03-31",
            previous_trading_date="2026-03-30",
        ),
    )
    def test_build_portfolio_outputs_raises_if_prediction_file_missing(self, _mock_context):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                build_portfolio_outputs(
                    date="2026-03-31",
                    output_dir=tmpdir,
                    track_run=False,
                )

    @patch(
        "runtime.adapters.modeling._resolve_execution_trading_context",
        return_value=TradingDateContext(
            trading_date="2026-03-31",
            previous_trading_date="2026-03-30",
        ),
    )
    def test_build_portfolio_outputs_applies_hold_band_before_capacity_cap(self, _mock_context):
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

    @patch(
        "runtime.adapters.modeling._resolve_execution_trading_context",
        return_value=TradingDateContext(
            trading_date="2026-03-31",
            previous_trading_date="2026-03-30",
        ),
    )
    @patch("runtime.adapters.modeling.build_model_runtime_state")
    def test_build_portfolio_outputs_tracks_run_via_runtime_state_when_enabled(
        self,
        mock_build_runtime_state,
        _mock_context,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            date = "2026-03-31"
            picks_path = os.path.join(tmpdir, f"top_picks_{date}.csv")
            pd.DataFrame(
                {
                    "instrument": ["A", "B"],
                    "Score": [2.0, 1.0],
                }
            ).to_csv(picks_path, index=False)

            runtime_state = Mock()
            mock_build_runtime_state.return_value = runtime_state

            result = build_portfolio_outputs(
                date=date,
                output_dir=tmpdir,
                top_k=2,
                max_weight=0.8,
                rebalance_threshold=0.001,
                track_run=True,
            )

            runtime_state.history.record.assert_called_once_with(
                "build_portfolio",
                date=date,
                picks_file=os.path.join(tmpdir, f"top_picks_{date}.csv"),
                target_file=result["target_path"],
                orders_file=result["orders_path"],
                **result["stats"],
            )

    @patch(
        "runtime.adapters.modeling._resolve_execution_trading_context",
        return_value=TradingDateContext(
            trading_date="2026-04-06",
            previous_trading_date="2026-04-03",
        ),
    )
    def test_build_portfolio_outputs_uses_previous_trading_day_across_weekend_gap(self, _mock_context):
        with tempfile.TemporaryDirectory() as tmpdir:
            date = "2026-04-06"
            picks_path = os.path.join(tmpdir, f"top_picks_{date}.csv")
            pd.DataFrame(
                {
                    "instrument": ["A", "B", "C", "D", "E", "F"],
                    "Score": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                }
            ).to_csv(picks_path, index=False)

            prev_target_path = os.path.join(tmpdir, "target_weights_2026-04-03.csv")
            pd.DataFrame(
                {
                    "instrument": ["D"],
                    "target_weight": [1.0],
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

    @patch("runtime.adapters.modeling._get_qlib_data_client")
    @patch("runtime.adapters.modeling._get_qlib_runtime")
    def test_build_portfolio_outputs_defaults_to_latest_local_trading_day(self, mock_get_runtime, mock_get_d):
        mock_qlib = Mock()
        mock_get_runtime.return_value = (mock_qlib, "cn", Mock(), lambda _conf: "dataset")
        mock_d = Mock()
        mock_d.calendar.return_value = [
            pd.Timestamp("2026-04-02"),
            pd.Timestamp("2026-04-03"),
            pd.Timestamp("2026-04-06"),
        ]
        mock_get_d.return_value = mock_d
        runtime_state = types.SimpleNamespace(
            settings=types.SimpleNamespace(qlib_provider_uri="provider://default"),
            history=Mock(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pd.DataFrame(
                {
                    "instrument": ["A", "B", "C", "D", "E", "F"],
                    "Score": [6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                }
            ).to_csv(os.path.join(tmpdir, "top_picks_2026-04-06.csv"), index=False)
            pd.DataFrame(
                {
                    "instrument": ["D"],
                    "target_weight": [1.0],
                }
            ).to_csv(os.path.join(tmpdir, "target_weights_2026-04-03.csv"), index=False)

            result = build_portfolio_outputs(
                date=None,
                output_dir=tmpdir,
                top_k=3,
                buy_rank=2,
                hold_rank=4,
                max_weight=0.8,
                rebalance_threshold=0.001,
                track_run=False,
                runtime_state=runtime_state,
            )

            self.assertEqual(result["date"], "2026-04-06")
            self.assertTrue(os.path.isfile(result["target_path"]))
            target_df = pd.read_csv(result["target_path"])
            self.assertEqual(target_df["instrument"].tolist(), ["A", "B", "D"])

        mock_qlib.init.assert_called_once_with(provider_uri="provider://default", region="cn")
        mock_d.calendar.assert_called_once_with(freq="day")


if __name__ == "__main__":
    unittest.main()
