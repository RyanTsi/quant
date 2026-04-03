"""Contract tests for runtime.adapters.fetching."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock

from runtime.adapters import fetching


class TestResolveFetchWindow(unittest.TestCase):
    def test_uses_fallback_last_end_date_when_history_missing(self):
        result = fetching.resolve_fetch_window(
            lookback_days=7,
            last_history=None,
            now=datetime(2026, 4, 1),
        )
        self.assertEqual(result["last_end_date"], "20100108")
        self.assertEqual(result["start_date"], "20100101")
        self.assertEqual(result["end_date"], "20260401")

    def test_uses_history_end_date_when_present(self):
        result = fetching.resolve_fetch_window(
            lookback_days=7,
            last_history={"end_date": "20260331"},
            now=datetime(2026, 4, 1),
        )
        self.assertEqual(result["last_end_date"], "20260331")
        self.assertEqual(result["start_date"], "20260324")
        self.assertEqual(result["end_date"], "20260401")


class TestFetchAndPackageMarketData(unittest.TestCase):
    def test_calls_fetcher_and_package_with_same_window_and_save_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            send_buffer_dir = os.path.join(tmpdir, "send_buffer")
            os.makedirs(send_buffer_dir, exist_ok=True)
            fetcher = MagicMock()
            package_data_fn = MagicMock()

            result = fetching.fetch_and_package_market_data(
                data_root=tmpdir,
                send_buffer_dir=send_buffer_dir,
                lookback_days=7,
                last_history={"end_date": "20260331"},
                fetcher_factory=lambda: fetcher,
                package_data_fn=package_data_fn,
                now=datetime(2026, 4, 1),
            )

            expected_save_dir = os.path.join(tmpdir, "20260324-20260401")
            self.assertEqual(result["start_date"], "20260324")
            self.assertEqual(result["end_date"], "20260401")
            self.assertEqual(result["last_end_date"], "20260331")
            self.assertEqual(result["lookback_days"], 7)
            self.assertEqual(result["save_dir"], expected_save_dir)
            self.assertEqual(result["send_buffer_dir"], send_buffer_dir)
            self.assertTrue(os.path.isdir(expected_save_dir))

            fetcher.fetch_all_stock_history.assert_called_once_with("20260324", "20260401", expected_save_dir)
            fetcher.fetch_all_index_history.assert_called_once_with("20260324", "20260401", expected_save_dir)
            package_data_fn.assert_called_once_with(expected_save_dir, send_buffer_dir)

    def test_propagates_fetcher_factory_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            package_data_fn = MagicMock()

            def _bad_factory():
                raise RuntimeError("fetcher init failed")

            with self.assertRaisesRegex(RuntimeError, "fetcher init failed"):
                fetching.fetch_and_package_market_data(
                    data_root=tmpdir,
                    send_buffer_dir=os.path.join(tmpdir, "send_buffer"),
                    lookback_days=7,
                    last_history=None,
                    fetcher_factory=_bad_factory,
                    package_data_fn=package_data_fn,
                    now=datetime(2026, 4, 1),
                )

            package_data_fn.assert_not_called()

    def test_propagates_package_data_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            send_buffer_dir = os.path.join(tmpdir, "send_buffer")
            fetcher = MagicMock()
            package_data_fn = MagicMock(side_effect=RuntimeError("package failed"))

            with self.assertRaisesRegex(RuntimeError, "package failed"):
                fetching.fetch_and_package_market_data(
                    data_root=tmpdir,
                    send_buffer_dir=send_buffer_dir,
                    lookback_days=1,
                    last_history={"end_date": "20260331"},
                    fetcher_factory=lambda: fetcher,
                    package_data_fn=package_data_fn,
                    now=datetime(2026, 4, 1),
                )

            expected_save_dir = os.path.join(tmpdir, "20260330-20260401")
            fetcher.fetch_all_stock_history.assert_called_once_with("20260330", "20260401", expected_save_dir)
            fetcher.fetch_all_index_history.assert_called_once_with("20260330", "20260401", expected_save_dir)


if __name__ == "__main__":
    unittest.main()
