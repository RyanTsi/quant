"""Focused tests for runtime dump-to-qlib adapter behavior."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from runtime.adapters import modeling


class TestModelingDumpAdapter(unittest.TestCase):
    @patch("runtime.adapters.modeling._get_dump_all_class")
    def test_dump_to_qlib_data_uses_dump_all_with_expected_defaults(self, mock_get_cls):
        fake_dumper = MagicMock()
        fake_cls = MagicMock(return_value=fake_dumper)
        mock_get_cls.return_value = fake_cls

        result = modeling.dump_to_qlib_data(csv_dir="/tmp/csvs", qlib_dir="/tmp/qlib")

        fake_cls.assert_called_once_with(
            data_path="/tmp/csvs",
            qlib_dir="/tmp/qlib",
            include_fields=modeling.DEFAULT_DUMP_INCLUDE_FIELDS,
            file_suffix=modeling.DEFAULT_DUMP_FILE_SUFFIX,
        )
        fake_dumper.dump.assert_called_once_with()
        self.assertEqual(result, {"csv_dir": "/tmp/csvs", "qlib_dir": "/tmp/qlib"})

    @patch("runtime.adapters.modeling._get_dump_all_class")
    def test_dump_to_qlib_data_forwards_custom_fields_and_suffix(self, mock_get_cls):
        fake_dumper = MagicMock()
        fake_cls = MagicMock(return_value=fake_dumper)
        mock_get_cls.return_value = fake_cls

        modeling.dump_to_qlib_data(
            csv_dir="/data/source",
            qlib_dir="/data/qlib",
            include_fields="open,close",
            file_suffix=".parquet",
        )

        fake_cls.assert_called_once_with(
            data_path="/data/source",
            qlib_dir="/data/qlib",
            include_fields="open,close",
            file_suffix=".parquet",
        )
        fake_dumper.dump.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
