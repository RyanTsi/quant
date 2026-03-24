"""Tests for config.settings singleton."""

import unittest

from config.settings import BASE_DIR, settings


class TestSettings(unittest.TestCase):
    def test_db_port_is_int(self):
        self.assertIsInstance(settings.db_port, int)

    def test_db_host_is_str(self):
        self.assertIsInstance(settings.db_host, str)

    def test_data_path_under_project(self):
        self.assertIn(str(BASE_DIR), settings.data_path)

    def test_qlib_ids_are_str(self):
        self.assertIsInstance(settings.qlib_recorder_id, str)
        self.assertIsInstance(settings.qlib_experiment_id, str)
        self.assertTrue(settings.qlib_recorder_id)
        self.assertTrue(settings.qlib_experiment_id)

    def test_qlib_mlruns_uri_scheme(self):
        self.assertTrue(
            settings.qlib_mlruns_uri.startswith("file:///")
            or settings.qlib_mlruns_uri.startswith("file://"),
        )


if __name__ == "__main__":
    unittest.main()
