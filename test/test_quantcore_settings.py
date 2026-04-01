"""Tests for quantcore.settings."""

import unittest
from unittest.mock import patch

from quantcore.settings import BASE_DIR, get_settings, load_settings


class TestQuantcoreSettings(unittest.TestCase):
    def test_load_settings_from_mapping(self):
        env = {
            "DB_HOST": "10.0.0.2",
            "DB_PORT": "9000",
            "PIPELINE_COOLDOWN_SECONDS": "1.25",
            "QLIB_EXPERIMENT_NAME": "exp_x",
        }
        s = load_settings(env=env)
        self.assertEqual(s.db_host, "10.0.0.2")
        self.assertEqual(s.db_port, 9000)
        self.assertAlmostEqual(s.pipeline_cooldown_seconds, 1.25, places=2)
        self.assertEqual(s.qlib_experiment_name, "exp_x")
        self.assertIn(str(BASE_DIR), s.data_path)

    def test_get_settings_refresh_reads_env(self):
        with patch.dict("os.environ", {"PIPELINE_COOLDOWN_SECONDS": "0.75"}, clear=False):
            s = get_settings(refresh=True)
        self.assertAlmostEqual(s.pipeline_cooldown_seconds, 0.75, places=2)


if __name__ == "__main__":
    unittest.main()
