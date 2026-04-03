import json
import os
import tempfile
import unittest
from unittest.mock import patch

import runtime.runlog as runtime_runlog
from runtime.runlog import RunLogStore, SCHEMA_VERSION


class TestRuntimeRunLog(unittest.TestCase):
    def test_legacy_flat_history_is_readable_and_migrates_on_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "run_history.json")
            legacy = {
                "fetch_stock": {"last_run": "2026-04-01 10:00:00", "status": "ok"},
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(legacy, f)

            store = RunLogStore(path)
            loaded = store.load()
            self.assertEqual(loaded["fetch_stock"]["status"], "ok")

            store.record("predict", date="2026-04-01")

            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.assertIn("_meta", raw)
            self.assertEqual(raw["_meta"]["schema_version"], SCHEMA_VERSION)
            self.assertIn("tasks", raw)
            self.assertIn("fetch_stock", raw["tasks"])
            self.assertIn("predict", raw["tasks"])

            loaded_again = store.load()
            self.assertIn("fetch_stock", loaded_again)
            self.assertIn("predict", loaded_again)

    def test_save_keeps_flat_load_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "run_history.json")
            store = RunLogStore(path)
            store.save({"task_a": {"last_run": "2026-04-01 08:00:00", "k": "v"}})

            loaded = store.load()
            self.assertEqual(list(loaded.keys()), ["task_a"])
            self.assertEqual(loaded["task_a"]["k"], "v")

    def test_malformed_schema_version_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "run_history.json")
            malformed = {
                "_meta": {"schema_version": "not-a-number"},
                "tasks": {"task_a": {"last_run": "2026-04-01 08:00:00"}},
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(malformed, f)

            store = RunLogStore(path)
            loaded = store.load()
            self.assertIn("task_a", loaded)

            store.record("task_b", status="ok")
            with open(path, "r", encoding="utf-8") as f:
                persisted = json.load(f)
            self.assertEqual(persisted["_meta"]["schema_version"], SCHEMA_VERSION)
            self.assertIn("task_b", persisted["tasks"])

    def test_record_uses_file_lock_when_available(self):
        if runtime_runlog.fcntl is None:
            self.skipTest("fcntl is not available on this platform")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "run_history.json")
            store = RunLogStore(path)

            with patch("runtime.runlog.fcntl.flock") as mock_flock:
                store.record("task_a", status="ok")

            self.assertGreaterEqual(mock_flock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
