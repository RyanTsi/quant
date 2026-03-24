import unittest
import json
import os
import tempfile
from unittest.mock import patch
from utils.run_tracker import record_run, get_last_run, today, _load, _save, init


class TestRunTracker(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        )
        self.tmp.write("{}")
        self.tmp.close()
        self.patcher = patch("utils.run_tracker._tracker_file", self.tmp.name)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)

    def test_record_run_creates_entry(self):
        entry = record_run("test_task")
        self.assertIn("last_run", entry)

    def test_record_run_with_kwargs(self):
        entry = record_run("test_task", status="ok")
        self.assertEqual(entry["status"], "ok")

    def test_get_last_run_returns_none_for_unknown(self):
        self.assertIsNone(get_last_run("nonexistent"))

    def test_get_last_run_returns_entry(self):
        record_run("my_task")
        result = get_last_run("my_task")
        self.assertIsNotNone(result)
        self.assertIn("last_run", result)

    def test_today_format(self):
        t = today()
        self.assertEqual(len(t), 8)
        self.assertTrue(t.isdigit())

    def test_multiple_tasks_independent(self):
        record_run("task_a", status="done")
        record_run("task_b", status="running")
        a = get_last_run("task_a")
        b = get_last_run("task_b")
        self.assertEqual(a["status"], "done")
        self.assertEqual(b["status"], "running")

    def test_init_sets_tracker_file(self):
        tmpdir = tempfile.mkdtemp()
        init(tmpdir)
        from utils import run_tracker
        self.assertEqual(run_tracker._tracker_file, os.path.join(tmpdir, "run_history.json"))
        os.rmdir(tmpdir)


if __name__ == "__main__":
    unittest.main()
