import unittest
import tempfile
from runtime.runlog import get_last_run, load_run_history, record_run, save_run_history, today


class TestRunLogHelpers(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmp.close()

    def tearDown(self):
        import os

        if os.path.exists(self.tmp.name):
            os.unlink(self.tmp.name)

    def test_record_run_creates_entry(self):
        entry = record_run("test_task", path=self.tmp.name)
        self.assertIn("last_run", entry)

    def test_record_run_with_kwargs(self):
        entry = record_run("test_task", path=self.tmp.name, status="ok")
        self.assertEqual(entry["status"], "ok")

    def test_get_last_run_returns_none_for_unknown(self):
        self.assertIsNone(get_last_run("nonexistent", path=self.tmp.name))

    def test_get_last_run_returns_entry(self):
        record_run("my_task", path=self.tmp.name)
        result = get_last_run("my_task", path=self.tmp.name)
        self.assertIsNotNone(result)
        self.assertIn("last_run", result)

    def test_today_format(self):
        t = today()
        self.assertEqual(len(t), 8)
        self.assertTrue(t.isdigit())

    def test_multiple_tasks_independent(self):
        record_run("task_a", path=self.tmp.name, status="done")
        record_run("task_b", path=self.tmp.name, status="running")
        a = get_last_run("task_a", path=self.tmp.name)
        b = get_last_run("task_b", path=self.tmp.name)
        self.assertEqual(a["status"], "done")
        self.assertEqual(b["status"], "running")

    def test_load_and_save_round_trip(self):
        save_run_history({"task_a": {"last_run": "2026-04-02 10:00:00", "status": "ok"}}, path=self.tmp.name)
        loaded = load_run_history(path=self.tmp.name)
        self.assertEqual(loaded["task_a"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
