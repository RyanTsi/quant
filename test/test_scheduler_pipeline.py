import unittest
from unittest.mock import patch

from scheduler.decorator import task, TaskFailed
from scheduler.pipelines import run_pipeline


class TestSchedulerPipeline(unittest.TestCase):
    def test_pipeline_stops_on_failure(self):
        seen = []

        @task("ok1")
        def ok1():
            seen.append("ok1")

        @task("boom")
        def boom():
            seen.append("boom")
            raise RuntimeError("fail")

        @task("ok2")
        def ok2():
            seen.append("ok2")

        with patch("time.sleep") as _:
            run_pipeline([ok1, boom, ok2])

        self.assertEqual(seen, ["ok1", "boom"])

    def test_pipeline_cooldown_called_between_tasks(self):
        @task("ok1")
        def ok1():
            return None

        @task("ok2")
        def ok2():
            return None

        with patch.dict("os.environ", {"PIPELINE_COOLDOWN_SECONDS": "0.5"}, clear=False):
            with patch("time.sleep") as sleep:
                run_pipeline([ok1, ok2])
                sleep.assert_called_once()
                args, _kwargs = sleep.call_args
                self.assertAlmostEqual(args[0], 0.5, places=2)


if __name__ == "__main__":
    unittest.main()

