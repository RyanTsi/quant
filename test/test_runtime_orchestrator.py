import logging
import unittest
from unittest.mock import patch

from runtime.contracts import TaskSpec
from runtime.orchestrator import SequentialOrchestrator


class TestRuntimeOrchestrator(unittest.TestCase):
    def test_pipeline_runs_in_order(self):
        seen = []

        def task_a():
            seen.append("a")

        def task_b():
            seen.append("b")

        orchestrator = SequentialOrchestrator(
            logger=logging.getLogger("test-runtime-orchestrator"),
            cooldown_provider=lambda: 0.0,
        )
        result = orchestrator.run_pipeline(
            "demo",
            [
                TaskSpec(name="task_a", fn=task_a),
                TaskSpec(name="task_b", fn=task_b),
            ],
        )

        self.assertTrue(result.success)
        self.assertEqual(seen, ["a", "b"])

    def test_pipeline_stops_on_failure(self):
        seen = []

        def ok():
            seen.append("ok")

        def boom():
            seen.append("boom")
            raise RuntimeError("fail")

        def never():
            seen.append("never")

        orchestrator = SequentialOrchestrator(
            logger=logging.getLogger("test-runtime-orchestrator"),
            cooldown_provider=lambda: 0.0,
        )
        result = orchestrator.run_pipeline(
            "demo",
            [
                TaskSpec(name="ok", fn=ok),
                TaskSpec(name="boom", fn=boom),
                TaskSpec(name="never", fn=never),
            ],
        )

        self.assertFalse(result.success)
        self.assertEqual(seen, ["ok", "boom"])
        self.assertEqual([t.name for t in result.tasks], ["ok", "boom"])

    def test_pipeline_applies_cooldown_between_tasks(self):
        orchestrator = SequentialOrchestrator(
            logger=logging.getLogger("test-runtime-orchestrator"),
            cooldown_provider=lambda: 0.5,
        )
        with patch("time.sleep") as sleep:
            orchestrator.run_pipeline(
                "demo",
                [TaskSpec(name="a", fn=lambda: None), TaskSpec(name="b", fn=lambda: None)],
            )
            sleep.assert_called_once_with(0.5)

    def test_run_task_can_raise_when_requested(self):
        orchestrator = SequentialOrchestrator(
            logger=logging.getLogger("test-runtime-orchestrator"),
            cooldown_provider=lambda: 0.0,
        )

        def boom():
            raise ValueError("x")

        with self.assertRaises(ValueError):
            orchestrator.run_task(TaskSpec(name="boom", fn=boom), raise_on_failure=True)


if __name__ == "__main__":
    unittest.main()
