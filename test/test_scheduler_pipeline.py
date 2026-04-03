import logging
import unittest
from unittest.mock import patch

from runtime.bootstrap import cooldown_seconds
from runtime.contracts import TaskSpec
from runtime.orchestrator import SequentialOrchestrator
from runtime.registry import RuntimeRegistry


class TestRuntimePipelineSemantics(unittest.TestCase):
    def test_pipeline_stops_on_failure(self):
        seen = []

        def ok1():
            seen.append("ok1")

        def boom():
            seen.append("boom")
            raise RuntimeError("fail")

        def ok2():
            seen.append("ok2")

        registry = RuntimeRegistry(
            task_map={"ok1": ok1, "boom": boom, "ok2": ok2},
            pipeline_map={"demo": ["ok1", "boom", "ok2"]},
            orchestrator=SequentialOrchestrator(
                logger=logging.getLogger("test-runtime-pipeline"),
                cooldown_provider=lambda: 0.0,
            ),
        )
        ok = registry.run("demo")

        self.assertFalse(ok)
        self.assertEqual(seen, ["ok1", "boom"])

    def test_pipeline_cooldown_called_between_tasks(self):
        orchestrator = SequentialOrchestrator(
            logger=logging.getLogger("test-runtime-pipeline"),
            cooldown_provider=lambda: 0.5,
        )
        with patch("time.sleep") as sleep:
            result = orchestrator.run_pipeline(
                "demo",
                [TaskSpec(name="ok1", fn=lambda: None), TaskSpec(name="ok2", fn=lambda: None)],
            )

        self.assertTrue(result.success)
        sleep.assert_called_once_with(0.5)

    def test_cooldown_seconds_prefers_env_override(self):
        with patch.dict("os.environ", {"PIPELINE_COOLDOWN_SECONDS": "0.75"}, clear=False):
            self.assertAlmostEqual(cooldown_seconds(), 0.75, places=2)


if __name__ == "__main__":
    unittest.main()
