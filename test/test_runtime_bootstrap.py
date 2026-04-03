import unittest

from runtime.bootstrap import build_default_registry
from runtime.constants import PIPELINE_TASK_NAMES, TASK_NAMES


class TestRuntimeBootstrap(unittest.TestCase):
    def test_build_default_registry_has_stable_names(self):
        registry = build_default_registry()

        self.assertEqual(set(registry.task_map.keys()), set(TASK_NAMES))
        self.assertEqual(registry.pipeline_map, PIPELINE_TASK_NAMES)
        self.assertEqual(registry.pipeline_map["full"], ["fetch", "ingest", "export", "dump", "train", "predict", "portfolio"])


if __name__ == "__main__":
    unittest.main()
