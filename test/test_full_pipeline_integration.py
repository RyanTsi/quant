"""Cross-module pipeline integration test (lightweight, mocked services)."""

import unittest
from unittest.mock import patch

from runtime.bootstrap import build_default_registry


class TestFullPipelineIntegration(unittest.TestCase):
    def test_full_pipeline_order_with_mocked_services(self):
        seen = []

        class FakeDataService:
            def fetch_data(self, **kwargs):
                seen.append("fetch")

            def ingest_to_db(self, **kwargs):
                seen.append("ingest")

            def export_from_db(self, **kwargs):
                seen.append("export")

        class FakeModelService:
            def dump_to_qlib(self, **kwargs):
                seen.append("dump")

            def train_model(self, **kwargs):
                seen.append("train")

            def predict(self, **kwargs):
                seen.append("predict")

            def build_portfolio(self, **kwargs):
                seen.append("portfolio")

        with patch("runtime.tasks.build_data_service", return_value=FakeDataService()):
            with patch("runtime.tasks.build_model_service", return_value=FakeModelService()):
                with patch.dict("os.environ", {"PIPELINE_COOLDOWN_SECONDS": "0"}, clear=False):
                    ok = build_default_registry().run("full")

        self.assertTrue(ok)
        self.assertEqual(seen, ["fetch", "ingest", "export", "dump", "train", "predict", "portfolio"])


if __name__ == "__main__":
    unittest.main()
