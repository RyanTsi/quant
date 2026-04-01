"""Cross-module pipeline integration test (lightweight, mocked services)."""

import unittest
from unittest.mock import patch

from scheduler.pipelines import FULL_PIPELINE, run_pipeline


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

        with patch("scheduler.data_tasks.build_data_service", return_value=FakeDataService()):
            with patch("scheduler.model_tasks.build_model_service", return_value=FakeModelService()):
                with patch("time.sleep"):
                    ok = run_pipeline(FULL_PIPELINE)

        self.assertTrue(ok)
        self.assertEqual(seen, ["fetch", "ingest", "export", "dump", "train", "predict", "portfolio"])


if __name__ == "__main__":
    unittest.main()
