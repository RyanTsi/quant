import unittest
from unittest.mock import MagicMock, patch

from runtime import tasks


class TestRuntimeTasks(unittest.TestCase):
    def test_task_names_are_stable(self):
        self.assertEqual(tasks.fetch_data.task_name, "fetch_data")
        self.assertEqual(tasks.ingest_to_db.task_name, "ingest_to_db")
        self.assertEqual(tasks.export_from_db.task_name, "export_from_db")
        self.assertEqual(tasks.dump_to_qlib.task_name, "dump_to_qlib")
        self.assertEqual(tasks.filter_training_universe.task_name, "filter_training_universe")
        self.assertEqual(tasks.train_model.task_name, "train_model")
        self.assertEqual(tasks.predict.task_name, "predict")
        self.assertEqual(tasks.build_portfolio.task_name, "build_portfolio")

    @patch("runtime.tasks.build_data_service")
    def test_fetch_data_uses_default_lookback(self, mock_build):
        service = MagicMock()
        mock_build.return_value = service

        tasks.fetch_data()

        mock_build.assert_called_once_with(refresh_settings=True)
        service.fetch_data.assert_called_once_with(lookback_days=7)

    @patch("runtime.tasks.build_data_service")
    def test_ingest_to_db_enables_delete_after_ingest(self, mock_build):
        service = MagicMock()
        mock_build.return_value = service

        tasks.ingest_to_db()

        mock_build.assert_called_once_with(refresh_settings=True)
        service.ingest_to_db.assert_called_once_with(delete_after_ingest=True)

    @patch("runtime.tasks.build_data_service")
    def test_export_from_db_uses_historical_start_date(self, mock_build):
        service = MagicMock()
        mock_build.return_value = service

        tasks.export_from_db()

        mock_build.assert_called_once_with(refresh_settings=True)
        service.export_from_db.assert_called_once_with(start_date="2010-01-01")

    @patch("runtime.tasks.build_model_service")
    def test_model_tasks_delegate_without_extra_arguments(self, mock_build):
        service = MagicMock()
        mock_build.return_value = service

        tasks.dump_to_qlib()
        tasks.filter_training_universe()
        tasks.train_model()
        tasks.predict()
        tasks.build_portfolio()

        self.assertEqual(mock_build.call_count, 5)
        for call in mock_build.call_args_list:
            self.assertEqual(call.kwargs, {"refresh_settings": True})
        service.dump_to_qlib.assert_called_once_with()
        service.build_training_universe.assert_called_once_with()
        service.train_model.assert_called_once_with()
        service.predict.assert_called_once_with()
        service.build_portfolio.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
