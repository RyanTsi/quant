import unittest
from unittest.mock import patch, MagicMock
from data_pipeline.database import DBClient


def _mock_response(json_data=None, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    return resp


class TestDBClient(unittest.TestCase):

    def setUp(self):
        self.client = DBClient("127.0.0.1", 8080)

    def test_base_url(self):
        self.assertEqual(self.client.base_url, "http://127.0.0.1:8080/api/v1")

    @patch.object(DBClient, "health", return_value={"status": "ok"})
    def test_health_ok(self, mock_health):
        result = self.client.health()
        self.assertEqual(result["status"], "ok")

    def test_health_unreachable(self):
        with patch.object(self.client.session, "get", side_effect=ConnectionError("refused")):
            result = self.client.health()
            self.assertEqual(result["status"], "unreachable")
            self.assertIn("refused", result["error"])

    def test_insert_data(self):
        payload = [{"symbol": "SH600000", "date": "2026-03-20", "open": 10.0}]
        with patch.object(self.client.session, "post", return_value=_mock_response({"inserted": 1})) as mock_post:
            resp = self.client.insert_data(payload)
            mock_post.assert_called_once()
            self.assertEqual(resp.json()["inserted"], 1)

    def test_query_data_success(self):
        rows = {"count": 1, "data": [{"symbol": "SH600000", "date": "2026-03-20"}]}
        with patch.object(self.client.session, "get", return_value=_mock_response(rows)) as mock_get:
            resp = self.client.query_data("SH600000", "2026-01-01", "2026-03-20")
            self.assertIsNotNone(resp)
            self.assertEqual(resp.json()["count"], 1)
            args, kwargs = mock_get.call_args
            self.assertEqual(kwargs["params"]["symbol"], "SH600000")

    def test_query_multiple(self):
        rows = {"count": 2, "data": [{"symbol": "SH600000"}, {"symbol": "SH600001"}]}
        with patch.object(self.client.session, "post", return_value=_mock_response(rows)) as mock_post:
            resp = self.client.query_multiple(["SH600000", "SH600001"], "2026-01-01", "2026-03-20")
            self.assertIsNotNone(resp)
            body = mock_post.call_args[1]["json"]
            self.assertEqual(body["symbols"], ["SH600000", "SH600001"])

    def test_query_latest(self):
        rows = {"count": 5, "data": [{"date": f"2026-03-{i}"} for i in range(16, 21)]}
        with patch.object(self.client.session, "get", return_value=_mock_response(rows)) as mock_get:
            resp = self.client.query_latest("SH600000", n=5)
            self.assertIsNotNone(resp)
            self.assertEqual(mock_get.call_args[1]["params"]["n"], 5)

    def test_get_stats(self):
        stats = {"avg_close": 12.5, "max_high": 15.0}
        with patch.object(self.client.session, "get", return_value=_mock_response(stats)):
            resp = self.client.get_stats("SH600000", "2026-01-01", "2026-03-20")
            self.assertEqual(resp.json()["avg_close"], 12.5)

    def test_list_symbols(self):
        data = {"symbols": ["SH600000", "SZ000001"]}
        with patch.object(self.client.session, "get", return_value=_mock_response(data)):
            resp = self.client.list_symbols()
            self.assertIn("SH600000", resp.json()["symbols"])

if __name__ == "__main__":
    unittest.main()
