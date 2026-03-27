"""HTTP client for the C++ data gateway (Drogon)."""

import logging

import requests
from config.settings import settings
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class DBClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}/api/v1"

        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        self.session.mount('http://', HTTPAdapter(max_retries=retries))

    def health(self):
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=5)
            return resp.json()
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}

    def insert_data(self, data):
        url = f"{self.base_url}/ingest/daily"
        response = self.session.post(url, json=data, timeout=30)
        return response

    def query_data(self, symbol, start_date, end_date, limit=5000, offset=0):
        url = f"{self.base_url}/query/daily/symbol"
        params = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit,
            "offset": offset,
        }
        try:
            response = self.session.get(url, params=params, timeout=10)
            return response
        except Exception as e:
            logger.error("Query failed for %s: %s", symbol, e)
            return None

    def query_multiple(self, symbols, start_date, end_date, limit=50000, offset=0):
        url = f"{self.base_url}/query/daily/symbols"
        body = {
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit,
            "offset": offset,
        }
        try:
            response = self.session.post(url, json=body, timeout=30)
            return response
        except Exception as e:
            logger.error("Multi-symbol query failed: %s", e)
            return None

    def query_latest(self, symbol, n=30):
        url = f"{self.base_url}/query/daily/latest"
        params = {"symbol": symbol, "n": n}
        try:
            response = self.session.get(url, params=params, timeout=10)
            return response
        except Exception as e:
            logger.error("Latest query failed for %s: %s", symbol, e)
            return None

    def get_stats(self, symbol, start_date, end_date):
        url = f"{self.base_url}/stats/summary"
        params = {"symbol": symbol, "start_date": start_date, "end_date": end_date}
        try:
            response = self.session.get(url, params=params, timeout=10)
            return response
        except Exception as e:
            logger.error("Stats query failed for %s: %s", symbol, e)
            return None

    def list_symbols(self):
        url = f"{self.base_url}/symbols"
        try:
            response = self.session.get(
                url, timeout=settings.gateway_list_symbols_timeout
            )
            return response
        except Exception as e:
            logger.error("List symbols failed: %s", e)
            return None

    def delete_data(self, symbol, start_date=None, end_date=None):
        url = f"{self.base_url}/data/daily"
        params = {"symbol": symbol}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        try:
            response = self.session.delete(url, params=params, timeout=10)
            return response
        except Exception as e:
            logger.error("Delete failed for %s: %s", symbol, e)
            return None
