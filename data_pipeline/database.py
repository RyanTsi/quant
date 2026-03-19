import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class DBClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}/api/v1"
        
        self.session = requests.Session()
        
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        self.session.mount('http://', HTTPAdapter(max_retries=retries))

    def insert_data(self, data):
        url = f"{self.base_url}/ingest/daily"
        response = self.session.post(url, json=data)
        return response

    def query_data(self, symbol, start_date, end_date):
        url = f"{self.base_url}/query/daily/symbol"
        query = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date
        }
        try:
            response = self.session.get(url, params=query, timeout=10)
            return response
        except Exception as e:
            print(f"Query failed for {symbol}: {e}")
            return None