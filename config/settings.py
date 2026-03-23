# config/settings.py
import os
import yaml
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(os.path.join(BASE_DIR, '.env'))

class Settings:
    def __init__(self):
        # path settings
        self.data_path = os.path.join(BASE_DIR, '.data')
        self.analysis_path = os.path.join(self.data_path, 'stock_analysis')
        # API tokens
        self.tu_token = os.getenv('TU_TOKEN')
        # server settings (C++ HTTP gateway, not the DB port)
        self.db_host = os.getenv('DB_HOST', '127.0.0.1')
        self.db_port = int(os.getenv('DB_PORT', 8080))
        # scraper settings
        self.timeout = 30
        retry_time = 3
        retry_delay = 1

        # init directory
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path, exist_ok=True)
        if not os.path.exists(self.analysis_path):
            os.makedirs(self.analysis_path, exist_ok=True)
        

settings = Settings()