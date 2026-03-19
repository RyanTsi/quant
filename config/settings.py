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
        self.output_path = os.path.join(BASE_DIR, 'output')
        # API tokens
        self.tu_token = os.getenv('TU_TOKEN')
        # database settings
        self.db_host = os.getenv('DB_HOST', 'localhost')
        self.db_port = int(os.getenv('DB_PORT', 8000))
        # scraper settings
        self.timeout = 30
        retry_time = 3
        retry_delay = 1
        

settings = Settings()