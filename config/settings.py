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
        # API tokens
        self.tu_token = os.getenv('TU_TOKEN')

settings = Settings()