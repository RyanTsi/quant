# config/settings.py
import os
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(os.path.join(BASE_DIR, '.env'))


class Settings:
    def __init__(self):
        # path settings
        self.data_path = os.path.join(BASE_DIR, '.data')
        self.analysis_path = os.path.join(self.data_path, 'stock_analysis')
        self.send_buffer_path = os.path.join(self.data_path, 'send_buffer')

        # API tokens
        self.tu_token = os.getenv('TU_TOKEN')

        # server settings (C++ HTTP gateway)
        self.db_host = os.getenv('DB_HOST', '127.0.0.1')
        self.db_port = int(os.getenv('DB_PORT', 8080))

        # scraper settings
        self.timeout = 30

        # qlib settings
        self.qlib_provider_uri = os.getenv(
            'QLIB_PROVIDER_URI',
            os.path.join(self.data_path, 'qlib_data'),
        )
        self.qlib_mlruns_uri = os.getenv(
            'QLIB_MLRUNS_URI',
            f"file:///{os.path.join(BASE_DIR, 'mlruns').replace(os.sep, '/')}",
        )
        self.qlib_experiment_name = os.getenv('QLIB_EXPERIMENT_NAME', 'transformer_alpha158')
        self.qlib_recorder_id = os.getenv('QLIB_RECORDER_ID', '6c6aaaec2fc4431eb78d5b17d709b348')
        self.qlib_experiment_id = os.getenv('QLIB_EXPERIMENT_ID', '379677092195942384')

        # init directories
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.analysis_path, exist_ok=True)
        os.makedirs(self.send_buffer_path, exist_ok=True)


settings = Settings()
