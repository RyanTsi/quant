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
        self.receive_buffer_path = os.path.join(self.data_path, 'receive_buffer')
        self.qlib_data_path = os.path.join(self.data_path, 'qlib_data')
        
        # API tokens
        self.tu_token = os.getenv('TU_TOKEN')

        # server settings (C++ HTTP gateway)
        self.db_host = os.getenv('DB_HOST', '127.0.0.1')
        self.db_port = int(os.getenv('DB_PORT', 8080))
        # list_symbols runs DISTINCT over market_data_daily; allow slow scans (seconds default).
        self.gateway_list_symbols_timeout = int(os.getenv('GATEWAY_LIST_SYMBOLS_TIMEOUT', '120'))

        # scraper settings
        self.timeout = 30

        # scheduler settings
        self.pipeline_cooldown_seconds = float(os.getenv("PIPELINE_COOLDOWN_SECONDS", "3"))

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
        self.qlib_recorder_id = os.getenv('QLIB_RECORDER_ID', '26026d1687f344f4971fa15df7d0734d')
        self.qlib_experiment_id = os.getenv('QLIB_EXPERIMENT_ID', '381621146439577988')
        self.qlib_workflow_config = os.getenv(
            "QLIB_WORKFLOW_CONFIG",
            str(BASE_DIR / "alpha_models" / "workflow_config_transformer_Alpha158.yaml"),
        )
        # For Windows stability; default to 0 unless explicitly set.
        self.qlib_torch_dataloader_workers = os.getenv("QLIB_TORCH_DATALOADER_WORKERS")

        # init directories
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.analysis_path, exist_ok=True)
        os.makedirs(self.send_buffer_path, exist_ok=True)


settings = Settings()
