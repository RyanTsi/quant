from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime

from quantcore.history import RunHistoryStore
from quantcore.settings import AppSettings


def _run_qlib_training() -> None:
    from alpha_models.qlib_workflow import main as qlib_main

    qlib_main()


class ModelPipelineService:
    def __init__(self, settings: AppSettings, *, history: RunHistoryStore | None = None):
        self.settings = settings
        self.history = history or RunHistoryStore(os.path.join(settings.data_path, "run_history.json"))

    @staticmethod
    def _today_dash() -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def dump_to_qlib(self) -> dict[str, str] | None:
        csv_dir = self.settings.receive_buffer_path
        qlib_dir = self.settings.qlib_data_path

        if not os.path.isdir(csv_dir) or not os.listdir(csv_dir):
            return None

        subprocess.run(
            [
                sys.executable,
                "scripts/dump_bin.py",
                "dump_all",
                f"--data_path={csv_dir}",
                f"--qlib_dir={qlib_dir}",
                "--include_fields=open,high,low,close,volume,amount,turn,isST,factor",
                "--file_suffix=.csv",
            ],
            check=True,
        )
        self.history.record("dump_to_qlib", csv_dir=csv_dir, qlib_dir=qlib_dir)
        return {"csv_dir": csv_dir, "qlib_dir": qlib_dir}

    def predict(self) -> dict[str, str]:
        subprocess.run([sys.executable, "-m", "scripts.predict"], check=True)
        self.history.record("predict", date=self._today_dash())
        return {"date": self._today_dash()}

    def build_portfolio(self) -> dict[str, str]:
        subprocess.run([sys.executable, "-m", "scripts.build_portfolio"], check=True)
        self.history.record("build_portfolio", date=self._today_dash())
        return {"date": self._today_dash()}

    def train_model(self) -> dict[str, str]:
        _run_qlib_training()
        self.history.record("train_model", date=self._today_dash())
        return {"date": self._today_dash()}
