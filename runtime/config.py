from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
_DOTENV_LOADED = False


@dataclass(frozen=True)
class AppSettings:
    data_path: str
    analysis_path: str
    send_buffer_path: str
    receive_buffer_path: str
    qlib_data_path: str
    tu_token: str | None
    db_host: str
    db_port: int
    gateway_list_symbols_timeout: int
    timeout: int
    pipeline_cooldown_seconds: float
    qlib_provider_uri: str
    qlib_mlruns_uri: str
    qlib_experiment_name: str
    qlib_recorder_id: str
    qlib_experiment_id: str
    qlib_workflow_config: str
    qlib_torch_dataloader_workers: str | None


def _load_dotenv_once() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    load_dotenv(BASE_DIR / ".env")
    _DOTENV_LOADED = True


def _ensure_dirs(paths: tuple[str, ...]) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def load_settings(env: Mapping[str, str] | None = None) -> AppSettings:
    _load_dotenv_once()
    source = env or os.environ

    data_path = str(BASE_DIR / ".data")
    analysis_path = str(Path(data_path) / "stock_analysis")
    send_buffer_path = str(Path(data_path) / "send_buffer")
    receive_buffer_path = str(Path(data_path) / "receive_buffer")
    qlib_data_path = str(Path(data_path) / "qlib_data")

    _ensure_dirs((data_path, analysis_path, send_buffer_path, receive_buffer_path, qlib_data_path))

    qlib_provider_uri = qlib_data_path
    default_mlruns = "file:///" + str((BASE_DIR / "mlruns").as_posix())
    qlib_mlruns_uri = source.get("QLIB_MLRUNS_URI", default_mlruns)

    return AppSettings(
        data_path=data_path,
        analysis_path=analysis_path,
        send_buffer_path=send_buffer_path,
        receive_buffer_path=receive_buffer_path,
        qlib_data_path=qlib_data_path,
        tu_token=source.get("TU_TOKEN"),
        db_host=source.get("DB_HOST", "127.0.0.1"),
        db_port=int(source.get("DB_PORT", "8080")),
        gateway_list_symbols_timeout=int(source.get("GATEWAY_LIST_SYMBOLS_TIMEOUT", "120")),
        timeout=int(source.get("TIMEOUT", "30")),
        pipeline_cooldown_seconds=float(source.get("PIPELINE_COOLDOWN_SECONDS", "3")),
        qlib_provider_uri=qlib_provider_uri,
        qlib_mlruns_uri=qlib_mlruns_uri,
        qlib_experiment_name=source.get("QLIB_EXPERIMENT_NAME", "quant_experiment"),
        qlib_recorder_id=source.get("QLIB_RECORDER_ID", "d0ca701f21c94c169d1a2b9b29c8bf40"),
        qlib_experiment_id=source.get("QLIB_EXPERIMENT_ID", "381621146439577988"),
        qlib_workflow_config=source.get(
            "QLIB_WORKFLOW_CONFIG",
            str(BASE_DIR / "alpha_models" / "workflow_config_transformer_Alpha158.yaml"),
        ),
        qlib_torch_dataloader_workers=source.get("QLIB_TORCH_DATALOADER_WORKERS"),
    )


_settings_cache: AppSettings | None = None


def get_settings(*, refresh: bool = False) -> AppSettings:
    global _settings_cache
    if refresh or _settings_cache is None:
        _settings_cache = load_settings()
    return _settings_cache
