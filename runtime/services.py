from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

from data_pipeline.database import DBClient
from runtime.adapters import modeling
from runtime.adapters.exporting import export_from_gateway
from runtime.adapters.fetching import fetch_and_package_market_data
from runtime.adapters.ingest import ingest_directory
from runtime.config import AppSettings, get_settings
from runtime.model_state import ModelRuntimeState, build_model_runtime_state
from runtime.runlog import RunLogStore


def _run_qlib_training(runtime_state: ModelRuntimeState):
    """Run the Qlib training workflow through the shared workflow wrapper."""

    from alpha_models.qlib_workflow import run_training

    return run_training(runtime_state=runtime_state)


class DataPipelineService:
    def __init__(self, settings: AppSettings, *, history: RunLogStore | None = None):
        self.settings = settings
        self.history = history or RunLogStore(os.path.join(settings.data_path, "run_history.json"))
        self.logger = logging.getLogger("scheduler")

    def fetch_data(self, *, lookback_days: int = 7) -> dict[str, str | int]:
        """Fetch stock/index data then package CSV chunks for ingest."""
        result = fetch_and_package_market_data(
            data_root=self.settings.data_path,
            send_buffer_dir=self.settings.send_buffer_path,
            lookback_days=lookback_days,
            last_history=self.history.get("fetch_stock"),
            logger=self.logger,
        )
        self.history.record("fetch_stock", **result)
        return result

    def ingest_to_db(
        self,
        *,
        data_dir: str | None = None,
        delete_after_ingest: bool = True,
    ) -> dict[str, Any] | None:
        """Push packaged CSV chunks to the C++ gateway."""
        resolved_data_dir = data_dir or self.settings.send_buffer_path
        if not os.path.isdir(resolved_data_dir):
            self.logger.warning("  No data directory found, skipping.")
            return None

        server_url = f"http://{self.settings.db_host}:{self.settings.db_port}"
        self.logger.info("  Server: %s", server_url)
        self.logger.info("  Data:   %s", resolved_data_dir)
        result = ingest_directory(
            server_url,
            resolved_data_dir,
            delete_after_ingest=delete_after_ingest,
            logger_override=self.logger,
        )
        self.history.record("ingest_to_db", **result)
        return result

    def export_from_db(self, *, start_date: str = "2010-01-01") -> dict[str, int | str | list[str]]:
        """Export all symbols from DB to per-symbol CSV files in receive buffer."""
        end_date = datetime.now().strftime("%Y-%m-%d")
        client = DBClient(self.settings.db_host, self.settings.db_port)
        symbol_fallback_paths = (
            os.path.join(self.settings.data_path, "stock_code_list"),
            os.path.join(self.settings.data_path, "index_code_list"),
            os.path.join(self.settings.qlib_data_path, "instruments", "all.txt"),
        )
        result = export_from_gateway(
            client,
            start_date=start_date,
            end_date=end_date,
            output_dir=self.settings.receive_buffer_path,
            logger=self.logger,
            symbol_fallback_paths=symbol_fallback_paths,
            prefer_local_symbol_fallback=True,
        )
        failed_symbols = list(result.get("failed_symbols", []))
        partial_symbols = list(result.get("partial_symbols", []))

        self.history.record(
            "export_from_db",
            output_dir=result["output_dir"],
            exported=result["exported"],
            total=result["total"],
            failed=len(failed_symbols),
            partial=len(partial_symbols),
            failed_symbols=failed_symbols,
            partial_symbols=partial_symbols,
        )
        return {
            "output_dir": result["output_dir"],
            "exported": result["exported"],
            "total": result["total"],
            "failed_symbols": failed_symbols,
            "partial_symbols": partial_symbols,
        }


class ModelPipelineService:
    def __init__(self, settings: AppSettings, *, history: RunLogStore | None = None):
        self.settings = settings
        self.history = history or RunLogStore(os.path.join(settings.data_path, "run_history.json"))
        self.runtime_state = build_model_runtime_state(settings=settings, history=self.history)

    @staticmethod
    def _today_dash() -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def dump_to_qlib(self) -> dict[str, str] | None:
        csv_dir = self.settings.receive_buffer_path
        qlib_dir = self.settings.qlib_data_path

        if not os.path.isdir(csv_dir) or not os.listdir(csv_dir):
            return None

        modeling.dump_to_qlib_data(
            csv_dir=csv_dir,
            qlib_dir=qlib_dir,
            include_fields=modeling.DEFAULT_DUMP_INCLUDE_FIELDS,
            file_suffix=modeling.DEFAULT_DUMP_FILE_SUFFIX,
        )
        self.history.record("dump_to_qlib", csv_dir=csv_dir, qlib_dir=qlib_dir)
        return {"csv_dir": csv_dir, "qlib_dir": qlib_dir}

    @staticmethod
    def _predict_history_payload(result: dict[str, Any]) -> dict[str, Any]:
        return {
            "date": str(result.get("predict_date", "")),
            "predict_date": str(result.get("predict_date", "")),
            "lookback_start": str(result.get("lookback_start", "")),
            "pool_size": int(result.get("pool_size", 0)),
            "recorder_id": str(result.get("recorder_id", "")),
            "experiment_id": str(result.get("experiment_id", "")),
            "output_path": str(result.get("output_path", "")),
        }

    @staticmethod
    def _portfolio_history_payload(result: dict[str, Any]) -> dict[str, Any]:
        stats = result.get("stats")
        normalized_stats = dict(stats) if isinstance(stats, dict) else {}
        return {
            "date": str(result.get("date", "")),
            "picks_file": str(result.get("picks_path", "")),
            "target_file": str(result.get("target_path", "")),
            "orders_file": str(result.get("orders_path", "")),
            **normalized_stats,
        }

    @staticmethod
    def _training_universe_history_payload(result: dict[str, Any]) -> dict[str, Any]:
        return {
            "output_path": str(result.get("output_path", "")),
            "start_year": int(result.get("start_year", 0)),
            "end_year": int(result.get("end_year", 0)),
            "top_n": int(result.get("top_n", 0)),
            "random_seed": int(result.get("random_seed", 0)),
            "effective_end": str(result.get("effective_end", "")),
            "source_month_count": int(result.get("source_month_count", 0)),
            "range_count": int(result.get("range_count", 0)),
            "symbol_count": int(result.get("symbol_count", 0)),
        }

    def predict(
        self,
        *,
        date: str | None = None,
        out: str | None = None,
    ) -> dict[str, Any]:
        """Generate predictions while keeping recorder resolution at the service boundary."""

        recorder_identity = self.runtime_state.resolve_recorder_identity()
        result = modeling.generate_predictions(
            date=date,
            out=out,
            runtime_state=self.runtime_state,
            recorder_identity=recorder_identity,
        )
        self.history.record("predict", **self._predict_history_payload(result))
        return result

    def build_training_universe(
        self,
        *,
        start_year: int = 2010,
        end_year: int = 2026,
        top_n: int = 2200,
        random_seed: int = 42,
    ) -> dict[str, Any]:
        result = modeling.build_training_universe_file(
            start_year=start_year,
            end_year=end_year,
            top_n=top_n,
            random_seed=random_seed,
            data_path=self.settings.data_path,
            qlib_dir=self.settings.qlib_data_path,
            db_host=self.settings.db_host,
            db_port=self.settings.db_port,
        )
        self.history.record("filter_training_universe", **self._training_universe_history_payload(result))
        return result

    def build_portfolio(
        self,
        *,
        date: str | None = None,
        top_k: int = 80,
        max_weight: float = 0.02,
        rebalance_threshold: float = 0.002,
        buy_rank: int = modeling.HOLDING_BUFFER_DEFAULTS.buy_rank,
        hold_rank: int = modeling.HOLDING_BUFFER_DEFAULTS.hold_rank,
    ) -> dict[str, Any]:
        """Build portfolio artifacts while keeping run-history writes at the service boundary."""

        result = modeling.build_portfolio_outputs(
            date=date,
            top_k=top_k,
            max_weight=max_weight,
            rebalance_threshold=rebalance_threshold,
            buy_rank=buy_rank,
            hold_rank=hold_rank,
            track_run=False,
            runtime_state=self.runtime_state,
        )
        self.history.record("build_portfolio", **self._portfolio_history_payload(result))
        return result

    def train_model(self) -> dict[str, str]:
        result = _run_qlib_training(self.runtime_state)
        self.runtime_state.record_training_result(result)
        self.history.record("train_model", date=self._today_dash())
        return {"date": self._today_dash()}


def build_history_store(*, refresh_settings: bool = False) -> RunLogStore:
    settings = get_settings(refresh=refresh_settings)
    return RunLogStore(os.path.join(settings.data_path, "run_history.json"))


def build_data_service(*, refresh_settings: bool = False) -> DataPipelineService:
    settings = get_settings(refresh=refresh_settings)
    history = build_history_store(refresh_settings=False)
    return DataPipelineService(settings, history=history)


def build_model_service(*, refresh_settings: bool = False) -> ModelPipelineService:
    settings = get_settings(refresh=refresh_settings)
    history = build_history_store(refresh_settings=False)
    return ModelPipelineService(settings, history=history)


__all__ = [
    "DataPipelineService",
    "ModelPipelineService",
    "build_data_service",
    "build_history_store",
    "build_model_service",
]
