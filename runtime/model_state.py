from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from model_function.qlib import (
    RecorderIdentity,
    build_training_run_payload,
    resolve_recorder_identity as resolve_model_recorder_identity,
    resolve_training_config_source,
)
from runtime.config import AppSettings, get_settings
from runtime.runlog import RunLogStore

NO_TRAINED_MODEL_ERROR = "No trained model found. Set QLIB_EXPERIMENT_ID/QLIB_RECORDER_ID or run `python main.py --run train` first."


@dataclass(frozen=True)
class TrainingWorkflowInputs:
    """Resolved runtime-owned inputs for a Qlib training or evaluation workflow."""

    config_source: str
    provider_uri: str
    mlruns_uri: str
    experiment_name: str


@dataclass(frozen=True)
class ModelRuntimeState:
    """Runtime-owned settings and run-history access for model pipeline flows."""

    settings: AppSettings
    history: RunLogStore

    def resolve_training_workflow_inputs(
        self,
        *,
        config_source: str | None = None,
    ) -> TrainingWorkflowInputs:
        """Resolve workflow config and storage inputs from the runtime settings boundary."""

        return TrainingWorkflowInputs(
            config_source=resolve_training_config_source(
                config_source,
                fallback_config_source=self.settings.qlib_workflow_config,
            ),
            provider_uri=self.settings.qlib_provider_uri,
            mlruns_uri=self.settings.qlib_mlruns_uri,
            experiment_name=self.settings.qlib_experiment_name,
        )

    def resolve_recorder_identity(
        self,
        *,
        experiment_id: str | None = None,
        recorder_id: str | None = None,
        env: Mapping[str, str] | None = None,
        allow_settings_fallback: bool = False,
        missing_error_message: str = NO_TRAINED_MODEL_ERROR,
    ) -> RecorderIdentity:
        """Resolve recorder coordinates with strict-by-default runtime-owned fallback policy."""

        try:
            return resolve_model_recorder_identity(
                experiment_id=experiment_id,
                recorder_id=recorder_id,
                env=env if env is not None else os.environ,
                runlog_entry=self.history.get("qlib_train") or {},
                fallback_experiment_id=self.settings.qlib_experiment_id if allow_settings_fallback else None,
                fallback_recorder_id=self.settings.qlib_recorder_id if allow_settings_fallback else None,
            )
        except RuntimeError as exc:
            if allow_settings_fallback:
                raise
            raise RuntimeError(missing_error_message) from exc

    def record_training_result(self, result: Any) -> dict[str, Any]:
        """Persist the canonical `qlib_train` payload for a successful training run."""

        return self.history.record("qlib_train", **build_training_run_payload(result))


def build_model_runtime_state(
    *,
    settings: AppSettings | None = None,
    history: RunLogStore | None = None,
    refresh_settings: bool = False,
) -> ModelRuntimeState:
    """Build the runtime-owned model state helper for wrappers, services, and adapters."""

    resolved_settings = settings or get_settings(refresh=refresh_settings)
    resolved_history = history or RunLogStore(os.path.join(resolved_settings.data_path, "run_history.json"))
    return ModelRuntimeState(settings=resolved_settings, history=resolved_history)


__all__ = [
    "ModelRuntimeState",
    "NO_TRAINED_MODEL_ERROR",
    "TrainingWorkflowInputs",
    "build_model_runtime_state",
]
