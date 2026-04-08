from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from alpha_models.workflow.runner import QlibWorkflowRunner, TrainResult
from utils.preprocess import ALPHA158_WEIGHTED_5D_LABEL


DEFAULT_WORKFLOW_CONFIG_PATH = Path("alpha_models/workflow_config_transformer_Alpha158.yaml")
TRAINED_MODEL_OBJECT_KEYS = ("trained_model", "trained_model.pkl", "params.pkl")


@dataclass(frozen=True)
class RecorderIdentity:
    """Resolved experiment/recorder coordinates for Qlib artifact access."""

    experiment_id: str
    recorder_id: str


def default_training_config_path() -> str:
    """Return the repository-default Qlib workflow config path."""

    return str(DEFAULT_WORKFLOW_CONFIG_PATH)


def resolve_training_config_source(
    config_source: str | None = None,
    *,
    fallback_config_source: str | None = None,
) -> str:
    """Resolve the workflow config source from explicit, wrapper-provided fallback, then repo default."""

    if config_source:
        return str(config_source)
    if fallback_config_source:
        return str(fallback_config_source)
    return default_training_config_path()


def run_training_workflow(
    *,
    config_source: str | None = None,
    fallback_config_source: str | None = None,
    provider_uri: str | None = None,
    default_provider_uri: str | None = None,
    mlruns_uri: str | None = None,
    default_mlruns_uri: str | None = None,
    experiment_name: str | None = None,
    default_experiment_name: str | None = None,
    runner: QlibWorkflowRunner | None = None,
) -> TrainResult:
    """Run the low-level workflow runner with caller-supplied defaults."""

    # Importing the model module registers the Transformer class for YAML-based instantiation.
    import qlib.contrib.model.pytorch_transformer_ts  # noqa: F401

    workflow_runner = runner or QlibWorkflowRunner()
    resolved_config_source = resolve_training_config_source(
        config_source,
        fallback_config_source=fallback_config_source,
    )
    return workflow_runner.run_from_yaml(
        config_source=resolved_config_source,
        provider_uri_override=provider_uri if provider_uri is not None else default_provider_uri,
        mlruns_uri=mlruns_uri if mlruns_uri is not None else default_mlruns_uri,
        experiment_name=experiment_name if experiment_name is not None else default_experiment_name,
    )


def build_training_run_payload(result: TrainResult) -> dict[str, Any]:
    """Convert a workflow result into the stable runlog payload used by training flows."""

    payload = {
        "config_source": result.config_source,
        "experiment_id": result.experiment_id,
        "recorder_id": result.recorder_id,
    }
    if result.metrics is None:
        return payload

    payload.update(
        {
            "IC": result.metrics.ic,
            "ICIR": result.metrics.icir,
            "Rank IC": result.metrics.rank_ic,
            "Rank ICIR": result.metrics.rank_icir,
        }
    )
    return payload


def resolve_recorder_identity(
    *,
    experiment_id: str | None = None,
    recorder_id: str | None = None,
    env: Mapping[str, str] | None = None,
    runlog_entry: Mapping[str, Any] | None = None,
    fallback_experiment_id: str | None = None,
    fallback_recorder_id: str | None = None,
) -> RecorderIdentity:
    """Resolve ids from explicit args, env, runlog payload, then wrapper-provided fallback ids."""

    if experiment_id and recorder_id:
        return RecorderIdentity(experiment_id=str(experiment_id), recorder_id=str(recorder_id))

    env_mapping = env if env is not None else os.environ
    env_experiment_id = env_mapping.get("QLIB_EXPERIMENT_ID")
    env_recorder_id = env_mapping.get("QLIB_RECORDER_ID")
    if env_experiment_id and env_recorder_id:
        return RecorderIdentity(experiment_id=str(env_experiment_id), recorder_id=str(env_recorder_id))

    runlog_payload = runlog_entry or {}
    runlog_experiment_id = runlog_payload.get("experiment_id")
    runlog_recorder_id = runlog_payload.get("recorder_id")
    if runlog_experiment_id and runlog_recorder_id:
        return RecorderIdentity(experiment_id=str(runlog_experiment_id), recorder_id=str(runlog_recorder_id))

    if fallback_experiment_id and fallback_recorder_id:
        return RecorderIdentity(
            experiment_id=str(fallback_experiment_id),
            recorder_id=str(fallback_recorder_id),
        )

    raise RuntimeError("Cannot resolve experiment/recorder ids for Qlib artifacts.")


def resolve_recorder_ids(
    *,
    experiment_id: str | None = None,
    recorder_id: str | None = None,
    env: Mapping[str, str] | None = None,
    runlog_entry: Mapping[str, Any] | None = None,
    fallback_experiment_id: str | None = None,
    fallback_recorder_id: str | None = None,
) -> tuple[str, str]:
    """Return resolved ids in `(experiment_id, recorder_id)` order."""

    identity = resolve_recorder_identity(
        experiment_id=experiment_id,
        recorder_id=recorder_id,
        env=env,
        runlog_entry=runlog_entry,
        fallback_experiment_id=fallback_experiment_id,
        fallback_recorder_id=fallback_recorder_id,
    )
    return identity.experiment_id, identity.recorder_id


def load_trained_model(recorder: Any, artifact_keys: Sequence[str] = TRAINED_MODEL_OBJECT_KEYS) -> Any:
    """Load the trained model artifact, preserving the historical fallback order."""

    for artifact_key in artifact_keys:
        try:
            return recorder.load_object(artifact_key)
        except Exception:
            continue
    raise RuntimeError("Failed to load trained model object from recorder artifacts.")


def build_alpha158_prediction_dataset_config(
    start_date: str,
    end_date: str,
    instruments: Sequence[str],
) -> dict[str, Any]:
    """Build the Alpha158/TSDatasetH scoring dataset config used by runtime prediction."""

    # Prediction scoring must mirror training feature/label semantics:
    # 1. keep the Alpha158 handler
    # 2. reuse the weighted 5-day forward-return label
    # 3. score only the requested instruments on the target segment
    return {
        "class": "TSDatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": start_date,
                    "end_time": end_date,
                    "fit_start_time": start_date,
                    "fit_end_time": end_date,
                    "instruments": list(instruments),
                    "infer_processors": [
                        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                        {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                    ],
                    "learn_processors": [
                        {"class": "DropnaLabel"},
                        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
                    ],
                    "label": [ALPHA158_WEIGHTED_5D_LABEL],
                },
            },
            "segments": {"test": [end_date, end_date]},
            "step_len": 20,
        },
    }


def _write_html_figures(figures: Any, output_dir: str, prefix: str) -> int:
    """Write Plotly figures to disk while tolerating missing optional charts."""

    written = 0
    if isinstance(figures, (list, tuple)):
        for index, figure in enumerate(figures):
            if figure is None:
                continue
            figure.write_html(os.path.join(output_dir, f"{prefix}_{index}.html"))
            written += 1
        return written

    if figures is None:
        return written

    figures.write_html(os.path.join(output_dir, f"{prefix}.html"))
    return written + 1


def _build_pred_label_frame(recorder: Any) -> pd.DataFrame:
    """Load and align prediction and label artifacts for Qlib report generation."""

    pred_df = recorder.load_object("pred.pkl")
    label_df = recorder.load_object("label.pkl")
    label_df.columns = ["label"]
    return pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)


def generate_analysis_view(
    *,
    identity: RecorderIdentity,
    provider_uri: str,
    analysis_path: str | Path,
    mlruns_uri: str | None = None,
) -> str:
    """Generate model and portfolio analysis HTML outputs for a resolved recorder."""

    import qlib
    from qlib.config import REG_CN
    from qlib.contrib.report import analysis_model, analysis_position
    from qlib.workflow import R

    qlib.init(provider_uri=provider_uri, region=REG_CN)
    if mlruns_uri:
        R.set_uri(mlruns_uri)

    recorder = R.get_recorder(
        experiment_id=identity.experiment_id,
        recorder_id=identity.recorder_id,
    )
    pred_label = _build_pred_label_frame(recorder)

    output_dir = os.path.join(str(analysis_path), identity.recorder_id)
    os.makedirs(output_dir, exist_ok=True)

    print("Generating IC/IR analysis charts...")
    model_figures = analysis_model.model_performance_graph(pred_label, show_notebook=False)
    _write_html_figures(model_figures, output_dir, "prediction_analysis")

    try:
        report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        print("Saving portfolio return charts...")
        portfolio_figures = analysis_position.report_graph(report_normal_df, show_notebook=False)
        _write_html_figures(portfolio_figures, output_dir, "portfolio_returns")
    except Exception as exc:
        print(f"Portfolio analysis artifact not available, skipped: {exc}")

    print(f"All charts saved to {output_dir}")
    return output_dir


def calculate_signal_metrics(pred: pd.DataFrame, label: pd.DataFrame) -> dict[str, float]:
    """Calculate IC/Rank-IC summary metrics for a prediction-vs-label pair."""

    from qlib.contrib.eva.alpha import calc_ic

    ic, rank_ic = calc_ic(pred.iloc[:, 0], label.iloc[:, 0])
    ic_mean = float(np.nanmean(ic))
    ic_std = float(np.nanstd(ic))
    rank_ic_mean = float(np.nanmean(rank_ic))
    rank_ic_std = float(np.nanstd(rank_ic))
    return {
        "IC": ic_mean,
        "ICIR": (ic_mean / ic_std) if ic_std else float("nan"),
        "Rank IC": rank_ic_mean,
        "Rank ICIR": (rank_ic_mean / rank_ic_std) if rank_ic_std else float("nan"),
    }


def _normalize_prediction_frame(prediction: Any) -> pd.DataFrame:
    """Normalize model predictions into the DataFrame shape expected by evaluation helpers."""

    if hasattr(prediction, "to_frame"):
        return prediction.to_frame("score")
    if isinstance(prediction, pd.DataFrame):
        return prediction
    return pd.DataFrame(prediction)


def evaluate_test_predictions(
    *,
    config_source: str,
    identity: RecorderIdentity,
    provider_uri: str,
    mlruns_uri: str | None = None,
) -> dict[str, float]:
    """Evaluate the resolved trained model against the full configured test segment."""

    import qlib
    from qlib.config import REG_CN
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R
    from qlib.workflow.record_temp import SignalRecord

    qlib.init(provider_uri=provider_uri, region=REG_CN)
    if mlruns_uri:
        R.set_uri(mlruns_uri)

    recorder = R.get_recorder(
        experiment_id=identity.experiment_id,
        recorder_id=identity.recorder_id,
    )
    model = load_trained_model(recorder)

    loaded = QlibWorkflowRunner.load_yaml_config(config_source)
    dataset_conf = dict(loaded.task["dataset"])
    dataset = init_instance_by_config(dataset_conf)

    pred = _normalize_prediction_frame(model.predict(dataset))
    label = SignalRecord.generate_label(dataset)
    return calculate_signal_metrics(pred, label)


__all__ = [
    "DEFAULT_WORKFLOW_CONFIG_PATH",
    "RecorderIdentity",
    "TRAINED_MODEL_OBJECT_KEYS",
    "build_alpha158_prediction_dataset_config",
    "build_training_run_payload",
    "calculate_signal_metrics",
    "default_training_config_path",
    "evaluate_test_predictions",
    "generate_analysis_view",
    "load_trained_model",
    "resolve_recorder_identity",
    "resolve_recorder_ids",
    "resolve_training_config_source",
    "run_training_workflow",
]
