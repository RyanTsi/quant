"""Evaluate a trained Qlib model on the full test segment.

Usage:
  python -m scripts.eval_test
  python -m scripts.eval_test --config alpha_models/workflow_config_transformer_Alpha158.yaml

Model selection:
  - If env QLIB_EXPERIMENT_ID + QLIB_RECORDER_ID are set, use them.
  - Else fallback to `.data/run_history.json` -> `qlib_train`.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import qlib
from qlib.config import REG_CN
from qlib.contrib.eva.alpha import calc_ic
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord

from alpha_models.workflow.runner import QlibWorkflowRunner
from runtime.config import get_settings
from runtime.runlog import get_last_run

settings = get_settings()


def _resolve_recorder_ids() -> Tuple[str, str]:
    env_recorder_id = os.getenv("QLIB_RECORDER_ID")
    env_experiment_id = os.getenv("QLIB_EXPERIMENT_ID")
    if env_recorder_id and env_experiment_id:
        return env_experiment_id, env_recorder_id

    last = get_last_run("qlib_train") or {}
    exp_id = last.get("experiment_id")
    rec_id = last.get("recorder_id")
    if not exp_id or not rec_id:
        raise RuntimeError(
            "No trained model found. Set QLIB_EXPERIMENT_ID/QLIB_RECORDER_ID or run `python main.py --run train` first."
        )
    return str(exp_id), str(rec_id)


def _load_trained_model(recorder) -> Any:
    # Prefer the object we save in workflow (`trained_model`), then fallback to historical names.
    for key in ("trained_model", "trained_model.pkl", "params.pkl"):
        try:
            return recorder.load_object(key)
        except Exception:
            continue
    raise RuntimeError("Failed to load trained model object from recorder artifacts.")


def _calc_metrics(pred, label) -> Dict[str, float]:
    ic, ric = calc_ic(pred.iloc[:, 0], label.iloc[:, 0])
    ic_mean = float(np.nanmean(ic))
    ic_std = float(np.nanstd(ic))
    ric_mean = float(np.nanmean(ric))
    ric_std = float(np.nanstd(ric))
    return {
        "IC": ic_mean,
        "ICIR": (ic_mean / ic_std) if ic_std else float("nan"),
        "Rank IC": ric_mean,
        "Rank ICIR": (ric_mean / ric_std) if ric_std else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained Qlib model on full test segment")
    parser.add_argument(
        "--config",
        type=str,
        default=settings.qlib_workflow_config,
        help="Workflow YAML config path (single file).",
    )
    args = parser.parse_args()

    qlib.init(provider_uri=settings.qlib_provider_uri, region=REG_CN)
    if settings.qlib_mlruns_uri:
        R.set_uri(settings.qlib_mlruns_uri)

    exp_id, rec_id = _resolve_recorder_ids()
    recorder = R.get_recorder(experiment_id=exp_id, recorder_id=rec_id)
    model = _load_trained_model(recorder)

    loaded = QlibWorkflowRunner.load_yaml_config(args.config)
    dataset_conf: Dict[str, Any] = dict(loaded.task["dataset"])
    dataset = init_instance_by_config(dataset_conf)

    pred = model.predict(dataset)
    if hasattr(pred, "to_frame"):
        pred = pred.to_frame("score")
    label = SignalRecord.generate_label(dataset)

    metrics = _calc_metrics(pred, label)
    print("\n== Full test segment metrics ==")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
