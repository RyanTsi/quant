"""Visualize Qlib experiment results: IC/IR analysis and portfolio returns.

Usage:
    python -m scripts.view
    python -m scripts.view --experiment_id <id> --recorder_id <id>
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

import pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.contrib.report import analysis_model, analysis_position
from qlib.workflow import R

from runtime.config import get_settings
from runtime.runlog import get_last_run

settings = get_settings()


def _resolve_recorder_ids(
    experiment_id: str | None = None,
    recorder_id: str | None = None,
) -> Tuple[str, str]:
    if experiment_id and recorder_id:
        return str(experiment_id), str(recorder_id)

    env_exp = os.getenv("QLIB_EXPERIMENT_ID")
    env_rec = os.getenv("QLIB_RECORDER_ID")
    if env_exp and env_rec:
        return str(env_exp), str(env_rec)

    last = get_last_run("qlib_train") or {}
    last_exp = last.get("experiment_id")
    last_rec = last.get("recorder_id")
    if last_exp and last_rec:
        return str(last_exp), str(last_rec)

    if settings.qlib_experiment_id and settings.qlib_recorder_id:
        return str(settings.qlib_experiment_id), str(settings.qlib_recorder_id)

    raise RuntimeError("Cannot resolve experiment/recorder ids for visualization.")


def _write_html_figs(figs, output_dir: str, prefix: str) -> int:
    written = 0
    if isinstance(figs, (list, tuple)):
        for i, f in enumerate(figs):
            if f is None:
                continue
            f.write_html(os.path.join(output_dir, f"{prefix}_{i}.html"))
            written += 1
    elif figs is not None:
        figs.write_html(os.path.join(output_dir, f"{prefix}.html"))
        written += 1
    return written


def generate_view(
    *,
    experiment_id: str | None = None,
    recorder_id: str | None = None,
    provider_uri: str | None = None,
    mlruns_uri: str | None = None,
) -> str:
    qlib.init(provider_uri=provider_uri or settings.qlib_provider_uri, region=REG_CN)
    R.set_uri(mlruns_uri or settings.qlib_mlruns_uri)

    exp_id, rec_id = _resolve_recorder_ids(experiment_id=experiment_id, recorder_id=recorder_id)
    recorder = R.get_recorder(experiment_id=exp_id, recorder_id=rec_id)

    pred_df = recorder.load_object("pred.pkl")
    label_df = recorder.load_object("label.pkl")
    label_df.columns = ["label"]
    pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)

    output_dir = os.path.join(settings.analysis_path, rec_id)
    os.makedirs(output_dir, exist_ok=True)

    print("Generating IC/IR analysis charts...")
    model_figs = analysis_model.model_performance_graph(pred_label, show_notebook=False)
    _write_html_figs(model_figs, output_dir, "prediction_analysis")

    try:
        report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        print("Saving portfolio return charts...")
        port_figs = analysis_position.report_graph(report_normal_df, show_notebook=False)
        _write_html_figs(port_figs, output_dir, "portfolio_returns")
    except Exception as exc:
        # Some lightweight configs skip PortAnaRecord. Keep view runnable with model charts.
        print(f"Portfolio analysis artifact not available, skipped: {exc}")

    print(f"All charts saved to {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Qlib run artifacts")
    parser.add_argument("--experiment_id", type=str, default=None, help="Qlib experiment id")
    parser.add_argument("--recorder_id", type=str, default=None, help="Qlib recorder id")
    args = parser.parse_args()

    generate_view(
        experiment_id=args.experiment_id,
        recorder_id=args.recorder_id,
    )


if __name__ == "__main__":
    main()
