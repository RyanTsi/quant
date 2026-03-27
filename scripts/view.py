"""Visualize Qlib experiment results: IC/IR analysis and portfolio returns.

Usage:
    python -m scripts.view
"""

import os
import pandas as pd
import qlib
from qlib.config import REG_CN
from qlib.workflow import R
from qlib.contrib.report import analysis_position, analysis_model

from config.settings import settings


def main():
    qlib.init(provider_uri=settings.qlib_provider_uri, region=REG_CN)
    R.set_uri(settings.qlib_mlruns_uri)

    recorder = R.get_exp(experiment_id=settings.qlib_experiment_id).get_recorder(
        recorder_id=settings.qlib_recorder_id
    )

    pred_df = recorder.load_object("pred.pkl")
    label_df = recorder.load_object("label.pkl")

    label_df.columns = ['label']
    pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)

    print("Generating IC/IR analysis charts...")
    figs = analysis_model.model_performance_graph(pred_label, show_notebook=False)

    output_dir = os.path.join(settings.analysis_path, settings.qlib_recorder_id)
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(figs, (list, tuple)):
        for i, f in enumerate(figs):
            if f is not None:
                f.write_html(os.path.join(output_dir, f"prediction_analysis_{i}.html"))
    elif figs is not None:
        figs.write_html(os.path.join(output_dir, "prediction_analysis.html"))

    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    figs = analysis_position.report_graph(report_normal_df, show_notebook=False)

    print("Saving portfolio return charts...")
    if isinstance(figs, tuple):
        for i, f in enumerate(figs):
            if f is not None:
                f.write_html(os.path.join(output_dir, f"portfolio_returns_{i}.html"))
    elif figs is not None:
        figs.write_html(os.path.join(output_dir, "portfolio_returns.html"))

    print(f"All charts saved to {output_dir}")


if __name__ == "__main__":
    main()
