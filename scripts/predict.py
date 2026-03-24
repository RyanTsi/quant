"""Generate stock predictions using the latest trained model.

Usage:
    python -m scripts.predict
"""

import qlib
import pandas as pd
from qlib.data import D
from qlib.workflow import R
from qlib.config import REG_CN
from qlib.utils import init_instance_by_config

from config.settings import settings


def get_predict_conf(start_date, end_date):
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
                    "instruments": "top_500_liquidity_stocks",
                    "infer_processors": [
                        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                        {"class": "Fillna", "kwargs": {"fields_group": "feature"}}
                    ],
                    "learn_processors": [
                        {"class": "DropnaLabel"},
                        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}
                    ],
                    "label": ["(Ref($close, -2)/Ref($close, -1) - 1 + Ref($close, -3)/Ref($close, -1) - 1 + Ref($close, -4)/Ref($close, -1) - 1 + Ref($close, -5)/Ref($close, -1) - 1 + Ref($close, -6)/Ref($close, -1) - 1) / 5"]
                },
            },
            "segments": {
                "test": [end_date, end_date],
            },
            "step_len": 20,
        },
    }


if __name__ == '__main__':
    qlib.init(provider_uri=settings.qlib_provider_uri, region=REG_CN)

    all_calendar = D.calendar(freq='day')
    latest_date = all_calendar[-1]
    # ~120 trading days lookback for Alpha158 indicator warm-up
    start_date_for_predict = all_calendar[-120]

    print(f"Latest trading day: {latest_date}")
    print(f"Lookback start:     {start_date_for_predict}")

    print("Loading model from MLflow...")
    try:
        recorder = R.get_recorder(
            recorder_id=settings.qlib_recorder_id,
            experiment_id=settings.qlib_experiment_id,
        )
        model = recorder.load_object("params.pkl")
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit(1)

    print("Computing Alpha158 features ...")
    predict_dataset_conf = get_predict_conf(start_date_for_predict, latest_date)
    dataset = init_instance_by_config(predict_dataset_conf)

    pred_score = model.predict(dataset)

    print("\n" + "=" * 50)
    print(f"{latest_date} Top Predictions")
    print("=" * 50)

    result_df = pred_score.sort_values(ascending=False).to_frame("Score")
    result_df.index = result_df.index.get_level_values('instrument')

    print(result_df)

    output_filename = "top_picks.csv"
    result_df.to_csv(output_filename)
    print(f"\nSaved to: {output_filename}")
