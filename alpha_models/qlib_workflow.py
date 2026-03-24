import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord, PortAnaRecord
from config.settings import settings
from utils.run_tracker import record_run, today_dash


PROVIDER_URI = settings.qlib_provider_uri
MARKET = "top_500_liquidity_stocks"
BENCHMARK = "SH000001"
SEED = 10059483
TODAY = today_dash()

data_handler_config = {
    "start_time": "2010-01-01",
    "end_time": TODAY,
    "fit_start_time": "2010-01-01",
    "fit_end_time": "2025-12-31",
    "instruments": MARKET,
    "infer_processors": [
        {
            "class": "RobustZScoreNorm",
            "kwargs": {"fields_group": "feature", "clip_outlier": True},
        },
        {
            "class": "Fillna",
            "kwargs": {"fields_group": "feature"},
        },
    ],
    "learn_processors": [
        {"class": "DropnaLabel"},
        {
            "class": "CSRankNorm",
            "kwargs": {"fields_group": "label"},
        },
    ],
    "label": [
        "(Ref($close, -2)/Ref($close, -1) - 1"
        " + Ref($close, -3)/Ref($close, -1) - 1"
        " + Ref($close, -4)/Ref($close, -1) - 1"
        " + Ref($close, -5)/Ref($close, -1) - 1"
        " + Ref($close, -6)/Ref($close, -1) - 1) / 5"
    ],
}

port_analysis_config = {
    "strategy": {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {
            "signal": "<PRED>",
            "topk": 50,
            "n_drop": 5,
        },
    },
    "backtest": {
        "start_time": "2026-03-01",
        "end_time": TODAY,
        "account": 100000000,
        "benchmark": BENCHMARK,
        "exchange_kwargs": {
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    },
}

task_config = {
    "model": {
        "class": "TransformerModel",
        "module_path": "qlib.contrib.model.pytorch_transformer_ts",
        "kwargs": {
            "seed": SEED,
            "n_jobs": 5,
            "batch_size": 4096,
            "d_feat": 158,
            "d_model": 128,
            "n_head": 8,
            "num_layers": 4,
            "dropout": 0.1,
        },
    },
    "dataset": {
        "class": "TSDatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": ["2010-01-01", "2025-12-31"],
                "valid": ["2026-01-01", "2026-02-28"],
                "test": ["2026-03-01", TODAY],
            },
            "step_len": 20,
        },
    },
}


def main():
    qlib.init(provider_uri=PROVIDER_URI, region=REG_CN)

    model = init_instance_by_config(task_config["model"])
    dataset = init_instance_by_config(task_config["dataset"])

    with R.start(experiment_name="transformer_alpha158"):
        R.log_params(**flatten_dict(task_config))

        model.fit(dataset)

        recorder = R.get_recorder()

        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        sar = SigAnaRecord(recorder, ana_long_short=False, ann_scaler=252)
        sar.generate()

        par = PortAnaRecord(recorder, config=port_analysis_config)
        par.generate()

        experiment_id = R.get_recorder().experiment_id
        recorder_id = R.get_recorder().id
        print(f"Experiment ID: {experiment_id}")
        print(f"Recorder  ID: {recorder_id}")

    record_run("qlib_train", end_date=TODAY, experiment_id=experiment_id, recorder_id=recorder_id)
    print(f"Run recorded: qlib_train -> {TODAY}")


if __name__ == "__main__":
    main()
