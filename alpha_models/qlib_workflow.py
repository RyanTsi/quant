import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData


if __name__ == "__main__":
    # use default data
    provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    market = "csi300"
    benchmark = "SH000300"

    data_handler_config = {
        "start_time": "2010-01-01",
        "end_time": "2026-02-13",
        "fit_start_time": "2010-01-01",
        "fit_end_time": "2019-12-31",
        "instruments": market,
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
        "label": ["Ref($close, -2) / Ref($close, -1) - 1"],
    }

    task_config = {
        "model": {
            "class": "TransformerModel",
            "module_path": "qlib.contrib.model.pytorch_transformer",
            "kwargs": {
                "d_feat": 6,
                "seed": 11150,
            },
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha360",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": data_handler_config,
                },
                "segments": {
                    "train": ["2010-01-01", "2019-12-31"],
                    "valid": ["2020-01-01", "2021-12-31"],
                    "test": ["2022-01-01", "2026-02-13"],
                },
            },
        },
    }

    model = init_instance_by_config(task_config["model"])
    dataset = init_instance_by_config(task_config["dataset"])

    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {
                "signal": (model, dataset),
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "start_time": "2022-01-01",
            "end_time": "2026-02-13",
            "account": 100000000,
            "benchmark": benchmark,
            "exchange_kwargs": {
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }

    with R.start(experiment_name="transformer_workflow"):
        # 记录参数
        R.log_params(**flatten_dict(task_config))
        
        # 训练模型
        print("Training Transformer Model...")
        model.fit(dataset)
        R.save_objects(**{"params.pkl": model})

        # 生成预测信号并记录 (对应 YAML 的 SignalRecord, <MODEL> 和 <DATASET>)
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # 信号分析 (对应 YAML 的 SigAnaRecord，附带自定义参数)
        sar = SigAnaRecord(recorder, ana_long_short=False, ann_scaler=252)
        sar.generate()

        # 回测及组合分析 (对应 YAML 的 PortAnaRecord)
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()