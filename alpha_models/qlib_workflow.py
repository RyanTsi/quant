import copy
import pandas as pd

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.tests.data import GetData
from qlib.backtest import backtest, executor
from qlib.contrib.strategy import TopkDropoutStrategy

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

    base_task_config = {
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

    test_years = [2022, 2023, 2024, 2025]
    all_predictions = []
    all_labels = []
    for target_year in test_years:
        print(f"\n{'='*40}")
        print(f"🚀 开始滚动训练，目标预测年份: {target_year}")
        print(f"{'='*40}")
        
        train_start = f"{target_year - 12}-01-01"
        train_end = f"{target_year - 3}-12-31"
        valid_start = f"{target_year - 2}-01-01"
        valid_end = f"{target_year - 1}-12-31"
        test_start = f"{target_year}-01-01"
        test_end = f"{target_year}-12-31"

        print(f"  > 训练集: {train_start} 至 {train_end}")
        print(f"  > 验证集: {valid_start} 至 {valid_end}")
        print(f"  > 测试集: {test_start} 至 {test_end}")

        current_task_config = copy.deepcopy(base_task_config)
        current_task_config["dataset"]["kwargs"]["handler"]["kwargs"]["fit_start_time"] = train_start
        current_task_config["dataset"]["kwargs"]["handler"]["kwargs"]["fit_end_time"] = train_end
        current_task_config["dataset"]["kwargs"]["segments"] = {
            "train": [train_start, train_end],
            "valid": [valid_start, valid_end],
            "test": [test_start, test_end],
        }
        model = init_instance_by_config(current_task_config["model"])
        dataset = init_instance_by_config(current_task_config["dataset"])

        model.fit(dataset)
        all_predictions.append(model.predict(dataset))
        label_df = dataset.prepare("test", col_set="label")
        all_labels.append(label_df)

    final_pred_series = pd.concat(all_predictions)
    final_pred_df = final_pred_series.to_frame("score")

    strategy = TopkDropoutStrategy(
        signal=final_pred_df,
        topk=50,
        n_drop=5
    )

    trade_executor = executor.SimulatorExecutor(
        time_per_step="day",
        generate_portfolio_metrics=True
    )

    portfolio_metric_dict, indicator_dict = backtest(
        start_time="2022-01-01", 
        end_time="2026-02-13",
        strategy=strategy,
        executor=trade_executor,
        account=100000000,
        benchmark=benchmark,
        exchange_kwargs={
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        }
    )

    # ==========================================
    # 🕵️ 核心信号分析 (Alpha 质量检验)
    # ==========================================
    print("\n" + "="*40)
    print("🔬 开始进行信号质量分析 (IC / IR)...")
    
    # 1. 拼接所有的真实标签
    final_label_df = pd.concat(all_labels)
    
    # 为了防止列名带来的困扰，我们统一取第一列的数据(Series)
    pred_s = final_pred_df.iloc[:, 0]
    label_s = final_label_df.iloc[:, 0]

    # 2. 调用 Qlib 底层的 IC 计算工具
    from qlib.contrib.evaluate import calc_ic
    
    # calc_ic 会按天计算预测分和真实收益的相关系数，返回 每日IC 和 每日Rank IC
    ic_series, rank_ic_series = calc_ic(pred_s, label_s)

    # 3. 计算均值和信息比率 (IR = IC均值 / IC标准差)
    # 乘以 252 的平方根 (约15.87) 是为了将 IR 年化，这是业界的标准做法
    import numpy as np
    ann_scaler = np.sqrt(252) 
    
    ic_mean = ic_series.mean()
    ic_ir = (ic_mean / ic_series.std()) * ann_scaler
    
    ric_mean = rank_ic_series.mean()
    ric_ir = (ric_mean / rank_ic_series.std()) * ann_scaler

    print(f"📌 Normal IC (皮尔逊相关系数): {ic_mean:.4f}")
    print(f"📌 Normal IR (年化IC信息比率): {ic_ir:.4f}")
    print(f"⭐ Rank IC (斯皮尔曼秩相关系数): {ric_mean:.4f}")
    print(f"⭐ Rank IR (年化Rank IC信息比率): {ric_ir:.4f}")
    
    # 保存每天的 Rank IC 到文件，方便以后画图排查哪天模型失效了
    rank_ic_series.to_csv("my_wfo_daily_rank_ic.csv")
    print("✅ 每日 Rank IC 已保存至 my_wfo_daily_rank_ic.csv\n")

    # ==========================================
    # 📝 打印核心回测指标
    # ==========================================
    print("\n" + "="*40)
    print("💰 回测资金表现 (Portfolio Metrics)")
    print("="*40)

    daily_indicators = indicator_dict.get('1day')[0] 
    
    print(f"📈 年化收益率 (Annual Return): {daily_indicators.loc['annualized_return'].values[0] * 100:.2f}%")
    print(f"📉 最大回撤 (Max Drawdown): {daily_indicators.loc['max_drawdown'].values[0] * 100:.2f}%")
    print(f"📊 夏普比率 (Sharpe Ratio): {daily_indicators.loc['information_ratio'].values[0]:.2f}") 
    # 注：在Qlib的纯多头组合中，information_ratio 通常代表扣除无风险利率后的夏普
    
    # 将完整的每日资金净值曲线保存下来，供后续画图分析
    report_normal_df = portfolio_metric_dict.get('1day')[0]
    report_normal_df.to_csv("my_wfo_portfolio_report.csv")
    print("\n✅ 回测完成！详细每日资金曲线已保存至 my_wfo_portfolio_report.csv")