import qlib
import pandas as pd
from qlib.data import D
from qlib.workflow import R
from qlib.config import REG_CN
from qlib.utils import init_instance_by_config

# --- 配置部分 ---
PROVIDER_URI = r"C:/Users/sola/Documents/quant/.data/qlib_data"

def get_predict_conf(start_date, end_date):
    """构造预测配置"""
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

# --- 执行部分 (必须放在保护块内) ---
if __name__ == '__main__':
    # 1. 初始化 Qlib
    print(f"正在初始化 Qlib 数据路径: {PROVIDER_URI}")
    qlib.init(provider_uri=PROVIDER_URI, region=REG_CN)

    # 2. 计算日期
    # 注意：这里需要在 init 之后调用 D.calendar
    all_calendar = D.calendar(freq='day')
    latest_date = all_calendar[-1]
    # 往前取 180 天确保指标计算完整
    start_date_for_predict = all_calendar[-120]
    
    
    print(f"最新交易日: {latest_date}")
    print(f"数据预热起点: {start_date_for_predict}")

    # 3. 加载模型
    print("正在从 MLflow 加载模型...")
    try:
        recorder = R.get_recorder(
            recorder_id="6c6aaaec2fc4431eb78d5b17d709b348", 
            experiment_id="379677092195942384"
        )
        model = recorder.load_object("params.pkl")
    except Exception as e:
        print(f"模型加载失败，请检查 Recorder ID 是否正确: {e}")
        exit()

    # 4. 准备数据并预测
    print("正在计算 Alpha158 特征 (多进程执行中，请稍候)...")
    predict_dataset_conf = get_predict_conf(start_date_for_predict, latest_date)
    dataset = init_instance_by_config(predict_dataset_conf)
    
    pred_score = model.predict(dataset)

    # 5. 格式化输出结果
    print("\n" + "="*50)
    print(f"🚀 {latest_date} 选股预测 TOP 20")
    print("="*50)
    
    # 整理结果 DataFrame
    result_df = pred_score.sort_values(ascending=False).to_frame("Score")
    result_df.index = result_df.index.get_level_values('instrument') # 简化索引

    print(result_df)
    
    # 6. 保存结果
    output_filename = f"top_picks.csv"
    result_df.to_csv(output_filename)
    print(f"\n结果已保存至: {output_filename}")