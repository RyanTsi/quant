import pickle
import pandas as pd
import numpy as np

# 文件路径 (确保和你的脚本在同一目录下，或者写绝对路径)
FILE_PATH = "train_data_v4.pkl"

def inspect_data():
    print(f"📂 正在加载 {FILE_PATH} ...")
    
    try:
        with open(FILE_PATH, "rb") as f:
            data_list = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ 找不到文件: {FILE_PATH}")
        return

    # 1. 检查整体结构
    print(f"\n=== 1. 整体结构 ===")
    print(f"数据类型: {type(data_list)}")
    print(f"列表长度 (股票数量): {len(data_list)}")
    
    if len(data_list) == 0:
        print("⚠️ 警告: 列表为空！之前的 get_data_with_cache 可能没下载到任何数据。")
        return

    # 2. 检查大盘指数 (通常是第0个)
    print(f"\n=== 2. 大盘指数 (Index 0) ===")
    index_df = data_list[0]
    analyze_dataframe(index_df, "指数/大盘")

    # 3. 检查第一只个股 (通常是第1个)
    if len(data_list) > 1:
        print(f"\n=== 3. 随机个股样本 (Index 1) ===")
        stock_df = data_list[1]
        analyze_dataframe(stock_df, "个股样本")
    else:
        print("\n⚠️ 警告: 只有指数数据，没有个股数据！")

def analyze_dataframe(df, name):
    """详细分析单个 DataFrame"""
    print(f"[{name}] 类型: {type(df)}")
    
    if not isinstance(df, pd.DataFrame):
        print(f"❌ 错误: 数据不是 DataFrame，而是 {type(df)}")
        return

    print(f"[{name}] 形状 (Rows, Cols): {df.shape}")
    print(f"[{name}] 列名: {list(df.columns)}")
    
    # 检查索引是否为时间
    is_time_index = isinstance(df.index, pd.DatetimeIndex)
    print(f"[{name}] Index是否为时间格式: {is_time_index}")
    
    if len(df) > 0:
        start_date = df.index.min()
        end_date = df.index.max()
        print(f"[{name}] 时间范围: {start_date} -> {end_date}")
        print(f"[{name}] ❌ 原始行数: {len(df)}")
        
        # 关键诊断：判断是否满足你的环境要求
        # 你的环境要求：Window(60) + Training(252) + Buffer(20) = 332
        required = 332
        if len(df) < required:
            print(f"⚠️ [关键问题] 行数不足！现有 {len(df)} < 需要 {required}。这会导致被环境丢弃。")
        else:
            print(f"✅ [通过] 行数充足 ({len(df)} > {required})。")
            
        print(f"[{name}] 头部数据预览:\n{df.head(3)}")
        print(f"[{name}] 尾部数据预览:\n{df.tail(3)}")
    else:
        print(f"⚠️ [关键问题] DataFrame 是空的！")

if __name__ == "__main__":
    inspect_data()