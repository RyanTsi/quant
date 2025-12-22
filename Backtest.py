import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from stable_baselines3 import SAC
from database.influx_manager import InfluxDBManager, InfluxDBConfig, InfluxDBCallbacks
from config import *

def backtest_from_db(db_manager, model_path, stock_code, start_date, end_date):
    # 1. 关键修改：多拉取 100 天的数据作为“缓冲区”，以计算第一天的 90 维历史涨幅
    fetch_start = start_date - timedelta(days=150) # 考虑到非交易日，多留点余量
    print(f"📡 正在从 InfluxDB 提取 {stock_code} 的历史数据 (包含缓冲区)...")
    
    df = db_manager.get_stock_data_by_range(stock_code, fetch_start, end_date)
    
    if df is None or df.empty:
        print("❌ 未能获取到数据。")
        return

    df = df.sort_values('time').reset_index(drop=True)
    
    # 定位回测真正的起始索引（找到大于等于 start_date 的第一行）
    try:
        start_idx = df[df['time'] >= pd.Timestamp(start_date)].index[0]
        # 确保前面有足够的 WINDOW_SIZE 步数
        if start_idx < WINDOW_SIZE:
            print(f"❌ 缓冲区不足，数据库中该日期前只有 {start_idx} 天数据，需要 {WINDOW_SIZE} 天。")
            return
    except IndexError:
        print("❌ 选定的开始日期在数据库中没有数据。")
        return

    # 2. 加载模型
    print(f"🧠 加载模型: {model_path}")
    model = SAC.load(model_path)
    
    # 3. 初始化状态
    balance = ORIGINAL_MONEY
    shares_held = 0
    net_worth_history = []
    actual_dates = []

    print(f"🚀 开始回测：从索引 {start_idx} ({df.iloc[start_idx]['time'].date()}) 开始")

    # 4. 模拟交易循环
    for i in range(start_idx, len(df)):
        # --- A. 特征工程重构 (90维 历史涨幅) ---
        stock_history = []
        # 获取 [i-WINDOW_SIZE] 到 [i] 范围的价格，共 91 个点，计算 90 个间隔
        window_prices = df['收盘'].iloc[i - WINDOW_SIZE : i + 1].values
        
        for j in range(WINDOW_SIZE):
            p_curr = window_prices[j] if window_prices[j] != 0 else 1e-5
            p_next = window_prices[j+1]
            # 计算涨幅并归一化
            delta_ratio = np.tanh((p_next - p_curr) / p_curr * INCR_PARA)
            stock_history.append(delta_ratio)
            
        # --- B. 资产特征重构 (3维) ---
        current_price = df.iloc[i]['收盘']
        current_net_worth = balance + (shares_held * current_price)
        
        # 必须与 _get_observation 逻辑完全一致
        total_asset_ratio = np.tanh(np.log(max(current_net_worth / ORIGINAL_MONEY, 1e-5)) * ASSET_PARA)
        cash_ratio = balance / current_net_worth if current_net_worth > 0 else 0.0
        position_ratio = 1.0 - cash_ratio
        
        # --- C. 拼接 93 维输入 ---
        obs = np.array(stock_history + [total_asset_ratio, cash_ratio, position_ratio], dtype=np.float32)
        
        # 5. 模型决策
        action, _ = model.predict(obs, deterministic=True)
        act_val = action[0]
        
        # 6. 执行交易逻辑 (根据你的 SAC 输出定义)
        if act_val > 0.5 and balance > 0: # 买入
            shares_held = (balance * 0.9995) / current_price 
            balance = 0
        elif act_val < -0.5 and shares_held > 0: # 卖出
            balance = shares_held * current_price * 0.9995
            shares_held = 0
            
        net_worth_history.append(balance + (shares_held * current_price))
        actual_dates.append(df.iloc[i]['time'])

    # 7. 计算指标与绘图
    nw_series = pd.Series(net_worth_history)
    final_nw = net_worth_history[-1]
    total_return = (final_nw - ORIGINAL_MONEY) / ORIGINAL_MONEY
    max_drawdown = (nw_series / nw_series.cummax() - 1).min()

    print("\n" + "="*30)
    print(f"📊 回测报告 [{stock_code}]")
    print(f"最终净值: {final_nw:.2f}")
    print(f"累计收益: {total_return*100:.2f}%")
    print(f"最大回撤: {max_drawdown*100:.2f}%")
    print("="*30)

    plt.figure(figsize=(12, 6))
    plt.plot(actual_dates, net_worth_history, label='Model Strategy', color='blue')
    # 基准线：买入持有
    benchmark = (df['收盘'].iloc[start_idx:] / df['收盘'].iloc[start_idx]) * ORIGINAL_MONEY
    plt.plot(actual_dates, benchmark.values, label='Buy & Hold Benchmark', linestyle='--', color='gray')
    plt.title(f'Backtest: {stock_code}')
    plt.legend()
    plt.grid(True)
    plt.show()

# 运行
stock_code_list = ["300496", "000001", "600519", "000651", "002475", "601318", "000333", "002594", "601166", "000725",
                     "600036", "601888", "601398", "600276", "002230", "600030", "601012", "600900", "600703", "600585"]
config = InfluxDBConfig(HOST, DATABASE, TOKEN)
manager = InfluxDBManager(config, InfluxDBCallbacks())
for code in stock_code_list:
    print(f"\n================ 回测股票: {code} ================\n")
    backtest_from_db(manager, "sac_random_stock_model2.zip", code, 
                     datetime(2023, 1, 1), datetime(2023, 12, 31))
# backtest_from_db(manager, "./checkpoints/sac_stock_auto_1000000_steps.zip", "300496", 
#                  datetime(2023, 1, 1), datetime(2023, 12, 31))