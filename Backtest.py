import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from rl.environment import SimpleStockEnv
from database.influx_manager import InfluxDBManager, InfluxDBConfig, InfluxDBCallbacks
from config import *
# 假设你的环境类名是 SimpleStockEnv，从你的文件名中导入
# from your_module import SimpleStockEnv 

def run_backtest(model_path, test_df_list, original_money=100000):
    """
    使用保存的模型在测试集上进行回测
    """
    # 1. 加载模型
    model = SAC.load(model_path)
    
    # 2. 准备环境 (这里建议直接使用你训练时的环境类)
    # 确保测试环境的种子固定，以便复现
    env = SimpleStockEnv(df_list=test_df_list) 
    obs, _ = env.reset()
    
    net_worth_history = []
    actions = []
    done = False
    
    print("开始回测...")
    
    while not done:
        # deterministic=True 很重要：回测时不需要探索，只要模型认为最稳的结果
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        net_worth_history.append(info['net_worth'])
        actions.append(action[0])
        
        done = terminated or truncated

    # 3. 数据分析与可视化
    df_result = pd.DataFrame({
        'net_worth': net_worth_history,
        'action': actions
    })
    
    # 计算量化指标
    total_return = (df_result['net_worth'].iloc[-1] / original_money) - 1
    # 简单的回撤计算
    df_result['max_so_far'] = df_result['net_worth'].cummax()
    df_result['drawdown'] = (df_result['net_worth'] - df_result['max_so_far']) / df_result['max_so_far']
    max_drawdown = df_result['drawdown'].min()

    print(f"\n--- 回测报告 ---")
    print(f"最终净值: {df_result['net_worth'].iloc[-1]:.2f}")
    print(f"累计收益率: {total_return * 100:.2f}%")
    print(f"最大回撤: {max_drawdown * 100:.2f}%")
    
    # 4. 绘图
    plt.figure(figsize=(15, 7))
    plt.subplot(2, 1, 1)
    plt.plot(df_result['net_worth'], label='Net Worth')
    plt.title('Backtest Net Worth')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.fill_between(range(len(df_result)), 0, df_result['action'], color='orange', alpha=0.5)
    plt.title('Agent Actions (Buy > 0, Sell < 0)')
    plt.tight_layout()
    plt.show()

    return df_result

# 使用示例
if __name__ == "__main__":
    # 建议选一段模型从未见过的 2024 年以后的数据
    test_data = [pd.read_csv('test_stock_data.csv')] 
    run_backtest("checkpoints/sac_stock_auto_1000000_steps.zip", test_data)