import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

# --- 必须引入自定义网络结构，否则模型加载会报错 ---
# (为了方便运行，这里复制了你的特征提取器定义，实际项目中建议 import)
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class LSTM_Attention_Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.window_size = observation_space.shape[0]
        self.input_features = observation_space.shape[1]
        hidden_size = 64
        self.lstm = nn.LSTM(input_size=self.input_features, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=0.0) # Eval模式下去掉dropout
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.Tanh(), nn.Linear(32, 1), nn.Softmax(dim=1)
        )
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, features_dim), nn.LayerNorm(features_dim), nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.float()
        lstm_out, _ = self.lstm(x)
        weights = self.attention(lstm_out)
        context_vector = torch.sum(weights * lstm_out, dim=1)
        return self.linear(context_vector)

# --- 引入你的环境 ---
from rl.signalEnv import AStockSignalEnv

# ==========================================
# 回测专用工具类
# ==========================================
class BacktestEngine:
    def __init__(self, env: AStockSignalEnv, model):
        self.env = env
        self.model = model
        
    def run_single_stock(self, stock_idx):
        """
        强制跑特定一只股票的全程数据
        """
        # 1. 强制设置环境内部状态
        # 获取该股票总长度
        data_len = len(self.env.data_list[stock_idx])
        valid_steps = data_len - self.env.window_size - 1
        
        if valid_steps <= 0:
            print(f"⚠️ 股票ID {stock_idx} 数据太短，跳过。")
            return None

        # 手动重置环境状态
        self.env.current_stock_idx = stock_idx
        self.env.day_idx = self.env.window_size # 从窗口期结束开始
        self.env.steps_taken = 0
        self.env.last_signal = 0.0
        self.env.portfolio_value = 1.0 # 归一化净值
        
        # 获取初始 Observation
        obs = self.env._get_observation()
        
        # 记录器
        history = {
            'signal': [],
            'action': [], # 实际持仓
            'stock_ret': [],
            'index_ret': [], # 近似推算
            'strategy_ret': [],
            'portfolio_value': [],
            'nav_stock': [], # 个股买入持有净值
            'nav_index': []  # 指数买入持有净值
        }
        
        curr_stock_nav = 1.0
        curr_index_nav = 1.0
        
        print(f"🔄 开始回测股票 ID: {stock_idx} (共 {valid_steps} 天)...")
        
        # 2. 步进循环
        for _ in range(valid_steps):
            # 模型预测 (deterministic=True 关闭随机探索)
            action, _ = self.model.predict(obs, deterministic=True)
            
            # 环境步进
            # 注意: step返回的 reward 是经过 scale 的，这里我们需要 info 里的原始数据
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            # 提取数据
            signal = info['Action/Signal']
            raw_stock_ret = info['Attribution/Abs_Ret_Day'] # 个股当日涨跌幅(%)
            # 估算指数涨跌幅: Stock_Abs - Alpha = Index (近似)
            raw_alpha = info['Attribution/Alpha_Ret_Day']
            raw_index_ret = raw_stock_ret - raw_alpha
            
            # 策略收益 (未扣费，简单模拟) = 信号 * 个股涨跌幅
            # 注意：这是多空策略。如果只是做多，逻辑不同。
            # 这里假设：Signal=1 全仓买入，Signal=-1 全仓做空
            # 如果是A股无法做空，你需要将 signal clip 到 [0, 1]
            strat_ret = signal * raw_stock_ret 
            
            # 更新净值
            curr_stock_nav *= (1 + raw_stock_ret/100.0)
            curr_index_nav *= (1 + raw_index_ret/100.0)
            
            history['signal'].append(signal)
            history['stock_ret'].append(raw_stock_ret)
            history['index_ret'].append(raw_index_ret)
            history['strategy_ret'].append(strat_ret)
            history['portfolio_value'].append(info['State/Portfolio_Value'])
            history['nav_stock'].append(curr_stock_nav)
            history['nav_index'].append(curr_index_nav)
            
            if terminated or truncated:
                break
                
        return pd.DataFrame(history)

    def plot_results(self, df, stock_name="Stock"):
        """绘图分析"""
        if df is None or len(df) == 0:
            return

        plt.figure(figsize=(16, 10))
        
        # 子图1: 净值对比
        plt.subplot(3, 1, 1)
        plt.title(f"Backtest Performance: {stock_name}")
        plt.plot(df['portfolio_value'], label='AI Strategy (Alpha)', color='red', linewidth=2)
        plt.plot(df['nav_stock'], label='Buy & Hold (Stock)', color='gray', alpha=0.5, linestyle='--')
        plt.plot(df['nav_index'], label='Benchmark (Index)', color='blue', alpha=0.5, linestyle='--')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylabel("Normalized Value")
        
        # 子图2: 信号与开仓
        plt.subplot(3, 1, 2)
        plt.title("Model Signal Strength (-1 to 1)")
        # 绘制信号区域
        plt.fill_between(df.index, df['signal'], 0, where=(df['signal']>0), color='red', alpha=0.3, label='Long')
        plt.fill_between(df.index, df['signal'], 0, where=(df['signal']<0), color='green', alpha=0.3, label='Short')
        plt.plot(df['signal'], color='black', linewidth=0.8)
        plt.axhline(0, color='black', linestyle='--')
        plt.ylabel("Signal")
        plt.grid(True, alpha=0.3)
        
        # 子图3: 累计超额收益 (Alpha)
        plt.subplot(3, 1, 3)
        # 简单计算累计超额：策略净值 / 指数净值
        cum_alpha = df['portfolio_value'] / df['nav_index']
        plt.plot(cum_alpha, color='purple', label='Relative Strength vs Index')
        plt.title("Cumulative Alpha (Strategy / Index)")
        plt.ylabel("Relative Value")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def calculate_metrics(self, df):
        if df is None: return
        
        # 计算日收益率
        # portfolio_value 是净值，转回 pct_change
        rets = df['portfolio_value'].pct_change().dropna()
        
        total_ret = (df['portfolio_value'].iloc[-1] - 1) * 100
        ann_ret = rets.mean() * 252 * 100
        volatility = rets.std() * np.sqrt(252) * 100
        sharpe = (ann_ret - 3.0) / volatility if volatility > 0 else 0 # 假设无风险利率3%
        
        # 最大回撤
        cum_max = df['portfolio_value'].cummax()
        drawdown = (df['portfolio_value'] - cum_max) / cum_max
        max_dd = drawdown.min() * 100
        
        # 胜率 (Alpha > 0 的天数 / 总交易天数)
        # 这里定义为：只要策略收益 > 0 就算赢（不严谨，但常用）
        win_rate = (df['strategy_ret'] > 0).mean() * 100
        
        print("-" * 40)
        print(f"📊 策略表现报告")
        print("-" * 40)
        print(f"累计收益: {total_ret:6.2f}%")
        print(f"年化收益: {ann_ret:6.2f}%")
        print(f"年化波动: {volatility:6.2f}%")
        print(f"夏普比率: {sharpe:6.2f}")
        print(f"最大回撤: {max_dd:6.2f}%")
        print(f"交易胜率: {win_rate:6.2f}%")
        print("-" * 40)

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    # 1. 路径设置
    MODEL_PATH = "./best_modelV4/best_model.zip"
    VAL_DATA_PATH = "val_data_v4.pkl"  # 确保你有这个文件
    
    # 2. 加载数据
    if not os.path.exists(VAL_DATA_PATH):
        print("❌ 找不到验证数据文件，请先运行训练脚本生成缓存。")
        exit()
        
    print(f"📂 加载数据 {VAL_DATA_PATH}...")
    with open(VAL_DATA_PATH, "rb") as f:
        val_dfs = pickle.load(f)
        
    # 3. 初始化环境 (Backtest Mode)
    # 这里的参数应该和训练时一致，但 training_days 要设得很大，防止被截断
    env_kwargs = {
        'window_size': 60,
        'training_days': 10000, # 设大一点，覆盖全时段
        'transaction_cost_pct': 0.0010, # 加上成本测试更真实
        'deadzone_level': 0.1,
        'reward_scale': 1
    }
    # 只需要原始环境类，不需要 VecEnv 包装，方便我们手动控制
    raw_env = AStockSignalEnv(val_dfs, **env_kwargs)
    
    # 4. 加载模型
    print(f"🧠 加载模型 {MODEL_PATH}...")
    # device='cpu' 方便回测，不需要gpu
    model = SAC.load(MODEL_PATH, device="cpu", custom_objects={
        "observation_space": raw_env.observation_space,
        "action_space": raw_env.action_space
    })
    
    # 5. 运行回测
    tester = BacktestEngine(raw_env, model)
    
    # --- 模式 A: 随机抽几只验证 ---
    import random
    # 假设 df_list[0] 是指数，我们从 1 开始抽
    test_ids = [1, 5, 10] if len(val_dfs) > 10 else [1]
    
    for stock_id in test_ids:
        if stock_id >= len(val_dfs): continue
        
        print(f"\n======== 测试股票 INDEX: {stock_id} ========")
        res_df = tester.run_single_stock(stock_id)
        
        if res_df is not None:
            tester.calculate_metrics(res_df)
            tester.plot_results(res_df, stock_name=f"Stock_{stock_id}")
            
    # --- 模式 B: (可选) 批量跑全市场看平均夏普 ---
    # 如果想跑所有股票的平均表现，可以写个循环把 metrics 存起来取平均