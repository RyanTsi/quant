import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import SAC
from datetime import datetime
from rl.environment import SimpleStockEnv
from database.influx_manager import InfluxDBManager, InfluxDBConfig, InfluxDBCallbacks
import rl.prehandle
from config import *

# --- 1. 定制一个用于回测的确定性环境 (适配 SimpleStockEnv v2) ---
class SingleStockTestEnv(SimpleStockEnv):
    """
    确定性回测环境
    修改重点：
    1. 移除价格噪声 (_calculate_noisy_price)
    2. 移除观测噪声 (dropout, noise)
    3. 固定 time_remaining = 0.0 (收盘决策)
    4. 补全 v2 版本所需的 open_gap, alpha 等特征
    """

    def _calculate_noisy_price(self, day_idx, time_rem):
        # [修改] 回测时强制返回准确的收盘价，不加噪声
        return self.prices_close[day_idx]

    def _get_observation(self):
        # --- A. 历史序列 (无噪声版) ---
        # 直接使用 self.stock_history，不做任何噪声处理
        history = np.array(self.stock_history.copy(), dtype=np.float32)

        # --- B. 技术指标 (逻辑完全对齐父类) ---
        current_idx = self.today
        start_idx = max(0, current_idx - 65)
        window_prices = self.prices_close[start_idx : current_idx + 1]

        # 计算开盘缺口 (Open Gap)
        # 注意：回测时我们已经有了全量数据，可以直接取
        current_open = self.prices_open[self.today]
        if self.today > 0:
            prev_close = self.prices_close[self.today - 1]
        else:
            prev_close = current_open
        
        if prev_close <= 0: prev_close = 1e-5
        open_gap = np.log(current_open / prev_close) * INCR_PARA

        def get_bias(p_array, period):
            if len(p_array) < period: return 0.0
            ma = np.mean(p_array[-period:])
            if ma == 0: return 0.0
            return (p_array[-1] - ma) / ma * INCR_PARA
            
        def get_ma(p_array, period):
            if len(p_array) < period: return p_array[-1]
            return np.mean(p_array[-period:])
        
        bias5  = get_bias(window_prices, 5)
        bias20 = get_bias(window_prices, 20)
        bias60 = get_bias(window_prices, 60)
        
        self.ma5 = get_ma(window_prices, 5)
        self.ma20 = get_ma(window_prices, 20)
        ma_dist5_20 = (self.ma5 - self.ma20) / (self.ma20 + 1e-8) * INCR_PARA

        # --- C. 仓位状态 ---
        # 此时 self.current_price 已经被 step 或 reset 更新为准确的 Close
        current_net_worth = self.my_cash + self.number_of_shares * self.current_price
        
        if current_net_worth <= 0:
            cash_ratio, position_ratio = 0.0, 0.0
        else:
            cash_ratio = self.my_cash / current_net_worth
            position_ratio = 1.0 - cash_ratio
        
        # --- D. 拼接特征 (必须匹配 WINDOW_SIZE + 9) ---
        # v2 特征顺序: history + [bias5, bias20, bias60, ma_dist] + [cash, pos] + [gap] + [time] + [alpha]
        obs = np.concatenate([
            history, 
            [bias5, bias20, bias60, ma_dist5_20],
            [cash_ratio, position_ratio],
            [open_gap],
            [self.time_remaining], # 固定为 0.0
            [self.alpha]           # 固定值
        ]).astype(np.float32)
        
        return obs

    def reset(self, seed=None, options=None):
        # [修改] 强制选择列表中的第一只股票
        self.current_df = self.df_list[0]
        
        # 准备价格数据
        self.prices_close = self.current_df['收盘'].values.astype(np.float32)
        if '开盘' in self.current_df.columns:
            self.prices_open = self.current_df['开盘'].values.astype(np.float32)
        else:
            self.prices_open = self.prices_close

        # 记录真实日期用于绘图
        if 'time' in self.current_df.columns:
            self.dates = pd.to_datetime(self.current_df['time'].values)
        else:
            # 如果没有时间列，生成虚拟时间
            self.dates = pd.date_range(start='2024-01-01', periods=len(self.prices_close))
        
        total_len = len(self.prices_close)
        start_index = 0 # 回测从头开始
        self.today = start_index + WINDOW_SIZE
        self.last_day = total_len - 1 

        # --- 账户重置 ---
        self.my_cash = ORIGINAL_MONEY
        self.number_of_shares = 0
        self.highest_worth = ORIGINAL_MONEY
        self.max_drawdown_cur = 0
        self.max_drawdown_global = 0
        
        self.episode_rewards = {
            "r_base": [], "r_base_pos": [], "r_base_neg": [], 
            "r_risk": [], "r_cash": [],
            "r_turnover": []
        }

        # --- [关键] 回测参数固定 ---
        # 你可以将 alpha 设为 0.1 (激进) 到 1.0 (保守) 之间的值来测试模型反应
        self.alpha = 0.1
        self.time_remaining = np.random.normal(0, 1)
        
        # 初始化价格 (无噪声)
        self.current_price = self.prices_close[self.today]

        # --- 初始化历史序列 ---
        self.ma5 = 0; self.ma20 = 0
        self.stock_history = []
        current_window_prices = self.prices_close[self.today - WINDOW_SIZE : self.today + 1]
        
        for i in range(WINDOW_SIZE):
            p_curr = max(current_window_prices[i], 1e-5)
            p_next = current_window_prices[i+1]
            delta_ratio = np.log(p_next / p_curr) * INCR_PARA
            self.stock_history.append(delta_ratio)

        return self._get_observation(), {}

    def step(self, action):
        # 调用父类的 step 计算逻辑 (含 Reward 计算)
        obs, reward, terminated, truncated, info = super().step(action)
        
        # [补充] 修正父类 step 结束时会随机化 time_remaining 和 price 的行为
        # 我们需要保持确定性
        self.time_remaining = 0.0
        if self.today < len(self.prices_close):
             self.current_price = self.prices_close[self.today]
        
        # [补充] 将当天的具体风险惩罚值注入 info，供绘图使用
        # 父类只记录在 self.episode_rewards 列表里
        if len(self.episode_rewards["r_risk"]) > 0:
            info['step_r_risk'] = self.episode_rewards["r_risk"][-1]
        else:
            info['step_r_risk'] = 0.0
            
        # [补充] 注入仓位比例 (父类 info 可能没带)
        current_net_worth = self.my_cash + self.number_of_shares * self.current_price
        if current_net_worth > 0:
            info['pos_ratio'] = (self.number_of_shares * self.current_price) / current_net_worth
        else:
            info['pos_ratio'] = 0.0

        return obs, reward, terminated, truncated, info

# --- 2. 绘图函数 (增强版) ---
def plot_backtest_results(stock_code, records):
    """
    records 包含: dates, prices, net_worths, actions, pos_ratio, r_risks, ma20
    """
    dates = records['dates']
    prices = np.array(records['prices'])
    net_worths = records['net_worths']
    actions = records['actions']
    pos_ratio = records['pos_ratio']
    r_risks = np.array(records['r_risks']) # 风险惩罚分
    ma20 = np.array(records['ma20'])       # 20日均线

    # 准备买卖信号点
    buy_x, buy_y = [], []
    sell_x, sell_y = [], []
    for i, act in enumerate(actions):
        if act > 0.15: 
            buy_x.append(dates[i]); buy_y.append(prices[i])
        elif act < -0.15: 
            sell_x.append(dates[i]); sell_y.append(prices[i])

    sns.set_theme(style="darkgrid")
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei'] # 试图设置中文支持
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    # 增加高度，容纳 5 个子图
    fig, axes = plt.subplots(5, 1, figsize=(16, 18), sharex=True, 
                             gridspec_kw={'height_ratios': [3, 2, 1, 1, 1.5]})
    fig.suptitle(f"Backtest Analysis: {stock_code}", fontsize=20, fontweight='bold')

    # Subplot 1: 股价 + 买卖点 + MA20
    axes[0].plot(dates, prices, label='Close Price', color='black', alpha=0.6)
    axes[0].plot(dates, ma20, label='MA20', color='blue', linestyle='--', alpha=0.4)
    axes[0].scatter(buy_x, buy_y, color='red', marker='^', s=80, label='Buy', zorder=5)
    axes[0].scatter(sell_x, sell_y, color='green', marker='v', s=80, label='Sell', zorder=5)
    axes[0].set_ylabel('Price')
    axes[0].legend(loc='upper left')

    # Subplot 2: 账户净值 vs 基准
    initial_price = prices[0]
    benchmark = [ORIGINAL_MONEY * (p / initial_price) for p in prices]
    axes[1].plot(dates, net_worths, label='AI Net Worth', color='purple', linewidth=2)
    axes[1].plot(dates, benchmark, label='Buy & Hold', color='gray', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Net Worth')
    axes[1].legend(loc='upper left')

    # Subplot 3: 仓位变化
    axes[2].fill_between(dates, pos_ratio, color='orange', alpha=0.3, label='Position %')
    axes[2].set_ylabel('Position')
    axes[2].set_ylim(-0.1, 1.1)

    # Subplot 4: 动作强度
    colors = np.where(np.array(actions)>0, 'red', 'green')
    axes[3].bar(dates, actions, color=colors, width=1.0)
    axes[3].axhline(0, color='black', linewidth=0.5)
    axes[3].set_ylabel('Action')
    axes[3].set_ylim(-1.1, 1.1)

    # --- Subplot 5: 风险痛感分析 (r_risk) ---
    ax5 = axes[4]
    # 绘制风险惩罚曲线 (通常是 0 或负数)
    ax5.plot(dates, r_risks, color='crimson', label='Risk Penalty (r_risk)', linewidth=1.5)
    
    # 标注那些惩罚特别大的时刻
    risk_threshold = -0.5 # 假设阈值
    pain_dates = [d for d, r in zip(dates, r_risks) if r < risk_threshold]
    pain_vals = [r for r in r_risks if r < risk_threshold]
    ax5.scatter(pain_dates, pain_vals, color='black', marker='x', s=30, label='High Pain')

    ax5.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax5.set_ylabel('Pain Score')
    ax5.set_title("Risk Penalty Analysis (Lower is more painful)")
    ax5.legend(loc='lower left')

    plt.tight_layout()
    plt.show()

# --- 3. 主程序 ---
if __name__ == "__main__":
    # 配置回测股票池
    target_stocks = [
        "600938",  # 中国海油 (高红利)
        "300308",  # 中际旭创 (高波动 AI)
        "600519",  # 贵州茅台 (白马)
        "300750"   # 宁德时代 (新能源)
    ]
    
    test_start = datetime(2024, 1, 1) # 建议拉长一点看
    test_end = datetime(2025, 12, 12)
    
    # 1. 加载模型
    model_path = "./best_model/best_model.zip" 
    print(f"📦 Loading Model: {model_path}")
    try:
        model = SAC.load(model_path, device="cuda")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    # 2. 准备数据连接
    config = InfluxDBConfig(HOST, DATABASE, TOKEN)
    manager = InfluxDBManager(config, InfluxDBCallbacks())

    for code in target_stocks:
        print(f"\n🚀 Testing: {code}")
        
        # 3. 获取单只股票数据
        df = manager.get_stock_data_by_range(code, test_start, test_end)
        df = rl.prehandle.preprocess_data(df)
        
        if df is None or len(df) < WINDOW_SIZE + 5:
            print(f"❌ Not enough data for {code}")
            continue
            
        # 4. 初始化测试环境
        # 注意: 传入只有一只股票的列表
        env = SingleStockTestEnv([df])
        obs, _ = env.reset()
        
        # 5. 运行回测循环
        records = {
            'dates': [], 'prices': [], 'net_worths': [], 
            'actions': [], 'pos_ratio': [], 'r_risks': [], 'ma20': []
        }
        
        done = False
        while not done:
            current_date = env.dates[env.today]
            # 此时 env.current_price 已经是准确的收盘价
            current_price = env.current_price 
            current_ma20 = env.ma20 
            
            # 预测
            action, _ = model.predict(obs, deterministic=True) 
            
            # 执行
            obs, reward, done, truncated, info = env.step(action)
            
            # 记录
            records['dates'].append(current_date)
            records['prices'].append(current_price)
            records['net_worths'].append(info['net_worth'])
            records['actions'].append(float(action[0]))
            records['pos_ratio'].append(info['pos_ratio'])
            records['ma20'].append(current_ma20)
            # 获取当步的风险惩罚
            records['r_risks'].append(info.get('step_r_risk', 0.0))

        # 6. 画图
        print(f"✅ Backtest finished for {code}. Net Worth: {info['net_worth']:.2f}")
        plot_backtest_results(code, records)

    manager.close()