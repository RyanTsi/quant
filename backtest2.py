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

# --- 1. 定制一个用于回测的确定性环境 ---
class SingleStockTestEnv(SimpleStockEnv):
    """
    继承自 SimpleStockEnv，但去掉了随机性。
    强制只使用传入的那一张 DataFrame，并且从第 WINDOW_SIZE 天一直跑到最后一天。
    """
    def reset(self, seed=None, options=None):
        # 不调用 super().reset() 因为我们要重写初始化逻辑
        # 但为了兼容性，保留 seed 处理
        if seed is not None:
            np.random.seed(seed)
        
        # 强制选择第一只（也是唯一一只）股票
        self.current_df = self.df_list[0]
        self.prices = self.current_df['收盘'].values.astype(np.float32)
        # 获取日期用于画图
        self.dates = pd.to_datetime(self.current_df['time'].values)
        
        total_len = len(self.prices)
        
        # --- 关键修改：不再随机选择开始时间 ---
        # 强制从数据能支持的最早时间开始
        start_index = 0
        self.today = start_index + WINDOW_SIZE
        # 强制跑到数据结束
        self.last_day = total_len - 1 

        # 初始化账户
        self.my_cash = ORIGINAL_MONEY
        self.number_of_shares = 0
        self.target_value = NEW_HIGH_TARGET
        self.new_high_reward = NEW_HIGH_REWARD
        self.times = 0
        self.ave_r_base = 0
        self.ave_r_risk_hold = 0
        self.ave_r_risk_down = 0
        self.ave_r_action_penalty = 0
        self.ave_r_position_uncertainty = 0
        self.ave_r_new_high = 0
        
        self.max_r_base = 0
        self.max_r_risk_hold = 0
        self.max_r_risk_down = 0
        self.max_r_action_penalty = 0
        self.max_r_position_uncertainty = 0
        self.max_r_new_high = 0
        self.pos_ratio = 0

        # 初始化历史
        self.stock_history = []
        current_window_prices = self.prices[self.today - WINDOW_SIZE : self.today + 1]
        for i in range(WINDOW_SIZE):
            p_curr = max(current_window_prices[i], 1e-5)
            p_next = current_window_prices[i+1]
            delta_ratio = np.log(p_next / p_curr) * INCR_PARA
            self.stock_history.append(delta_ratio)

        return self._get_observation(), {}

# --- 2. 绘图函数 ---
def plot_backtest_results(stock_code, records):
    """
    records 包含: dates, prices, net_worths, actions, pos_ratios, rewards_breakdown
    """
    # 提取数据
    dates = records['dates']
    prices = records['prices']
    net_worths = records['net_worths']
    actions = records['actions']
    pos_ratios = records['pos_ratios']
    
    # 准备 Buy/Sell 信号用于画图
    buy_x, buy_y = [], []
    sell_x, sell_y = [], []
    
    for i, act in enumerate(actions):
        if act > 0.15: # 只有明显买入才标记
            buy_x.append(dates[i])
            buy_y.append(prices[i])
        elif act < -0.15: # 只有明显卖出才标记
            sell_x.append(dates[i])
            sell_y.append(prices[i])

    # 设置风格
    sns.set_theme(style="darkgrid")
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建 4 个子图
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [3, 2, 1, 1]})
    fig.suptitle(f"个股回测分析: {stock_code}", fontsize=20, fontweight='bold')

    # Subplot 1: 股价 + 买卖点
    ax1 = axes[0]
    ax1.plot(dates, prices, label='股价 (Close)', color='black', alpha=0.6, linewidth=1.5)
    # 画买卖信号
    ax1.scatter(buy_x, buy_y, color='red', marker='^', s=100, label='买入', zorder=5)
    ax1.scatter(sell_x, sell_y, color='green', marker='v', s=100, label='卖出', zorder=5)
    ax1.set_ylabel('股价')
    ax1.legend(loc='upper left')
    ax1.set_title("股价走势与交易信号")

    # Subplot 2: 账户净值
    ax2 = axes[1]
    # 计算基准收益（如果全仓持有不动）
    initial_price = prices[0]
    benchmark = [ORIGINAL_MONEY * (p / initial_price) for p in prices]
    
    ax2.plot(dates, net_worths, label='AI 策略净值', color='purple', linewidth=2)
    ax2.plot(dates, benchmark, label='基准(买入持有)', color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('资金')
    ax2.legend(loc='upper left')
    ax2.set_title("策略净值 vs 基准收益")

    # Subplot 3: 仓位变化
    ax3 = axes[2]
    ax3.fill_between(dates, pos_ratios, color='orange', alpha=0.3, label='仓位占比')
    ax3.plot(dates, pos_ratios, color='orange', linewidth=1)
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_ylabel('仓位 (0-1)')
    ax3.set_title("持仓比例变化")

    # Subplot 4: 动作强度 (Action)
    ax4 = axes[3]
    ax4.bar(dates, actions, color=np.where(np.array(actions)>0, 'red', 'green'), width=1.0)
    ax4.set_ylim(-1.1, 1.1)
    ax4.set_ylabel('动作 (-1卖 ~ 1买)')
    ax4.axhline(0, color='black', linewidth=0.5)
    ax4.set_title("AI 决策强度")

    plt.tight_layout()
    plt.show()

# --- 3. 主程序 ---
if __name__ == "__main__":
    # 配置
    target_stocks = ["600519", "300750", "300496", "000001", "600519", "000651", "002475", "601318", "000333", "002594", "601166", "000725"]
    test_start = datetime(2024, 1, 1)
    test_end = datetime(2025, 12, 12)
    
    # 1. 加载模型
    model_path = "./best_model/best_model.zip" # 确保路径正确
    print(f"📦 正在加载模型: {model_path}")
    model = SAC.load(model_path, device="cuda")

    # 2. 准备数据连接
    config = InfluxDBConfig(HOST, DATABASE, TOKEN)
    manager = InfluxDBManager(config, InfluxDBCallbacks())

    for code in target_stocks:
        print(f"\n🚀 正在测试: {code}")
        
        # 3. 获取单只股票数据
        df = manager.get_stock_data_by_range(code, test_start, test_end)
        df = rl.prehandle.preprocess_data(df)
        
        if df is None or len(df) < WINDOW_SIZE + 5:
            print(f"❌ 数据不足，跳过 {code}")
            continue
            
        # 4. 初始化测试环境
        # 注意：这里我们传入只有一只股票的列表
        env = SingleStockTestEnv([df])
        obs, _ = env.reset()
        
        # 5. 运行回测循环
        records = {
            'dates': [], 'prices': [], 'net_worths': [], 
            'actions': [], 'pos_ratios': []
        }
        
        done = False
        while not done:
            # 记录 T 时刻的数据
            current_date = env.dates[env.today] # 从 dataframe 获取真实日期
            current_price = env.prices[env.today]
            
            # AI 预测
            action, _ = model.predict(obs, deterministic=True) # ⚠️ 必须 deterministic=True
            
            # 执行一步
            obs, reward, done, truncated, info = env.step(action)
            
            # 收集数据
            records['dates'].append(current_date)
            records['prices'].append(current_price)
            records['net_worths'].append(info['net_worth'])
            records['actions'].append(float(action[0])) # 记录动作数值
            records['pos_ratios'].append(info['pos_ratios']) # 这里的 key 要和你 step 返回的 info 一致

        # 6. 画图
        print(f"✅ 回测完成，正在绘图...")
        plot_backtest_results(code, records)

    manager.close()