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
    确定性回测环境：继承自 SimpleStockEnv
    """
    def _get_observation(self):
        # 1. 获取基础历史数据（不加噪声，不加掩码）
        history = np.array(self.stock_history.copy(), dtype=np.float32)
        
        # 2. 派生特征计算（逻辑与父类完全一致，但确保没有随机扰动）
        current_idx = self.today
        lookback = 65 
        start_idx = max(0, current_idx - lookback)
        window_prices = self.prices[start_idx : current_idx + 1]
        
        def get_bias(p_array, period):
            if len(p_array) < period:
                return 0.0
            ma = np.mean(p_array[-period:])
            return (p_array[-1] - ma) / ma * INCR_PARA
            
        def get_ma(p_array, period):
            return np.mean(p_array[-period:])
        
        bias5  = get_bias(window_prices, 5)
        bias20 = get_bias(window_prices, 20)
        bias60 = get_bias(window_prices, 60)
        
        self.ma5 = get_ma(window_prices, 5)
        self.ma20 = get_ma(window_prices, 20)
        ma_dist5_20 = (self.ma5 - self.ma20) / self.ma20 * INCR_PARA

        # 3. 账户状态
        current_price = self.prices[self.today]
        current_net_worth = self.my_cash + self.number_of_shares * current_price
        cash_ratio = self.my_cash / current_net_worth if current_net_worth > 0 else 0.0
        position_ratio = 1.0 - cash_ratio
        
        # 4. 拼接最终向量
        obs = np.concatenate([
            history, 
            [bias5, bias20, bias60, ma_dist5_20],
            [cash_ratio, position_ratio]
        ]).astype(np.float32)
        
        return obs

    def reset(self, seed=None, options=None):
        # 1. 基础初始化
        if seed is not None:
            np.random.seed(seed)
        
        # 强制选择第一只股票数据
        self.current_df = self.df_list[0]
        self.prices = self.current_df['收盘'].values.astype(np.float32)
        self.dates = pd.to_datetime(self.current_df['time'].values)
        
        total_len = len(self.prices)
        
        # 2. 确定性起跑点：从 WINDOW_SIZE 开始
        start_index = 0
        self.today = start_index + WINDOW_SIZE
        self.last_day = total_len - 1 

        # 3. 初始化账户 (严格同步父类变量名)
        self.my_cash = ORIGINAL_MONEY
        self.number_of_shares = 0
        self.highest_worth = ORIGINAL_MONEY
        self.highest_worth_day = self.today
        
        # 4. 初始化 Reward 计算相关变量
        self.alpha = 1.0  # 回测时 alpha 通常设为起始值，或根据你的策略调整
        self.target_value = NEW_HIGH_TARGET
        self.new_high_reward = NEW_HIGH_REWARD
        self.times = 1 # 对应父类的 self.times 迭代
        
        # 5. 初始化 info 统计变量
        self.ma5 = 0
        self.ma20 = 0
        self.ave_r_base = 0
        self.ave_r_risk = 0
        self.ave_r_new_high = 0
        self.max_r_base = 0
        self.max_r_risk = 0
        self.max_r_new_high = 0
        self.max_drawdown = 0
        
        # 6. 初始化历史数据 (同步父类逻辑)
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
    records 包含: dates, prices, net_worths, actions, pos_ratio, rewards_breakdown
    """
    # 提取数据
    dates = records['dates']
    prices = records['prices']
    net_worths = records['net_worths']
    actions = records['actions']
    pos_ratio = records['pos_ratio']
    
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
    ax3.fill_between(dates, pos_ratio, color='orange', alpha=0.3, label='仓位占比')
    ax3.plot(dates, pos_ratio, color='orange', linewidth=1)
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
    target_stocks = [
    # --- 能源与红利板块 (低波动、高股息、独立行情) ---
    "600938",  # 中国海油 - 国际油价驱动，高股息
    "600900",  # 长江电力 - 防御性极强的类债资产
    "601088",  # 中国神华 - 煤炭龙头，红利风格代表
    "601899",  # 紫金矿业 - 黄金+铜，受国际大宗商品定价

    # --- 核心科技与AI (高弹性、受美股科技股映射) ---
    "300308",  # 中际旭创 - AI光模块龙头，波动率极大
    "601138",  # 工业富联 - AI服务器+苹果概念，流动性极佳
    "002371",  # 北方华创 - 半导体设备，国产化替代核心
    "603986",  # 兆易创新 - 存储芯片，半导体周期拐点代表

    # --- 权重白马与内需消费 (指数定海神针) ---
    "600519",  # 贵州茅台 - 消费总龙头，市场信心指标
    "000333",  # 美的集团 - 家电白马，业绩极其稳健
    "603605",  # 珀莱雅   - 消费细分领域(美妆)的长牛代表
    "000651",  # 格力电器 - 传统白马，高分红+低估值

    # --- 新能源与高端制造 (全球定价、出海逻辑) ---
    "300750",  # 宁德时代 - 锂电绝对龙头，创业板权重
    "002594",  # 比亚迪   - 新能源车龙头，制造能力代表
    "600031",  # 三一重工 - 机械出海，老牌周期白马复苏
    "002475",  # 立讯精密 - 电子制造服务，精密制造代表

    # --- 金融与市场情绪 (牛市旗手、宏观beta) ---
    "600030",  # 中信证券 - 券商龙头，反应市场活跃度
    "601318",  # 中国平安 - 保险/金融，宏观经济晴雨表
    "601166",  # 兴业银行 - 低估值银行，高流动性金融权重
    "000725"   # 京东方A  - 面板周期龙头，极其庞大的成交量
]
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
            'actions': [], 'pos_ratio': []
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
            records['actions'].append(float(action[0]))
            records['pos_ratio'].append(info['pos_ratio'])

        # 6. 画图
        print(f"✅ 回测完成，正在绘图...")
        plot_backtest_results(code, records)

    manager.close()