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
    确定性回测环境：严格对齐 SimpleStockEnv 的最新逻辑
    """
    def _get_observation(self):
        # 1. 基础历史数据（回测关闭噪声和掩码，保持确定性）
        history = np.array(self.stock_history.copy(), dtype=np.float32)
        
        # 2. 派生特征计算（逻辑与父类完全一致）
        current_idx = self.today
        start_idx = max(0, current_idx - 65)
        window_prices = self.prices[start_idx : current_idx + 1]
        
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
        
        # 同步父类的均线距离公式 (ma5 - ma20) / (ma20 + 1e-8)
        ma_dist5_20 = (self.ma5 - self.ma20) / (self.ma20 + 1e-8) * INCR_PARA

        # 3. 仓位状态 (同步父类逻辑)
        current_price = self.prices[self.today]
        current_net_worth = self.my_cash + self.number_of_shares * current_price
        
        if current_net_worth <= 0:
            cash_ratio = 0.0
            position_ratio = 0.0
        else:
            cash_ratio = self.my_cash / current_net_worth
            position_ratio = 1.0 - cash_ratio
        
        obs = np.concatenate([
            history, 
            [bias5, bias20, bias60, ma_dist5_20],
            [cash_ratio, position_ratio]
        ]).astype(np.float32)
        
        return obs

    def reset(self, seed=None, options=None):
        # 强制选择传入的第一只股票
        self.current_df = self.df_list[0]
        self.prices = self.current_df['收盘'].values.astype(np.float32)
        # 记录真实日期用于绘图
        self.dates = pd.to_datetime(self.current_df['time'].values)
        
        total_len = len(self.prices)
        start_index = 0 # 回测从头开始
        self.today = start_index + WINDOW_SIZE
        self.last_day = total_len - 1 

        # 3. 初始化账户 (严格同步父类变量名)
        self.my_cash = ORIGINAL_MONEY
        self.number_of_shares = 0
        self.highest_worth = ORIGINAL_MONEY
        self.highest_worth_day = self.today
        self.max_drawdown = 0
        
        # 4. 回测时的 Alpha 设置
        # 建议设为 1.0 或 训练结束时的最终 Alpha，用于观察风险惩罚下的收益
        self.alpha = 1.0 
        
        # 5. 初始化容器
        self.episode_rewards = {"r_base": [], "r_risk": [], "r_new_high": []}
        
        # 6. 初始化历史序列
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
    records 包含: dates, prices, net_worths, actions, pos_ratio, r_risks, ma20
    """
    dates = records['dates']
    prices = np.array(records['prices'])
    net_worths = records['net_worths']
    actions = records['actions']
    pos_ratio = records['pos_ratio']
    r_risks = np.array(records['r_risks']) # 新增：风险惩罚分
    ma20 = np.array(records['ma20'])       # 新增：20日均线

    # 准备买卖信号
    buy_x, buy_y = [], []
    sell_x, sell_y = [], []
    for i, act in enumerate(actions):
        if act > 0.15: 
            buy_x.append(dates[i]); buy_y.append(prices[i])
        elif act < -0.15: 
            sell_x.append(dates[i]); sell_y.append(prices[i])

    sns.set_theme(style="darkgrid")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 增加高度，容纳 5 个子图
    fig, axes = plt.subplots(5, 1, figsize=(16, 16), sharex=True, 
                             gridspec_kw={'height_ratios': [3, 2, 1, 1, 1.5]})
    fig.suptitle(f"个股回测与风险痛感分析: {stock_code}", fontsize=20, fontweight='bold')

    # Subplot 1: 股价 + 买卖点
    axes[0].plot(dates, prices, label='股价 (Close)', color='black', alpha=0.6)
    axes[0].plot(dates, ma20, label='MA20', color='blue', linestyle='--', alpha=0.4) # 画出均线
    axes[0].scatter(buy_x, buy_y, color='red', marker='^', s=100, label='买入', zorder=5)
    axes[0].scatter(sell_x, sell_y, color='green', marker='v', s=100, label='卖出', zorder=5)
    axes[0].legend(loc='upper left')

    # Subplot 2: 账户净值
    initial_price = prices[0]
    benchmark = [ORIGINAL_MONEY * (p / initial_price) for p in prices]
    axes[1].plot(dates, net_worths, label='AI 策略净值', color='purple', linewidth=2)
    axes[1].plot(dates, benchmark, label='基准(买入持有)', color='gray', linestyle='--', alpha=0.5)
    axes[1].legend(loc='upper left')

    # Subplot 3: 仓位变化
    axes[2].fill_between(dates, pos_ratio, color='orange', alpha=0.3)
    axes[2].set_ylabel('仓位')

    # Subplot 4: 动作强度
    axes[3].bar(dates, actions, color=np.where(np.array(actions)>0, 'red', 'green'), width=1.0)
    axes[3].set_ylabel('Action')

    # --- Subplot 5: 核心：风险痛感分析 (r_risk) ---
    ax5 = axes[4]
    # 绘制风险惩罚曲线
    ax5.plot(dates, r_risks, color='red', label='风险惩罚 (r_risk)', linewidth=1.5)
    
    # 填充均线保护区：当价格 > MA20 时，惩罚减半的区域
    safe_zone = prices > ma20
    ax5.fill_between(dates, min(r_risks), 0, where=safe_zone, color='green', alpha=0.1, label='MA20 保护开启')
    
    ax5.axhline(0, color='black', linewidth=0.8)
    ax5.set_ylabel('痛感得分')
    ax5.set_title("AI 承受的风险压力 (数值越低越痛)")
    ax5.legend(loc='lower left')

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
            'actions': [], 'pos_ratio': [], 'r_risks': [], 'ma20': []
        }
        
        done = False
        while not done:
            current_date = env.dates[env.today]
            current_price = env.prices[env.today]
            current_ma20 = env.ma20 # 记录当前均线值
            
            action, _ = model.predict(obs, deterministic=True) 
            obs, reward, done, truncated, info = env.step(action)
            
            records['dates'].append(current_date)
            records['prices'].append(current_price)
            records['net_worths'].append(info['net_worth'])
            records['actions'].append(float(action[0]))
            records['pos_ratio'].append(info['pos_ratio'])
            # --- 新增记录 ---
            records['r_risks'].append(info.get('ave_r_risk', 0)) # 记录平均风险分
            records['ma20'].append(current_ma20)

        # 6. 画图
        print(f"✅ 回测完成，正在绘图...")
        plot_backtest_results(code, records)

    manager.close()