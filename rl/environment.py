import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

# ==========================================
# Config 部分 
# ==========================================
try:
    from config import *
except ImportError:
    # 默认配置（如果 config.py 不存在）
    WINDOW_SIZE = 30       
    TRAINING_DAYS = 200    
    ORIGINAL_MONEY = 100000 
    INCR_PARA = 100.0      

class SimpleStockEnv(gym.Env):
    """
    单一股票交易环境 (增强版)
    
    特性:
    1. 盘中价格模拟: 线性模型 + 布朗桥噪声 (Brownian Bridge Noise)
    2. 风险控制: 回撤扩张惩罚 (Drawdown Expansion Penalty)
    3. 状态空间: 增加 '距离收盘时间' 特征
    """
    def __init__(self, df_list: list):
        super(SimpleStockEnv, self).__init__()

        self.df_list = df_list
        self.stock_list_len = len(self.df_list)
        
        # 风险厌恶系数
        self.alpha = 0.05
        
        # 1. 动作空间: [-1, 1] (买入/卖出比例)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # 2. 状态空间: 
        # [0:WINDOW_SIZE] -> 历史价格 Log Return
        # [+0..+3] -> 技术指标 (bias, ma_dist)
        # [+4..+5] -> 仓位信息 (cash, position)
        # [+7] -> open_gap (开盘相对于昨收的涨跌幅)
        # [+7] -> time_remaining (0.0=收盘, 1.0=开盘)
        # [+8] -> alpha
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(WINDOW_SIZE + 9,), 
            dtype=np.float32
        )

        # 数据容器
        self.current_df = None
        self.prices_close = None 
        self.prices_open = None  
        self.current_price = 0.0
        
        # 状态指针
        self.today = 0
        self.last_day = 0
        self.my_cash = 0
        self.number_of_shares = 0
        self.stock_history = []
        self.ma5 = 0
        self.ma20 = 0
        
        # 时间模拟 (1.0 = 开盘, 0.0 = 收盘)
        self.time_remaining = 0.0 
        
        # 统计指标
        self.highest_worth = 0
        self.max_drawdown_cur = 0    # 当前波段最大回撤
        self.max_drawdown_global = 0 # 全局最大回撤
        self.episode_rewards = {}

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- 1. 数据准备 ---
        self.alpha = np.random.uniform(0, 1)
        stock_idx = random.randint(0, self.stock_list_len - 1)
        self.current_df = self.df_list[stock_idx]
        
        self.prices_close = self.current_df['收盘'].values.astype(np.float32)
        # 尝试获取开盘价，如果没有则用收盘价兜底
        if '开盘' in self.current_df.columns:
            self.prices_open = self.current_df['开盘'].values.astype(np.float32)
        else:
            self.prices_open = self.prices_close

        total_len = len(self.prices_close)
        
        # --- 2. 时间窗口选择 ---
        valid_range_len = total_len - WINDOW_SIZE - TRAINING_DAYS
        if valid_range_len <= 0:
            start_index = 0
            self.last_day = total_len - 1
        else:
            start_index = random.randint(0, valid_range_len - 1)
            self.last_day = start_index + WINDOW_SIZE + TRAINING_DAYS
        
        self.today = start_index + WINDOW_SIZE

        # --- 3. 账户与统计重置 ---
        self.my_cash = ORIGINAL_MONEY 
        self.number_of_shares = 0 
        
        self.highest_worth = ORIGINAL_MONEY
        self.max_drawdown_cur = 0 
        self.max_drawdown_global = 0
        
        self.episode_rewards = {
            "r_base": [], "r_base_pos": [], "r_base_neg": [], "r_risk": []
        }

        # --- 4. 初始化模拟价格 ---
        # 随机生成一个初始时间点
        self.time_remaining = random.random() 
        self.current_price = self._calculate_noisy_price(self.today, self.time_remaining)

        # --- 5. 历史数据预热 (Window) ---
        self.ma5 = 0; self.ma20 = 0
        self.stock_history = []
        
        # 这里的 history 依然使用标准的日线收盘价 (Stable features)
        current_window_prices = self.prices_close[self.today - WINDOW_SIZE : self.today + 1]
        
        for i in range(WINDOW_SIZE):
            p_curr = current_window_prices[i]
            p_next = current_window_prices[i+1]
            if p_curr == 0: p_curr = 1e-5 
            delta_ratio = np.log(p_next / p_curr) * INCR_PARA
            self.stock_history.append(delta_ratio)

        return self._get_observation(), {}

    def _calculate_noisy_price(self, day_idx, time_rem):
        """
        核心函数: 计算带噪声的盘中价格
        公式: P = Linear(Open, Close) + Noise
        特性: 噪声在 Open 和 Close 处为 0，在中间最大
        """
        p_open = self.prices_open[day_idx]
        p_close = self.prices_close[day_idx]
        
        # 1. 线性基准 (Linear Baseline)
        # time_rem: 1.0 (Open) -> 0.0 (Close)
        linear_price = p_open + (p_close - p_open) * (1.0 - time_rem)
        
        # 2. 动态波动率估计 (Volatility Estimation)
        # 估计当天的波动范围，如果 Open==Close，给一个极小的默认波动
        daily_volatility = abs(p_close - p_open) + (p_open * 0.002) 
        
        # 3. 布朗桥噪声系数 (Bridge Factor)
        # 在 time=1 和 time=0 时为 0，在 time=0.5 时最大(0.25)
        # 乘以 4 是为了让中间的系数归一化到 1.0 左右
        bridge_factor = time_rem * (1.0 - time_rem) * 4.0 
        
        # 4. 生成噪声
        # np.random.normal(0, 0.1) -> 假设标准差为日波动的 10%
        noise = np.random.normal(0, 0.1) * daily_volatility * bridge_factor
        
        final_price = linear_price + noise
        
        # 兜底防止负价格
        return max(1e-5, final_price)

    def _get_observation(self):
        # --- A. 历史序列 ---
        history = np.array(self.stock_history.copy(), dtype=np.float32)
        # 观测噪声 (Observation Noise)
        noise = np.random.normal(0, 0.005, size=history.shape) 
        history = history + noise
        # 随机 Dropout
        if np.random.rand() < 0.05:
            history = history * np.random.binomial(1, 0.9, size=history.shape)

        # --- B. 技术指标 (基于收盘价) ---
        current_idx = self.today
        start_idx = max(0, current_idx - 65)
        window_prices = self.prices_close[start_idx : current_idx + 1]
        # 获取昨天的收盘价
        if self.today > 0:
            prev_close = self.prices_close[self.today - 1]
        else:
            # 如果是第一天，用当天的开盘价模拟昨收，即认为没有跳空
            prev_close = self.prices_open[self.today]
            
        current_open = self.prices_open[self.today]
        
        # 防止除零 (虽然股价通常不为0)
        if prev_close <= 0: prev_close = 1e-5
        
        # 计算 Log Return 并缩放
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

        # --- C. 仓位 (基于模拟成交价) ---
        # 使用 self.current_price 即使盘中计算净值
        current_net_worth = self.my_cash + self.number_of_shares * self.current_price
        
        if current_net_worth <= 0:
            cash_ratio, position_ratio = 0.0, 0.0
        else:
            cash_ratio = self.my_cash / current_net_worth
            position_ratio = 1.0 - cash_ratio
        
        # --- D. 拼接特征 ---
        obs = np.concatenate([
            history, 
            [bias5, bias20, bias60, ma_dist5_20],
            [cash_ratio, position_ratio],
            [open_gap],
            [self.time_remaining],
            [self.alpha]
        ]).astype(np.float32)
        
        return obs

    def step(self, action):
        # --- 1. 执行交易 (基于当前模拟价格) ---
        # 此时的 self.current_price 是上一次 step 或 reset 计算出的
        prev_net_worth = self.my_cash + self.number_of_shares * self.current_price
        
        act = np.clip(action[0], -1, 1)
        
        if act > 0: # 买入
            can_buy_cash = self.my_cash * act
            shares_to_buy = int(can_buy_cash // self.current_price)
            if shares_to_buy > 0:
                # 包含万分之五滑点/手续费
                cost = shares_to_buy * self.current_price * 1.0005 
                self.my_cash -= cost
                self.number_of_shares += shares_to_buy
        elif act < 0: # 卖出
            shares_to_sell = int(self.number_of_shares * abs(act))
            if shares_to_sell > 0:
                gain = shares_to_sell * self.current_price * 0.9995 
                self.my_cash += gain
                self.number_of_shares -= shares_to_sell

        # --- 2. 状态推演 (T -> T+1) ---
        terminated = self.today >= self.last_day
        if not terminated:
            self.today += 1 
            
        # --- 3. 结算净值 ---
        current_net_worth = self.my_cash + self.number_of_shares * self.prices_close[self.today]

        # 创新高逻辑
        if current_net_worth > self.highest_worth:
            self.max_drawdown_cur = 0 # 重置回撤
            self.highest_worth = current_net_worth

        # --- 4. Reward 计算 ---
        
        # A. 基础收益
        r_base = np.log((current_net_worth + 1e-8) / (prev_net_worth + 1e-8)) * INCR_PARA 

        # B. 风险调整 (回撤扩张惩罚)
        drawdown = (self.highest_worth - current_net_worth) / self.highest_worth * INCR_PARA
        
        r_risk_down = 0.0
        # 只有当回撤【扩大】时才惩罚
        if drawdown > self.max_drawdown_cur:
            r_risk_down = (drawdown - self.max_drawdown_cur)
        
        # 更新记录
        self.max_drawdown_cur = max(self.max_drawdown_cur, drawdown)
        self.max_drawdown_global = max(self.max_drawdown_global, drawdown)

        # 风险惩罚项
        r_risk = -r_risk_down * self.alpha * 0.8
        
        # 最终 Reward
        total_reward = r_base + r_risk
        total_reward = np.clip(total_reward, -10.0, 10.0)

        # --- 6. 统计记录 ---
        self.episode_rewards["r_base"].append(r_base)
        self.episode_rewards["r_risk"].append(r_risk)
        
        if r_base > 0:
            self.episode_rewards["r_base_pos"].append(r_base)
        elif r_base < 0:
            self.episode_rewards["r_base_neg"].append(r_base)
        
        def safe_mean(lst): return np.mean(lst) if len(lst) > 0 else 0.0

        info = {
            "net_worth": float(current_net_worth),
            "price": float(self.current_price),    # 当前成交价
            "time_rem": float(self.time_remaining),# 离收盘还有多久
            "max_dd_global": float(self.max_drawdown_global),
            "max_dd_cur": float(self.max_drawdown_cur),
            "alpha": float(self.alpha),
            
            "ave_r_base": safe_mean(self.episode_rewards["r_base"]),
            "ave_r_base_pos": safe_mean(self.episode_rewards["r_base_pos"]),
            "ave_r_base_neg": safe_mean(self.episode_rewards["r_base_neg"]),
            "ave_r_risk": safe_mean(self.episode_rewards["r_risk"]),
        }

        # --- 7. 更新历史序列 (Rolling) ---
        # 使用收盘价更新 History (保持特征稳定性)
        p_close_curr = self.prices_close[self.today-1] if self.today > 0 else self.prices_close[0]
        p_close_next = self.prices_close[self.today]
        delta_ratio = np.log(p_close_next / p_close_curr) * INCR_PARA
        
        self.stock_history.pop(0)
        self.stock_history.append(delta_ratio)

        return self._get_observation(), float(total_reward), terminated, False, info