import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from config import *


class SimpleStockEnv(gym.Env):
    """
    单一股票交易环境 
    特性: 
    1. 适配单 CPU 训练
    2. 自适应 Alpha (风险厌恶系数)
    3. Alpha 作为一个特征输入到 Observation 中
    4. 分别统计 r_base 的正向均值和负向均值
    """
    def __init__(self, df_list: list):
        super(SimpleStockEnv, self).__init__()

        self.df_list = df_list
        self.stock_list_len = len(self.df_list)
        
        # --- 核心变量: Alpha ---
        self.alpha = 0.05
        
        # 1. 动作空间: [-1, 1] 
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # 2. 状态空间: 
        # [0:WINDOW_SIZE] -> 历史价格 Log Return 序列
        # [+0] -> bias5
        # [+1] -> bias20
        # [+2] -> bias60
        # [+3] -> ma_dist5_20
        # [+4] -> cash_ratio
        # [+5] -> position_ratio
        # [+6] -> alpha
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(WINDOW_SIZE + 7,), 
            dtype=np.float32
        )

        # 初始化占位符
        self.current_df = None
        self.prices = None
        self.today = 0
        self.last_day = 0
        self.my_cash = 0
        self.number_of_shares = 0
        self.stock_history = []
        self.ma5 = 0
        self.ma20 = 0
        
        # 统计相关
        self.highest_worth = 0
        self.max_drawdown_cur = 0
        self.max_drawdown_rec = 0
        self.episode_rewards = {}

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. 随机选择一只股票
        stock_idx = random.randint(0, self.stock_list_len - 1)
        self.current_df = self.df_list[stock_idx]
        
        self.prices = self.current_df['收盘'].values.astype(np.float32)
        total_len = len(self.prices)
        
        # 2. 随机选择时间段
        valid_range_len = total_len - WINDOW_SIZE - TRAINING_DAYS
        
        if valid_range_len <= 0:
            start_index = 0
            self.last_day = total_len - 1
        else:
            start_index = random.randint(0, valid_range_len - 1)
            self.last_day = start_index + WINDOW_SIZE + TRAINING_DAYS
        
        # 当前时间指针
        self.today = start_index + WINDOW_SIZE

        # 3. 初始化账户
        self.my_cash = ORIGINAL_MONEY 
        self.number_of_shares = 0 

        # 4. 初始化统计指标
        self.highest_worth = ORIGINAL_MONEY
        self.max_drawdown_cur = 0
        self.max_drawdown_rec = 0
        self.episode_rewards = {
            "r_base": [],      # 所有的 base reward
            "r_base_pos": [],  # 盈利
            "r_base_neg": [],  # 亏损
            "r_risk": []       # 所有的 risk reward
        }

        # 5. 初始化 Ma/Info 占位符
        self.ma5 = 0
        self.ma20 = 0

        # 6. 初始化历史价格序列 (预热 Window)
        self.stock_history = []
        current_window_prices = self.prices[self.today - WINDOW_SIZE : self.today + 1]
        
        for i in range(WINDOW_SIZE):
            p_curr = current_window_prices[i]
            p_next = current_window_prices[i+1]
            
            if p_curr == 0: p_curr = 1e-5 
            
            delta_ratio = np.log(p_next / p_curr) * INCR_PARA
            self.stock_history.append(delta_ratio)

        return self._get_observation(), {}

    def _get_observation(self):
        # --- A. 基础价格历史 ---
        history = np.array(self.stock_history.copy(), dtype=np.float32)
        
        # 噪声注入
        noise = np.random.normal(0, 0.005, size=history.shape) 
        history = history + noise
        
        # Dropout
        if np.random.rand() < 0.05:
            mask = np.random.binomial(1, 0.9, size=history.shape)
            history = history * mask 

        # --- B. 均线与乖离率 ---
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
        
        ma_dist5_20 = (self.ma5 - self.ma20) / (self.ma20 + 1e-8) * INCR_PARA

        # --- C. 仓位状态 ---
        current_price = self.prices[self.today]
        current_net_worth = self.my_cash + self.number_of_shares * current_price
        
        if current_net_worth <= 0:
            cash_ratio = 0.0
            position_ratio = 0.0
        else:
            cash_ratio = self.my_cash / current_net_worth
            position_ratio = 1.0 - cash_ratio
        
        # --- D. 拼接最终 Observation ---
        obs = np.concatenate([
            history,                            # 历史 Log Return
            [bias5, bias20, bias60, ma_dist5_20], # 技术指标
            [cash_ratio, position_ratio],       # 仓位信息
            [self.alpha]                        # Alpha 参数
        ]).astype(np.float32)
        
        return obs

    def step(self, action):
        # --- 1. 执行交易 (Time T) ---
        current_price = self.prices[self.today]
        prev_net_worth = self.my_cash + self.number_of_shares * current_price
        
        act = np.clip(action[0], -1, 1)
        
        if act > 0: # 买入
            can_buy_cash = self.my_cash * act
            shares_to_buy = int(can_buy_cash // current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * 1.0005 
                self.my_cash -= cost
                self.number_of_shares += shares_to_buy
        elif act < 0: # 卖出
            shares_to_sell = int(self.number_of_shares * abs(act))
            if shares_to_sell > 0:
                gain = shares_to_sell * current_price * 0.9995 
                self.my_cash += gain
                self.number_of_shares -= shares_to_sell

        # --- 2. 时间流逝 (T -> T+1) ---
        terminated = self.today >= self.last_day
        if not terminated:
            self.today += 1 
        
        # --- 3. 结算 (Time T+1) ---
        next_price = self.prices[self.today]
        current_net_worth = self.my_cash + self.number_of_shares * next_price

        if current_net_worth > self.highest_worth:
            self.max_drawdown_cur = 0
            self.highest_worth = current_net_worth

        # --- 4. Reward 计算 ---
        
        # A. 基础收益 (Base Reward)
        r_base = np.log((current_net_worth + 1e-8) / (prev_net_worth + 1e-8)) * INCR_PARA 

        # B. 风险调整 (Risk Penalty)
        drawdown = (self.highest_worth - current_net_worth) / self.highest_worth * INCR_PARA
        r_risk_down = 0
        if drawdown > self.max_drawdown_cur:
            r_risk_down = (drawdown - self.max_drawdown_cur)
        self.max_drawdown_cur = max(self.max_drawdown_cur, drawdown)
        self.max_drawdown_rec = max(self.max_drawdown_rec, drawdown)

        
        # 这里使用了你提供的简化版 risk 计算逻辑
        r_risk = -r_risk_down * self.alpha * 0.8
        
        # --- 5. 最终加权 (Combined Reward) ---
        total_reward = r_base + r_risk
        total_reward = np.clip(total_reward, -10.0, 10.0)

        # --- 6. 记录统计信息 ---
        
        # 记录所有的 base reward
        self.episode_rewards["r_base"].append(r_base)
        self.episode_rewards["r_risk"].append(r_risk)
        
        # --- 修改点 2: 分别记录正负收益 ---
        if r_base > 0:
            self.episode_rewards["r_base_pos"].append(r_base)
        elif r_base < 0:
            self.episode_rewards["r_base_neg"].append(r_base)
        
        # 辅助函数：计算均值，如果是空列表则返回 0
        def safe_mean(lst):
            return np.mean(lst) if len(lst) > 0 else 0.0

        # 构造 Info
        info = {
            "net_worth": float(current_net_worth),
            "max_drawdown": float(self.max_drawdown_rec),
            "alpha": float(self.alpha),
            "pos_ratio": float((self.number_of_shares * next_price) / current_net_worth) if current_net_worth > 0 else 0,
            
            # 全体均值
            "ave_r_base": safe_mean(self.episode_rewards["r_base"]),
            # 盈利时的平均值 (应该 > 0)
            "ave_r_base_pos": safe_mean(self.episode_rewards["r_base_pos"]),
            # 亏损时的平均值 (应该 < 0)
            "ave_r_base_neg": safe_mean(self.episode_rewards["r_base_neg"]),
            
            "ave_r_risk": safe_mean(self.episode_rewards["r_risk"]),
        }

        # --- 7. 更新历史状态 (Rolling Update) ---
        if prev_net_worth == 0: prev_net_worth = 1e-8 
        delta_ratio = np.log(next_price / current_price) * INCR_PARA
        
        self.stock_history.pop(0)
        self.stock_history.append(delta_ratio)

        return self._get_observation(), float(total_reward), terminated, False, info