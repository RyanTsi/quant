import numpy as np
import gymnasium as gym
from gymnasium import spaces
from config import *
import random

class SimpleStockEnv(gym.Env):
    """
    单一股票交易环境 (适配单 CPU + 自适应 Alpha)
    """
    def __init__(self, df_list: list):
        super(SimpleStockEnv, self).__init__()

        self.df_list = df_list
        self.stock_list_len = len(self.df_list)
        
        # --- 核心修改 1: Alpha 初始值 ---
        # 这里的 alpha 会被 Callback 通过 set_attr 动态修改
        self.alpha = 0.05
        
        # 1. 动作空间: [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # 2. 状态空间: Window + 6 个特征
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(WINDOW_SIZE + 6,), dtype=np.float32)

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
        
        self.today = start_index + WINDOW_SIZE

        # 3. 初始化账户
        self.my_cash = ORIGINAL_MONEY 
        self.number_of_shares = 0 

        # 4. 初始化统计指标 (Reset 时必须重置，否则 Info 会污染)
        self.highest_worth = ORIGINAL_MONEY
        self.highest_worth_day = self.today
        self.max_drawdown = 0
        
        # 统计平均 Reward 的容器
        self.episode_rewards = {
            "r_base": [], "r_risk": [], "r_new_high": []
        }

        # 5. 初始化 Ma/Info 占位符
        self.ma5 = 0
        self.ma20 = 0

        # 6. 初始化历史价格序列
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
        # 基础价格历史
        history = np.array(self.stock_history.copy(), dtype=np.float32)
        
        # 噪声注入 (Noise Injection)
        noise = np.random.normal(0, 0.01, size=history.shape) 
        history = history + noise
        if np.random.rand() < 0.05:
            mask = np.random.binomial(1, 0.9, size=history.shape)
            history = history * mask 

        # 均线与乖离率 (Bias)
        current_idx = self.today
        # 防止索引越界，取最近 65 天
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
        
        # 均线距离 (避免除零)
        ma_dist5_20 = (self.ma5 - self.ma20) / (self.ma20 + 1e-8) * INCR_PARA

        # 仓位状态
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

    def step(self, action):
        # --- 1. 执行交易 (Time T) ---
        current_price = self.prices[self.today]
        prev_net_worth = self.my_cash + self.number_of_shares * current_price
        
        act = np.clip(action[0], -1, 1)
        
        # 交易逻辑
        if act > 0: # 买入
            can_buy_cash = self.my_cash * act
            shares_to_buy = int(can_buy_cash // current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * 1.0005 # 手续费
                self.my_cash -= cost
                self.number_of_shares += shares_to_buy
        elif act < 0: # 卖出
            shares_to_sell = int(self.number_of_shares * abs(act))
            if shares_to_sell > 0:
                gain = shares_to_sell * current_price * 0.9995 # 手续费
                self.my_cash += gain
                self.number_of_shares -= shares_to_sell

        # --- 2. 时间流逝 (T -> T+1) ---
        terminated = self.today >= self.last_day
        if not terminated:
            self.today += 1 
        
        # --- 3. 结算 (Time T+1) ---
        next_price = self.prices[self.today]
        current_net_worth = self.my_cash + self.number_of_shares * next_price

        # 更新最高资产记录 (用于计算回撤)
        if current_net_worth > self.highest_worth:
            self.highest_worth_day = self.today
            self.highest_worth = current_net_worth

        # --- 4. Reward 计算 ---
        
        # A. 基础收益 (Base Reward)
        r_base = np.log((current_net_worth + 1e-8) / (prev_net_worth + 1e-8)) * INCR_PARA 

        # B. 风险调整 (Risk)
        # 动态计算回撤幅度
        drawdown = (self.highest_worth - current_net_worth) / self.highest_worth * INCR_PARA
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # 随着回撤天数增加，惩罚加重 (最大约 2.4 倍)
        time_penalty = np.log1p(min(10, self.today - self.highest_worth_day))
        
        r_risk_down = drawdown * time_penalty
        
        # 如果价格在20日均线之上，说明可能是良性回调，惩罚减半
        if next_price > self.ma20:
            r_risk_down *= 0.5

        # 资产回升给予微量补偿
        # r_repair = min((current_net_worth - prev_net_worth) / self.highest_worth, r_risk_down * 0.5) # to check
        r_repair = 0
        # 风险分
        r_risk = -r_risk_down + r_repair
        r_risk *= 0.1
        # C. 创新高 (简化)
        r_new_high = 0.0
        # 这里去掉了 target_value 的循环逻辑，简化为单纯的 Log 收益驱动

        # --- 5. 最终加权 (Combined Reward) ---
        # 调整量级：如果 r_risk_down 太大，可以乘个系数比如 0.5
        total_reward = r_base + (r_risk * self.alpha) 

        # 裁剪防止梯度爆炸
        total_reward = np.clip(total_reward, -10.0, 10.0)

        # --- 6. 记录统计信息 ---
        self.episode_rewards["r_base"].append(r_base)
        self.episode_rewards["r_risk"].append(r_risk)
        self.episode_rewards["r_new_high"].append(r_new_high)

        # 构造 Info
        info = {
            "net_worth": float(current_net_worth),
            "max_drawdown": float(self.max_drawdown),
            "alpha": float(self.alpha),
            "pos_ratio": float((self.number_of_shares * next_price) / current_net_worth) if current_net_worth > 0 else 0,
            # 计算本局目前的平均值，供 Callback 使用
            "ave_r_base": np.mean(self.episode_rewards["r_base"]),
            "ave_r_risk": np.mean(self.episode_rewards["r_risk"]),
            "max_r_base": np.max(np.abs(self.episode_rewards["r_base"])),
            "max_r_risk": np.max(np.abs(self.episode_rewards["r_risk"])),
        }

        # 更新历史状态 (Rolling Update)
        if prev_net_worth == 0: prev_net_worth = 1e-8
        delta_ratio = np.log(next_price / current_price) * INCR_PARA
        
        self.stock_history.pop(0)
        self.stock_history.append(delta_ratio)

        return self._get_observation(), float(total_reward), terminated, False, info