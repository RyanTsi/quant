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
    """
    def __init__(self, df_list: list):
        super(SimpleStockEnv, self).__init__()

        self.df_list = df_list
        self.stock_list_len = len(self.df_list)
        
        # --- 核心变量: Alpha ---
        # 这个值可以通过 Callback 在外部动态修改
        self.alpha = 0.05
        
        # 1. 动作空间: [-1, 1] 
        # >0 买入比例, <0 卖出比例
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

        # 初始化一些占位符
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
        self.max_drawdown = 0
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
        # 必须保证有足够的历史数据做 Window (WINDOW_SIZE) 且有足够的未来数据做训练
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
        self.highest_worth_day = self.today
        self.max_drawdown = 0
        
        # 清空 Reward 记录容器
        self.episode_rewards = {
            "r_base": [], "r_risk": [], "r_new_high": []
        }

        # 5. 初始化 Ma/Info 占位符
        self.ma5 = 0
        self.ma20 = 0

        # 6. 初始化历史价格序列 (预热 Window)
        self.stock_history = []
        # 获取当前时间点之前的窗口价格
        current_window_prices = self.prices[self.today - WINDOW_SIZE : self.today + 1]
        
        for i in range(WINDOW_SIZE):
            p_curr = current_window_prices[i]
            p_next = current_window_prices[i+1]
            
            if p_curr == 0: p_curr = 1e-5 # 防止除零
            
            # 计算对数收益率并缩放
            delta_ratio = np.log(p_next / p_curr) * INCR_PARA
            self.stock_history.append(delta_ratio)

        return self._get_observation(), {}

    def _get_observation(self):
        # --- A. 基础价格历史 ---
        history = np.array(self.stock_history.copy(), dtype=np.float32)
        
        # 噪声注入 (Noise Injection): 增加模型鲁棒性
        noise = np.random.normal(0, 0.005, size=history.shape) 
        history = history + noise
        
        # Dropout (Mask): 随机遮盖部分历史数据
        if np.random.rand() < 0.05:
            mask = np.random.binomial(1, 0.9, size=history.shape)
            history = history * mask 

        # --- B. 均线与乖离率 (Technical Indicators) ---
        current_idx = self.today
        # 防止索引越界，取最近 65 天 (计算 Bias60 需要)
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
        
        # 均线距离
        ma_dist5_20 = (self.ma5 - self.ma20) / (self.ma20 + 1e-8) * INCR_PARA

        # --- C. 仓位状态 (Position Status) ---
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
            history,                                # 历史 Log Return
            [bias5, bias20, bias60, ma_dist5_20],   # 技术指标
            [cash_ratio, position_ratio],           # 仓位信息
            [self.alpha]                            # Alpha 参数
        ]).astype(np.float32)
        
        return obs

    def step(self, action):
        # --- 1. 执行交易 (Time T) ---
        current_price = self.prices[self.today]
        prev_net_worth = self.my_cash + self.number_of_shares * current_price
        
        # 截断动作范围
        act = np.clip(action[0], -1, 1)
        
        # 交易逻辑
        if act > 0: # 买入
            can_buy_cash = self.my_cash * act
            shares_to_buy = int(can_buy_cash // current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * 1.0005 # 买入费率 0.05%
                self.my_cash -= cost
                self.number_of_shares += shares_to_buy
        elif act < 0: # 卖出
            shares_to_sell = int(self.number_of_shares * abs(act))
            if shares_to_sell > 0:
                gain = shares_to_sell * current_price * 0.9995 # 卖出费率 0.05%
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

        # B. 风险调整 (Risk Penalty)
        # 动态计算回撤幅度
        drawdown = (self.highest_worth - current_net_worth) / self.highest_worth * INCR_PARA
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # 时间惩罚系数 (目前设为1，可调整)
        time_penalty = 1
        r_risk_down = drawdown * time_penalty
        
        r_risk = -r_risk_down
        r_risk *= 0.005 # 缩放系数
        

        # --- 5. 最终加权 (Combined Reward) ---
        # 使用 self.alpha 动态调节风险权重
        total_reward = r_base + (r_risk * self.alpha) 

        # 裁剪防止梯度爆炸
        total_reward = np.clip(total_reward, -10.0, 10.0)

        # --- 6. 记录统计信息 ---
        self.episode_rewards["r_base"].append(r_base)
        self.episode_rewards["r_risk"].append(r_risk)

        # 构造 Info
        info = {
            "net_worth": float(current_net_worth),
            "max_drawdown": float(self.max_drawdown),
            "alpha": float(self.alpha),
            "pos_ratio": float((self.number_of_shares * next_price) / current_net_worth) if current_net_worth > 0 else 0,
            # 统计均值供 Callback 使用
            "ave_r_base": np.mean(self.episode_rewards["r_base"]) if self.episode_rewards["r_base"] else 0,
            "ave_r_risk": np.mean(self.episode_rewards["r_risk"]) if self.episode_rewards["r_risk"] else 0,
        }

        # --- 7. 更新历史状态 (Rolling Update) ---
        # 计算 T 到 T+1 的收益率，推入 history
        if prev_net_worth == 0: prev_net_worth = 1e-8 # 理论上不应发生
        delta_ratio = np.log(next_price / current_price) * INCR_PARA
        
        self.stock_history.pop(0)
        self.stock_history.append(delta_ratio)

        # 返回 (observation, reward, terminated, truncated, info)
        return self._get_observation(), float(total_reward), terminated, False, info