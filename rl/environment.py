import numpy as np
import gymnasium as gym
from gymnasium import spaces
from config import *
import random

class SimpleStockEnv(gym.Env):
    """
    单一股票交易环境
    """
    def __init__(self, df_list: list):
        super(SimpleStockEnv, self).__init__()

        self.df_list = df_list
        self.stock_list_len = len(self.df_list)
        self.current_step = 0
        # 1. 动作空间 (Action Space): 
        # 范围 [-1, 1]。 1 表示全仓买入，-1 表示全仓卖出，0 表示观望
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # 2. 状态空间 (Observation Space): 
        # 我们给 AI 看 windowsize + 4 + 2 个维度：[[当前对比前一天的涨幅(对数)], BIMA, 我的现金占总资产的比例, 当前持仓状态(0到1, 1-现金占比)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(WINDOW_SIZE + 6,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        # 必须调用 super().reset 来处理随机种子
        super().reset(seed=seed)
        
        # 1. 随机选择一只股票
        stock_idx = random.randint(0, self.stock_list_len - 1)
        self.current_df = self.df_list[stock_idx]
        self.prices = self.current_df['收盘'].values.astype(np.float32)
        total_len = len(self.prices)
        
        # 2. 随机选择开始时间
        # 确保：start_index + WINDOW_SIZE + TRAINING_DAYS <= total_len
        # max_start_index 是我们可以开始交易的最晚的一天的索引（相对于 prices 数组）
        # 注意：self.today 指向的是“当前看到的价格”的索引
        valid_range_len = total_len - WINDOW_SIZE - TRAINING_DAYS
        if valid_range_len <= 0:
            # 理论上预处理已经过滤了，防止万一，这里做个兜底，如果数据不够长，就固定从头开始
            start_index = 0
            self.last_day = total_len - 1
        else:
            # 随机选择一个起跑点
            start_index = random.randint(0, valid_range_len - 1)
            # 结束点是起跑点 + 训练时长
            self.last_day = start_index + WINDOW_SIZE + TRAINING_DAYS
        self.today = start_index + WINDOW_SIZE

        # 3. 初始化账户
        self.my_cash = ORIGINAL_MONEY  # 现金
        self.number_of_shares = 0 # 持股数

        # 4. 初始化参数
        self.alpha = 1.0
        self.target_value = NEW_HIGH_TARGET
        self.new_high_reward = NEW_HIGH_REWARD
        self.highest_worth_day = self.today
        self.highest_worth = ORIGINAL_MONEY

        # 5. 初始化 info 参数
        self.ma5 = 0
        self.ma20 = 0
        self.ave_r_base = 0
        self.ave_r_risk = 0
        self.ave_r_new_high = 0

        self.max_r_base = 0
        self.max_r_risk = 0
        self.max_r_new_high = 0

        self.max_drawdown = 0
        # 5. 初始化状态历史
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
        history = np.array(self.stock_history.copy(), dtype=np.float32)
        noise = np.random.normal(0, 0.01, size=history.shape) # noise 1 new add
        history = history + noise
        if np.random.rand() < 0.05:
            mask = np.random.binomial(1, 0.9, size=history.shape)
            history = history * mask # noise 2 new add

        # ma 派生特征
        current_idx = self.today
        lookback = 65 
        start_idx = max(0, current_idx - lookback)
        window_prices = self.prices[start_idx : current_idx + 1]
        current_p = window_prices[-1]
        
        def get_bias(p_array, period):
            if len(p_array) < period:
                return 0.0
            ma = np.mean(p_array[-period:])
            return (p_array[-1] - ma) / ma * INCR_PARA
        def get_ma(p_array, period):
            return np.mean(window_prices[-period:])
        
        bias5  = get_bias(window_prices, 5)
        bias20 = get_bias(window_prices, 20)
        bias60 = get_bias(window_prices, 60)
        # 均线距离
        self.ma5 = get_ma(window_prices, 5)
        self.ma20 = get_ma(window_prices, 20)
        ma_dist5_20 = (self.ma5 - self.ma20) / self.ma20 * INCR_PARA # new add

        # my data
        current_price = self.prices[self.today]
        current_net_worth = self.my_cash + self.number_of_shares * current_price
        cash_ratio = self.my_cash / current_net_worth if current_net_worth > 0 else 0.0
        position_ratio = 1.0 - cash_ratio
        
        obs = np.concatenate([
            history, 
            [bias5, bias20, bias60, ma_dist5_20],
            [cash_ratio, position_ratio]
        ]).astype(np.float32)
        return obs

    def step(self, action):
        # --- 1. 此时此刻 (Time T) ---
        # 当前价格
        current_price = self.prices[self.today]
        # 记录交易前的资产总额 (T 时刻)
        prev_net_worth = self.my_cash + self.number_of_shares * current_price
        # 交易前的仓位
        prev_pos_ratio = (self.number_of_shares * current_price) / prev_net_worth
        # 执行交易：按当前价格 current_price 成交
        act = np.clip(action[0], -1, 1)
        if act > 0:
            shares_to_buy = int((self.my_cash * act) // current_price)
            self.my_cash -= shares_to_buy * current_price * 1.0005  # 扣除交易费用 0.05%
            self.number_of_shares += shares_to_buy
        elif act < 0:
            shares_to_sell = int(self.number_of_shares * abs(act))
            self.my_cash += shares_to_sell * current_price * 0.9995  # 扣除交易费用 0.05%
            self.number_of_shares -= shares_to_sell

        # --- 2. 时间流逝 (T -> T+1) ---
        terminated = self.today >= self.last_day
        if not terminated:
            self.today += 1 
        
        # --- 3. 到达下一刻 (Time T+1) ---
        # 新时刻的价格
        next_price = self.prices[self.today]
        # 计算新的总资产
        current_net_worth = self.my_cash + self.number_of_shares * next_price

        if current_net_worth > self.highest_worth: # new add
            self.highest_worth_day = self.today
            self.highest_worth = current_net_worth
        # 计算新的仓位 (用于 Risk Reward)
        # 注意：这里用的是 T+1 时刻的仓位价值占比
        current_pos_value = self.number_of_shares * next_price
        current_pos_ratio = current_pos_value / current_net_worth if current_net_worth > 0 else 0

        # --- 4. Reward 计算核心 ---
        
        # A. 基础收益 (Base Reward)
        r_base = np.log(current_net_worth / prev_net_worth) * 10 # [-1, 0.9]
        # B. 风险调整 (Risk-Adjusted Reward)
        # B1. 回撤惩罚 (Drawdown Penalty)
        # 约 10 天达到最大
        time_penalty_factor = np.log1p(min(10, self.today - self.highest_worth_day)) # 最大 2.4
        drawdown = (self.highest_worth - current_net_worth) / self.highest_worth
        r_risk_down = drawdown * time_penalty_factor # new add
        if current_price > self.ma20:
            r_risk_down *= 0.5
        # B2. 恢复补偿
        r_repair = min((current_net_worth - prev_net_worth) / self.highest_worth, r_risk_down * 0.8)  # new add
        r_risk = -r_risk_down + r_repair
        
        # C. 创新高奖励 (New High Reward)
        r_new_high = 0
        while current_net_worth >= ORIGINAL_MONEY * self.target_value:
            r_new_high += self.new_high_reward  # 给予一次性大奖 (例如 1.0)
            self.target_value *= 1.1 # 提高目标

        # --- 5. 总 Reward 汇总 ---
        # 权重分配：
        r_base *= 1
        r_risk *= 0.1 * (self.alpha - 1)
        r_new_high *= 0 # r_base 已经足够，不需要新的非线性正向reward了

        total_reward =  r_base + r_risk + r_new_high
        

        # 裁剪，防止梯度爆炸
        total_reward = np.clip(total_reward, -10.0, 10.0)

        self.max_r_base = max(self.max_r_base, abs(r_base))
        self.max_r_risk = max(self.max_r_risk, abs(r_risk))
        self.max_r_new_high = max(self.max_r_new_high, abs(r_new_high))

        self.ave_r_base = self.ave_r_base * self.times + r_base
        self.ave_r_risk = self.ave_r_risk * self.times + r_risk
        self.ave_r_new_high = self.ave_r_new_high * self.times + r_new_high

        self.ave_r_base /= self.times
        self.ave_r_risk /= self.times
        self.ave_r_new_high /= self.times

        info = {
            "net_worth": float(current_net_worth),
            "ave_r_base": self.ave_r_base,
            "ave_r_risk": self.ave_r_risk,
            "ave_r_new_high": self.ave_r_new_high,
            "max_r_base": self.max_r_base,
            "max_r_risk": self.max_r_risk,
            "max_r_new_high": self.max_r_new_high,
            "max_drawdown": self.max_drawdown,
            "alpha": self.alpha,
            "ma5": self.ma5,
            "ma20": self.ma20,
            "pos_ratios": current_pos_ratio
        }

        # 更新历史状态
        self.times += 1
        self.alpha = min(2.0, self.alpha * 1.0000011552459682)
        delta_ratio = np.log(next_price / current_price) * INCR_PARA
        self.stock_history.pop(0)
        self.stock_history.append(delta_ratio)
        # 返回的是 T+1 的观察值，AI 将根据这个新状态做下一次(T+1)决策
        return self._get_observation(), float(total_reward), terminated, False, info