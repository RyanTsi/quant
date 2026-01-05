import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
from config import *

   

class SimpleStockEnv(gym.Env):
    """
    单一股票交易环境 (增强版 - v2)
    
    修改记录:
    1. Alpha 分布优化: 混合分布 (Uniform + Normal)
    2. 风险权重调整: Lambda = 1.0
    3. 新增机制: 现金无风险收益 (防止横盘躺平)
    """
    def __init__(self, df_list: list):
        super(SimpleStockEnv, self).__init__()

        self.df_list = df_list
        self.stock_list_len = len(self.df_list)
        
        # 风险厌恶系数 (将在 reset 中动态生成)
        self.alpha = 0.5
        
        # [配置] 年化无风险利率 (2.5%) -> 用于计算现金奖励
        self.rf_annual = 0.025
        self.rf_daily = (1 + self.rf_annual) ** (1/250) - 1
        
        self.turnover_coef = 0.002 * INCR_PARA
        # 1. 动作空间: [-1, 1] (买入/卖出比例)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # 2. 状态空间: 
        # [0:WINDOW_SIZE] -> 历史价格 Log Return
        # [+0..+3] -> 技术指标 (bias, ma_dist)
        # [+4..+5] -> 仓位信息 (cash, position)
        # [+6] -> open_gap (开盘相对于昨收的涨跌幅)
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
        self.last_action = 0.0

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
        
        # --- 1. Alpha 设置策略 ---
        rand_val = np.random.random()
        if rand_val < 0.8:
            self.alpha = float(np.random.uniform(0.0, 1.0))
        else:
            # 均值 0.5, 标准差 0.15, 截断在 [0, 1]
            val = np.random.normal(0.5, 0.15)
            self.alpha = float(np.clip(val, 0.0, 1.0))

        # --- 数据准备 ---
        stock_idx = random.randint(0, self.stock_list_len - 1)
        self.current_df = self.df_list[stock_idx]
        
        self.prices_close = self.current_df['收盘'].values.astype(np.float32)
        if '开盘' in self.current_df.columns:
            self.prices_open = self.current_df['开盘'].values.astype(np.float32)
        else:
            self.prices_open = self.prices_close

        total_len = len(self.prices_close)
        
        # --- 时间窗口选择 ---
        valid_range_len = total_len - WINDOW_SIZE - TRAINING_DAYS
        if valid_range_len <= 0:
            start_index = 0
            self.last_day = total_len - 1
        else:
            start_index = random.randint(0, valid_range_len - 1)
            self.last_day = start_index + WINDOW_SIZE + TRAINING_DAYS
        
        self.today = start_index + WINDOW_SIZE

        # --- 账户与统计重置 ---
        self.my_cash = ORIGINAL_MONEY 
        self.number_of_shares = 0 
        
        self.highest_worth = ORIGINAL_MONEY
        self.max_drawdown_cur = 0 
        self.max_drawdown_global = 0
        self.last_action = 0.0

        self.episode_rewards = {
            "r_base": [], "r_base_pos": [], "r_base_neg": [], 
            "r_risk": [], "r_cash": [],
            "r_turnover": []
        }

        # --- 初始化模拟价格 ---
        self.time_remaining = random.random() 
        self.current_price = self._calculate_noisy_price(self.today, self.time_remaining)

        # --- 历史数据预热 ---
        self.ma5 = 0; self.ma20 = 0
        self.stock_history = []
        
        current_window_prices = self.prices_close[self.today - WINDOW_SIZE : self.today + 1]
        
        for i in range(WINDOW_SIZE):
            p_curr = current_window_prices[i]
            p_next = current_window_prices[i+1]
            if p_curr == 0: p_curr = 1e-5 
            delta_ratio = np.log(p_next / p_curr) * INCR_PARA
            self.stock_history.append(delta_ratio)

        return self._get_observation(), {}

    def _calculate_noisy_price(self, day_idx, time_rem):
        """ 计算带噪声的盘中价格 (Brownian Bridge) """
        p_open = self.prices_open[day_idx]
        p_close = self.prices_close[day_idx]
        
        linear_price = p_open + (p_close - p_open) * (1.0 - time_rem)
        daily_volatility = abs(p_close - p_open) + (p_open * 0.002) 
        
        # 桥接因子：两端为0，中间最大
        bridge_factor = time_rem * (1.0 - time_rem) * 4.0 
        
        noise = np.random.normal(0, 0.1) * daily_volatility * bridge_factor
        final_price = linear_price + noise
        
        return max(1e-5, final_price)

    def _get_observation(self):
        # --- A. 历史序列 ---
        history = np.array(self.stock_history.copy(), dtype=np.float32)
        
        # 观测噪声
        noise = np.random.normal(0, 0.005, size=history.shape) 
        history = history + noise
        
        # 随机 Dropout (增强鲁棒性)
        if np.random.rand() < 0.05:
            history = history * np.random.binomial(1, 0.9, size=history.shape)

        # --- B. 技术指标 ---
        current_idx = self.today
        start_idx = max(0, current_idx - 65)
        window_prices = self.prices_close[start_idx : current_idx + 1]
        
        if self.today > 0:
            prev_close = self.prices_close[self.today - 1]
        else:
            prev_close = self.prices_open[self.today]
            
        current_open = self.prices_open[self.today]
        if prev_close <= 0: prev_close = 1e-5
        
        # 开盘跳空
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

        # --- C. 仓位 ---
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
        # --- 1. 执行交易 ---
        # 记录上一步净值用于计算 Reward
        prev_net_worth = self.my_cash + self.number_of_shares * self.current_price
        
        act = np.clip(action[0], -1, 1)
        
        if act > 0: # 买入
            can_buy_cash = self.my_cash * act
            shares_to_buy = int(can_buy_cash // self.current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * self.current_price * 1.0005 
                self.my_cash -= cost
                self.number_of_shares += shares_to_buy
        elif act < 0: # 卖出
            shares_to_sell = int(self.number_of_shares * abs(act))
            if shares_to_sell > 0:
                gain = shares_to_sell * self.current_price * 0.9995 
                self.my_cash += gain
                self.number_of_shares -= shares_to_sell

        # --- 2. 状态推演 ---
        terminated = self.today >= self.last_day
        if not terminated:
            self.today += 1 
            
        # --- 3. 结算净值 ---
        current_net_worth = self.my_cash + self.number_of_shares * self.prices_close[self.today]

        # 创新高逻辑
        if current_net_worth > self.highest_worth:
            self.max_drawdown_cur = 0 
            self.highest_worth = current_net_worth

        # --- 4. Reward 计算 (核心修改部分) ---
        
        # A. 基础收益 (Base Return)
        # 包含股票涨跌 + 交易滑点成本
        r_base = np.log((current_net_worth + 1e-8) / (prev_net_worth + 1e-8)) * INCR_PARA 

        # B. 现金无风险收益 (Risk-Free Reward)
        # 目的：防止横盘时死拿股票。如果股票不涨，持有现金是有奖励的。
        if current_net_worth > 0:
            curr_cash_ratio = self.my_cash / current_net_worth
        else:
            curr_cash_ratio = 0.0
            
        # 计算逻辑：(现金比例 * 日化无风险收益) * 缩放系数
        # 这样 r_risk_free 和 r_base 就在同一个数量级上
        r_risk_free = curr_cash_ratio * self.rf_daily * INCR_PARA

        # C. 风险调整 (回撤惩罚)
        drawdown = (self.highest_worth - current_net_worth) / self.highest_worth * INCR_PARA
        
        r_risk_val = 0.0
        if drawdown > self.max_drawdown_cur:
            dd_diff = drawdown - self.max_drawdown_cur
            r_risk_val = - (dd_diff * self.alpha * 1.0)
            
        self.max_drawdown_cur = max(self.max_drawdown_cur, drawdown)
        self.max_drawdown_global = max(self.max_drawdown_global, drawdown)
        
        # D. 交易摩擦惩罚 (Turnover Penalty)
        # 计算动作变化幅度 (例如: 上次 0.5, 这次 -0.5, 变化量是 1.0)
        action_change = np.abs(act - self.last_action)
        
        # 惩罚项: 系数 * 变化幅度
        # 这就像是给频繁换手征收的 "智商税"，强迫它思考清楚再动
        r_turnover = - self.turnover_coef * action_change
        
        # 更新上一步动作，供下一步使用
        self.last_action = act
        
        # E. 总 Reward
        total_reward = r_base 
        # + r_risk_val + r_risk_free + r_turnover
        total_reward = np.clip(total_reward, -10.0, 10.0)

        # --- 6. 统计记录 ---
        self.episode_rewards["r_base"].append(r_base)
        self.episode_rewards["r_risk"].append(r_risk_val)
        self.episode_rewards["r_cash"].append(r_risk_free)
        self.episode_rewards["r_turnover"].append(r_turnover)
        
        if r_base > 0:
            self.episode_rewards["r_base_pos"].append(r_base)
        elif r_base < 0:
            self.episode_rewards["r_base_neg"].append(r_base)
        
        def safe_mean(lst): return np.mean(lst) if len(lst) > 0 else 0.0

        info = {
            "net_worth": float(current_net_worth),
            "price": float(self.current_price),
            "max_dd": float(self.max_drawdown_global),
            "alpha": float(self.alpha),
            
            "ave_r_base": safe_mean(self.episode_rewards["r_base"]),
            "ave_r_base_pos": safe_mean(self.episode_rewards["r_base_pos"]),
            "ave_r_base_neg": safe_mean(self.episode_rewards["r_base_neg"]),
            "ave_r_turnover": safe_mean(self.episode_rewards["r_turnover"]),
            "ave_r_risk": safe_mean(self.episode_rewards["r_risk"]),
            "ave_r_cash": safe_mean(self.episode_rewards["r_cash"]),
        }

        # --- 7. 更新历史序列 ---
        p_close_curr = self.prices_close[self.today-1] if self.today > 0 else self.prices_close[0]
        p_close_next = self.prices_close[self.today]
        delta_ratio = np.log(p_close_next / p_close_curr) * INCR_PARA
        self.time_remaining = random.random() # 下一步的随机时间点
        self.current_price = self._calculate_noisy_price(self.today, self.time_remaining)

        self.stock_history.pop(0)
        self.stock_history.append(delta_ratio)

        return self._get_observation(), float(total_reward), terminated, False, info