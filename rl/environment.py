from datetime import datetime
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
from database.influx_manager import InfluxDBManager, InfluxDBConfig, InfluxDBCallbacks
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
        
        # 1. 动作空间 (Action Space): 
        # 范围 [-1, 1]。 1 表示全仓买入，-1 表示全仓卖出，0 表示观望
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # 2. 状态空间 (Observation Space): 
        # 我们给 AI 看 windowsize + 2 个维度：[[当前对比前一天的涨幅(对数)], 我的现金占总资产的比例, 当前持仓状态(0到1, 1-现金占比)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(WINDOW_SIZE + 2,), dtype=np.float32)

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
        self.target_value = NEW_HIGH_TARGET
        self.new_high_reward = NEW_HIGH_REWARD

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
        history = self.stock_history.copy()
        
        current_price = self.prices[self.today]
        current_net_worth = self.my_cash + self.number_of_shares * current_price
        cash_ratio = self.my_cash / current_net_worth if current_net_worth > 0 else 0.0
        position_ratio = 1.0 - cash_ratio
        
        history.extend([cash_ratio, position_ratio])
        return np.array(history, dtype=np.float32)

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
        # 计算新的仓位 (用于 Risk Reward)
        # 注意：这里用的是 T+1 时刻的仓位价值占比
        current_pos_value = self.number_of_shares * next_price
        current_pos_ratio = current_pos_value / current_net_worth if current_net_worth > 0 else 0

        # --- 4. Reward 计算核心 ---
        
        # A. 基础收益 (Base Reward)
        r_base = np.log((current_net_worth + 1e-8) / (prev_net_worth + 1e-8)) * 10 # 绝对值 1 左右

        # B. 风险调整 (Risk-Adjusted Reward)
        # 获取过去 WINDOW_SIZE 天的价格（如果没有 WINDOW_SIZE 天，就取尽量多的天数）
        lookback = WINDOW_SIZE
        start_idx = max(0, self.today - lookback)
        window_prices = self.prices[start_idx : self.today + 1]
        recent_prices_max = np.max(window_prices)
        max_idx_in_window = np.argmax(window_prices)
        days_since_high = (len(window_prices) - 1) - max_idx_in_window
        sigma = np.std(self.stock_history) # 最大 0.1 左右

        # B1. 持有风险惩罚 (Hold Risk)
        # 仓位越重，且市场波动越大，惩罚越重
        r_risk_hold = 10 * current_pos_ratio * sigma # 绝对值 1 左右

        # B2. 下行风险/回撤惩罚 (Drawdown Penalty)
        # 如果当前价格距离 90 天高点很远，且还持有重仓，Q: 90 天内价格走向突变，应该重仓但是会被惩罚，是否应该乘上离最高点的时间跨度，越远这个权重越低？
        drawdown = (recent_prices_max - next_price) / recent_prices_max
        decay_factor = 0.95 ** days_since_high
        r_risk_down = current_pos_ratio * drawdown * decay_factor # 绝对值 1 左右
        r_risk_down = np.clip(r_risk_down, 0.0, 1.0)

        # C. 死区/空仓奖励 (Empty/Deadzone Reward)
        # 惩罚微小调仓：如果动作绝对值在 0.05 到 0.15 之间，视为无效操作
        # 同时，如果在半仓 (0.5) 附近晃悠，也给予惩罚
        r_action_penalty = 1.0 if 0.05 <= abs(act) <= 0.15 else 0 # 绝对值 1 左右

        # 惩罚不坚定的仓位 (半仓惩罚) Q: 这个惩罚是否正确且有必要？
        # 当仓位在 0.5 时，Punish 最大，逼迫 AI 选 0 或 1
        r_position_uncertainty = 4 * (current_pos_ratio * (1 - current_pos_ratio)) # 绝对值 1 左右

        # D. 创新高奖励 (New High Reward)
        r_new_high = 0
        while current_net_worth >= ORIGINAL_MONEY * self.target_value:
            r_new_high += self.new_high_reward  # 给予一次性大奖 (例如 1.0)
            self.target_value *= 1.1 # 提高目标

        # --- 5. 总 Reward 汇总 ---
        # 权重分配：
        # Base: 1.0 (主导)
        # Risk: -0.1, -0.1
        # Empty: -0.01
        # Position Uncertainty: -0.01
        # NewHigh: 1.0

        total_reward = 2.0 * r_base + \
                      -0.1 *  r_risk_hold + -0.1 * r_risk_down + \
                     -0.01 * r_action_penalty + \
                     -0.01 * r_position_uncertainty + \
                       0.3 * r_new_high
        
        # 稍微给一点生存奖励，防止因为全是负分而自杀
        total_reward += 0.001 

        # 裁剪，防止梯度爆炸
        total_reward = np.clip(total_reward, -10.0, 10.0)


        info = {
            "net_worth": float(current_net_worth),
            "shares": self.number_of_shares,
            "today": self.today,
            "r_base": r_base,
            "r_risk_hold": r_risk_hold,
            "r_risk_down": r_risk_down,
            "r_act_pen": r_action_penalty,
            "r_pos_unc": r_position_uncertainty,
            "drawdown": drawdown * decay_factor # 观察一下衰减后的回撤是多少
        }

        # 更新历史状态
        delta_ratio = np.log(next_price / current_price) * INCR_PARA
        self.stock_history.pop(0)
        self.stock_history.append(delta_ratio)
        # 返回的是 T+1 的观察值，AI 将根据这个新状态做下一次(T+1)决策
        return self._get_observation(), float(total_reward), terminated, False, info