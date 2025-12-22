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
        # 1. 数据准备：仅使用收盘价，并做简单的归一化处理
        self.df_list = df_list
        self.stock_list_len = len(self.df_list)
        # self.df = df.reset_index(drop=True)
        # self.prices = self.df['收盘'].values.astype(np.float32)
        # self.last_day = min(len(self.prices) - 1, TRAINING_DAYS + WINDOW_SIZE - 1)
        
        # 2. 动作空间 (Action Space): 
        # 范围 [-1, 1]。 1 表示全仓买入，-1 表示全仓卖出，0 表示观望
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # 3. 状态空间 (Observation Space): 
        # 我们给 AI 看 windowsize + 3 个维度：[[当前对比前一天的涨幅(十分比)], 我的总资产/原始资金, 我的现金占总资产的比例, 当前持仓状态(0到1, 1-现金占比)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(WINDOW_SIZE + 3,), dtype=np.float32)

        # 初始化内部状态
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
        # 我们需要确保：start_index + WINDOW_SIZE + TRAINING_DAYS <= total_len
        # max_start_index 是我们可以开始交易的最晚的一天的索引（相对于 prices 数组）
        # 注意：self.today 指向的是“当前看到的价格”的索引
        valid_range_len = total_len - WINDOW_SIZE - TRAINING_DAYS
        if valid_range_len <= 0:
            # 理论上预处理已经过滤了，防止万一，这里做个兜底
            # 如果数据不够长，就固定从头开始，或者报错
            start_index = 0
            self.last_day = total_len - 1
        else:
            # 随机选择一个起跑点
            start_index = random.randint(0, valid_range_len - 1)
            # 结束点是起跑点 + 训练时长
            self.last_day = start_index + WINDOW_SIZE + TRAINING_DAYS
            
        # 3. 初始化账户
        self.my_cash = ORIGINAL_MONEY  # 现金
        self.number_of_shares = 0 # 持股数
        # self.today 是当前指针，初始位置需要先预留 WINDOW_SIZE 的历史数据
        self.today = start_index + WINDOW_SIZE
        # parameters
        self.target_value = 1.1 # 目标：资产增值到 130%
        self.new_high_reward = NEW_HIGH_REWARD

        # 4. 初始化状态历史
        self.stock_history = []
        current_window_prices = self.prices[self.today - WINDOW_SIZE : self.today + 1]
        for i in range(WINDOW_SIZE):
            # 计算 i 和 i+1 之间的涨幅
            p_curr = current_window_prices[i]
            p_next = current_window_prices[i+1]
            # 防止除以0
            if p_curr == 0: p_curr = 1e-5 
            
            delta_ratio = np.tanh((p_next - p_curr) / p_curr * INCR_PARA)
            self.stock_history.append(delta_ratio)

        # 返回初始状态和空信息
        return self._get_observation(), {}

    def _get_observation(self):
        history = self.stock_history.copy()
        
        current_price = self.prices[self.today]
        # 此时此刻的总资产
        current_net_worth = self.my_cash + self.number_of_shares * current_price
        
        total_asset_ratio = np.tanh(np.log(current_net_worth / ORIGINAL_MONEY) * ASSET_PARA)
        cash_ratio = self.my_cash / current_net_worth if current_net_worth > 0 else 0.0
        position_ratio = 1.0 - cash_ratio
        
        # 拼接到列表末尾
        history.extend([total_asset_ratio, cash_ratio, position_ratio])
        return np.array(history, dtype=np.float32)

    def step(self, action):
        # --- 1. 此时此刻 (Time T) ---
        # 看到的是 self.prices[self.today] (当前的观察值)
        current_price = self.prices[self.today]
        
        # 记录交易前的资产总额 (T 时刻)
        prev_net_worth = self.my_cash + self.number_of_shares * current_price

        # 执行交易：按当前价格 current_price 成交
        # 这符合“看到价格立刻下单”的逻辑
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
        # 获取新时刻的价格
        next_price = self.prices[self.today]
        
        # 计算新的总资产
        current_net_worth = self.my_cash + self.number_of_shares * next_price
        info = {
            "net_worth": float(current_net_worth),
            "shares": self.number_of_shares,
            "today": self.today
        }
        # 奖励 = 这一段时间内的资产增值比率 + 达到新高奖励 + 大幅度下跌惩罚
        reward = np.log(current_net_worth / prev_net_worth + 1e-8) * 10
        return_rate = np.sinh((current_net_worth - prev_net_worth) / prev_net_worth * 10)
        # 大幅度下跌惩罚
        if return_rate < 0:
            k = 0.3
        else:
            k = 0.3
            if current_net_worth >= ORIGINAL_MONEY * self.target_value:
                reward += self.new_high_reward
                self.target_value = current_net_worth / ORIGINAL_MONEY * 1.1 # 提高目标
        reward += return_rate * k
        reward = np.clip(reward, -5.0, 5.0)

        # 更新历史状态
        delta_ratio = np.tanh((next_price - current_price) / current_price * INCR_PARA)
        self.stock_history.pop(0)
        self.stock_history.append(delta_ratio)
        # 返回的是 T+1 的观察值，AI 将根据这个新状态做下一次(T+1)决策
        return self._get_observation(), float(reward), terminated, False, info