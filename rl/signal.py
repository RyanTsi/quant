import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import pandas as pd
from config import *

class SignalStockEnv(gym.Env):
    """
    SignalStockEnv v4: 纯粹的信号指示器训练环境
    """
    def __init__(self, stock_df_list: list, index_df: pd.DataFrame):
        super(SignalStockEnv, self).__init__()

        self.df_list = stock_df_list
        self.index_df = index_df  # 必须传入大盘数据 (如沪深300)
        self.stock_list_len = len(self.df_list)
        
        # --- 1. 动作空间 ---
        # [-1, 1]: -1(确信跌/做空), 0(观望), 1(确信涨/做多)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # --- 2. 状态空间 (Feature Engineering) ---
        # 计算 Observation 的维度:
        # 1. History Sequence: WINDOW_SIZE
        # 2. Trend (MA dist): 3 (5, 20, 60)
        # 3. Bias (乖离率): 3 (6, 12, 24)
        # 4. Indicators: 2 (RSI, Boll_Pos)
        # 5. Volume: 2 (Log_Turnover, Log_Vol_Ratio)
        # 6. Context (大盘): 2 (Market_Ret, RS_Daily)
        # 7. Intraday (日内): 5 (Stock_Ret, Index_Ret, RS_Intra, Stock_Gap, Time)
        # Total = 60 + 3 + 3 + 2 + 2 + 2 + 5 = 77
        self.obs_dim = WINDOW_SIZE + 17
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # 运行时容器
        self.current_stock_df = None
        self.day_idx = 0      # 当前在数据中的绝对索引
        self.stop_idx = 0     # 结束索引
        
        # 模拟状态
        self.time_remaining = 0.0 
        self.current_price = 0.0       # 个股当前模拟价
        self.current_index_price = 0.0 # 大盘当前模拟价

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. 随机选择一只股票
        idx = random.randint(0, self.stock_list_len - 1)
        self.current_stock_df = self.df_list[idx]
        
        # 2. 确保日期对齐 (这里简化处理，假设 index_df 和 stock_df 长度一致且日期对应)
        # 实际工程中需要用 pd.merge 或 date索引对齐
        max_len = len(self.current_stock_df)
        valid_start = WINDOW_SIZE + 30 # 留足计算 MA 的 buffer
        valid_end = max_len - TRAINING_DAYS - 2
        
        if valid_end <= valid_start:
            return self.reset() # 数据太短，重开

        # 3. 随机选择开始日期
        self.day_idx = random.randint(valid_start, valid_end)
        self.stop_idx = self.day_idx + TRAINING_DAYS
        
        # 4. 初始化模拟状态
        # 随机生成这一局的"看盘时间" (或者在 step 中每一步都变，这里假设每一步随机)
        self.time_remaining = random.random() 
        self._update_simulated_prices()

        return self._get_observation(), {}

    def _update_simulated_prices(self):
        """同时模拟个股和大盘的当前时刻价格"""
        # 1. 模拟 个股
        s_open = self.current_stock_df['开盘'].iloc[self.day_idx]
        s_close = self.current_stock_df['收盘'].iloc[self.day_idx]
        self.current_price = self._calculate_noisy_price(
            s_open, s_close, self.time_remaining, scale=VOL_SCALE
        )
        
        # 2. 模拟 大盘 (使用相同的时间进度，但噪声更小)
        i_open = self.index_df['开盘'].iloc[self.day_idx]
        i_close = self.index_df['收盘'].iloc[self.day_idx]
        self.current_index_price = self._calculate_noisy_price(
            i_open, i_close, self.time_remaining, scale=INDEX_VOL_SCALE
        )

    def _calculate_noisy_price(self, p_open, p_close, t_rem, scale=1.0):
        """Brownian Bridge 价格模拟"""
        progress = 1.0 - t_rem
        
        # 线性基准
        base_price = p_open + (p_close - p_open) * progress
        
        # 波动项 (两头小，中间大)
        volatility = p_open * 0.015 * scale # 基础波动率 1.5%
        bridge_factor = np.sqrt(progress * t_rem) # 简单的桥接因子
        noise = np.random.normal(0, 1) * volatility * bridge_factor
        
        return max(0.01, base_price + noise)

    def _get_observation(self):
        # ==========================================
        # 1. 准备基础数据
        # ==========================================
        # 历史切片 (不包含今天)
        history_end = self.day_idx
        history_start = history_end - WINDOW_SIZE
        
        # 获取价格序列 (用于计算 technical indicators)
        closes = self.current_stock_df['收盘'].iloc[history_start : history_end].values
        opens = self.current_stock_df['开盘'].iloc[history_start : history_end].values
        # 假设有 '换手率' 列，如果没有请用 Volume / Capital 计算
        turnovers = self.current_stock_df['换手率'].iloc[history_start : history_end].values 
        
        # 当前模拟价格
        curr_price = self.current_price
        curr_idx_price = self.current_index_price

        # ==========================================
        # 2. 特征工程 (Feature Engineering)
        # ==========================================
        
        # --- A. 记忆层 (Memory) ---
        # Log Return Sequence
        # 为了避免除0，加一个小常数
        prices_seq = closes
        if len(prices_seq) > 0:
            log_returns = np.diff(np.log(prices_seq + 1e-8)) * INCR_PARA
            # 补齐长度 (diff 会少一个)
            log_returns = np.insert(log_returns, 0, 0)
        else:
            log_returns = np.zeros(WINDOW_SIZE)

        # --- B. 宏观趋势层 (Trend & Position) ---
        # 计算简单的 MA (利用 numpy)
        def calc_ma(period):
            if len(closes) < period: return curr_price
            return np.mean(closes[-period:])
        
        ma5 = calc_ma(5)
        ma20 = calc_ma(20)
        ma60 = calc_ma(60)
        
        # 价格相对于均线的位置 (Log Distance)
        dist_ma5 = np.log(curr_price / (ma5 + 1e-8)) * INCR_PARA
        dist_ma20 = np.log(curr_price / (ma20 + 1e-8)) * INCR_PARA
        dist_ma60 = np.log(curr_price / (ma60 + 1e-8)) * INCR_PARA
        
        # 乖离率 (Bias)
        bias_6 = (curr_price - calc_ma(6)) / (calc_ma(6) + 1e-8) * INCR_PARA
        bias_12 = (curr_price - calc_ma(12)) / (calc_ma(12) + 1e-8) * INCR_PARA
        bias_24 = (curr_price - calc_ma(24)) / (calc_ma(24) + 1e-8) * INCR_PARA
        
        # 布林带位置 (Boll Position) - 使用过去 20 天
        std20 = np.std(closes[-20:]) if len(closes) >= 20 else 1.0
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        # 归一化: 0=中轨, 1=上轨, -1=下轨 (大致范围)
        bb_pos = (curr_price - ma20) / (2 * std20 + 1e-8)

        # RSI (简化版，仅用最后14天计算一个点)
        # 实际中最好维护一个 rolling rsi，这里简化计算
        deltas = np.diff(closes[-15:])
        gain = deltas[deltas > 0].sum()
        loss = -deltas[deltas < 0].sum()
        rs = gain / (loss + 1e-8)
        rsi_val = 1.0 - (1.0 / (1.0 + rs)) # 0~1
        rsi_val = (rsi_val - 0.5) * 2.0    # 映射到 -1 ~ 1

        # --- C. 量能层 (Volume) ---
        # 1. 绝对热度 (Log Turnover)
        curr_turnover = turnovers[-1] if len(turnovers) > 0 else 1.0
        log_turnover = np.log(curr_turnover + 1.0)
        
        # 2. 相对突变 (Volume Ratio)
        ma_turnover_5 = np.mean(turnovers[-5:]) if len(turnovers) >= 5 else curr_turnover
        log_vol_ratio = np.log((curr_turnover + 1e-8) / (ma_turnover_5 + 1e-8))

        # --- D. 市场环境层 (Context) ---
        # 获取大盘前一天收盘
        idx_prev_close = self.index_df['收盘'].iloc[self.day_idx - 1]
        
        # 大盘当前涨跌 (相对于昨收)
        market_ret = np.log(curr_idx_price / idx_prev_close) * INCR_PARA
        
        # 个股当前涨跌 (相对于昨收)
        stock_prev_close = self.current_stock_df['收盘'].iloc[self.day_idx - 1]
        stock_ret_daily = np.log(curr_price / stock_prev_close) * INCR_PARA
        
        # 相对强弱 (RS Daily)
        rs_daily = stock_ret_daily - market_ret

        # --- E. 微观博弈层 (Intraday) ---
        # 今开
        stock_open = self.current_stock_df['开盘'].iloc[self.day_idx]
        index_open = self.index_df['开盘'].iloc[self.day_idx]
        
        # 缺口 (Gap)
        stock_gap = np.log(stock_open / stock_prev_close) * INCR_PARA
        index_gap = np.log(index_open / idx_prev_close) * INCR_PARA
        
        # 日内涨幅 (Intraday Return)
        intra_ret = np.log(curr_price / stock_open) * INCR_PARA
        index_intra_ret = np.log(curr_idx_price / index_open) * INCR_PARA
        
        # 日内相对强弱 (Intraday RS)
        intra_rs = intra_ret - index_intra_ret
        
        # 时间语义 (两头大，中间小)
        time_feature = abs(self.time_remaining - 0.5) * 2.0

        # ==========================================
        # 3. 拼接 Observation
        # ==========================================
        obs = np.concatenate([
            log_returns,                       # [0:60] 历史走势
            [dist_ma5, dist_ma20, dist_ma60],  # [60:63] 趋势位置
            [bias_6, bias_12, bias_24],        # [63:66] 乖离率
            [rsi_val, bb_pos],                 # [66:68] 震荡指标
            [log_turnover, log_vol_ratio],     # [68:70] 量能
            [market_ret, rs_daily],            # [70:72] 大盘环境
            [intra_ret, intra_rs, stock_gap, index_gap, time_feature] # [72:77] 日内博弈
        ]).astype(np.float32)
        
        return np.nan_to_num(obs)

    def step(self, action):
        # Action 是 [-1, 1] 的置信度
        confidence = float(action[0])
        
        # ==========================================
        # 1. 计算真实的收益 (Reward Calculation)
        # ==========================================
        # 逻辑：我们在 self.current_price (基于 time_remaining) 时刻做出决策
        # 收益结算：假设持有到【明天收盘】或者【今天收盘】。
        # 对于波段信号指示器，通常计算到 "第二天收盘" 的收益更合理，代表趋势预测能力。
        
        # 获取 "未来" 价格 (明天收盘)
        # 注意边界检查
        if self.day_idx + 1 >= len(self.current_stock_df):
            next_close = self.current_stock_df['收盘'].iloc[self.day_idx] # 如果没明天了，就按今天收盘算
        else:
            next_close = self.current_stock_df['收盘'].iloc[self.day_idx + 1]
            
        # 计算实际收益率 (Log Return)
        actual_return = np.log(next_close / self.current_price) * 100.0 # 百分比
        
        # 奖励公式：Alpha * Confidence
        # 如果 actual_return 是 +5%, confidence 是 1.0 -> Reward = 5.0
        # 如果 actual_return 是 -5%, confidence 是 1.0 -> Reward = -5.0 (重罚)
        # 如果 actual_return 是 +5%, confidence 是 0.0 -> Reward = 0.0 (踏空无惩罚，或者给微小负值)
        
        # 改进版 Reward: 鼓励抓住大波动
        reward = actual_return * confidence
        
        # 可选：惩罚过度自信但方向错误的 (Risk Control Penalty)
        if (actual_return < -2.0) and (confidence > 0.5):
            reward *= 1.5 # 大跌还重仓，加倍惩罚
            
        # ==========================================
        # 2. 状态流转
        # ==========================================
        self.day_idx += 1
        terminated = (self.day_idx >= self.stop_idx) or (self.day_idx >= len(self.current_stock_df) - 1)
        
        # 更新时间步 (这里每一步随机一个新的 time_remaining，模拟每次都在不同时间看盘)
        # 如果你想模拟连续的一天，可以在这里逻辑处理
        self.time_remaining = random.random()
        
        # 更新价格状态
        if not terminated:
            self._update_simulated_prices()
            next_obs = self._get_observation()
        else:
            next_obs = np.zeros(self.obs_dim, dtype=np.float32)

        info = {
            "date": str(self.current_stock_df.index[self.day_idx]) if not terminated else "End",
            "return": actual_return,
            "confidence": confidence
        }

        return next_obs, float(reward), terminated, False, info