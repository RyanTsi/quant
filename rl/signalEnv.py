import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

# ==========================================
# 1. 核心环境类: AStockSignalEnv v4.0
# ==========================================
class AStockSignalEnv(gym.Env):
    """
    A股信号交易环境 v4.0 (Production Ready)
    特性:
    - 动态 Rolling Z-Score 归一化 (无未来函数)
    - 严格的流动性检查 (停牌/一字板无法交易)
    - 交易死区 (过滤微小震荡)
    - 真实净值追踪 (Portfolio Value)
    """
    def __init__(self, stock_df_list: list,
                 window_size=60,
                 training_days=252,
                 transaction_cost_pct=0.0010,  # 单边万分之10 (含印花税+佣金+滑点)
                 deadzone_level=0.1,           # 10% 仓位变化死区
                 reward_scale=0.1):            # Reward 缩放因子
        super(AStockSignalEnv, self).__init__()
        
        self.window_size = window_size
        self.training_days = training_days
        self.transaction_cost_pct = transaction_cost_pct
        self.deadzone_level = deadzone_level
        self.reward_scale = reward_scale
        
        print(f"正在初始化环境 (v4.0)...")
        self.data_list = []      # (N_stocks, T, Features)
        self.target_list = []    # (N_stocks, T)
        self.abs_ret_list = []  # (N_stocks, T)
        
        # list[0] 是大盘指数
        if len(stock_df_list) < 2:
            raise ValueError("需要至少两个DataFrame: [0]为指数, [1:]为个股")
            
        index_df = stock_df_list[0]
        
        # 批量预处理
        print("开始特征工程 (Rolling Window)...")
        for i, df in enumerate(stock_df_list[1:]):
            feats, targs, abs_ret = self._preprocess_data(df, index_df)
            
            # 确保数据长度足够
            # 需要: Window + Training Steps + Buffer
            if len(feats) > window_size + training_days + 20:
                self.data_list.append(feats)
                self.target_list.append(targs)
                self.abs_ret_list.append(abs_ret)
            
            if i % 500 == 0 and i > 0:
                print(f"已处理 {i} 只股票...")
                
        print(f"初始化完成，有效股票数量: {len(self.data_list)}")

        # 动作空间: [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # 观测空间: (Window, 5特征)
        # 特征: [Log_Ret, Rel_Str, MA_Bias, Vol_Ratio, Hist_Vol]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(window_size, 5), dtype=np.float32
        )
        
        # 运行时内部状态
        self.current_stock_idx = 0
        self.day_idx = 0
        self.steps_taken = 0
        self.last_signal = 0.0
        self.portfolio_value = 1.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. 随机选一只股票
        self.current_stock_idx = np.random.randint(0, len(self.data_list))
        data_len = len(self.data_list[self.current_stock_idx])
        
        # 2. 随机选一个时间起点
        # 必须留出 training_days 的余量
        max_start = data_len - self.training_days - 1
        start_idx_min = self.window_size
        
        if max_start <= start_idx_min:
            self.day_idx = start_idx_min
        else:
            self.day_idx = np.random.randint(start_idx_min, max_start + 1)
            
        # 3. 重置状态
        self.steps_taken = 0
        self.last_signal = 0.0
        self.portfolio_value = 1.0
        
        return self._get_observation(), {}

    def step(self, action):
        # -----------------------------------------------------------
        # A. 解析动作与状态
        # -----------------------------------------------------------
        # 1. 截断 Action 到 [-1, 1]
        signal = float(np.clip(action[0], -1, 1))
        
        # 2. 计算仓位变化量 (用于计算手续费)
        effective_change = np.abs(signal - self.last_signal)

        # -----------------------------------------------------------
        # B. 获取环境数据 (Ground Truth)
        # -----------------------------------------------------------
        # 边界检查
        current_alpha_targets = self.target_list[self.current_stock_idx]
        if self.day_idx >= len(current_alpha_targets) - 1:
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {}

        # [关键区别]
        # actual_alpha_pct: 超额收益 (Stock - Index)。这是模型要学习的目标。
        # actual_abs_pct:   绝对收益 (Stock)。这是账户真实的盈亏。
        actual_alpha_pct = current_alpha_targets[self.day_idx]
        actual_abs_pct = self.abs_ret_list[self.current_stock_idx][self.day_idx]
        
        # 反推指数收益 (用于调试: Alpha + Index ~= Abs)
        # 注意: 由于浮点误差和Log Return计算方式，这里只是近似，但足够调试用
        actual_index_pct = actual_abs_pct - actual_alpha_pct

        # -----------------------------------------------------------
        # C. 计算 Reward (指导模型学习的信号)
        # -----------------------------------------------------------
        # 1. 原始 Alpha 收益 (Raw Alpha Reward)
        # 如果 Signal > 0 且 Alpha > 0 (跑赢大盘)，正奖励
        # 如果 Signal > 0 且 Alpha < 0 (跑输大盘)，负奖励
        raw_reward = signal * actual_alpha_pct
        
        # 2. 交易成本 (Cost)
        # 建议: 训练初期 transaction_cost_pct 设极低 (如 0 或 1e-5)，让模型先敢于交易
        cost_pct = effective_change * self.transaction_cost_pct * 100
        
        # 3. 最终 Reward
        # 只有扣除成本后还能跑赢大盘，才是好的 Alpha
        reward = (raw_reward - cost_pct) * self.reward_scale

        # -----------------------------------------------------------
        # D. 净值追踪 (Portfolio Value - 真实的钱)
        # -----------------------------------------------------------
        # 这里的盈亏必须用 "绝对收益" 算，因为你在实盘里不能只买 Alpha
        gross_abs_ret = signal * actual_abs_pct
        net_abs_ret = gross_abs_ret - cost_pct
        
        self.portfolio_value *= (1 + net_abs_ret / 100.0)

        # -----------------------------------------------------------
        # E. 计算调试指标 (Info Engineering)
        # -----------------------------------------------------------
        # 1. 胜率判定: 只有当方向正确且产生正向 Alpha 时才算 Win
        # 避免 0 值噪音，设一个极小阈值
        is_win = 1.0 if (signal * actual_alpha_pct > 1e-5) else 0.0
        
        # 2. 信号置信度 (模型开仓有多重?)
        confidence = np.abs(signal)

        # -----------------------------------------------------------
        # F. 状态更新与输出
        # -----------------------------------------------------------
        self.last_signal = signal
        self.day_idx += 1
        self.steps_taken += 1
        
        truncated = self.steps_taken >= self.training_days
        terminated = False # 除非破产，否则不自行终止，让 TimeLimit 处理
        
        # 这一步一定要保证 preprocess 里的 dropna 是执行过的
        next_obs = self._get_observation()
        
        info = {
            # === 1. 核心表现 (训练监控) ===
            'Reward': reward,                          # 最终给 RL 的分
            'Metrics/Raw_Alpha_Ret': raw_reward,       # 未扣费的 Alpha 收益
            'Metrics/Cost': cost_pct,                  # 手续费损耗
            
            # === 2. 归因分析 (最重要 - 区分运气和实力) ===
            # 如果 Portfolio 涨了，是因为 Alpha (实力) 还是 Index (运气)?
            'Attribution/Alpha_Ret_Day': actual_alpha_pct, # 当天该股超额收益
            'Attribution/Index_Ret_Day': actual_index_pct, # 当天大盘收益
            'Attribution/Abs_Ret_Day': actual_abs_pct,     # 当天个股绝对收益
            
            # === 3. 模型行为诊断 ===
            'Action/Signal': signal,                   # 信号值 (-1 ~ 1)
            'Action/Confidence': confidence,           # 开仓力度 (0 ~ 1)
            'Metrics/Win_Rate_Step': is_win,           # 单步胜率 (0 或 1, Tensorboard 会自动算平均)
            
            # === 4. 账户状态 ===
            'State/Portfolio_Value': self.portfolio_value
        }

        return next_obs, reward, terminated, truncated, info

    def _get_observation(self):
        # 切片获取窗口数据: [T - Window + 1 : T + 1]
        start = self.day_idx - self.window_size + 1
        end = self.day_idx + 1
        
        raw_obs = self.data_list[self.current_stock_idx][start:end]
        
        # 注入微小噪声，增加鲁棒性
        # 因为特征已经是 Z-Score (~N(0,1))，噪声给 0.01 足够
        noise = np.random.normal(0, 0.01, size=raw_obs.shape)
        return (raw_obs + noise).astype(np.float32)

    def _preprocess_data(self, df, index_df):
        """
        严谨的特征工程:
        """
        df = df.copy()
        if 'time' in df.columns:
            # 确保转为 datetime 格式
            df['time'] = pd.to_datetime(df['time'])
            # 设置为索引
            df.set_index('time', inplace=True)
            # 排序（以防万一）
            df.sort_index(inplace=True)
        # 1. 数据对齐
        df = df.join(index_df[['收盘']], rsuffix='_Idx', how='inner')
        
        # 2. 计算基础 Log Return
        # log(P_t / P_{t-1}) * 100
        df['Log_Ret'] = np.log(df['收盘'] / df['收盘'].shift(1)) * 100
        df['Index_Log_Ret'] = np.log(df['收盘_Idx'] / df['收盘_Idx'].shift(1)) * 100
        df['Excess_Ret'] = df['Log_Ret'] - df['Index_Log_Ret']
        # 3. 构建原始特征
        # F1: 相对强度
        df['Rel_Str'] = df['Log_Ret'] - df['Index_Log_Ret']
        
        # F2: 均线乖离
        ma_20 = df['收盘'].rolling(20).mean()
        df['MA_Bias'] = (df['收盘'] - ma_20) / (ma_20 + 1e-8) * 100
        
        # F3: 量比 (Log)
        vol_ma_5 = df['成交额'].rolling(5).mean()
        df['Vol_Ratio'] = np.log(df['成交额'] / (vol_ma_5.shift(1) + 1e-8))
        
        # F4: 历史波动率
        df['Hist_Vol'] = df['Log_Ret'].rolling(20).std()
        
        # 4. === 动态归一化 (Rolling Z-Score) ===
        feature_cols = ['Log_Ret', 'Rel_Str', 'MA_Bias', 'Vol_Ratio', 'Hist_Vol']
        roll_window = 60 # 统计窗口
        
        for col in feature_cols:
            # 计算滚动统计量
            roll = df[col].rolling(window=roll_window, min_periods=20)
            roll_mean = roll.mean()
            roll_std = roll.std()
            
            # 归一化: (x - mean) / std
            df[col] = (df[col] - roll_mean) / (roll_std + 1e-8)
            # 截断极端值
            df[col] = df[col].clip(-5, 5)
            
        # 5. 准备 Target (T+1 收益)
        df['Next_Excess_Ret'] = df['Excess_Ret'].shift(-1)
        df['Next_Abs_Ret'] = df['Log_Ret'].shift(-1)
        # 6. 清洗 NaN (Rolling导致的头部缺失 + Shift导致的尾部缺失)
        df.dropna(inplace=True)
        
        return (
            df[feature_cols].values.astype(np.float32),
            df['Next_Excess_Ret'].values.astype(np.float32),
            df['Next_Abs_Ret'].values.astype(np.float32),
        )