import numpy as np
import gymnasium as gym
from gymnasium import spaces

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
        self.tradable_list = []  # (N_stocks, T)
        
        # list[0] 是大盘指数
        if len(stock_df_list) < 2:
            raise ValueError("需要至少两个DataFrame: [0]为指数, [1:]为个股")
            
        index_df = stock_df_list[0]
        
        # 批量预处理
        print("开始特征工程 (Rolling Window)...")
        for i, df in enumerate(stock_df_list[1:]):
            feats, targs, tradables = self._preprocess_data(df, index_df)
            
            # 确保数据长度足够
            # 需要: Window + Training Steps + Buffer
            if len(feats) > window_size + training_days + 20:
                self.data_list.append(feats)
                self.target_list.append(targs)
                self.tradable_list.append(tradables)
            
            if i % 10 == 0 and i > 0:
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
        # --- A. 获取当前环境信息 ---
        # 是否可交易 (由预处理计算好的 T 日状态决定)
        is_tradable = self.tradable_list[self.current_stock_idx][self.day_idx]
        
        raw_signal = float(np.clip(action[0], -1, 1))
        
        # --- B. 动作过滤 (Deadzone & Force Hold) ---
        if not is_tradable:
            # 不可交易(涨跌停/停牌): 强制保持上一日信号
            signal = self.last_signal
        else:
            # 只有当信号变化超过死区阈值时，才执行改变
            if abs(raw_signal - self.last_signal) < self.deadzone_level:
                signal = self.last_signal
            else:
                signal = raw_signal
        
        # 计算实际发生的变化量
        effective_change = np.abs(signal - self.last_signal)

        # --- C. 计算回报 (Reward) ---
        # 获取 T+1 日的真实收益 (百分比, 如 1.5 代表 1.5%)
        current_targets = self.target_list[self.current_stock_idx]
        
        # 边界保护
        if self.day_idx >= len(current_targets) - 1:
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, False, {}

        actual_next_ret_pct = current_targets[self.day_idx]
        
        # 1. 投资毛收益 (Gross PnL)
        gross_ret_pct = signal * actual_next_ret_pct
        
        # 2. 交易成本 (Transaction Cost)
        cost_pct = effective_change * self.transaction_cost_pct * 100
        
        # 3. 净收益 (Net PnL)
        net_ret_pct = gross_ret_pct - cost_pct
        
        # 4. 缩放 Reward (便于神经网络训练)
        reward = net_ret_pct * self.reward_scale
        
        # --- D. 追踪真实净值 (Portfolio Value) ---
        # 复利计算
        self.portfolio_value *= (1 + net_ret_pct / 100.0)

        # --- E. 状态更新与输出 ---
        self.last_signal = signal
        self.day_idx += 1
        self.steps_taken += 1
        
        # 检查是否结束
        truncated = self.steps_taken >= self.training_days
        terminated = False
        
        next_obs = self._get_observation()
        
        info = {
            'signal': signal,
            'real_ret': actual_next_ret_pct,
            'cost': cost_pct,
            'portfolio_value': self.portfolio_value, # 关键监控指标
            'is_tradable': is_tradable
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
        1. Inner Join 对齐日期
        2. Rolling Z-Score 归一化 (防未来函数)
        3. 识别停牌与一字板
        """
        df = df.copy()
        
        # 1. 数据对齐
        df = df.join(index_df[['收盘']], rsuffix='_Idx', how='inner')
        
        # 2. 计算基础 Log Return
        # log(P_t / P_{t-1}) * 100
        df['Log_Ret'] = np.log(df['收盘'] / df['收盘'].shift(1)) * 100
        df['Index_Log_Ret'] = np.log(df['收盘_Idx'] / df['收盘_Idx'].shift(1)) * 100
        
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
            
        # 5. === 流动性状态判定 ===
        # 判定一字板: High=Low 且 绝对涨跌幅 > 9% (粗略判定)
        is_limit = (df['最高'] == df['最低']) & (df['Log_Ret'].abs() > 9.0)
        # 判定停牌: High=Low 且 波动极小
        is_suspend = (df['最高'] == df['最低']) & (df['Log_Ret'].abs() < 1e-6)
        
        # 可交易标记
        df['Is_Tradable'] = ~(is_limit | is_suspend)
        
        # 6. 准备 Target (T+1 收益)
        df['Next_Ret'] = df['Log_Ret'].shift(-1)
        
        # 7. 清洗 NaN (Rolling导致的头部缺失 + Shift导致的尾部缺失)
        df.dropna(inplace=True)
        
        return (
            df[feature_cols].values.astype(np.float32), 
            df['Next_Ret'].values.astype(np.float32), 
            df['Is_Tradable'].values.astype(bool)
        )