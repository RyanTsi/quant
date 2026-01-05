import random
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import torch
import torch.nn as nn

# --- 自定义模块导入 ---
import rl.prehandle
from rl.signalEnv import AStockSignalEnv
from database.influx_manager import InfluxDBManager, InfluxDBConfig, InfluxDBCallbacks
from config import * 

# ==========================================
# 1. 日志回调
# ==========================================
class DetailedLogCallback(BaseCallback):
    """
    从 Env 的 info 中提取自定义指标并记录到 TensorBoard
    """
    def _on_step(self) -> bool:
        # 1. 获取当前 Step 所有环境返回的 info (列表，长度为环境数量)
        infos = self.locals.get('infos', [])
        
        # 2. 遍历环境 (通常你用 DummyVecEnv 只有一个环境，但为了通用性这里用循环)
        for info in infos:
            
            # --- A. 账户状态 (最核心) ---
            if 'State/Portfolio_Value' in info:
                # 记录净值曲线
                self.logger.record("main/Portfolio_Value", info['State/Portfolio_Value'])

            # --- B. 训练监控 (Metrics) ---
            if 'Metrics/Raw_Alpha_Ret' in info:
                # 原始 Alpha 收益 (未扣费)
                self.logger.record("train/Raw_Alpha_Ret", info['Metrics/Raw_Alpha_Ret'])
            
            if 'Metrics/Cost' in info:
                # 交易成本损耗
                self.logger.record("train/Transaction_Cost", info['Metrics/Cost'])
                
            if 'Metrics/Win_Rate_Step' in info:
                # 胜率 (SB3 会自动计算 dump 间隔内的平均值)
                self.logger.record("train/Win_Rate", info['Metrics/Win_Rate_Step'])

            # --- C. 归因分析 (Attribution) ---
            # 这里的目的是看：你的收益到底来自于 Alpha 还是大盘 Beta
            if 'Attribution/Alpha_Ret_Day' in info:
                self.logger.record("attribution/Alpha_Ret", info['Attribution/Alpha_Ret_Day'])
            
            if 'Attribution/Index_Ret_Day' in info:
                self.logger.record("attribution/Index_Ret", info['Attribution/Index_Ret_Day'])
                
            if 'Attribution/Abs_Ret_Day' in info:
                self.logger.record("attribution/Abs_Ret", info['Attribution/Abs_Ret_Day'])

            # --- D. 行为诊断 (Behavior) ---
            # 观察模型是不是只会输出 0，或者疯狂输出 1/-1
            if 'Action/Signal' in info:
                self.logger.record("behavior/Signal_Mean", info['Action/Signal'])
                
            if 'Action/Confidence' in info:
                self.logger.record("behavior/Confidence", info['Action/Confidence'])

        return True

# ==========================================
# 2. 数据加载工具 (已修改：移除外部预处理)
# ==========================================
def get_data_with_cache(manager, codes, start_date, end_date, cache_name):
    """
    修改后的数据加载逻辑：
    1. 从数据库拉取原始数据
    2. 使用 rl.prehandle.preprocess_data 进行清洗 (剔除ST、死股)
    3. 只有清洗合格的数据才进入列表
    """
    if os.path.exists(cache_name):
        print(f"📦 发现缓存 {cache_name}，快速加载中...")
        with open(cache_name, "rb") as f:
            return pickle.load(f)
    
    print(f"🚀 本地无缓存，开始下载及清洗 {len(codes)} 只股票数据...")
    df_list = []
    
    # 必须保证 index=0 是大盘指数
    # 我们假设 codes[0] 是 sh000001

    # 处理个股、指数
    valid_count = 0
    skipped_count = 0
    
    for code in codes:
        try:
            df_temp = manager.get_stock_data_by_range(stock_code=code, start_time=start_date, end_time=end_date)
            
            # === 调用您的清洗逻辑 ===
            # 注意：这里我们传入了 code，用于前缀判断
            df_clean = rl.prehandle.preprocess_data(df_temp)
            
            if df_clean is not None:
                df_list.append(df_clean)
                valid_count += 1
            else:
                skipped_count += 1
                
        except Exception as e:
            print(f"❌ {code} 处理异常: {e}")
            skipped_count += 1
            
        # 进度打印
        if (valid_count + skipped_count) % 500 == 0:
            print(f"处理进度: 有效 {valid_count} / 跳过 {skipped_count} ...")
    
    print(f"📊 数据清洗完成: 输入 {len(codes)-1} -> 输出 {valid_count} (剔除率 {skipped_count/(len(codes)-1):.1%})")

    if len(df_list) > 1: # 至少要有 1个指数 + 1个股票
        print(f"💾 保存缓存至 {cache_name}...")
        with open(cache_name, "wb") as f:
            pickle.dump(df_list, f)
            
    return df_list

class LSTM_Attention_Extractor(BaseFeaturesExtractor):
    """
    工业级时序特征提取器
    结构: Input -> LSTM -> (Attention) -> Linear -> Output to Policy
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        # 初始化父类，features_dim 是输出给 SAC Actor/Critic 的向量长度
        super().__init__(observation_space, features_dim)
        
        # 1. 自动推断输入维度
        # observation_space.shape 通常是 (Window_Size, Feature_Num)
        # 例如 (60, 5)
        self.window_size = observation_space.shape[0]
        self.input_features = observation_space.shape[1]
        
        # 2. 定义 LSTM 层
        # hidden_size: 隐层维度，越大拟合能力越强，但越难训练
        hidden_size = 64
        self.lstm = nn.LSTM(
            input_size=self.input_features,
            hidden_size=hidden_size,
            num_layers=2,           # 堆叠两层 LSTM 提取深层特征
            batch_first=True,       # 输入格式 (Batch, Seq, Feature)
            dropout=0.2             # 防止过拟合
        )
        
        # 3. (可选) 简单的注意力机制层
        # 用于计算 LSTM 输出序列中每个时间步的权重
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Softmax(dim=1)
        )
        
        # 4. 最终映射层
        # 将 LSTM/Attention 的输出映射到 features_dim (256)
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, features_dim),
            nn.LayerNorm(features_dim), # LayerNorm 对金融时序非常重要，稳定梯度
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        前向传播逻辑
        observations shape: (Batch_Size, Window_Size, Features)
        """
        # 1. 确保输入是 Float 类型
        x = observations.float()
        
        # 2. LSTM 前向传播
        # out: (Batch, Window, Hidden)
        # (h_n, c_n): 最后时刻的隐状态
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # --- 策略 A: 仅使用最后一步 (经典做法) ---
        # feature_vector = lstm_out[:, -1, :] 
        
        # --- 策略 B: 使用注意力机制 (进阶做法 - 推荐) ---
        # 计算权重: (Batch, Window, 1)
        weights = self.attention(lstm_out) 
        # 加权求和: (Batch, Hidden)
        # 这里的含义是：模型自动学会这60天里，哪几天对预测T+1最重要
        context_vector = torch.sum(weights * lstm_out, dim=1)
        
        # 3. 最终映射
        return self.linear(context_vector)
    

# ==========================================
# 3. 主程序
# ==========================================
SEED = 541438
ADDITIONAL_STEPS = 2_000_000 

if __name__ == "__main__":
    set_random_seed(SEED)
    
    # --- A. 数据准备 ---
    # 确保 config.py 中定义了 HOST, DATABASE, TOKEN 等
    config_db = InfluxDBConfig(HOST, DATABASE, TOKEN)
    manager = InfluxDBManager(config_db, InfluxDBCallbacks())
    
    # 获取股票列表
    target_date = datetime(2023, 12, 12)
    all_codes = manager.get_stock_code_list_by_date(target_date)
    
    # 确保大盘指数在第一位
    index_code = "sh000001"
    if index_code in all_codes:
        all_codes.remove(index_code)
    all_codes.insert(0, index_code)

    # 加载数据 (Train/Val/Test)
    # 这里的 train_range 等变量需在 config.py 中定义
    print("正在加载训练集...")
    train_dfs = get_data_with_cache(manager, all_codes, train_range[0], train_range[1], "train_data_v4.pkl")
    print("正在加载验证集...")
    val_dfs   = get_data_with_cache(manager, all_codes, val_range[0], val_range[1], "val_data_v4.pkl")
    
    manager.close()

    # --- B. 环境构建 ---
    # 注意：使用 v4.0 的参数配置
    env_kwargs = {
        'window_size': 60,
        'training_days': 252,
        'transaction_cost_pct': 0.0000,
        'deadzone_level': 0.1,
        'reward_scale': 1
    }

    print("构建训练环境...")
    train_env = DummyVecEnv([lambda: AStockSignalEnv(train_dfs, **env_kwargs)])
    train_env = VecMonitor(train_env, TRAIN_LOG_DIR)

    print("构建验证环境...")
    val_env = DummyVecEnv([lambda: AStockSignalEnv(val_dfs, **env_kwargs)])
    val_env = VecMonitor(val_env, VAL_LOG_DIR)

    # --- C. 回调函数组装 ---
    
    # 1. 验证回调
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path='./best_modelV4/',
        log_path=VAL_LOG_DIR,
        eval_freq=5_000,        # 稍微降低频率，加快训练速度
        n_eval_episodes=20,     # 验证20个Episode (20只随机股票/时间段)
        deterministic=True,
        render=False
    )
    
    # 2. 检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=5_000, 
        save_path='./checkpoints_v4/', 
        name_prefix='sac_v4'
    )
    
    # 3. 详细日志回调
    log_callback = DetailedLogCallback()

    callback_list = CallbackList([eval_callback, checkpoint_callback, log_callback])

    # --- D. 模型加载与训练 ---
    best_model_path = "./best_modelV4/best_model.zip"
    
    if os.path.exists(best_model_path):
        print(f"🔄 发现现有模型 {best_model_path}，正在加载...")
        model = SAC.load(best_model_path, env=train_env, device="cuda")
        
        # 尝试加载 Replay Buffer
        buffer_path = "./best_modelV4/replay_buffer.pkl"
        if os.path.exists(buffer_path):
            try:
                print("💾 加载 Replay Buffer...")
                model.load_replay_buffer(buffer_path)
            except Exception as e:
                print(f"⚠️ Buffer 加载失败 (可能是Obs Shape变了): {e}")
                
        print(f"📈 继续训练，目标步数: {ADDITIONAL_STEPS}")
        model.learn(total_timesteps=ADDITIONAL_STEPS, callback=callback_list, reset_num_timesteps=False)
        
    else:
        print("🆕 创建全新 SAC 模型 (V4 Environment)...")
        policy_kwargs = dict(
            # 1. 指定自定义提取器
            features_extractor_class=LSTM_Attention_Extractor,
            
            # 2. 传递参数给提取器 (对应 __init__ 中的参数)
            features_extractor_kwargs=dict(features_dim=256),
            
            # 3. 定义提取器之后的网络结构 (Actor 和 Critic)
            # 因为 LSTM 已经提取了强力的特征，后面的网络可以稍微简单点
            net_arch=dict(pi=[128, 64], qf=[128, 64]),
            
            # 4. 优化器参数 (可选，微调)
            optimizer_kwargs=dict(weight_decay=1e-5) # L2 正则化，防止过拟合
        )
        # 针对金融时间序列调整的 SAC 参数
        model = SAC(
            "MlpPolicy", 
            train_env, 
            verbose=1, 
            tensorboard_log=TRAIN_LOG_DIR,
            device="cuda",
            buffer_size=500_000,
            learning_starts=20_000,
            batch_size=512,
            ent_coef='auto',
            # policy_kwargs=dict(net_arch=[256, 256])
            policy_kwargs=policy_kwargs
        )
        
        print("🚀 开始训练...")
        model.learn(total_timesteps=ADDITIONAL_STEPS, callback=callback_list)

    # --- E. 保存最终结果 ---
    print("✅ 训练结束。保存最终模型...")
    model.save("./best_modelV4/final_model")
    try:
        model.save_replay_buffer("./best_modelV4/replay_buffer.pkl")
    except Exception as e:
        print(f"Buffer保存失败: {e}")