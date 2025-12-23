import rl.prehandle
import numpy as np
from datetime import datetime
from rl.environment import SimpleStockEnv
from database.influx_manager import InfluxDBManager, InfluxDBConfig, InfluxDBCallbacks
from config import *
from stable_baselines3 import SAC
import os
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
import pickle
import glob

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # 获取当前 step 的 info 字典
        # self.locals['infos'] 是一个列表，因为可能有多个并行环境
        info = self.locals['infos'][0]
        if "net_worth" in info:
            # 将资产净值记录到 TensorBoard 的 "Custom/NetWorth" 路径下
            self.logger.record("custom/net_worth", info["net_worth"])
        if "shares" in info:
            self.logger.record("custom/shares_held", info["shares"])
        if "r_base" in info:
            self.logger.record("custom/reward_base", info["r_base"])
        if "r_risk_hold" in info:
            self.logger.record("custom/reward_risk_hold", info["r_risk_hold"])
        if "r_risk_down" in info:
            self.logger.record("custom/reward_risk_down", info["r_risk_down"])
        if "r_act_pen" in info:
            self.logger.record("custom/reward_action_penalty", info["r_act_pen"])
        if "r_pos_unc" in info:
            self.logger.record("custom/reward_position_uncertainty", info["r_pos_unc"])
        if "drawdown" in info:
            self.logger.record("custom/drawdown", info["drawdown"])
        return True
    
def get_data_with_cache(manager, codes, start_date, end_date, cache_name="stock_data_cache.pkl"):
    # 检查本地是否存在缓存文件
    if os.path.exists(cache_name):
        print(f"📦 发现本地缓存 {cache_name}，正在快速加载...")
        with open(cache_name, "rb") as f:
            return pickle.load(f)
    
    # 如果没有缓存，则执行原有的下载逻辑
    print("🚀 本地无缓存，开始从 InfluxDB 提取数据...")
    df_list = []
    for code in codes:
        try:
            df_temp = manager.get_stock_data_by_range(
                stock_code=code,
                start_time=start_date, 
                end_time=end_date
            )
            df_clean = rl.prehandle.preprocess_data(df_temp)
            if df_clean is not None and len(df_clean) > WINDOW_SIZE + TRAINING_DAYS:
                df_list.append(df_clean)
                print(f"✅ {code} 加载成功")
        except Exception as e:
            print(f"❌ {code} 加载失败: {e}")
    
    # 下载完成后，保存到本地
    if df_list:
        print(f"💾 正在将 {len(df_list)} 只股票保存至本地缓存...")
        with open(cache_name, "wb") as f:
            pickle.dump(df_list, f)
            
    return df_list

SEED = 5418

def make_env(df_list, rank, seed=0):
    def _init():
        env = SimpleStockEnv(df_list)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    # ------------ 数据准备 ------------
    # 1. 初始化 InfluxDB
    config = InfluxDBConfig(HOST, DATABASE, TOKEN)
    manager = InfluxDBManager(config, InfluxDBCallbacks())
    
    # 2. 获取股票列表并随机筛选
    all_stock_codes = manager.get_stock_code_list_by_date(target_date=datetime(2025, 12, 12))
    selected_codes = np.random.choice(all_stock_codes, size=1200, replace=False)

    # 3. 通过缓存获取数据
    print("正在加载训练集...")
    train_dfs = get_data_with_cache(manager, selected_codes, train_range[0], train_range[1], "train_data.pkl")
    
    print("正在加载验证集...")
    val_dfs = get_data_with_cache(manager, selected_codes, val_range[0], val_range[1], "val_data.pkl")
    
    print("正在加载测试集...")
    test_dfs = get_data_with_cache(manager, selected_codes, test_range[0], test_range[1], "test_data.pkl")

    manager.close()
    # ------------ 环境构建 ------------ 

    num_cpu = 20
    train_env = SubprocVecEnv([make_env(train_dfs, i, SEED) for i in range(num_cpu)])
    train_env = VecMonitor(train_env, TRAIN_LOG_DIR)

    val_env = SubprocVecEnv([make_env(val_dfs, i, SEED + 7324) for i in range(num_cpu // 2)])
    val_env = VecMonitor(val_env, VAL_LOG_DIR)
    # ------------ 回调函数 ------------ 

    # A. 验证回调 (EvalCallback) - 核心部分
    # 它的作用：每隔 eval_freq 步，暂停训练，用当前模型在 val_env 里跑 n_eval_episodes 局
    # 如果平均奖励创新高，就保存到 best_model_save_path
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path='./best_model/',
        log_path=VAL_LOG_DIR,
        eval_freq=10000,        # 每训练 1万步(env steps) 验证一次
        n_eval_episodes=50,     # 每次验证跑 50 局取平均，消除随机性
        deterministic=True,     # 验证时由确定性策略(去除随机探索)，看真实实力
        render=False
    )

    # B. 定期保存 (Checkpoint)
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path='./checkpoints/', name_prefix='sac_stock')
    
    # C. Tensorboard 记录细节
    tb_callback = TensorboardCallback()

    # 组合回调
    callback_list = CallbackList([eval_callback, checkpoint_callback, tb_callback])

    # ------------ 模型训练 ------------

    model = SAC(
        "MlpPolicy", 
        train_env, 
        tensorboard_log=TRAIN_LOG_DIR,
        learning_rate=3e-4, 
        buffer_size=1_000_000, 
        learning_starts=5000,
        batch_size=4096,
        train_freq=(100, "step"),
        gradient_steps=100,
        ent_coef='auto',
        target_entropy=-0.5,
        verbose=1,
        use_sde=True,
        device="cuda"
    )
    print("开始训练...")
    model.learn(
        total_timesteps=5_000_000, 
        callback=callback_list,
        reset_num_timesteps=False 
    )
    print("训练结束。")

    # --- 最终测试 (Backtest) ---
    print("开始在测试集上回测最佳模型...")
    
    # 加载验证集上表现最好的模型
    best_model_path = os.path.join('./best_model/', "best_model.zip")
    if os.path.exists(best_model_path):
        model = SAC.load(best_model_path, device="cuda")
        print("已加载最佳模型。")
    else:
        print("未找到最佳模型，使用当前最终模型。")

    # 构建测试环境 (这里可以用 DummyVecEnv 方便调试，或者 SubprocVecEnv 加速)
    # 测试集是 2024-2025 的数据
    test_env = SubprocVecEnv([make_env(test_dfs, i, SEED + 906) for i in range(num_cpu)])
    
    # 跑测试
    obs = test_env.reset()
    total_episodes = 100 # 测试 100 个不同的股票/时间段
    episode_counts = 0
    test_rewards = []
    
    # 用于记录资产曲线
    # 注意：并行环境很难画出单一的连续曲线，通常我们统计分布
    
    while episode_counts < total_episodes:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = test_env.step(action)
        
        for i, done in enumerate(dones):
            if done:
                episode_counts += 1
                # 获取该局结束时的信息
                if "net_worth" in infos[i]:
                     final_value = infos[i]["net_worth"]
                     roi = (final_value - ORIGINAL_MONEY) / ORIGINAL_MONEY
                     test_rewards.append(roi)
                     print(f"测试局 {episode_counts}: 收益率 {roi*100:.2f}%")

    print(f"平均测试收益率: {np.mean(test_rewards)*100:.2f}%")
    print(f"正收益比例: {np.sum(np.array(test_rewards) > 0) / len(test_rewards) * 100:.2f}%")
    
    test_env.close()
    train_env.close()
    val_env.close()