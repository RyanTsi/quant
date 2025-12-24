import random
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt  # 引入绘图库
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import torch

# --- 自定义模块导入 ---
import rl.prehandle
from rl.environment import SimpleStockEnv
from database.influx_manager import InfluxDBManager, InfluxDBConfig, InfluxDBCallbacks
from config import * 

# ==========================================
# 1. 核心组件：自适应 Alpha 回调函数
# ==========================================
class AdaptiveAlphaCallback(BaseCallback):
    """
    根据模型表现自动调整 Alpha (风险惩罚权重)
    逻辑：当平均奖励超过阈值，且过了冷却期，就增加 Alpha
    """
    def __init__(self, verbose=0, start_alpha=0.1, max_alpha=1.8, cooldown_steps=30000, warmup_steps=61000):
        super(AdaptiveAlphaCallback, self).__init__(verbose)
        self.current_alpha = start_alpha
        self.max_alpha = max_alpha
        self.reward_threshold = 0.5  # 初始门槛：平均收益达到 0.5 才加压
        self.warmup_steps = warmup_steps
        self.last_update_step = 0
        self.cooldown = cooldown_steps
        self.step_size = 0.2         # 每次增加 0.2

    def _on_step(self) -> bool:
        global_step = self.num_timesteps
        
        # 获取最近 100 局的平均奖励 (SB3 自动维护该指标)
        # 这里的 key 必须是 SB3 原生记录的 rollout/ep_rew_mean
        ep_rew_mean = self.logger.name_to_value.get("rollout/ep_rew_mean", -np.inf)

        # --- 自适应判断逻辑 ---
        # 1. 超过预热
        # 2. 奖励超过当前门槛
        # 3. 距离上次调整已经过了冷却期 (防止频繁震荡)
        # 4. Alpha 还没到上限
        if (global_step > self.warmup_steps and
            ep_rew_mean > self.reward_threshold and 
            (global_step - self.last_update_step) > self.cooldown and 
            self.current_alpha < self.max_alpha):
            
            # 执行升级
            self.current_alpha += self.step_size
            self.reward_threshold += 0.5  # 提高下一次的门槛，逼迫模型进化
            self.last_update_step = global_step
            
            # 注入环境 (DummyVecEnv 下 set_attr 是即时生效的)
            self.training_env.set_attr("alpha", self.current_alpha)
            
            print(f"\n🔥 [进化时刻] Step {global_step}: Alpha 提升至 {self.current_alpha:.1f}, 下一目标 Reward > {self.reward_threshold:.1f}")

        # --- Tensorboard 记录 ---
        # 记录环境参数变化
        self.logger.record("env/adaptive_alpha", self.current_alpha)
        self.logger.record("env/target_threshold", self.reward_threshold)
        
        # 记录关键性能指标 (从 Info 中提取)
        if len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]
            if "net_worth" in info:
                self.logger.record("performance/net_worth", info["net_worth"])
            if "max_drawdown" in info:
                self.logger.record("performance/max_drawdown", info["max_drawdown"])
            if "pos_ratio" in info:
                self.logger.record("performance/position_ratio", info["pos_ratio"])
                self.logger.record("performance/alpha", info["alpha"])
            
            # 记录奖励细节
            reward_keys = ["ave_r_base", "ave_r_risk", "max_r_base", "max_r_risk"]
            for key in reward_keys:
                if key in info:
                    self.logger.record(f"rewards/{key}", info[key])

        return True

# ==========================================
# 2. 数据加载工具 (带缓存)
# ==========================================
def get_data_with_cache(manager, codes, start_date, end_date, cache_name):
    """优先从本地 pickle 读取，否则从 InfluxDB 下载并缓存"""
    if os.path.exists(cache_name):
        print(f"📦 发现缓存 {cache_name}，快速加载中...")
        with open(cache_name, "rb") as f:
            return pickle.load(f)
    
    print(f"🚀 本地无缓存，开始下载 {len(codes)} 只股票数据...")
    df_list = []
    for code in codes:
        try:
            df_temp = manager.get_stock_data_by_range(stock_code=code, start_time=start_date, end_time=end_date)
            df_clean = rl.prehandle.preprocess_data(df_temp)
            # 简单过滤：数据长度不够的不要
            if df_clean is not None and len(df_clean) > WINDOW_SIZE + 200:
                df_list.append(df_clean)
        except Exception as e:
            print(f"❌ {code} 失败: {e}")
    
    if df_list:
        print(f"💾 保存缓存至 {cache_name}...")
        with open(cache_name, "wb") as f:
            pickle.dump(df_list, f)
            
    return df_list

# ==========================================
# 3. 主程序
# ==========================================
SEED = 5418
ADDITIONAL_STEPS = 2_000_000

if __name__ == "__main__":
    set_random_seed(SEED)
    
    # --- A. 数据准备 ---
    config = InfluxDBConfig(HOST, DATABASE, TOKEN)
    manager = InfluxDBManager(config, InfluxDBCallbacks())
    
    # 获取股票列表
    target_date = datetime(2025, 12, 12)
    all_codes = manager.get_stock_code_list_by_date(target_date)
    # 过滤主板
    valid_prefixes = ('600', '601', '603', '000', '002')
    main_codes = [c for c in all_codes if c.startswith(valid_prefixes)]
    
    # 随机抽 1200 只
    selected_codes = np.random.choice(main_codes, size=min(1200, len(main_codes)), replace=False)
    print(f"📊 选中股票数量: {len(selected_codes)}")

    train_dfs = get_data_with_cache(manager, selected_codes, train_range[0], train_range[1], "train_data.pkl")
    val_dfs   = get_data_with_cache(manager, selected_codes, val_range[0], val_range[1], "val_data.pkl")
    test_dfs  = get_data_with_cache(manager, selected_codes, test_range[0], test_range[1], "test_data.pkl")
    manager.close()

    # --- B. 环境构建 (单进程 DummyVecEnv) ---
    train_env = DummyVecEnv([lambda: SimpleStockEnv(train_dfs)])
    train_env = VecMonitor(train_env, TRAIN_LOG_DIR)

    val_env = DummyVecEnv([lambda: SimpleStockEnv(val_dfs)])
    val_env = VecMonitor(val_env, VAL_LOG_DIR)

    # --- C. 回调函数组装 ---
    
    # 1. 验证回调：定期在验证集上测试
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path='./best_model/',
        log_path=VAL_LOG_DIR,
        eval_freq=10_000,
        n_eval_episodes=100,     # 验证 100 局
        deterministic=True,
        render=False
    )
    
    # 2. 检查点回调：定期保存模型文件
    checkpoint_callback = CheckpointCallback(
        save_freq=20000, 
        save_path='./checkpoints/', 
        name_prefix='sac_adaptive'
    )
    
    # 3. 自适应 Alpha 回调：核心逻辑
    # 假设我们从 0.0 开始，最高到 1.8
    adaptive_cb = AdaptiveAlphaCallback(start_alpha=0.1, max_alpha=1.8)

    callback_list = CallbackList([eval_callback, checkpoint_callback, adaptive_cb])

    # --- D. 模型加载与训练 (断点续传核心) ---
    best_model_path = "./best_model/best_model.zip"
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], qf=[256, 256]), # 网络大小适中
        activation_fn=torch.nn.ReLU
    )
    if os.path.exists(best_model_path):
        print(f"🔄 发现现有模型 {best_model_path}，正在加载...")
        model = SAC.load(best_model_path, env=train_env, device="cuda")
        # 计算新的目标步数
        current_steps = model.num_timesteps
        target_steps = current_steps + ADDITIONAL_STEPS
        print(f"📈 历史步数: {current_steps}")
        print(f"🎯 目标步数: {target_steps} (+{ADDITIONAL_STEPS})")
        
        # 尝试加载 Replay Buffer (如果存在)，这会让训练更平滑
        buffer_path = "./best_model/replay_buffer.pkl"
        if os.path.exists(buffer_path):
            print("💾 加载 Replay Buffer...")
            model.load_replay_buffer(buffer_path)
            
    else:
        print("🆕 创建全新 SAC 模型...")
        model = SAC(
            "MlpPolicy", 
            train_env, 
            verbose=1, 
            tensorboard_log=TRAIN_LOG_DIR,
            device="cuda",
            policy_kwargs=policy_kwargs,
            buffer_size=1_000_000,
            learning_starts=60_000, # 预收集：先跑 2万步 (约300局) 
            batch_size=4096,        # 大 Batch：一次看 4096 条数据
            tau=0.005,
            gamma=0.99,
            learning_rate=1e-4,
            train_freq=7,
            gradient_steps=1,
            ent_coef='auto',
        )
        target_steps = ADDITIONAL_STEPS

    print("🚀 开始训练...")
    try:
        model.learn(
            total_timesteps=target_steps, 
            callback=callback_list,
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        print("⚠️ 检测到中断，正在保存当前模型...")
        model.save("./best_model/interrupted_model")
        # 手动保存 buffer
        model.save_replay_buffer("./best_model/replay_buffer.pkl")

    print("✅ 训练结束。保存最终模型...")
    model.save("./best_model/final_model")
    model.save_replay_buffer("./best_model/replay_buffer.pkl")

    # --- E. 最终回测与可视化 ---
    print("\n🔍 开始回测可视化...")
    
    # 加载最佳模型进行测试
    test_model = SAC.load("./best_model/best_model.zip", device="cuda")
    test_env = DummyVecEnv([lambda: SimpleStockEnv(test_dfs)]) # 使用测试集
    
    returns = []
    obs = test_env.reset()
    
    # 测试 100 个Episode
    for i in range(100):
        done = False
        while not done:
            action, _ = test_model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            
            if done:
                # 提取收益率
                net_worth = info[0]["net_worth"]
                roi = (net_worth - ORIGINAL_MONEY) / ORIGINAL_MONEY
                returns.append(roi)
                print(f"测试局 {i+1}: 收益率 {roi*100:.2f}%")

    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', label='盈亏平衡线')
    plt.title('模型在测试集上的收益分布 (100局)')
    plt.xlabel('收益率 (ROI)')
    plt.ylabel('频次')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    
    # 保存图片
    plot_path = os.path.join(TRAIN_LOG_DIR, "backtest_distribution.png")
    plt.savefig(plot_path)
    print(f"📊 收益分布图已保存至: {plot_path}")
    
    # 打印统计数据
    returns = np.array(returns)
    print(f"\n🏆 最终成绩单:")
    print(f"平均收益: {np.mean(returns)*100:.2f}%")
    print(f"正收益比例: {np.sum(returns > 0)} / {len(returns)} ({np.sum(returns > 0)/len(returns)*100:.0f}%)")
    print(f"最大单局盈利: {np.max(returns)*100:.2f}%")
    print(f"最大单局亏损: {np.min(returns)*100:.2f}%")