import random
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, SubprocVecEnv
import torch

# --- 自定义模块导入 ---
import rl.prehandle
from rl.environment import SimpleStockEnv
from database.influx_manager import InfluxDBManager, InfluxDBConfig, InfluxDBCallbacks
from config import * 

# ==========================================
# 1. 核心组件：详细日志回调
# ==========================================
class DetailedLogCallback(BaseCallback):
    """
    专门用于记录新环境特性的日志回调
    不再控制 Alpha，而是观察 Agent 在不同 Alpha 下的表现，以及现金奖励的获取情况
    """
    def __init__(self, verbose=0):
        super(DetailedLogCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # 记录关键性能指标 (从 Info 中提取)
        # SB3 的 VecEnv 会自动堆叠 Info，这里取第一个环境的 Info
        if len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]
            
            # --- 1. 记录 Reward 组成 (验证现金奖励机制) ---
            # 这里的 key 要对应环境 info 中的 key
            if "ave_r_base" in info:
                self.logger.record("rewards/1_base_return", info["ave_r_base"])
            if "ave_r_base_pos" in info:
                self.logger.record("rewards/2_base_pos_return", info["ave_r_base_pos"])
            if "ave_r_base_neg" in info:
                self.logger.record("rewards/3_base_neg_return", info["ave_r_base_neg"])
            if "ave_r_cash" in info:
                self.logger.record("rewards/4_cash_interest", info["ave_r_cash"])
            if "ave_r_risk" in info:
                self.logger.record("rewards/5_risk_penalty", info["ave_r_risk"])
            if "ave_r_turnover" in info:
                self.logger.record("rewards/6_turnover_penalty", info["ave_r_turnover"])

            # --- 2. 记录资产状态 ---
            if "net_worth" in info:
                self.logger.record("status/net_worth", info["net_worth"])
            if "max_dd" in info:
                self.logger.record("status/max_drawdown", info["max_dd"])
            if "alpha" in info:
                self.logger.record("status/current_alpha", info["alpha"])
            if "price" in info:
                self.logger.record("status/price", info["price"])

        return True

# ==========================================
# 2. 数据加载工具 (保持不变)
# ==========================================
def get_data_with_cache(manager: InfluxDBManager, codes, start_date, end_date, cache_name):
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
SEED = 541438
ADDITIONAL_STEPS = 2_000_000 # 训练步数

if __name__ == "__main__":
    set_random_seed(SEED)
    
    # --- A. 数据准备 ---
    # 确保 config.py 中有 train_range, val_range 等定义
    config = InfluxDBConfig(HOST, DATABASE, TOKEN)
    manager = InfluxDBManager(config, InfluxDBCallbacks())
    
    # 获取股票列表
    target_date = datetime(2025, 12, 12) # 注意：这个日期需要确保在数据库范围内
    # 如果是回测，通常用过去的时间；如果是模拟，确保能取到代码表
    try:
        all_codes = manager.get_stock_code_list_by_date(target_date)
    except:
        # 如果取不到，用一个兜底逻辑或者取最新
        print("⚠️ 无法获取指定日期代码，尝试获取所有...")
        # 这里需要你自己根据数据库接口调整，假设获取成功
        all_codes = [] 

    valid_prefixes = ('600', '601', '603', '000', '002')
    main_codes = [c for c in all_codes if c.startswith(valid_prefixes)]
    
    # 随机抽样
    selected_codes = np.random.choice(main_codes, size=min(1200, len(main_codes)), replace=False)
    print(f"📊 选中股票数量: {len(selected_codes)}")

    train_dfs = get_data_with_cache(manager, selected_codes, train_range[0], train_range[1], "train_data.pkl")
    val_dfs   = get_data_with_cache(manager, selected_codes, val_range[0], val_range[1], "val_data.pkl")
    test_dfs  = get_data_with_cache(manager, selected_codes, test_range[0], test_range[1], "test_data.pkl")
    manager.close()

    # --- B. 环境构建 ---
    train_env = DummyVecEnv([lambda: SimpleStockEnv(train_dfs)])
    train_env = VecMonitor(train_env, TRAIN_LOG_DIR)

    val_env = DummyVecEnv([lambda: SimpleStockEnv(val_dfs)])
    val_env = VecMonitor(val_env, VAL_LOG_DIR)

    # --- C. 回调函数组装 ---
    
    # 1. 验证回调
    eval_callback = EvalCallback(
        val_env,
        best_model_save_path='./best_model/',
        log_path=VAL_LOG_DIR,
        eval_freq=1_000,
        n_eval_episodes=50,     
        deterministic=True,
        render=False
    )
    
    # 2. 检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path='./checkpoints/', 
        name_prefix='sac_v2'
    )
    
    # 3. 详细日志回调
    log_callback = DetailedLogCallback()

    callback_list = CallbackList([eval_callback, checkpoint_callback, log_callback])

    # --- D. 模型加载与训练 ---
    best_model_path = "./best_model/best_model.zip"
    
    # 网络架构：可以适当加宽，以处理更复杂的状态（Alpha输入）
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], qf=[128, 128]),
        activation_fn=torch.nn.ReLU
    )
    
    if os.path.exists(best_model_path):
        print(f"🔄 发现现有模型 {best_model_path}，正在加载...")
        # custom_objects 用于处理版本兼容性或特定参数变化
        model = SAC.load(best_model_path, env=train_env, device="cuda")
        
        current_steps = model.num_timesteps
        target_steps = current_steps + ADDITIONAL_STEPS
        print(f"📈 历史步数: {current_steps}")
        print(f"🎯 目标步数: {target_steps} (+{ADDITIONAL_STEPS})")
        
        # 尝试加载 Buffer
        buffer_path = "./best_model/replay_buffer.pkl"
        if os.path.exists(buffer_path):
            try:
                print("💾 加载 Replay Buffer...")
                model.load_replay_buffer(buffer_path)
            except Exception as e:
                print(f"⚠️ Buffer 加载失败 (可能是环境Obs空间变了): {e}")
                print("⚠️ 将使用空 Buffer 继续训练")
    else:
        print("🆕 创建全新 SAC 模型 (V2 Environment)...")
        model = SAC(
            "MlpPolicy", 
            train_env, 
            verbose=1, 
            tensorboard_log=TRAIN_LOG_DIR,
            device="cuda",
            policy_kwargs=policy_kwargs,
            buffer_size=1_000_000,
            learning_starts=10_000, 
            batch_size=4096,        
            tau=0.005,
            gamma=0.99,
            learning_rate=3e-4, # 稍微调大一点初始 LR，因为有了 Cash Reward 容易陷入局部最优
            train_freq=20,       # 增加更新频率
            gradient_steps=20,
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
    print("\n🔍 开始回测可视化 (测试集)...")
    
    test_model = SAC.load("./best_model/best_model.zip", device="cuda")
    test_env = DummyVecEnv([lambda: SimpleStockEnv(test_dfs)]) 
    
    returns = []
    alphas = [] # 记录每一局的 Alpha
    
    obs = test_env.reset()
    
    # 测试 100 个Episode
    for i in range(100):
        done = False
        while not done:
            action, _ = test_model.predict(obs, deterministic=True)
            obs, reward, done, info_list = test_env.step(action)
            
            if done:
                info = info_list[0]
                net_worth = info["net_worth"]
                roi = (net_worth - ORIGINAL_MONEY) / ORIGINAL_MONEY
                returns.append(roi)
                alphas.append(info["alpha"])
                
                print(f"测试局 {i+1} | Alpha: {info['alpha']:.2f} | 收益率: {roi*100:.2f}% | 回撤: {info['max_dd']:.2f}")
    
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', label='盈亏平衡线')
    plt.title('模型测试集收益分布 (100局)')
    plt.xlabel('收益率 (ROI)')
    plt.ylabel('频次')
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    
    plot_path = os.path.join(TRAIN_LOG_DIR, "backtest_distribution.png")
    plt.savefig(plot_path)
    print(f"📊 收益分布图已保存至: {plot_path}")
    
    # 打印统计数据
    returns = np.array(returns)
    print(f"\n🏆 最终成绩单:")
    print(f"平均收益: {np.mean(returns)*100:.2f}%")
    print(f"正收益比例: {np.sum(returns > 0)} / {len(returns)} ({np.sum(returns > 0)/len(returns)*100:.0f}%)")
    # 简单的相关性分析：看看 Alpha 高的时候表现如何
    corr = np.corrcoef(alphas, returns)[0, 1]
    print(f"Alpha与收益的相关性: {corr:.2f} (正数表示越保守越赚钱，负数表示越激进越赚钱)")