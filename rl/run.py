import rl.prehandle
import numpy as np
from datetime import datetime
from rl.environment import SimpleStockEnv
from database.influx_manager import InfluxDBManager, InfluxDBConfig, InfluxDBCallbacks
from config import *
from stable_baselines3 import SAC
import os
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
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

# 定义时间范围
train_range = (datetime(2010, 1, 1), datetime(2021, 12, 31))
val_range   = (datetime(2022, 1, 1), datetime(2023, 12, 31))
test_range  = (datetime(2024, 1, 1), datetime(2025, 12, 31))

SEED = 215450649
np.random.seed(SEED)

MODEL_PATH = "sac_random_stock_model_1000.zip"
LOG_DIR = "./tensorboard_logs/"

def make_env(df_list, rank, seed=0):
    def _init():
        # 这里确保引用你定义的 SimpleStockEnv
        env = SimpleStockEnv(df_list)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    np.random.seed(SEED) 
    
    # 1. 初始化 InfluxDB
    config = InfluxDBConfig(HOST, DATABASE, TOKEN)
    manager = InfluxDBManager(config, InfluxDBCallbacks())
    
    # 2. 获取股票列表并随机筛选
    all_stock_codes = manager.get_stock_code_list_by_date(target_date=datetime(2025, 12, 12))
    random_selected_codes = np.random.choice(all_stock_codes, size=1200, replace=False)

    # 3. 通过缓存获取数据
    # 注意：如果你更改了 train_range，记得手动删除旧的 .pkl 文件
    df_list = get_data_with_cache(
        manager, 
        random_selected_codes, 
        train_range[0], 
        train_range[1],
        cache_name="train_1000_stocks.pkl"
    )

    manager.close()
    
    # 3. 创建环境
    num_cpu = 20
    env = SubprocVecEnv([make_env(df_list, i, SEED) for i in range(num_cpu)])
    env = VecMonitor(env, LOG_DIR) # 并行版日志监控
    # env = SimpleStockEnv(df_list)


    checkpoint_list = glob.glob("./checkpoints/sac_stock_auto_*.zip") # 获取所有自动保存的模型
    os.makedirs(LOG_DIR, exist_ok=True)

    # 2. 检查模型是否存在
    if os.path.exists(MODEL_PATH):
        print(f"📦 加载主模型: {MODEL_PATH}")
        load_path = MODEL_PATH
    elif checkpoint_list:
        # 找到修改时间最晚（最新）的一个备份文件
        latest_checkpoint = max(checkpoint_list, key=os.path.getmtime)
        print(f"🔄 未发现主模型，正在加载最新备份: {latest_checkpoint}")
        load_path = latest_checkpoint
    else:
        load_path = None

    if load_path:
        model = SAC.load(load_path, env=env, device="cuda")
    else:
        print("未发现历史模型，正在初始化新模型...")
        model = SAC(
            "MlpPolicy", 
            env, 
            tensorboard_log=LOG_DIR,
            learning_rate=3e-4, 
            buffer_size=1000000, 
            learning_starts=1000,
            batch_size=256,
            ent_coef='auto',
            target_entropy='auto',
            verbose=1,
            device="cuda"
        )

    # 3. 设置自动保存回调（防止训练中途断电）
    # 每 10,000 步保存一次，存放在 ./checkpoints/ 文件夹下
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path='./checkpoints/',
        name_prefix='sac_stock_auto'
    )
    tb_callback = TensorboardCallback()
    callback_list = CallbackList([tb_callback, checkpoint_callback])
    # 4. 开始/继续训练
    # reset_num_timesteps=False 是关键：它保证了 Tensorboard 曲线和学习率调度器的连续性
    model.learn(
        total_timesteps=1000000, 
        callback=callback_list,
        reset_num_timesteps=False 
    )

    # 5. 手动保存最终模型
    model.save(MODEL_PATH)
    # 6. 关闭环境
    env.close()