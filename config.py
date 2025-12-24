from datetime import datetime

# ------- databse 配置信息 -------
HOST = "http://localhost:8181"
DATABASE = "stock_history_db"
TOKEN = "apiv3_DfumAJrYFgvwzRLausV9rI4_74-JlbekNQRlqf5gFT1wMnE4nc_ObRCNNtqtlynztO_pokRMII08bIhAbGoEyw"
# --------------------------------

# ------- RL 训练参数配置 -------
WINDOW_SIZE = 90  # 用于状态表示的窗口大小
TRAINING_DAYS = 252  # 交易日数量
ORIGINAL_MONEY = 1000000.0  # 原始资金
NEW_HIGH_REWARD = 1  # 达到新高奖励
NEW_HIGH_TARGET = 1.1  # 新高目标
INCR_PARA  = 10  # 用于计算涨幅的参数
ASSET_PARA = 2.3 # 用于计算总资产比率的参数
ADDITIONAL_STEPS = 500_000
# --------------------------------

# ---------- 时间范围 ----------
train_range = (datetime(2010, 1, 1), datetime(2021, 12, 31))
val_range   = (datetime(2022, 1, 1), datetime(2023, 12, 31))
test_range  = (datetime(2024, 1, 1), datetime(2025, 12, 31))

# ---------- 路径配置 ----------
MODEL_PATH = "sac_random_stock_model_6.zip"
TRAIN_LOG_DIR = "./tensorboard_logs/"
VAL_LOG_DIR = "./tensorboard_logs/val/"
# --------------------------------