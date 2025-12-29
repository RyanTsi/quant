from datetime import datetime

# ------- databse 配置信息 -------
HOST = "http://localhost:8181"
DATABASE = "stock_history_db"
TOKEN = "apiv3_DfumAJrYFgvwzRLausV9rI4_74-JlbekNQRlqf5gFT1wMnE4nc_ObRCNNtqtlynztO_pokRMII08bIhAbGoEyw"
# --------------------------------

# ------- RL 训练参数配置 -------

# --------------------------------

# ---------- 时间范围 ----------
train_range = (datetime(2010, 1, 1), datetime(2021, 12, 31))
val_range   = (datetime(2022, 1, 1), datetime(2023, 12, 31))
test_range  = (datetime(2024, 1, 1), datetime(2025, 12, 31))

# ---------- 路径配置 ----------
MODEL_PATH = "sac_random_stock_model_6.zip"
TRAIN_LOG_DIR = "./tensorboard_logs/"
VAL_LOG_DIR = "./tensorboard_logs/val/"
TOTAL_TIMESTEPS = 3_000_000
# --------------------------------