# ------- databse 配置信息 -------
HOST = "http://localhost:8181"
DATABASE = "stock_history_db"
TOKEN = "apiv3_DfumAJrYFgvwzRLausV9rI4_74-JlbekNQRlqf5gFT1wMnE4nc_ObRCNNtqtlynztO_pokRMII08bIhAbGoEyw"
# --------------------------------

# ------- RL 训练参数配置 -------
WINDOW_SIZE = 90  # 用于状态表示的窗口大小
TRAINING_DAYS = 252  # 交易日数量
ORIGINAL_MONEY = 100000.0  # 原始资金
NEW_HIGH_REWARD = 0.3  # 达到新高奖励
INCR_PARA  = 10  # 用于计算涨幅的参数
ASSET_PARA = 2.3 # 用于计算总资产比率的参数
# --------------------------------