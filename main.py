# from datetime import datetime
# import pandas as pd
import numpy as np
# import rl.prehandle
# from database.influx_manager import InfluxDBManager, InfluxDBConfig, InfluxDBCallbacks
# from config import *

# # 1. 初始化 InfluxDB
# config = InfluxDBConfig(HOST, DATABASE, TOKEN)
# manager = InfluxDBManager(config, InfluxDBCallbacks())
# # 定义时间范围
# train_range = (datetime(2010, 1, 1), datetime(2021, 12, 31))
# val_range   = (datetime(2022, 1, 1), datetime(2023, 12, 31))
# test_range  = (datetime(2024, 1, 1), datetime(2025, 12, 31))

# # 2. 获取数据
# df_list = []
# all_stock_codes = ['000001']

# print("正在加载并清洗数据...")
# for code in all_stock_codes:
#     try:
#         df_temp = manager.get_stock_data_by_range(
#             stock_code=code,
#             start_time=train_range[0], 
#             end_time=train_range[1]
#         )
#         # 清洗数据
#         df_clean = rl.prehandle.preprocess_data(df_temp)
#         if df_clean is not None:
#             df_list.append(df_clean)
#             print(f"股票 {code} 加载成功，长度: {len(df_clean)}")
#         else:
#             print(f"股票 {code} 数据无效或过短，已跳过")
#     except Exception as e:
#         print(f"加载 {code} 失败: {e}")
# if df is not None and not df.empty:
#     print("📊 查询结果预览:")
#     df.reset_index(drop=True)
#     print(df)
#     # 接下来你可以直接用 df.plot() 或者进行量化分析
# else:
#     print("📭 未找到相关数据。")

# if __name__ == "__main__":
#     a = 2.718281828459045
#     # print(np.log(a) * 4)
#     print(np.tanh(1))

1.5 ** (1/252) - 1
a = np.log(1.5 ** (1/252)) * 100
print(a)