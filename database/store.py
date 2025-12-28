from datetime import datetime
from influx_manager import StockData, InfluxDBConfig, InfluxDBManager, InfluxDBCallbacks
import pandas as pd
from typing import Dict, List, Optional, Union
import glob
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# ------- 配置信息 -------
HOST = "http://localhost:8181"
DATABASE = "stock_history_db"
TOKEN = "apiv3_DfumAJrYFgvwzRLausV9rI4_74-JlbekNQRlqf5gFT1wMnE4nc_ObRCNNtqtlynztO_pokRMII08bIhAbGoEyw"
# ------------------------



# 建议将解析函数放在全局作用域，以便多进程序列化
def parse_single_csv(file_path: str):
    """
    负责最耗时的解析工作，运行在独立的进程中
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        if df.empty: return []
        
        df['日期'] = pd.to_datetime(df['日期'])
        # 向量化转换比 iterrows 快得多
        stock_code = str(df['股票代码'].iloc[0]).zfill(6)
        field_cols = ['开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅', '涨跌幅', '涨跌额', '换手率']
        
        # 预先过滤掉含空值的列，转为字典列表，减少跨进程传输开销
        records = df.to_dict('records')
        batch_results = []
        for row in records:
            fields = {col: float(row[col]) for col in field_cols if pd.notna(row[col])}
            # 传输基础类型字典，比传输 StockData 对象更轻量
            batch_results.append({
                'time': row['日期'],
                'code': stock_code,
                'fields': fields
            })
        return batch_results
    except Exception as e:
        return []

if __name__ == '__main__':
    # --- 参数配置 ---
    DIRECTORY = "C:/Users/sola/Documents/quant/history/past15year_stock_data_daily"
    BATCH_WRITE_SIZE = 500000  # 凑够 50 万行写一次
    MAX_WORKERS = os.cpu_count()  # 充分利用所有核心
    
    # 1. 初始化 InfluxDB
    config = InfluxDBConfig(HOST, DATABASE, TOKEN)
    manager = InfluxDBManager(config, InfluxDBCallbacks())
    
    csv_files = glob.glob(os.path.join(DIRECTORY, "*.csv"))
    # csv_files = [ DIRECTORY + "/index_sh000001.csv"]
    print(f"🔥 启动多进程解析引擎 (Workers: {MAX_WORKERS})...")
    
    pending_buffer = []
    total_count = 0
    start_time = time.time()

    # 2. 使用进程池并行解析
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # as_completed 保证哪个文件先解析完就先处理哪个，不再按顺序死等
        future_to_file = {executor.submit(parse_single_csv, f): f for f in csv_files}
        
        for future in as_completed(future_to_file):
            data = future.result()
            if data:
                pending_buffer.extend(data)
                
                # 3. 达到批次大小，异步写入 IO
                if len(pending_buffer) >= BATCH_WRITE_SIZE:
                    # 转换为 StockData 对象
                    write_list = [StockData(timestamp=d['time'], stock_code=d['code'], fields=d['fields']) 
                                 for d in pending_buffer]
                    print(f"📦 缓冲区已满 ({len(write_list)}行)，正在提交 InfluxDB...")
                    manager.write_stock_batch(write_list)
                    total_count += len(write_list)
                    pending_buffer = []

    # 4. 清空剩余数据
    if pending_buffer:
        write_list = [StockData(timestamp=d['time'], stock_code=d['code'], fields=d['fields']) 
                     for d in pending_buffer]
        manager.write_stock_batch(write_list)
        total_count += len(write_list)

    end_time = time.time()
    manager.close()
    print(f"\n✨ 重构完成！")
    print(f"⏱️ 总耗时: {end_time - start_time:.2f} 秒")
    print(f"📈 总记录数: {total_count}")