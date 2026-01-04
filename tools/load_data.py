import os
import pickle
from database.influx_manager import InfluxDBManager
import tools.preprocess_data as predata

def get_data_with_cache(manager: InfluxDBManager, codes, start_date, end_date, cache_name):
    """优先从本地 pickle 读取，否则从 InfluxDB 下载并缓存"""
    if os.path.exists(cache_name):
        print(f"📦 发现缓存 {cache_name}，快速加载中...")
        with open(cache_name, "rb") as f:
            return pickle.load(f)
    
    print(f"🚀 本地无缓存，开始下载 {len(codes)} 只股票数据...")
    df_list = []
    df_index = manager.get_stock_data_by_range(stock_code="sh000001", start_time=start_date, end_time=end_date)
    for code in codes:
        try:
            df_temp = manager.get_stock_data_by_range(stock_code=code, start_time=start_date, end_time=end_date)
            df_clean = predata.preprocess_data(df_temp, df_index)
            if df_clean is not None and len(df_clean) > 300:
                df_list.append(df_clean)
        except Exception as e:
            print(f"❌ {code} 失败: {e}")
    
    if df_list:
        print(f"💾 保存缓存至 {cache_name}...")
        with open(cache_name, "wb") as f:
            pickle.dump(df_list, f)
            
    return df_list