import pandas as pd
from config import *

def preprocess_data(df_raw):
    """
    清洗股票数据
    1. 去除空值
    2. 去除成交量为0（停牌）的行
    3. 确保数据按时间排序
    4. (可选) 计算好涨跌幅，避免在Env中重复计算
    """
    if df_raw is None or len(df_raw) < 100: # 如果数据太少，直接丢弃
        return None

    df = df_raw.copy()
    
    # 确保时间格式正确并排序
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    # 1. 剔除非法数据（收盘价为空 或 <=0）
    df = df[df['收盘'] > 0].dropna(subset=['收盘'])
    
    # 2. 剔除停牌数据（成交量为0）
    # 注意：如果停牌时间过长，这种暴力剔除会导致时间不连续。
    # 对于RL模型，通常关注K线序列的形态，时间上的跳跃（不连续）是可以接受的，
    # 只要价格接续是市场的真实开盘价格即可。
    df = df[df['成交量'] > 0]
    
    # 3. 再次重置索引
    df = df.reset_index(drop=True)
    # 4. 检查长度
    # 如果清洗后剩下的长度不足以支撑一个完整的训练窗口 (Window + Training Days)
    # 则该股票不适合用于训练
    min_length = WINDOW_SIZE + TRAINING_DAYS + 10
    if len(df) < min_length:
        return None
        
    return df