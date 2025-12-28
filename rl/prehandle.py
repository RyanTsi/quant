import pandas as pd
from config import *

def preprocess_data(df_raw):
    """
    清洗股票数据 v2.0 (包含 ST 剔除逻辑)
    1. 去除空值
    2. 去除成交量为0（停牌）的行
    3. 确保数据按时间排序
    4. 剔除 ST 股（基于名称或价格行为）
    5. 长度检查
    """
    if df_raw is None or len(df_raw) < 100: 
        return None

    df = df_raw.copy()
    
    # 确保时间格式正确并排序
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
    
    # 1. 剔除非法数据
    df = df[df['收盘'] > 0].dropna(subset=['收盘'])
    
    # 2. 剔除停牌数据（成交量为0）
    df = df[df['成交量'] > 0]
    
    # ==========================================
    # 3. === 新增逻辑: ST / 科创 / 创业板 过滤 ===
    # ==========================================
    
    # --- A. 基于代码前缀的过滤 (只做深沪主板) ---
    # 假设有一列 'code'，如果没有则跳过此步
    if 'stock_code' in df.columns:
        # 取第一行的代码进行判断
        code_str = str(df['stock_code'].iloc[0])
        # 如果是 688(科创), 300(创业), 8xx/4xx(北交所) 则直接丢弃
        if code_str.startswith(('688', '300', '8', '4')):
            return None

    # --- B. 基于名称的显式过滤 (最准确) ---
    # 如果数据源包含股票名称（如 'name', '股票名称'）
    name_col = next((col for col in df.columns if col in ['name', '名称', '股票名称']), None)
    if name_col:
        # 只要历史上任何一天名字里带 'ST'，直接整只股票丢弃 (黑名单策略)
        if df[name_col].str.contains('ST', case=False, na=False).any():
            return None

    # --- C. 基于价格行为的隐式过滤 (兜底策略) ---
    # 计算涨跌幅 (绝对值)
    # 注意：这里临时计算用于检查，不影响后续特征工程
    pct_chg = df['收盘'].pct_change().abs() * 100
    
    # C1. 检查“死股”或“长期ST股”
    # 逻辑：如果一只股票在全历史中，最大单日波动从未超过 5.1%，
    # 说明它要么是长期 ST，要么是流动性枯竭的死股。
    # 正常主板股票一定会有涨停(10%)的时候。
    if pct_chg.max() < 5.1: 
        return None

    # C2. 检查“曾经戴帽”特征 (更严格)
    # 逻辑：ST 股的特征是“5%一字板”。
    # 如果发现某天：最高价==最低价 (一字板) 且 涨跌幅在 4.9%~5.1% 之间
    # 那么这只股极大概率在那段时间是 ST。为了数据纯净，建议丢弃。
    
    is_limit_lock = (df['最高'] == df['最低'])
    is_st_range = (pct_chg > 4.9) & (pct_chg < 5.1)
    
    # 如果存在这样的日子，视为被 ST 污染过，丢弃
    if (is_limit_lock & is_st_range).any():
        return None

    # ==========================================
    # 逻辑结束
    # ==========================================

    # 4. 再次重置索引 (因为 dropna 和 停牌过滤可能删了行)
    df = df.reset_index(drop=True)
    
    # 5. 检查长度
    min_length = WINDOW_SIZE + TRAINING_DAYS + 22
    if len(df) < min_length:
        return None
        
    return df