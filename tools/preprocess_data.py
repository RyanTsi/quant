import numpy as np
import pandas as pd
import talib
from config import *

def preprocess_data(df_raw, df_index_raw):
    """
    清洗股票数据 v3
    """
    # --- 1. 基础检查 ---
    if df_raw is None or len(df_raw) < 100: return None
    if df_index_raw is None or len(df_index_raw) < 100: return None
    
    df = df_raw.copy()
    df_index = df_index_raw.copy()

    # --- 2. 时间清洗 ---
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    df_index['time'] = pd.to_datetime(df_index['time'])
    df_index = df_index.sort_values('time').reset_index(drop=True)
    # --- 3. 数据对齐 (个股 + 大盘) ---
    df_merged = pd.merge(
        df,
        df_index[['time', '收盘', '成交量']],
        on='time',
        how='left',
        suffixes=('', '_index')
    )
    df_merged['收盘_index'] = df_merged['收盘_index'].ffill()
    df_merged['收盘'] = df_merged['收盘'].ffill()
    df_merged['最高'] = df_merged['最高'].ffill()
    df_merged['最低'] = df_merged['最低'].ffill()
    df_merged['成交量'] = df_merged['成交量'].fillna(0)

    is_trading_day = df_merged['成交量'] > 0

    if len(df_merged) < 100: return None

    # ==========================================
    # 4. 特征工程 (Features)
    # ==========================================
    
    # 准备 Numpy 数组
    
    close = df_merged['收盘'].values
    high = df_merged['最高'].values if '最高' in df_merged.columns else close
    low = df_merged['最低'].values if '最低' in df_merged.columns else close
    volume = df_merged['成交量'].values.astype(float)
    # --- A. 动量类 ---
    # 1. RSI (相对强弱)
    df_merged['feat_rsi_14'] = talib.RSI(close, timeperiod=14) / 100.0
    
    # 2. MFI
    df_merged['feat_mfi_14'] = talib.MFI(high, low, close, volume, timeperiod=14) / 100.0

    # --- B. 趋势类 ---
    # 3. MACD Hist (柱状图)
    _, _, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df_merged['feat_macd_hist'] = hist / close

    # 4. 个股/大盘 Bias (5, 20, 60)
    for window in [5, 20, 60]:
        ma = df_merged['收盘'].rolling(window=window).mean()
        df_merged[f'feat_bias_{window}'] = (df_merged['收盘'] / ma) - 1.0
        idx_ma = df_merged['收盘_index'].rolling(window=window).mean()
        df_merged[f'feat_idx_bias_{window}'] = (df_merged['收盘_index'] / idx_ma) - 1.0
        
    # --- C. 波动类 ---
    # 6. ATR (归一化)
    if '最高' in df_merged.columns:
        df_merged['feat_atr_norm'] = talib.ATR(high, low, close, timeperiod=14) / close
    else:
        # 如果没有高低价，用滚动波动率代替
        df_merged['feat_atr_norm'] = df_merged['收盘'].pct_change().rolling(14).std()

    # 7. Bollinger Position (布林带位置)
    upper, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    bb_width = upper - lower
    # 避免除以0
    df_merged['feat_bb_pos'] = np.where(bb_width == 0, 0, (close - lower) / bb_width)

    # --- D. 量能类 ---
    # 8. OBV Trend
    obv = talib.OBV(close, volume)
    obv_series = pd.Series(obv)
    obv_change = obv_series.diff(20)
    vol_sum = df_merged['成交量'].rolling(20).sum()
    df_merged['feat_obv_trend'] = obv_change / (vol_sum + 1e-8)
    
    # 9. Volume Ratio (量比)
    vol_ma20 = df_merged['成交量'].rolling(20).mean()
    df_merged['feat_vol_ratio'] = df_merged['成交量'] / (vol_ma20 + 1e-8)

    # --- E. 其他逻辑特征 ---
    # 10. Time Gap (对数时间间隔)
    df_merged['diff_days'] = df_merged['time'].diff().dt.days.fillna(1.0)
    df_merged['feat_log_time_gap'] = np.log1p(df_merged['diff_days'])

    # 11. Excess Return (超额收益)
    df_merged['feat_log_ret'] = np.log(df_merged['收盘'] / df_merged['收盘'].shift(1))
    df_merged['feat_log_ret_index'] = np.log(df_merged['收盘_index'] / df_merged['收盘_index'].shift(1))
    df_merged['feat_excess_ret'] = df_merged['feat_log_ret'] - df_merged['feat_log_ret_index']
    
    # 12. 20日波动率
    df_merged['feat_vol_20'] = df_merged['feat_log_ret'].rolling(20).std()

    # ==========================================
    # 5. 标签生成 (Labels)
    # ==========================================

    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=5)
    # Label 1: 明日收益 (预测目标)
    df_merged['label_1d'] = df_merged['feat_log_ret'].shift(-1)
    
    # Label 2: 未来 5 日均价收益
    df_merged['label_5d'] = np.log(df_merged['收盘'].shift(-1).rolling(window=indexer).mean() / df_merged['收盘'])

    # Label 3: 未来 5 日波动风险
    df_merged['label_vol'] = df_merged['label_1d'].rolling(window=indexer).std()

    # --- 6. 清理 NaN ---
    # 剔除停牌日
    df_final = df_merged[is_trading_day].copy()
    # df_final = df_merged[df_merged['成交量'] > 0].copy()
    df_final = df_final.replace([np.inf, -np.inf], np.nan).dropna()
    
    return df_final