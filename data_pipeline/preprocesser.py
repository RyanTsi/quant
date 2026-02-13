import numpy as np
import pandas as pd
import akshare as ak
import talib

class Preprocesser:
    def __init__(self, raw_df, index_symbol):
        self.raw_df = raw_df
        self.index_symbol = index_symbol
        self.tar_df = None

    def _is_main_board(self, df):
        if 'symbol' not in df.columns or len(df) == 0:
            return False
        symbol_str = str(df['symbol'].iloc[0])
        valid_prefixes = ('600','601','603','000','002')
        exclude_prefixes = ('688','300','8','4')
        if symbol_str.startswith(exclude_prefixes):
            return False
        return symbol_str.startswith(valid_prefixes)

    def _filter_clean(self, df):
        df = df[df['close'] > 0].dropna(subset=['close'])
        df = df[df['volume'] > 0]
        if 'high' in df.columns and 'low' in df.columns and 'pct_chg' in df.columns:
            is_limit_lock = (df['high'] == df['low'])
            is_st_range = (df['pct_chg'] > 4.9) & (df['pct_chg'] < 5.1)
            if (is_limit_lock & is_st_range).any():
                return None
        df = df.reset_index(drop=True)
        min_length = 60 + 252 + 22
        if len(df) < min_length:
            return None
        return df

    def _merge_index(self, df, idx_df):
        m = pd.merge(df, idx_df, on='date', how='left')
        m['index_close'] = m['index_close'].ffill()
        return m

    def _ema(self, series, span):
        return series.ewm(span=span, adjust=False).mean()

    def _compute_features(self, df):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        close = df['close']
        high = df['high'] if 'high' in df.columns else close
        low = df['low'] if 'low' in df.columns else close
        volume = df['volume'].astype(float)
        df['log_ret'] = np.log(close / close.shift(1))
        if 'index_close' in df.columns:
            df['log_ret_index'] = np.log(df['index_close'] / df['index_close'].shift(1))
            df['excess_ret'] = df['log_ret'] - df['log_ret_index']
        else:
            df['log_ret_index'] = 0.0
            df['excess_ret'] = df['log_ret']
        for window in [5,20,60]:
            ma = close.rolling(window).mean()
            df[f'bias_{window}'] = (close / (ma + 1e-8)) - 1.0
            if 'index_close' in df.columns:
                idx_ma = df['index_close'].rolling(window).mean()
                df[f'idx_bias_{window}'] = (df['index_close'] / (idx_ma + 1e-8)) - 1.0
            else:
                df[f'idx_bias_{window}'] = 0.0
        ema12 = self._ema(close, 12)
        ema26 = self._ema(close, 26)
        macd_line = ema12 - ema26
        signal = self._ema(macd_line, 9)
        hist = macd_line - signal
        df['macd_hist'] = hist / (close + 1e-8)
        prev_close = close.shift(1).fillna(close)
        tr1 = (high - low).abs()
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        df['atr_norm'] = atr / (close + 1e-8)
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        width = (upper - lower)
        df['bb_pos'] = np.where(width == 0, 0, (close - lower) / (width + 1e-8))
        sign_change = np.sign(close.diff().fillna(0.0))
        obv = (sign_change * volume).cumsum()
        obv_change = obv.diff(20)
        vol_sum = volume.rolling(20).sum()
        df['obv_trend'] = obv_change / (vol_sum + 1e-8)
        vol_ma20 = volume.rolling(20).mean()
        df['vol_ratio'] = volume / (vol_ma20 + 1e-8)
        df['diff_days'] = df['date'].diff().dt.days.fillna(1.0)
        df['log_time_gap'] = np.log1p(df['diff_days'])
        df['vol_20'] = df['log_ret'].rolling(20).std()
        try:
            indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=5)
            df['label_1d'] = df['log_ret'].shift(-1)
            df['label_5d'] = np.log(close.shift(-1).rolling(window=indexer).mean() / close)
            df['label_vol'] = df['label_1d'].rolling(window=indexer).std()
        except Exception:
            df['label_1d'] = df['log_ret'].shift(-1)
            df['label_5d'] = df['log_ret'].rolling(5).mean().shift(-4)
            df['label_vol'] = df['label_1d'].rolling(5).std().shift(-4)
        df_final = df[df['volume'] > 0].copy()
        df_final = df_final.replace([np.inf, -np.inf], np.nan).dropna()
        if len(df_final) < 100:
            return None
        return df_final

    def process(self):
        df = self._rename_stock_columns(self.raw_df.copy())
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
        for col in self.schema:
            if col not in df.columns:
                df[col] = None
        df = df[self.schema]
        if not self._is_main_board(df):
            self.tar_df = None
            return None
        df = self._filter_clean(df)
        if df is None:
            self.tar_df = None
            return None
        idx_df = self._load_index_df(self.index_symbol)
        if idx_df is not None:
            df = self._merge_index(df, idx_df)
        res = self._compute_features(df)
        self.tar_df = res
        return res
    
