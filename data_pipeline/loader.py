import functools
import akshare as ak
import pandas as pd
import time
import os

def retry(max_retries=3, delay=2):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    print(f"[{func.__name__}] 报错重试 ({i+1}/{max_retries}): {e}")
                    time.sleep(delay)
            # 如果重试全部失败，打印错误并返回 None 
            print(f"[{func.__name__}] 最终失败，已放弃。")
            return None 
        return wrapper
    return decorator


class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.period = 'daily'
        self.adjust = 'hfq'

    @retry(max_retries=3, delay=2)
    def fetch_current_stock_spot_df(self):
        df = ak.stock_zh_a_spot_em()
        df = self._filter_current_stock_spot_df(df)
        return df
    
    def _filter_current_stock_spot_df(self, df):
        if df is None: return None
        df['最新价'] = pd.to_numeric(df['最新价'], errors='coerce')
        df = df.dropna(subset=['最新价'])
        mask_not_st = ~df['名称'].str.contains('ST', na=False, case=False)
        mask_main_board = df['代码'].str.startswith(('00', '60'), na=False)
        filtered_df = df[mask_not_st & mask_main_board].copy()
        return filtered_df.reset_index(drop=True)

    @retry(max_retries=3, delay=2)
    def fetch_current_index_spot_df(self):
        df = ak.stock_zh_index_spot_em()
        return df

    @retry(max_retries=3, delay=2)
    def fetch_stock_history_by_symbol(self, symbol: str, start_date: str, end_date: str):
        df = ak.stock_zh_a_hist(symbol, period=self.period, start_date=start_date, end_date=end_date, adjust=self.adjust)
        target_df = self.rename_and_clean_df_columns(df)
        return target_df
    
    @retry(max_retries=3, delay=2)
    def fetch_index_history_by_symbol(self, symbol: str):
        df = ak.stock_zh_index_daily(symbol)
        if df is None or df.empty: return None
        df['symbol'] = symbol
        target_df = self.rename_and_clean_df_columns(df)
        return target_df
    
    def fetch_all_stock_history(self, start_date: str, end_date: str):
        full_dir_path = os.path.join(self.data_path, f'{start_date}-{end_date}')
        os.makedirs(full_dir_path, exist_ok=True)
        df = self.fetch_current_stock_spot_df()
        if df is None: return

        total = len(df)
        print(f"获取成功，共 {total} 只股票。")

        for i, symbol in enumerate(df['代码']):
            if i % 100 == 0: print(f"进度: {i}/{total} ...")
            
            file_path = os.path.join(full_dir_path, f'{symbol}.csv')
            if os.path.exists(file_path): continue
            history_df = self.fetch_stock_history_by_symbol(symbol, start_date, end_date)
            time.sleep(2)
            if history_df is not None and not history_df.empty:
                history_df.to_csv(file_path, index=False)
            else:
                print(f'{symbol} skipped (no data)')

    def rename_and_clean_df_columns(self, df):
        if df is None: return None
        rename_map = {
            '日期': 'date',
            '股票代码': 'symbol',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
        }
        df = df.rename(columns=rename_map)
        target_columns = [col for col in rename_map.values() if col in df.columns]
        return df[target_columns]

    def get_df_from_csv(self, csv_file):
        file_path = os.path.join(self.data_path, csv_file)
        if not os.path.exists(file_path): return None
        df = pd.read_csv(file_path, dtype={'symbol': str})
        return df