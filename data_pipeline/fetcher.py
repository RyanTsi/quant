import functools
import akshare as ak
import tushare as ts
import pandas as pd
import time
import os
import utils.io
import utils.format
from config.settings import settings

def retry(max_retries=3, delay=2):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"[{func.__name__}] retry ({i+1}/{max_retries}): {e}")
                    time.sleep(delay)
            print(f"[{func.__name__}] failed :(")
            return None 
        return wrapper
    return decorator


class DataFetcher:
    def __init__(self):
        self.data_path = settings.data_path
        ts.set_token(settings.tu_token)
        self.pro = ts.pro_api()
        stock_code_list_path = os.path.join(self.data_path, 'stock_code_list')
        index_code_list_path = os.path.join(self.data_path, 'index_code_list')
        self.stock_code_list = utils.io.read_file_lines(stock_code_list_path) if os.path.exists(stock_code_list_path) else None
        self.index_code_list = utils.io.read_file_lines(index_code_list_path) if os.path.exists(index_code_list_path) else None
        self.period = 'daily'
        self.adjust = 'hfq'

    @retry(max_retries=3, delay=2)
    def fetch_current_stock_spot_df(self):
        df = ak.stock_zh_a_spot_em()
        df = self._filter_current_stock_spot_df(df)
        return df

    @retry(max_retries=3, delay=2)
    def fetch_current_index_spot_df(self):
        df = ak.stock_zh_index_spot_em()
        return df

    @retry(max_retries=3, delay=2)
    def fetch_stock_history_by_symbol(self, symbol: str, start_date: str, end_date: str):
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        if end_year - start_year > 20:
            middle_year = (start_year + end_year) // 2
            start_date_list = [start_date, f'{middle_year}0101']
            end_date_list = [f'{middle_year-1}1231', end_date]
        else:
            start_date_list = [start_date]
            end_date_list = [end_date]

        df_all = pd.DataFrame()
        for m_start_date, m_end_date in zip(start_date_list, end_date_list):
            df = self.pro.daily(ts_code=symbol, start_date=m_start_date, end_date=m_end_date)
            time.sleep(1.5)
            if df is None or df.empty:
                continue
            df_all = pd.concat([df_all, df], ignore_index=True)
        df_all.sort_values(by='trade_date', inplace=True)
        return df_all
    
    @retry(max_retries=3, delay=2)
    def fetch_index_history_by_symbol(self, symbol: str):
        symbol = symbol.lower()
        try:
            df = ak.stock_zh_index_daily_em(symbol)
            if df is None or df.empty: return None
        except Exception as e:
            df = ak.stock_zh_index_daily(symbol)
        target_df = self.rename_and_clean_df_columns(df)
        return target_df
    
    def fetch_all_stock_history(self, start_date: str, end_date: str):
        full_dir_path = os.path.join(self.data_path, f'{start_date}-{end_date}')
        os.makedirs(full_dir_path, exist_ok=True)
        if os.path.exists(self.stock_code_list_path):
            stock_code_list = utils.io.read_file_lines(self.stock_code_list_path)
        else:
            print("fetch stock code list online...")
            df = self.fetch_current_stock_spot_df()
            if df is None: return
            stock_code_list = df['代码']
        for i in range(len(stock_code_list)):
            stock_code_list[i] = utils.format.format_stock_code(stock_code_list[i])
        total = len(stock_code_list)
        print(f"获取成功，共 {total} 只股票。")

        for i, symbol in enumerate(stock_code_list):
            if i % 100 == 0: print(f"进度: {i}/{total} ...")
            file_path = os.path.join(full_dir_path, f'{symbol}.csv')
            if os.path.exists(file_path): continue
            history_df = self.fetch_stock_history_by_symbol(symbol, start_date, end_date)
            time.sleep(2)
            if history_df is not None and not history_df.empty:
                history_df.to_csv(file_path, index=False)
            else:
                print(f'{symbol} skipped (no data)')

    def fetch_all_index_history(self):
        if os.path.exists(self.index_code_list_path):
            index_code_list = utils.io.read_file_lines(self.index_code_list_path)
        else:
            print("fetch index code list online...")
            df = self.fetch_current_index_spot_df()
            if df is None: return
            index_code_list = df['代码']
        total = len(index_code_list)
        print(f"获取成功，共 {total} 只指数。")

        for i, symbol in enumerate(index_code_list):
            symbol = symbol.lower()
            if i % 20 == 0: print(f"进度: {i}/{total} ...")
            file_path = os.path.join(self.data_path, 'cn_index', f'{symbol}.csv')
            if os.path.exists(file_path): continue
            history_df = self.fetch_index_history_by_symbol(symbol)
            time.sleep(2)
            if history_df is not None and not history_df.empty:
                history_df.to_csv(file_path, index=False)
            else:
                print(f'{symbol} skipped (no data)')

    def rename_and_clean_df_columns(self, df):
        if df is None: return None
        rename_map = {
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '复权因子': 'factor',
        }
        df = df.rename(columns=rename_map)
        if 'factor' not in df.columns:
            df['factor'] = 1.0
        target_columns = [col for col in rename_map.values() if col in df.columns]
        return df[target_columns]

    def get_df_from_csv(self, csv_file):
        file_path = os.path.join(self.data_path, csv_file)
        if not os.path.exists(file_path): return None
        df = pd.read_csv(file_path, dtype={'symbol': str})
        return df
    
    def _pure_stock_code(self, code):
        return "".join(c for c in code if c.isdigit())