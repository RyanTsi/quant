import functools
import akshare as ak
import pandas as pd
import time
import os
import utils.io
import utils.format
from config.settings import settings
import baostock as bs

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


class StockDataFetcher:
    def __init__(self):
        self.data_path = settings.data_path
        stock_code_list_path = os.path.join(self.data_path, 'stock_code_list')
        index_code_list_path = os.path.join(self.data_path, 'index_code_list')
        csi500_code_list_path = os.path.join(self.data_path, 'csi500_code_list')
        self.stock_code_list = utils.io.read_file_lines(stock_code_list_path)
        self.index_code_list = utils.io.read_file_lines(index_code_list_path)
        self.csi500_code_list = utils.io.read_file_lines(csi500_code_list_path)
        self.period = 'daily'
        self.adjust = 'hfq'
        lg = bs.login()
        print('login respond error_code:'+lg.error_code)
        print('login respond  error_msg:'+lg.error_msg)
    
    def __del__(self):
        bs.logout()

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
        symbol = utils.format.format_stock_code(symbol, prefix=True, uppercase=False)
        pure_symbol = utils.format.pure_stock_code(symbol)
        try:
            df = ak.stock_zh_a_hist(pure_symbol, period=self.period, start_date=start_date, end_date=end_date, adjust=self.adjust)
        except Exception as e:
            print(f"[fetch_stock_history_by_symbol] Error occurred while fetching history for {symbol}: {e}")
            df = ak.stock_zh_a_daily(symbol, start_date=start_date, end_date=end_date, adjust=self.adjust)
            df['volume'] /= 100.0
        target_df = self.rename_and_clean_df_columns(df)
        return target_df
    
    @retry(max_retries=3, delay=2)
    def fetch_index_history_by_symbol(self, symbol: str):
        symbol = utils.format.format_stock_code(symbol, prefix=True, uppercase=False)
        try:
            df = ak.stock_zh_index_daily_em(symbol)
            if df is None or df.empty: return None
        except Exception as e:
            df = ak.stock_zh_index_daily(symbol)
        target_df = self.rename_and_clean_df_columns(df)
        return target_df
    
    def fetch_history_data_via_baostock(self, symbol: str, start_date: str, end_date: str):
        start_date = utils.format.format_date(start_date, format='YYYY-MM-DD')
        end_date = utils.format.format_date(end_date, format='YYYY-MM-DD')
        symbol = utils.format.format_stock_code(symbol, prefix=True, uppercase=False, has_dot=True)
        adjustflag = "1" if self.adjust == 'hfq' else "2" if self.adjust == 'qfq' else "3"
        rs = bs.query_history_k_data_plus(symbol,"date,open,high,low,close,volume,amount,turn,tradestatus,isST",
                                        start_date=start_date, end_date=end_date, frequency="d", adjustflag=adjustflag)
        if rs.error_code == '0':
            result = rs.get_data()
        else:
            result = pd.DataFrame()
        return result

    def fetch_list_stock_history(self, stock_code_list: list, start_date: str, end_date: str):
        full_dir_path = os.path.join(self.data_path, f'{start_date}-{end_date}')
        os.makedirs(full_dir_path, exist_ok=True)
        for i in range(len(stock_code_list)):
            stock_code_list[i] = utils.format.format_stock_code(stock_code_list[i])
        stock_code_list_temp = []
        for symbol in stock_code_list:
            if symbol.startswith('BJ'): continue
            stock_code_list_temp.append(symbol)
        stock_code_list = stock_code_list_temp
        total = len(stock_code_list)
        print(f"获取成功，共 {total} 只股票。")
        for i, symbol in enumerate(stock_code_list):
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            if i % 100 == 0: print(f"进度: {i}/{total} ...")
            file_path = os.path.join(full_dir_path, f'{symbol}.csv')
            if os.path.exists(file_path): continue
            history_df = self.fetch_history_data_via_baostock(symbol, start_date, end_date)
            time.sleep(1.5)
            if history_df is not None and not history_df.empty:
                history_df.to_csv(file_path, index=False)
            else:
                print(f'{symbol} skipped (no data)')

    def fetch_all_stock_history(self, start_date: str, end_date: str):
        if self.stock_code_list:
            stock_code_list = self.stock_code_list
        else:
            print("fetch stock code list online...")
            df = self.fetch_current_stock_spot_df()
            if df is None: return
            stock_code_list = df['代码']
        for i in range(len(stock_code_list)):
            stock_code_list[i] = utils.format.format_stock_code(stock_code_list[i])
        self.fetch_list_stock_history(stock_code_list, start_date, end_date)
        
    def fetch_csi500_stock_history(self, start_date: str, end_date: str):
        if self.csi500_code_list:
            stock_code_list = self.csi500_code_list
        else:
            return
        print(f"获取成功，共 {total} 只股票。")
        self.fetch_list_stock_history(stock_code_list, start_date, end_date)

    def fetch_all_index_history(self, start_date: str, end_date: str):
        if self.index_code_list:
            index_code_list = self.index_code_list
        else:
            return
        total = len(index_code_list)
        print(f"获取成功，共 {total} 只指数。")
        self.fetch_list_stock_history(index_code_list, start_date, end_date)

    # def fetch_all_index_history(self):
    #     if not self.index_code_list:
    #         print("fetch index code list online...")
    #         df = self.fetch_current_index_spot_df()
    #         if df is None: return
    #         self.index_code_list = df['代码']
    #     total = len(self.index_code_list)
    #     print(f"获取成功，共 {total} 只指数。")

    #     for i, symbol in enumerate(self.index_code_list):
    #         if i % 100 == 0: print(f"进度: {i}/{total} ...")
    #         file_path = os.path.join(self.data_path, 'cn_index', f'{symbol}.csv')
    #         # if os.path.exists(file_path): continue
    #         history_df = self.fetch_index_history_by_symbol(symbol)
    #         history_df = self.rename_and_clean_df_columns(history_df)
    #         time.sleep(1.5)
    #         if history_df is not None and not history_df.empty:
    #             history_df.to_csv(file_path, index=False)
    #         else:
    #             print(f'{symbol} skipped (no data)')

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
            'total_turnover': 'amount',
            '成交额': 'amount',
            '换手率': 'turn',
            '交易状态': 'tradestatus',
            '是否ST股': 'isST'
        }
        df = df.rename(columns=rename_map)
        if 'factor' not in df.columns:
            df['factor'] = 1.0
        target_columns = [col for col in rename_map.values() if col in df.columns]
        return df[target_columns]
    
    def _filter_current_stock_spot_df(self, df):
        if df is None: return None
        df['最新价'] = pd.to_numeric(df['最新价'], errors='coerce')
        df = df.dropna(subset=['最新价'])
        mask_not_st = ~df['名称'].str.contains('ST', na=False, case=False)
        filtered_df = df[mask_not_st].copy()
        return filtered_df.reset_index(drop=True)
    
class NewsDataFetcher:
    def __init__(self):
        self.data_path = settings.data_path
        