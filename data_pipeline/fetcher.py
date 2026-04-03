"""Fetch A-share market data via baostock."""

import functools
import logging
import os
import time

import baostock as bs
import pandas as pd

import utils.format
import utils.io
from runtime.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class StockDataFetcher:
    def __init__(self):
        self.data_path = settings.data_path
        stock_code_list_path = os.path.join(self.data_path, 'stock_code_list')
        index_code_list_path = os.path.join(self.data_path, 'index_code_list')
        self.stock_code_list = utils.io.read_file_lines(stock_code_list_path)
        self.index_code_list = utils.io.read_file_lines(index_code_list_path)
        self.period = 'daily'
        self.adjust = 'hfq'
        lg = bs.login()
        logger.info("baostock login: code=%s msg=%s", lg.error_code, lg.error_msg)

    def __del__(self):
        bs.logout()

    def fetch_history_data_via_baostock(self, symbol: str, start_date: str, end_date: str):
        start_date = utils.format.format_date(start_date, format='YYYY-MM-DD')
        end_date = utils.format.format_date(end_date, format='YYYY-MM-DD')
        symbol = utils.format.format_stock_code(symbol, prefix=True, uppercase=False, has_dot=True)
        adjustflag = "1" if self.adjust == 'hfq' else "2" if self.adjust == 'qfq' else "3"
        rs = bs.query_history_k_data_plus(
            symbol,
            "date,open,high,low,close,volume,amount,turn,tradestatus,isST",
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag=adjustflag,
        )
        if rs.error_code == '0':
            df = rs.get_data()
            df = self.rename_and_clean_df_columns(df)
            return df
        return pd.DataFrame()

    def _fetch_list_stock_history(self, stock_code_list: list, start_date: str, end_date: str, save_dir: str = None):
        if save_dir is None:
            full_dir_path = os.path.join(self.data_path, f'{start_date}-{end_date}')
        else:
            full_dir_path = save_dir
        os.makedirs(full_dir_path, exist_ok=True)

        stock_code_list = [
            utils.format.format_stock_code(c) for c in stock_code_list
            if not utils.format.format_stock_code(c).startswith('BJ')
        ]
        total = len(stock_code_list)
        logger.info("Fetching %d symbols: %s -> %s", total, start_date, end_date)

        for i, symbol in enumerate(stock_code_list):
            if i % 100 == 0:
                logger.info("Progress: %d/%d", i, total)
            file_path = os.path.join(full_dir_path, f'{symbol}.csv')
            if os.path.exists(file_path):
                continue
            history_df = self.fetch_history_data_via_baostock(symbol, start_date, end_date)
            time.sleep(1.0)
            if history_df is not None and not history_df.empty:
                history_df.to_csv(file_path, index=False)
            else:
                logger.debug("%s skipped (no data)", symbol)

    def fetch_all_stock_history(self, start_date: str, end_date: str, save_dir: str = None):
        if not self.stock_code_list:
            logger.warning("No stock code list loaded, skipping.")
            return
        stock_code_list = [utils.format.format_stock_code(c) for c in self.stock_code_list]
        self._fetch_list_stock_history(stock_code_list, start_date, end_date, save_dir)

    def fetch_all_index_history(self, start_date: str, end_date: str, save_dir: str = None):
        if not self.index_code_list:
            logger.warning("No index code list loaded, skipping.")
            return
        self._fetch_list_stock_history(self.index_code_list, start_date, end_date, save_dir)

    def rename_and_clean_df_columns(self, df):
        if df is None:
            return None
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
            '是否ST股': 'isST',
        }
        df = df.rename(columns=rename_map)
        if 'factor' not in df.columns:
            df['factor'] = 1.0
        target_columns = [col for col in rename_map.values() if col in df.columns]
        return df[target_columns]

    def _filter_current_stock_spot_df(self, df):
        if df is None:
            return None
        df['最新价'] = pd.to_numeric(df['最新价'], errors='coerce')
        df = df.dropna(subset=['最新价'])
        mask_not_st = ~df['名称'].str.contains('ST', na=False, case=False)
        filtered_df = df[mask_not_st].copy()
        return filtered_df.reset_index(drop=True)
