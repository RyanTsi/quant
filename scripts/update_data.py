import pandas as pd
import yaml
import random
import json
import os
from data_pipeline.fetcher import StockDataFetcher
from data_pipeline.preprocesser import Preprocesser
import utils.io
from config.settings import settings

def main():
    data_fetcher = StockDataFetcher()
    start_date = '20100101'
    end_date = '20260319'
    dir_name = f'{start_date}-{end_date}'
    data_fetcher.fetch_all_stock_history(start_date, end_date)
    data_fetcher.fetch_all_index_history(start_date, end_date)
    

if __name__ == '__main__':
    main()
