import pandas as pd
import yaml
import random
import json
import os
from data_pipeline.loader import DataLoader
from data_pipeline.preprocesser import Preprocesser
import utils.io

def main():
    with open('config/settings.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    data_path = config['data_path']
    stock_code_list = utils.io.read_file_lines(os.path.join(data_path, 'stock_code_list'))
    data_loader = DataLoader(data_path)
    start_date = '20100101'
    end_date = '20260213'
    dir_name = f'{start_date}-{end_date}'
    # data_loader.fetch_all_stock_history(start_date, end_date)
    data_loader.fetch_all_stock_history(start_date, end_date)
    

if __name__ == '__main__':
    main()
