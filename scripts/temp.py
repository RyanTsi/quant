import os
import pandas as pd
import yaml
import utils.io
from data_pipeline.loader import DataLoader

index_code_list = utils.io.read_file_lines('C:/Users/sola/Documents/quant/.data/index_code_list')
with open('config/settings.yaml', 'r', encoding='utf-8') as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
data_loader = DataLoader(config['data_path'])
for symbol in index_code_list:
    symbol = symbol.lower()
    # print(symbol)
    df = data_loader.fetch_index_history_by_symbol(symbol)
    df['factor'] = 1.0
    path = os.path.join(config['data_path'], 'cn_index')
    os.makedirs(path, exist_ok=True)
    if df is not None and not df.empty:
        df.to_csv(os.path.join(path, f'{symbol}.csv'), index=False)