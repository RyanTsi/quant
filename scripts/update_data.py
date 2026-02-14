import pandas as pd
import yaml
import random
import json
import os
from data_pipeline.loader import DataLoader
from data_pipeline.preprocesser import Preprocesser

def main():
    with open('config/settings.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    data_loader = DataLoader(config['data_path'])
    start_date = '20100101'
    end_date = '20260213'
    dir_name = f'{start_date}-{end_date}'
    # data_loader.fetch_all_stock_history(start_date, end_date)
    file_list = os.listdir(os.path.join(config['data_path'], dir_name))
    file_list = random.sample(file_list, 100)
    df_list = []
    for i, file in enumerate(file_list):
        df = data_loader.get_df_from_csv(f"{dir_name}/{file}")
        df_list.append(df)
    df_raw = pd.concat(df_list, axis=0, ignore_index=True)
    preprocesser = Preprocesser(df_raw)
    preprocesser.run()
    preprocesser.tar_df.to_csv(os.path.join(config['data_path'], 'processed_data.csv'), index=False)
    

if __name__ == '__main__':
    main()
