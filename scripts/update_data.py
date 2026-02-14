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
    dir_name = "20100101-20260213"
    file_list = os.listdir(os.path.join(config['data_path'], dir_name))
    file_list = random.sample(file_list, 100)
    df_list = []
    for i, file in enumerate(file_list):
        if i % 100 == 0:
            print(f"已读取 {i} 个文件...")
        df = data_loader.get_df_from_csv(f"{dir_name}/{file}")
        df_list.append(df)
    df_raw = pd.concat(df_list, axis=0, ignore_index=True)
    preprocesser = Preprocesser(df_raw)
    preprocesser.run()
    preprocesser.analyze_feature_correlation(preprocesser.tar_df)
    print(preprocesser.tar_df)
    

if __name__ == '__main__':
    main()
