import os
import pandas as pd
import yaml
import time
import requests
import tushare as ts
import utils.io

token = ''
ts.set_token(token)
pro = ts.pro_api()

def main():
    with open('config/settings.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    data_path = config['data_path']
    stock_code_list = utils.io.read_file_lines(os.path.join(data_path, 'stock_code_list'))

    start_date_list = ['20100101', '20140101']
    end_date_list = ['20131231','20260227']
    for stock_code in stock_code_list:
        dir_name = f'{start_date_list[0]}-{end_date_list[-1]}'
        stock_code = stock_code[2:] + '.' + stock_code[:2]
        file_path = os.path.join(data_path, dir_name, f'{stock_code}.csv')
        if os.path.exists(file_path):
            continue
        print(f'Fetching history for {stock_code}...')
        os.makedirs(os.path.join(data_path, dir_name), exist_ok=True)
        df_all = pd.DataFrame()
        for start_date, end_date in zip(start_date_list, end_date_list):
            df = pro.daily(ts_code=stock_code, start_date=start_date, end_date=end_date)
            time.sleep(1.5)
            if df is None or df.empty:
                continue
            df_all = pd.concat([df_all, df], ignore_index=True)
        df_all.sort_values(by='trade_date', inplace=True)
        df_all.to_csv(os.path.join(data_path, dir_name, f'{stock_code}.csv'), index=False)

if __name__ == '__main__':
    main()
