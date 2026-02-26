import os
import pandas as pd
import yaml
import utils.io
from data_pipeline.loader import DataLoader
import time
import requests

# index_code_list = utils.io.read_file_lines('C:/Users/sola/Documents/quant/.data/index_code_list')
# with open('config/settings.yaml', 'r', encoding='utf-8') as f:
#     config = yaml.load(f, Loader=yaml.SafeLoader)
# data_loader = DataLoader(config['data_path'])
# for symbol in index_code_list:
#     symbol = symbol.lower()
#     # print(symbol)
#     df = data_loader.fetch_index_history_by_symbol(symbol)
#     df['factor'] = 1.0
#     path = os.path.join(config['data_path'], 'cn_index')
#     os.makedirs(path, exist_ok=True)
#     if df is not None and not df.empty:
#         df.to_csv(os.path.join(path, f'{symbol}.csv'), index=False)
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["all_proxy"] = ""
os.environ["ALL_PROXY"] = ""

_original_request = requests.Session.request

def _patched_request(self, method, url, **kwargs):
    # 拦截请求，强行塞入真实的浏览器 Headers
    headers = kwargs.get("headers", {})
    headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Connection": "keep-alive"
    })
    kwargs["headers"] = headers
    # 调用原生的发包方法，把伪装后的请求发出去
    return _original_request(self, method, url, **kwargs)

# 狸猫换太子：用我们的伪装方法替换系统的原生方法
requests.Session.request = _patched_request
import akshare as ak
time.sleep(2)
df = ak.stock_zh_a_hist('000001', period='daily', start_date='20200101', end_date='20231231', adjust='hfq')
print(df)