import akshare as ak
import time
import random
import pickle
import os

start_date = "20100101"
end_date = "202501214"
period = "daily"
adjust = "hfq"
save_path = "history/past15year_stock_data_daily"
if(os.path.exists(save_path) == False):
    os.makedirs(save_path)
path = "all_stock_code.pkl"

all_stock_code = pickle.load(open(path, "rb"))

for symbol in all_stock_code:
    retry = 0
    success = False
    file_path = os.path.join(save_path, f"{symbol}.csv")
    print(file_path)
    if os.path.exists(file_path):
        print(f"{symbol} 数据已存在, skip")
        continue
    while retry <= 3 and not success:
        try:
            stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol, period, start_date, end_date, adjust)
            
            if not stock_zh_a_hist_df.empty:
                print(f"{symbol} 获取成功")
                stock_zh_a_hist_df.to_csv(f"{save_path}/{symbol}.csv", index=False)
                success = True
                # 每次请求后固定随机休眠（0-2秒）
                time.sleep(random.uniform(0, 1.5))
            else:
                print(f"{symbol} not found")
                success = True  # 空数据视为处理完成，不再重试
                
        except Exception as e:
            retry += 1
            print(e)
            print(f"{symbol} 获取失败（第{retry}次重试）")
            
            if retry > 10:
                print(f"{symbol} 达到最大重试次数")
                break
                
            time.sleep(min(retry * 1.5, 3))  # 采用指数退避策略，1.5/3/4.5秒
    
    # 每个股票处理结束后固定休眠
    time.sleep(0.5)