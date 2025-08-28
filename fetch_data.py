import akshare as ak
import time
import random
import pickle

start_date = "20100101"
end_date = "20250827"
period = "daily"
adjust = "hfq"

path = "all_stock_code.pkl"

all_stock_code = pickle.load(open(path, "rb"))

for symbol in all_stock_code:
    retry = 0
    success = False
    
    while retry <= 3 and not success:
        try:
            stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol, period, start_date, end_date, adjust)
            
            if not stock_zh_a_hist_df.empty:
                print(f"{symbol} 获取成功")
                stock_zh_a_hist_df.to_excel(f"history/{symbol}.xlsx", index=False)
                success = True
                # 每次请求后固定随机休眠（0-2秒）
                time.sleep(random.uniform(0, 2))
            else:
                print(f"{symbol} not found")
                success = True  # 空数据视为处理完成，不再重试
                
        except Exception as e:
            retry += 1
            print(e)
            print(f"{symbol} 获取失败（第{retry}次重试）")
            
            if retry > 3:
                print(f"{symbol} 达到最大重试次数")
                break
                
            time.sleep(min(retry * 1.5, 3))  # 采用指数退避策略，1.5/3/4.5秒
    
    # 每个股票处理结束后固定休眠
    time.sleep(0.5)