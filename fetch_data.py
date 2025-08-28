import akshare as ak

start_date = "20100101"
end_date = "20250827"
period="daily"
adjust="hfq"

for stock_num in range(1, 999999):
    symbol = f"{stock_num:06d}"
    try:
        stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol, period, start_date, end_date, adjust)
        if stock_zh_a_hist_df.empty:
            print(f"{symbol} 获取成功")
        else:
            pass
    except:
        print(f"{symbol} 获取失败")
