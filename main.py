import akshare as ak
import mplfinance as mpf  # Please install mplfinance as follows: pip install mplfinance

stock_zh_a_spot_em_df = ak.stock_zh_a_spot_em()

# stock_zh_a_spot_em_df.to_excel("stock_data.xlsx", index=False)



stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="19600301", end_date='19600528', adjust="hfq")

print(stock_zh_a_hist_df)