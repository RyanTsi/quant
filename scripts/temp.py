import pandas as pd

df = pd.read_parquet("C:/Users/sola/Documents/quant/.data/stock/2010/20100105.parquet")
df.to_csv("cur.csv", index=False)
print(df)