import pickle
from tools.preprocess_data import preprocess_data, analyze_feature_correlation

with open("train_data_v4.pkl","rb") as f:
    df = pickle.load(f)

res = preprocess_data(df[1], df[0])
analyze_feature_correlation(res)
# with open("res.csv", "w", newline='', encoding='utf-8') as f:
#     f.write(res.to_csv(index=False))