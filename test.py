import pickle

path = "all_stock_code.pkl"

a = pickle.load(open(path, "rb"))
print(a)