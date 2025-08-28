import pickle
import os

path = "code_dict"

res = set()

list = os.listdir(path)

for i in list:
    with open(path + "/" + i, "rb") as f:
        code_dict = pickle.load(f)
        for i in code_dict.keys():
            c = i.split(".")[0]
            res.add(c)
        # print(res)
result_list = sorted(res)
print(result_list)
with open("all_stock_code.pkl", "wb") as f:
    pickle.dump(result_list, f)