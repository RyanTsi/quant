import os

list_path = os.path.join(os.path.dirname(__file__), '..\\temp\\stock_code_list')
if os.path.exists(list_path):
    with open(list_path, 'r') as f:
        stock_code_list = f.read().splitlines()
        print(stock_code_list)