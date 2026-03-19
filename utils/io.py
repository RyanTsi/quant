import os
import pandas as pd

def read_file_lines(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    return lines

def get_df_from_csv(csv_file):
    if not os.path.exists(csv_file): return None
    df = pd.read_csv(csv_file, dtype={'symbol': str})
    return df