import os

def read_file_lines(file_path):
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()
    return lines