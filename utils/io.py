import os
import pandas as pd
from utils.format import format_stock_code

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

def package_data(input_dir, output_dir, max_rows=500_000):
    """Merge per-symbol CSVs into chunked output files.

    When the accumulated row count exceeds *max_rows*, the current
    chunk is flushed to disk and a new chunk is started.  Output files
    are named ``all_data_0.csv``, ``all_data_1.csv``, etc.
    """
    if not os.path.exists(input_dir):
        return
    os.makedirs(output_dir, exist_ok=True)

    csv_files = sorted(f for f in os.listdir(input_dir) if f.endswith('.csv'))
    if not csv_files:
        return

    chunk_idx = 0
    parts: list[pd.DataFrame] = []
    row_count = 0

    def _flush():
        nonlocal chunk_idx, parts, row_count
        if not parts:
            return
        merged = pd.concat(parts, ignore_index=True)
        merged.to_csv(
            os.path.join(output_dir, f"all_data_{chunk_idx}.csv"), index=False,
        )
        chunk_idx += 1
        parts = []
        row_count = 0

    for file in csv_files:
        symbol = format_stock_code(file.replace('.csv', ''))
        current_df = pd.read_csv(os.path.join(input_dir, file))
        current_df['symbol'] = symbol
        parts.append(current_df)
        row_count += len(current_df)

        if row_count >= max_rows:
            _flush()

    _flush()

