import os
import random
from datetime import timedelta

import pandas as pd

import utils.io
from config.settings import settings
from data_pipeline.database import DBClient


def _quarter_ranges(start_year: int, end_date: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(f"{start_year}-01-01")
    quarters = pd.period_range(start=start, end=end_date, freq="Q")
    return [(q.start_time.normalize(), q.end_time.normalize()) for q in quarters]


def _target_source_quarter_pairs(
    quarters: list[tuple[pd.Timestamp, pd.Timestamp]]
) -> list[tuple[pd.Period, pd.Timestamp, pd.Timestamp]]:
    """
    Build (source_quarter, target_start, target_end) pairs.
    Target quarter uses previous quarter's liquidity (one-quarter lag).
    """
    pairs: list[tuple[pd.Period, pd.Timestamp, pd.Timestamp]] = []
    if len(quarters) <= 1:
        return pairs
    for idx in range(1, len(quarters)):
        target_start, target_end = quarters[idx]
        source_start, _source_end = quarters[idx - 1]
        source_quarter = pd.Period(source_start, freq="Q")
        pairs.append((source_quarter, target_start, target_end))
    return pairs


def _sum_turnover_by_quarter(rows: list[dict], start_year: int, end_year: int) -> pd.Series:
    if not rows:
        return pd.Series(dtype=float)

    df = pd.DataFrame(rows)
    if "date" not in df.columns:
        return pd.Series(dtype=float)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    # to_period("Q") drops timezone anyway; strip it explicitly to avoid noisy warnings.
    if getattr(df["date"].dt, "tz", None) is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df = df[(df["date"] >= f"{start_year}-01-01") & (df["date"] <= f"{end_year}-12-31")]
    if df.empty:
        return pd.Series(dtype=float)

    # Use amount when available; fallback to close*volume.
    if "amount" in df.columns:
        turnover = pd.to_numeric(df["amount"], errors="coerce")
    else:
        close = pd.to_numeric(df.get("close", 0), errors="coerce").fillna(0)
        volume = pd.to_numeric(df.get("volume", 0), errors="coerce").fillna(0)
        turnover = close * volume

    if "tradestatus" in df.columns:
        tradestatus = pd.to_numeric(df["tradestatus"], errors="coerce").fillna(1)
        turnover = turnover.where(tradestatus != 0, 0)

    df["turnover"] = turnover.fillna(0)
    df["quarter"] = df["date"].dt.to_period("Q")
    return df.groupby("quarter")["turnover"].sum()


def _sample_symbols_for_quarter(df_q: pd.DataFrame, rng: random.Random, top_n: int, segment_count: int) -> list[str]:
    top = df_q.sort_values("turnover", ascending=False).head(top_n)
    if top.empty:
        return []

    selected: list[str] = []
    segment_size = max(1, top_n // max(1, segment_count))
    for seg in pd.Series(top["symbol"]).reset_index(drop=True).groupby(lambda i: i // segment_size):
        symbols = list(seg[1])
        if len(symbols) <= 40:
            continue
        middle = symbols[20:-20]
        k = min(80, len(middle))
        selected.extend(rng.sample(middle, k))
    return selected


def _merge_contiguous_ranges(df_ranges: pd.DataFrame) -> pd.DataFrame:
    if df_ranges.empty:
        return df_ranges

    merged_rows = []
    for symbol, grp in df_ranges.sort_values(["symbol", "start_date"]).groupby("symbol"):
        start = pd.Timestamp(grp.iloc[0]["start_date"])
        end = pd.Timestamp(grp.iloc[0]["end_date"])
        for _, row in grp.iloc[1:].iterrows():
            cur_start = pd.Timestamp(row["start_date"])
            cur_end = pd.Timestamp(row["end_date"])
            if cur_start <= end + timedelta(days=1):
                end = max(end, cur_end)
            else:
                merged_rows.append(
                    {"symbol": symbol, "start_date": start.strftime("%Y-%m-%d"), "end_date": end.strftime("%Y-%m-%d")}
                )
                start, end = cur_start, cur_end
        merged_rows.append({"symbol": symbol, "start_date": start.strftime("%Y-%m-%d"), "end_date": end.strftime("%Y-%m-%d")})
    return pd.DataFrame(merged_rows)


def filter_top_liquidity(start_year=2010, end_year=2026, top_n=2000, random_seed=42):
    db_client = DBClient(settings.db_host, settings.db_port)
    all_stock_code_list_path = os.path.join(settings.data_path, "stock_code_list")
    all_stock_list = utils.io.read_file_lines(all_stock_code_list_path)
    today = pd.Timestamp.today().normalize()
    requested_end = pd.Timestamp(f"{end_year}-12-31")
    effective_end = min(today, requested_end)
    quarters = _quarter_ranges(start_year, effective_end)
    rng = random.Random(random_seed)

    stock_quarter_turnover = []
    query_start = f"{start_year}-01-01"
    query_end = effective_end.strftime("%Y-%m-%d")
    for stock_code in all_stock_list:
        response = db_client.query_data(stock_code, query_start, query_end)
        if response is None or response.status_code != 200:
            continue
        rows = (response.json() or {}).get("data", [])
        quarter_sum = _sum_turnover_by_quarter(rows, start_year, end_year)
        for q, turnover in quarter_sum.items():
            stock_quarter_turnover.append({"symbol": stock_code, "quarter": q, "turnover": float(turnover)})

    if not stock_quarter_turnover:
        out_txt = os.path.join(settings.data_path, "my_800_stocks.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("")
        return

    df = pd.DataFrame(stock_quarter_turnover)
    selected_ranges = []
    for source_q, q_start, q_end in _target_source_quarter_pairs(quarters):
        df_q = df[df["quarter"] == source_q]
        symbols = _sample_symbols_for_quarter(df_q, rng, top_n=top_n, segment_count=10)
        for sym in symbols:
            selected_ranges.append({"symbol": sym, "start_date": q_start.strftime("%Y-%m-%d"), "end_date": q_end.strftime("%Y-%m-%d")})

    if not selected_ranges:
        out_txt = os.path.join(settings.data_path, "my_800_stocks.txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("")
        return

    df_ranges = pd.DataFrame(selected_ranges)

    # If the final period is a partial quarter, merge it with previous quarter.
    if quarters:
        last_start, last_end = quarters[-1]
        if effective_end < last_end and len(quarters) >= 2:
            prev_start, _prev_end = quarters[-2]
            last_start_str = last_start.strftime("%Y-%m-%d")
            prev_start_str = prev_start.strftime("%Y-%m-%d")
            df_ranges.loc[df_ranges["start_date"] == last_start_str, "start_date"] = prev_start_str

    merged = _merge_contiguous_ranges(df_ranges)

    out_txt = os.path.join(settings.data_path, "my_800_stocks.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        for _, row in merged.iterrows():
            f.write(f"{row['symbol']}\t{row['start_date']}\t{row['end_date']}\n")
    print(f"Saved to {out_txt}")


if __name__ == "__main__":
    filter_top_liquidity()
