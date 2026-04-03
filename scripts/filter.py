import os
import random
from datetime import timedelta

import pandas as pd

import utils.io
from data_pipeline.database import DBClient
from model_function.universe import (
    TRAINING_UNIVERSE_DEFAULTS,
    TrainingUniverseConfig,
    collect_training_month_liquidity,
    select_training_symbols,
)
from runtime.config import get_settings
from utils.preprocess import read_symbol_list

settings = get_settings()


def _output_instruments_path() -> str:
    qlib_data_path = getattr(settings, "qlib_data_path", None)
    if qlib_data_path:
        out_dir = os.path.join(str(qlib_data_path), "instruments")
    else:
        out_dir = settings.data_path
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "my_800_stocks.txt")


def _month_ranges(start_year: int, end_date: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(f"{start_year}-01-01")
    months = pd.period_range(start=start, end=end_date, freq="M")
    return [(month.start_time.normalize(), month.end_time.normalize()) for month in months]


def _target_source_month_pairs(
    months: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[tuple[pd.Period, pd.Timestamp, pd.Timestamp]]:
    """
    Build (source_month, target_start, target_end) pairs.
    Target month uses previous month's liquidity (one-month lag).
    """
    pairs: list[tuple[pd.Period, pd.Timestamp, pd.Timestamp]] = []
    if len(months) <= 1:
        return pairs
    for idx in range(1, len(months)):
        target_start, target_end = months[idx]
        source_start, _source_end = months[idx - 1]
        pairs.append((pd.Period(source_start, freq="M"), target_start, target_end))
    return pairs


def _sample_symbols_for_month(
    df_m: pd.DataFrame,
    rng: random.Random,
    top_n: int,
    segment_count: int,
    *,
    total_select: int = 800,
    min_per_segment: int = 20,
    previous_symbols: set[str] | None = None,
    source_month: pd.Period | str = "unknown",
    entry_limit: int = TRAINING_UNIVERSE_DEFAULTS.entry_limit,
    exit_limit: int = TRAINING_UNIVERSE_DEFAULTS.exit_limit,
    seed: int = TRAINING_UNIVERSE_DEFAULTS.seed,
    excluded_symbols: set[str] | None = None,
) -> list[str]:
    del rng  # The redesign keeps training-only downsampling deterministic.
    if df_m.empty:
        return []

    month_frame = df_m.copy()
    if "lagged_liquidity" not in month_frame.columns:
        if "mean_turnover" in month_frame.columns:
            month_frame["lagged_liquidity"] = pd.to_numeric(month_frame["mean_turnover"], errors="coerce").fillna(0.0)
        else:
            return []

    if top_n > 0:
        month_frame = month_frame.sort_values("lagged_liquidity", ascending=False).head(int(top_n))

    config = TrainingUniverseConfig(
        liquidity_lookback_days=TRAINING_UNIVERSE_DEFAULTS.liquidity_lookback_days,
        liquidity_min_periods=TRAINING_UNIVERSE_DEFAULTS.liquidity_min_periods,
        entry_limit=entry_limit,
        exit_limit=exit_limit,
        sample_size=total_select,
        segment_count=segment_count,
        min_per_segment=min_per_segment,
        seed=seed,
    )
    return select_training_symbols(
        month_frame,
        previous_symbols,
        source_month=source_month,
        excluded_symbols=excluded_symbols,
        config=config,
    )


def _merge_contiguous_ranges(df_ranges: pd.DataFrame) -> pd.DataFrame:
    if df_ranges.empty:
        return df_ranges

    merged_rows = []
    for symbol, group in df_ranges.sort_values(["symbol", "start_date"]).groupby("symbol"):
        start = pd.Timestamp(group.iloc[0]["start_date"])
        end = pd.Timestamp(group.iloc[0]["end_date"])
        for _, row in group.iloc[1:].iterrows():
            current_start = pd.Timestamp(row["start_date"])
            current_end = pd.Timestamp(row["end_date"])
            if current_start <= end + timedelta(days=1):
                end = max(end, current_end)
                continue
            merged_rows.append({"symbol": symbol, "start_date": start.strftime("%Y-%m-%d"), "end_date": end.strftime("%Y-%m-%d")})
            start, end = current_start, current_end
        merged_rows.append({"symbol": symbol, "start_date": start.strftime("%Y-%m-%d"), "end_date": end.strftime("%Y-%m-%d")})
    return pd.DataFrame(merged_rows)


def filter_top_liquidity(start_year=2010, end_year=2026, top_n=2200, random_seed=42):
    db_client = DBClient(settings.db_host, settings.db_port)
    all_stock_code_list_path = os.path.join(settings.data_path, "stock_code_list")
    all_stock_list = utils.io.read_file_lines(all_stock_code_list_path)
    index_symbols = read_symbol_list(os.path.join(settings.data_path, "index_code_list"))

    today = pd.Timestamp.today().normalize()
    requested_end = pd.Timestamp(f"{end_year}-12-31")
    effective_end = min(today, requested_end)
    months = _month_ranges(start_year, effective_end)
    month_pairs = _target_source_month_pairs(months)
    source_months = [pair[0] for pair in month_pairs]
    rng = random.Random(random_seed)

    training_config = TrainingUniverseConfig(
        liquidity_lookback_days=TRAINING_UNIVERSE_DEFAULTS.liquidity_lookback_days,
        liquidity_min_periods=TRAINING_UNIVERSE_DEFAULTS.liquidity_min_periods,
        entry_limit=TRAINING_UNIVERSE_DEFAULTS.entry_limit,
        exit_limit=TRAINING_UNIVERSE_DEFAULTS.exit_limit,
        sample_size=TRAINING_UNIVERSE_DEFAULTS.sample_size,
        segment_count=TRAINING_UNIVERSE_DEFAULTS.segment_count,
        min_per_segment=TRAINING_UNIVERSE_DEFAULTS.min_per_segment,
        seed=random_seed,
    )

    monthly_metrics: list[dict] = []
    query_start = f"{start_year}-01-01"
    query_end = effective_end.strftime("%Y-%m-%d")
    for stock_code in all_stock_list:
        response = db_client.query_data(stock_code, query_start, query_end)
        if response is None or response.status_code != 200:
            continue
        rows = (response.json() or {}).get("data", [])
        monthly_metrics.extend(
            collect_training_month_liquidity(
                symbol=stock_code,
                rows=rows,
                source_months=source_months,
                start_year=start_year,
                end_year=end_year,
                config=training_config,
            )
        )

    out_txt = _output_instruments_path()
    if not monthly_metrics:
        with open(out_txt, "w", encoding="utf-8") as handle:
            handle.write("")
        return

    monthly_frame = pd.DataFrame(monthly_metrics)
    selected_ranges: list[dict] = []
    previous_symbols: set[str] = set()
    for source_month, month_start, month_end in month_pairs:
        month_frame = monthly_frame[monthly_frame["source_month"] == source_month]
        if month_frame.empty:
            previous_symbols = set()
            continue

        symbols = _sample_symbols_for_month(
            month_frame,
            rng=rng,
            top_n=top_n,
            segment_count=training_config.segment_count,
            total_select=training_config.sample_size,
            min_per_segment=training_config.min_per_segment,
            previous_symbols=previous_symbols,
            source_month=source_month,
            entry_limit=training_config.entry_limit,
            exit_limit=training_config.exit_limit,
            seed=training_config.seed,
            excluded_symbols=index_symbols,
        )
        previous_symbols = set(symbols)
        for symbol in symbols:
            selected_ranges.append(
                {
                    "symbol": symbol,
                    "start_date": month_start.strftime("%Y-%m-%d"),
                    "end_date": month_end.strftime("%Y-%m-%d"),
                }
            )

    if not selected_ranges:
        with open(out_txt, "w", encoding="utf-8") as handle:
            handle.write("")
        return

    df_ranges = pd.DataFrame(selected_ranges)
    if months:
        last_start, last_end = months[-1]
        if effective_end < last_end and len(months) >= 2:
            prev_start, _prev_end = months[-2]
            df_ranges.loc[
                df_ranges["start_date"] == last_start.strftime("%Y-%m-%d"),
                "start_date",
            ] = prev_start.strftime("%Y-%m-%d")

    merged = _merge_contiguous_ranges(df_ranges)
    with open(out_txt, "w", encoding="utf-8") as handle:
        for _, row in merged.iterrows():
            handle.write(f"{row['symbol']}\t{row['start_date']}\t{row['end_date']}\n")
    print(f"Saved to {out_txt}")


if __name__ == "__main__":
    filter_top_liquidity()
