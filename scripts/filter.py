import os
import random
from datetime import timedelta
import math

import pandas as pd

import utils.io
from config.settings import settings
from data_pipeline.database import DBClient
from utils.preprocess import compute_liquidity


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
    return [(m.start_time.normalize(), m.end_time.normalize()) for m in months]


def _target_source_month_pairs(
    months: list[tuple[pd.Timestamp, pd.Timestamp]]
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
        source_month = pd.Period(source_start, freq="M")
        pairs.append((source_month, target_start, target_end))
    return pairs


def _normalize_symbol_daily_rows(rows: list[dict], start_year: int, end_year: int) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "date" not in df.columns:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    if getattr(df["date"].dt, "tz", None) is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df = df[(df["date"] >= f"{start_year}-01-01") & (df["date"] <= f"{end_year}-12-31")]
    if df.empty:
        return pd.DataFrame()

    turnover = compute_liquidity(df, amount_col="amount", close_col="close", volume_col="volume")
    if "tradestatus" in df.columns:
        tradestatus = pd.to_numeric(df["tradestatus"], errors="coerce").fillna(1)
        turnover = turnover.where(tradestatus != 0, 0)

    close = pd.to_numeric(df.get("close"), errors="coerce")
    ret = close.pct_change()

    out = pd.DataFrame(
        {
            "date": df["date"],
            "turnover": pd.to_numeric(turnover, errors="coerce").fillna(0.0),
            "ret": pd.to_numeric(ret, errors="coerce"),
            "month": df["date"].dt.to_period("M"),
        }
    )
    if "isST" in df.columns:
        out["isST"] = pd.to_numeric(df["isST"], errors="coerce").fillna(0.0)
    else:
        out["isST"] = 0.0
    return out


def _quarter_lookback_window(source_month: pd.Period) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = source_month.end_time.normalize()
    start = (source_month.start_time - pd.DateOffset(months=2)).normalize()
    return start, end


def _collect_symbol_month_metrics(
    symbol: str,
    rows: list[dict],
    source_months: list[pd.Period],
    start_year: int,
    end_year: int,
) -> list[dict]:
    df = _normalize_symbol_daily_rows(rows, start_year, end_year)
    if df.empty:
        return []

    st_months = set(df.loc[df["isST"] != 0, "month"].tolist())
    records: list[dict] = []

    for source_m in source_months:
        # Exclude ST sources directly from training membership generation.
        if source_m in st_months:
            continue

        win_start, win_end = _quarter_lookback_window(source_m)
        win_df = df[(df["date"] >= win_start) & (df["date"] <= win_end)]
        if win_df.empty:
            continue

        liq = pd.to_numeric(win_df["turnover"], errors="coerce").dropna()
        if liq.empty:
            continue
        liq_mean = float(liq.mean())
        liq_std = float(liq.std(ddof=0))
        # Liquidity stability over the past quarter (higher is more stable + liquid).
        stability = liq_mean / (liq_std + 1e-12)

        vol = float(pd.to_numeric(win_df["ret"], errors="coerce").dropna().std(ddof=0))
        if math.isnan(vol):
            vol = 0.0

        records.append(
            {
                "symbol": symbol,
                "source_month": source_m,
                "mean_turnover": liq_mean,
                "stability": float(stability),
                "volatility": vol,
            }
        )

    return records


def _exclude_volatility_extremes(df_m: pd.DataFrame, lower_q: float = 0.05, upper_q: float = 0.95) -> pd.DataFrame:
    if df_m.empty or "volatility" not in df_m.columns:
        return df_m
    vol = pd.to_numeric(df_m["volatility"], errors="coerce")
    valid = vol.notna()
    if int(valid.sum()) < 3:
        return df_m.loc[valid].copy()
    low = float(vol[valid].quantile(lower_q))
    high = float(vol[valid].quantile(upper_q))
    keep = valid & vol.between(low, high, inclusive="both")
    return df_m.loc[keep].copy()


def _weighted_segment_targets(
    segment_count: int,
    total_select: int,
    min_per_segment: int,
) -> list[int]:
    if segment_count <= 0 or total_select <= 0:
        return []
    min_per_segment = max(1, int(min_per_segment))
    if min_per_segment * segment_count > total_select:
        min_per_segment = max(1, total_select // segment_count)
    base = [min_per_segment] * segment_count
    remain = max(0, total_select - sum(base))

    weights = [segment_count - i for i in range(segment_count)]
    w_sum = sum(weights) or 1
    raw_extra = [remain * w / w_sum for w in weights]
    extra = [int(x) for x in raw_extra]
    left = remain - sum(extra)
    # Largest-remainder with higher-rank priority on ties.
    frac_order = sorted(
        range(segment_count),
        key=lambda i: (raw_extra[i] - extra[i], weights[i]),
        reverse=True,
    )
    for i in frac_order:
        if left <= 0:
            break
        extra[i] += 1
        left -= 1

    return [base[i] + extra[i] for i in range(segment_count)]


def _weighted_group_pick(
    ranked_symbols: list[str],
    *,
    segment_count: int,
    total_select: int,
    min_per_segment: int,
) -> list[str]:
    ranked = [str(s) for s in ranked_symbols if str(s)]
    if not ranked or segment_count <= 0:
        return []

    segments: list[list[str]] = []
    n = len(ranked)
    for i in range(segment_count):
        start = int(i * n / segment_count)
        end = int((i + 1) * n / segment_count)
        segments.append(ranked[start:end])

    targets = _weighted_segment_targets(segment_count, total_select, min_per_segment)
    picked = [0] * segment_count
    capacities = [len(seg) for seg in segments]

    shortfall = 0
    for i in range(segment_count):
        take = min(targets[i], capacities[i])
        picked[i] = take
        shortfall += targets[i] - take

    # Re-distribute unmet quota to higher-ranked groups first.
    if shortfall > 0:
        for i in range(segment_count):
            if shortfall <= 0:
                break
            avail = capacities[i] - picked[i]
            if avail <= 0:
                continue
            add = min(avail, shortfall)
            picked[i] += add
            shortfall -= add

    out: list[str] = []
    for i, seg in enumerate(segments):
        out.extend(seg[: picked[i]])
    return list(dict.fromkeys(out))


def _sample_symbols_for_month(
    df_m: pd.DataFrame,
    rng: random.Random,
    top_n: int,
    segment_count: int,
    *,
    total_select: int = 800,
    min_per_segment: int = 20,
) -> list[str]:
    del rng  # deterministic allocation by rank for reproducibility
    if df_m.empty:
        return []
    top = df_m.sort_values(["stability", "mean_turnover"], ascending=[False, False]).head(top_n)
    ranked = pd.Series(top["symbol"]).reset_index(drop=True).astype(str).tolist()
    return _weighted_group_pick(
        ranked,
        segment_count=max(1, segment_count),
        total_select=max(1, int(total_select)),
        min_per_segment=max(1, int(min_per_segment)),
    )


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
    months = _month_ranges(start_year, effective_end)
    month_pairs = _target_source_month_pairs(months)
    source_months = [p[0] for p in month_pairs]
    rng = random.Random(random_seed)

    monthly_metrics = []
    query_start = f"{start_year}-01-01"
    query_end = effective_end.strftime("%Y-%m-%d")
    for stock_code in all_stock_list:
        response = db_client.query_data(stock_code, query_start, query_end)
        if response is None or response.status_code != 200:
            continue
        rows = (response.json() or {}).get("data", [])
        monthly_metrics.extend(
            _collect_symbol_month_metrics(
                symbol=stock_code,
                rows=rows,
                source_months=source_months,
                start_year=start_year,
                end_year=end_year,
            )
        )

    if not monthly_metrics:
        out_txt = _output_instruments_path()
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("")
        return

    df = pd.DataFrame(monthly_metrics)
    selected_ranges = []
    for source_m, m_start, m_end in month_pairs:
        df_m = df[df["source_month"] == source_m]
        if df_m.empty:
            continue
        df_m = _exclude_volatility_extremes(df_m, lower_q=0.05, upper_q=0.95)
        symbols = _sample_symbols_for_month(df_m, rng, top_n=top_n, segment_count=10)
        for sym in symbols:
            selected_ranges.append({"symbol": sym, "start_date": m_start.strftime("%Y-%m-%d"), "end_date": m_end.strftime("%Y-%m-%d")})

    if not selected_ranges:
        out_txt = _output_instruments_path()
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("")
        return

    df_ranges = pd.DataFrame(selected_ranges)

    # If the final period is a partial month, merge it with previous month.
    if months:
        last_start, last_end = months[-1]
        if effective_end < last_end and len(months) >= 2:
            prev_start, _prev_end = months[-2]
            last_start_str = last_start.strftime("%Y-%m-%d")
            prev_start_str = prev_start.strftime("%Y-%m-%d")
            df_ranges.loc[df_ranges["start_date"] == last_start_str, "start_date"] = prev_start_str

    merged = _merge_contiguous_ranges(df_ranges)

    out_txt = _output_instruments_path()
    with open(out_txt, "w", encoding="utf-8") as f:
        for _, row in merged.iterrows():
            f.write(f"{row['symbol']}\t{row['start_date']}\t{row['end_date']}\n")
    print(f"Saved to {out_txt}")


if __name__ == "__main__":
    filter_top_liquidity()
