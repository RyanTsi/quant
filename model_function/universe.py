from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd

from utils.preprocess import compute_liquidity, exclude_symbols


@dataclass(frozen=True)
class TrainingUniverseConfig:
    """Configuration for month-lagged training-universe construction."""

    # Trailing trading-day window used to compute the lagged-liquidity score.
    liquidity_lookback_days: int = 60
    # Minimum valid observations required before the rolling median is trusted.
    liquidity_min_periods: int = 20
    # New symbols may only enter the training universe from this top-liquidity band.
    entry_limit: int = 1800
    # Existing members may remain while they stay inside this wider band.
    exit_limit: int = 2200
    # Final compute cap after buffering and deterministic downsampling.
    sample_size: int = 1200
    # Number of rank buckets used for reproducible segment-based sampling.
    segment_count: int = 10
    # Minimum quota reserved for each segment before weighted redistribution.
    min_per_segment: int = 20
    # Stable seed used in hashed sampling order for reproducibility.
    seed: int = 42


@dataclass(frozen=True)
class PredictionUniverseConfig:
    """Configuration for daily prediction-universe construction."""

    # Trailing trading-day window used to rank symbols by lagged liquidity.
    liquidity_lookback_days: int = 60
    # Fresh names must rank inside this top-liquidity band.
    entry_limit: int = 1000
    # Existing holdings may remain while still inside this wider band.
    exit_limit: int = 1200


@dataclass(frozen=True)
class HoldingBufferConfig:
    """Configuration for portfolio buy/hold rank buffers."""

    # New buys must rank inside this model-score band.
    buy_rank: int = 300
    # Existing holdings may remain while still inside this wider score band.
    hold_rank: int = 500


# Canonical defaults reused by training, prediction, and portfolio paths.
TRAINING_UNIVERSE_DEFAULTS = TrainingUniverseConfig()
PREDICTION_UNIVERSE_DEFAULTS = PredictionUniverseConfig()
HOLDING_BUFFER_DEFAULTS = HoldingBufferConfig()
TRAINING_UNIVERSE_ARTIFACT_NAME = "my_800_stocks.txt"


def _normalize_limits(entry_limit: int, exit_limit: int) -> tuple[int, int]:
    """Clamp limits to non-negative integers and ensure exit >= entry."""

    entry = max(0, int(entry_limit))
    exit_ = max(entry, int(exit_limit))
    return entry, exit_


def _stable_symbol_order(symbol: str, *, salt: str, seed: int) -> str:
    """Return a stable hash key for deterministic per-symbol ordering."""

    payload = f"{salt}|{seed}|{symbol}".encode("utf-8")
    return blake2b(payload, digest_size=16).hexdigest()


def _segment_slices(ranked_symbols: Sequence[str], segment_count: int) -> list[list[str]]:
    """Split a ranked symbol list into contiguous rank segments."""

    # Normalize symbols once so downstream helpers never deal with empty/non-string items.
    ranked = [str(sym) for sym in ranked_symbols if str(sym)]
    if not ranked:
        return []

    segment_count = max(1, int(segment_count))
    # ``total`` is the full ranked list length; each segment keeps the original rank order.
    total = len(ranked)
    segments: list[list[str]] = []
    for idx in range(segment_count):
        start = int(idx * total / segment_count)
        end = int((idx + 1) * total / segment_count)
        segment = ranked[start:end]
        if segment:
            segments.append(segment)
    return segments


def _weighted_segment_targets(segment_count: int, total_select: int, min_per_segment: int) -> list[int]:
    """Allocate a deterministic per-segment quota with top-heavy weighting."""

    if segment_count <= 0 or total_select <= 0:
        return []

    min_per_segment = max(1, int(min_per_segment))
    if min_per_segment * segment_count > total_select:
        min_per_segment = max(1, total_select // segment_count)

    # ``base`` guarantees each segment a floor allocation before weighted extras are applied.
    base = [min_per_segment] * segment_count
    # ``remain`` is the quota still available for top-heavy redistribution.
    remain = max(0, int(total_select) - sum(base))
    if remain <= 0:
        return base

    # Earlier segments get larger weights because they correspond to higher-liquidity ranks.
    weights = [segment_count - idx for idx in range(segment_count)]
    weight_sum = sum(weights) or 1
    raw_extra = [remain * weight / weight_sum for weight in weights]
    extra = [int(value) for value in raw_extra]
    left = remain - sum(extra)

    # Largest-remainder tie-breaking keeps any leftover quota biased toward higher ranks.
    frac_order = sorted(
        range(segment_count),
        key=lambda idx: (raw_extra[idx] - extra[idx], weights[idx]),
        reverse=True,
    )
    for idx in frac_order:
        if left <= 0:
            break
        extra[idx] += 1
        left -= 1

    return [base[idx] + extra[idx] for idx in range(segment_count)]


def apply_entry_exit_buffer(
    ranked_symbols: Sequence[str],
    previous_symbols: Iterable[str] | None,
    *,
    entry_limit: int,
    exit_limit: int,
) -> list[str]:
    """Apply entry/exit buffering to a ranked list.

    Entry semantics:
    - symbols ranked within ``entry_limit`` are always kept
    - symbols ranked outside ``entry_limit`` may remain only if they were already present
    - no symbol ranked outside ``exit_limit`` can survive
    """

    entry, exit_ = _normalize_limits(entry_limit, exit_limit)
    ranked = [str(sym) for sym in ranked_symbols if str(sym)]
    if not ranked or exit_ <= 0:
        return []

    # ``previous_set`` models the incumbents that are eligible for the wider exit band.
    previous_set = {str(sym) for sym in (previous_symbols or []) if str(sym)}
    # ``buffered`` preserves the original ranking order after applying the buffer rules.
    buffered: list[str] = []
    for idx, symbol in enumerate(ranked[:exit_], start=1):
        if idx <= entry or symbol in previous_set:
            buffered.append(symbol)
    return buffered


def deterministic_segment_sample(
    ranked_symbols: Sequence[str],
    *,
    total_select: int,
    segment_count: int,
    min_per_segment: int,
    salt: str,
    seed: int,
) -> list[str]:
    """Downsample a ranked universe reproducibly while preserving rank diversity.

    The sampling order inside each segment is driven by a stable hash derived from:
    - the symbol itself
    - a caller-provided ``salt`` such as the source month
    - the configured seed
    """

    ranked = [str(sym) for sym in ranked_symbols if str(sym)]
    total_select = max(0, int(total_select))
    if not ranked or total_select <= 0:
        return []
    if len(ranked) <= total_select:
        return ranked

    # Segments preserve rank locality so sampling does not collapse onto only the top names.
    segments = _segment_slices(ranked, max(1, int(segment_count)))
    if not segments:
        return ranked[:total_select]

    # ``targets`` is the final per-segment quota after top-heavy redistribution.
    targets = _weighted_segment_targets(len(segments), total_select, min_per_segment)
    # ``selected`` accumulates chosen symbols while keeping each segment's original rank order.
    selected: list[str] = []
    for idx, segment in enumerate(segments):
        target = min(len(segment), targets[idx]) if idx < len(targets) else 0
        if target <= 0:
            continue
        # Order symbols deterministically inside the segment before taking the quota.
        ordered = sorted(
            (
                (_stable_symbol_order(symbol, salt=f"{salt}:{idx}", seed=seed), pos, symbol)
                for pos, symbol in enumerate(segment)
            ),
            key=lambda item: (item[0], item[1]),
        )
        chosen = {symbol for _, _, symbol in ordered[:target]}
        selected.extend([symbol for symbol in segment if symbol in chosen])

    if len(selected) < total_select:
        # If any segment under-fills, backfill from the remaining ranked list in rank order.
        already_selected = set(selected)
        selected.extend([symbol for symbol in ranked if symbol not in already_selected][: total_select - len(selected)])

    return list(dict.fromkeys(selected))[:total_select]


def _normalize_symbol_daily_rows(rows: list[dict], start_year: int, end_year: int) -> pd.DataFrame:
    """Normalize raw daily DB rows into the minimal fields needed by universe builders."""

    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    if "date" not in frame.columns:
        return pd.DataFrame()

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    if getattr(frame["date"].dt, "tz", None) is not None:
        frame["date"] = frame["date"].dt.tz_localize(None)

    start = pd.Timestamp(f"{start_year}-01-01")
    end = pd.Timestamp(f"{end_year}-12-31")
    frame = frame[(frame["date"] >= start) & (frame["date"] <= end)]
    if frame.empty:
        return pd.DataFrame()

    # ``turnover`` is the liquidity proxy used throughout the universe contract.
    turnover = compute_liquidity(frame, amount_col="amount", close_col="close", volume_col="volume")
    if "tradestatus" in frame.columns:
        # Suspended days are forced to zero turnover so they do not inflate liquidity ranks.
        tradestatus = pd.to_numeric(frame["tradestatus"], errors="coerce").fillna(1)
        turnover = turnover.where(tradestatus != 0, 0.0)

    if "isST" in frame.columns:
        is_st = pd.to_numeric(frame["isST"], errors="coerce").fillna(0.0)
    else:
        is_st = pd.Series(0.0, index=frame.index, dtype=float)

    normalized = pd.DataFrame(
        {
            "date": frame["date"],
            "turnover": pd.to_numeric(turnover, errors="coerce").fillna(0.0),
            "month": frame["date"].dt.to_period("M"),
            "isST": is_st,
        }
    )
    return normalized


def collect_training_month_liquidity(
    *,
    symbol: str,
    rows: list[dict],
    source_months: Sequence[pd.Period],
    start_year: int,
    end_year: int,
    config: TrainingUniverseConfig = TRAINING_UNIVERSE_DEFAULTS,
) -> list[dict]:
    """Collect one lagged-liquidity record per symbol/source month for training.

    Output schema:
    - ``symbol``: normalized symbol string
    - ``source_month``: month whose data is used to decide next month's membership
    - ``lagged_liquidity``: trailing median turnover measured at the end of that month
    """

    frame = _normalize_symbol_daily_rows(rows, start_year, end_year)
    if frame.empty:
        return []

    # ``min_periods`` prevents tiny partial windows from being treated as stable liquidity history.
    min_periods = max(1, min(int(config.liquidity_min_periods), int(config.liquidity_lookback_days)))
    # Lagged liquidity score:
    # score(symbol, month_t) = median(turnover) over the trailing lookback window
    # ending on the last available trading day inside month_t.
    frame["lagged_liquidity"] = (
        pd.to_numeric(frame["turnover"], errors="coerce")
        .fillna(0.0)
        .rolling(window=int(config.liquidity_lookback_days), min_periods=min_periods)
        .median()
    )

    # Any month with ST status is excluded from serving as a source month.
    st_months = set(frame.loc[frame["isST"] != 0, "month"].tolist())
    records: list[dict] = []
    for source_month in source_months:
        if source_month in st_months:
            continue
        # ``month_frame`` isolates the source month whose end-of-month liquidity will be recorded.
        month_frame = frame[frame["month"] == source_month]
        if month_frame.empty:
            continue
        liquidity = pd.to_numeric(month_frame["lagged_liquidity"], errors="coerce").dropna()
        if liquidity.empty:
            continue
        records.append(
            {
                "symbol": str(symbol),
                "source_month": source_month,
                "lagged_liquidity": float(liquidity.iloc[-1]),
            }
    )
    return records


def resolve_training_universe_output_path(
    *,
    data_path: str | Path,
    qlib_data_path: str | Path | None = None,
) -> Path:
    """Resolve the canonical instrument-file path for the training universe."""

    output_dir = Path(qlib_data_path) / "instruments" if qlib_data_path else Path(data_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / TRAINING_UNIVERSE_ARTIFACT_NAME


def build_training_source_month_pairs(
    months: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> list[tuple[pd.Period, pd.Timestamp, pd.Timestamp]]:
    """Build one-month-lagged (source_month, target_start, target_end) tuples."""

    pairs: list[tuple[pd.Period, pd.Timestamp, pd.Timestamp]] = []
    if len(months) <= 1:
        return pairs

    for idx in range(1, len(months)):
        target_start, target_end = months[idx]
        source_start, _source_end = months[idx - 1]
        pairs.append((pd.Period(source_start, freq="M"), target_start, target_end))
    return pairs


def merge_contiguous_symbol_ranges(df_ranges: pd.DataFrame) -> pd.DataFrame:
    """Merge adjacent monthly membership rows for the same symbol."""

    if df_ranges.empty:
        return df_ranges

    merged_rows = []
    for symbol, group in df_ranges.sort_values(["symbol", "start_date"]).groupby("symbol"):
        start = pd.Timestamp(group.iloc[0]["start_date"])
        end = pd.Timestamp(group.iloc[0]["end_date"])
        for _, row in group.iloc[1:].iterrows():
            current_start = pd.Timestamp(row["start_date"])
            current_end = pd.Timestamp(row["end_date"])
            if current_start <= end + pd.Timedelta(days=1):
                end = max(end, current_end)
                continue
            merged_rows.append(
                {
                    "symbol": symbol,
                    "start_date": start.strftime("%Y-%m-%d"),
                    "end_date": end.strftime("%Y-%m-%d"),
                }
            )
            start, end = current_start, current_end
        merged_rows.append(
            {
                "symbol": symbol,
                "start_date": start.strftime("%Y-%m-%d"),
                "end_date": end.strftime("%Y-%m-%d"),
            }
        )
    return pd.DataFrame(merged_rows)


def _build_month_ranges(
    start_year: int,
    effective_end: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Build normalized month windows from January of ``start_year`` to ``effective_end``."""

    start = pd.Timestamp(f"{int(start_year)}-01-01")
    months = pd.period_range(start=start, end=effective_end, freq="M")
    return [(month.start_time.normalize(), month.end_time.normalize()) for month in months]


def _write_training_universe_rows(df_ranges: pd.DataFrame, output_path: Path) -> None:
    """Persist the training-universe instrument file in Qlib txt format."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        if df_ranges.empty:
            handle.write("")
            return
        for _, row in df_ranges.iterrows():
            handle.write(f"{row['symbol']}\t{row['start_date']}\t{row['end_date']}\n")


def select_training_symbols(
    month_frame: pd.DataFrame,
    previous_symbols: Iterable[str] | None,
    *,
    source_month: pd.Period | str,
    top_n: int | None = None,
    excluded_symbols: set[str] | None = None,
    config: TrainingUniverseConfig = TRAINING_UNIVERSE_DEFAULTS,
) -> list[str]:
    """Select the final training symbols for one source month.

    Process:
    1. rank by lagged liquidity
    2. remove excluded symbols such as index constituents
    3. apply entry/exit buffering against prior members
    4. deterministically downsample to the compute cap
    """

    if month_frame.empty or "symbol" not in month_frame.columns:
        return []

    # ``ranked`` is the full month-level liquidity ranking before exclusions and buffering.
    ranked_frame = month_frame.sort_values("lagged_liquidity", ascending=False)
    if top_n is not None and int(top_n) > 0:
        ranked_frame = ranked_frame.head(int(top_n))

    ranked = ranked_frame["symbol"].astype(str).tolist()
    ranked = exclude_symbols(ranked, excluded_symbols or set())
    # ``candidates`` is the buffered universe before the final compute cap is applied.
    candidates = apply_entry_exit_buffer(
        ranked,
        previous_symbols,
        entry_limit=config.entry_limit,
        exit_limit=config.exit_limit,
    )
    if not candidates:
        return []
    return deterministic_segment_sample(
        candidates,
        total_select=config.sample_size,
        segment_count=config.segment_count,
        min_per_segment=config.min_per_segment,
        salt=f"training:{source_month}",
        seed=config.seed,
    )


def build_training_universe_artifact(
    *,
    symbols: Sequence[str],
    fetch_symbol_rows: Callable[[str], Sequence[dict] | None],
    output_path: str | Path,
    start_year: int,
    end_year: int,
    top_n: int = TRAINING_UNIVERSE_DEFAULTS.exit_limit,
    random_seed: int = TRAINING_UNIVERSE_DEFAULTS.seed,
    excluded_symbols: set[str] | None = None,
    effective_end: pd.Timestamp | None = None,
) -> dict[str, int | str]:
    """Build and persist the month-lagged training-universe instrument file."""

    output_file = Path(output_path)
    normalized_symbols = [str(symbol) for symbol in symbols if str(symbol)]
    normalized_excluded = {str(symbol) for symbol in (excluded_symbols or set()) if str(symbol)}
    requested_end = pd.Timestamp(f"{int(end_year)}-12-31").normalize()
    resolved_end = pd.Timestamp.today().normalize() if effective_end is None else pd.Timestamp(effective_end).normalize()
    effective_end = min(resolved_end, requested_end)

    months = _build_month_ranges(start_year, effective_end)
    month_pairs = build_training_source_month_pairs(months)
    source_months = [pair[0] for pair in month_pairs]
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
    for symbol in normalized_symbols:
        rows = list(fetch_symbol_rows(symbol) or [])
        if not rows:
            continue
        monthly_metrics.extend(
            collect_training_month_liquidity(
                symbol=symbol,
                rows=rows,
                source_months=source_months,
                start_year=start_year,
                end_year=end_year,
                config=training_config,
            )
        )

    if not monthly_metrics:
        _write_training_universe_rows(pd.DataFrame(), output_file)
        return {
            "output_path": str(output_file),
            "start_year": int(start_year),
            "end_year": int(end_year),
            "top_n": int(top_n),
            "random_seed": int(random_seed),
            "effective_end": effective_end.strftime("%Y-%m-%d"),
            "source_month_count": len(source_months),
            "range_count": 0,
            "symbol_count": 0,
        }

    monthly_frame = pd.DataFrame(monthly_metrics)
    selected_ranges: list[dict[str, str]] = []
    previous_symbols: set[str] = set()
    for source_month, month_start, month_end in month_pairs:
        month_frame = monthly_frame[monthly_frame["source_month"] == source_month]
        if month_frame.empty:
            previous_symbols = set()
            continue

        symbols_for_month = select_training_symbols(
            month_frame,
            previous_symbols,
            source_month=source_month,
            top_n=top_n,
            excluded_symbols=normalized_excluded,
            config=training_config,
        )
        previous_symbols = set(symbols_for_month)
        for symbol in symbols_for_month:
            selected_ranges.append(
                {
                    "symbol": symbol,
                    "start_date": month_start.strftime("%Y-%m-%d"),
                    "end_date": month_end.strftime("%Y-%m-%d"),
                }
            )

    if not selected_ranges:
        _write_training_universe_rows(pd.DataFrame(), output_file)
        return {
            "output_path": str(output_file),
            "start_year": int(start_year),
            "end_year": int(end_year),
            "top_n": int(top_n),
            "random_seed": int(random_seed),
            "effective_end": effective_end.strftime("%Y-%m-%d"),
            "source_month_count": len(source_months),
            "range_count": 0,
            "symbol_count": 0,
        }

    df_ranges = pd.DataFrame(selected_ranges)
    if months:
        last_start, last_end = months[-1]
        if effective_end < last_end and len(months) >= 2:
            prev_start, _prev_end = months[-2]
            df_ranges.loc[
                df_ranges["start_date"] == last_start.strftime("%Y-%m-%d"),
                "start_date",
            ] = prev_start.strftime("%Y-%m-%d")

    merged = merge_contiguous_symbol_ranges(df_ranges)
    _write_training_universe_rows(merged, output_file)
    return {
        "output_path": str(output_file),
        "start_year": int(start_year),
        "end_year": int(end_year),
        "top_n": int(top_n),
        "random_seed": int(random_seed),
        "effective_end": effective_end.strftime("%Y-%m-%d"),
        "source_month_count": len(source_months),
        "range_count": int(len(merged)),
        "symbol_count": int(merged["symbol"].nunique()) if not merged.empty else 0,
    }


def build_prediction_pool_from_features(
    features: pd.DataFrame,
    previous_holdings: Iterable[str] | None,
    *,
    excluded_symbols: set[str] | None = None,
    config: PredictionUniverseConfig = PREDICTION_UNIVERSE_DEFAULTS,
) -> list[str]:
    """Build the deterministic prediction pool from per-day feature data.

    The prediction pool does not sample randomly. It keeps:
    - all symbols inside the entry liquidity band
    - previous holdings that remain inside the wider exit band
    after index/ST exclusions are applied
    """

    if features.empty:
        return []

    # ``liquidity`` is computed on the feature frame so prediction and training share the same proxy.
    liquidity = compute_liquidity(features, amount_col="$amount", close_col="$close", volume_col="$volume")
    # ``ranked`` is a per-symbol median-liquidity ordering over the recent prediction window.
    ranked = (
        pd.DataFrame({"liq": liquidity})
        .groupby(level="instrument")["liq"]
        .median()
        .sort_values(ascending=False)
        .index
        .astype(str)
        .tolist()
    )

    # ``st_symbols`` are removed even if they would otherwise rank inside the liquidity bands.
    st_symbols: set[str] = set()
    if "$isst" in features.columns:
        is_st = pd.to_numeric(features["$isst"], errors="coerce").fillna(0.0)
        if isinstance(is_st.index, pd.MultiIndex) and "instrument" in is_st.index.names:
            latest = is_st.groupby(level="instrument").last()
            st_symbols = {str(symbol) for symbol, value in latest.items() if float(value) != 0.0}

    ranked = exclude_symbols(ranked, (excluded_symbols or set()) | st_symbols)
    return apply_entry_exit_buffer(
        ranked,
        previous_holdings,
        entry_limit=config.entry_limit,
        exit_limit=config.exit_limit,
    )


def apply_portfolio_hold_buffer(
    picks_df: pd.DataFrame,
    current_instruments: Iterable[str] | None,
    *,
    buy_rank: int,
    hold_rank: int,
) -> pd.DataFrame:
    """Filter prediction picks through explicit buy/hold score bands.

    Output rows keep the same schema as ``picks_df`` but contain only symbols that are:
    - inside the buy band, or
    - already held and still inside the wider hold band
    """

    if picks_df.empty:
        return pd.DataFrame(columns=list(picks_df.columns))
    if "instrument" not in picks_df.columns or "Score" not in picks_df.columns:
        raise ValueError("picks_df must contain columns: instrument, Score")

    buy_rank, hold_rank = _normalize_limits(buy_rank, hold_rank)
    if buy_rank <= 0 or hold_rank <= 0:
        return picks_df.iloc[0:0].copy()

    # ``current_set`` is the incumbent portfolio used for hold-band retention.
    current_set = {str(symbol) for symbol in (current_instruments or []) if str(symbol)}
    # ``ranked`` is the score-ordered candidate table used to compute explicit buy/hold ranks.
    ranked = picks_df.copy().sort_values("Score", ascending=False)
    ranked["instrument"] = ranked["instrument"].astype(str)
    ranked = ranked.drop_duplicates(subset=["instrument"], keep="first").reset_index(drop=True)
    ranked["rank"] = ranked.index + 1

    # ``allowed`` is the boolean mask that encodes the buy band plus hold-band retention.
    allowed = ranked["rank"] <= buy_rank
    if current_set:
        allowed = allowed | (
            ranked["instrument"].isin(current_set)
            & (ranked["rank"] <= hold_rank)
        )

    return ranked.loc[allowed].drop(columns=["rank"]).reset_index(drop=True)
