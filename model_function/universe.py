from __future__ import annotations

from dataclasses import dataclass
from hashlib import blake2b
from typing import Iterable, Sequence

import pandas as pd

from utils.preprocess import compute_liquidity, exclude_symbols


@dataclass(frozen=True)
class TrainingUniverseConfig:
    liquidity_lookback_days: int = 60
    liquidity_min_periods: int = 20
    entry_limit: int = 1800
    exit_limit: int = 2200
    sample_size: int = 800
    segment_count: int = 10
    min_per_segment: int = 20
    seed: int = 42


@dataclass(frozen=True)
class PredictionUniverseConfig:
    liquidity_lookback_days: int = 60
    entry_limit: int = 1000
    exit_limit: int = 1200


@dataclass(frozen=True)
class HoldingBufferConfig:
    buy_rank: int = 300
    hold_rank: int = 500


TRAINING_UNIVERSE_DEFAULTS = TrainingUniverseConfig()
PREDICTION_UNIVERSE_DEFAULTS = PredictionUniverseConfig()
HOLDING_BUFFER_DEFAULTS = HoldingBufferConfig()


def _normalize_limits(entry_limit: int, exit_limit: int) -> tuple[int, int]:
    entry = max(0, int(entry_limit))
    exit_ = max(entry, int(exit_limit))
    return entry, exit_


def _stable_symbol_order(symbol: str, *, salt: str, seed: int) -> str:
    payload = f"{salt}|{seed}|{symbol}".encode("utf-8")
    return blake2b(payload, digest_size=16).hexdigest()


def _segment_slices(ranked_symbols: Sequence[str], segment_count: int) -> list[list[str]]:
    ranked = [str(sym) for sym in ranked_symbols if str(sym)]
    if not ranked:
        return []

    segment_count = max(1, int(segment_count))
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
    if segment_count <= 0 or total_select <= 0:
        return []

    min_per_segment = max(1, int(min_per_segment))
    if min_per_segment * segment_count > total_select:
        min_per_segment = max(1, total_select // segment_count)

    base = [min_per_segment] * segment_count
    remain = max(0, int(total_select) - sum(base))
    if remain <= 0:
        return base

    weights = [segment_count - idx for idx in range(segment_count)]
    weight_sum = sum(weights) or 1
    raw_extra = [remain * weight / weight_sum for weight in weights]
    extra = [int(value) for value in raw_extra]
    left = remain - sum(extra)

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
    entry, exit_ = _normalize_limits(entry_limit, exit_limit)
    ranked = [str(sym) for sym in ranked_symbols if str(sym)]
    if not ranked or exit_ <= 0:
        return []

    previous_set = {str(sym) for sym in (previous_symbols or []) if str(sym)}
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
    ranked = [str(sym) for sym in ranked_symbols if str(sym)]
    total_select = max(0, int(total_select))
    if not ranked or total_select <= 0:
        return []
    if len(ranked) <= total_select:
        return ranked

    segments = _segment_slices(ranked, max(1, int(segment_count)))
    if not segments:
        return ranked[:total_select]

    targets = _weighted_segment_targets(len(segments), total_select, min_per_segment)
    selected: list[str] = []
    for idx, segment in enumerate(segments):
        target = min(len(segment), targets[idx]) if idx < len(targets) else 0
        if target <= 0:
            continue
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
        already_selected = set(selected)
        selected.extend([symbol for symbol in ranked if symbol not in already_selected][: total_select - len(selected)])

    return list(dict.fromkeys(selected))[:total_select]


def _normalize_symbol_daily_rows(rows: list[dict], start_year: int, end_year: int) -> pd.DataFrame:
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

    turnover = compute_liquidity(frame, amount_col="amount", close_col="close", volume_col="volume")
    if "tradestatus" in frame.columns:
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
    frame = _normalize_symbol_daily_rows(rows, start_year, end_year)
    if frame.empty:
        return []

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

    st_months = set(frame.loc[frame["isST"] != 0, "month"].tolist())
    records: list[dict] = []
    for source_month in source_months:
        if source_month in st_months:
            continue
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


def select_training_symbols(
    month_frame: pd.DataFrame,
    previous_symbols: Iterable[str] | None,
    *,
    source_month: pd.Period | str,
    excluded_symbols: set[str] | None = None,
    config: TrainingUniverseConfig = TRAINING_UNIVERSE_DEFAULTS,
) -> list[str]:
    if month_frame.empty or "symbol" not in month_frame.columns:
        return []

    ranked = (
        month_frame.sort_values("lagged_liquidity", ascending=False)["symbol"]
        .astype(str)
        .tolist()
    )
    ranked = exclude_symbols(ranked, excluded_symbols or set())
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


def build_prediction_pool_from_features(
    features: pd.DataFrame,
    previous_holdings: Iterable[str] | None,
    *,
    excluded_symbols: set[str] | None = None,
    config: PredictionUniverseConfig = PREDICTION_UNIVERSE_DEFAULTS,
) -> list[str]:
    if features.empty:
        return []

    liquidity = compute_liquidity(features, amount_col="$amount", close_col="$close", volume_col="$volume")
    ranked = (
        pd.DataFrame({"liq": liquidity})
        .groupby(level="instrument")["liq"]
        .median()
        .sort_values(ascending=False)
        .index
        .astype(str)
        .tolist()
    )

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
    if picks_df.empty:
        return pd.DataFrame(columns=list(picks_df.columns))
    if "instrument" not in picks_df.columns or "Score" not in picks_df.columns:
        raise ValueError("picks_df must contain columns: instrument, Score")

    buy_rank, hold_rank = _normalize_limits(buy_rank, hold_rank)
    if buy_rank <= 0 or hold_rank <= 0:
        return picks_df.iloc[0:0].copy()

    current_set = {str(symbol) for symbol in (current_instruments or []) if str(symbol)}
    ranked = picks_df.copy().sort_values("Score", ascending=False)
    ranked["instrument"] = ranked["instrument"].astype(str)
    ranked = ranked.drop_duplicates(subset=["instrument"], keep="first").reset_index(drop=True)
    ranked["rank"] = ranked.index + 1

    allowed = ranked["rank"] <= buy_rank
    if current_set:
        allowed = allowed | (
            ranked["instrument"].isin(current_set)
            & (ranked["rank"] <= hold_rank)
        )

    return ranked.loc[allowed].drop(columns=["rank"]).reset_index(drop=True)
