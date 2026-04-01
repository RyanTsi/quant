from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

# Keep training/prediction label expression aligned from one source.
ALPHA158_WEIGHTED_5D_LABEL = (
    "(5 * (Ref($close, -2)/Ref($close, -1) - 1) + 4 * (Ref($close, -3)/Ref($close, -1) - 1) + "
    "3 * (Ref($close, -4)/Ref($close, -1) - 1) + 2 * (Ref($close, -5)/Ref($close, -1) - 1) + "
    "1 * (Ref($close, -6)/Ref($close, -1) - 1)) / 15"
)


def compute_liquidity(
    df: pd.DataFrame,
    *,
    amount_col: str = "amount",
    close_col: str = "close",
    volume_col: str = "volume",
) -> pd.Series:
    """Compute liquidity by amount; fallback to close*volume."""
    if amount_col in df.columns:
        amount = pd.to_numeric(df[amount_col], errors="coerce")
        if not amount.isna().all():
            return amount.fillna(0)

    if close_col in df.columns:
        close = pd.to_numeric(df[close_col], errors="coerce").fillna(0)
    else:
        close = pd.Series(0.0, index=df.index, dtype=float)

    if volume_col in df.columns:
        volume = pd.to_numeric(df[volume_col], errors="coerce").fillna(0)
    else:
        volume = pd.Series(0.0, index=df.index, dtype=float)

    return (close * volume).fillna(0)


def sample_ranked_symbols(
    ranked_symbols: Sequence[str],
    *,
    segment_count: int,
    sample_per_segment: int,
    segment_size: int | None = None,
    trim_ratio: float | None = None,
    trim_count: int = 0,
    min_segment_size: int = 1,
    rng: random.Random | None = None,
) -> list[str]:
    """
    Segment ranked symbols and sample from each segment's middle part.

    - If ``trim_ratio`` is set, each segment trims ``max(1, int(len(seg)*trim_ratio))``
      symbols from both head and tail.
    - Else ``trim_count`` is used on both sides.
    """
    if segment_count <= 0 or sample_per_segment <= 0:
        return []

    ranked = [str(s) for s in ranked_symbols if str(s)]
    if not ranked:
        return []

    rng = rng or random.Random(42)
    segment_size = max(1, int(segment_size)) if segment_size is not None else max(1, len(ranked) // segment_count)
    selected: list[str] = []

    for i in range(segment_count):
        start = i * segment_size
        end = (i + 1) * segment_size if i < segment_count - 1 else len(ranked)
        seg = ranked[start:end]
        if len(seg) < min_segment_size:
            continue

        if trim_ratio is not None:
            cut = max(1, int(len(seg) * trim_ratio))
        else:
            cut = max(0, trim_count)

        if len(seg) <= cut * 2:
            continue

        middle = seg[cut : len(seg) - cut]
        if not middle:
            continue

        k = min(sample_per_segment, len(middle))
        selected.extend(rng.sample(middle, k))

    return list(dict.fromkeys(selected))


def read_symbol_list(path: str | Path) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    return {line.strip().upper() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}


def exclude_symbols(symbols: Iterable[str], excluded_symbols: set[str]) -> list[str]:
    if not excluded_symbols:
        return [str(s) for s in symbols if str(s)]
    excluded_upper = {s.upper() for s in excluded_symbols}
    out: list[str] = []
    for sym in symbols:
        s = str(sym)
        if not s:
            continue
        if s.upper() in excluded_upper:
            continue
        out.append(s)
    return out
