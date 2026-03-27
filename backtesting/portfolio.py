from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd


@dataclass(frozen=True)
class PortfolioConfig:
    top_k: int = 80
    max_weight: float = 0.02
    rebalance_threshold: float = 0.002


def build_target_weights(picks_df: pd.DataFrame, cfg: PortfolioConfig) -> pd.DataFrame:
    if picks_df.empty:
        return pd.DataFrame(columns=["instrument", "target_weight", "score"])
    df = picks_df.copy()
    if "instrument" not in df.columns or "Score" not in df.columns:
        raise ValueError("picks_df must contain columns: instrument, Score")

    top = df.sort_values("Score", ascending=False).head(cfg.top_k).copy()
    if top.empty:
        return pd.DataFrame(columns=["instrument", "target_weight", "score"])

    # Stable positive weights from ranking score. Shift to avoid negative totals.
    min_score = float(top["Score"].min())
    shifted = top["Score"] - min_score + 1e-6
    raw = shifted / shifted.sum()
    top["target_weight"] = raw

    # Apply per-stock cap, then re-normalize.
    top["target_weight"] = top["target_weight"].clip(upper=cfg.max_weight)
    weight_sum = float(top["target_weight"].sum())
    if weight_sum <= 0:
        top["target_weight"] = 1.0 / len(top)
    else:
        top["target_weight"] = top["target_weight"] / weight_sum

    out = top.rename(columns={"Score": "score"})[["instrument", "target_weight", "score"]]
    out["target_weight"] = out["target_weight"].astype(float)
    return out


def build_rebalance_orders(
    target_df: pd.DataFrame,
    current_df: pd.DataFrame | None,
    threshold: float,
) -> pd.DataFrame:
    tgt_map: Dict[str, float] = dict(zip(target_df["instrument"], target_df["target_weight"]))
    cur_map: Dict[str, float] = {}
    if current_df is not None and not current_df.empty:
        if "instrument" not in current_df.columns or "target_weight" not in current_df.columns:
            raise ValueError("current_df must contain columns: instrument, target_weight")
        cur_map = dict(zip(current_df["instrument"], current_df["target_weight"]))

    all_symbols = sorted(set(tgt_map) | set(cur_map))
    rows = []
    for sym in all_symbols:
        current_w = float(cur_map.get(sym, 0.0))
        target_w = float(tgt_map.get(sym, 0.0))
        delta = target_w - current_w
        action = "HOLD"
        if abs(delta) >= threshold:
            action = "BUY" if delta > 0 else "SELL"
        rows.append(
            {
                "instrument": sym,
                "current_weight": current_w,
                "target_weight": target_w,
                "delta_weight": delta,
                "action": action,
            }
        )
    return pd.DataFrame(rows)


def summarize_orders(orders_df: pd.DataFrame) -> Dict[str, float]:
    if orders_df.empty:
        return {"buy_count": 0, "sell_count": 0, "hold_count": 0, "turnover": 0.0}
    buy_count = int((orders_df["action"] == "BUY").sum())
    sell_count = int((orders_df["action"] == "SELL").sum())
    hold_count = int((orders_df["action"] == "HOLD").sum())
    turnover = float(orders_df["delta_weight"].abs().sum())
    return {
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
        "turnover": turnover,
    }

