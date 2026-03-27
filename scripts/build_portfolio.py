"""Build target portfolio weights and rebalance orders from prediction picks.

Usage:
    python -m scripts.build_portfolio
    python -m scripts.build_portfolio --date 2026-03-26
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtesting.portfolio import (
    PortfolioConfig,
    build_rebalance_orders,
    build_target_weights,
    summarize_orders,
)
from utils.run_tracker import record_run


def _resolve_date(args_date: str | None) -> str:
    if args_date:
        return pd.Timestamp(args_date).strftime("%Y-%m-%d")
    return pd.Timestamp.today().strftime("%Y-%m-%d")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build target portfolio and rebalance orders")
    parser.add_argument("--date", type=str, default=None, help="Prediction date YYYY-MM-DD")
    parser.add_argument("--top_k", type=int, default=80, help="Top-k picks for portfolio construction")
    parser.add_argument("--max_weight", type=float, default=0.02, help="Per-stock max target weight")
    parser.add_argument("--rebalance_threshold", type=float, default=0.002, help="Min abs delta weight to trade")
    args = parser.parse_args()

    date_str = _resolve_date(args.date)
    picks_path = Path("output") / f"top_picks_{date_str}.csv"
    if not picks_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {picks_path}")

    picks_df = pd.read_csv(picks_path)
    if "instrument" not in picks_df.columns:
        # Old outputs may put instrument in index column.
        first_col = picks_df.columns[0]
        if first_col.lower() in {"unnamed: 0", "symbol"}:
            picks_df = picks_df.rename(columns={first_col: "instrument"})
    if "Score" not in picks_df.columns:
        score_col = [c for c in picks_df.columns if c.lower() == "score"]
        if score_col:
            picks_df = picks_df.rename(columns={score_col[0]: "Score"})

    cfg = PortfolioConfig(
        top_k=args.top_k,
        max_weight=args.max_weight,
        rebalance_threshold=args.rebalance_threshold,
    )
    target_df = build_target_weights(picks_df, cfg)

    target_path = Path("output") / f"target_weights_{date_str}.csv"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_df.to_csv(target_path, index=False)

    prev_date = (pd.Timestamp(date_str) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    prev_target_path = Path("output") / f"target_weights_{prev_date}.csv"
    current_df = pd.read_csv(prev_target_path) if prev_target_path.exists() else None

    orders_df = build_rebalance_orders(target_df, current_df, threshold=cfg.rebalance_threshold)
    orders_path = Path("output") / f"orders_{date_str}.csv"
    orders_df.to_csv(orders_path, index=False)

    stats = summarize_orders(orders_df)
    print(f"Target weights saved: {target_path}")
    print(f"Orders saved: {orders_path}")
    print(f"Order stats: {stats}")

    record_run(
        "build_portfolio",
        date=date_str,
        picks_file=str(picks_path),
        target_file=str(target_path),
        orders_file=str(orders_path),
        **stats,
    )


if __name__ == "__main__":
    main()

