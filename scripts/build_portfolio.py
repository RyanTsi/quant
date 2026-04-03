"""Build target portfolio weights and rebalance orders from prediction picks.

Usage:
    python -m scripts.build_portfolio
    python -m scripts.build_portfolio --date 2026-03-26
"""

from __future__ import annotations

import argparse
from runtime.adapters.modeling import build_portfolio_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build target portfolio and rebalance orders")
    parser.add_argument("--date", type=str, default=None, help="Prediction date YYYY-MM-DD")
    parser.add_argument("--top_k", type=int, default=80, help="Top-k picks for portfolio construction")
    parser.add_argument("--buy_rank", type=int, default=300, help="New buys must rank within this model-score band")
    parser.add_argument("--hold_rank", type=int, default=500, help="Existing holdings may remain inside this wider score band")
    parser.add_argument("--max_weight", type=float, default=0.02, help="Per-stock max target weight")
    parser.add_argument("--rebalance_threshold", type=float, default=0.002, help="Min abs delta weight to trade")
    args = parser.parse_args()

    try:
        result = build_portfolio_outputs(
            date=args.date,
            top_k=args.top_k,
            buy_rank=args.buy_rank,
            hold_rank=args.hold_rank,
            max_weight=args.max_weight,
            rebalance_threshold=args.rebalance_threshold,
        )
    except Exception as exc:
        # Keep CLI failure surface concise for operator-facing runs.
        print(f"Portfolio build failed: {exc}")
        raise SystemExit(1)

    target_path = result["target_path"]
    orders_path = result["orders_path"]
    stats = result["stats"]
    print(f"Target weights saved: {target_path}")
    print(f"Orders saved: {orders_path}")
    print(f"Order stats: {stats}")


if __name__ == "__main__":
    main()
