"""Build the month-lagged training-universe instrument file.

Usage:
    python -m scripts.filter
    python -m scripts.filter --start_year 2010 --end_year 2026 --top_n 2200 --random_seed 42
"""

from __future__ import annotations

import argparse

from runtime.services import build_model_service


def filter_top_liquidity(
    start_year: int = 2010,
    end_year: int = 2026,
    top_n: int = 2200,
    random_seed: int = 42,
) -> dict[str, int | str]:
    """Compatibility wrapper over the runtime-managed training-universe build path."""

    service = build_model_service(refresh_settings=True)
    return service.build_training_universe(
        start_year=start_year,
        end_year=end_year,
        top_n=top_n,
        random_seed=random_seed,
    )


def main() -> None:
    """Parse CLI arguments and dispatch the training-universe builder."""

    parser = argparse.ArgumentParser(description="Build the monthly lagged training-universe txt")
    parser.add_argument("--start_year", type=int, default=2010, help="Source-history start year")
    parser.add_argument("--end_year", type=int, default=2026, help="Requested target end year")
    parser.add_argument("--top_n", type=int, default=2200, help="Liquidity ranking cap before buffering")
    parser.add_argument("--random_seed", type=int, default=42, help="Deterministic training-universe seed")
    args = parser.parse_args()

    try:
        result = filter_top_liquidity(
            start_year=args.start_year,
            end_year=args.end_year,
            top_n=args.top_n,
            random_seed=args.random_seed,
        )
    except Exception as exc:
        print(f"Filter build failed: {exc}")
        raise SystemExit(1)

    print(f"Saved to: {result['output_path']}")
    print(f"Effective end: {result['effective_end']}")
    print(f"Source months: {result['source_month_count']}")
    print(f"Unique symbols in artifact: {result['symbol_count']}")


if __name__ == "__main__":
    main()
