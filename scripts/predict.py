"""Generate stock predictions using the latest trained model.

Usage:
    python -m scripts.predict
    python -m scripts.predict --date 2026-03-05
    python -m scripts.predict --date 2026-03-05 --out output/top_picks_2026-03-05.csv
"""

import argparse
from runtime.adapters.modeling import generate_predictions


def main():
    parser = argparse.ArgumentParser(description="Predict for a given local trading day")
    parser.add_argument("--date", type=str, default=None, help="Target trading day (YYYY-MM-DD). Default: latest.")
    parser.add_argument("--out", type=str, default=None, help="Output csv path. Default: output/top_picks_<date>.csv")
    args = parser.parse_args()

    try:
        result = generate_predictions(date=args.date, out=args.out)
    except Exception as exc:
        # Keep CLI failure surface concise for operator-facing runs.
        print(f"Prediction failed: {exc}")
        raise SystemExit(1)

    predict_date = result["predict_date"]
    lookback_start = result["lookback_start"]
    pool_size = result["pool_size"]
    result_df = result["result_df"]
    out_path = result["output_path"]

    print(f"Predict date:       {predict_date}")
    print(f"Lookback start:     {lookback_start}")
    print(f"Pool size:          {pool_size}")
    print("Loading model from MLflow...")
    print("Computing Alpha158 features ...")

    print("\n" + "=" * 50)
    print(f"{predict_date} Top Predictions")
    print("=" * 50)
    print(result_df)
    print(f"\nSaved to: {out_path}")


if __name__ == '__main__':
    main()
