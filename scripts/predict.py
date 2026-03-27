"""Generate stock predictions using the latest trained model.

Usage:
    python -m scripts.predict
    python -m scripts.predict --date 2026-03-05
    python -m scripts.predict --date 2026-03-05 --out output/top_picks_2026-03-05.csv
"""

import argparse
import random
import os
from pathlib import Path

import qlib
import pandas as pd
from qlib.data import D
from qlib.workflow import R
from qlib.config import REG_CN
from qlib.utils import init_instance_by_config

from config.settings import settings
from utils.run_tracker import get_last_run


def get_predict_conf(start_date, end_date, instruments):
    return {
        "class": "TSDatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": start_date,
                    "end_time": end_date,
                    "fit_start_time": start_date,
                    "fit_end_time": end_date,
                    "instruments": instruments,
                    "infer_processors": [
                        {"class": "RobustZScoreNorm", "kwargs": {"fields_group": "feature", "clip_outlier": True}},
                        {"class": "Fillna", "kwargs": {"fields_group": "feature"}}
                    ],
                    "learn_processors": [
                        {"class": "DropnaLabel"},
                        {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}
                    ],
                    "label": ["(Ref($close, -2)/Ref($close, -1) - 1 + Ref($close, -3)/Ref($close, -1) - 1 + Ref($close, -4)/Ref($close, -1) - 1 + Ref($close, -5)/Ref($close, -1) - 1 + Ref($close, -6)/Ref($close, -1) - 1) / 5"]
                },
            },
            "segments": {
                "test": [end_date, end_date],
            },
            "step_len": 20,
        },
    }


def _resolve_predict_date(calendar, date_str: str | None):
    if calendar is None or len(calendar) == 0:
        raise RuntimeError("Empty Qlib calendar. Check your provider_uri and dumped data.")
    if not date_str:
        return pd.Timestamp(calendar[-1]).strftime("%Y-%m-%d")
    target = pd.Timestamp(date_str).strftime("%Y-%m-%d")
    cal_str = [pd.Timestamp(x).strftime("%Y-%m-%d") for x in calendar]
    if target not in cal_str:
        raise ValueError(f"Date {target} not in local trading calendar (available: {cal_str[0]} ... {cal_str[-1]}).")
    return target


def _lookback_start(calendar, end_date: str, lookback: int = 120) -> str:
    cal_str = [pd.Timestamp(x).strftime("%Y-%m-%d") for x in calendar]
    end_date_str = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    idx = cal_str.index(end_date_str)
    start_idx = max(0, idx - lookback)
    return cal_str[start_idx]


def _trading_index(calendar, target_date: str) -> int:
    cal_str = [pd.Timestamp(x).strftime("%Y-%m-%d") for x in calendar]
    return cal_str.index(pd.Timestamp(target_date).strftime("%Y-%m-%d"))


def _build_today_pool(calendar, predict_date: str, seed: int = 42) -> list[str]:
    idx = _trading_index(calendar, predict_date)
    lookback_start_idx = max(0, idx - 59)
    start = pd.Timestamp(calendar[lookback_start_idx]).strftime("%Y-%m-%d")
    end = pd.Timestamp(calendar[idx]).strftime("%Y-%m-%d")

    # Liquidity ranking in last 60 trading days.
    feats = D.features(D.instruments("all"), ["$amount", "$close", "$volume"], start_time=start, end_time=end)
    if feats.empty:
        raise RuntimeError("No feature data available to build prediction pool.")

    amount = pd.to_numeric(feats.get("$amount"), errors="coerce")
    if amount is None or amount.isna().all():
        close = pd.to_numeric(feats.get("$close"), errors="coerce").fillna(0)
        volume = pd.to_numeric(feats.get("$volume"), errors="coerce").fillna(0)
        liquidity = (close * volume).fillna(0)
    else:
        liquidity = amount.fillna(0)

    ranked = (
        pd.DataFrame({"liq": liquidity})
        .groupby(level="instrument")["liq"]
        .sum()
        .sort_values(ascending=False)
        .head(1000)
        .index
        .tolist()
    )

    # Segment sampling: 1000 -> 10*100, remove top/bottom 10%, sample 40 each segment.
    rng = random.Random(seed)
    selected = []
    segment_size = 100
    for i in range(10):
        seg = ranked[i * segment_size : (i + 1) * segment_size]
        if len(seg) < 20:
            continue
        cut = max(1, int(len(seg) * 0.1))
        middle = seg[cut : len(seg) - cut]
        k = min(40, len(middle))
        if k > 0:
            selected.extend(rng.sample(middle, k))
    base_pool = list(dict.fromkeys(selected))
    selected_set = set(base_pool)

    # Expand from previous trading day prediction until total pool reaches 500.
    if idx > 0:
        prev_date = pd.Timestamp(calendar[idx - 1]).strftime("%Y-%m-%d")
        prev_file = Path("output") / f"top_picks_{prev_date}.csv"
        if prev_file.exists():
            prev_df = pd.read_csv(prev_file)
            if "instrument" in prev_df.columns and "Score" in prev_df.columns and not prev_df.empty:
                need = max(0, 500 - len(selected_set))
                if need > 0:
                    prev_ranked = (
                        prev_df.sort_values("Score", ascending=False)["instrument"]
                        .astype(str)
                        .tolist()
                    )
                    for sym in prev_ranked:
                        if sym in selected_set:
                            continue
                        selected_set.add(sym)
                        need -= 1
                        if need <= 0:
                            break

    return sorted(selected_set)


def main():
    parser = argparse.ArgumentParser(description="Predict for a given local trading day")
    parser.add_argument("--date", type=str, default=None, help="Target trading day (YYYY-MM-DD). Default: latest.")
    parser.add_argument("--out", type=str, default=None, help="Output csv path. Default: output/top_picks_<date>.csv")
    args = parser.parse_args()

    qlib.init(provider_uri=settings.qlib_provider_uri, region=REG_CN)

    all_calendar = D.calendar(freq='day')
    predict_date = _resolve_predict_date(all_calendar, args.date)
    # ~120 trading days lookback for Alpha158 indicator warm-up
    start_date_for_predict = _lookback_start(all_calendar, predict_date, lookback=120)
    pool_symbols = _build_today_pool(all_calendar, predict_date, seed=42)

    print(f"Predict date:       {predict_date}")
    print(f"Lookback start:     {start_date_for_predict}")
    print(f"Pool size:          {len(pool_symbols)}")

    print("Loading model from MLflow...")
    try:
        env_recorder_id = os.getenv("QLIB_RECORDER_ID")
        env_experiment_id = os.getenv("QLIB_EXPERIMENT_ID")

        if env_recorder_id and env_experiment_id:
            recorder_id = env_recorder_id
            experiment_id = env_experiment_id
        else:
            last = get_last_run("qlib_train")
            if not last:
                raise RuntimeError("No trained model found: run_history.json.qlib_train is empty.")
            recorder_id = last.get("recorder_id")
            experiment_id = last.get("experiment_id")
            if not recorder_id or not experiment_id:
                raise RuntimeError("No trained model found: missing recorder_id/experiment_id in run_history.json.qlib_train.")

        recorder = R.get_recorder(
            recorder_id=recorder_id,
            experiment_id=experiment_id,
        )
        # Prefer our saved key from workflow, fallback to historical qlib examples.
        try:
            model = recorder.load_object("trained_model")
        except Exception:
            model = recorder.load_object("params.pkl")
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit(1)

    print("Computing Alpha158 features ...")
    predict_dataset_conf = get_predict_conf(start_date_for_predict, predict_date, instruments=pool_symbols)
    dataset = init_instance_by_config(predict_dataset_conf)

    pred_score = model.predict(dataset)

    print("\n" + "=" * 50)
    print(f"{predict_date} Top Predictions")
    print("=" * 50)

    result_df = pred_score.sort_values(ascending=False).to_frame("Score")
    result_df.index = result_df.index.get_level_values('instrument')

    print(result_df)

    default_out = Path("output") / f"top_picks_{predict_date}.csv"
    out_path = Path(args.out) if args.out else default_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(out_path)
    print(f"\nSaved to: {out_path}")


if __name__ == '__main__':
    main()
