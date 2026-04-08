from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backtesting.portfolio import (
    PortfolioConfig,
    build_rebalance_orders,
    build_target_weights,
    summarize_orders,
)
from data_pipeline.database import DBClient
from model_function.qlib import (
    RecorderIdentity,
    build_alpha158_prediction_dataset_config,
    load_trained_model,
)
from model_function.universe import (
    HOLDING_BUFFER_DEFAULTS,
    PREDICTION_UNIVERSE_DEFAULTS,
    TRAINING_UNIVERSE_DEFAULTS,
    apply_portfolio_hold_buffer,
    build_training_universe_artifact,
    build_prediction_pool_from_features,
    resolve_training_universe_output_path,
)
from runtime.model_state import ModelRuntimeState, build_model_runtime_state
import utils.io as io_utils
from utils.preprocess import read_symbol_list

DEFAULT_DUMP_INCLUDE_FIELDS = "open,high,low,close,volume,amount,turn,isST,factor"
DEFAULT_DUMP_FILE_SUFFIX = ".csv"


@dataclass(frozen=True)
class TradingDateContext:
    """Resolved trading-day context for model artifact generation."""

    trading_date: str
    previous_trading_date: str | None


def _get_qlib_runtime():
    import qlib
    from qlib.config import REG_CN
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R

    return qlib, REG_CN, R, init_instance_by_config


def _get_qlib_data_client():
    from qlib.data import D

    return D


def _get_dump_all_class():
    from runtime.adapters.dump_bin_core import DumpDataAll

    return DumpDataAll


def dump_to_qlib_data(
    *,
    csv_dir: str | Path,
    qlib_dir: str | Path,
    include_fields: str = DEFAULT_DUMP_INCLUDE_FIELDS,
    file_suffix: str = DEFAULT_DUMP_FILE_SUFFIX,
) -> dict[str, str]:
    """Convert CSV market data into qlib binary format via direct Python call."""
    dump_all_cls = _get_dump_all_class()
    dumper = dump_all_cls(
        data_path=str(csv_dir),
        qlib_dir=str(qlib_dir),
        include_fields=include_fields,
        file_suffix=file_suffix,
    )
    dumper.dump()
    return {"csv_dir": str(csv_dir), "qlib_dir": str(qlib_dir)}


def _load_training_rows(
    db_client: DBClient,
    symbol: str,
    *,
    query_start: str,
    query_end: str,
) -> list[dict]:
    """Fetch one symbol's daily rows and normalize failed responses to an empty list."""

    response = db_client.query_data(symbol, query_start, query_end)
    if response is None or getattr(response, "status_code", None) != 200:
        return []
    payload = response.json() or {}
    rows = payload.get("data", [])
    if not isinstance(rows, list):
        return []
    return rows


def build_training_universe_file(
    *,
    start_year: int = 2010,
    end_year: int = 2026,
    top_n: int = TRAINING_UNIVERSE_DEFAULTS.exit_limit,
    random_seed: int = TRAINING_UNIVERSE_DEFAULTS.seed,
    data_path: str | Path | None = None,
    qlib_dir: str | Path | None = None,
    db_host: str | None = None,
    db_port: int | None = None,
) -> dict[str, Any]:
    """Build the training-universe instrument file through the shared model helper."""

    runtime_state = build_model_runtime_state()
    data_root = Path(data_path) if data_path is not None else Path(runtime_state.settings.data_path)
    qlib_root = Path(qlib_dir) if qlib_dir is not None else Path(runtime_state.settings.qlib_data_path)
    output_path = resolve_training_universe_output_path(data_path=data_root, qlib_data_path=qlib_root)
    all_stock_list = io_utils.read_file_lines(str(data_root / "stock_code_list"))
    index_symbols = read_symbol_list(data_root / "index_code_list")
    client = DBClient(
        db_host or runtime_state.settings.db_host,
        db_port or runtime_state.settings.db_port,
    )
    query_start = f"{int(start_year)}-01-01"
    query_end = pd.Timestamp(f"{int(end_year)}-12-31").strftime("%Y-%m-%d")

    return build_training_universe_artifact(
        symbols=all_stock_list,
        fetch_symbol_rows=lambda symbol: _load_training_rows(
            client,
            symbol,
            query_start=query_start,
            query_end=query_end,
        ),
        output_path=output_path,
        start_year=start_year,
        end_year=end_year,
        top_n=top_n,
        random_seed=random_seed,
        excluded_symbols=set(index_symbols),
    )


def get_predict_conf(start_date: str, end_date: str, instruments: list[str]) -> dict[str, Any]:
    """Return the shared Alpha158 prediction dataset config used by runtime scoring."""

    return build_alpha158_prediction_dataset_config(start_date, end_date, instruments)


def _calendar_to_strings(calendar) -> list[str]:
    """Normalize Qlib calendar values to stable trading-day strings."""

    if calendar is None or len(calendar) == 0:
        raise RuntimeError("Empty Qlib calendar. Check your provider_uri and dumped data.")
    return [pd.Timestamp(value).strftime("%Y-%m-%d") for value in calendar]


def _resolve_trading_date_context(calendar, date_str: str | None) -> TradingDateContext:
    """Resolve an execution date and its previous trading day from the local calendar."""

    cal_str = _calendar_to_strings(calendar)
    if not date_str:
        trading_date = cal_str[-1]
    else:
        trading_date = pd.Timestamp(date_str).strftime("%Y-%m-%d")
    target = trading_date
    if target not in cal_str:
        raise ValueError(f"Date {target} not in local trading calendar (available: {cal_str[0]} ... {cal_str[-1]}).")
    index = cal_str.index(target)
    previous_trading_date = cal_str[index - 1] if index > 0 else None
    return TradingDateContext(
        trading_date=target,
        previous_trading_date=previous_trading_date,
    )


def _resolve_execution_trading_context(
    *,
    date: str | None,
    runtime_state: ModelRuntimeState | None = None,
    provider_uri: str | None = None,
) -> TradingDateContext:
    """Load the local trading calendar through the runtime-owned provider boundary."""

    resolved_runtime_state = runtime_state or build_model_runtime_state()
    settings = resolved_runtime_state.settings
    qlib, reg_cn, _, _ = _get_qlib_runtime()
    d_client = _get_qlib_data_client()
    qlib.init(provider_uri=provider_uri or settings.qlib_provider_uri, region=reg_cn)
    calendar = d_client.calendar(freq="day")
    return _resolve_trading_date_context(calendar, date)


def _lookback_start(calendar, end_date: str, lookback: int = 120) -> str:
    """Return the first trading day inside the requested lookback window."""

    cal_str = _calendar_to_strings(calendar)
    end_date_str = pd.Timestamp(end_date).strftime("%Y-%m-%d")
    idx = cal_str.index(end_date_str)
    start_idx = max(0, idx - lookback)
    return cal_str[start_idx]


def _trading_index(calendar, target_date: str) -> int:
    """Return the calendar index for a normalized trading day."""

    cal_str = _calendar_to_strings(calendar)
    return cal_str.index(pd.Timestamp(target_date).strftime("%Y-%m-%d"))


def _resolve_recorder_ids(
    runtime_state: ModelRuntimeState,
    *,
    recorder_identity: RecorderIdentity | None = None,
) -> tuple[str, str]:
    """Preserve the adapter's historical `(recorder_id, experiment_id)` return order."""

    identity = recorder_identity or runtime_state.resolve_recorder_identity(env=os.environ)
    return identity.recorder_id, identity.experiment_id


def _build_today_pool(
    calendar,
    predict_date: str,
    *,
    seed: int = 42,
    data_path: str | Path | None = None,
    output_dir: str | Path = "output",
) -> list[str]:
    """Build the deterministic prediction pool for one resolved trading day."""

    del seed  # Prediction universe is deterministic under the redesigned contract.
    d_client = _get_qlib_data_client()
    runtime_state = build_model_runtime_state()
    data_root = Path(data_path) if data_path is not None else Path(runtime_state.settings.data_path)
    output_root = Path(output_dir)

    index_symbols = read_symbol_list(data_root / "index_code_list")
    date_context = _resolve_trading_date_context(calendar, predict_date)
    idx = _trading_index(calendar, date_context.trading_date)
    lookback_days = max(1, int(PREDICTION_UNIVERSE_DEFAULTS.liquidity_lookback_days))
    lookback_start_idx = max(0, idx - (lookback_days - 1))
    start = pd.Timestamp(calendar[lookback_start_idx]).strftime("%Y-%m-%d")
    end = pd.Timestamp(calendar[idx]).strftime("%Y-%m-%d")

    feats = d_client.features(
        d_client.instruments("all"),
        ["$amount", "$close", "$volume", "$isst"],
        start_time=start,
        end_time=end,
    )
    if feats.empty:
        raise RuntimeError("No feature data available to build prediction pool.")

    previous_holdings: list[str] = []
    if date_context.previous_trading_date is not None:
        prev_target_path = output_root / f"target_weights_{date_context.previous_trading_date}.csv"
        prev_target_df = _read_csv_if_available(prev_target_path)
        if prev_target_df is not None and "instrument" in prev_target_df.columns and not prev_target_df.empty:
            previous_holdings = prev_target_df["instrument"].astype(str).tolist()

    pool_symbols = build_prediction_pool_from_features(
        feats,
        previous_holdings,
        excluded_symbols=set(index_symbols),
        config=PREDICTION_UNIVERSE_DEFAULTS,
    )
    if not pool_symbols:
        raise RuntimeError("No eligible symbols remain after prediction-universe filtering.")
    return pool_symbols


def _to_result_dataframe(pred_score) -> pd.DataFrame:
    result_df = pred_score.sort_values(ascending=False).to_frame("Score")
    result_df.index = result_df.index.get_level_values("instrument")
    return result_df


def _atomic_write_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp.{os.getpid()}")
    df.to_csv(tmp_path)
    os.replace(tmp_path, out_path)


def _read_csv_if_available(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def generate_predictions(
    *,
    date: str | None = None,
    out: str | None = None,
    seed: int = 42,
    output_dir: str | Path = "output",
    provider_uri: str | None = None,
    mlruns_uri: str | None = None,
    runtime_state: ModelRuntimeState | None = None,
    recorder_identity: RecorderIdentity | None = None,
) -> dict[str, Any]:
    """Generate prediction artifacts for one resolved trading day."""

    resolved_runtime_state = runtime_state or build_model_runtime_state()
    settings = resolved_runtime_state.settings
    qlib, reg_cn, recorder_client, init_instance_by_config = _get_qlib_runtime()
    d_client = _get_qlib_data_client()

    qlib.init(provider_uri=provider_uri or settings.qlib_provider_uri, region=reg_cn)
    if mlruns_uri or settings.qlib_mlruns_uri:
        recorder_client.set_uri(mlruns_uri or settings.qlib_mlruns_uri)

    all_calendar = d_client.calendar(freq="day")
    date_context = _resolve_trading_date_context(all_calendar, date)
    predict_date = date_context.trading_date
    lookback_start = _lookback_start(all_calendar, predict_date, lookback=120)
    pool_symbols = _build_today_pool(
        all_calendar,
        predict_date,
        seed=seed,
        data_path=settings.data_path,
        output_dir=output_dir,
    )

    recorder_id, experiment_id = _resolve_recorder_ids(
        resolved_runtime_state,
        recorder_identity=recorder_identity,
    )
    recorder = recorder_client.get_recorder(recorder_id=recorder_id, experiment_id=experiment_id)
    model = load_trained_model(recorder, artifact_keys=("trained_model", "params.pkl"))

    predict_dataset_conf = get_predict_conf(lookback_start, predict_date, instruments=pool_symbols)
    dataset = init_instance_by_config(predict_dataset_conf)
    pred_score = model.predict(dataset)
    result_df = _to_result_dataframe(pred_score)

    default_out = Path(output_dir) / f"top_picks_{predict_date}.csv"
    out_path = Path(out) if out else default_out
    _atomic_write_csv(result_df, out_path)

    return {
        "predict_date": predict_date,
        "lookback_start": lookback_start,
        "pool_size": len(pool_symbols),
        "recorder_id": recorder_id,
        "experiment_id": experiment_id,
        "output_path": str(out_path),
        "result_df": result_df,
    }


def _normalize_picks_columns(picks_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common script-side pick CSV column names for portfolio construction."""

    normalized = picks_df.copy()
    if "instrument" not in normalized.columns:
        first_col = normalized.columns[0]
        if first_col.lower() in {"unnamed: 0", "symbol"}:
            normalized = normalized.rename(columns={first_col: "instrument"})
    if "Score" not in normalized.columns:
        score_col = [c for c in normalized.columns if c.lower() == "score"]
        if score_col:
            normalized = normalized.rename(columns={score_col[0]: "Score"})
    return normalized


def build_portfolio_outputs(
    *,
    date: str | None = None,
    top_k: int = 80,
    max_weight: float = 0.02,
    rebalance_threshold: float = 0.002,
    buy_rank: int = HOLDING_BUFFER_DEFAULTS.buy_rank,
    hold_rank: int = HOLDING_BUFFER_DEFAULTS.hold_rank,
    output_dir: str | Path = "output",
    track_run: bool = True,
    runtime_state: ModelRuntimeState | None = None,
) -> dict[str, Any]:
    """Build target weights and rebalance orders for one resolved trading day."""

    resolved_runtime_state = runtime_state or build_model_runtime_state()
    date_context = _resolve_execution_trading_context(
        date=date,
        runtime_state=resolved_runtime_state,
    )
    date_str = date_context.trading_date
    output_root = Path(output_dir)
    picks_path = output_root / f"top_picks_{date_str}.csv"
    if not picks_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {picks_path}")
    if hold_rank < buy_rank:
        raise ValueError("hold_rank must be greater than or equal to buy_rank.")

    picks_df = _normalize_picks_columns(pd.read_csv(picks_path))
    current_df = None
    if date_context.previous_trading_date is not None:
        prev_target_path = output_root / f"target_weights_{date_context.previous_trading_date}.csv"
        current_df = _read_csv_if_available(prev_target_path)

    current_instruments = []
    if current_df is not None and "instrument" in current_df.columns:
        current_instruments = current_df["instrument"].astype(str).tolist()

    cfg = PortfolioConfig(top_k=top_k, max_weight=max_weight, rebalance_threshold=rebalance_threshold)
    eligible_picks = apply_portfolio_hold_buffer(
        picks_df,
        current_instruments,
        buy_rank=buy_rank,
        hold_rank=hold_rank,
    )
    target_df = build_target_weights(eligible_picks, cfg)

    target_path = output_root / f"target_weights_{date_str}.csv"
    orders_path = output_root / f"orders_{date_str}.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    target_tmp = target_path.with_name(f".{target_path.name}.tmp.{os.getpid()}")
    orders_tmp = orders_path.with_name(f".{orders_path.name}.tmp.{os.getpid()}")
    target_df.to_csv(target_tmp, index=False)
    orders_df = build_rebalance_orders(target_df, current_df, threshold=cfg.rebalance_threshold)
    orders_df.to_csv(orders_tmp, index=False)

    os.replace(target_tmp, target_path)
    os.replace(orders_tmp, orders_path)

    stats = summarize_orders(orders_df)
    if track_run:
        resolved_runtime_state.history.record(
            "build_portfolio",
            date=date_str,
            picks_file=str(picks_path),
            target_file=str(target_path),
            orders_file=str(orders_path),
            **stats,
        )

    return {
        "date": date_str,
        "picks_path": str(picks_path),
        "target_path": str(target_path),
        "orders_path": str(orders_path),
        "stats": stats,
    }
