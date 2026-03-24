"""Ingest per-symbol CSV files into the C++ data gateway via HTTP batch API."""

import logging
import os
import requests

import utils.io

logger = logging.getLogger(__name__)

BATCH_SIZE = 4096


def _safe_float(val, default=0.0):
    try:
        return float(val) if val.strip() else default
    except (ValueError, AttributeError):
        return default


def _safe_int(val, default=0):
    try:
        return int(float(val)) if val.strip() else default
    except (ValueError, AttributeError):
        return default


def ingest_directory(base_url: str, data_dir: str):
    """Read all CSVs in *data_dir* and POST them in batches to the gateway."""
    if not os.path.isdir(data_dir):
        logger.warning("Directory not found: %s", data_dir)
        return

    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    total = len(files)
    logger.info("Found %d CSV files in %s", total, data_dir)

    for idx, filename in enumerate(files, 1):
        filepath = os.path.join(data_dir, filename)
        symbol = filename.split(".")[0]
        lines = utils.io.read_file_lines(filepath)
        if len(lines) < 2:
            continue

        header = lines[0].split(",")
        col_map = {k.strip(): v for v, k in enumerate(header)}

        data_list = []
        for line in lines[1:]:
            cols = line.split(",")
            if len(cols) < len(header):
                continue
            payload = {
                "date":        cols[col_map["date"]].strip(),
                "symbol":      cols[col_map["symbol"]].strip() if "symbol" in col_map else symbol,
                "open":        _safe_float(cols[col_map["open"]]),
                "high":        _safe_float(cols[col_map["high"]]),
                "low":         _safe_float(cols[col_map["low"]]),
                "close":       _safe_float(cols[col_map["close"]]),
                "volume":      _safe_float(cols[col_map["volume"]]) if "volume" in col_map else 0.0,
                "amount":      _safe_float(cols[col_map["amount"]]) if "amount" in col_map else 0.0,
                "turn":        _safe_float(cols[col_map["turn"]]) if "turn" in col_map else 0.0,
                "tradestatus": _safe_int(cols[col_map["tradestatus"]], 1) if "tradestatus" in col_map else 1,
                "isST":        _safe_int(cols[col_map["isST"]]) if "isST" in col_map else 0,
            }
            data_list.append(payload)

        for i in range(0, len(data_list), BATCH_SIZE):
            batch = data_list[i:i + BATCH_SIZE]
            try:
                resp = requests.post(f"{base_url}/api/v1/ingest/daily", json=batch, timeout=30)
                if resp.status_code != 200:
                    logger.error("[%s] batch %d failed: %d %s", symbol, i // BATCH_SIZE + 1, resp.status_code, resp.text)
            except Exception as e:
                logger.error("[%s] batch %d error: %s", symbol, i // BATCH_SIZE + 1, e)

        logger.info("[%d/%d] %s: %d rows sent.", idx, total, symbol, len(data_list))
