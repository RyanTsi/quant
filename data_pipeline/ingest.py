"""Ingest CSV files into the C++ data gateway via HTTP batch API."""

from __future__ import annotations

import csv
import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

BATCH_SIZE = 4096


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        text = str(val).strip()
        return float(text) if text else default
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        text = str(val).strip()
        return int(float(text)) if text else default
    except (ValueError, TypeError):
        return default


def _row_to_payload(row: dict[str, Any], fallback_symbol: str) -> dict[str, Any]:
    return {
        "date": str(row.get("date", "")).strip(),
        "symbol": str(row.get("symbol") or fallback_symbol).strip(),
        "open": _safe_float(row.get("open")),
        "high": _safe_float(row.get("high")),
        "low": _safe_float(row.get("low")),
        "close": _safe_float(row.get("close")),
        "volume": _safe_float(row.get("volume")),
        "amount": _safe_float(row.get("amount")),
        "turn": _safe_float(row.get("turn")),
        "tradestatus": _safe_int(row.get("tradestatus"), 1),
        "isST": _safe_int(row.get("isST")),
    }


def ingest_directory(
    base_url: str,
    data_dir: str,
    *,
    delete_after_ingest: bool = False,
) -> None:
    """
    Read all CSVs in *data_dir* and POST rows in batches to the gateway.

    `delete_after_ingest=False` by default to avoid hidden filesystem side effects.
    Pipeline callers that need one-shot consumption should pass `True`.
    """
    if not os.path.isdir(data_dir):
        logger.warning("Directory not found: %s", data_dir)
        return

    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    total = len(files)
    logger.info("Found %d CSV files in %s", total, data_dir)

    for idx, filename in enumerate(files, 1):
        filepath = os.path.join(data_dir, filename)
        symbol = os.path.splitext(filename)[0]

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            if delete_after_ingest:
                os.remove(filepath)
            continue

        data_list = [_row_to_payload(row, symbol) for row in rows]
        for i in range(0, len(data_list), BATCH_SIZE):
            batch = data_list[i : i + BATCH_SIZE]
            try:
                resp = requests.post(f"{base_url}/api/v1/ingest/daily", json=batch, timeout=30)
                if resp.status_code != 200:
                    logger.error(
                        "[%s] batch %d failed: %d %s",
                        symbol,
                        i // BATCH_SIZE + 1,
                        resp.status_code,
                        resp.text,
                    )
            except Exception as exc:
                logger.error("[%s] batch %d error: %s", symbol, i // BATCH_SIZE + 1, exc)

        if delete_after_ingest:
            os.remove(filepath)
        logger.info("[%d/%d] %s: %d rows sent.", idx, total, symbol, len(data_list))
