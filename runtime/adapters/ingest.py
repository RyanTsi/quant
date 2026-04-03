"""Runtime-owned ingest adapter for gateway CSV batch uploads."""

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


def _empty_result(base_url: str, data_dir: str) -> dict[str, Any]:
    return {
        "data_dir": data_dir,
        "server_url": base_url,
        "files_found": 0,
        "files_ingested": 0,
        "skipped_files": [],
        "rows_sent": 0,
        "failed_files": [],
        "failed_batches": [],
        "deleted_files": [],
    }


def ingest_directory(
    base_url: str,
    data_dir: str,
    *,
    delete_after_ingest: bool = False,
    batch_size: int = BATCH_SIZE,
    logger_override: logging.Logger | None = None,
) -> dict[str, Any]:
    """
    Read all CSVs in *data_dir* and POST rows in batches to the gateway.

    `delete_after_ingest=False` by default to avoid hidden filesystem side effects.
    Pipeline callers that need one-shot consumption should pass `True`.
    `rows_sent` counts rows from batches that returned HTTP 200.
    """
    active_logger = logger_override or logger
    result = _empty_result(base_url, data_dir)

    if not os.path.isdir(data_dir):
        active_logger.warning("Directory not found: %s", data_dir)
        return result

    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".csv"))
    total = len(files)
    result["files_found"] = total
    active_logger.info("Found %d CSV files in %s", total, data_dir)

    for idx, filename in enumerate(files, 1):
        filepath = os.path.join(data_dir, filename)
        symbol = os.path.splitext(filename)[0]

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            result["skipped_files"].append(filename)
            if delete_after_ingest:
                os.remove(filepath)
                result["deleted_files"].append(filename)
            continue

        data_list = [_row_to_payload(row, symbol) for row in rows]
        file_failed = False
        file_rows_sent = 0

        for i in range(0, len(data_list), batch_size):
            batch = data_list[i : i + batch_size]
            batch_id = i // batch_size + 1
            try:
                resp = requests.post(f"{base_url}/api/v1/ingest/daily", json=batch, timeout=30)
                if resp.status_code != 200:
                    file_failed = True
                    result["failed_batches"].append(
                        {
                            "file": filename,
                            "symbol": symbol,
                            "batch": batch_id,
                            "status_code": resp.status_code,
                            "error": resp.text,
                        }
                    )
                    active_logger.error("[%s] batch %d failed: %d %s", symbol, batch_id, resp.status_code, resp.text)
                else:
                    file_rows_sent += len(batch)
            except Exception as exc:
                file_failed = True
                result["failed_batches"].append(
                    {
                        "file": filename,
                        "symbol": symbol,
                        "batch": batch_id,
                        "status_code": None,
                        "error": str(exc),
                    }
                )
                active_logger.error("[%s] batch %d error: %s", symbol, batch_id, exc)

        result["rows_sent"] += file_rows_sent
        if file_failed:
            result["failed_files"].append(filename)
        else:
            result["files_ingested"] += 1

        if delete_after_ingest:
            os.remove(filepath)
            result["deleted_files"].append(filename)
        active_logger.info("[%d/%d] %s: %d rows sent.", idx, total, symbol, file_rows_sent)

    return result


__all__ = [
    "BATCH_SIZE",
    "ingest_directory",
]
