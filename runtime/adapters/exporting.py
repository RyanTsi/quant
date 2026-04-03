from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

DEFAULT_EXPORT_PAGE_SIZE = 3000
DEFAULT_EXPORT_CSV_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turn",
    "isST",
    "factor",
]
DB_TO_CSV_COLUMN_RENAMES = {"is_st": "isST"}


def normalize_gateway_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    csv_columns: Sequence[str] = DEFAULT_EXPORT_CSV_COLUMNS,
) -> pd.DataFrame:
    """Normalize gateway rows into export CSV schema."""
    if not rows:
        return pd.DataFrame(columns=list(csv_columns))

    df = pd.DataFrame(list(rows))
    df.rename(columns=DB_TO_CSV_COLUMN_RENAMES, inplace=True)

    if "factor" not in df.columns:
        df["factor"] = 1.0
    if "date" in df.columns:
        df["date"] = df["date"].astype(str).str[:10]
    if "symbol" in df.columns:
        df.drop(columns=["symbol"], inplace=True)
    if "tradestatus" in df.columns:
        df = df[df["tradestatus"] != 0]

    ordered_columns = [column for column in csv_columns if column in df.columns]
    return df[ordered_columns]


def fetch_symbol_rows(
    client: Any,
    *,
    symbol: str,
    start_date: str,
    end_date: str,
    page_size: int = DEFAULT_EXPORT_PAGE_SIZE,
) -> tuple[list[dict[str, Any]], bool]:
    """Fetch all pages for one symbol. Returns rows and query failure flag."""
    rows: list[dict[str, Any]] = []
    offset = 0
    query_failed = False

    while True:
        response = client.query_data(symbol, start_date, end_date, limit=page_size, offset=offset)
        if response is None or response.status_code != 200:
            query_failed = True
            break

        data = response.json().get("data", [])
        rows.extend(data)
        if len(data) < page_size:
            break
        offset += page_size

    return rows, query_failed


def export_symbol_csvs(
    client: Any,
    *,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    output_dir: str,
    logger: logging.Logger | None = None,
    page_size: int = DEFAULT_EXPORT_PAGE_SIZE,
) -> dict[str, Any]:
    """Export each symbol to one CSV file with normalized schema."""
    os.makedirs(output_dir, exist_ok=True)

    exported = 0
    failed_symbols: list[str] = []
    partial_symbols: list[str] = []
    total = len(symbols)

    for index, symbol in enumerate(symbols):
        rows, query_failed = fetch_symbol_rows(
            client,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            page_size=page_size,
        )

        if not rows:
            if query_failed:
                failed_symbols.append(symbol)
                if logger is not None:
                    logger.warning("  Skipping symbol %s due to gateway query failure.", symbol)
            continue

        normalized_df = normalize_gateway_rows(rows)
        normalized_df.to_csv(os.path.join(output_dir, f"{symbol}.csv"), index=False)
        exported += 1

        if query_failed:
            partial_symbols.append(symbol)
            if logger is not None:
                logger.warning("  Symbol %s exported from partial data after gateway query failure.", symbol)

        if logger is not None and (index + 1) % 500 == 0:
            logger.info("  Progress: %d/%d", index + 1, total)

    return {
        "output_dir": output_dir,
        "exported": exported,
        "total": total,
        "failed_symbols": failed_symbols,
        "partial_symbols": partial_symbols,
    }


def export_from_gateway(
    client: Any,
    *,
    start_date: str,
    end_date: str,
    output_dir: str,
    logger: logging.Logger | None = None,
    page_size: int = DEFAULT_EXPORT_PAGE_SIZE,
) -> dict[str, Any]:
    """Export all symbols from gateway to per-symbol CSV files."""
    health = client.health()
    if health.get("status") != "healthy":
        raise RuntimeError(f"DB unreachable: {health}")

    response = client.list_symbols()
    if response is None or response.status_code != 200:
        raise RuntimeError("Failed to list symbols from DB.")

    symbols = response.json().get("symbols", [])
    if logger is not None:
        logger.info("  Exporting %d symbols: %s -> %s", len(symbols), start_date, end_date)

    result = export_symbol_csvs(
        client,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        logger=logger,
        page_size=page_size,
    )

    if logger is not None:
        logger.info("  Exported %d/%d symbols to %s", result["exported"], result["total"], output_dir)
        if result["failed_symbols"]:
            logger.warning("  Failed to export %d symbols.", len(result["failed_symbols"]))
        if result["partial_symbols"]:
            logger.warning("  Exported %d symbols from partial data.", len(result["partial_symbols"]))

    return result


__all__ = [
    "DEFAULT_EXPORT_CSV_COLUMNS",
    "DEFAULT_EXPORT_PAGE_SIZE",
    "export_from_gateway",
    "export_symbol_csvs",
    "fetch_symbol_rows",
    "normalize_gateway_rows",
]
