from pathlib import Path
from typing import Union

import polars as pl


def prepare_sequential_data(
    folder_path: Union[str, Path],
    date_col: str = "date",
    symbol_col: str = "symbol",
    fill_null_strategy: str = "forward",
    default_fill_value: float = 0.0,
) -> pl.DataFrame:
    """
    Reads all supported market data files (CSV, Parquet) from a specified directory,
    merges them into a single large DataFrame, and performs time alignment,
    length padding, and missing value processing.

    This Quality-of-Life (QoL) function ensures that multi-asset data is perfectly
    synchronized for the SequentialBacktester.

    Args:
        folder_path (str | Path): URL or local path to the directory containing data files.
        date_col (str): The name of the datetime/date column. Defaults to "date".
        symbol_col (str): The name of the ticker/symbol column. If missing in a file,
                          the file's base name will be automatically used.
        fill_null_strategy (str): Strategy to handle missing data ('forward', 'backward', 'zero').
                                  Defaults to 'forward' (avoids look-ahead bias).
        default_fill_value (float): The numeric value to use for filling remaining nulls
                                    (e.g., leading nulls before the first valid price).

    Returns:
        pl.DataFrame: A globally aligned, sorted, and cleaned Polars DataFrame.
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(
            f"The directory '{folder_path}' does not exist or is not a directory."
        )

    lazy_frames = []

    for file_path in folder.iterdir():
        if file_path.suffix.lower() == ".csv":
            lf = pl.scan_csv(file_path)
        elif file_path.suffix.lower() in (".parquet", ".pqt"):
            lf = pl.scan_parquet(file_path)
        else:
            continue

        schema_names = lf.collect_schema().names()
        if symbol_col not in schema_names:
            lf = lf.with_columns(pl.lit(file_path.stem).alias(symbol_col))

        lazy_frames.append(lf)

    if not lazy_frames:
        raise ValueError(f"No valid CSV or Parquet files found in '{folder_path}'.")

    master_lf = pl.concat(lazy_frames, how="diagonal_relaxed")

    unique_dates = master_lf.select(date_col).unique()
    unique_symbols = master_lf.select(symbol_col).unique()

    grid_lf = unique_dates.join(unique_symbols, how="cross")

    aligned_lf = grid_lf.join(master_lf, on=[date_col, symbol_col], how="left")

    aligned_lf = aligned_lf.sort([date_col, symbol_col])

    value_cols = [
        col
        for col in master_lf.collect_schema().names()
        if col not in (date_col, symbol_col)
    ]

    if value_cols:
        if fill_null_strategy == "forward":
            aligned_lf = aligned_lf.with_columns(
                pl.col(value_cols).forward_fill().over(symbol_col)
            )
        elif fill_null_strategy == "backward":
            aligned_lf = aligned_lf.with_columns(
                pl.col(value_cols).backward_fill().over(symbol_col)
            )
        elif fill_null_strategy == "zero":
            aligned_lf = aligned_lf.with_columns(pl.col(value_cols).fill_null(0.0))

        aligned_lf = aligned_lf.with_columns(
            pl.col(value_cols).fill_null(default_fill_value)
        )

    return aligned_lf.collect()
