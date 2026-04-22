import contextlib
import io

import baostock as bs
import polars as pl
import pytest


def fetch_baostock_data(
    code="sh.600000", start_date="2010-01-01", end_date="2023-12-31"
):
    """Fetch historical k-line data using baostock."""
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        bs.login()
        rs = bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,volume,amount,pctChg",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3",
        )

        data_list = []
        while (rs.error_code == "0") and rs.next():
            data_list.append(rs.get_row_data())
        bs.logout()

    df = pl.DataFrame(
        data_list,
        schema=[
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "pctChg",
        ],
        orient="row",
    )

    # Cast necessary columns to float
    df = df.with_columns(
        [
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
            pl.col("amount").cast(pl.Float64),
        ]
    ).drop_nulls()
    return df


@pytest.fixture(scope="session")
def base_data():
    """Fetch data once per session to avoid repeated network calls."""
    return fetch_baostock_data()


@pytest.fixture(scope="session")
def small_stock_data(base_data):
    """Provide a small dataset (original size, approx 3.4k rows)."""
    print(f"\n--- Small Dataset Initialized: {base_data.height:,} rows ---")
    return base_data


@pytest.fixture(scope="session")
def large_stock_data(base_data):
    """Provide a large dataset by duplicating 100 times (approx 340k rows)."""
    large_df = pl.concat([base_data] * 100)
    print(f"\n--- Large Dataset Initialized: {large_df.height:,} rows ---")
    return large_df


@pytest.fixture(scope="session")
def stock_data(small_stock_data):
    """Alias for backwards compatibility with existing tests that rely on 'stock_data'."""
    return small_stock_data
