# ruff: noqa
import polars as pl
from pathlib import Path
from polars.plugins import register_plugin_function

_LIB = str(Path(__file__).parent.parent)


def ATR(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    timeperiod: int = 14,
) -> pl.Expr:
    """ATR - Average True Range"""
    return register_plugin_function(
        args=[high, low, close],
        kwargs={"timeperiod": timeperiod},
        plugin_path=_LIB,
        function_name="atr",
        is_elementwise=False,
    )


def NATR(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    timeperiod: int = 14,
) -> pl.Expr:
    """NATR - Normalized Average True Range"""
    return register_plugin_function(
        args=[high, low, close],
        kwargs={"timeperiod": timeperiod},
        plugin_path=_LIB,
        function_name="natr",
        is_elementwise=False,
    )


def TRANGE(high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    """TRANGE - True Range"""
    return register_plugin_function(
        args=[high, low, close],
        plugin_path=_LIB,
        function_name="trange",
        is_elementwise=False,
    )
