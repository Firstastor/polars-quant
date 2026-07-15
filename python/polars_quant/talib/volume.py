# ruff: noqa
import polars as pl
from polars.plugins import register_plugin_function
from pathlib import Path

_LIB = str(Path(__file__).parent.parent)


def AD(high: pl.Expr, low: pl.Expr, close: pl.Expr, volume: pl.Expr) -> pl.Expr:
    """AD - Chaikin A/D Line"""
    return register_plugin_function(
        args=[high, low, close, volume],
        plugin_path=_LIB,
        function_name="ad",
        is_elementwise=False,
    )


def ADOSC(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    volume: pl.Expr,
    fastperiod: int = 3,
    slowperiod: int = 10,
) -> pl.Expr:
    """ADOSC - Chaikin A/D Oscillator"""
    return register_plugin_function(
        args=[high, low, close, volume],
        kwargs={"fastperiod": fastperiod, "slowperiod": slowperiod},
        plugin_path=_LIB,
        function_name="adosc",
        is_elementwise=False,
    )


def OBV(real: pl.Expr, volume: pl.Expr) -> pl.Expr:
    """OBV - On Balance Volume"""
    return register_plugin_function(
        args=[real, volume],
        plugin_path=_LIB,
        function_name="obv",
        is_elementwise=False,
    )
