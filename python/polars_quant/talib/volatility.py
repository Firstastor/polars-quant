# ruff: noqa
import polars as pl
from pathlib import Path
from polars.plugins import register_plugin_function

_LIB = Path(__file__).parent.parent / "polars_quant.abi3.so"


def ATR(
    high: pl.Series,
    low: pl.Series,
    close: pl.Series,
    timeperiod: int = 14,
) -> pl.Series:
    """ATR - Average True Range"""
    expr = register_plugin_function(
        args=[high, low, close, timeperiod],
        plugin_path=_LIB,
        function_name="atr",
        is_elementwise=False,
    )
    if isinstance(high, pl.Series):
        return high.to_frame().select(expr).to_series()
    return expr


def NATR(
    high: pl.Series,
    low: pl.Series,
    close: pl.Series,
    timeperiod: int = 14,
) -> pl.Series:
    """NATR - Normalized Average True Range"""
    expr = register_plugin_function(
        args=[high, low, close, timeperiod],
        plugin_path=_LIB,
        function_name="natr",
        is_elementwise=False,
    )
    if isinstance(high, pl.Series):
        return high.to_frame().select(expr).to_series()
    return expr


def TRANGE(high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """TRANGE - True Range"""
    expr = register_plugin_function(
        args=[high, low, close],
        plugin_path=_LIB,
        function_name="trange",
        is_elementwise=False,
    )
    if isinstance(high, pl.Series):
        return high.to_frame().select(expr).to_series()
    return expr
