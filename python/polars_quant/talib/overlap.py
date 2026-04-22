# ruff: noqa
import polars as pl
from polars.plugins import register_plugin_function
from pathlib import Path

_LIB = Path(__file__).parent.parent / "polars_quant.abi3.so"


def BBANDS(
    real: pl.Expr, timeperiod: int = 20, nbdevup: float = 2.0, nbdevdn: float = 2.0
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """BBANDS - Bollinger Bands (Upper, Middle, Lower)"""
    expr = register_plugin_function(
        args=[real, timeperiod, nbdevup, nbdevdn],
        plugin_path=_LIB,
        function_name="bbands",
        is_elementwise=False,
    )
    return (
        expr.struct.field("bb_upper"),
        expr.struct.field("bb_middle"),
        expr.struct.field("bb_lower"),
    )


def DEMA(real: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    """DEMA - Double Exponential Moving Average"""
    return register_plugin_function(
        args=[real, timeperiod],
        plugin_path=_LIB,
        function_name="dema",
        is_elementwise=False,
    )


def EMA(real: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    """EMA - Exponential Moving Average"""
    return register_plugin_function(
        args=[real, timeperiod],
        plugin_path=_LIB,
        function_name="ema",
        is_elementwise=False,
    )


def KAMA(real: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    """KAMA - Kaufman Adaptive Moving Average"""
    return register_plugin_function(
        args=[real, timeperiod],
        plugin_path=_LIB,
        function_name="kama",
        is_elementwise=False,
    )


def MA(real: pl.Expr, timeperiod: int = 30, matype: int = 0) -> pl.Expr:
    """MA - Moving Average"""
    return register_plugin_function(
        args=[real, timeperiod, matype],
        plugin_path=_LIB,
        function_name="ma",
        is_elementwise=False,
    )


def MAMA(
    real: pl.Expr, fastlimit: float = 0.0, slowlimit: float = 0.0
) -> tuple[pl.Expr, pl.Expr]:
    """MAMA - MESA Adaptive Moving Average"""
    expr = register_plugin_function(
        args=[real, fastlimit, slowlimit],
        plugin_path=_LIB,
        function_name="mama",
        is_elementwise=False,
    )
    return (expr.struct.field("mama"), expr.struct.field("fama"))


def MAVP(
    real: pl.Expr,
    periods: pl.Expr,
    minperiod: int = 2,
    maxperiod: int = 30,
    matype: int = 0,
) -> pl.Expr:
    """MAVP - Moving Average with Variable Period"""
    return register_plugin_function(
        args=[real, periods, minperiod, maxperiod, matype],
        plugin_path=_LIB,
        function_name="mavp",
        is_elementwise=False,
    )


def MIDPOINT(real: pl.Expr, timeperiod: int = 14) -> pl.Expr:
    """MIDPOINT - Midpoint over period"""
    return register_plugin_function(
        args=[real, timeperiod],
        plugin_path=_LIB,
        function_name="midpoint",
        is_elementwise=False,
    )


def MIDPRICE(high: pl.Expr, low: pl.Expr, timeperiod: int = 14) -> pl.Expr:
    """MIDPRICE - Midpoint Price over period"""
    return register_plugin_function(
        args=[high, low, timeperiod],
        plugin_path=_LIB,
        function_name="midprice",
        is_elementwise=False,
    )


def SAR(
    high: pl.Expr, low: pl.Expr, acceleration: float = 0.0, maximum: float = 0.0
) -> pl.Expr:
    """SAR - Parabolic SAR"""
    return register_plugin_function(
        args=[high, low, acceleration, maximum],
        plugin_path=_LIB,
        function_name="sar",
        is_elementwise=False,
    )


def SAREXT(
    high: pl.Expr,
    low: pl.Expr,
    startvalue: float = 0.0,
    offsetonreverse: float = 0.0,
    accelerationinitlong: float = 0.0,
    accelerationlong: float = 0.0,
    accelerationmaxlong: float = 0.0,
    accelerationinitshort: float = 0.0,
    accelerationshort: float = 0.0,
    accelerationmaxshort: float = 0.0,
) -> pl.Expr:
    """SAREXT - Parabolic SAR - Extended"""
    return register_plugin_function(
        args=[
            high,
            low,
            startvalue,
            offsetonreverse,
            accelerationinitlong,
            accelerationlong,
            accelerationmaxlong,
            accelerationinitshort,
            accelerationshort,
            accelerationmaxshort,
        ],
        plugin_path=_LIB,
        function_name="sarext",
        is_elementwise=False,
    )


def SMA(real: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    """SMA - Simple Moving Average"""
    return register_plugin_function(
        args=[real, timeperiod],
        plugin_path=_LIB,
        function_name="sma",
        is_elementwise=False,
    )


def T3(real: pl.Expr, timeperiod: int = 5, vfactor: float = 0.7) -> pl.Expr:
    """T3 - Triple Exponential Moving Average (T3)"""
    return register_plugin_function(
        args=[real, timeperiod, vfactor],
        plugin_path=_LIB,
        function_name="t3",
        is_elementwise=False,
    )


def TEMA(real: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    """TEMA - Triple Exponential Moving Average"""
    return register_plugin_function(
        args=[real, timeperiod],
        plugin_path=_LIB,
        function_name="tema",
        is_elementwise=False,
    )


def TRIMA(real: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    """TRIMA - Triangular Moving Average"""
    return register_plugin_function(
        args=[real, timeperiod],
        plugin_path=_LIB,
        function_name="trima",
        is_elementwise=False,
    )


def WMA(real: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    """WMA - Weighted Moving Average"""
    return register_plugin_function(
        args=[real, timeperiod],
        plugin_path=_LIB,
        function_name="wma",
        is_elementwise=False,
    )
