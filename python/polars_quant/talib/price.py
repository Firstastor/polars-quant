# ruff: noqa
import polars as pl
from polars.plugins import register_plugin_function
from pathlib import Path

_LIB = str(Path(__file__).parent.parent)


def AVGPRICE(open: pl.Expr, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    """AVGPRICE - Average Price"""
    return register_plugin_function(
        args=[open, high, low, close],
        plugin_path=_LIB,
        function_name="avgprice",
        is_elementwise=False,
    )


def MEDPRICE(high: pl.Expr, low: pl.Expr) -> pl.Expr:
    """MEDPRICE - Median Price"""
    return register_plugin_function(
        args=[high, low],
        plugin_path=_LIB,
        function_name="medprice",
        is_elementwise=False,
    )


def TYPPRICE(high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    """TYPPRICE - Typical Price"""
    return register_plugin_function(
        args=[high, low, close],
        plugin_path=_LIB,
        function_name="typprice",
        is_elementwise=False,
    )


def WCLPRICE(high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    """WCLPRICE - Weighted Close Price"""
    return register_plugin_function(
        args=[high, low, close],
        plugin_path=_LIB,
        function_name="wclprice",
        is_elementwise=False,
    )
