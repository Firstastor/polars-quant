# ruff: noqa
import polars as pl
from polars.plugins import register_plugin_function
import inspect
from pathlib import Path

_LIB = Path(__file__).parent.parent / "polars_quant.abi3.so"

import polars as pl

def AVGPRICE(open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """AVGPRICE - Average Price"""
    expr = register_plugin_function(args=[open, high, low, close], plugin_path=_LIB, function_name='avgprice', is_elementwise=False)
    if isinstance(open, pl.Series):
        return open.to_frame().select(expr).to_series()
    return expr

def MEDPRICE(high: pl.Series, low: pl.Series) -> pl.Series:
    """MEDPRICE - Median Price"""
    expr = register_plugin_function(args=[high, low], plugin_path=_LIB, function_name='medprice', is_elementwise=False)
    if isinstance(high, pl.Series):
        return high.to_frame().select(expr).to_series()
    return expr

def TYPPRICE(high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """TYPPRICE - Typical Price"""
    expr = register_plugin_function(args=[high, low, close], plugin_path=_LIB, function_name='typprice', is_elementwise=False)
    if isinstance(high, pl.Series):
        return high.to_frame().select(expr).to_series()
    return expr

def WCLPRICE(high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """WCLPRICE - Weighted Close Price"""
    expr = register_plugin_function(args=[high, low, close], plugin_path=_LIB, function_name='wclprice', is_elementwise=False)
    if isinstance(high, pl.Series):
        return high.to_frame().select(expr).to_series()
    return expr