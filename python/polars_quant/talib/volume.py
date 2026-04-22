# ruff: noqa
import polars as pl
from polars.plugins import register_plugin_function
import inspect
from pathlib import Path

_LIB = Path(__file__).parent.parent / "polars_quant.abi3.so"

import polars as pl

def AD(high: pl.Series, low: pl.Series, close: pl.Series, volume: pl.Series) -> pl.Series:
    """AD - Chaikin A/D Line"""
    expr = register_plugin_function(args=[high, low, close, volume], plugin_path=_LIB, function_name='ad', is_elementwise=False)
    if isinstance(high, pl.Series):
        return high.to_frame().select(expr).to_series()
    return expr

def ADOSC(high: pl.Series, low: pl.Series, close: pl.Series, volume: pl.Series, fastperiod: int=3, slowperiod: int=10) -> pl.Series:
    """ADOSC - Chaikin A/D Oscillator"""
    expr = register_plugin_function(args=[high, low, close, volume, fastperiod, slowperiod], plugin_path=_LIB, function_name='adosc', is_elementwise=False)
    if isinstance(high, pl.Series):
        return high.to_frame().select(expr).to_series()
    return expr

def OBV(real: pl.Series, volume: pl.Series) -> pl.Series:
    """OBV - On Balance Volume"""
    expr = register_plugin_function(args=[real, volume], plugin_path=_LIB, function_name='obv', is_elementwise=False)
    if isinstance(real, pl.Series):
        return real.to_frame().select(expr).to_series()
    return expr