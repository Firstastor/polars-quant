# ruff: noqa
import polars as pl
from polars.plugins import register_plugin_function
from pathlib import Path

_LIB = str(Path(__file__).parent.parent)


def HT_DCPERIOD(real: pl.Expr) -> pl.Expr:
    """HT_DCPERIOD - Hilbert Transform - Dominant Cycle Period"""
    return register_plugin_function(
        args=[real],
        plugin_path=_LIB,
        function_name="ht_dcperiod",
        is_elementwise=False,
    )


def HT_DCPHASE(real: pl.Expr) -> pl.Expr:
    """HT_DCPHASE - Hilbert Transform - Dominant Cycle Phase"""
    return register_plugin_function(
        args=[real],
        plugin_path=_LIB,
        function_name="ht_dcphase",
        is_elementwise=False,
    )


def HT_PHASOR(real: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """HT_PHASOR - Hilbert Transform - Phasor Components (InPhase, Quadrature)"""
    expr = register_plugin_function(
        args=[real],
        plugin_path=_LIB,
        function_name="ht_phasor",
        is_elementwise=False,
    )
    return expr.struct.field("inphase"), expr.struct.field("quadrature")


def HT_SINE(real: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    """HT_SINE - Hilbert Transform - Sine Wave (Sine, LeadSine)"""
    expr = register_plugin_function(
        args=[real],
        plugin_path=_LIB,
        function_name="ht_sine",
        is_elementwise=False,
    )
    return expr.struct.field("sine"), expr.struct.field("leadsine")


def HT_TRENDLINE(real: pl.Expr) -> pl.Expr:
    """HT_TRENDLINE - Hilbert Transform - Trendline"""
    return register_plugin_function(
        args=[real],
        plugin_path=_LIB,
        function_name="ht_trendline",
        is_elementwise=False,
    )


def HT_TRENDMODE(real: pl.Expr) -> pl.Expr:
    """HT_TRENDMODE - Hilbert Transform - Trend Mode"""
    return register_plugin_function(
        args=[real],
        plugin_path=_LIB,
        function_name="ht_trendmode",
        is_elementwise=False,
    )
