# ruff: noqa
import polars as pl
from polars.plugins import register_plugin_function
import inspect
from pathlib import Path

_LIB = Path(__file__).parent.parent / "polars_quant.abi3.so"

import math
import polars as pl
_TWO_PI = 2.0 * math.pi
_RAD2DEG = 180.0 / math.pi

def _wma4(data: list, i: int) -> float:
    if i < 3:
        return 0.0
    d0 = data[i] if data[i] is not None else 0.0
    d1 = data[i - 1] if data[i - 1] is not None else 0.0
    d2 = data[i - 2] if data[i - 2] is not None else 0.0
    d3 = data[i - 3] if data[i - 3] is not None else 0.0
    return (4.0 * d0 + 3.0 * d1 + 2.0 * d2 + d3) * 0.1

def _hilbert_transform(inp: list, i: int, adj: float) -> float:
    if i < 6:
        return 0.0
    v0 = inp[i] if inp[i] is not None else 0.0
    v2 = inp[i - 2] if i >= 2 and inp[i - 2] is not None else 0.0
    v4 = inp[i - 4] if i >= 4 and inp[i - 4] is not None else 0.0
    v6 = inp[i - 6] if i >= 6 and inp[i - 6] is not None else 0.0
    return (0.0962 * v0 + 0.5769 * v2 - 0.5769 * v4 - 0.0962 * v6) * adj

def _compute_mesa_state(r: list):
    n = len(r)
    smooth = [0.0] * n
    detrend = [0.0] * n
    q1 = [0.0] * n
    i1 = [0.0] * n
    ji = [0.0] * n
    jq = [0.0] * n
    i2 = [0.0] * n
    q2 = [0.0] * n
    re_ = [0.0] * n
    im_ = [0.0] * n
    period = [0.0] * n
    smooth_period = [0.0] * n
    phase = [0.0] * n
    wma = [0.0] * n
    for i in range(n):
        if r[i] is None:
            if i > 0:
                wma[i] = wma[i - 1]
            continue
        wma[i] = _wma4(r, i)
    for i in range(3, n):
        smooth[i] = _wma4(wma, i)
    for i in range(6, n):
        prev_p = period[i - 1] if i > 0 else 6.0
        adj = 0.075 * prev_p + 0.54
        detrend[i] = _hilbert_transform(smooth, i, adj)
        q1[i] = _hilbert_transform(detrend, i, adj)
        i1[i] = detrend[i - 3] if i >= 3 else 0.0
        ji[i] = _hilbert_transform(i1, i, adj)
        jq[i] = _hilbert_transform(q1, i, adj)
        i2[i] = i1[i] - jq[i]
        q2[i] = q1[i] + ji[i]
        i2[i] = 0.2 * i2[i] + 0.8 * i2[i - 1]
        q2[i] = 0.2 * q2[i] + 0.8 * q2[i - 1]
        re_[i] = i2[i] * i2[i - 1] + q2[i] * q2[i - 1]
        im_[i] = i2[i] * q2[i - 1] - q2[i] * i2[i - 1]
        re_[i] = 0.2 * re_[i] + 0.8 * re_[i - 1]
        im_[i] = 0.2 * im_[i] + 0.8 * im_[i - 1]
        if im_[i] != 0.0 and re_[i] != 0.0:
            period[i] = _TWO_PI / math.atan2(im_[i], re_[i])
        else:
            period[i] = prev_p
        if period[i] > 1.5 * prev_p:
            period[i] = 1.5 * prev_p
        if period[i] < 0.67 * prev_p:
            period[i] = 0.67 * prev_p
        if period[i] < 6.0:
            period[i] = 6.0
        if period[i] > 50.0:
            period[i] = 50.0
        period[i] = 0.2 * period[i] + 0.8 * prev_p
        smooth_period[i] = 0.33 * period[i] + 0.67 * smooth_period[i - 1]
        if i1[i] != 0.0:
            phase[i] = math.atan(q1[i] / i1[i]) * _RAD2DEG
        else:
            phase[i] = phase[i - 1] if i > 0 else 0.0
    return (smooth, detrend, q1, i1, i2, q2, re_, im_, period, smooth_period, phase)
_LOOKBACK = 32

def HT_DCPERIOD(real: pl.Series) -> pl.Series:
    """HT_DCPERIOD - Hilbert Transform - Dominant Cycle Period"""
    expr = register_plugin_function(args=[real], plugin_path=_LIB, function_name='ht_dcperiod', is_elementwise=False)
    if isinstance(real, pl.Series):
        return real.to_frame().select(expr).to_series()
    return expr

def HT_DCPHASE(real: pl.Series) -> pl.Series:
    """HT_DCPHASE - Hilbert Transform - Dominant Cycle Phase"""
    expr = register_plugin_function(args=[real], plugin_path=_LIB, function_name='ht_dcphase', is_elementwise=False)
    if isinstance(real, pl.Series):
        return real.to_frame().select(expr).to_series()
    return expr

def HT_PHASOR(real: pl.Series) -> tuple[pl.Series, pl.Series]:
    """HT_PHASOR - Hilbert Transform - Phasor Components (InPhase, Quadrature)"""
    expr = register_plugin_function(args=[real], plugin_path=_LIB, function_name='ht_phasor', is_elementwise=False)
    if isinstance(real, pl.Series):
        df = real.to_frame().select(expr).unnest(expr.meta.output_name())
        return tuple((df[col] for col in df.columns))
    return (expr.struct.field('inphase'), expr.struct.field('quadrature'))

def HT_SINE(real: pl.Series) -> tuple[pl.Series, pl.Series]:
    """HT_SINE - Hilbert Transform - Sine Wave (Sine, LeadSine)"""
    expr = register_plugin_function(args=[real], plugin_path=_LIB, function_name='ht_sine', is_elementwise=False)
    if isinstance(real, pl.Series):
        df = real.to_frame().select(expr).unnest(expr.meta.output_name())
        return tuple((df[col] for col in df.columns))
    return (expr.struct.field('sine'), expr.struct.field('leadsine'))

def HT_TRENDLINE(real: pl.Series) -> pl.Series:
    """HT_TRENDLINE - Hilbert Transform - Trendline"""
    expr = register_plugin_function(args=[real], plugin_path=_LIB, function_name='ht_trendline', is_elementwise=False)
    if isinstance(real, pl.Series):
        return real.to_frame().select(expr).to_series()
    return expr

def HT_TRENDMODE(real: pl.Series) -> pl.Series:
    """HT_TRENDMODE - Hilbert Transform - Trend Mode"""
    expr = register_plugin_function(args=[real], plugin_path=_LIB, function_name='ht_trendmode', is_elementwise=False)
    if isinstance(real, pl.Series):
        return real.to_frame().select(expr).to_series()
    return expr