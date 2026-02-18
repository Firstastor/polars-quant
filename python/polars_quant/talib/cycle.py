import math

import polars as pl

# ====================================================================
# Cycle Indicators - 周期指标
# ====================================================================

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

    return smooth, detrend, q1, i1, i2, q2, re_, im_, period, smooth_period, phase


_LOOKBACK = 32


def HT_DCPERIOD(real: pl.Series) -> pl.Series:
    """HT_DCPERIOD - Hilbert Transform - Dominant Cycle Period"""
    r = real.to_list()
    n = len(r)
    result = [None] * n
    if n < _LOOKBACK:
        return pl.Series("", result, dtype=pl.Float64)

    _, _, _, _, _, _, _, _, _, smooth_period, _ = _compute_mesa_state(r)

    for i in range(_LOOKBACK, n):
        result[i] = smooth_period[i]
    return pl.Series("", result, dtype=pl.Float64)


def HT_DCPHASE(real: pl.Series) -> pl.Series:
    """HT_DCPHASE - Hilbert Transform - Dominant Cycle Phase"""
    r = real.to_list()
    n = len(r)
    result = [None] * n
    if n < _LOOKBACK + 1:
        return pl.Series("", result, dtype=pl.Float64)

    _, _, _, _, _, _, _, _, period, smooth_period, phase = _compute_mesa_state(r)

    for i in range(_LOOKBACK + 1, n):
        dc_period = smooth_period[i]
        dc_period_int = int(dc_period + 0.5)
        if dc_period_int < 1:
            dc_period_int = 1

        real_part = 0.0
        imag_part = 0.0
        idx_start = max(0, i - dc_period_int + 1)
        for j in range(idx_start, i + 1):
            weight_angle = _TWO_PI * (j - idx_start) / dc_period_int
            if r[j] is not None:
                real_part += math.cos(weight_angle) * r[j]
                imag_part += math.sin(weight_angle) * r[j]

        dc_phase = 0.0
        if abs(real_part) > 0.001:
            dc_phase = math.atan(imag_part / real_part) * _RAD2DEG
        elif abs(imag_part) > 0.001:
            dc_phase = 90.0 if imag_part > 0 else -90.0

        if real_part < 0.0:
            dc_phase += 180.0
        dc_phase += 90.0
        if dc_phase > 315.0:
            dc_phase -= 360.0

        result[i] = dc_phase
    return pl.Series("", result, dtype=pl.Float64)


def HT_PHASOR(real: pl.Series) -> tuple[pl.Series, pl.Series]:
    """HT_PHASOR - Hilbert Transform - Phasor Components (InPhase, Quadrature)"""
    r = real.to_list()
    n = len(r)
    inphase_out = [None] * n
    quadrature_out = [None] * n
    if n < _LOOKBACK:
        return (
            pl.Series("", inphase_out, dtype=pl.Float64),
            pl.Series("", quadrature_out, dtype=pl.Float64),
        )

    _, _, q1, i1, _, _, _, _, _, _, _ = _compute_mesa_state(r)

    for i in range(_LOOKBACK, n):
        inphase_out[i] = i1[i]
        quadrature_out[i] = q1[i]
    return (
        pl.Series("", inphase_out, dtype=pl.Float64),
        pl.Series("", quadrature_out, dtype=pl.Float64),
    )


def HT_SINE(real: pl.Series) -> tuple[pl.Series, pl.Series]:
    """HT_SINE - Hilbert Transform - Sine Wave (Sine, LeadSine)"""
    r = real.to_list()
    n = len(r)
    sine_out = [None] * n
    leadsine_out = [None] * n
    if n < _LOOKBACK + 1:
        return (
            pl.Series("", sine_out, dtype=pl.Float64),
            pl.Series("", leadsine_out, dtype=pl.Float64),
        )

    _, _, _, _, _, _, _, _, period, smooth_period, phase = _compute_mesa_state(r)

    for i in range(_LOOKBACK + 1, n):
        dc_period = smooth_period[i]
        dc_period_int = int(dc_period + 0.5)
        if dc_period_int < 1:
            dc_period_int = 1

        real_part = 0.0
        imag_part = 0.0
        idx_start = max(0, i - dc_period_int + 1)
        for j in range(idx_start, i + 1):
            weight_angle = _TWO_PI * (j - idx_start) / dc_period_int
            if r[j] is not None:
                real_part += math.cos(weight_angle) * r[j]
                imag_part += math.sin(weight_angle) * r[j]

        dc_phase = 0.0
        if abs(real_part) > 0.001:
            dc_phase = math.atan(imag_part / real_part) * _RAD2DEG
        elif abs(imag_part) > 0.001:
            dc_phase = 90.0 if imag_part > 0 else -90.0

        if real_part < 0.0:
            dc_phase += 180.0
        dc_phase += 90.0
        if dc_phase > 315.0:
            dc_phase -= 360.0

        sine_out[i] = math.sin(dc_phase * math.pi / 180.0)
        leadsine_out[i] = math.sin((dc_phase + 45.0) * math.pi / 180.0)

    return (
        pl.Series("", sine_out, dtype=pl.Float64),
        pl.Series("", leadsine_out, dtype=pl.Float64),
    )


def HT_TRENDLINE(real: pl.Series) -> pl.Series:
    """HT_TRENDLINE - Hilbert Transform - Trendline"""
    r = real.to_list()
    n = len(r)
    result = [None] * n
    if n < _LOOKBACK:
        return pl.Series("", result, dtype=pl.Float64)

    _, _, _, _, _, _, _, _, _, smooth_period, _ = _compute_mesa_state(r)

    for i in range(_LOOKBACK, n):
        dc_period = smooth_period[i]
        dc_period_int = int(dc_period + 0.5)
        if dc_period_int < 1:
            dc_period_int = 1
        trendline_val = 0.0
        count = 0
        for j in range(max(0, i - dc_period_int + 1), i + 1):
            if r[j] is not None:
                trendline_val += r[j]
                count += 1
        if count > 0:
            result[i] = trendline_val / count
        else:
            result[i] = r[i]
    return pl.Series("", result, dtype=pl.Float64)


def HT_TRENDMODE(real: pl.Series) -> pl.Series:
    """HT_TRENDMODE - Hilbert Transform - Trend Mode"""
    r = real.to_list()
    n = len(r)
    result = [None] * n
    if n < _LOOKBACK + 1:
        return pl.Series("", result, dtype=pl.Float64)

    _, _, _, _, _, _, _, _, _, smooth_period, phase = _compute_mesa_state(r)

    trend = 0
    prev_dc_phase = 0.0
    day_count = 0

    for i in range(_LOOKBACK + 1, n):
        dc_period = smooth_period[i]
        dc_period_int = int(dc_period + 0.5)
        if dc_period_int < 1:
            dc_period_int = 1

        real_part = 0.0
        imag_part = 0.0
        idx_start = max(0, i - dc_period_int + 1)
        for j in range(idx_start, i + 1):
            weight_angle = _TWO_PI * (j - idx_start) / dc_period_int
            if r[j] is not None:
                real_part += math.cos(weight_angle) * r[j]
                imag_part += math.sin(weight_angle) * r[j]

        dc_phase = 0.0
        if abs(real_part) > 0.001:
            dc_phase = math.atan(imag_part / real_part) * _RAD2DEG
        elif abs(imag_part) > 0.001:
            dc_phase = 90.0 if imag_part > 0 else -90.0

        if real_part < 0.0:
            dc_phase += 180.0
        dc_phase += 90.0
        if dc_phase > 315.0:
            dc_phase -= 360.0

        delta_phase = prev_dc_phase - dc_phase
        if prev_dc_phase < 90.0 and dc_phase > 315.0:
            delta_phase = prev_dc_phase + 360.0 - dc_phase
        prev_dc_phase = dc_phase

        if delta_phase < 1.0:
            delta_phase = 1.0
        if delta_phase > 1.0:
            trend = 1
            day_count = 0
        else:
            day_count += 1
            if day_count > 0.5 * smooth_period[i]:
                trend = 0

        trendline_val = 0.0
        count = 0
        for j in range(max(0, i - dc_period_int + 1), i + 1):
            if r[j] is not None:
                trendline_val += r[j]
                count += 1
        if count > 0:
            trendline_val /= count
        else:
            trendline_val = r[i] if r[i] is not None else 0.0

        sine_val = math.sin(dc_phase * math.pi / 180.0)
        leadsine_val = math.sin((dc_phase + 45.0) * math.pi / 180.0)

        cur_val = r[i] if r[i] is not None else 0.0
        if abs(cur_val - trendline_val) > 0.015 * cur_val if cur_val != 0.0 else False:
            trend = 1
        elif abs(sine_val - leadsine_val) < 0.3:
            trend = 0

        result[i] = trend
    return pl.Series("", result, dtype=pl.Int32)
