import polars as pl

# ====================================================================
# Overlap Studies - 重叠指标
# ====================================================================


def BBANDS(
    real: pl.Series, timeperiod: int = 20, nbdevup: float = 2.0, nbdevdn: float = 2.0
) -> tuple[pl.Series, pl.Series, pl.Series]:
    """BBANDS - Bollinger Bands (Upper, Middle, Lower)"""
    middleband = real.rolling_mean(timeperiod)
    stdev = real.rolling_std(timeperiod)
    return (
        (middleband + stdev * nbdevup).rename("bb_upper"),
        middleband.rename("bb_middle"),
        (middleband - stdev * nbdevdn).rename("bb_lower"),
    )


def DEMA(real: pl.Series, timeperiod: int = 30) -> pl.Series:
    """DEMA - Double Exponential Moving Average"""
    ema1 = real.ewm_mean(span=timeperiod, adjust=False)
    return (ema1 * 2.0 - ema1.ewm_mean(span=timeperiod, adjust=False)).rename("dema")


def EMA(real: pl.Series, timeperiod: int = 30) -> pl.Series:
    """EMA - Exponential Moving Average"""
    return real.ewm_mean(span=timeperiod, adjust=False).rename("ema")


def KAMA(real: pl.Series, timeperiod: int = 30) -> pl.Series:
    """KAMA - Kaufman Adaptive Moving Average"""
    sc = (
        (real.diff(timeperiod).abs() / real.diff().abs().rolling_sum(timeperiod))
        * 0.6022
        + 0.0645
    ).pow(2)
    ln_decay = (1 - sc).log1p()
    cum_ln_decay = ln_decay.cum_sum()
    cum_prod_decay = cum_ln_decay.exp()
    return (real * sc / cum_prod_decay).cum_sum() * cum_prod_decay


def MA(real: pl.Series, timeperiod: int = 30, matype: int = 0) -> pl.Series:
    """MA - Moving Average"""
    if matype == 0:
        return SMA(real, timeperiod)
    if matype == 1:
        return EMA(real, timeperiod)
    if matype == 2:
        return WMA(real, timeperiod)
    if matype == 3:
        return DEMA(real, timeperiod)
    if matype == 4:
        return TEMA(real, timeperiod)
    if matype == 5:
        return TRIMA(real, timeperiod)
    if matype == 6:
        return KAMA(real, timeperiod)
    if matype == 7:
        return MAMA(real, timeperiod)
    if matype == 8:
        return T3(real, timeperiod)
    return SMA(real, timeperiod)


def MAMA(
    real: pl.Series, fastlimit: float = 0.0, slowlimit: float = 0.0
) -> tuple[pl.Series, pl.Series]:
    """MAMA - MESA Adaptive Moving Average"""
    smooth = (4 * real.shift(3) + 3 * real.shift(2) + 2 * real.shift(1) + real) / 10
    return pl.Series(), pl.Series()


def MAVP(
    real: pl.Series,
    periods: pl.Series,
    minperiod: int = 2,
    maxperiod: int = 30,
    matype: int = 0,
) -> pl.Series:
    """MAVP - Moving Average with Variable Period"""
    return pl.Series()

def MIDPOINT(real: pl.Series, timeperiod: int = 14) -> pl.Series:
    """MIDPOINT - Midpoint over period"""
    return (real.rolling_max(timeperiod) + real.rolling_min(timeperiod)) * 0.5


def MIDPRICE(high: pl.Series, low: pl.Series, timeperiod: int = 14) -> pl.Series:
    """MIDPRICE - Midpoint Price over period"""
    return (high.rolling_max(timeperiod) + low.rolling_min(timeperiod)) * 0.5


def SAR(
    high: pl.Series, low: pl.Series, acceleration: float = 0.0, maximum: float = 0.0
) -> pl.Series:
    """SAR - Parabolic SAR"""
    return pl.Series()


def SAREXT(
    high: pl.Series,
    low: pl.Series,
    startvalue: float = 0.0,
    offsetonreverse: float = 0.0,
    accelerationinitlong: float = 0.0,
    accelerationlong: float = 0.0,
    accelerationmaxlong: float = 0.0,
    accelerationinitshort: float = 0.0,
    accelerationshort: float = 0.0,
    accelerationmaxshort: float = 0.0,
) -> pl.Series:
    """SAREXT - Parabolic SAR - Extended"""
    return pl.Series()


def SMA(real: pl.Series, timeperiod: int = 30) -> pl.Series:
    """SMA - Simple Moving Average"""
    return real.rolling_mean(timeperiod)


def T3(real: pl.Series, timeperiod: int = 5, vfactor: float = 0.7) -> pl.Series:
    """T3 - Triple Exponential Moving Average (T3)"""
    vfactor2 = vfactor**2
    vfactor3 = vfactor**3
    c1 = -(vfactor3)
    c2 = 3.0 * vfactor3 + 3.0 * vfactor3
    c3 = -6.0 * vfactor2 - 3.0 * vfactor - 3.0 * vfactor3
    c4 = 1.0 + 3.0 * vfactor + vfactor3 + 3.0 * vfactor2
    e1 = real.ewm_mean(span=timeperiod, adjust=False)
    e2 = e1.ewm_mean(span=timeperiod, adjust=False)
    e3 = e2.ewm_mean(span=timeperiod, adjust=False)
    e4 = e3.ewm_mean(span=timeperiod, adjust=False)
    e5 = e4.ewm_mean(span=timeperiod, adjust=False)
    e6 = e5.ewm_mean(span=timeperiod, adjust=False)
    return e6 * c1 + e5 * c2 + e4 * c3 + e3 * c4


def TEMA(real: pl.Series, timeperiod: int = 30) -> pl.Series:
    """TEMA - Triple Exponential Moving Average"""
    e1 = real.ewm_mean(span=timeperiod, adjust=False)
    e2 = e1.ewm_mean(span=timeperiod, adjust=False)
    e3 = e2.ewm_mean(span=timeperiod, adjust=False)
    return e1 * 3.0 - e2 * 3.0 + e3


def TRIMA(real: pl.Series, timeperiod: int = 30) -> pl.Series:
    """TRIMA - Triangular Moving Average"""
    n1 = (timeperiod + 1) // 2
    n2 = (timeperiod // 2) + 1 if timeperiod % 2 == 0 else n1
    return real.rolling_mean(n1).rolling_mean(n2)


def WMA(real: pl.Series, timeperiod: int = 30) -> pl.Series:
    """WMA - Weighted Moving Average"""
    weights = [
        i / (timeperiod * (timeperiod + 1) / 2) for i in range(1, timeperiod + 1)
    ]
    return real.rolling_mean(window_size=timeperiod, weights=weights)
