import polars as pl

# ====================================================================
# Volume Indicators - 成交量指标
# ====================================================================


def AD(
    high: pl.Series, low: pl.Series, close: pl.Series, volume: pl.Series
) -> pl.Series:
    """AD - Chaikin A/D Line"""
    return ((2 * close - high - low) / (high - low) * volume).cum_sum()


def ADOSC(
    high: pl.Series,
    low: pl.Series,
    close: pl.Series,
    volume: pl.Series,
    fastperiod: int = 3,
    slowperiod: int = 10,
) -> pl.Series:
    """ADOSC - Chaikin A/D Oscillator"""
    return AD(high, low, close, volume).ewm_mean(
        span=fastperiod, adjust=False
    ) - AD(high, low, close, volume).ewm_mean(span=slowperiod, adjust=False)


def OBV(real: pl.Series, volume: pl.Series) -> pl.Series:
    """OBV - On Balance Volume"""
    return (real.diff().sign() * volume).cum_sum()
