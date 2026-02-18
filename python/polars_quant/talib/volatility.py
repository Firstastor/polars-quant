import polars as pl

# ====================================================================
# Volatility Indicators - 波动率指标
# ====================================================================

def ATR(
    high: pl.Series, low: pl.Series, close: pl.Series, timeperiod: int = 14
) -> pl.Series:
    """ATR - Average True Range"""
    return TRANGE(high, low, close).ewm_mean(span=timeperiod, adjust=False)


def NATR(
    high: pl.Series, low: pl.Series, close: pl.Series, timeperiod: int = 14
) -> pl.Series:
    """NATR - Normalized Average True Range"""
    return ATR(high, low, close, timeperiod) / close * 100


def TRANGE(high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """TRANGE - True Range"""
    return max(high, close.shift(1)) - min(low, close.shift(1))
