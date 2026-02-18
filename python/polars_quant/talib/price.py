import polars as pl

# ====================================================================
# Price Transform - 价格变换
# ====================================================================

def AVGPRICE(
    open: pl.Series, high: pl.Series, low: pl.Series, close: pl.Series
) -> pl.Series:
    """AVGPRICE - Average Price"""
    return (open + high + low + close) * 0.25


def MEDPRICE(high: pl.Series, low: pl.Series) -> pl.Series:
    """MEDPRICE - Median Price"""
    return (high + low) * 0.5


def TYPPRICE(high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """TYPPRICE - Typical Price"""
    return (high + low + close) * (1.0 / 3.0)


def WCLPRICE(high: pl.Series, low: pl.Series, close: pl.Series) -> pl.Series:
    """WCLPRICE - Weighted Close Price"""
    return (high + low + close * 2.0) * 0.25
