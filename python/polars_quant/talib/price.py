import polars as pl


def AVGPRICE(open: pl.Expr, high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    """AVGPRICE - Average Price"""
    return (open + high + low + close) / 4.0


def MEDPRICE(high: pl.Expr, low: pl.Expr) -> pl.Expr:
    """MEDPRICE - Median Price"""
    return (high + low) / 2.0


def TYPPRICE(high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    """TYPPRICE - Typical Price"""
    return (high + low + close) / 3.0


def WCLPRICE(high: pl.Expr, low: pl.Expr, close: pl.Expr) -> pl.Expr:
    """WCLPRICE - Weighted Close Price"""
    return (high + low + close * 2.0) / 4.0
