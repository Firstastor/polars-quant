# ruff: noqa
import polars as pl
from polars.plugins import register_plugin_function
from pathlib import Path

_LIB = str(Path(__file__).parent.parent)


def ADX(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    timeperiod: int = 14,
) -> pl.Expr:
    """ADX - Average Directional Movement Index"""
    return register_plugin_function(
        args=[high, low, close],
        kwargs={"timeperiod": timeperiod},
        plugin_path=_LIB,
        function_name="adx",
        is_elementwise=False,
    )


def ADXR(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    timeperiod: int = 14,
) -> pl.Expr:
    """ADXR - Average Directional Movement Index Rating"""
    return register_plugin_function(
        args=[high, low, close],
        kwargs={"timeperiod": timeperiod},
        plugin_path=_LIB,
        function_name="adxr",
        is_elementwise=False,
    )


def APO(
    real: pl.Expr,
    fastperiod: int = 12,
    slowperiod: int = 26,
    matype: int = 0,
) -> pl.Expr:
    """APO - Absolute Price Oscillator"""
    return register_plugin_function(
        args=[real],
        kwargs={"fastperiod": fastperiod, "slowperiod": slowperiod, "matype": matype},
        plugin_path=_LIB,
        function_name="apo",
        is_elementwise=False,
    )


def AROON(high: pl.Expr, low: pl.Expr, timeperiod: int = 14) -> tuple[pl.Expr, pl.Expr]:
    """AROON - Aroon"""
    expr = register_plugin_function(
        args=[high, low],
        kwargs={"timeperiod": timeperiod},
        plugin_path=_LIB,
        function_name="aroon",
        is_elementwise=False,
    )
    return expr.struct.field("aroon_up"), expr.struct.field("aroon_down")


def AROONOSC(high: pl.Expr, low: pl.Expr, timeperiod: int = 14) -> pl.Expr:
    """AROONOSC - Aroon Oscillator"""
    return register_plugin_function(
        args=[high, low],
        kwargs={"timeperiod": timeperiod},
        plugin_path=_LIB,
        function_name="aroonosc",
        is_elementwise=False,
    )


def BOP(
    open: pl.Expr,
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
) -> pl.Expr:
    """BOP - Balance Of Power"""
    return register_plugin_function(
        args=[open, high, low, close],
        plugin_path=_LIB,
        function_name="bop",
        is_elementwise=False,
    )


def CCI(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    timeperiod: int = 14,
) -> pl.Expr:
    """CCI - Commodity Channel Index"""
    return register_plugin_function(
        args=[high, low, close],
        kwargs={"timeperiod": timeperiod},
        plugin_path=_LIB,
        function_name="cci",
        is_elementwise=False,
    )


def CMO(real: pl.Expr, timeperiod: int = 14) -> pl.Expr:
    """CMO - Chande Momentum Oscillator"""
    return register_plugin_function(
        args=[real],
        kwargs={"timeperiod": timeperiod},
        plugin_path=_LIB,
        function_name="cmo",
        is_elementwise=False,
    )


def DX(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    timeperiod: int = 14,
) -> pl.Expr:
    """DX - Directional Movement Index"""
    return register_plugin_function(
        args=[high, low, close],
        kwargs={"timeperiod": timeperiod},
        plugin_path=_LIB,
        function_name="dx",
        is_elementwise=False,
    )


def MACD(
    real: pl.Expr,
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9,
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """MACD - Moving Average Convergence/Divergence (MACD, Signal, Hist)"""
    expr = register_plugin_function(
        args=[real],
        kwargs={
            "fastperiod": fastperiod,
            "slowperiod": slowperiod,
            "signalperiod": signalperiod,
        },
        plugin_path=_LIB,
        function_name="macd",
        is_elementwise=False,
    )
    return (
        expr.struct.field("macd"),
        expr.struct.field("macd_signal"),
        expr.struct.field("macd_hist"),
    )


def MACDEXT(
    real: pl.Expr,
    fastperiod: int = 12,
    fastmatype: int = 0,
    slowperiod: int = 26,
    slowmatype: int = 0,
    signalperiod: int = 9,
    signalmatype: int = 0,
) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """MACDEXT - MACD with controllable MA type"""
    from .overlap import MA

    macd_line = MA(real, fastperiod, fastmatype) - MA(real, slowperiod, slowmatype)
    signal_line = MA(macd_line, signalperiod, signalmatype)
    return (
        macd_line.alias("macd_dif"),
        signal_line.alias("macd_dea"),
        (macd_line - signal_line).alias("macd_hist"),
    )


def MACDFIX(real: pl.Expr, signalperiod: int = 9) -> tuple[pl.Expr, pl.Expr, pl.Expr]:
    """MACDFIX - Moving Average Convergence/Divergence Fixed 12/26/9"""
    return MACD(real, 12, 26, signalperiod)


def MFI(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    volume: pl.Expr,
    timeperiod: int = 14,
) -> pl.Expr:
    """MFI - Money Flow Index"""
    return register_plugin_function(
        args=[high, low, close, volume],
        kwargs={"timeperiod": timeperiod},
        plugin_path=_LIB,
        function_name="mfi",
        is_elementwise=False,
    )


def MINUS_DI(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    timeperiod: int = 14,
) -> pl.Expr:
    """MINUS_DI - Minus Directional Indicator"""
    return register_plugin_function(
        args=[high, low, close],
        kwargs={"timeperiod": timeperiod},
        plugin_path=_LIB,
        function_name="minus_di",
        is_elementwise=False,
    )


def MINUS_DM(high: pl.Expr, low: pl.Expr, timeperiod: int = 14) -> pl.Expr:
    """MINUS_DM - Minus Directional Movement"""
    return register_plugin_function(
        args=[high, low],
        kwargs={"timeperiod": timeperiod},
        plugin_path=_LIB,
        function_name="minus_dm",
        is_elementwise=False,
    )


def MOM(real: pl.Expr, timeperiod: int = 10) -> pl.Expr:
    """MOM - Momentum"""
    return register_plugin_function(
        args=[real],
        kwargs={"timeperiod": timeperiod},
        plugin_path=_LIB,
        function_name="mom",
        is_elementwise=False,
    )


def PLUS_DI(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    timeperiod: int = 14,
) -> pl.Expr:
    """PLUS_DI - Plus Directional Indicator"""
    return register_plugin_function(
        args=[high, low, close],
        kwargs={"timeperiod": timeperiod},
        plugin_path=_LIB,
        function_name="plus_di",
        is_elementwise=False,
    )


def PLUS_DM(high: pl.Expr, low: pl.Expr, timeperiod: int = 14) -> pl.Expr:
    """PLUS_DM - Plus Directional Movement"""
    return register_plugin_function(
        args=[high, low],
        kwargs={"timeperiod": timeperiod},
        plugin_path=_LIB,
        function_name="plus_dm",
        is_elementwise=False,
    )


def PPO(
    real: pl.Expr,
    fastperiod: int = 12,
    slowperiod: int = 26,
    matype: int = 0,
) -> pl.Expr:
    """PPO - Percentage Price Oscillator"""
    return register_plugin_function(
        args=[real],
        kwargs={"fastperiod": fastperiod, "slowperiod": slowperiod, "matype": matype},
        plugin_path=_LIB,
        function_name="ppo",
        is_elementwise=False,
    )


def ROC(real: pl.Expr, timeperiod: int = 10) -> pl.Expr:
    """ROC - Rate of change"""
    return register_plugin_function(
        args=[real],
        kwargs={"timeperiod": timeperiod},
        plugin_path=_LIB,
        function_name="roc",
        is_elementwise=False,
    )


def ROCP(real: pl.Expr, timeperiod: int = 10) -> pl.Expr:
    """ROCP - Rate of change Percentage"""
    return register_plugin_function(
        args=[real],
        kwargs={"timeperiod": timeperiod},
        plugin_path=_LIB,
        function_name="rocp",
        is_elementwise=False,
    )


def ROCR(real: pl.Expr, timeperiod: int = 10) -> pl.Expr:
    """ROCR - Rate of change ratio: (price/prevPrice)"""
    return register_plugin_function(
        args=[real],
        kwargs={"timeperiod": timeperiod},
        plugin_path=_LIB,
        function_name="rocr",
        is_elementwise=False,
    )


def ROCR100(real: pl.Expr, timeperiod: int = 10) -> pl.Expr:
    """ROCR100 - Rate of change ratio 100: (price/prevPrice)*100"""
    return register_plugin_function(
        args=[real],
        kwargs={"timeperiod": timeperiod},
        plugin_path=_LIB,
        function_name="rocr100",
        is_elementwise=False,
    )


def RSI(real: pl.Expr, timeperiod: int = 14) -> pl.Expr:
    """RSI - Relative Strength Index"""
    return register_plugin_function(
        args=[real],
        kwargs={"timeperiod": timeperiod},
        plugin_path=_LIB,
        function_name="rsi",
        is_elementwise=False,
    )


def STOCH(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    fastk_period: int = 5,
    slowk_period: int = 3,
    slowk_matype: int = 0,
    slowd_period: int = 3,
    slowd_matype: int = 0,
) -> tuple[pl.Expr, pl.Expr]:
    """STOCH - Stochastic (SlowK, SlowD)"""
    from .overlap import MA

    ln = low.rolling_min(fastk_period)
    hn = high.rolling_max(fastk_period)
    fastk = (close - ln) * 100.0 / (hn - ln)
    slowk = MA(fastk, slowk_period, slowk_matype).alias("slowk")
    slowd = MA(slowk, slowd_period, slowd_matype).alias("slowd")
    return (slowk, slowd)


def STOCHF(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    fastk_period: int = 5,
    fastd_period: int = 3,
    fastd_matype: int = 0,
) -> tuple[pl.Expr, pl.Expr]:
    """STOCHF - Stochastic Fast (FastK, FastD)"""
    from .overlap import MA

    ln = low.rolling_min(fastk_period)
    hn = high.rolling_max(fastk_period)
    fastk = ((close - ln) * 100.0 / (hn - ln)).alias("fastk")
    fastd = MA(fastk, fastd_period, fastd_matype).alias("fastd")
    return (fastk, fastd)


def STOCHRSI(
    real: pl.Expr,
    timeperiod: int = 14,
    fastk_period: int = 5,
    fastd_period: int = 3,
    fastd_matype: int = 0,
) -> tuple[pl.Expr, pl.Expr]:
    """STOCHRSI - Stochastic Relative Strength Index (FastK, FastD)"""
    from .overlap import MA

    rsi = RSI(real, timeperiod)
    ln = rsi.rolling_min(fastk_period)
    hn = rsi.rolling_max(fastk_period)
    fastk = ((rsi - ln) * 100.0 / (hn - ln)).alias("fastk_rsi")
    fastd = MA(fastk, fastd_period, fastd_matype).alias("fastd_rsi")
    return (fastk, fastd)


def TRIX(real: pl.Expr, timeperiod: int = 30) -> pl.Expr:
    """TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA"""
    return register_plugin_function(
        args=[real],
        kwargs={"timeperiod": timeperiod},
        plugin_path=_LIB,
        function_name="trix",
        is_elementwise=False,
    )


def ULTOSC(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    timeperiod1: int = 7,
    timeperiod2: int = 14,
    timeperiod3: int = 28,
) -> pl.Expr:
    """ULTOSC - Ultimate Oscillator"""
    return register_plugin_function(
        args=[high, low, close],
        kwargs={
            "timeperiod1": timeperiod1,
            "timeperiod2": timeperiod2,
            "timeperiod3": timeperiod3,
        },
        plugin_path=_LIB,
        function_name="ultosc",
        is_elementwise=False,
    )


def WILLR(
    high: pl.Expr,
    low: pl.Expr,
    close: pl.Expr,
    timeperiod: int = 14,
) -> pl.Expr:
    """WILLR - Williams' %R"""
    return register_plugin_function(
        args=[high, low, close],
        kwargs={"timeperiod": timeperiod},
        plugin_path=_LIB,
        function_name="willr",
        is_elementwise=False,
    )
